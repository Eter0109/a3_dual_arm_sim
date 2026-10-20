"""Two simulation workers, one shared GPU model, independent action queues/RNG.

Own-process pipes carry RGB/proprioception only. Privileged audits stay in the
simulation workers. A suite is finite and produces one immutable case per seed.
"""
import argparse
import copy
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def clone_session(base):
    session=copy.copy(base)
    session.policy=copy.copy(base.policy)
    session.policy.reset()  # creates independent queue objects; weights shared
    session.pre=copy.deepcopy(base.pre);session.post=copy.deepcopy(base.post)
    session._right_hold=None;session.last_raw_action=None;session.steps=0
    return session


def serve(checkpoint,job,adapter_dir,pipes,control,ready,log):
    with open(log,'w',buffering=1) as stream:
        from contextlib import redirect_stdout,redirect_stderr
        with redirect_stdout(stream),redirect_stderr(stream):
            try:
                import torch
                torch.set_num_threads(2)
                sys.path.insert(0,adapter_dir)
                from pi05_adapter import Pi05BenchmarkPolicy
                base=Pi05BenchmarkPolicy(Path(checkpoint),Path(job))
                ready.send(dict(ready=True,n_action_steps=base.config.n_action_steps))
                sessions={};random_states={};active=[*pipes,control]
                while True:
                    for connection in wait(active):
                        if connection is control:
                            if connection.recv()=='shutdown':base.close();return
                            continue
                        try:request=connection.recv()
                        except EOFError:active.remove(connection);continue
                        try:
                            op=request['op']
                            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                                if op=='new':
                                    sessions[connection]=clone_session(base)
                                    torch.manual_seed(request['seed'])
                                else:
                                    cpu,gpu=random_states[connection]
                                    torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
                                session=sessions[connection]
                                if op=='reset':
                                    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
                                    session.reset(request['context'])
                                    torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
                                    value=None
                                elif op=='act':value=session.act(request['observation'],request['task'])
                                elif op=='close':value=None
                                elif op=='new':value=dict(n_action_steps=base.config.n_action_steps)
                                else:raise ValueError('Unknown inference operation')
                                random_states[connection]=(torch.get_rng_state(),torch.cuda.get_rng_state())
                            connection.send(dict(ok=True,value=value))
                            if op=='close':sessions.pop(connection,None);random_states.pop(connection,None)
                        except Exception:
                            connection.send(dict(ok=False,error=traceback.format_exc()))
            except Exception:
                ready.send(dict(ready=False,error=traceback.format_exc()));raise


class RemotePolicy:
    action_mode='joint_position'
    def __init__(self,connection,seed):
        from types import SimpleNamespace
        self.connection=connection
        self.config=SimpleNamespace(**self.call('new',seed=seed))
    def call(self,op,**kwargs):
        self.connection.send(dict(op=op,**kwargs));result=self.connection.recv()
        if not result['ok']:raise RuntimeError(result['error'])
        return result['value']
    def reset(self,context=None):return self.call('reset',context=context)
    def act(self,observation,task):
        names=('observation.state','observation.images.front','observation.images.left_wrist','observation.images.right_wrist')
        return self.call('act',observation={k:observation[k] for k in names},task=task)
    def close(self):self.call('close')


def queued_cases(tasks):
    """Block until a case/sentinel arrives; never use Queue.empty/get_nowait."""
    while True:
        case=tasks.get()
        if case is None:return
        yield case


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def worker(index,connection,tasks,root,common):
    os.environ['CUDA_VISIBLE_DEVICES']=''  # rendering uses EGL, no Torch CUDA context
    from .visual_policy_run import main
    from contextlib import redirect_stdout,redirect_stderr
    for case in queued_cases(tasks):
        output=Path(root)/'cases'/case['name']
        if (Path(root)/'STOP_AFTER_ACTIVE').exists():
            write_json(output.with_suffix('.exit.json'),dict(case=case,exit_code=130,
                worker=index,not_started=True,reason='Suite stopped after active cases'))
            continue
        args=[*common,'--output',str(output),'--seed',str(case['seed']),
              '--task',case['task'],'--profile',case['profile']]
        if case.get('max_batches'):args+=['--max-batches',str(case['max_batches'])]
        try:
            with open(output.with_suffix('.log'),'w',buffering=1) as stream,redirect_stdout(stream),redirect_stderr(stream):
                code=main(args,policy_factory=lambda *a,**k:RemotePolicy(connection,case['seed']))
            record=dict(case=case,exit_code=code,worker=index)
        except BaseException:
            record=dict(case=case,exit_code=-1,worker=index,error=traceback.format_exc())
        write_json(output.with_suffix('.exit.json'),record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--job',required=True)
    p.add_argument('--policy-module-dir',required=True);p.add_argument('--camera-reference',required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--seeds',type=int,nargs='+',default=[26380,26381])
    p.add_argument('--profiles',nargs='+',choices=('basic','medium'),default=['basic','medium'])
    p.add_argument('--task',choices=('2x5','2x10'),default='2x5')
    p.add_argument('--source-column',choices=('auto','1','2','3','4'),default='auto')
    p.add_argument('--max-batches',type=int)
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite a suite')
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<14*1024**3:p.error('Need 14 GiB free for one model')
    cases=[dict(name=f'{a.task}_{profile}_{seed}',task=a.task,profile=profile,seed=seed,max_batches=a.max_batches)
           for profile in a.profiles for seed in a.seeds]
    if len({c['name'] for c in cases})!=len(cases):p.error('Duplicate cases')
    a.output.mkdir(parents=True);(a.output/'cases').mkdir()
    (a.output/'protocol.json').write_text(json.dumps(dict(cases=cases,workers=a.workers,gpu_models=1,
        independent_action_queues=True,independent_rng=True,automatic_training=False,checkpoint=a.checkpoint,
        source_column=a.source_column),indent=2))
    context=mp.get_context('spawn')
    tasks=context.Queue()
    for case in cases:tasks.put(case)
    for _ in range(a.workers):tasks.put(None)
    previous_handlers={}
    def stop_after_active(signum,frame):
        (a.output/'STOP_AFTER_ACTIVE').touch()
    for signum in (signal.SIGTERM,signal.SIGINT):
        previous_handlers[signum]=signal.signal(signum,stop_after_active)
    pairs=[context.Pipe() for _ in range(a.workers)]
    control_parent,control_child=context.Pipe();ready_parent,ready_child=context.Pipe()
    server=context.Process(target=serve,args=(a.checkpoint,a.job,a.policy_module_dir,
        [p[0] for p in pairs],control_child,ready_child,str(a.output/'model.log')))
    server.start();workers=[]
    try:
        if not ready_parent.poll(180):raise RuntimeError('Model service did not initialize')
        ready=ready_parent.recv()
        if not ready['ready']:raise RuntimeError(ready['error'])
        common=['--checkpoint',a.checkpoint,'--job',a.job,'--policy-module-dir',a.policy_module_dir,
                '--camera-reference',a.camera_reference,'--placement-control','rgbd-servo',
                '--source-column',a.source_column,
                '--record-placement-probes','--audit-placement-truth','--video']
        for i,(_,client) in enumerate(pairs):
            process=context.Process(target=worker,args=(i,client,tasks,str(a.output),common));process.start();workers.append(process)
        write_json(a.output/'processes.json',dict(root=os.getpid(),model=server.pid,
            assignment='blocking_fifo',workers=[dict(pid=w.pid,index=i) for i,w in enumerate(workers)]))
        started=time.monotonic()
        while True:
            finished=[];live=[]
            for case in cases:
                root=a.output/'cases'/case['name'];exit_path=root.with_suffix('.exit.json')
                if exit_path.exists():
                    result=json.loads(exit_path.read_text());path=root/'result.json'
                    if path.exists():result.update(json.loads(path.read_text()))
                    finished.append(result)
                elif (root/'status.json').exists():
                    try:live.append(dict(case=case,**json.loads((root/'status.json').read_text())))
                    except json.JSONDecodeError:pass
            status=dict(completed=len(finished),planned=len(cases),active=live,
                        workers_alive=sum(w.is_alive() for w in workers),
                        skipped=sum(bool(r.get('not_started')) for r in finished),
                        scope_successes=sum(bool(r.get('requested_scope_success')) for r in finished),
                        full_successes=sum(bool(r.get('native_success_evaluation_only')) for r in finished))
            write_json(a.output/'status.json',status)
            write_json(a.output/'report.json',finished)
            if not server.is_alive():raise RuntimeError('Shared model exited unexpectedly')
            if all(not w.is_alive() for w in workers):break
            if time.monotonic()-started>5400:raise RuntimeError('Suite exceeded its bounded wall budget')
            time.sleep(5)
        for w in workers:w.join()
        if len(finished)!=len(cases):raise RuntimeError('Some cases did not produce exit records')
        return 130 if any(r.get('not_started') for r in finished) else 0
    finally:
        for w in workers:
            if w.is_alive():w.terminate()
            w.join(timeout=5)
        if server.is_alive():control_parent.send('shutdown');server.join(timeout=15)
        if server.is_alive():server.terminate();server.join(timeout=5)
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)


if __name__=='__main__':raise SystemExit(main())
