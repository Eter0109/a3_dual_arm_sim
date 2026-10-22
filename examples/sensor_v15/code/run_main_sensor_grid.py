"""Detached/resumable six-cell grid using main's actual episode runner.

Each case is a separate bounded subprocess. Results are committed after every
case; one failed episode or renderer cannot abort the remaining combinations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from a3_continuability.native_grid_reporting import cases,key,summarize
from a3_continuability.protocol import atomic_json


def worker(case):
    from a3_continuability.runtime import setup
    sys.path.insert(0,case['skill_code']);setup(Path(case['job']))
    import torch
    torch.set_num_threads(2)
    output=Path(case['output']);output.mkdir(parents=True,exist_ok=True)
    os.environ.update(A3_SENSOR_NATIVE_OUTPUT=str(output/'policy'),A3_SENSOR_REQUIRE_RELEASE='1',
        A3_SENSOR_SOURCE_COLUMN='1',A3_SENSOR_SNAPSHOTS='failure',
        A3_SENSOR_ACTIVE_RIGHT='0',A3_SENSOR_RIGHT_EACH_BATCH='0',
        A3_SENSOR_REFINEMENT='1' if case.get('refine') else '0',
        A3_SENSOR_HANDEYE='1' if case.get('handeye') else '0')
    if case.get('table_height') is not None:os.environ['A3_SENSOR_TABLE_HEIGHT_M']=str(case['table_height'])
    else:os.environ.pop('A3_SENSOR_TABLE_HEIGHT_M',None)
    from a3_dual_arm_sim.workflows import benchmark_cookie_batch_cli
    cli=['main-grid','--policy','a3_continuability.native_sensor_policy:make_policy',
        '--task',case['task'],'--profile',case['profile'],'--source-column','1',
        '--episodes','1','--seed-start',str(case['seed']),'--max-steps',str(case['max_steps']),
        '--workers','1','--prompt-style','training' if case['task']=='2x5' else 'task',
        '--output',str(output/'native_benchmark.json')]
    if case['task']=='2x10':cli+=['--first-grasp','5']
    sys.argv=cli
    started=time.monotonic();benchmark_cookie_batch_cli.main()
    native=json.loads((output/'native_benchmark.json').read_text())['episodes'][0]
    controller=json.loads((output/'policy'/f"seed_{case['seed']}"/'controller.json').read_text())
    row=dict(native,task=case['task'],profile=case['profile'],seed=case['seed'],
        controller_phase=controller['phase'],controller_failure=controller['failure_reason'],
        visual_completed_batches=controller['completed_batches'],infrastructure_error=None,
        measured_wall_seconds=time.monotonic()-started,max_steps=case['max_steps'])
    atomic_json(output/'result.json',row)


def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--job',type=Path,default=Path('/root/autodl-tmp/pi05_repro_20261015'))
    p.add_argument('--skill-code',type=Path,default=Path('/root/autodl-tmp/pi05_repro_20261015/batch_skill_code/47402a61dc8fe29c/src'))
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-start',type=int,default=13000)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--tasks',nargs='+',choices=('2x5','2x10'),default=['2x5','2x10'])
    p.add_argument('--profiles',nargs='+',choices=('basic','medium','advanced'),default=['basic','medium','advanced'])
    p.add_argument('--table-height',type=float)
    p.add_argument('--handeye',action='store_true')
    p.add_argument('--refine',action='store_true')
    p.add_argument('--case-timeout',type=int,default=1800)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--case-json')
    a=p.parse_args()
    if a.case_json:worker(json.loads(a.case_json));return
    if a.output is None or min(a.episodes,a.workers,a.case_timeout)<1:p.error('Invalid grid arguments')
    if a.handeye and a.table_height is None:p.error('--handeye requires --table-height')
    if a.output.exists() and not a.resume:p.error('Output exists; use --resume with the same code/protocol')
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    run_lock=(a.output/'.run.lock').open('a')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('This grid already has an active supervisor')
    code_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in (Path(__file__).parent/'a3_continuability').glob('*.py')}
    code_hashes[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=dict(tasks=a.tasks,profiles=a.profiles,episodes_per_cell=a.episodes,
        seed_start=a.seed_start,workers=a.workers,case_timeout_seconds=a.case_timeout,
        budgets={'2x5':1500,'2x10':3000},source_column='1',first_grasp=5,
        original_main_success_predicate=True,native_cameras_unchanged=True,
        continue_after_early_native_success_until_visual_release=True,
        action_source='classic_three_rgbd_visual_expert',learned_policy_loaded=False,
        object_truth_for_control=False,checkpoint=None,code_hashes=code_hashes,
        observed_handeye_calibration=a.handeye,table_height_m=a.table_height,active_right=False,
        task_conditioned_completion=a.refine,
        source_commit=(a.job/'benchmark_main_29fe90e/SOURCE_COMMIT.txt').read_text().strip(),
        storage='per-case logs/results; bounded calibration images and failure RGB-D only; no physics snapshots')
    previous=read_json(a.output/'protocol.json')
    if previous is not None and previous!=protocol:raise RuntimeError('Resume protocol/code differs')
    atomic_json(a.output/'protocol.json',protocol)
    rows_path=a.output/'results.jsonl'
    rows=[json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done={key(r) for r in rows}
    plan=cases(a.seed_start,a.episodes,tasks=a.tasks,profiles=a.profiles)
    active={};lock=threading.Lock();stop=threading.Event();started=time.time()
    def on_signal(signum,frame):stop.set()
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)

    def run_case(item):
        output=a.output/'cases'/key(item)
        cached=read_json(output/'result.json')
        if cached:return cached
        if output.exists():
            # Preserve interrupted sensor/log files; policy.reset requires a fresh
            # case directory. Never overwrite artifacts during --resume.
            attempts=a.output/'interrupted_attempts';attempts.mkdir(exist_ok=True)
            output.rename(attempts/(key(item)+'_'+str(time.time_ns())))
        payload=dict(item,job=str(a.job),skill_code=str(a.skill_code),output=str(output),
                     max_steps=protocol['budgets'][item['task']],handeye=a.handeye,table_height=a.table_height,refine=a.refine)
        output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--case-json',json.dumps(payload)]
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (output/'worker.log').open('ab') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env=env)
            with lock:active[key(item)]=dict(**item,pid=process.pid,started=time.time(),output=str(output))
            timeout=False
            while process.poll() is None:
                with lock:age=time.time()-active[key(item)]['started']
                if stop.is_set() or age>a.case_timeout:
                    timeout=age>a.case_timeout
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
        with lock:active.pop(key(item),None)
        if stop.is_set():return None
        row=read_json(output/'result.json')
        if row:return row
        error='case_timeout' if timeout else f'worker_exit_{process.returncode}'
        row=dict(item,infrastructure_error=error,log_tail=(output/'worker.log').read_text()[-1500:])
        atomic_json(output/'result.json',row)
        return row

    def save(status):
        with lock:running=[dict(v) for v in active.values()]
        for item in running:
            item['controller']=read_json(Path(item['output'])/'policy'/f"seed_{item['seed']}"/'status.json')
        cells=summarize(rows,a.episodes,tasks=a.tasks,profiles=a.profiles)
        atomic_json(a.output/'report.json',dict(status=status,completed=len(rows),planned=len(plan),
            running=running,cells=cells,episodes=rows,updated=time.time(),wall_seconds=time.time()-started))
        atomic_json(a.output/'status.json',dict(status=status,completed=len(rows),planned=len(plan),running=running))
        lines=['# main 三摄 RGB-D 网格测试','',f'状态：{status}；完成 {len(rows)}/{len(plan)}。',
            '', '| 任务 | 随机化 | 完成 | 原判据通过 | 松爪后视觉确认通过 | 平均最终数量 | 基础设施错误 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for c in cells:
            mean='—' if c['mean_final_count'] is None else f"{c['mean_final_count']:.2f}"
            lines.append(f"| {c['task']} | {c['profile']} | {c['completed']}/{c['planned']} | {c['native_successes']}/{c['valid_episodes']} | {c['release_verified_successes']}/{c['valid_episodes']} | {mean} | {c['infrastructure_errors']} |")
        lines+=['','固定供料第 1 列，每批五块；预算 1500/3000。main 原判据保留，回合延续至释放后视觉检查。',
                '运行中分母仅为已完成的有效回合；失败原因详见 report.json。']
        (a.output/'summary.md').write_text('\n'.join(lines)+'\n')

    save('running')
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            futures={executor.submit(run_case,c):c for c in plan if key(c) not in done}
            while futures:
                ready,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for future in ready:
                    item=futures.pop(future)
                    try:row=future.result()
                    except Exception as exc:row=dict(item,infrastructure_error=repr(exc))
                    if row is not None:
                        with rows_path.open('a') as stream:
                            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                        rows.append(row)
                        print(json.dumps(dict(completed=len(rows),planned=len(plan),case=key(item),
                            count=row.get('cookies_in_target'),success=row.get('success'),
                            error=row.get('infrastructure_error'))),flush=True)
                if stop.is_set():
                    for future in list(futures):
                        if future.cancel():futures.pop(future)
                save('stopping' if stop.is_set() else 'running')
        save('stopped' if stop.is_set() else 'completed')
    except BaseException:
        stop.set();save('error');raise


if __name__=='__main__':main()
