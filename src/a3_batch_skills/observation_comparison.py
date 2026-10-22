"""Finite paired frozen-policy comparisons; never launches training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .train_visual import atomic_json


VARIANTS=(('legacy','legacy-fixed','passive','strict'),
          ('native_passive','native-three','passive','strict'),
          ('native_active','native-three','active-left-wrist','strict'),
          ('native_active_loose','native-three','active-left-wrist','contained_unstacked'))


def used_seeds(job):
    seeds=set()
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('seed'),int):seeds.add(value['seed'])
            for k,v in value.items():
                if k=='seeds' and isinstance(v,list):seeds.update(s for s in v if isinstance(s,int))
                elif isinstance(v,(dict,list)):collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for folder in ('batch_skill_results','continuability_results'):
        for pattern in ('*/protocol.json','*/selected_manifest.json','*/cases/*/result.json'):
            for p in (job/folder).glob(pattern):
                try:collect(json.loads(p.read_text()))
                except (OSError,ValueError):pass
    return seeds


def summarize(rows):
    return [dict(seed=r.get('seed'),task=r.get('task'),runtime_exit=r.get('exit_code'),
        model_started=any(e['event']=='visual_batch_started' for e in r.get('events',[])),
        completed_batches=r.get('completed_batches',0),scope_success=r.get('requested_scope_success',False),
        native_success=r.get('native_success_evaluation_only',False),native_count=r.get('native_count_evaluation_only',0),
        failure_reason=r.get('failure_reason'),observation_steps=r.get('observation_steps',0),
        action_sources=r.get('action_sources'),batch_counts=[e.get('observed_total') for e in r.get('events',[])
            if e['event']=='visual_batch_verified']) for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--policy-module-dir',type=Path,required=True);p.add_argument('--camera-reference',type=Path,required=True)
    p.add_argument('--wait-for-run',type=Path,help='Wait for this owned diagnostic to finish before loading a model')
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite comparison outputs')
    a.output.mkdir(parents=True);all_results=[];started=time.monotonic()
    weight=a.checkpoint/'model.safetensors';initial=(weight.stat().st_size,weight.stat().st_mtime_ns)
    unused=used_seeds(a.job);holdout=[];candidate=26510
    while len(holdout)<5:
        if candidate not in unused:holdout.append(candidate)
        candidate+=1
    protocol=dict(checkpoint=str(a.checkpoint),weight_size_mtime=initial,development=[26380,26490,26491],
        reserved_holdout=holdout,variants=VARIANTS,weights_frozen=True,automatic_training=False,
        original_benchmark_scored_independently=True,development_examples_not_formal_success_rate=True)
    atomic_json(a.output/'protocol.json',protocol)
    try:
        if a.wait_for_run:
            if not a.wait_for_run.resolve().is_relative_to((a.job/'batch_skill_results').resolve()):raise ValueError('Diagnostic is outside this job')
            deadline=time.monotonic()+7200
            while not a.wait_for_run.with_suffix('.exit.json').exists():
                atomic_json(a.output/'status.json',dict(stage='waiting_for_owned_diagnostic',path=str(a.wait_for_run)))
                if time.monotonic()>deadline:raise RuntimeError('Owned diagnostic wait exceeded finite budget')
                time.sleep(15)
        for stage,seeds in (('development',[26380,26490,26491]),('holdout',holdout)):
            for name,sensor,observation,placement in VARIANTS:
                for task in ('2x5','2x10'):
                    wait_started=time.monotonic()
                    while True:
                        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                        if free>=24*1024:break
                        atomic_json(a.output/'status.json',dict(stage='waiting_for_free_gpu',free_mib=free))
                        if time.monotonic()-wait_started>4*3600:raise RuntimeError('GPU remained occupied; no other job stopped')
                        time.sleep(30)
                    target=a.output/f'{stage}_{name}_{task}'
                    atomic_json(a.output/'status.json',dict(stage=stage,variant=name,task=task,
                        completed_suites=len(all_results),elapsed_seconds=time.monotonic()-started))
                    command=[sys.executable,'-u','-m','a3_batch_skills.visual_suite','--output',str(target),
                        '--checkpoint',str(a.checkpoint),'--job',str(a.job),'--policy-module-dir',str(a.policy_module_dir),
                        '--camera-reference',str(a.camera_reference),'--workers','2','--seeds',*map(str,seeds),
                        '--profiles','basic','--task',task,'--source-column','auto','--placement-control','vla',
                        '--sensor-mode',sensor,'--observation-mode',observation,'--placement-mode',placement]
                    with target.with_suffix('.log').open('w') as stream:
                        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        try:code=process.wait(timeout=7200)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                            raise RuntimeError('Comparison suite exceeded finite wall budget')
                    if code:raise RuntimeError('Comparison runtime failed: '+target.name)
                    rows=json.loads((target/'report.json').read_text())
                    if any(r.get('exit_code',-1)<0 for r in rows):raise RuntimeError('Case runtime exception: '+target.name)
                    all_results.append(dict(stage=stage,variant=name,task=task,results=summarize(rows)))
                    atomic_json(a.output/'comparisons.json',all_results)
                    if (weight.stat().st_size,weight.stat().st_mtime_ns)!=initial:raise RuntimeError('Frozen checkpoint changed externally')
            if stage=='development':
                new=[r for suite in all_results if suite['variant'] in ('native_active','native_active_loose') for r in suite['results']]
                retained=any(r['task']=='2x5' and r['seed']==26490 and (r['native_count']>=5 or r['completed_batches']>=1) for r in new)
                entry=any(r['seed']==26491 and r['model_started'] for r in new)
                atomic_json(a.output/'development_gate.json',dict(first_batch_retained=retained,blocked_seed_admitted=entry,passed=retained and entry))
                if not retained or not entry:
                    atomic_json(a.output/'result.json',dict(status='development_gate_failed',comparisons=all_results,
                        holdout_started=False,automatic_training=False,claimed_improvement=False))
                    atomic_json(a.output/'status.json',dict(stage='development_gate_failed',elapsed_seconds=time.monotonic()-started));return
        atomic_json(a.output/'result.json',dict(status='complete',comparisons=all_results,
            holdout_started=True,automatic_training=False,claimed_improvement=False))
        atomic_json(a.output/'status.json',dict(stage='complete',elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        atomic_json(a.output/'error.json',dict(type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
