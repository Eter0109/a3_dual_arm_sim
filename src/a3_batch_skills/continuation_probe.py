"""Bounded frozen-policy diagnostics of contact tolerance and batch handoff."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .train_visual import atomic_json
from .observation_comparison import summarize


VARIANTS=(('handoff4','initial-open',4,'2x5'),
          ('preserve16','preserve',16,'2x5'),
          ('handoff16','initial-open',16,'2x5'),
          ('handoff4','initial-open',4,'2x10'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    for key in ('checkpoint','job','policy-module-dir','camera-reference'):p.add_argument('--'+key,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[26380,26490,26491])
    p.add_argument('--target-floor-fraction',type=float,default=.80)
    p.add_argument('--target-overlap-tolerance-mm',type=float,default=12.)
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite a diagnostic')
    a.output.mkdir(parents=True)
    weight=Path(a.checkpoint)/'model.safetensors';initial=(weight.stat().st_size,weight.stat().st_mtime_ns)
    protocol=dict(checkpoint=a.checkpoint,seeds=a.seeds,variants=VARIANTS,
        target_floor_fraction=a.target_floor_fraction,target_overlap_tolerance_mm=a.target_overlap_tolerance_mm,
        experimental_contact_probe=True,development_only=True,physical_resets_between_batches=0,
        frozen_weights=True,automatic_training=False,manipulation_controller='pi05',
        boundary_handoff_controller='empty-hand limited joint motion only',original_benchmark_scored_independently=True)
    atomic_json(a.output/'protocol.json',protocol)
    results=[];started=time.monotonic()
    try:
        for name,handoff,horizon,task in VARIANTS:
            wait_started=time.monotonic()
            while True:
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=16*1024:break
                atomic_json(a.output/'status.json',dict(stage='waiting_for_free_gpu',free_mib=free,completed_suites=len(results)))
                if time.monotonic()-wait_started>7200:raise RuntimeError('GPU wait exceeded diagnostic budget')
                time.sleep(15)
            target=a.output/(name+'_'+task)
            atomic_json(a.output/'status.json',dict(stage='testing',variant=name,task=task,completed_suites=len(results),elapsed_seconds=time.monotonic()-started))
            command=[sys.executable,'-u','-m','a3_batch_skills.visual_suite','--output',str(target),
                '--checkpoint',a.checkpoint,'--job',a.job,'--policy-module-dir',a.policy_module_dir,
                '--camera-reference',a.camera_reference,'--workers','2','--seeds',*map(str,a.seeds),
                '--profiles','basic','--task',task,'--source-column','auto','--placement-control','vla',
                '--sensor-mode','native-three','--observation-mode','active-left-wrist',
                '--placement-mode','contained_unstacked','--batch-handoff',handoff,
                '--n-action-steps','16','--continuation-action-steps',str(horizon),
                '--target-floor-fraction',str(a.target_floor_fraction),
                '--target-overlap-tolerance-mm',str(a.target_overlap_tolerance_mm)]
            with target.with_suffix('.log').open('w') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Continuation suite exceeded wall budget')
            if code:raise RuntimeError('Continuation runtime failed: '+target.name)
            rows=json.loads((target/'report.json').read_text())
            if any(r.get('exit_code',-1)<0 for r in rows):raise RuntimeError('Case exception: '+target.name)
            summary=summarize(rows)
            for r,row in zip(summary,rows):
                r.update(batch_handoff=row.get('batch_handoff'),
                    handoff_steps=row.get('action_sources',{}).get('boundary_handoff',0),
                    final_n_action_steps=row.get('n_action_steps'),
                    second_batch_started=any(e.get('event')=='visual_batch_started' and e.get('index')==1 for e in row.get('events',[])))
            results.append(dict(variant=name,task=task,results=summary))
            atomic_json(a.output/'comparisons.json',results)
            if (weight.stat().st_size,weight.stat().st_mtime_ns)!=initial:raise RuntimeError('Frozen weight file changed')
        atomic_json(a.output/'result.json',dict(status='complete',comparisons=results,automatic_training=False,claimed_improvement=False))
        atomic_json(a.output/'status.json',dict(stage='complete',elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        atomic_json(a.output/'error.json',dict(type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
