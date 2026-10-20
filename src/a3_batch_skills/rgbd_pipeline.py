"""Finite collect -> audit -> adapt -> two-example evaluation pipeline.

Training uses ONLY 2x5 skill demonstrations plus original 2x5 replay. The 2x10
teacher debugging runs are deliberately excluded from the learner's dataset.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from .train_visual import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--verified-smoke',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if not 1<=args.workers<=4:p.error('Use one to four render workers')
    smoke=json.loads((args.verified_smoke/'result.json').read_text())
    gradient=json.loads((args.verified_smoke/'gradient_check.json').read_text())
    manifest=json.loads((args.verified_smoke/'manifest.json').read_text())
    training_hash=hashlib.sha256(Path(__file__).with_name('train_visual.py').read_bytes()).hexdigest()
    if smoke.get('status')!='smoke_complete' or not gradient.get('passed') or manifest.get('training_code_sha256')!=training_hash:
        p.error('The exact trainer must pass real-model gradient smoke tests first')
    args.output.mkdir(parents=True,exist_ok=False)
    job=args.job
    camera=job/'artifacts/training_camera_profile.json'
    checkpoint=job/'training/checkpoints/030000/pretrained_model'
    policy_dir=job/'benchmark_research_v3'
    started=time.monotonic()
    def phase(name,**fields):
        atomic_json(args.output/'status.json',dict(stage=name,elapsed_seconds=time.monotonic()-started,**fields))
        print(json.dumps(dict(stage=name,**fields)),flush=True)
    def run(module,extra,log,timeout):
        with Path(log).open('w') as stream:
            process=subprocess.run([sys.executable,'-u','-m',module,*map(str,extra)],
                stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        return process.returncode
    try:
        phase('collecting_2x5_only')
        cases=[dict(task='2x5',profile=profile,column=column,seed=seed)
               for profile in ('basic','medium') for column in (1,2) for seed in range(26340,26344)]
        def collect(index,case):
            output=args.output/'collection'/f'episode_{index:02d}'
            output.parent.mkdir(exist_ok=True)
            code=run('a3_batch_skills',['--output',output,'--task','2x5','--profile',case['profile'],
                '--source-column',case['column'],'--seed',case['seed'],'--executor','grounded',
                '--camera-reference',camera,'--record-skills','--rgbd-goals'],
                output.with_suffix('.log'),1800)
            return dict(case=case,index=index,exit_code=code,output=str(output))
        records=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(collect,i,c) for i,c in enumerate(cases)]
            for future in as_completed(futures):
                records.append(future.result())
                phase('collecting_2x5_only',completed=len(records),total=len(cases))
        atomic_json(args.output/'collection_results.json',records)
        metadata=[json.loads(path.read_text()) for path in (args.output/'collection').glob('*/skill_clips/batch_*/metadata.json')]
        successful=[m for m in metadata if m['clip_success'] and m.get('visual_boundary_verified')]
        frames=sum(m['frames'] for m in successful)
        columns={c:sum(m['visual_goal']['source_column']==c for m in successful) for c in (1,2)}
        gate=dict(passed=len(successful)>=24 and frames>=7000 and min(columns.values())>=6,
            verified_clips=len(successful),frames=frames,source_columns=columns,
            failed_clips_excluded=sum(not m['clip_success'] for m in metadata),training_tasks=['2x5'])
        atomic_json(args.output/'data_gate.json',gate)
        if not gate['passed']:raise RuntimeError('Too few visually verified clips; no automatic neural training')
        dataset=args.output/'dataset'
        phase('exporting_left8',**gate)
        clip_roots=[str(Path(row['output'])/'skill_clips') for row in records]
        code=run('a3_batch_skills.export',['--inputs',*clip_roots,'--output',dataset,
            '--repo-id','local/a3_rgbd_batch_2x5_left8'],args.output/'export.log',1800)
        if code:raise RuntimeError('Dataset export failed')
        phase('auditing_real_dataset_reader')
        code=run('a3_batch_skills.audit',['--root',dataset,'--repo-id','local/a3_rgbd_batch_2x5_left8',
            '--checkpoint',checkpoint,'--job',job,'--policy-module-dir',policy_dir],args.output/'audit.log',900)
        if code:raise RuntimeError('Action timing or padded-loss audit failed')
        training=args.output/'training'
        phase('training_5000_updates',replay_fraction=.5,save_every=5000,training_tasks=['2x5'])
        code=run('a3_batch_skills.train_visual',['--output',training,'--dataset',dataset,'--checkpoint',checkpoint,
            '--job',job,'--policy-module-dir',policy_dir,'--steps','5000'],args.output/'training.log',18*3600)
        if code:raise RuntimeError('Neural adaptation failed; source checkpoint preserved')
        adapted=training/'checkpoints/005000/pretrained_model'
        phase('evaluating_two_new_examples')
        examples=[]
        for task in ('2x5','2x10'):
            output=args.output/('evaluation_'+task)
            code=run('a3_batch_skills.visual_policy_run',['--output',output,'--task',task,'--seed','26380',
                '--checkpoint',adapted,'--job',job,'--policy-module-dir',policy_dir,'--camera-reference',camera,'--video'],
                args.output/('evaluation_'+task+'.log'),3600)
            result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else dict(runtime_failed=True)
            examples.append(dict(task=task,exit_code=code,result=result))
        atomic_json(args.output/'result.json',dict(stage='complete',training_updates=5000,
            training_tasks=['2x5'],examples=examples,wall_seconds=time.monotonic()-started,
            note='Two development examples are not a success-rate estimate or a proven improvement.'))
        phase('complete')
    except Exception as exc:
        atomic_json(args.output/'error.json',dict(type=type(exc).__name__,message=str(exc)))
        phase('failed',reason=str(exc));raise


if __name__=='__main__':main()
