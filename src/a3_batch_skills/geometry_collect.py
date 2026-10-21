"""Finite concurrent geometry bootstrap collection; teacher actions are labeled."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--camera-reference',required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[26440,26441])
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite a collection')
    a.output.mkdir(parents=True);(a.output/'cases').mkdir()
    cases=[dict(seed=seed,profile=profile,column=column,name=f'{profile}_col{column}_{seed}')
           for profile in ('basic','medium') for column in (1,2) for seed in a.seeds]
    (a.output/'protocol.json').write_text(json.dumps(dict(cases=cases,workers=a.workers,
        task='2x5',actions='privileged_teacher',geometry='calibrated_RGBD_and_robot_FK_only',
        automatic_training=False),indent=2))
    active=[];records=[];pending=list(cases)
    try:
        while pending or active:
            while pending and len(active)<a.workers:
                case=pending.pop(0);output=a.output/'cases'/case['name']
                command=[sys.executable,'-u','-m','a3_batch_skills','--output',str(output),
                    '--task','2x5','--executor','grounded','--record-skills','--rgbd-goals',
                    '--record-geometry','--source-column',str(case['column']),'--seed',str(case['seed']),
                    '--profile',case['profile'],'--camera-reference',a.camera_reference]
                log=output.with_suffix('.log').open('w')
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                active.append((case,process,log,time.monotonic()))
            for entry in list(active):
                case,process,log,started=entry
                if process.poll() is None and time.monotonic()-started>2400:
                    process.terminate();process.wait(timeout=20)
                if process.poll() is None:continue
                log.close();records.append(dict(case=case,exit_code=process.returncode))
                active.remove(entry)
            clips=[json.loads(f.read_text()) for f in (a.output/'cases').glob('*/skill_clips/batch_*/metadata.json')]
            accepted=[x for x in clips if x.get('clip_success')]
            status=dict(completed=len(records),planned=len(cases),active=[x[0] for x in active],
                accepted_clips=len(accepted),accepted_frames=sum(x['frames'] for x in accepted))
            temporary=a.output/'status.pending';temporary.write_text(json.dumps(status,indent=2));temporary.replace(a.output/'status.json')
            if pending or active:time.sleep(5)
        (a.output/'result.json').write_text(json.dumps(dict(cases=records,**status),indent=2))
        return 0
    finally:
        for _,process,log,_ in active:
            if process.poll() is None:process.terminate();process.wait(timeout=20)
            log.close()


if __name__=='__main__':raise SystemExit(main())
