"""Bounded independent RGB-D captures across layouts, seeds and sensor settings."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=3)
    args=p.parse_args()
    if not 1<=args.workers<=4:p.error("Use one to four independent render workers")
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for task in ("2x5","2x10"):
        for profile in ("basic","medium"):
            for seed in (26310,26311):
                cases.append(dict(task=task,profile=profile,seed=seed,elevation=-90,azimuth=90,depth_noise_m=0.))
        for noise in (.0005,.001):
            cases.append(dict(task=task,profile="basic",seed=26312,elevation=-90,azimuth=90,depth_noise_m=noise))
        for azimuth in (0,180):
            cases.append(dict(task=task,profile="basic",seed=26310,elevation=-65,azimuth=azimuth,depth_noise_m=0.))
    def run(index,case):
        directory=args.output/f"case_{index:02d}"
        command=[sys.executable,"-m","a3_batch_skills.rgbd_probe","--output",str(directory),"--audit-truth"]
        for name,value in case.items():command.extend(["--"+name.replace("_","-"),str(value)])
        capture=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        result=dict(case=case,index=index,capture_exit=capture.returncode)
        (args.output/f"case_{index:02d}.capture.log").write_text(capture.stdout)
        if capture.returncode:return dict(result,status="capture_failed")
        audit=subprocess.run([sys.executable,"-m","a3_batch_skills.rgbd_audit","--capture",str(directory),
            "--output",str(directory/"audit"),"--audit-truth"],text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (args.output/f"case_{index:02d}.audit.log").write_text(audit.stdout)
        result["audit_exit"]=audit.returncode
        if audit.returncode:return dict(result,status="localization_rejected")
        report=json.loads((directory/"audit/result.json").read_text())
        return dict(result,status=report["status"],accepted_source=report["accepted_source_cookies"],
                    evaluation_only=report.get("evaluation_only"))
    started=time.monotonic();rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,i,case) for i,case in enumerate(cases)]
        for future in as_completed(futures):
            rows.append(future.result())
            status=dict(completed=len(rows),total=len(cases),results=sorted(rows,key=lambda r:r["index"]))
            (args.output/"status.json").write_text(json.dumps(status,indent=2))
            print(json.dumps(rows[-1]),flush=True)
    result=dict(status,wall_seconds=time.monotonic()-started,policy_training_steps=0,
        note="Localization evaluation only. Not robot task success or real-sensor accuracy.")
    (args.output/"result.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
