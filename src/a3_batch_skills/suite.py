"""Bounded paired CPU checks for the geometric and legacy reference executors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from .__main__ import write_json


def roster(tasks, executors, seeds, columns, profile):
    return [dict(task=t, executor=e, seed=s, source_column=c, profile=profile,
                 max_steps=1500 if t == "2x5" else 3000, batch_steps=750)
            for t in tasks for e in executors for c in columns for s in seeds]


def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.95996398454
    p = successes / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def summarize(rows):
    groups = {}
    for row in rows:
        key = f"{row['task']}/{row['executor']}/column_{row['source_column']}/{row['profile']}"
        groups.setdefault(key, []).append(row)
    result = {}
    for key, group in groups.items():
        valid = [r for r in group if not r.get("process_error")]
        good = sum(bool(r.get("selected_success")) for r in valid)
        result[key] = dict(episodes=len(group), valid_episodes=len(valid), successes=good,
            process_errors=len(group)-len(valid), success_rate=good/len(valid) if valid else None,
            descriptive_wilson_95=wilson(good, len(valid)),
            failures=[dict(seed=r["seed"], reason=r.get("failure_reason"),
                           completed_batches=r.get("completed_batches"),
                           native_count=r.get("native_count"))
                      for r in group if not r.get("selected_success")])
        batches = 2 if group[0]["task"] == "2x5" else 4
        counts = [sum(r.get("completed_batches",0) >= i for r in valid) for i in range(1,batches+1)]
        denom = [len(valid),*counts[:-1]]
        result[key].update(native_successes=sum(bool(r.get("native_success")) for r in valid),
            mean_native_count=sum(r.get("native_count",0) for r in valid)/len(valid) if valid else None,
            batch_completion_counts=counts,
            conditional_batch_completion_rates=[n/d if d else None for n,d in zip(counts,denom)],
            max_right_target_error=max((r.get("right_target_max_error",0.) for r in valid),default=None),
            mean_control_steps=sum(r.get("interaction_steps",0) for r in valid)/len(valid) if valid else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", choices=("2x5", "2x10"), default=["2x5", "2x10"])
    p.add_argument("--executors", nargs="+", choices=("grounded", "expert"), default=["grounded"])
    p.add_argument("--seed-start", type=int, default=26200)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--columns", nargs="+", choices=("1", "2", "3", "4"), default=["1"])
    p.add_argument("--profile", choices=("basic", "medium", "diverse"), default="basic")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--camera-reference", type=Path)
    p.add_argument("--record-skills", action="store_true", help="Collect verified short clips during the suite")
    a = p.parse_args()
    if a.episodes < 1 or a.workers < 1:
        p.error("episodes and workers must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    cases = roster(a.tasks, a.executors, range(a.seed_start, a.seed_start+a.episodes), a.columns, a.profile)
    write_json(a.output/"protocol.json", dict(cases=cases, workers=a.workers,
        cameras=str(a.camera_reference), localization="simulator_truth",
        phase="development", physics_reset_between_batches=False, policy_training_steps=0))

    def run(case):
        name = f"{case['task']}_{case['executor']}_c{case['source_column']}_s{case['seed']}"
        root = a.output/name
        command = [sys.executable, "-m", "a3_batch_skills", "--output", str(root)]
        for key in ("task", "executor", "seed", "profile", "source_column", "max_steps", "batch_steps"):
            command += ["--"+key.replace("_", "-"), str(case[key])]
        if a.camera_reference:
            command += ["--camera-reference", str(a.camera_reference)]
        if a.record_skills:
            if case["executor"] != "grounded":
                raise ValueError("Short skill collection requires the goal-bound grounded executor")
            command += ["--record-skills"]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (a.output/(name+".log")).open("w") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((root/"result.json").read_text()) if (root/"result.json").is_file() else {}
        return dict(**case, **{k:v for k,v in result.items() if k not in case},
                    process_error=process.returncode not in (0, 2) or not result,
                    process_exit=process.returncode)

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run, case) for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            write_json(a.output/"results.json", rows)
            write_json(a.output/"summary.json", summarize(rows))
            print(json.dumps(dict(completed=len(rows), total=len(cases),
                last={k:rows[-1].get(k) for k in ("task", "executor", "seed", "selected_success", "failure_reason")})), flush=True)
    return 1 if any(r["process_error"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
