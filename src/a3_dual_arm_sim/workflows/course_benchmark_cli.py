"""Run the 14 base and four bonus course tasks under a versioned protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from a3_dual_arm_sim.paths import resource_root

from .benchmark import make_cookie_benchmark
from .course_protocol import (
    COURSE_BASE_TASKS,
    COURSE_BONUS_TASKS,
    COURSE_PROMPT,
    COURSE_PROTOCOL,
    COURSE_TASKS,
    course_max_steps,
)


def evaluate_course(*, policy, episodes=5, seed_start=0, standard_profile="advanced",
                    max_steps=None, workers=1, on_task_end=None):
    """Evaluate every task; task errors remain incomplete, never counted as failures."""
    if episodes < 1 or workers < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("episodes, workers and max_steps must be positive")
    if standard_profile not in ("basic", "medium", "advanced"):
        raise ValueError("unknown standard task profile")
    report = {
        "status": "running",
        "protocol": COURSE_PROTOCOL,
        "prompt": COURSE_PROMPT,
        "policy": policy if isinstance(policy, str) else getattr(policy, "name", type(policy).__name__),
        "episodes_per_task": episodes,
        "seed_start": seed_start,
        "standard_profile": standard_profile,
        "expanded_profile": "basic",
        "document_deviations": [
            "expanded tasks use basic instead of advanced randomization",
            "source selection and prefilled-cookie identity do not affect success",
        ],
        "tasks": {},
    }
    package = Path(__file__).resolve().parents[1]
    protocol_sources = (
        "tasks/cookie_transfer.py", "tasks/cookie_2x10.py", "tasks/cookie_2x10_settings.py",
        "tasks/cookie_scenarios.py", "config.py", "sim/env.py", "sim/actions.py",
        "sim/object_builder.py", "sim/randomization.py", "controllers/batch_expert.py",
        "controllers/same_column_batch_expert.py", "controllers/cookie_2x10_expert.py",
        "tasks/source_clearance.py", "tasks/expanded_ik.py", "controllers/course_expert.py",
        "controllers/course_sparse_expert.py", "workflows/benchmark.py",
        "workflows/episode_execution.py", "workflows/cookie_2x10_execution.py",
        "workflows/course_protocol.py", "workflows/course_benchmark_cli.py",
    )
    def source_hashes():
        hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                  for name in protocol_sources}
        for name in ("cookie_batch.yaml", "cookie_2x10.yaml", "randomization.yaml",
                     "randomization_2x10.yaml"):
            hashes[f"configs/{name}"] = hashlib.sha256(
                (resource_root() / "configs" / name).read_bytes()
            ).hexdigest()
        return hashes
    report["protocol_source_sha256"] = source_hashes()
    for task in COURSE_TASKS:
        profile = standard_profile if task in ("2x5", "2x10") else "basic"
        limit = max_steps if max_steps is not None else course_max_steps(task)
        try:
            if source_hashes() != report["protocol_source_sha256"]:
                raise RuntimeError("protocol source changed during the suite; start a new run")
            benchmark = make_cookie_benchmark(task, protocol=COURSE_PROTOCOL,
                                              profile=profile, max_steps=limit)
            result = benchmark.evaluate(policy, num_episodes=episodes, seed_start=seed_start,
                                        workers=workers, verbose=False)
            if source_hashes() != report["protocol_source_sha256"]:
                raise RuntimeError("protocol source changed during the task; start a new run")
            report["tasks"][task] = {
                "profile": profile,
                "max_steps": limit,
                "horizon": benchmark.config.horizon,
                **result.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - preserve and report other tasks if one fails
            report["tasks"][task] = {
                "profile": profile, "max_steps": limit,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if on_task_end is not None:
            on_task_end(task, report)
    def counts(tasks):
        records = [report["tasks"][task] for task in tasks]
        completed = sum(len(record.get("episodes", [])) for record in records)
        successes = sum(sum(bool(ep["success"]) for ep in record.get("episodes", []))
                        for record in records)
        return {"completed": completed, "expected": len(tasks) * episodes,
                "successes": successes}
    report["base"] = counts(COURSE_BASE_TASKS)
    report["bonus"] = counts(COURSE_BONUS_TASKS)
    report["bonus"]["points"] = (report["bonus"]["successes"]
                                 if episodes == 5 and report["bonus"]["completed"] == 20 else None)
    report["status"] = ("complete" if report["base"]["completed"] == len(COURSE_BASE_TASKS) * episodes
                        and report["bonus"]["completed"] == len(COURSE_BONUS_TASKS) * episodes
                        else "incomplete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="expert", help="expert or policy adapter specification")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--standard-profile", choices=("basic", "medium", "advanced"),
                        default="advanced")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def save(task, report):
        record = report["tasks"][task]
        if "error" in record:
            print(f"{task}: ERROR {record['error']}", flush=True)
        else:
            episodes = record["episodes"]
            print(f"{task}: {sum(bool(e['success']) for e in episodes)}/{len(episodes)} full successes", flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    report = evaluate_course(policy=args.policy, episodes=args.episodes,
                             seed_start=args.seed_start, standard_profile=args.standard_profile,
                             max_steps=args.max_steps, workers=args.workers, on_task_end=save)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "base", "bonus")}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
