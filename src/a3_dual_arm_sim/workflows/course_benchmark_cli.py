"""Run the course assessment: 120 base and 20 bonus episodes by default."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from a3_dual_arm_sim.paths import resource_root

from .benchmark import make_cookie_benchmark
from .course_protocol import (
    COURSE_BONUS_TASKS,
    COURSE_EXPANDED_TASKS,
    COURSE_PROMPT,
    COURSE_PROTOCOL,
    COURSE_STANDARD_PROFILES,
    COURSE_STANDARD_TASKS,
    course_max_steps,
)

COURSE_ASSESSMENT_PLAN = "course-140-v1"


def build_course_schedule(*, episodes=None, standard_episodes=None, expanded_episodes=None,
                          bonus_episodes=None, standard_profile=None):
    """Expand task types into independently reported task/profile cases.

    ``episodes`` preserves the old uniform override; group-specific overrides
    take precedence. Without overrides, standard cases use 10 and others 5.
    """
    for name, value in (("episodes", episodes), ("standard_episodes", standard_episodes),
                        ("expanded_episodes", expanded_episodes), ("bonus_episodes", bonus_episodes)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if standard_profile is not None and standard_profile not in COURSE_STANDARD_PROFILES:
        raise ValueError("unknown standard task profile")
    profiles = (standard_profile,) if standard_profile is not None else COURSE_STANDARD_PROFILES
    common = episodes
    counts = {
        "standard": standard_episodes if standard_episodes is not None else common if common is not None else 10,
        "expanded": expanded_episodes if expanded_episodes is not None else common if common is not None else 5,
        "bonus": bonus_episodes if bonus_episodes is not None else common if common is not None else 5,
    }
    schedule = [
        {"id": f"{task}@{profile}", "task": task, "group": "standard",
         "profile": profile, "episodes": counts["standard"]}
        for task in COURSE_STANDARD_TASKS for profile in profiles
    ]
    schedule.extend(
        {"id": task, "task": task, "group": "expanded", "profile": "basic",
         "episodes": counts["expanded"]}
        for task in COURSE_EXPANDED_TASKS
    )
    schedule.extend(
        {"id": task, "task": task, "group": "bonus", "profile": "basic",
         "episodes": counts["bonus"]}
        for task in COURSE_BONUS_TASKS
    )
    return schedule, counts, profiles


def _source_hashes():
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
    hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
              for name in protocol_sources}
    for name in ("cookie_batch.yaml", "cookie_2x10.yaml", "randomization.yaml",
                 "randomization_2x10.yaml"):
        hashes[f"configs/{name}"] = hashlib.sha256(
            (resource_root() / "configs" / name).read_bytes()
        ).hexdigest()
    return hashes


def _update_counts(report, schedule):
    def counts(groups):
        selected = [case for case in schedule if case["group"] in groups]
        records = [report["tasks"][case["id"]] for case in selected]
        return {
            "completed": sum(len(record.get("episodes", [])) for record in records),
            "expected": sum(case["episodes"] for case in selected),
            "successes": sum(sum(bool(ep["success"]) for ep in record.get("episodes", []))
                             for record in records),
        }

    report["standard"] = counts({"standard"})
    report["expanded"] = counts({"expanded"})
    report["base"] = counts({"standard", "expanded"})
    report["bonus"] = counts({"bonus"})
    bonus_cases = [case for case in schedule if case["group"] == "bonus"]
    report["bonus"]["points"] = (
        report["bonus"]["successes"]
        if len(bonus_cases) == 4 and all(case["episodes"] == 5 for case in bonus_cases)
        and all(report["tasks"][case["id"]]["status"] == "complete" for case in bonus_cases)
        else None
    )
    report["total"] = counts({"standard", "expanded", "bonus"})


def evaluate_course(*, policy, episodes=None, standard_episodes=None, expanded_episodes=None,
                    bonus_episodes=None, seed_start=0, standard_profile=None,
                    max_steps=None, workers=1, on_task_end=None, dry_run=False):
    """Evaluate the requested cases; errors remain incomplete, not failures.

    Each standard task/profile combination has its own result record. An
    interrupted case keeps completed episodes when the source did not change.
    """
    schedule, counts, profiles = build_course_schedule(
        episodes=episodes, standard_episodes=standard_episodes,
        expanded_episodes=expanded_episodes, bonus_episodes=bonus_episodes,
        standard_profile=standard_profile,
    )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    uniform_episodes = next(iter(counts.values())) if len(set(counts.values())) == 1 else None
    report = {
        "status": "planned" if dry_run else "running",
        "protocol": COURSE_PROTOCOL,
        "assessment_plan": COURSE_ASSESSMENT_PLAN,
        "is_standard_assessment": (
            tuple(profiles) == COURSE_STANDARD_PROFILES
            and counts == {"standard": 10, "expanded": 5, "bonus": 5}
            and max_steps is None
        ),
        "prompt": COURSE_PROMPT,
        "policy": policy if isinstance(policy, str) else getattr(policy, "name", type(policy).__name__),
        "episodes_per_task": uniform_episodes,
        "episodes_per_group": counts,
        "seed_start": seed_start,
        "max_steps_override": max_steps,
        "standard_profile": standard_profile,
        "standard_profiles": list(profiles),
        "expanded_profile": "basic",
        "document_deviations": [
            "standard tasks use three profiles with ten episodes each",
            "expanded tasks use basic instead of advanced randomization",
            "source selection and prefilled-cookie identity do not affect success",
        ],
        "protocol_source_sha256": _source_hashes(),
        "tasks": {
            case["id"]: {
                "task": case["task"], "group": case["group"], "profile": case["profile"],
                "episodes_requested": case["episodes"],
                "max_steps": max_steps if max_steps is not None else course_max_steps(case["task"]),
                "status": "planned",
            }
            for case in schedule
        },
    }
    _update_counts(report, schedule)
    if dry_run:
        return report

    for case in schedule:
        record = report["tasks"][case["id"]]
        partial = []

        def on_episode_end(episode, captured=partial):
            captured.append(episode.to_dict())

        try:
            if _source_hashes() != report["protocol_source_sha256"]:
                raise RuntimeError("protocol source changed during the suite; start a new run")
            benchmark = make_cookie_benchmark(case["task"], protocol=COURSE_PROTOCOL,
                                              profile=case["profile"], max_steps=record["max_steps"])
            result = benchmark.evaluate(
                policy, num_episodes=case["episodes"], seed_start=seed_start,
                workers=workers, verbose=False, on_episode_end=on_episode_end,
            )
            if _source_hashes() != report["protocol_source_sha256"]:
                raise RuntimeError("protocol source changed during the task; start a new run")
            data = result.to_dict()
            completed = data.get("episodes", [])
            if len(completed) > case["episodes"]:
                raise RuntimeError("benchmark returned more episodes than requested")
            record.update(data)
            record["horizon"] = benchmark.config.horizon
            record["status"] = "complete" if len(completed) == case["episodes"] else "incomplete"
        except Exception as exc:  # noqa: BLE001 - preserve other tasks and completed episodes
            unchanged = _source_hashes() == report["protocol_source_sha256"]
            if unchanged:
                record["episodes"] = sorted(partial, key=lambda ep: ep.get("episode", 0))
            else:
                record["episodes"] = []
                record["invalid_episodes"] = partial
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        _update_counts(report, schedule)
        if on_task_end is not None:
            on_task_end(case["id"], report)
    report["status"] = (
        "complete" if all(record["status"] == "complete" for record in report["tasks"].values())
        else "incomplete"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="expert", help="expert or policy adapter specification")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override all case counts (default: standard 10, other cases 5)")
    parser.add_argument("--standard-episodes", type=int, default=None,
                        help="override the count for each standard task/profile case")
    parser.add_argument("--expanded-episodes", type=int, default=None,
                        help="override the count for each expanded base task")
    parser.add_argument("--bonus-episodes", type=int, default=None,
                        help="override the count for each bonus task")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--standard-profile", choices=COURSE_STANDARD_PROFILES, default=None,
                        help="evaluate only this standard profile; default evaluates all three")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the schedule without evaluating")
    args = parser.parse_args()

    def save(case_id, report):
        record = report["tasks"][case_id]
        if record["status"] == "error":
            print(f"{case_id}: ERROR {record['error']}", flush=True)
        else:
            completed = record.get("episodes", [])
            print(f"{case_id}: {sum(bool(e['success']) for e in completed)}/{len(completed)} "
                  "full successes", flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")

    try:
        report = evaluate_course(
            policy=args.policy, episodes=args.episodes, standard_episodes=args.standard_episodes,
            expanded_episodes=args.expanded_episodes, bonus_episodes=args.bonus_episodes,
            seed_start=args.seed_start, standard_profile=args.standard_profile,
            max_steps=args.max_steps, workers=args.workers, on_task_end=save, dry_run=args.dry_run,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({key: report[key] for key in ("status", "base", "bonus", "total")},
                         ensure_ascii=False))
    return 0 if report["status"] in ("complete", "planned") else 1


if __name__ == "__main__":
    raise SystemExit(main())
