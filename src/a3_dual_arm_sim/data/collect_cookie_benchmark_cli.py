"""Collect successful randomized single-box episodes in LeRobot v3 format."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.data.recording import LeRobotV3Recorder
from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark

TASK = "transfer 10 cookies into target box"
DEFAULT_ROOT = Path("datasets/a3_front_close_left_100")
DEFAULT_CONFIG = Path("configs/cookie_batch.yaml")
DEFAULT_REPO_ID = "local/a3-front-close-left-100"


def _from_project(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_attempts is not None and args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")

    root = _from_project(args.root)
    config_path = _from_project(args.config)
    benchmark = CookieBatchBenchmark(config_path, max_steps=1000)
    summary_path = root / "collection_summary.json"
    summary = {
        "schema_version": 2,
        "task": "cookie_transfer",
        "task_instruction": TASK,
        "policy": "same_column",
        "source_action_mode": "cartesian_delta",
        "stored_action_mode": "joint_position",
        "config": asdict(benchmark.config),
        "seed_start": args.seed_start,
    }
    if args.resume:
        if not summary_path.is_file():
            raise FileNotFoundError(f"cannot resume without {summary_path}")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("seed_start", 0) != args.seed_start:
            raise RuntimeError("seed_start changed since the initial run")
        for key in ("schema_version", "task", "policy", "stored_action_mode", "config"):
            if previous.get(key) != json.loads(json.dumps(summary[key])):
                raise RuntimeError(f"collection setting {key!r} changed since the initial run")

    recorder = LeRobotV3Recorder(
        root,
        repo_id=args.repo_id,
        fps=benchmark.config.control_hz,
        image_height=benchmark.config.image_height,
        image_width=benchmark.config.image_width,
        resume=args.resume,
    )
    if not args.resume:
        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True
        ).strip()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    attempts_path = root / "attempts.jsonl"
    attempts = _read_jsonl(attempts_path)
    saved = _read_jsonl(root / "a3_episode_metadata.jsonl")
    seed = max((item["seed"] for item in attempts + saved), default=args.seed_start - 1) + 1
    failures = 0
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing the current episode before closing.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while recorder._episode_index < args.episodes:
            if args.max_attempts is not None and len(attempts) >= args.max_attempts:
                break
            try:
                result = benchmark.run_episode("same_column", seed, env=None, recorder=recorder)
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(asdict(result)) + "\n")
                attempts.append(asdict(result))
                print(
                    json.dumps({"saved": recorder._episode_index, **asdict(result)}),
                    flush=True,
                )
                failures = 0 if result.success else failures + 1
            except (RuntimeError, ValueError) as err:
                print(f"Skipping seed {seed} due to setup/unreachable error: {err}", flush=True)
                attempt_err = {
                    "seed": seed,
                    "success": False,
                    "score": 0,
                    "failure_reason": str(err),
                }
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(attempt_err) + "\n")
                attempts.append(attempt_err)
                failures += 1
            if failures >= 25:
                raise RuntimeError("25 consecutive failed attempts; inspect before resuming")
            seed += 1
            if stop_requested:
                break
    finally:
        recorder.close()

    if recorder._episode_index == args.episodes:
        audit = audit_training_dataset(root, repo_id=args.repo_id)
        print("Dataset audit passed:", json.dumps(audit, indent=2), flush=True)
    else:
        print(f"Paused with {recorder._episode_index}/{args.episodes} saved episodes.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
