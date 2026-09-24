"""Collect successful randomized single-box episodes in LeRobot v3 format."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.benchmark import CookieBatchBenchmark, DIVERSE_RANDOMIZATION, EpisodeScore
from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.recording import LeRobotV3Recorder
from a3_dual_arm_sim.training import audit_training_dataset

TASK = "transfer 10 cookies into target box"
DEFAULT_ROOT = Path("datasets/a3_single_box_same_column_100")
DEFAULT_CONFIG = Path("configs/cookie_batch.yaml")
DEFAULT_REPO_ID = "local/a3-single-box-same-column-100"
DIVERSE_ROOT = Path("datasets/a3_single_box_diverse")
DIVERSE_REPO_ID = "local/a3-single-box-diverse"
ALL_COLUMNS_ROOT = Path("datasets/a3_single_box_diverse_all_columns")
ALL_COLUMNS_REPO_ID = "local/a3-single-box-diverse-all-columns"


def _from_project(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--repo-id")
    parser.add_argument("--profile", choices=("baseline", "diverse"), default="baseline")
    parser.add_argument(
        "--source-column", choices=("first", "random", "1", "2", "3", "4"),
        default="first", help="source column for diverse collection (1-4 or balanced random)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    diverse = args.profile == "diverse"
    if not diverse and args.source_column != "first":
        parser.error("--source-column requires --profile diverse")
    if args.source_column == "random":
        policy = "varied_column"
        default_root, default_repo_id = ALL_COLUMNS_ROOT, ALL_COLUMNS_REPO_ID
    elif args.source_column != "first":
        column = int(args.source_column)
        policy = f"column_{column - 1}"
        default_root = Path(f"datasets/a3_single_box_diverse_column_{column}")
        default_repo_id = f"local/a3-single-box-diverse-column-{column}"
    else:
        policy = "same_column"
        default_root = DIVERSE_ROOT if diverse else DEFAULT_ROOT
        default_repo_id = DIVERSE_REPO_ID if diverse else DEFAULT_REPO_ID
    root = _from_project(args.root or default_root)
    config_path = _from_project(args.config)
    randomization = DIVERSE_RANDOMIZATION if diverse else {}
    benchmark = CookieBatchBenchmark(config_path, max_steps=1000, **randomization)
    repo_id = args.repo_id or default_repo_id
    summary_path = root / "collection_summary.json"
    summary = {
        "schema_version": 2,
        "task": "cookie_transfer",
        "task_instruction": TASK,
        "policy": policy,
        "source_action_mode": "cartesian_delta",
        "stored_action_mode": "joint_position",
        "config": asdict(benchmark.config),
    }
    if diverse:
        summary["randomization"] = randomization
    if args.source_column != "first":
        summary["source_column"] = args.source_column
        summary["task_instruction_template"] = (
            "Transfer 10 cookies from source column {column} "
            "into the target box in two batches of five."
        )
        summary["task_instruction"] = summary["task_instruction_template"]
    if args.resume:
        if not summary_path.is_file():
            raise FileNotFoundError(f"cannot resume without {summary_path}")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version", "task", "policy", "stored_action_mode", "config",
            "randomization", "source_column", "task_instruction_template",
        ):
            if previous.get(key) != json.loads(json.dumps(summary.get(key))):
                raise RuntimeError(f"collection setting {key!r} changed since the initial run")

    recorder = LeRobotV3Recorder(
        root,
        repo_id=repo_id,
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
    seed = max((item["seed"] for item in attempts + saved), default=-1) + 1
    env = benchmark.create_env(render_cameras=True)
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
            episode_idx = recorder._episode_index + 1
            try:
                result = benchmark.run_episode(
                    policy, seed, episode_idx=episode_idx, env=env, recorder=recorder
                )
            except RuntimeError as exc:
                if "unreachable with vertical grasp" not in str(exc):
                    raise
                result = EpisodeScore(
                    episode=episode_idx, seed=seed, score=0, success=False,
                    failure_reason=f"expert_preflight: {exc}",
                )
            with attempts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(result)) + "\n")
            print(
                json.dumps({"saved": recorder._episode_index, **asdict(result)}),
                flush=True,
            )
            failures = 0 if result.success else failures + 1
            if failures >= 5:
                raise RuntimeError("five consecutive failed attempts; inspect before resuming")
            seed += 1
            if stop_requested:
                break
    finally:
        recorder.close()
        env.close()

    if recorder._episode_index == args.episodes:
        audit = audit_training_dataset(root, repo_id=repo_id)
        print("Dataset audit passed:", json.dumps(audit, indent=2), flush=True)
    else:
        print(f"Paused with {recorder._episode_index}/{args.episodes} saved episodes.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
