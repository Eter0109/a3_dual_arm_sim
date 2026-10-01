"""Collect successful 2x10 episodes with a controllable first-grasp count in LeRobot v3 format."""

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
from a3_dual_arm_sim.sim.randomization import (
    choose_column,
    profile_parameters,
)
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan, parse_first_grasp
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from a3_dual_arm_sim.workflows.cookie_2x10_execution import Cookie2x10EpisodeExecution

TASK = "Transfer 20 cookies into the 2x10 target box, ten per column."
DEFAULT_ROOT = Path("datasets/a3_cookie_2x10")
DEFAULT_CONFIG = Path("configs/cookie_2x10.yaml")
DEFAULT_REPO_ID = "local/a3-cookie-2x10"


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
    parser.add_argument(
        "--first-grasp",
        type=parse_first_grasp,
        default=None,
        help="Optional override of YAML first_grasp (0..9 or random)",
    )
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and seeded prompts without collecting",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--randomization-config",
        type=Path,
        default=None,
        help="Independent 2x10 YAML; defaults to configs/randomization_2x10.yaml",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help=(
            "Store camera frames inline in parquet instead of as AV1 video. "
            "Inlining costs roughly ten times the disk (about 45 MB per episode "
            "versus 4 MB), so it is opt-in rather than the default."
        ),
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_attempts is not None and args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")

    root = _from_project(args.root)
    config_path = _from_project(args.config)
    settings, first_grasp = load_cookie_2x10_settings(
        _from_project(args.randomization_config) if args.randomization_config else None
    )
    if args.first_grasp is not None:
        first_grasp = args.first_grasp
    profile = settings["profile"]
    source_column = settings["source_column"]
    benchmark = Cookie2x10EpisodeExecution(
        config_path,
        max_steps=args.max_steps,
        first_grasp=first_grasp,
        profile=profile,
        source_column=source_column,
        randomization_settings=settings,
    )
    summary_path = root / "collection_summary.json"
    summary = {
        "schema_version": 2,
        "task": "cookie_transfer",
        "task_instruction": TASK,
        "repo_id": args.repo_id,
        "policy": "cookie_2x10_variable_grasp",
        "target_layout": "2x10",
        "required_cookies": 20,
        "first_grasp_mode": first_grasp,
        "first_grasp_sampling_version": 1,
        "zero_grasp_behavior": "skip",
        "grasp_plan": "[n, 10-n, n, 10-n]",
        "task_instruction_version": 1,
        "source_action_mode": "cartesian_delta",
        "stored_action_mode": "joint_position",
        "config": asdict(benchmark.config),
        "seed_start": args.seed_start,
        "randomization_profile": profile,
        "randomization_parameters": profile_parameters(profile, benchmark.randomization_settings),
        "randomization_settings": benchmark.randomization_settings,
        "source_column": source_column,
        "randomization_version": 1,
        "max_steps": args.max_steps,
        "use_videos": not args.no_videos,
    }
    if args.dry_run:
        previews = []
        for seed in range(args.seed_start, args.seed_start + 3):
            column = choose_column(source_column, seed)
            plan = Cookie2x10Plan.from_seed(first_grasp, seed)
            previews.append(
                {
                    "seed": seed,
                    "source_column": column,
                    **plan.metadata(),
                    "prompt": plan.prompt(column),
                }
            )
        print(json.dumps({"collection": summary, "preview": previews}, indent=2))
        return 0
    if args.resume:
        if not summary_path.is_file():
            raise FileNotFoundError(f"cannot resume without {summary_path}")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("seed_start", 0) != args.seed_start:
            raise RuntimeError("seed_start changed since the initial run")
        for key in (
            "schema_version",
            "task",
            "policy",
            "target_layout",
            "required_cookies",
            "first_grasp_mode",
            "first_grasp_sampling_version",
            "zero_grasp_behavior",
            "task_instruction_version",
            "repo_id",
            "stored_action_mode",
            "config",
            "randomization_profile",
            "randomization_parameters",
            "randomization_settings",
            "source_column",
            "randomization_version",
            "max_steps",
            "use_videos",
        ):
            if previous.get(key) != json.loads(json.dumps(summary[key])):
                raise RuntimeError(f"collection setting {key!r} changed since the initial run")

    recorder = LeRobotV3Recorder(
        root,
        repo_id=args.repo_id,
        fps=benchmark.config.control_hz,
        image_height=benchmark.config.image_height,
        image_width=benchmark.config.image_width,
        use_videos=not args.no_videos,
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
                result = benchmark.run_episode("variable_grasp", seed, env=None, recorder=recorder)
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
                    "phase": "SETUP_OR_EXPERT_EXCEPTION",
                    "source_column": choose_column(source_column, seed),
                    "randomization": benchmark.last_attempt_metadata,
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
