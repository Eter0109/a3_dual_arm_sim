"""Collect single-box PICK_FIVE/PLACE_FIVE LeRobot v3 skill episodes.

One successful continuous expert rollout creates four training episodes.  The
existing whole-task collector and dataset are intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.core.contracts import EpisodeContext
from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.data.skills import (
    _FRAME_KEYS,
    CookieSkillEpisodeRecorder,
    CookieSkillLabel,
)
from a3_dual_arm_sim.evaluation.benchmark import (
    DIVERSE_RANDOMIZATION,
    CookieBatchBenchmark,
    EpisodeScore,
)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class _SkippedSeed:
    seed: int
    source_bin_x_m: float


def _probe_source_bin_x(benchmark: CookieBatchBenchmark, env: Any, seed: int) -> float:
    """Recreate benchmark's seeded layout without camera rendering or expert steps."""
    # Keep these reset options identical to CookieBatchBenchmark.run_episode.
    env.reset(seed=seed, options={
        "randomize_cookies": benchmark.randomize_cookies,
        "randomize_boxes": benchmark.randomize_boxes,
        "randomize_source_bin": benchmark.randomize_boxes,
        "randomize_target_bin": benchmark.randomize_boxes,
        "source_bin_noise_m": benchmark.source_bin_noise_m,
        "target_bin_noise_m": benchmark.target_bin_noise_m,
        "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
        "camera_position_noise_m": benchmark.camera_position_noise_m,
        "camera_fovy_noise_deg": benchmark.camera_fovy_noise_deg,
        "light_noise_fraction": benchmark.light_noise_fraction,
        "color_noise_fraction": benchmark.color_noise_fraction,
    })
    return float(env.model.body_pos[env._source_bin_body, 0])


def _log_skipped_seed(path: Path, skipped: _SkippedSeed, minimum: float) -> None:
    item = {
        "seed": skipped.seed,
        "reason": "source_bin_x_below_min",
        "source_bin_x_m": skipped.source_bin_x_m,
        "required_min_m": minimum,
    }
    _append_jsonl(path, item)
    print(json.dumps({"skipped_seed": item}), flush=True)


class _BufferedSkillRolloutRecorder:
    """Buffer one worker rollout without opening a LeRobot writer in that worker."""

    def __init__(self) -> None:
        self.context: EpisodeContext | None = None
        self.controller_type = "unknown"
        self.frames: list[tuple[dict[str, np.ndarray], np.ndarray, CookieSkillLabel]] = []
        self.success = False

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        self.context = context
        self.controller_type = controller_type
        self.frames = []
        self.success = False

    def add_expert_frame(
        self, observation: dict[str, Any], action: np.ndarray, label: CookieSkillLabel,
    ) -> None:
        if self.context is None:
            raise RuntimeError("start_episode must be called before add_expert_frame")
        frame = {key: np.asarray(observation[key]).copy() for key in _FRAME_KEYS}
        self.frames.append((frame, np.asarray(action, dtype=np.float32).copy(), label))

    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None:
        raise RuntimeError("skill recording requires expert phase annotations")

    def finish_episode(self, success: bool | None = None) -> None:
        if self.context is None:
            raise RuntimeError("no rollout is active")
        self.success = success is True
        if not self.success:
            self.discard_episode()

    def discard_episode(self) -> None:
        self.context = None
        self.frames = []
        self.success = False


_WORKER_CACHE: dict[str, Any] = {}
_WORKER_CLEANUP_REGISTERED = False
_WORKER_PROBE_CLEANUP_REGISTERED = False


def _close_probe_env() -> None:
    probe_env = _WORKER_CACHE.pop("probe_env", None)
    if probe_env is not None:
        probe_env.close()


def _close_worker_env() -> None:
    env = _WORKER_CACHE.pop("env", None)
    if env is not None:
        env.close()
    _close_probe_env()
    _WORKER_CACHE.clear()


def _register_worker_cleanup() -> None:
    """Close the renderer before MuJoCo's later-registered EGL termination."""
    global _WORKER_CLEANUP_REGISTERED
    if not _WORKER_CLEANUP_REGISTERED:
        atexit.register(_close_worker_env)
        _WORKER_CLEANUP_REGISTERED = True


def _run_skill_attempt_worker(
    spec: dict[str, Any],
) -> tuple[EpisodeScore | _SkippedSeed, _BufferedSkillRolloutRecorder | None]:
    """Run one seed in a spawned process; never write to the shared dataset."""
    identity = (
        spec["config_path"], spec["max_steps"],
        json.dumps(spec["randomization"], sort_keys=True),
    )
    if _WORKER_CACHE.get("identity") != identity:
        _close_worker_env()
        benchmark = CookieBatchBenchmark(
            Path(spec["config_path"]),
            max_steps=spec["max_steps"],
            **spec["randomization"],
        )
        _WORKER_CACHE.update(
            identity=identity,
            benchmark=benchmark,
            env=benchmark.create_env(render_cameras=True),
        )
    minimum = spec.get("source_bin_x_min_m")
    if minimum is not None:
        if "probe_env" not in _WORKER_CACHE:
            global _WORKER_PROBE_CLEANUP_REGISTERED
            _WORKER_CACHE["probe_env"] = _WORKER_CACHE["benchmark"].create_env(
                render_cameras=False,
            )
            if not _WORKER_PROBE_CLEANUP_REGISTERED:
                # No EGL context is created by the probe.  It still needs to
                # be closed when every dispatched seed is filtered out.
                atexit.register(_close_probe_env)
                _WORKER_PROBE_CLEANUP_REGISTERED = True
        source_x = _probe_source_bin_x(
            _WORKER_CACHE["benchmark"], _WORKER_CACHE["probe_env"], spec["seed"],
        )
        if source_x < minimum:
            return _SkippedSeed(spec["seed"], source_x), None
    buffered = _BufferedSkillRolloutRecorder()
    try:
        result = _WORKER_CACHE["benchmark"].run_episode(
            spec["expert_policy"], spec["seed"], episode_idx=spec["episode_idx"],
            env=_WORKER_CACHE["env"], recorder=buffered,
        )
    finally:
        # MuJoCo registers eglTerminate when the first camera render creates a
        # GL context.  atexit callbacks run last-in-first-out, so registering
        # only after that render lets env.close() release its Renderer first.
        _register_worker_cleanup()
    if result.success != buffered.success:
        raise RuntimeError("worker success and buffered skill rollout disagree")
    return result, buffered if result.success else None


def _collect_parallel(
    *,
    args: argparse.Namespace,
    recorder: CookieSkillEpisodeRecorder,
    config_path: Path,
    randomization: dict[str, Any],
    expert_policy: str,
    attempts_path: Path,
    skipped_path: Path,
    first_seed: int,
    stop_requested: Callable[[], bool],
) -> None:
    """Bound in-flight rollouts and commit successful results through one writer."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from multiprocessing import get_context

    # Spawn imports MuJoCo separately in each process.  EGL is the headless
    # default on Linux, while an explicitly selected rendering backend wins.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    reservations_path = recorder.root / "seed_reservations.jsonl"
    next_seed = first_seed
    consecutive_failures = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"),
    ) as executor:
        pending: dict[Any, int] = {}
        while recorder.saved_rollouts < args.rollouts or pending:
            while (
                not stop_requested()
                and len(pending) < args.workers
                and recorder.saved_rollouts + len(pending) < args.rollouts
            ):
                seed = next_seed
                next_seed += 1
                _append_jsonl(
                    reservations_path, {"seed": seed, "worker_mode": "parallel"},
                )
                spec = {
                    "config_path": str(config_path),
                    "max_steps": args.max_steps,
                    "randomization": randomization,
                    "expert_policy": expert_policy,
                    "seed": seed,
                    "source_bin_x_min_m": getattr(args, "source_bin_x_min", None),
                    # Attempt identity must not depend on completion order.
                    "episode_idx": seed + 1,
                }
                pending[executor.submit(_run_skill_attempt_worker, spec)] = seed
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                seed = pending.pop(future)
                result, buffered = future.result()
                if isinstance(result, _SkippedSeed):
                    if result.seed != seed or buffered is not None:
                        raise RuntimeError("worker returned an inconsistent skipped seed")
                    _log_skipped_seed(skipped_path, result, args.source_bin_x_min)
                    continue
                if result.seed != seed:
                    raise RuntimeError("worker returned a different seed than assigned")
                if result.success:
                    if buffered is None or buffered.context is None or not buffered.success:
                        raise RuntimeError("successful worker returned no skill frames")
                    recorder.start_episode(buffered.context, buffered.controller_type)
                    for frame, action, label in buffered.frames:
                        recorder.add_expert_frame(frame, action, label)
                    recorder.finish_episode(success=True)
                elif buffered is not None:
                    raise RuntimeError("failed worker unexpectedly returned skill frames")
                _append_jsonl(attempts_path, asdict(result))
                print(
                    json.dumps({"saved_rollouts": recorder.saved_rollouts,
                                "skill_episodes": recorder.writer._episode_index,
                                **asdict(result)}),
                    flush=True,
                )
                consecutive_failures = 0 if result.success else consecutive_failures + 1
                if consecutive_failures >= 5:
                    raise RuntimeError("five consecutive failed attempts; inspect before resuming")


def collect_skill_dataset(args: argparse.Namespace) -> int:
    """Collect complete rollouts and commit their four labeled skill episodes."""
    workers = getattr(args, "workers", 1)
    source_bin_x_min = getattr(args, "source_bin_x_min", None)
    seed_start = getattr(args, "seed_start", 0)
    if args.rollouts < 1 or args.max_steps < 1 or workers < 1:
        raise ValueError("rollouts, max_steps and workers must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if source_bin_x_min is not None and not np.isfinite(source_bin_x_min):
        raise ValueError("source_bin_x_min must be finite")
    if args.profile == "baseline" and args.source_column != "first":
        raise ValueError("source_column requires profile diverse")
    if args.source_column == "first":
        expert_policy = "same_column"
    elif args.source_column == "random":
        expert_policy = "varied_column"
    else:
        expert_policy = f"column_{int(args.source_column) - 1}"

    suffix = ""
    if args.profile == "diverse":
        suffix = "_diverse"
        if args.source_column == "random":
            suffix += "_all_columns"
        elif args.source_column != "first":
            suffix += f"_column_{args.source_column}"
    default_name = f"a3_single_box_skills{suffix}"
    root = _project_path(args.root or Path("datasets") / default_name)
    repo_id = args.repo_id or f"local/{default_name.replace('_', '-')}"
    randomization = DIVERSE_RANDOMIZATION if args.profile == "diverse" else {}
    benchmark = CookieBatchBenchmark(
        _project_path(args.config), max_steps=args.max_steps, **randomization
    )
    if source_bin_x_min is not None:
        source_center_x = benchmark.config.cookie_transfer.source_bin_center_m[0]
        source_noise = benchmark.source_bin_noise_m if benchmark.randomize_boxes else 0.0
        source_x_upper = source_center_x + (
            source_noise
        )
        if source_bin_x_min > source_x_upper or (
            source_noise > 0 and source_bin_x_min == source_x_upper
        ):
            raise ValueError(
                f"source_bin_x_min={source_bin_x_min} cannot be reached with "
                f"source X upper bound {source_x_upper}"
            )
    summary = {
        "schema_version": 3,
        "task": "cookie_skills",
        "skills": ["PICK_FIVE", "PLACE_FIVE"],
        "segments_per_successful_rollout": 4,
        "expert_policy": expert_policy,
        "profile": args.profile,
        "source_column": args.source_column,
        "stored_action_mode": "joint_position",
        "config": asdict(benchmark.config),
        "randomization": randomization,
    }
    if source_bin_x_min is not None:
        summary["source_bin_x_min_m"] = source_bin_x_min
    if seed_start:
        summary["seed_start"] = seed_start
    summary_path = root / "collection_summary.json"
    if args.resume:
        if not summary_path.is_file():
            raise FileNotFoundError(f"cannot resume without {summary_path}")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("source_bin_x_min_m") != source_bin_x_min:
            raise RuntimeError("collection setting 'source_bin_x_min_m' changed since the initial run")
        if previous.get("seed_start", 0) != seed_start:
            raise RuntimeError("collection setting 'seed_start' changed since the initial run")
        for key, value in summary.items():
            if previous.get(key) != json.loads(json.dumps(value)):
                raise RuntimeError(f"collection setting {key!r} changed since the initial run")

    recorder = CookieSkillEpisodeRecorder(
        root,
        repo_id=repo_id,
        fps=benchmark.config.control_hz,
        image_height=benchmark.config.image_height,
        image_width=benchmark.config.image_width,
        resume=args.resume,
    )
    if not args.resume:
        summary["repo_id"] = repo_id
        summary["git_commit"] = subprocess.check_output(
            ["git", "-c", f"safe.directory={project_root()}", "rev-parse", "HEAD"],
            cwd=project_root(), text=True,
        ).strip()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    elif previous.get("repo_id") != repo_id:
        raise RuntimeError("repo_id changed since the initial run")

    attempts_path = root / "attempts.jsonl"
    skipped_path = root / "skipped_seeds.jsonl"
    attempts = _jsonl(attempts_path)
    saved = _jsonl(root / "a3_skill_segments.jsonl")
    reservations = _jsonl(root / "seed_reservations.jsonl")
    skipped = _jsonl(skipped_path)
    seed = max(
        [item["seed"] for item in attempts]
        + [item["parent_seed"] for item in saved]
        + [item["seed"] for item in reservations]
        + [item["seed"] for item in skipped],
        default=seed_start - 1,
    ) + 1
    env = benchmark.create_env(render_cameras=True) if workers == 1 else None
    probe_env = (
        benchmark.create_env(render_cameras=False)
        if workers == 1 and source_bin_x_min is not None else None
    )
    failures = 0
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing in-flight rollouts before closing.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        if workers > 1:
            _collect_parallel(
                args=args, recorder=recorder, config_path=_project_path(args.config),
                randomization=randomization, expert_policy=expert_policy,
                attempts_path=attempts_path, skipped_path=skipped_path, first_seed=seed,
                stop_requested=lambda: stop_requested,
            )
        else:
            while recorder.saved_rollouts < args.rollouts:
                if probe_env is not None:
                    source_x = _probe_source_bin_x(benchmark, probe_env, seed)
                    if source_x < source_bin_x_min:
                        _log_skipped_seed(
                            skipped_path, _SkippedSeed(seed, source_x), source_bin_x_min,
                        )
                        seed += 1
                        if stop_requested:
                            break
                        continue
                result = benchmark.run_episode(
                    expert_policy, seed, episode_idx=recorder.saved_rollouts + 1,
                    env=env, recorder=recorder,
                )
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(asdict(result)) + "\n")
                print(
                    json.dumps({"saved_rollouts": recorder.saved_rollouts,
                                "skill_episodes": recorder.writer._episode_index,
                                **asdict(result)}),
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
        if env is not None:
            env.close()
        if probe_env is not None:
            probe_env.close()

    if recorder.saved_rollouts == args.rollouts:
        audit = audit_training_dataset(root, repo_id=repo_id)
        print("Skill dataset audit passed:", json.dumps(audit, indent=2), flush=True)
    else:
        print(f"Paused with {recorder.saved_rollouts}/{args.rollouts} rollouts.", flush=True)
    return 0
