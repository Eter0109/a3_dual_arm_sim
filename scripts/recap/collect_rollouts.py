#!/usr/bin/env python3
"""Collect v9 rollouts for RECAP: LeRobot dataset + per-frame trace, failures included.

Why this file exists instead of a flag on the benchmark driver
-------------------------------------------------------------
`bench_pi05_on_main.py` measures; it does not train-ready record.  Three things the RECAP
pipeline needs are absent from it, and one of them is a correctness issue rather than a
convenience:

1. **`a_previous` is never updated on that path.**  `BridgeChunkPolicy` keeps the
   `a_previous` history channel in `_applied_action`, which is only written by its
   `observe_step` hook.  main's `episode_execution.run_episode` does not call
   `observe_step` -- only this repo's own `runner.py` does -- so every benchmark episode
   feeds the encoder a history window whose third channel is sixteen zeros.  Training data
   has the real applied action there (`add_history_window.py`), so the benchmark was
   measuring a distribution shift, not the checkpoint.  This collector calls the hook.

2. **Failed episodes are discarded.**  `run_episode` ends with
   `recorder.finish_episode()` on success and `recorder.discard_episode()` otherwise.  For
   RECAP the failures are the point: they are what teaches the value function that a state
   is bad, and they are 85-95 % of the episodes.  `KeepAllRecorder` below turns the
   discard into a save with `success=False`, which needs no change on main's side.

3. **Per-frame privileged counts have to sit next to the images.**  `cookies_in_target`
   per frame is the reward (`r_t = count[t+1] - count[t]`) and its episode-final value is
   the value target.  It is written here into `trace.jsonl` in the same loop iteration as
   the dataset frame, so the two can never drift apart -- which is the failure mode the
   plan's §2.4 warns about.

What is deliberately reused
---------------------------
The environment, its task config, the instruction sentence and the success predicate all
come from main's benchmark object rather than being rebuilt here.  The instruction
especially: 2x10 draws one of eleven plan variants from the seed
(`Cookie2x10Plan.from_seed`), and a collector that invented its own sentence would train
RECAP on text the checkpoint never sees.

Usage
-----
    python scripts/recap/collect_rollouts.py \
        --socket /tmp/graft_v9.sock \
        --output /root/autodl-tmp/datasets/a3_recap_rollout_p00 \
        --seeds 200-219 --task 2x10 --max-steps 1500 --history-frames 10

One process owns one output directory.  `LeRobotDataset.create` does not tolerate two
writers, so parallelism is N processes with N roots, merged later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# main's source tree first, for exactly the reason `bench_pi05_on_main.py` does it: the
# import must resolve to the worktree that owns the environment, never to a checkout of
# this feature branch.
MAIN_ROOT = os.environ.get("A3_MAIN_ROOT", "/tmp/a3-main")
sys.path.insert(0, os.path.join(MAIN_ROOT, "src"))
# The bridge policy lives in the benchmark driver; importing it is safe (its module level
# only extends `sys.path` and loads the wire protocol) and avoids a 300-line copy that
# would then have to be kept in step with the encoder it serves.
# `parents[1]` is the repo's `scripts/` directory -- bench_pi05_on_main.py sits directly in
# it, so appending "scripts" again resolves to scripts/scripts and breaks the import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from a3_dual_arm_sim.contracts import (  # noqa: E402
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    EpisodeContext,
)
from a3_dual_arm_sim.data.recording import ACTION, TASK  # noqa: E402
from a3_dual_arm_sim.sim.randomization import choose_column  # noqa: E402

from bench_pi05_on_main import BridgeChunkPolicy  # noqa: E402

#: Camera keys are repeated rather than imported from the dataset module so this file stays
#: readable on its own; the recorder validates them anyway.
TRACE_NAME = "trace.jsonl"
LABELS_NAME = "episode_labels.jsonl"

#: The two extra observation channels the v9 training set carries.  A rollout dataset
#: without them cannot be mixed with `a3_clean_v9h` -- LeRobot loads by feature name, so a
#: missing `observation.history` means the encoder gets nothing and the value model cannot
#: read the one input that makes it better than RLinf's image-only critic.
HISTORY = "observation.history"
PROGRESS = "observation.progress"


class NullRecorder:
    """`RecapRecorder`'s interface with no pixels behind it.

    Why an evaluation may skip the dataset
    -------------------------------------
    A scored episode needs three numbers -- `success`, `cookies_in_target` and `steps` -- and all
    three come from the trace and the environment.  The three camera streams exist because RECAP
    stage 1 needs *training data*; an evaluation re-derives nothing from them.

    They are also not free: `LeRobotDataset` encodes three 480x640 streams per control step
    through PNG writers, and measured on this machine an episode that steps at ~6 Hz while
    recording runs at ~1 Hz.  At 56 seeds an arm that is 23 hours instead of 4.

    What is *not* skipped, and why the numbers stay comparable
    ---------------------------------------------------------
    §9.80 of docs/pi05-training-history.md requires one harness for every arm in a comparison.
    That rule is about how the policy is driven -- `observe_step` on every control step so the
    per-step history window fills, the same chunk/execute split, the same reset options, the same
    success contract -- and none of that touches the recorder.  Only the archived pixels go away,
    so an episode scored here is scored by the same code as one scored with `--record`.
    """

    frames_recorded = 0

    def start_episode(self, context, controller_type: str) -> None:
        self.frames_recorded = 0

    def add_frame(self, observation, action, *, history=None, progress=None) -> None:
        self.frames_recorded += 1

    def set_episode_metadata(self, metadata: dict) -> None:
        return None

    def finish_episode(self, success: bool | None = None) -> None:
        self.frames_recorded = 0

    def discard_episode(self) -> None:
        self.frames_recorded = 0

    def close(self) -> None:
        return None


class KeepAllRecorder:
    """Save failed episodes instead of discarding them.

    main's runner decides success itself and calls `discard_episode()` on anything that is
    not a success.  RECAP needs those trajectories, so the discard is redirected to a
    save with `success=False`.  Everything else is passed straight through -- this is a
    shim, not a reimplementation.

    `discard_episode` is also reached when an episode recorded zero frames, and saving an
    empty episode would corrupt the dataset (LeRobot would write a zero-length parquet
    shard).  `_frames` guards that case, so the wrapper is only ever more permissive than
    the original.
    """

    def __init__(self, inner: "RecapRecorder") -> None:
        self.inner = inner
        #: Set by the caller before `finish_episode`/`discard_episode` so the saved
        #: metadata carries the real reason.
        self.last_success = False

    def _has_frames(self) -> bool:
        return self.inner.frames_recorded > 0

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        self.inner.start_episode(context, controller_type)

    def add_frame(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        *,
        history: np.ndarray | None = None,
        progress: np.ndarray | None = None,
    ) -> None:
        self.inner.add_frame(observation, action, history=history, progress=progress)

    def set_episode_metadata(self, metadata: dict[str, Any]) -> None:
        self.inner.set_episode_metadata(metadata)

    def finish_episode(self, success: bool | None = None) -> None:
        self.inner.finish_episode(success=success)

    def discard_episode(self) -> None:
        if self._has_frames():
            self.inner.finish_episode(success=False)
        else:
            self.inner.discard_episode()

    def close(self) -> None:
        self.inner.close()


class _ProgressTracker:
    """The two `observation.progress` channels, computed exactly as the policy computes them.

    `BridgeChunkPolicy._state_with_progress` is **not idempotent** -- it advances
    `self._deliveries` and the right-continuous entry rule -- so calling it a second time
    to obtain the training-data column would double-count every delivery.  On top of that
    it only runs inside `act`, i.e. once per 16-step chunk, while the dataset needs a value
    on every frame.

    So this is a separate tracker fed the same observations in the same order.  It is a
    copy of that rule, and duplication is a real risk: if the two ever disagree, the rollout
    data's progress channel means something slightly different from the training data's.
    `test_progress_tracker_matches_policy` pins them together on a synthetic trajectory, so
    drift is a failing test rather than a silent distribution shift.
    """

    def __init__(self) -> None:
        self._deliveries = 0
        self._inside = False
        self._inside_frames = 0
        self._gap = 10**9

    def update(self, observation: dict[str, Any]) -> np.ndarray:
        eef = np.asarray(observation.get(EEF_POSE, np.zeros(14)), dtype=float)
        inside = bool(eef.shape[0] >= 2 and eef[1] < 0.18)
        if inside:
            self._inside_frames += 1
            if not self._inside:
                if self._gap >= 20:
                    self._deliveries += 1
                self._inside = True
            self._gap = 0
        else:
            if self._inside and self._inside_frames < 5:
                self._deliveries -= 1
            self._inside = False
            self._inside_frames = 0
            self._gap = min(self._gap + 1, 10**9)
        counts = float(self._deliveries)
        cookies = min(counts * 5.0, 20.0)
        return np.array(
            [(counts / 4.0) * 2.0 - 1.0, (cookies / 20.0) * 2.0 - 1.0], dtype=np.float32
        )


class RecapRecorder:
    """LeRobot v3 writer whose schema matches `a3_clean_v9h`, failures included.

    `LeRobotV3Recorder` hard-codes `lerobot_features(...)`, which carries no history or
    progress column, so it cannot produce a dataset that mixes with the training set.  This
    writer declares the same fifteen features the training set has and nothing else.

    The feature declaration is what matters, not just the payload: LeRobot decides video vs
    inline image from the declared dtype, and the loader selects columns by name.  A
    dataset that is correct on disk but declares the wrong schema is worse than a broken
    one, because it loads.
    """

    def __init__(
        self,
        root: Path,
        *,
        repo_id: str,
        fps: int,
        image_height: int,
        image_width: int,
        history_frames: int,
        use_videos: bool = True,
        resume: bool = False,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.history_frames = max(int(history_frames), 0)
        image = {
            "dtype": "video" if use_videos else "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
        features: dict[str, Any] = {
            FRONT_IMAGE: image.copy(),
            LEFT_WRIST_IMAGE: image.copy(),
            RIGHT_WRIST_IMAGE: image.copy(),
            STATE: {"dtype": "float32", "shape": (16,), "names": [f"state_{i}" for i in range(16)]},
            VELOCITY: {
                "dtype": "float32",
                "shape": (16,),
                "names": [f"velocity_{i}" for i in range(16)],
            },
            EEF_POSE: {
                "dtype": "float32",
                "shape": (14,),
                "names": [f"eef_pose_{i}" for i in range(14)],
            },
            FORCE: {"dtype": "float32", "shape": (18,), "names": [f"force_{i}" for i in range(18)]},
            ACTION: {"dtype": "float32", "shape": (16,), "names": [f"action_{i}" for i in range(16)]},
            PROGRESS: {"dtype": "float32", "shape": (2,), "names": ["batches", "cookies"]},
        }
        if self.history_frames > 0:
            width = self.history_frames * 48
            features[HISTORY] = {
                "dtype": "float32",
                "shape": (width,),
                "names": [f"history_{i}" for i in range(width)],
            }

        self._metadata_path = self.root / "a3_episode_metadata.jsonl"
        self.dataset = (
            LeRobotDataset.resume(repo_id=repo_id, root=self.root, image_writer_threads=3)
            if resume
            else LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=self.root,
                robot_type="A3_dual_arm",
                features=features,
                use_videos=use_videos,
                image_writer_threads=3,
            )
        )
        self._context: EpisodeContext | None = None
        self._controller_type = "unknown"
        self._randomization_metadata: dict[str, Any] = {}
        self.frames_recorded = 0
        self.episode_index = 0
        if resume and self._metadata_path.is_file():
            self.episode_index = sum(
                1 for line in self._metadata_path.read_text().splitlines() if line.strip()
            )

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        self._context = context
        self._controller_type = controller_type
        self._randomization_metadata = {}
        self.frames_recorded = 0

    def add_frame(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        *,
        history: np.ndarray | None = None,
        progress: np.ndarray | None = None,
    ) -> None:
        if self._context is None:
            raise RuntimeError("start_episode must be called before add_frame")
        frame: dict[str, Any] = {
            FRONT_IMAGE: observation[FRONT_IMAGE],
            LEFT_WRIST_IMAGE: observation[LEFT_WRIST_IMAGE],
            RIGHT_WRIST_IMAGE: observation[RIGHT_WRIST_IMAGE],
            STATE: np.asarray(observation[STATE], dtype=np.float32),
            VELOCITY: np.asarray(observation[VELOCITY], dtype=np.float32),
            EEF_POSE: np.asarray(observation[EEF_POSE], dtype=np.float32),
            FORCE: np.asarray(observation[FORCE], dtype=np.float32),
            ACTION: np.asarray(action, dtype=np.float32),
            TASK: self._context.task,
        }
        if progress is not None:
            frame[PROGRESS] = np.asarray(progress, dtype=np.float32).reshape(2)
        if self.history_frames > 0:
            if history is None:
                # The expert has no proprioceptive window of its own.  Writing zeros keeps the
                # schema identical to a rollout's -- which is what lets both be loaded by one
                # trainer -- and is honest: a zero window is what "this policy has no history
                # channel" looks like, not a fabricated trajectory.  Recording it as zeros
                # rather than raising means the expert set can be mixed in without a schema
                # fork; the alternative is two dataset layouts and a loader branch.
                window = np.zeros(self.history_frames * 48, dtype=np.float32)
            else:
                window = np.asarray(history, dtype=np.float32).reshape(-1)
            if window.size != self.history_frames * 48:
                raise ValueError(
                    f"history window is {window.size} wide, expected "
                    f"{self.history_frames * 48} ({self.history_frames} x 48)"
                )
            frame[HISTORY] = window
        self.dataset.add_frame(frame)
        self.frames_recorded += 1

    def set_episode_metadata(self, metadata: dict[str, Any]) -> None:
        self._randomization_metadata = json.loads(json.dumps(metadata, default=str))

    def finish_episode(self, success: bool | None = None) -> None:
        if self._context is None:
            raise RuntimeError("no episode is active")
        self.dataset.save_episode(parallel_encoding=True)
        metadata = {
            "episode_index": self.episode_index,
            "seed": self._context.seed,
            "task": self._context.task,
            "controller_type": self._controller_type,
            "stored_action_mode": "joint_position",
            "frames": self.frames_recorded,
            "success": success,
        }
        if self._randomization_metadata:
            metadata["randomization"] = self._randomization_metadata
        with self._metadata_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False, default=str) + "\n")
        self.episode_index += 1
        self._context = None
        self.frames_recorded = 0

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self._context = None
        self.frames_recorded = 0

    def close(self) -> None:
        # `finalize` writes the meta files; without it the directory looks complete but no
        # loader can open it.
        try:
            self.dataset.finalize()
        except Exception:
            pass



def parse_seed_range(text: str) -> list[int]:
    """`200-219`, `200,201,300`, or a single number."""
    seeds: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError(f"no seeds in {text!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {text!r}")
    return seeds


def build_trace_entry(
    step: int,
    requested: np.ndarray,
    info: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """One trace row: what the policy asked for and what the world looked like after.

    The `after` block is copied from main's own `info`, not recomputed.  A second
    implementation of the judge is a second definition of success, and the value targets
    have to agree with the benchmark's verdict or the advantage labels are measuring a
    different task.
    """
    return {
        "step": step,
        "requested_action": np.asarray(requested, dtype=np.float32).tolist(),
        "applied_action": np.asarray(info["applied_action"], dtype=np.float32).tolist(),
        "expert_phase": phase,
        "after": {
            "cookies_in_target": int(info.get("cookies_in_target", 0)),
            "cookies_in_source": int(info.get("cookies_in_source", 0)),
            "success": bool(info.get("success", False)),
            "success_hold_count": int(info.get("success_hold_count", 0)),
            "target_slot_occupancy": [
                int(v) for v in info.get("target_slot_occupancy", []) or []
            ],
            "target_touches_all_walls": bool(info.get("target_touches_all_walls", False)),
            "safety_reason": info.get("safety_reason"),
        },
    }


def run_episode(
    bench,
    policy_factory: Any,
    seed: int,
    episode_index: int,
    *,
    recorder: KeepAllRecorder,
    trace_dir: Path,
    task_suffix: str = "",
) -> dict[str, Any]:
    """Drive one seed, recording every frame to the dataset and the trace.

    The loop mirrors main's `episode_execution.run_episode` (so the environment, the reset
    options and the success rule stay identical) with the two additions this collector
    exists for: `policy.observe_step(...)` after every step, and a trace row per frame.

    `policy_factory` is called per episode rather than once.  An expert adapter holds the
    environment it was bound to -- `bind_env` constructs the controller around `env` -- and
    this function closes its environment at the end of each episode, so reusing one adapter
    would drive a closed scene.  The bridge policy is stateless across episodes apart from
    its window, so the same factory shape works for it too.
    """
    policy = policy_factory()
    column = choose_column(bench.source_column, seed) if bench.source_column is not None else None
    task = bench._episode_task(column, seed)
    if task_suffix:
        # Advantage conditioning at inference is a property of the *sentence*, not of the model
        # (see `build_task_variants.py`), so asking for good behaviour means appending the same
        # suffix the positive training frames carried.  Kept as a parameter rather than baked in
        # so the same script can run the unconditional control on identical code.
        task = task + task_suffix

    # Same reset options as main's runner.  Copied rather than shared because
    # `run_episode` builds them inline; the plan's alternative -- calling it and losing
    # the per-step hook -- is the thing this collector is here to avoid.
    reset_options = {
        "randomize_cookies": bench.randomize_cookies,
        "randomize_boxes": bench.randomize_boxes,
        "randomize_source_bin": bench.randomize_boxes,
        "randomize_target_bin": bench.randomize_boxes,
        "source_bin_noise_m": bench.source_bin_noise_m,
        "target_bin_noise_m": bench.target_bin_noise_m,
        "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
        "randomization_profile": bench.profile or "basic",
        "appearance_seed": seed,
        "randomization_settings": bench.randomization_settings,
    }

    env = bench.create_env(render_cameras=True)
    trace_path = trace_dir / f"seed_{seed}" / TRACE_NAME
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = bench._episode_metadata(column, seed)
    trace_stream = trace_path.open("w", encoding="utf-8")
    progress_tracker = _ProgressTracker()
    steps = 0
    terminated = truncated = False
    info: dict[str, Any] = {}
    started = time.monotonic()
    try:
        observation, _ = env.reset(seed=seed, options=reset_options)
        metadata.update(getattr(env, "randomization_metadata", {}))
        metadata["reset_options"] = reset_options
        # Two policy protocols meet here.  `BridgeChunkPolicy` keeps its own history window and
        # needs `reset`; main's `ExpertPolicyAdapter` needs `bind_env` first (it constructs the
        # expert around the live environment) and has no `observe_step`.  Both are handled by
        # duck typing rather than by a branch on the class name, so a future policy that
        # implements either subset works unchanged.
        bind_env = getattr(policy, "bind_env", None)
        if callable(bind_env):
            bind_env(env)
        if hasattr(policy, "reset"):
            policy.reset(EpisodeContext(seed=seed, task=task, action_mode=env.action_mode))
        recorder.start_episode(
            EpisodeContext(seed=seed, task=task, action_mode=env.action_mode),
            getattr(policy, "name", type(policy).__name__),
        )
        recorder.set_episode_metadata(metadata)

        observe_step = getattr(policy, "observe_step", None)
        while steps < bench.max_steps:
            steps += 1
            # The progress channels belong to the observation the policy is about to act
            # on, so they are computed before `act` -- mirroring the training data, where
            # a frame's progress describes that frame.  Computed here rather than by
            # calling the policy's own helper: that one is not idempotent and already ran
            # inside `act` for the previous step (see `_ProgressTracker`).
            progress = progress_tracker.update(observation)
            requested = policy.act(observation, task)
            previous_observation = observation
            observation, _, terminated, truncated, info = env.step(requested)
            # Before the next `act`, so the window that step t+1 sends ends at t -- the
            # training data's definition of window t.  Absent on the expert, which has no
            # history channel to maintain.
            if callable(observe_step):
                observe_step(
                    requested=requested, applied=info["applied_action"], info=info
                )
            # `_history_vector` is a pure read of the window `act` just advanced, so the
            # stored column is exactly what the encoder was fed.  For the expert there is no
            # window, so the column is written as zeros -- the value model must not be trained
            # on a fake history, and `history_frames=0` makes the recorder reject the omission
            # rather than invent one.
            history = (
                policy._history_vector() if getattr(policy, "history_window", False) else None
            )
            recorder.add_frame(
                previous_observation,
                info["applied_action"],
                history=history,
                progress=progress,
            )
            trace_stream.write(
                json.dumps(
                    build_trace_entry(
                        steps, requested, info, phase=getattr(policy, "phase_name", "")
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            if getattr(policy, "finished", False) or terminated or truncated:
                break
    finally:
        trace_stream.close()
        env.close()
        # The policy is built per episode, so it is closed per episode.  Skipping this leaks
        # one bridge connection per seed, and the server keeps a thread and a slot for each --
        # after a few hundred seeds the bridge is serving hundreds of idle clients and the
        # forward latency climbs, which looks like a policy slowdown.
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()

    wall = time.monotonic() - started
    cookies_in_target = int(info.get("cookies_in_target", 0))
    cookies_in_source = int(info.get("cookies_in_source", 0))
    expected_source = (
        len(bench.config.cookie_transfer.cookie_source_positions_m) - bench.required_cookies
    )
    # main's rule, byte for byte (episode_execution.py:276): the environment's own verdict
    # plus the exact counts plus no safety stop.
    success = bool(
        info.get("success", False)
        and cookies_in_target == bench.required_cookies
        and cookies_in_source == expected_source
        and not info.get("safety_reason")
    )

    recorder.finish_episode(success=success)

    label = {
        "episode_index": episode_index,
        "seed": seed,
        "task": "2x10" if bench.task == "2x10" else "2x5",
        "instruction": task,
        "source_column": column,
        "steps": steps,
        "wall_seconds": round(wall, 2),
        "success": success,
        "cookies_in_target": cookies_in_target,
        "cookies_in_source": cookies_in_source,
        "required_cookies": bench.required_cookies,
        "target_slot_occupancy": [int(v) for v in info.get("target_slot_occupancy", []) or []],
        "target_touches_all_walls": bool(info.get("target_touches_all_walls", False)),
        "unmet_success_conditions": info.get("unmet_success_conditions"),
        "safety_reason": info.get("safety_reason"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        #: The value target this episode contributes (`cookies_in_target` at T) is
        #: `cookies_in_target` above; the reward is its frame-to-frame difference, which the
        #: trace already carries step by step.
        "failure_reason": None
        if success
        else ("environment_safety_terminated" if terminated else "max_steps_exceeded"),
    }
    return label


def make_policy(kind: str, args: argparse.Namespace, bench):
    """The acting policy, either the v9 bridge or the task's own expert.

    `expert` records demonstration-quality data through exactly the same loop, schema and
    traces as a rollout.  That matters for RECAP: the positive class is meant to be
    genuinely-better behaviour, and the cleanest source of it is the expert, not the top 30 %
    of a policy that succeeds 6 % of the time.  Using one collector for both also means the
    two sets cannot drift apart in schema.

    The expert is main's `variable_grasp` adapter for 2x10 -- the one whose `_episode_policy`
    builds `A3Cookie2x10Expert` with the seed's plan -- so the instructions match the plan the
    expert actually executes.
    """

    if kind == "bridge":
        def bridge_factory(seed: int = 0):
            # `seed` is accepted and ignored: the bridge's plan comes from `_episode_task`,
            # not from the policy.  The signature matches the expert factory so `run_episode`
            # can call both the same way.
            del seed
            return BridgeChunkPolicy(
                args.socket,
                execute=args.execute,
                history_window=args.history_frames > 0,
                history_frames=max(args.history_frames, 1),
                task_prompt=None,
            )

        return bridge_factory
    if kind == "expert":
        spec = "variable_grasp" if args.task == "2x10" else "same_column"
        # `_episode_policy` is the function `run_episode` itself would call, and it returns a
        # *constructed* adapter -- `make_policy_adapter("variable_grasp")` alone leaves the
        # string unresolved, which is the difference between an object with `act` and one with
        # `AttributeError: 'str' object has no attribute 'act'`.
        #
        # Wrapped in a factory because the adapter is bound to one environment (`bind_env`
        # builds the controller around it) and each episode closes its environment.
        # `sampling_seed` decides the plan the expert executes, so it has to follow the seed
        # even though the sentence the *policy* reads comes from `_episode_task`.
        def expert_factory(seed: int = 0):
            adapter = bench._episode_policy(spec, args.source_column, seed)  # noqa: SLF001
            return adapter

        return expert_factory
    raise SystemExit(f"unknown --policy {kind!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default="", help="bridge unix socket (required for --policy bridge)")
    parser.add_argument("--policy", default="bridge", choices=("bridge", "expert"))
    parser.add_argument("--output", required=True, help="dataset root (must be empty or new)")
    parser.add_argument("--seeds", required=True, help='e.g. "200-219"')
    parser.add_argument("--task", default="2x10", choices=("2x5", "2x10"))
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--execute", type=int, default=16, help="actions per bridge forward")
    parser.add_argument("--history-frames", type=int, default=10, help="0 disables the window")
    parser.add_argument(
        "--no-record",
        dest="record",
        action="store_false",
        help=(
            "skip writing the dataset, keeping only the trace and the episode labels.  For "
            "evaluation: an episode's score comes from the trace, so the three encoded camera "
            "streams are pure cost there.  The control loop is untouched, so scores stay "
            "comparable to a recorded run."
        ),
    )
    parser.set_defaults(record=True)
    parser.add_argument("--prompt-style", default="task", choices=("task", "short"))
    parser.add_argument(
        "--task-suffix",
        default="",
        help=(
            "appended to every episode's instruction, e.g. ' Advantage: positive'.  This is how "
            "the advantage-conditioned checkpoints are steered, and the unconditional control "
            "is the same run with this left empty."
        ),
    )
    parser.add_argument("--source-column", type=int, default=1)
    parser.add_argument("--repo-id", default="a3_recap_rollout")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing dataset; seeds already in episode_labels.jsonl are skipped",
    )
    args = parser.parse_args(argv)

    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    output = Path(args.output)
    # `LeRobotDataset.create` calls `mkdir(exist_ok=False)`, so the collector must not
    # create the root itself.  Refusing a non-empty directory is the same guard main's own
    # recorder applies: a silent merge into an existing dataset would interleave two
    # collection runs into one episode numbering.
    if not args.resume:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty dataset: {output}")
        if output.exists():
            output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    # With `--no-record` nothing else creates the root: `RecapRecorder` normally does it, and the
    # labels file is written first, so without this the run fails on the first `open("w")` after
    # the episode has already been stepped -- losing the work rather than the run.
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / LABELS_NAME
    trace_dir = output / "traces"

    done: set[int] = set()
    if args.resume and labels_path.is_file():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(int(json.loads(line)["seed"]))
        print(f"[collect] resuming: {len(done)} seeds already recorded", flush=True)

    seeds = [s for s in parse_seed_range(args.seeds) if s not in done]
    if not seeds:
        print("[collect] nothing to do", flush=True)
        return 0

    bench = make_cookie_benchmark(
        task=args.task,
        prompt_style=args.prompt_style,
        max_steps=args.max_steps,
        randomize_boxes=True,
        randomize_cookies=True,
        source_column=args.source_column,
        render=False,
        # 2x10 has no early exit on success (`terminate_on_success=False`), which is what
        # makes the full-budget episodes comparable across arms.
        success_hold_steps=20,
    )
    config = bench.config
    # `--record` is the default because collecting training data is this script's main job;
    # an evaluation turns it off and keeps everything except the pixels.  See `NullRecorder`.
    recorder = (
        RecapRecorder(
            output,
            repo_id=args.repo_id,
            fps=config.control_hz,
            image_height=config.image_height,
            image_width=config.image_width,
            history_frames=args.history_frames,
            # Video, not inline PNG: the 400-episode training set is video too, and LeRobot
            # decides storage from the declared dtype.  Declaring `image` here would produce a
            # dataset ten times larger that the loader treats differently.
            use_videos=True,
            resume=args.resume and (output / "meta").is_dir(),
        )
        if args.record
        else NullRecorder()
    )
    keep_all = KeepAllRecorder(recorder)

    policy_factory = make_policy(args.policy, args, bench)
    if args.policy == "bridge" and not args.socket:
        raise SystemExit("--policy bridge needs --socket")
    print(
        f"[collect] {len(seeds)} seeds -> {output}  task={args.task} policy={args.policy} "
        f"max_steps={args.max_steps} history={args.history_frames} exec={args.execute}",
        flush=True,
    )

    accepted = 0
    try:
        with labels_path.open("a" if args.resume else "w", encoding="utf-8") as labels:
            for i, seed in enumerate(seeds, 1):
                label = run_episode(
                    bench,
                    lambda seed=seed: policy_factory(seed),
                    seed,
                    i,
                    recorder=keep_all,
                    trace_dir=trace_dir,
                    task_suffix=args.task_suffix,
                )
                labels.write(json.dumps(label, ensure_ascii=False) + "\n")
                labels.flush()
                accepted += int(label["success"])
                print(
                    f"[{i}/{len(seeds)}] seed={seed} score={label['cookies_in_target']}"
                    f"/{label['required_cookies']} success={label['success']}"
                    f" steps={label['steps']} {label['wall_seconds']}s"
                    f" | accepted {accepted}/{i}",
                    flush=True,
                )
    finally:
        recorder.close()

    print(
        f"[collect] done: {len(seeds)} episodes, {accepted} strict successes "
        f"({100.0 * accepted / len(seeds):.1f} %), labels={labels_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
