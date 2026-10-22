#!/usr/bin/env python3
"""Collect grafted demonstrations: the policy's first batch, the expert's second.

The datasets so far contain one behaviour each -- all expert, or all policy -- and
neither fixes the failure this project is on: a fine-tune that places the first
five and then does nothing useful with the second five.  The epiphanic measurement
(``docs/pi05-training-history.md`` section 9.35-9.36) is that the *state* after a
policy's own first batch is recoverable; the expert can finish from it, at a rate
worth collecting from.  So this collector records exactly that episode:

    policy phase   4.05 (or any bridge model) drives from the empty scene until it
                   stops making progress on the source -- the same trigger the gate
                   uses, so the handover state here is the handover state measured
                   there;
    expert phase   `A3CookieBatchExpert` in ``adopt`` mode takes over from that
                   state and places the second batch without moving anything else.

Both phases go into one trajectory, so a policy trained on it learns the state it
actually produces *and* what to do from it -- which is the whole point of the
graft, and is what a pure-expert dataset cannot teach (its first batch is
impeccable, so it never visits the state a policy leaves).

Acceptance is the scene's own success contract (ten Cookies released in the
target box, seventy left in the source), the same rule every dataset here was
collected under.  The tier is recorded per episode so a reader can tell a clean
graft (``A``: the expert's five reached slots) from a box that merely got full
(``B``), and a policy that finished the job itself (``policy_only``) from either.

Usage:

    python scripts/collect_graft.py \\
        --root /root/autodl-tmp/datasets/graft_v1/shard_00 \\
        --repo-id local/a3_graft_v1-shard-00 \\
        --episodes 25 --max-attempts 100 --shard-index 0 --shard-count 8

The bridge must already be serving a model (``scripts/a3_bridge_server.py
--listen unix:///tmp/probe.sock --adapter ... --config ckpt=...``): this process
never loads the policy, it only sends observations and receives action chunks.
Several shards may share one server; the server holds the model once and
serialises the calls.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_dual_arm_sim.batch_expert import A3CookieBatchExpert  # noqa: E402
from a3_dual_arm_sim.bridge.protocol import (  # noqa: E402
    decode_message,
    encode_request,
    read_frame,
    write_frame,
)
from a3_dual_arm_sim.collection import collect_dataset, scene_by_name  # noqa: E402
from a3_dual_arm_sim.env import (  # noqa: E402
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
)
from a3_dual_arm_sim.main_alignment import apply_scene, load_scene  # noqa: E402

#: main's `basic` profile, term for term (9.22).  The arrangement is main's; the
#: variation is these sigmas, drawn per seed on top of it.  Same numbers the gate
#: measured the takeover rate under, so this run's episodes are the ones that rate
#: describes rather than a new distribution nobody has measured.
BASIC = dict(
    bin_noise_m=0.01,
    bin_noise_dist="uniform",
    bin_noise_horizontal_only=True,
    cookie_noise_m=0.0003,
    cookie_noise_dist="uniform",
    cookie_noise_horizontal_only=True,
    cookie_yaw_noise_rad=0.015,
    bin_yaw_noise_rad=0.03,
    pose_noise_rad=0.0,
)


class Client:
    """One connection to the bridge, handing over observations and taking chunks."""

    def __init__(self, path: str, chunk: int) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.reader = self.sock.makefile("rb")
        self.writer = self.sock.makefile("wb")
        kind, _, _ = decode_message(read_frame(self.reader))
        if kind != "ready":
            raise RuntimeError(f"expected 'ready' from the bridge, got {kind!r}")
        self.chunk = chunk
        self.step = 0
        self.episode_id = ""
        self.calls = 0

    def begin(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.step = 0

    def infer(self, observation: dict[str, Any], task: str) -> np.ndarray:
        def view(key: str) -> np.ndarray:
            image = np.asarray(observation[key])
            if image.shape[:2] != (224, 224):
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(image, dtype=np.uint8)

        arrays = {
            "front_view": view(FRONT_IMAGE),
            "left_wrist_view": view(LEFT_WRIST_IMAGE),
            "right_wrist_view": view(RIGHT_WRIST_IMAGE),
            "observation.state": np.asarray(observation[STATE], dtype=np.float32).reshape(16),
        }
        write_frame(
            self.writer,
            encode_request(
                episode_id=self.episode_id,
                step=self.step,
                task=task,
                arrays=arrays,
                chunk=self.chunk,
            ),
        )
        self.step += 1
        self.calls += 1
        kind, _fields, reply = decode_message(read_frame(self.reader))
        if kind == "error":
            raise RuntimeError(f"bridge reported {reply!r}")
        if kind != "actions" or "actions" not in reply:
            raise RuntimeError(f"bridge replied {kind!r} without actions")
        return np.asarray(reply["actions"], dtype=np.float64)

    def close(self) -> None:
        try:
            self.writer.close()
            self.reader.close()
            self.sock.close()
        except OSError:
            pass


class GraftPolicy:
    """Phase one the bridge policy, phase two the expert; one trajectory out.

    The runner drives this exactly as it drives a scripted expert: ``act`` returns
    the next command, and everything the episode is worth knowing is in ``metrics``
    and ``success_override``.  The handover rule is the gate's, unchanged
    (``/tmp/gate_takeover.py``): once five Cookies are in the target, wait a
    seed-drawn U[delay_low, delay_high] steps, and hand over at the first step the
    source has stopped shrinking.  Two conditions rather than one because the
    policy often pauses mid-batch with five already down; handing over there would
    abandon a batch it was about to finish.
    """

    action_mode = "joint_position"

    def __init__(
        self,
        env: Any,
        client: Client,
        *,
        execute: int,
        policy_steps: int,
        expert_steps: int,
        delay_low: int = 100,
        delay_high: int = 200,
        hold_steps: int = 40,
        accept_tier_b: bool = False,
    ) -> None:
        self.env = env
        self.client = client
        self.execute = execute
        self.policy_steps = policy_steps
        self.expert_steps = expert_steps
        self.delay_low = delay_low
        self.delay_high = delay_high
        self.hold_steps = hold_steps
        self.accept_tier_b = accept_tier_b
        self.expert: A3CookieBatchExpert | None = None
        self._reset_state()

    # -- lifecycle ---------------------------------------------------------
    def _reset_state(self) -> None:
        self.queue: list[np.ndarray] = []
        self.phase = "policy"
        self.steps = 0
        self.policy_steps_taken = 0
        self.expert_steps_taken = 0
        self.first_five: int | None = None
        self.source_at_first_five: int | None = None
        self.triggered = False
        self.stop = False
        self.failed = False
        self.done_steps = 0
        self.completed_step: int | None = None
        self.takeover: dict[str, Any] = {}
        self.last_action: np.ndarray | None = None
        self.last_info: dict[str, Any] = {}
        self.delay = self.delay_low
        self.seed = None
        self.error: str | None = None

    def reset(self, context: Any = None) -> None:
        self._reset_state()
        self.seed = getattr(context, "seed", None)
        rng = np.random.default_rng(10_000 + int(self.seed or 0))
        self.delay = int(rng.integers(self.delay_low, self.delay_high + 1))
        self.client.begin(f"graft-seed{self.seed}")
        if self.expert is not None:
            self.expert = None

    def close(self) -> None:
        return None

    # -- the two phases ----------------------------------------------------
    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        if self.phase == "policy":
            if self.policy_steps_taken >= self.policy_steps:
                # The trigger never fired: the policy is neither finishing nor
                # stalling, and waiting out the horizon records a stuck arm.
                self.failed = True
                self.stop = True
                return self._hold()
            if self._should_hand_over():
                self._hand_over()
                return self._expert_act(observation, task)
            if self.completed_step is not None and self.steps - self.completed_step >= self.hold_steps:
                # The policy finished the job itself.  Keep driving for the same hold window
                # the expert gets, because the contract's own success flag is counted over
                # stable steps and a cookie still pinched when the last command lands does not
                # count as released.  Measured: one such episode ended 10/10 in the target with
                # success=False, i.e. a real success thrown away by stopping one command early.
                self.stop = True
            if not self.queue:
                actions = self.client.infer(observation, task)
                self.queue = [np.asarray(a, dtype=np.float64) for a in actions[: self.execute]]
            action = self.queue.pop(0)
            self.policy_steps_taken += 1
            self.last_action = action
            return action
        return self._expert_act(observation, task)

    def _should_hand_over(self) -> bool:
        if self.first_five is None:
            return False
        if self.steps - self.first_five < self.delay:
            return False
        return int(self.last_info.get("cookies_in_source", 0)) >= int(
            self.source_at_first_five or 0
        )

    def _hand_over(self) -> None:
        # `adopt` rather than the baseline mode: the expert must read the occupancy
        # the policy left and place its five into the free column, not act as if the
        # first batch had gone in cleanly.
        self.env.second_batch_mode = "adopt"
        expert = A3CookieBatchExpert(self.env)
        try:
            expert.reset()
        except Exception as exc:  # noqa: BLE001 - a refused layout is a rejected attempt
            self.error = f"{type(exc).__name__}: {exc}"
            self.failed = True
            self.stop = True
            self.expert = expert
            return
        self.expert = expert
        self.phase = "expert"
        self.triggered = True
        inside = tuple(
            self.env._cookie_inside_target(i) for i in range(len(self.env.cookie_positions))
        )
        slots = sum(1 for index in self.env._target_slot_occupancy(inside) if index >= 0)
        self.takeover = {
            "policy_steps": self.policy_steps_taken,
            "first_five_step": self.first_five,
            "delay_draw": self.delay,
            "in_target": int(self.last_info.get("cookies_in_target", 0)),
            "in_source": int(self.last_info.get("cookies_in_source", 0)),
            "slots": slots,
            "target_column": getattr(expert, "target_column", None),
            "adopt_infeasible": bool(getattr(expert, "adopt_infeasible", False)),
            "adopted_cookies": len(getattr(expert, "completed_cookie_indices", ())),
        }

    def _expert_act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        expert = self.expert
        if expert is None or expert.failed or self.error is not None:
            self.failed = True
            self.stop = True
            return self._hold()
        if expert.finished:
            self.done_steps += 1
            if self.done_steps >= self.hold_steps:
                self.stop = True
        if self.expert_steps_taken >= self.expert_steps:
            self.failed = True
            self.stop = True
            return self._hold()
        try:
            action = expert.act(observation, task)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            self.error = f"{type(exc).__name__}: {exc}"
            self.failed = True
            self.stop = True
            return self._hold()
        self.expert_steps_taken += 1
        self.last_action = np.asarray(action, dtype=np.float64)
        return self.last_action

    def _hold(self) -> np.ndarray:
        if self.last_action is not None:
            return self.last_action
        return np.zeros(16, dtype=np.float64)

    # -- what the runner reads --------------------------------------------
    def observe_step(self, *, requested: Any, applied: Any, info: dict[str, Any]) -> None:
        del requested, applied
        self.steps += 1
        self.last_info = info
        target = int(info.get("cookies_in_target", 0))
        source = int(info.get("cookies_in_source", 0))
        if self.first_five is None and target >= 5:
            self.first_five = self.steps
            self.source_at_first_five = source
        if self.completed_step is None and bool(info.get("success", False)):
            self.completed_step = self.steps

    @property
    def stop_requested(self) -> bool:
        return self.stop

    def _slots_now(self) -> int:
        try:
            inside = tuple(
                self.env._cookie_inside_target(i) for i in range(len(self.env.cookie_positions))
            )
            return sum(1 for index in self.env._target_slot_occupancy(inside) if index >= 0)
        except Exception:  # noqa: BLE001 - a torn-down env must not mask the result
            return -1

    @property
    def tier(self) -> str:
        if not self.triggered:
            return "no_trigger" if self.failed else "policy_only"
        final_slots = self._slots_now()
        final_target = int(self.last_info.get("cookies_in_target", 0))
        gained = final_slots - int(self.takeover.get("slots", 0))
        if final_target >= 10 and gained >= 5:
            return "A"
        if final_target >= 10:
            return "B"
        return "C"

    @property
    def success_override(self) -> bool:
        """Accepted when the episode reached the contract *and* is worth learning from.

        A grafted episode whose second batch went into the box without reaching
        slots teaches a placement precisely as well as a failure does, so ``B`` is
        refused by default and counted rather than silently kept.  ``--accept-tier-b``
        exists because the scene's own contract does not distinguish them, and a
        caller who only cares about the benchmark score may disagree.
        """

        tier = self.tier
        if tier == "A":
            return True
        if tier == "B":
            return self.accept_tier_b
        if tier == "policy_only":
            return bool(self.last_info.get("success", False))
        return False

    @property
    def metrics(self) -> dict[str, Any]:
        expert = self.expert
        final_slots = self._slots_now()
        return {
            "graft_tier": self.tier,
            "graft_triggered": self.triggered,
            "graft_failed": self.failed,
            "graft_error": self.error,
            "graft_policy_steps": self.policy_steps_taken,
            "graft_expert_steps": self.expert_steps_taken,
            "graft_model_calls": self.client.calls,
            "graft_delay_draw": self.delay,
            "graft_first_five_step": self.first_five,
            "graft_completed_step": self.completed_step,
            "graft_total_steps": self.steps,
            "graft_slots_at_takeover": self.takeover.get("slots"),
            "graft_slots_final": final_slots,
            "graft_in_target_at_takeover": self.takeover.get("in_target"),
            "graft_target_column": self.takeover.get("target_column"),
            "graft_adopted_cookies": self.takeover.get("adopted_cookies"),
            "graft_expert_failure_phase": getattr(expert, "failure_phase", None),
            "graft_expert_failure_reason": getattr(expert, "failure_reason", None),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="dataset root for this shard")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, required=True, help="accepted episodes wanted")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="first seed of the sweep")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--socket", default="/tmp/probe.sock")
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--execute", type=int, default=16, help="env steps per 50-action chunk")
    parser.add_argument("--policy-steps", type=int, default=700)
    parser.add_argument("--expert-steps", type=int, default=700)
    parser.add_argument("--delay-low", type=int, default=100)
    parser.add_argument("--delay-high", type=int, default=200)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--accept-tier-b", action="store_true")
    parser.add_argument("--scene", default="a3_cookie_batch_second_batch_aligned")
    parser.add_argument(
        "--scene-dump",
        type=Path,
        default=Path(__file__).resolve().parent / "main_bench_scene.json",
        help=(
            "main's bench arrangement: every Cookie, bin and camera pose, dumped from "
            "main's own CookieBatchBenchmark.  In the repository rather than /tmp so a "
            "rerun on another machine reaches the same scene."
        ),
    )
    parser.add_argument("--position-noise", type=float, default=BASIC["cookie_noise_m"])
    parser.add_argument("--yaw-noise", type=float, default=BASIC["cookie_yaw_noise_rad"])
    parser.add_argument(
        "--fast-render",
        dest="fast_render",
        action="store_true",
        default=False,
        help=(
            "Render without the shadow and reflection passes.  Off by default because the "
            "takeover rate this collector was built around (50%) was measured with them on: "
            "measured here, two seeds under fast render ended 700 steps with in_target=0, the "
            "same two under full render reached 5 and 10.  Turning it back on is a different "
            "observation distribution, not a faster version of this one."
        ),
    )
    parser.add_argument("--no-fast-render", dest="fast_render", action="store_false")
    parser.add_argument("--videos", dest="videos", action="store_true", default=True)
    parser.add_argument("--no-videos", dest="videos", action="store_false")
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--rollover-mb", type=float, default=500.0)
    parser.add_argument("--limit-attempts", type=int, default=None, help="stop after N attempts")
    args = parser.parse_args()

    scene_dump = load_scene(str(args.scene_dump))
    client = Client(args.socket, args.chunk)

    base = scene_by_name(args.scene)

    def build_policy(env: Any) -> GraftPolicy:
        return GraftPolicy(
            env,
            client,
            execute=args.execute,
            policy_steps=args.policy_steps,
            expert_steps=args.expert_steps,
            delay_low=args.delay_low,
            delay_high=args.delay_high,
            hold_steps=args.hold_steps,
            accept_tier_b=args.accept_tier_b,
        )

    def post_reset(env: Any, seed: int) -> None:
        # main's arrangement, jittered by this seed's draw of the BASIC profile.  Done
        # after `env.reset`, because reset re-derives the layout, and re-read by the
        # runner afterwards, because the observation reset returned describes the old
        # one.
        apply_scene(env, scene_dump, rng=np.random.default_rng(int(seed)), **BASIC)
        # Phase one starts from the empty bench.  The scene draws one of four
        # second-batch conditions per seed, and every other one of them has the expert
        # *prefill* the box -- which would hand the policy a first batch it did not do.
        env.second_batch_mode = "full"

    details = dict(base.details)
    details.update(
        {
            "graft": "policy phase then A3CookieBatchExpert in adopt mode",
            "graft_socket": args.socket,
            "graft_execute": args.execute,
            "graft_chunk": args.chunk,
            "graft_delay_range": [args.delay_low, args.delay_high],
            "graft_accept_tier_b": bool(args.accept_tier_b),
            "post_reset_profile": "main basic randomisation",
        }
    )
    scene = replace(
        base,
        build_policy=build_policy,
        details=details,
        # The progress line prints these per attempt, which is the only way a run
        # that is going badly says *why* while it is still going: "success=False"
        # alone cannot distinguish a policy that never placed five from an expert
        # that was handed a mess.
        metric_keys=(
            "cookies_in_target",
            "cookies_in_source",
            "graft_tier",
            "graft_triggered",
            "graft_policy_steps",
            "graft_expert_steps",
            "graft_model_calls",
            "graft_slots_at_takeover",
            "graft_slots_final",
            "graft_first_five_step",
            "graft_expert_failure_phase",
        ),
    )

    description = {
        "source": "main's CookieBatchBenchmark arrangement, pinned",
        "profile": dict(BASIC),
        "second_batch_mode": "full for phase one, adopt for phase two",
    }

    started = time.time()
    print(
        f"graft shard {args.shard_index}/{args.shard_count} "
        f"episodes={args.episodes} seeds from {args.seed} socket={args.socket}",
        flush=True,
    )
    summary = None
    try:
        summary = collect_dataset(
            args.root,
            scene,
            repo_id=args.repo_id or f"local/graft-shard-{args.shard_index:02d}",
            episodes=args.episodes,
            start_seed=args.seed,
            max_attempts=args.max_attempts or args.episodes * 4,
            position_noise_m=args.position_noise,
            yaw_noise_rad=args.yaw_noise,
            hold_steps=args.hold_steps,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            fast_render=args.fast_render,
            use_videos=args.videos,
            save_failed_episodes=args.keep_failures,
            rollover_mb=args.rollover_mb,
            post_reset=post_reset,
            post_reset_description=description,
        )
    finally:
        elapsed = time.time() - started
        if summary is None:
            path = Path(args.root) / "collection_summary.json"
            summary = json.loads(path.read_text()) if path.exists() else {}
        accepted = summary.get("accepted_episodes")
        attempts = summary.get("attempts")
        print(
            f"graft shard {args.shard_index}/{args.shard_count}: "
            f"{accepted}/{args.episodes} accepted in {attempts} attempts, "
            f"{elapsed / 60:.1f} min"
            f"{'' if accepted == args.episodes else '  (short of the quota)'}",
            flush=True,
        )
        client.close()
    return 0 if summary.get("accepted_episodes") == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
