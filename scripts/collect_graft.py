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
from a3_dual_arm_sim.batch_expert_2x10 import A3Cookie2x10RecoveryExpert  # noqa: E402
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

    #: Entries per history frame: ``[q, q_dot, a_previous]``, 16 each.  The training data's
    #: layout (`add_history_window.py`), and the reason the adapter's `history_dims` is
    #: ``frames * 48``.
    HISTORY_PER_FRAME = 48

    def __init__(self, path: str, chunk: int, history_frames: int = 10) -> None:
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
        #: Frames of proprioceptive history, 0 to send none.
        #:
        #: **This exists because its absence was a silent, total failure.**  The collector
        #: used to send four arrays -- three views and the 16 joint values -- and no
        #: ``history``.  The adapter's `_history_vector` catches the missing key and returns
        #: ``None`` (by design, so a pre-v6 checkpoint still loads), the batch then has no
        #: ``observation.history``, and the launcher's patched `embed_prefix` returns early.
        #: Net effect: **the history token was never appended**, so every episode collected
        #: before this was driven by a v9 whose prefix was one token shorter than the one it
        #: was trained with -- not "a policy with a zero history", a different architecture.
        #: Nothing raised, and the only symptom was a takeover rate that happened to look
        #: plausible (42 %, against 9.40's 46.7 % on a checkpoint that had no encoder at all).
        self.history_frames = history_frames
        self._history: list[np.ndarray] = []

    def begin(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.step = 0
        self._history = []

    def observe(self, observation: dict[str, Any], applied_action: Any) -> None:
        """Append one history frame for the control step that is about to be taken.

        Called on **every** step, not only when a request goes out.  The training data has
        one window per frame (`add_history_window.py::build_window`), so a buffer that only
        advanced once per chunk would span ``frames * execute`` = 160 steps instead of 10 and
        sample the trajectory at a different rate than the encoder was trained on.  That is
        what the benchmark did and, because this collector copied it, what it did too.

        ``applied_action`` is what the environment applied on the *previous* step, matching
        the training data's ``a_previous`` (the recorded `action` column shifted one row, and
        that column is the applied command).
        """

        if self.history_frames <= 0:
            return
        state = np.asarray(observation[STATE], dtype=np.float32).reshape(-1)[:16]
        velocity = np.asarray(
            observation.get("observation.velocity", np.zeros(16)), dtype=np.float32
        ).reshape(-1)[:16]
        previous = (
            np.zeros(16, dtype=np.float32)
            if applied_action is None
            else np.asarray(applied_action, dtype=np.float32).reshape(-1)[:16]
        )
        self._history.append(np.concatenate([state, velocity, previous]).astype(np.float32))
        if len(self._history) > self.history_frames:
            del self._history[0]

    def _history_vector(self) -> np.ndarray:
        """The buffered window, oldest first, leading frames repeated from the first.

        Term for term the training data's `build_window`: repeat frame zero rather than
        zero-fill, because "every joint at zero" is a pose the robot is never in and padding
        with it would teach the encoder that episode start looks like an impossible state.
        """

        frames = list(self._history)
        if not frames:
            return np.zeros(self.history_frames * self.HISTORY_PER_FRAME, dtype=np.float32)
        while len(frames) < self.history_frames:
            frames.insert(0, frames[0])
        return np.concatenate(frames)

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
        if self.history_frames > 0:
            arrays["history"] = self._history_vector()
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
        stall_steps: int = 0,
        hold_steps: int = 40,
        accept_tier_b: bool = False,
        expert_class: type | None = None,
    ) -> None:
        self.env = env
        self.client = client
        self.execute = execute
        self.policy_steps = policy_steps
        self.expert_steps = expert_steps
        self.delay_low = delay_low
        self.delay_high = delay_high
        #: Steps of no target growth that count as "a batch was taken and lost".  A whole
        #: attempt cycle rather than the handover delay -- see `_should_hand_over`.
        self.stall_steps = stall_steps
        self.hold_steps = hold_steps
        self.accept_tier_b = accept_tier_b
        #: Which controller takes over.  The 2x5 box needs the two-column `adopt`; the 2x10
        #: box needs one that plans five-Cookie groups from a twenty-slot occupancy.  Passed
        #: in rather than imported here because the two live in different modules and a
        #: default would silently collect the wrong task's recoveries under the right
        #: task's name.
        self.expert_class = expert_class or A3CookieBatchExpert
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
        #: Highest `cookies_in_target` seen and lowest `cookies_in_source` seen, with the
        #: step the target last improved.  The stall trigger reads the latter; see
        #: `_should_hand_over`.
        self.best_target = 0
        self.best_source = 1 << 30
        self.last_target_gain_step = 0
        self.last_progress_step = 0
        #: The action the environment applied on the previous step.  Carried into the next
        #: history frame as `a_previous`, matching the training data (whose `action` column
        #: is the recorded -- i.e. applied -- command, shifted one row).
        self.applied_action: np.ndarray | None = None

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
        # **Every control step, in both phases.**  The window is the robot's own recent
        # trajectory, so it has to advance with the environment rather than with the
        # policy's requests -- and during the expert phase the robot is still moving, so a
        # window that only advanced in phase one would be stale by the expert's duration.
        # (Nothing reads it after the handover today, because the policy never resumes; a
        # hand-back rotation would, and this keeps it correct for one.)
        self.client.observe(observation, self.applied_action)
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
        # The source has stopped shrinking: the policy has stopped reaching.
        if int(self.last_info.get("cookies_in_source", 0)) >= int(
            self.source_at_first_five or 0
        ):
            return True
        # Or **the policy took a batch and never delivered it**.  This is the case 9.40's
        # rule cannot see: the source *did* shrink (so the policy has not stopped reaching),
        # but the box never gained a Cookie.
        #
        # **Off by default, because it was measured and it loses.**  Over the same twelve
        # 2x5 seeds, enabling it (``--stall-steps 350``) took the acceptance rate from
        # 5/12 to 4/12 and the tier-A rate among triggered episodes from 5/8 to 4/10.  The
        # episodes it adds are the ones the expert then makes *worse*: seed 2 triggered with
        # four slots already filled and ended with **zero**, because the group of five came
        # apart during transport and scattered the box.  A policy that is still reaching is
        # usually still making progress somewhere, and taking the box off it costs more than
        # the batch it was going to lose.
        #
        # The window also has to be a **whole attempt cycle**, not the drawn delay: a batch
        # is grasp (source falls) -> carry -> release (target rises), and the middle moves
        # neither number for ~150-250 steps.  With the delay as the window, three of three
        # episodes took over *while the policy was still carrying its second batch*, and the
        # expert then failed in `MOVE_TO_SLOT` or `DESCEND_TO_PLACE`.
        if self.stall_steps <= 0:
            return False
        if int(self.last_info.get("cookies_in_source", 0)) <= int(
            self.source_at_first_five or 0
        ) - 2:
            return (self.steps - self.last_target_gain_step) >= self.stall_steps
        return False

    def _hand_over(self) -> None:
        # `adopt` rather than the baseline mode: the expert must read the occupancy
        # the policy left and place its five into the free column, not act as if the
        # first batch had gone in cleanly.
        self.env.second_batch_mode = "adopt"
        expert = self.expert_class(self.env)
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
            # **The scene the expert is about to work in.**  The expert never reads the
            # policy's outputs, so anything that changes its success rate has to arrive
            # through the box -- and until this was recorded, that was the one quantity in
            # the whole pipeline nothing measured.  See `_source_report`.
            **self._source_report(expert),
        }

    def _source_report(self, expert: Any) -> dict[str, Any]:
        """How disturbed the source box is at the moment of takeover.

        Written because the expert's success rate fell from 4-5 per run to zero when the
        policy started receiving its history channel, and every existing number in this
        collector is about the *target* box -- what the policy left behind there, how many
        slots it filled.  The expert takes its five out of the **source**, and nothing
        measured the source.

        Every threshold here is the expert's own, so a count means the same thing to this
        report as it does to `_candidate_batches`: a Cookie is *placeable* when it is inside
        the source, upright, and within `SOURCE_IN_PLACE_TOLERANCE_M` of where the layout
        put it.  A run of five placeable neighbours is what the expert needs, so
        `src_runs` is the length of the longest such run per column -- the number that
        decides whether a batch can be formed at all.

        `jaw_gap_mm` is the other half and is a different question: the five in the jaws can
        be placeable and still fail, because the pads stand *outside* the row and the
        neighbour past each end is in their path.  Measured against the tightest neighbour
        beside the held group; a value near zero is the geometry that produces
        `DESCEND` blocked at 20-75 N.
        """

        env = self.env
        source = np.asarray(env.SOURCE_POSITIONS, dtype=np.float64)
        actual = np.asarray(env.cookie_positions, dtype=np.float64)
        half = np.asarray(env.COOKIE_HALF_SIZE, dtype=np.float64).ravel()
        bodies = list(env._cookie_bodies)  # noqa: SLF001 - the judge's own objects

        def upright(index: int) -> bool:
            # The expert's test: the Cookie's own z axis still points up.
            return abs(float(env.data.xmat[bodies[index], 8])) >= float(
                np.cos(np.deg2rad(10.0))
            )

        def in_place(index: int, tolerance: float = 0.0016) -> bool:
            return float(np.linalg.norm(actual[index][:2] - source[index][:2])) <= tolerance

        laid_out = range(min(len(source), len(actual)))
        inside_source = [i for i in laid_out if env.privileged_cookie_in_source(i)]
        standing = [i for i in inside_source if upright(i)]
        placeable = [i for i in standing if in_place(i)]

        # Longest run of placeable neighbours per column, which is what a grasp needs.
        runs: list[int] = []
        for x in sorted(set(float(v) for v in source[:, 0])):
            column = sorted(
                (i for i in range(len(source)) if np.isclose(source[i, 0], x)),
                key=lambda i: float(source[i, 1]),
            )
            best = run = 0
            for index in column:
                run = run + 1 if index in placeable else 0
                best = max(best, run)
                if index not in placeable:
                    run = 0
            runs.append(int(best))

        held = list(getattr(expert, "_cookies_in_jaws", lambda: [])())
        gap_mm = -1.0
        if held:
            held_ys = np.sort(actual[held, 1])
            others = [i for i in inside_source if i not in held]
            if others:
                other_ys = actual[others, 1]
                below = other_ys[other_ys < held_ys[0]]
                above = other_ys[other_ys > held_ys[-1]]
                gaps = []
                if below.size:
                    gaps.append(float(held_ys[0] - below.max()))
                if above.size:
                    gaps.append(float(above.min() - held_ys[-1]))
                if gaps:
                    gap_mm = round(min(gaps) * 1000.0, 2)

        return {
            "src_inside": len(inside_source),
            "src_upright": len(standing),
            "src_placeable": len(placeable),
            "src_runs": runs,
            "jaws_held": len(held),
            "jaw_gap_mm": gap_mm,
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
        del requested
        self.steps += 1
        # Before anything else: the next `act` builds its history frame from this.
        if applied is not None:
            self.applied_action = np.asarray(applied, dtype=np.float32).copy()
        self.last_info = info
        target = int(info.get("cookies_in_target", 0))
        source = int(info.get("cookies_in_source", 0))
        if self.first_five is None and target >= 5:
            self.first_five = self.steps
            self.source_at_first_five = source
        # **A policy can keep taking from the source and still be stuck.**  The handover
        # used to require the source count to stop falling, which describes a policy that
        # has given up reaching -- not one that reaches, misses, and reaches again.  Measured
        # on v9 over 2x5 seeds 0-11: seed 2 ended with five in the box, five in slots and
        # `graft_triggered=False`, because it was still removing Cookies from the source
        # every batch and placing none of them.  So *either* number improving counts as
        # progress, and the trigger reads when progress last happened.
        if target > self.best_target:
            self.best_target = target
            self.last_target_gain_step = self.steps
            self.last_progress_step = self.steps
        if source < self.best_source:
            self.best_source = source
            self.last_progress_step = self.steps
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
        # **The target is the scene's number, not ten.**  This read `final_target >= 10`
        # because every scene it was written for was the 2x5 box, and on the twenty-slot box
        # that makes a half-finished episode look finished -- and worse, `success_override`
        # trusts this verdict, so it would accept an episode the environment itself refused.
        required = int(getattr(self.env.task_config, "required_cookies", 10))
        final_target = int(self.last_info.get("cookies_in_target", 0))
        # **Progress is measured in Cookies that arrived, not slots they landed on.**  A
        # five-Cookie group placed from one wide grip lands 3-5 of them on slot centres even
        # when all five are inside and upright -- measured on the 2x10 probe as +5 in the box
        # with +1, +3 and +4 in slots -- so a slot-gain rule calls a clean recovery a `B` for
        # a reason that is the *policy's* placement quality, not the expert's.  What makes an
        # episode worth learning from is that the group arrived *and* the expert did not
        # report a placement failure; `VERIFY_RELEASE` already tests upright, settled and
        # contained, so a group that merely landed on the box floor still reads `B`.
        target_gained = final_target - int(self.takeover.get("in_target") or 0)
        expert_failed = bool(getattr(self.expert, "failed", False))
        if final_target >= required and target_gained >= 5 and not expert_failed:
            return "A"
        if final_target >= required:
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
            # The final count is recorded because `tier` reads it and a tier without the
            # numbers behind it cannot be re-derived once the log is gone.
            "graft_in_target_final": int(self.last_info.get("cookies_in_target", 0)),
            "graft_target_column": self.takeover.get("target_column"),
            "graft_adopted_cookies": self.takeover.get("adopted_cookies"),
            # The source box at takeover, which is what the expert actually works in.  These
            # are the fields that make a "why did the expert fail" question answerable from
            # the metadata instead of from a re-run; see `_source_report`.
            "graft_src_inside": self.takeover.get("src_inside"),
            "graft_src_upright": self.takeover.get("src_upright"),
            "graft_src_placeable": self.takeover.get("src_placeable"),
            "graft_src_runs": self.takeover.get("src_runs"),
            "graft_jaws_held": self.takeover.get("jaws_held"),
            "graft_jaw_gap_mm": self.takeover.get("jaw_gap_mm"),
            "graft_in_source_at_takeover": self.takeover.get("in_source"),
            "graft_in_source_final": int(self.last_info.get("cookies_in_source", 0)),
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
    parser.add_argument(
        "--history-frames",
        type=int,
        default=10,
        help=(
            "Frames of proprioceptive history to send, 0 for none.  Must match the checkpoint: "
            "a model with a history encoder expects frames*48 values, and sending none does "
            "NOT fail -- the adapter treats the absent vector as 'old checkpoint', the batch "
            "has no observation.history, and the launcher's embed_prefix returns early, so "
            "the history token is simply never appended.  That silent path is what this "
            "default exists to close: every episode collected before it was driven by a "
            "shorter prefix than the checkpoint was trained with."
        ),
    )
    parser.add_argument("--policy-steps", type=int, default=700)
    parser.add_argument("--expert-steps", type=int, default=700)
    parser.add_argument("--delay-low", type=int, default=100)
    parser.add_argument("--delay-high", type=int, default=200)
    parser.add_argument(
        "--stall-steps",
        type=int,
        default=0,
        help=(
            "Steps without a Cookie entering the box that count as a lost batch.  0 (the "
            "default) disables the rule.  Enabling it was measured to *lower* the acceptance "
            "rate (5/12 -> 4/12 over the same seeds) because the extra takeovers are states "
            "the expert then scatters; see `_should_hand_over`.  When enabled it must be a "
            "whole attempt cycle (>= 300), never the handover delay: the carry between a "
            "grasp and a release moves neither count for 150-250 steps."
        ),
    )
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--accept-tier-b", action="store_true")
    parser.add_argument(
        "--expert",
        choices=("2x5", "2x10"),
        default=None,
        help=(
            "Which controller takes over.  Default: inferred from `--scene` -- "
            "`a3_cookie_2x10_recovery` means `2x10`, anything else means `2x5`.  They are not "
            "interchangeable: the 2x5 `adopt` decides a *column* (five slots is a whole "
            "column there), while the 2x10 box is two columns of ten and needs a plan built "
            "from the occupancy's contiguous free rows."
        ),
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        help=(
            "The sentence the *policy* is driven with.  Default is the scene's own prompt, "
            "which is this branch's training sentence -- and for any checkpoint that was "
            "fine-tuned on main's benchmark sentence that is the wrong one.  It is not a "
            "cosmetic difference: 9.42 measured the same weights at 7/20 under the sentence "
            "they had seen and 0/20 under the other.  Phase one is the policy, so a collector "
            "that gets this wrong measures a policy that is not the one being recovered, and "
            "every takeover state is drawn from a distribution it never occupies at "
            "evaluation.  The expert phase is unaffected -- it reads no language."
        ),
    )
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
            # `%%`, not `%`: argparse runs every help string through `%`-formatting, so a
            # literal per-cent followed by `)` raises "unsupported format character" and
            # `--help` dies with a traceback instead of printing.  Latent until now because
            # nothing had ever run `--help` on this script.
            "takeover rate this collector was built around (50%%) was measured with them on: "
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
    client = Client(args.socket, args.chunk, history_frames=args.history_frames)

    base = scene_by_name(args.scene)

    # Which controller takes over, resolved once so a mistyped `--scene` cannot silently
    # collect one task's recoveries under the other's name.
    expert_arm = args.expert or ("2x10" if "2x10" in args.scene else "2x5")
    expert_class = A3Cookie2x10RecoveryExpert if expert_arm == "2x10" else A3CookieBatchExpert
    if expert_arm == "2x10" and "2x10" not in args.scene:
        raise SystemExit(
            "--expert 2x10 needs a twenty-slot scene; the 2x5 box has ten slots and the "
            "recovery expert plans against the layout's own columns"
        )
    if expert_arm == "2x5" and "2x10" in args.scene:
        raise SystemExit(
            "a twenty-slot scene cannot be recovered by the 2x5 expert: its adopt decides a "
            "column, and in a ten-row column that lands the group on the Cookies already there"
        )

    def build_policy(env: Any) -> GraftPolicy:
        return GraftPolicy(
            env,
            client,
            execute=args.execute,
            policy_steps=args.policy_steps,
            expert_steps=args.expert_steps,
            delay_low=args.delay_low,
            delay_high=args.delay_high,
            stall_steps=args.stall_steps,
            hold_steps=args.hold_steps,
            accept_tier_b=args.accept_tier_b,
            expert_class=expert_class,
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
        # The policy is driven with `--task-prompt` when one is given; the recorder writes
        # that same sentence, so an episode's own instruction matches the one that produced
        # it.  `replace` on the frozen `TaskSpec` rather than on the scene, because the
        # runner reads `scene.task.prompt` and nothing else.
        task=replace(base.task, prompt=args.task_prompt) if args.task_prompt else base.task,
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
    # Stated, not inferred: whether the policy is being driven as the architecture it was
    # trained as is the single fact that decides whether this run's numbers mean anything,
    # and the failure mode (no history token, no error) leaves no trace in the episode.
    if args.history_frames > 0:
        width = args.history_frames * Client.HISTORY_PER_FRAME
        print(
            f"history            : {args.history_frames} frames ({width}-wide "
            f"observation.history) -- must match the checkpoint's encoder",
            flush=True,
        )
    else:
        print(
            "history            : OFF -- any checkpoint with a history encoder will be "
            "driven as a shorter-prefix model (see --history-frames)",
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
