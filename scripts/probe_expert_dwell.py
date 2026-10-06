"""Which phase of the scripted expert owns the stand-still, and how much of the episode.

The claim this exists to check
------------------------------
The A3 Cookie demonstrations dwell: on **57.2%** of frames the expert commands the arm to
stay where it is (`|action - state| < 0.20` of the per-step cap), against **11.7%** for
LIBERO's human teleoperation, and the distribution is bimodal -- only 4% of frames sit
between 20% and 50% of the cap, so the expert either stands still or moves at 70-87%.  Under
an MSE-style objective the conditional mean of a distribution that is 57% "do not move" is
very close to "do not move", and measured, that is what every checkpoint trained on it
does: it collapses towards an identity map, reproducing the state to 0.001 rad on hold
frames while getting the residual -- the 4-9% of the signal that is the actual task --
**58 degrees wrong**.

That argument has a hole in it, and it is the reason this file exists.  It is a statement
about a *fraction*, and a fraction is not actionable: "reduce the dwell" needs to name what
to change.  The obvious candidate was `hold_steps`, which appears in
`collection_summary.json` as 40 -- but reading the code, `hold_steps` is not the dwell at
all.  It is the length of the recording tail after the expert declares DONE
(`CookieCollectionPolicy.stop_requested`, "down by the time the expert finishes its
retract").  There is no dwell parameter anywhere: `CookieTransferTaskConfig`'s
`success_hold_steps` is the *task's* criterion that a Cookie stays in the bin for 20 steps,
not a pacing the expert waits out.

The dwell is emergent.  The expert is a sixteen-phase state machine
(`expert.py`, `CookiePhase`), and the still frames are phases waiting on a condition --
`VERIFY_GRASP`, `VERIFY_LIFT`, `VERIFY_RELEASE` -- where the arm has arrived and the
controller keeps commanding the pose it is already in.  So the question "how do we reduce
it" has an answer only after "which phases produce it".

What it measures
----------------
Runs the untouched expert through the real `EpisodeRunner` and records, per control step,
the phase that produced the action, the state, and the **applied** action (the row the
dataset would have stored).  Then reports, per phase: how many steps it owns, and what
fraction of them are stand-still at the same thresholds the dataset probe uses.

Nothing about the run is special-cased: the same scene builder, the same expert, the same
runner, the same seeds-as-episodes convention.  The only addition is observation, through
the `observe_step` hook `runner.py` already reads duck-typed off the policy plus a wrapper
around `act` (which is where the state is, since `observe_step` is handed the actions and
`info` but not the observation).

Rendering is turned off, so this needs no GPU and can run beside a training job.

Usage::

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/probe_expert_dwell.py --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from a3_dual_arm_sim.collection import scene_by_name  # noqa: E402
from a3_dual_arm_sim.contracts import STATE  # noqa: E402
from a3_dual_arm_sim.runner import EpisodeRunner  # noqa: E402

#: Fractions of the per-step cap below which a step counts as "standing still".  The same
#: three the LIBERO comparison uses, so the numbers line up with the 15.8% / 23.9% / 57.2%
#: that started this.  The per-step cap is `max_joint_step_rad` (0.05 rad here, 1.0 for
#: LIBERO's normalised delta), and the measurement is the largest single-joint excursion.
DWELL_THRESHOLDS = (0.02, 0.05, 0.20)


class _PhaseLog:
    """Per-step (phase, state, applied action) for one episode, filled from two hooks.

    Two hooks because neither alone has everything: `act` is the only place the observation
    is, and `observe_step` is the only place the applied action is.  They are called once
    each per step, `act` first, so the pairing is positional rather than by matching values.
    """

    def __init__(self, policy: object) -> None:
        self.policy = policy
        #: The bound method being replaced.  `self.policy.act` cannot be called from inside
        #: the wrapper -- by then it *is* the wrapper -- so the original has to be held here.
        self.original_act = policy.act  # type: ignore[attr-defined]
        self.phase: list[str] = []
        self.state: list[np.ndarray] = []
        self.requested: list[np.ndarray] = []
        self.applied: list[np.ndarray] = []
        self._pending: tuple[str, np.ndarray, np.ndarray | None] | None = None

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        phase = getattr(getattr(self.policy, "expert", None), "phase", None)
        self._pending = (
            "DONE" if phase is None else str(getattr(phase, "name", phase)),
            np.asarray(observation[STATE], dtype=np.float64),
            None,
        )
        action = self.original_act(observation, task)
        name, state, _ = self._pending  # type: ignore[misc]
        self._pending = (name, state, np.asarray(action, dtype=np.float64))
        return action

    def observe_step(self, *, requested: np.ndarray, applied: np.ndarray, info: dict) -> None:
        del requested, info
        if self._pending is None:
            return
        name, state, requested_action = self._pending
        self.phase.append(name)
        self.state.append(state)
        self.requested.append(
            np.asarray(requested_action if requested_action is not None else applied,
                       dtype=np.float64)
        )
        self.applied.append(np.asarray(applied, dtype=np.float64))
        self._pending = None


def _displacement(state: np.ndarray, applied: np.ndarray, cap: float) -> float:
    """Largest single-arm-joint excursion, as a fraction of the per-step cap.

    The first seven entries of both vectors are the left arm's joints -- the arm the task
    uses and the one the wrist camera is on -- and the eighth is its gripper.  Only the
    joints are compared: a gripper command is not a step of the arm.
    """

    return float(np.max(np.abs(applied[:7] - state[:7])) / cap)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_cookie_single_piece")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene = scene_by_name(args.scene)
    env = scene.build_env(0.002, 0.05)
    # No renderer is created until a camera is asked for one, so turning the flag off keeps
    # this process off the GPU entirely rather than merely off the frame path.
    env.render_cameras = False
    policy = scene.build_policy(env)
    log = _PhaseLog(policy)
    policy.act = log.act  # type: ignore[method-assign]
    policy.observe_step = log.observe_step  # type: ignore[attr-defined]

    cap = float(env.config.max_joint_step_rad)
    runner = EpisodeRunner(
        env,
        policy,
        task=scene.task.prompt,
        recorder=None,
        reset_options={
            "randomize_scene": scene.randomize_placements,
            "randomize_cookies": scene.randomize_cookies,
        },
    )

    per_phase: dict[str, list[float]] = defaultdict(list)
    accepted = 0
    steps_total = 0
    try:
        for index in range(args.episodes):
            result = runner.run(seed=args.seed + index)
            accepted += int(bool(result.success))
            if not log.phase:
                continue
            steps_total += len(log.phase)
            for name, state, applied in zip(log.phase, log.state, log.applied, strict=True):
                per_phase[name].append(_displacement(state, applied, cap))
            log.phase.clear()
            log.state.clear()
            log.requested.clear()
            log.applied.clear()
    finally:
        runner.close()

    print(f"{args.scene}: {args.episodes} attempts, {accepted} accepted, "
          f"{steps_total} control steps, cap {cap:.3f} rad/step")
    print()

    every = np.concatenate([np.asarray(v) for v in per_phase.values()]) if per_phase else np.zeros(1)
    print("overall stand-still fraction (reproduces the dataset probe)")
    for threshold in DWELL_THRESHOLDS:
        print(f"  < {threshold*100:>3.0f}% of the cap: {np.mean(every < threshold) * 100:>5.1f}%")
    print()

    order = sorted(per_phase, key=lambda name: -len(per_phase[name]))
    print(f"{'phase':<18} {'steps':>7} {'share':>7} " +
          " ".join(f"{f'<{t*100:.0f}%':>7}" for t in DWELL_THRESHOLDS) + f" {'median':>7}")
    for name in order:
        values = np.asarray(per_phase[name])
        share = len(values) / max(steps_total, 1)
        cells = " ".join(f"{np.mean(values < t) * 100:>6.1f}%" for t in DWELL_THRESHOLDS)
        print(f"{name:<18} {len(values):>7} {share * 100:>6.1f}% {cells} "
              f"{np.median(values) * 100:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
