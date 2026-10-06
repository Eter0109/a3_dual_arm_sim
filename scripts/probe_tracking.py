"""Does the arm actually go where the policy points it? Measured, one command at a time.

Why this exists
---------------
`configs/policies/pi05_libero_a3_cartesian.yaml` gives `lerobot/pi05_libero_finetuned` a
correct LIBERO interface on the A3 arm: end-effector deltas, the scene's own `output_max`,
a healthy rate-limit fraction.  On `a3_libero_props` -- where the bowl and plate are
LIBERO's *own* assets and the task string is LIBERO's own -- three seeds give the same
fingerprint:

    seed   closest fingertip-bowl gap   at step   bowl moved   gripper flips
    0              128.7 mm                18        0.63 mm        26
    1               95.2 mm                19        0.63 mm        21
    2              141.6 mm                19        0.63 mm        28

The bowl's rim is 39 mm out, so the pads stop 60-100 mm short, **at the same step on every
seed**, and then retreat monotonically to about a metre.  The bowl never moves at all (0.63
mm is settling, and it is identical on all three seeds).  Meanwhile the model's own commands
are not small: median 10 mm per step, up to 35 mm, against a 50 mm cap -- and only 11.5% of
them are rate-limited.

That is the shape of an arm that does not go where it is told.  A policy that pushes toward
an object, sees the object not get closer, and goes elsewhere produces exactly this: approach,
give up, retreat, repeat.  The alternative -- that the policy simply cannot see or cannot plan
-- is contradicted by the front camera showing the bowl at **269 pixels** and by the commands
being of sensible magnitude and direction.

What it measures
----------------
Two things, because there are two places a commanded motion can be lost, and neither needs
the policy, the GPU, or a fixed scene:

**1. The joint servo.**  Hold the arm still, then step the *commanded* pose by the rate limit
and watch where the arm actually is for the next several control steps.  This is the servo's
step response: how much of one step it delivers before the next one arrives.

**2. Cartesian tracking.**  Switch the environment to `cartesian_delta`, send the same
constant delta every step (10 mm, the median the policy asked for), and compare what the
end-effector actually achieves per step against what was asked.  The adapter solves IK for
`current + delta * cartesian_translation_scale_m`, so the *target* is right by construction;
this measures whether the arm gets there.

Both are run headless: no renderer is created unless a camera is asked for, so this needs no
GPU and can sit beside a training run.

Usage::

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/probe_tracking.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from a3_dual_arm_sim.collection import scene_by_name  # noqa: E402
from a3_dual_arm_sim.contracts import (  # noqa: E402
    CARTESIAN_ACTION_DIM,
    JOINT_ACTION_DIM,
    LEFT_JOINTS,
)

TRAJECTORY = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def _eef(env: object) -> np.ndarray:
    """World position of the left arm's end-effector site."""

    return env.data.site_xpos[env._eef_sites[0]].copy()  # noqa: SLF001


def _eef_for(env: object, applied: np.ndarray) -> np.ndarray:
    """Where the left eef *would* be if the arm were exactly at the commanded joints.

    This is the third of the three numbers a tracking measurement needs, and the one that
    separates the two ways a commanded motion goes missing.  Asking for 10 mm and achieving
    5.5 mm is consistent with either the adapter solving IK for less than was asked (or the
    rate limiter scaling the answer down), or the servo failing to reach what was asked.  The
    first is a calibration error and the second is a dynamics one, and they call for opposite
    fixes, so the command itself has to be measured rather than assumed.
    """

    import mujoco

    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    qids = [env._qpos_ids[name] for name in LEFT_JOINTS]  # noqa: SLF001
    for address, value in zip(qids, applied[:7], strict=True):
        scratch.qpos[address] = value
    mujoco.mj_forward(env.model, scratch)
    return scratch.site_xpos[env._eef_sites[0]].copy()  # noqa: SLF001


def _step_response(env: object, steps: int) -> dict:
    """Step the commanded pose by the rate limit and follow the arm's response.

    The command is a jump in *joint* space of exactly `max_joint_step_rad`, applied on every
    arm joint at once and then held.  Held, because a single control step is not enough to
    see where the arm settles: the question is how many control steps it takes to arrive, and
    a policy that re-issues a command every step only ever gets the first part of that.
    """

    env.reset(seed=0)
    limit = float(env.config.max_joint_step_rad)
    start_eef = _eef(env)
    # `_last_applied_action` is what the rate limiter measures against, so a command built
    # from it is guaranteed to sit exactly on the limit rather than being clipped below it.
    base = env._last_applied_action.copy()  # noqa: SLF001
    command = base.copy()
    command[:7] = base[:7] + limit

    trace = []
    for _ in range(steps):
        env.step(command)
        trace.append(float(np.linalg.norm(_eef(env) - start_eef)))
    return {"limit": limit, "trace": np.asarray(trace)}


def _ringing(trace: np.ndarray, tolerance: float = 0.02) -> dict:
    """How much the response overshoots the value it settles at, and how long it shakes.

    Tracking accuracy and smoothness are separate costs and the knob below trades one for
    the other, so measuring only the fraction delivered per step hides the half of it that a
    person watching the arm sees.  A servo that delivers 95% of a step while overshooting by
    12% is *shaking* its way to the target: the policy then sees a position that varies in a
    direction its own plan does not, and corrects for it, and the correction is the next
    command.

    ``overshoot`` is the largest excursion past where the arm ends up, as a fraction of the
    total travel; ``reversals`` counts the direction changes before it settles, which is what
    "ringing" means as a number rather than as an adjective.
    """

    values = np.asarray(trace, dtype=np.float64)
    total = float(values[-1])
    if total <= 0.0 or len(values) < 3:
        return {"overshoot": 0.0, "reversals": 0, "settled_at": len(values)}
    settled_at = len(values)
    for index in range(len(values) - 1, -1, -1):
        if abs(values[index] - total) > tolerance * total:
            settled_at = index + 1
            break
    # Only past the first step: the first step is the servo's rise, not a reversal.  And
    # only excursions that clear the tolerance: counting every wobble of the last digit would
    # report numerical noise as ringing, which is how this metric reported 14 reversals for a
    # response that is inside 2% by step 6.
    error = values - total
    significant = np.where(np.abs(error) > tolerance * total, np.sign(error), 0.0)
    significant = significant[significant != 0]
    reversals = int(np.count_nonzero(np.diff(significant) != 0))
    return {
        "overshoot": float(max(0.0, (values[1:].max() - total) / total)),
        "reversals": reversals,
        "settled_at": settled_at,
    }


def _cartesian_tracking(env: object, delta_mm: float, steps: int, axis: int,
                        hold: int = 1) -> dict:
    """Send one constant Cartesian delta every `hold` steps and measure what is achieved.

    The direction matters more than it looks.  A downward delta walks the arm into the table
    within about twenty steps, and from then on the achieved displacement is the table's
    answer rather than the servo's -- which reads as a steady-state tracking ratio of 21% and
    is a measurement of nothing.

    `hold` is the mechanism test.  If the shortfall is a *scale* error the ratio will not move
    when the same command is repeated for two or three control steps; if it is a *cadence*
    problem -- the servo cannot cover a fresh command every 50 ms -- then giving it 100 or
    150 ms per command should recover most of the motion, and it will say so directly.
    """

    env.action_mode = "cartesian_delta"
    env.reset(seed=0)
    requested = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
    scale = float(env.config.cartesian_translation_scale_m)
    requested[axis] = (delta_mm / 1000.0) / scale
    requested[6] = -1.0  # hold the gripper open
    requested[13] = 0.0

    achieved = []
    commanded = []
    for step in range(steps):
        before = _eef(env)
        # One command per `hold` steps; the intervening steps re-send the *last applied*
        # command, which is what a policy that has run out of chunk rows does, and what the
        # joint-space test commands implicitly.
        if step % hold == 0:
            payload = requested
        else:
            # The cartesian action is 14 wide -- two arms of position, rotation and gripper --
            # so the grippers are at 6 and 13, not 7 and 15 (which is the joint layout).
            payload = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
            payload[6] = requested[6]
            payload[13] = requested[13]
        _, _, _, _, info = env.step(payload)
        achieved.append(float(np.linalg.norm(_eef(env) - before)))
        commanded.append(float(np.linalg.norm(_eef_for(env, info["applied_action"]) - before)))
    return {
        "requested_m": delta_mm / 1000.0,
        "achieved": np.asarray(achieved),
        "commanded": np.asarray(commanded),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_libero_props",
                        help="the scene the props run used")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--delta-mm", type=float, default=10.0,
                        help="the per-step Cartesian delta to test (the policy's median)")
    parser.add_argument("--axis", type=int, default=2, choices=(0, 1, 2),
                        help="0/1/2 = x/y/z; z is up, which is free space")
    parser.add_argument("--hold", type=int, default=1,
                        help="control steps each command is given before the next one")
    args = parser.parse_args()

    scene = scene_by_name(args.scene)
    env = scene.build_env(0.002, 0.05)
    env.render_cameras = False
    try:
        print(f"scene {args.scene}   control {env.config.control_hz} Hz   "
              f"physics {env.config.physics_hz} Hz   "
              f"joint rate limit {env.config.max_joint_step_rad} rad/step")
        print()

        response = _step_response(env, args.steps)
        trace = response["trace"]
        print(f"1. joint servo step response -- commanded {response['limit']:.3f} rad on every "
              f"arm joint, then held")
        print(f"   eef distance travelled, mm, at control step:")
        cells = "  ".join(f"{int(s)}:{trace[int(s)-1]*1000:6.2f}" for s in TRAJECTORY
                          if int(s) <= len(trace))
        print(f"   {cells}")
        final = float(trace[-1]) * 1000
        first = float(trace[0]) * 1000
        print(f"   after 1 step {first:.2f} mm   after {len(trace)} steps {final:.2f} mm   "
              f"=> one step delivers {first / max(final, 1e-9) * 100:.1f}% of the total")
        shake = _ringing(trace)
        print(f"   overshoot past where it settles {shake['overshoot'] * 100:.1f}%, "
              f"{shake['reversals']} direction reversal(s), settled by step "
              f"{shake['settled_at']} ({shake['settled_at'] / env.config.control_hz * 1000:.0f} ms)")
        print()

        tracking = _cartesian_tracking(env, args.delta_mm, args.steps, axis=args.axis,
                                       hold=args.hold)
        achieved = tracking["achieved"] * 1000
        commanded = tracking["commanded"] * 1000
        asked = tracking["requested_m"] * 1000
        print(f"2. cartesian tracking -- {asked:.1f} mm requested every "
              f"{args.hold} control step(s), axis {'xyz'[args.axis]}")
        print("   per step, mm: requested / commanded / achieved")
        print("   " + "  ".join(
            f"{int(s)}:{asked:.1f}/{commanded[int(s)-1]:.2f}/{achieved[int(s)-1]:.2f}"
            for s in TRAJECTORY if int(s) <= len(achieved)))
        window = slice(-10 * args.hold, None)
        per_command = achieved[window].mean() * args.hold
        # The ratio that means something is against what was *asked for*, not against the
        # FK'd command: the command is measured from the arm's current position, and while the
        # arm is still trailing the previous command that base already sits ahead of it, so
        # `achieved / commanded` understates the tracking and moves the wrong way with `hold`.
        print(f"   per command (the {asked:.1f} mm asked for) achieved "
              f"{per_command:.2f} mm = {per_command / asked * 100:.0f}%, "
              f"over {args.hold} control step(s) = {args.hold / env.config.control_hz * 1000:.0f} ms")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
