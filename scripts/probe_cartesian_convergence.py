"""Does a cartesian command actually close on its goal?  Measure, do not infer.

`_motion_done` is what ends APPROACH -- and APPROACH has no timeout and no retry, so a goal the
command never reaches does not fail, it *hangs*: the rollout runs to its cap with no reason and no
retries, which is exactly what the first gate run reported for both Panda configurations.

This prints, every few steps, the phase, the distance to the current waypoint's pose, whether the
expert considers itself arrived, and the six numbers it is sending.  That separates the three
possibilities that look identical from outside:

  * the delta is wrong and the arm goes nowhere (error flat, delta saturated in one direction),
  * the delta is right and the arm converges but not far enough (error falls, plateaus above the
    tolerance),
  * the goal itself is out of reach (error falls to a floor and the delta stays saturated).
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

A3_LAYOUT = dict(matched_cameras=True, a3_single_cookie=True, a3_layout_offset_m=(0.0, 0.0))
COLLECT_LAYOUT = dict(
    matched_cameras=True, a3_single_cookie=True, a3_layout_offset_m=(0.020, -0.200)
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panda", action="store_true")
    parser.add_argument("--layout", choices=("a3", "collect"), default="a3")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--approach-height", type=float, default=None)
    args = parser.parse_args()

    layout = COLLECT_LAYOUT if args.layout == "collect" else A3_LAYOUT
    scene = libero_props_scene(
        task_name="a3_libero_panda_single_cookie",
        hold_steps=40,
        cartesian=True,
        panda=args.panda,
        **layout,
    )
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    expert = policy.expert
    if args.approach_height is not None:
        expert.approach_height_m = args.approach_height
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()
    print(
        f"panda={args.panda} layout={args.layout}  approach_height={expert.approach_height_m}  "
        f"cartesian_translation_scale_m={env.config.cartesian_translation_scale_m}  "
        f"cartesian_rotation_scale_rad={env.config.cartesian_rotation_scale_rad}"
    )
    print(
        f"tolerance: {expert.cartesian_arrival_tolerance_m * 1000:.1f} mm, "
        f"{expert.cartesian_arrival_tolerance_rad:.3f} rad"
    )
    previous = None
    for step in range(1, args.steps + 1):
        action = expert.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        phase = expert.phase.name
        # The waypoint this phase is aiming at, for the phases that have one.
        key = {
            "APPROACH": "approach",
            "DESCEND": "grasp",
            "CLOSE": "grasp",
            "VERIFY_GRASP": "grasp",
            "LIFT": "lift",
            "VERIFY_LIFT": "lift",
        }.get(phase)
        line = f"  {step:4d} {phase:16s}"
        if key and key in expert._waypoints:
            position_error, rotation_error = expert._goal_error(expert._waypoints[key])
            line += (
                f"  |dpos| {np.linalg.norm(position_error) * 1000:8.2f} mm"
                f"  |drot| {np.linalg.norm(rotation_error):7.4f} rad"
                f"  done={expert._motion_done!s:5s}"
                f"  cmd {np.round(action[:6], 3)}"
            )
        else:
            line += f"  (no tracked waypoint)  cmd {np.round(action[:6], 3)}"
        if previous != phase:
            print(line + "   <- phase entered")
            previous = phase
        elif step % args.every == 0:
            print(line)
        if terminated or truncated:
            print(f"  ended at step {step}: terminated={terminated} truncated={truncated}")
            break
    print(f"  final: phase={expert.phase.name} reason={expert.failure_reason}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
