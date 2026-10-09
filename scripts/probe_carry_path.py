"""Where the arm actually goes between the grasp and the box, step by step.

Watching the two-Cookie recording, the arm carries a Cookie out of the source bin,
moves a little way **away** from the target box, and then comes back and puts the
Cookie down.  A small motion in the wrong direction is easy to read as servo
settling, and it is not: `MOVE_TO_SLOT` is a two-stage move, and its first stage
drives to `q_transit`, a joint vector solved at reset for a **hard-coded** world
pose (0.10, 0.30, 0.89).  Whether that point lies on the way to the box is not
something the code asks.

So this prints the tool's position projected onto the line from the Cookie's start
to the target box: positive is towards the box, and any negative excursion is the
arm going the wrong way.  It also prints where the waypoints themselves are, so
the excursion can be attributed to a named waypoint rather than to the controller.

Usage::

    python scripts/probe_carry_path.py --seed 0
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

#: Kept in step with `run_expert_gate.py`, so this measures the configuration that ships.
LAYOUT = {
    "matched_cameras": True,
    "a3_single_cookie": True,
    "a3_layout_offset_m": (0.020, -0.200),
    "a3_cookie_positions_m": ((0.1194, 0.04875), (0.1194, 0.11941667)),
}


def tool_of(env, joints: np.ndarray, scratch: mujoco.MjData, qpos_index: np.ndarray, site: int) -> np.ndarray:
    """Forward kinematics: where the tool would be if the arm held ``joints``."""

    scratch.qpos[:] = env.data.qpos
    scratch.qpos[qpos_index] = joints
    mujoco.mj_kinematics(env.model, scratch)
    return np.asarray(scratch.site_xpos[site], dtype=np.float64).copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=4)
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name="a3_libero_panda_two_cookies",
        hold_steps=40,
        panda=True,
        cartesian=True,
        **LAYOUT,
    )
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    expert = policy.expert
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()

    start = env.privileged_cookie_position(0).copy()
    box = env.data.xpos[env._target_bin_body].copy()
    axis = box - start
    axis[2] = 0.0
    axis = axis / np.linalg.norm(axis)

    print(f"  Cookie 0 starts at {np.round(start * 1000, 1)} mm")
    print(f"  the target box is at {np.round(box * 1000, 1)} mm")
    print(f"  travel axis (horizontal) {np.round(axis, 3)}, {np.linalg.norm(box - start) * 1000:.1f} mm long")

    scratch = mujoco.MjData(env.model)
    qpos_index = expert._l_qpos
    site = expert._l_site
    print("\n  the waypoints the expert will drive to, from forward kinematics:")
    for name in ("q_transit",):
        joints = getattr(expert, name)
        print(
            f"    {name:10s} tool at"
            f" {np.round(tool_of(env, joints, scratch, qpos_index, site) * 1000, 1)} mm"
        )

    print("\n  step  phase            stage     tool xyz (mm)                   along  across")
    previous = None
    worst = 0.0
    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

        tool = np.asarray(env.data.site_xpos[env.ik.left_site], dtype=np.float64)
        along = float((tool - start) @ axis) * 1000.0
        across = float(np.linalg.norm((tool - start) - (tool - start) @ axis * axis)) * 1000.0
        stage = expert._move_stage if expert.phase.name == "MOVE_TO_SLOT" else ""
        if along < worst:
            worst = along
        if previous != expert.phase.name or step % args.every == 0:
            print(
                f"  {step:4d}  {expert.phase.name:16s} {stage:9s}"
                f" [{tool[0] * 1000:7.1f} {tool[1] * 1000:7.1f} {tool[2] * 1000:7.1f}]"
                f"  {along:7.1f} {across:7.1f}"
            )
            previous = expert.phase.name

        if expert.phase.name == "DESCEND_TO_PLACE":
            break
        if terminated or truncated:
            print(f"  episode ended at step {step}")
            break

    print(f"\n  furthest the tool ever went *away* from the box: {worst:.1f} mm")
    print("  (0 means it never went backwards; negative numbers are the excursion)")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
