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
from typing import Any
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

#: Kept in step with `run_expert_gate.py`, so this measures the configuration that ships.
LAYOUTS = {
    "two cookies": {
        "matched_cameras": True,
        "a3_single_cookie": True,
        "a3_layout_offset_m": (0.020, -0.200),
        "a3_cookie_positions_m": ((0.1194, 0.04875), (0.1194, 0.11941667)),
    },
    "a3": {
        "matched_cameras": True,
        "a3_single_cookie": True,
        "a3_layout_offset_m": (0.0, 0.0),
    },
}


def tool_of(env, joints: np.ndarray, scratch: mujoco.MjData, qpos_index: np.ndarray, site: int) -> np.ndarray:
    """Forward kinematics: where the tool would be if the arm held ``joints``."""

    scratch.qpos[:] = env.data.qpos
    scratch.qpos[qpos_index] = joints
    mujoco.mj_kinematics(env.model, scratch)
    return np.asarray(scratch.site_xpos[site], dtype=np.float64).copy()


def target_wall_top(env) -> float:
    """The highest point of the target box's walls, in world z.

    Taken from the geoms themselves rather than from the config's wall height, because the config
    gives a height above a floor whose datum is the box body's, and the question here is where the
    wall is in the world -- the Cookie hangs below the tool and meets whatever is highest.
    """

    highest = -np.inf
    for geom in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if "wall" not in name or "source" in name:
            continue
        half = np.asarray(env.model.geom_size[geom], dtype=np.float64)
        centre = np.asarray(env.data.geom_xpos[geom], dtype=np.float64)
        rotation = np.asarray(env.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        corners = np.array(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world = corners @ rotation.T + centre
        highest = max(highest, float(world[:, 2].max()))
    return highest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--skip", action="store_true", help="Never take the transit leg")
    parser.add_argument("--config", choices=sorted(LAYOUTS), default="two cookies")
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name=(
            "a3_libero_panda_single_cookie"
            if args.config == "a3"
            else "a3_libero_panda_two_cookies"
        ),
        hold_steps=40,
        panda=True,
        cartesian=True,
        **LAYOUTS[args.config],
    )
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    expert = policy.expert
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()

    if args.skip:
        original = expert._advance

        def advance(phase, _original=original, _expert=expert):
            _original(phase)
            if phase.name == "MOVE_TO_SLOT":
                _expert._move_stage = "preplace"

        expert._advance = advance

    wall_top = target_wall_top(env)
    print(f"  the target box's wall top is at world z = {wall_top * 1000:.1f} mm")

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
    # Every step's tool displacement, split into the part that is lateral (in the table plane) and
    # the part that is vertical, accumulated per leg.  An L-shaped pick and place is a vertical
    # leg, then a lateral one, then a vertical one -- so a leg that has both components is a
    # diagonal, and a diagonal is what brings the hanging Cookie into the box's wall.
    legs: dict[str, list[float]] = {}
    last_point: np.ndarray | None = None
    #: worst (Cookie bottom - wall top) while the Cookie is over the box, the step, and the phase.
    clearance_report: list[Any] = [np.inf, 0, ""]
    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, _ = env.step(action)

        tool = np.asarray(env.data.site_xpos[env.ik.left_site], dtype=np.float64)
        along = float((tool - start) @ axis) * 1000.0
        across = float(np.linalg.norm((tool - start) - (tool - start) @ axis * axis)) * 1000.0
        stage = expert._move_stage if expert.phase.name == "MOVE_TO_SLOT" else ""
        leg = f"{expert.phase.name}{'/' + stage if stage else ''}"
        if last_point is not None:
            step_vector = (tool - last_point) * 1000.0
            entry = legs.setdefault(leg, [0.0, 0.0])
            entry[0] += float(np.linalg.norm(step_vector[:2]))
            entry[1] += abs(float(step_vector[2]))
        last_point = tool.copy()

        # The clearance that matters for the L-shape.  A pure lift, translate, drop never takes any
        # part of the Cookie over the box's **wall band** below the wall top.  Checking the Cookie's
        # centre against the box's outer footprint misses the case that actually happened -- the
        # centre is still outside the box while the lower edge pokes over the wall -- so all eight
        # corners are tested, and the corner has to be over the wall band rather than over the
        # opening, which is where the Cookie legitimately goes.
        carried = expert._current_cookie()
        if carried is not None:
            body = env._cookie_bodies[carried]
            box_position = np.asarray(env.data.xpos[env._target_bin_body], dtype=np.float64)
            box_rotation = np.asarray(
                env.data.xmat[env._target_bin_body], dtype=np.float64
            ).reshape(3, 3)
            centre = np.asarray(env.data.xpos[body], dtype=np.float64)
            rotation = env._cookie_rotation(carried)
            outer = np.asarray(env.config.cookie_transfer.target_bin_half_size_m, dtype=np.float64)
            inner = np.asarray(env.TARGET_INNER_HALF_SIZE, dtype=np.float64)
            wall_top_local = wall_top - box_position[2]
            half = np.asarray(env.COOKIE_HALF_SIZE, dtype=np.float64)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = centre + rotation @ np.array(
                            [sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64
                        )
                        local = box_rotation.T @ (corner - box_position)
                        inside_outer = bool(np.all(np.abs(local[:2]) <= outer + 0.002))
                        over_opening = bool(np.all(np.abs(local[:2]) <= inner))
                        if inside_outer and not over_opening:
                            clearance = (float(local[2]) - wall_top_local) * 1000.0
                            if clearance < clearance_report[0]:
                                clearance_report[0] = clearance
                                clearance_report[1] = step
                                clearance_report[2] = expert.phase.name
        worst = min(worst, along)
        if previous != expert.phase.name or step % args.every == 0:
            print(
                f"  {step:4d}  {expert.phase.name:16s} {stage:9s}"
                f" [{tool[0] * 1000:7.1f} {tool[1] * 1000:7.1f} {tool[2] * 1000:7.1f}]"
                f"  {along:7.1f} {across:7.1f}"
            )
            previous = expert.phase.name

        # The descent is where the wall is met, so the trace has to run through it rather than
        # stopping at its first step -- stopping there measured the clearance before the collision
        # and reported it as comfortable.
        if terminated or truncated:
            print(f"  episode ended at step {step}")
            break

    print(f"\n  furthest the tool ever went *away* from the box: {worst:.1f} mm")
    print("  (0 means it never went backwards; negative numbers are the excursion)")
    print(
        f"\n  worst clearance of the carried Cookie's lowest point above the box wall top,"
        f" while it was over the box: {clearance_report[0]:+.1f} mm"
        f"  (step {clearance_report[1]}, {clearance_report[2]})"
    )
    print("  (negative means the Cookie's bottom went below the wall top while it was over the box)")
    print("\n  each leg, split into lateral and vertical travel (mm):")
    print(f"    {'leg':<26} {'lateral':>9} {'vertical':>10}   shape")
    for name, (lateral, vertical) in legs.items():
        if lateral < 1.0 and vertical < 1.0:
            continue
        if lateral < 0.15 * max(vertical, 1e-9):
            shape = "vertical"
        elif vertical < 0.15 * max(lateral, 1e-9):
            shape = "lateral"
        else:
            shape = f"DIAGONAL ({lateral / max(vertical, 1e-9):.1f}x lateral)"
        print(f"    {name:<26} {lateral:9.1f} {vertical:10.1f}   {shape}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
