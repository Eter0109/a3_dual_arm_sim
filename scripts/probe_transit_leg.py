"""What the transit leg does for the placement, by taking it away and watching.

`MOVE_TO_SLOT` moves in two stages: to `q_transit` first, and then to the place pose.
The obvious question is why it does not go straight to the place pose, since that is
where it is going anyway, and the first stage is what carried the Cookie away from
the box on the Panda scene (measured: a 250 mm excursion against a 150 mm trip, see
`scripts/probe_carry_path.py`).

The answer is not "nothing" -- removing the stage makes the three configurations
that can be placed comfortably *faster* (clear of wall 155 -> 119 steps, two
cookies 356 -> 273) and breaks `a3 layout`, which is the one configuration where the
Cookie stands at A3's own bin position, the position the usable-region measurement
puts outside the region the dataset may draw from.  "It breaks one config" is not an
answer, so this prints what actually differs: the tool's error against the place pose
through the descent, and the joint configuration the descent starts from.

Usage::

    python scripts/probe_transit_leg.py --config a3 --skip
    python scripts/probe_transit_leg.py --config a3
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

#: Kept in step with `scripts/run_expert_gate.py`.
LAYOUTS = {
    "a3": dict(matched_cameras=True, a3_single_cookie=True, a3_layout_offset_m=(0.0, 0.0)),
    "clear": dict(
        matched_cameras=True,
        a3_single_cookie=True,
        a3_layout_offset_m=(0.020, -0.200),
        a3_cookie_position_m=(0.1194, 0.0711),
    ),
}


def dump_mechanism(env, expert) -> None:
    """Everything that could be holding the Cookie over, at one instant.

    The placement is refused because the Cookie is not upright, and a Cookie leaning 19.5 degrees
    is not explained by a tool that is 1.3 degrees off: the amplification is the question.  So this
    prints the tool's rotation residual and its axis, the Cookie's tilt and *its* axis, the force on
    each of the Cookie's contacts, and how much room each joint has left -- the four things that
    could be doing it.
    """

    import mujoco

    index = expert._current_cookie()
    tool_matrix = np.asarray(env.data.site_xmat[env.ik.left_site]).reshape(3, 3)
    target_matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(target_matrix, expert._grasp_quat)
    box_matrix = env.data.xmat[env._target_bin_body].reshape(3, 3)
    cookie = env.data.xmat[env._cookie_bodies[index]].reshape(3, 3)
    relative = box_matrix.T @ cookie

    print("\n  --- mechanism snapshot ---")
    position_error, rotation_error = expert._goal_error(expert._waypoints["place"])
    axis = rotation_error / max(float(np.linalg.norm(rotation_error)), 1e-12)
    print(
        f"    tool: position off {np.linalg.norm(position_error) * 1000:.3f} mm,"
        f" orientation off {np.degrees(np.linalg.norm(rotation_error)):.3f} deg"
        f" about world {np.round(axis, 3)}"
    )
    lean = np.arccos(np.clip(abs(relative[2, 2]), -1.0, 1.0))
    # The Cookie's own local z, in the box frame: which way it is falling.
    print(
        f"    Cookie: leaning {np.degrees(lean):.2f} deg, its up axis points"
        f" {np.round(relative[:, 2], 3)} in the box frame"
    )
    print("    contacts on the Cookie:")
    geoms = set(env._cookie_collision_geoms[index])
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        if not {int(contact.geom1), int(contact.geom2)} & geoms:
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, contact_index, force)
        names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or "?"
            for geom in (contact.geom1, contact.geom2)
        ]
        print(
            f"      {names[0]:<30} <-> {names[1]:<30}"
            f" dist {contact.dist * 1000:+7.3f} mm  normal {force[0]:+8.4f} N"
        )
    print("    joints, and how much travel is left before the limit:")
    for joint in expert._l_qpos:
        joint_id = int(joint)
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "?"
        low, high = env.model.jnt_range[joint_id]
        value = float(env.data.qpos[joint_id])
        print(
            f"      {name:<28} {value:+8.4f}   range [{low:+.3f}, {high:+.3f}]"
            f"   room {min(value - low, high - value):+.4f} rad"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(LAYOUTS), default="a3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--skip", action="store_true", help="Never take the transit leg")
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument(
        "--dump-at",
        type=int,
        default=0,
        help="Print one full mechanism snapshot at this step",
    )
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name="a3_libero_panda_single_cookie",
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
        # Reproduce the variant that was measured and rejected: the leg is never taken, so the arm
        # drives from the lift pose straight to the place pose.
        original = expert._advance

        def advance(phase, _original=original, _expert=expert):
            _original(phase)
            if phase.name == "MOVE_TO_SLOT":
                _expert._move_stage = "preplace"

        expert._advance = advance

    print(f"config {args.config}  skip_transit={args.skip}  seed {args.seed}")
    print("  step  phase            stage     pos err (mm)  rot err (deg)   joints 1-3 (rad)")

    warned = False
    previous_up = None
    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

        phase = expert.phase.name
        if phase in ("MOVE_TO_SLOT", "DESCEND_TO_PLACE") and step % args.every == 0:
            # The Cookie's pose **in the tool's frame**.  While the jaws hold it this is constant;
            # any change is the Cookie turning inside a closed grip, which is what a lean that the
            # tool is not making has to be.
            index = expert._current_cookie()
            tool_matrix = np.asarray(env.data.site_xmat[env.ik.left_site]).reshape(3, 3)
            cookie_matrix = env.data.xmat[env._cookie_bodies[index]].reshape(3, 3)
            in_tool = tool_matrix.T @ cookie_matrix
            offset = tool_matrix.T @ (
                np.asarray(env.data.xpos[env._cookie_bodies[index]])
                - np.asarray(env.data.site_xpos[env.ik.left_site])
            )
            print(
                f"        grip: Cookie in tool frame, up axis {np.round(in_tool[:, 2], 3)}"
                f"   offset {np.round(offset * 1000, 2)} mm"
            )
        if phase == "DESCEND_TO_PLACE" and step % args.every == 0:
            # The three clauses of `aligned`, which is what actually ends this phase.  They are all
            # about the *Cookie*, not the tool, so a descent can be perfect in tool terms and still
            # never satisfy them.
            index = expert._current_cookie()
            live = expert.env.privileged_cookie_target_position(index)
            slot = np.asarray(expert.env.TARGET_SLOTS_LOCAL[expert._current_slot()])
            box_rotation = expert.data.xmat[expert._tb_id].reshape(3, 3)
            relative = box_rotation.T @ expert.data.xmat[expert.env._cookie_bodies[index]].reshape(3, 3)
            print(
                f"        cookie xy off slot {np.linalg.norm(live[:2] - slot) * 1000:7.3f} mm"
                f"   height off plan {abs(live[2] - expert._placement_height) * 1000:7.3f} mm"
                f"   tilt {np.degrees(np.arccos(np.clip(abs(relative[2, 2]), -1.0, 1.0))):6.2f} deg"
                f"   settle {expert._settle_count}"
            )
        if phase in ("MOVE_TO_SLOT", "DESCEND_TO_PLACE"):
            place = expert._waypoints.get("place")
            if place is not None:
                position_error, rotation_error = expert._goal_error(place)
                if step % args.every == 0 or not warned:
                    stage = expert._move_stage if phase == "MOVE_TO_SLOT" else ""
                    print(
                        f"  {step:4d}  {phase:16s} {stage:9s}"
                        f"  {np.linalg.norm(position_error) * 1000:11.3f}"
                        f"  {np.degrees(np.linalg.norm(rotation_error)):13.3f}"
                        f"   {np.round(expert.data.qpos[expert._l_qpos][:3], 3)}"
                    )
                    warned = True
        # The moment the Cookie changes how it sits in the jaws is the moment to look at what it is
        # touching, because a jaw cannot lean it without something else pushing back.
        if phase in ("MOVE_TO_SLOT", "DESCEND_TO_PLACE"):
            index = expert._current_cookie()
            tool_matrix = np.asarray(env.data.site_xmat[env.ik.left_site]).reshape(3, 3)
            in_tool = tool_matrix.T @ env.data.xmat[env._cookie_bodies[index]].reshape(3, 3)
            if previous_up is not None and float(in_tool[:, 2] @ previous_up) < np.cos(np.radians(1.0)):
                print(f"\n    *** step {step} ({phase}): the Cookie turned inside the grip")
                import mujoco as _mj

                for contact_index in range(env.data.ncon):
                    contact = env.data.contact[contact_index]
                    names = [
                        _mj.mj_id2name(env.model, _mj.mjtObj.mjOBJ_GEOM, int(geom)) or "?"
                        for geom in (contact.geom1, contact.geom2)
                    ]
                    if "cookie_" not in names[0] and "cookie_" not in names[1]:
                        continue
                    force = np.zeros(6, dtype=np.float64)
                    _mj.mj_contactForce(env.model, env.data, contact_index, force)
                    print(
                        f"      {names[0]:<30} <-> {names[1]:<30}"
                        f" dist {contact.dist * 1000:+7.3f} mm  normal {force[0]:+8.4f} N"
                    )
            previous_up = in_tool[:, 2].copy()

        if args.dump_at and step == args.dump_at:
            dump_mechanism(env, expert)

        if terminated or truncated:
            break

    print(f"\n  final: phase={expert.phase.name} reason={expert.failure_reason}")
    print(f"  success={info.get('success')} steps={step}")
    for index in range(len(env._cookie_bodies)):
        body = env._cookie_bodies[index]
        print(
            f"  Cookie {index} at {np.round(env.data.xpos[body] * 1000, 2)} mm,"
            f" in target {env.privileged_cookie_in_target(index)}"
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
