"""Where does the Cookie's spin come from?

The Panda walks the whole state machine and stops at the last check because the Cookie is turning
at 0.65 rad/s about the vertical, and the scene cannot damp that: its friction is
`[0.8, 0.003, 0.0002]`, so torsional friction is 2e-4 and nothing stops a spinning slab.  So the
spin has to not be there in the first place, which means finding the phase that puts it there.

This prints the Cookie's angular speed at every phase transition and the speed of the jaw that is
holding it, so the two can be compared: a spin that appears while the Cookie is in the jaws is the
transfer's, and one that appears at the moment they open is the release's.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import mujoco

from a3_dual_arm_sim.collection import libero_props_scene

BASE = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
    cartesian=True,
    panda=True,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear-of-wall", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--settle-every", type=int, default=500)
    parser.add_argument("--rolling-friction", type=float, default=None)
    parser.add_argument("--dump-contacts", action="store_true")
    parser.add_argument("--trace", type=int, default=0)
    parser.add_argument("--release-steps", type=int, default=0)
    args = parser.parse_args()

    layout = dict(BASE)
    if args.clear_of_wall:
        layout["a3_cookie_position_m"] = (0.1194, 0.0711)
    scene = libero_props_scene(task_name="a3_libero_panda_single_cookie", hold_steps=40, **layout)
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    policy.reset()
    expert = policy.expert
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()
    if args.release_steps:
        expert.release_steps = args.release_steps
        print(f"  release_steps = {args.release_steps}")

    print("  step  phase            cookie w   cookie v    in-jaw  tip sightline speed")
    previous = None
    for step in range(1, args.steps + 1):
        action = expert.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        phase = expert.phase.name
        index = expert._current_cookie()
        if index is None:
            index = 0
        if args.trace and args.trace <= step <= args.trace + 20:
            # The question is whether the jaws open as a ramp or as a step, and what the Cookie does
            # while they do.  A single-step jump in the command would be a bug; a ramp means the
            # impulse is physics, and the numbers below say which.
            applied = np.asarray(env.last_applied_action, dtype=np.float64).ravel()
            grip = applied[7] if applied.size >= 8 else float("nan")
            finger_qpos = [
                float(env.data.qpos[env.model.jnt_qposadr[joint]])
                for joint in range(env.model.njnt)
                if "finger" in (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint) or "")
            ]
            cookie_velocity = np.asarray(
                env.data.qvel[
                    int(env.model.jnt_dofadr[env._cookie_joints[index]]):
                    int(env.model.jnt_dofadr[env._cookie_joints[index]]) + 6
                ],
                dtype=np.float64,
            )
            print(
                f"    trace {step:4d}  {phase:14s} applied {np.round(applied, 3)}"
                f"\n          fingers {np.round(finger_qpos, 4)}"
                f"  jaws {env.privileged_left_finger_contacts(index)}"
                f"  cookie w {np.linalg.norm(cookie_velocity[3:]):8.4f}"
            )
        if phase != previous or phase in ("OPEN", "VERIFY_RELEASE") or step % 40 == 0:
            # The Cookie's own body velocity, not relative to the box: the question here is the
            # spin the object carries, wherever it came from.
            joint = env._cookie_joints[index]
            dof = int(env.model.jnt_dofadr[joint])
            velocity = np.asarray(env.data.qvel[dof : dof + 6], dtype=np.float64)
            contacts = env.privileged_left_finger_contacts(index)
            marking = "  <- entered" if phase != previous else ""
            print(
                f"  {step:4d}  {phase:16s} {np.linalg.norm(velocity[3:]):9.4f}"
                f"  {np.linalg.norm(velocity[:3]):9.4f}"
                f"  {('yes' if any(contacts) else 'no'):5s}"
                f"  {np.linalg.norm(np.asarray(env.data.cvel[env._target_bin_body][:3])):8.4f}"
                f"{marking}"
            )
            previous = phase
        if terminated or truncated:
            print(f"  ended at step {step} terminated={terminated} truncated={truncated}")
            break
    print(f"  final: phase={expert.phase.name} reason={expert.failure_reason}")

    # **Can the residual spin die of its own accord?**  That single question decides the fix: if it
    # decays, the expert only has to wait longer before judging `settled`; if it does not, waiting
    # is not a fix and the release must stop imparting it.  The expert is done, so the environment
    # is stepped with a still command and the Cookie watched.
    if args.settle_steps:
        if args.rolling_friction is not None:
            # A pure measurement: the Cookie's own geoms get a rolling friction a slab on a table
            # could plausibly have, and the same held-still run is repeated.  Nothing else about the
            # scene changes, and the value is not written anywhere -- if this is the fix, the fix
            # belongs in the prop config with a reason, not here.
            cookie_geoms = sorted(env._cookie_collision_geoms[0])
            was = float(env.model.geom_friction[cookie_geoms[0]][2])
            for geom in cookie_geoms:
                friction = np.asarray(env.model.geom_friction[geom], dtype=np.float64).copy()
                friction[2] = args.rolling_friction
                env.model.geom_friction[geom] = friction
            print(
                f"\n  rolling friction raised to {args.rolling_friction} (was {was}),"
                f" across {len(cookie_geoms)} Cookie geoms"
            )
        print(f"\n  holding still for {args.settle_steps} steps to see whether the spin decays:")
        print("     step   cookie w   cookie v   contacts")
        for extra in range(1, args.settle_steps + 1):
            observation, _, terminated, truncated, info = env.step(expert.act(observation))
            if extra % args.settle_every == 0 or extra == 1:
                joint = env._cookie_joints[0]
                dof = int(env.model.jnt_dofadr[joint])
                velocity = np.asarray(env.data.qvel[dof : dof + 6], dtype=np.float64)
                print(
                    f"  {extra:6d}  {np.linalg.norm(velocity[3:]):9.5f}"
                    f"  {np.linalg.norm(velocity[:3]):9.5f}"
                    f"  {env.privileged_left_finger_contacts(0)}"
                )
            if terminated or truncated:
                print(f"  the environment ended the episode at extra step {extra}")
                break
    if args.dump_contacts:
        # What is the Cookie actually resting on while it keeps turning?  The force each contact
        # carries is printed in full, including the torsional and rolling terms, because a flat slab
        # spinning about the vertical axis is opposed by the *torsional* friction component and not
        # by the rolling one -- and the Cookie's own entry for that is 0.003.
        joint = env._cookie_joints[0]
        address = int(env.model.jnt_qposadr[joint])
        position = np.asarray(env.data.qpos[address : address + 3], dtype=np.float64)
        quaternion = np.asarray(env.data.qpos[address + 3 : address + 7], dtype=np.float64)
        upright = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(upright, quaternion)
        tilt = np.degrees(
            np.arccos(np.clip(abs(upright.reshape(3, 3)[2, 2]), -1.0, 1.0))
        )
        angular = np.asarray(
            env.data.qvel[
                int(env.model.jnt_dofadr[joint]) + 3 : int(env.model.jnt_dofadr[joint]) + 6
            ],
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(angular))
        axis = np.round(angular / speed, 3) if speed > 1e-9 else np.zeros(3)
        print(f"\n  Cookie centre in world: {np.round(position * 1000, 2)} mm")
        print(
            f"  tilt from vertical {tilt:5.1f} deg"
            f"   angular speed {speed:6.4f} rad/s along world axis {axis}"
        )
        print("  (z is up: an axis of (0,0,1) is a slab spinning flat, a horizontal axis is one rolling on an edge)")
        print(f"  Cookie collision geoms: {sorted(env._cookie_collision_geoms[0])}")
        cookie_geoms = set(env._cookie_collision_geoms[0])
        print("  contacts involving the Cookie:")
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            if contact.geom1 not in cookie_geoms and contact.geom2 not in cookie_geoms:
                continue
            other = contact.geom2 if contact.geom1 in cookie_geoms else contact.geom1
            other_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or "?"
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(env.model, env.data, index, force)
            print(
                f"    vs {other_name:<32} dist {contact.dist * 1000:+7.3f} mm"
                f"  normal {force[0]:+8.4f}  tangential {force[1]:+8.4f} {force[2]:+8.4f}"
                f"  torsional {force[3]:+8.4f}  rolling {force[4]:+8.4f} {force[5]:+8.4f}"
            )
        friction = np.asarray(env.model.geom_friction[sorted(cookie_geoms)[0]], dtype=np.float64)
        print(f"  Cookie friction [sliding, torsional, rolling]: {friction}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
