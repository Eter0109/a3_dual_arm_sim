"""Why does the Panda never verify a grasp?  Measure the pads and the signals, not the code.

`_has_verified_grasp` needs two things and the run reports neither: a MuJoCo contact between each
left finger geom and the Cookie, and a touch sensor reading of at least 0.02 on both fingertips.
Which of the two is missing decides where to look next, and the geometry decides whether it is a
placement problem or a grip problem -- so this prints all three together, per step, from the moment
the jaws are commanded shut.

Static first, because the Panda scene renames robosuite's gripper onto A3's names and the whole
measurement is worthless if `L_finger_inner_geom` is not the hand that is actually moving.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import imageio.v2 as imageio
import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

FINGER_GEOMS = ("L_finger_inner_geom", "L_finger_outer_geom")
FINGERTIP_SITES = ("L_finger_inner_touch", "L_finger_outer_touch")
TOUCH_SENSORS = ("L_finger_inner_touch_sensor", "L_finger_outer_touch_sensor")


def name_of(model, kind, index):
    return mujoco.mj_id2name(model, kind, index) or f"<unnamed {index}>"


def static_report(model, label):
    print(f"\n  {label}")
    for geom_name in FINGER_GEOMS:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom < 0:
            print(f"    {geom_name:22s} MISSING")
            continue
        body = int(model.geom_bodyid[geom])
        print(
            f"    {geom_name:22s} body {name_of(model, mujoco.mjtObj.mjOBJ_BODY, body):24s}"
            f" size {np.round(model.geom_size[geom], 5)}"
        )
    for site_name in FINGERTIP_SITES:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site < 0:
            print(f"    {site_name:22s} MISSING")
            continue
        body = int(model.site_bodyid[site])
        print(
            f"    {site_name:22s} body {name_of(model, mujoco.mjtObj.mjOBJ_BODY, body):24s}"
            f" local {np.round(model.site_pos[site], 5)}"
        )
    for sensor_name in TOUCH_SENSORS:
        sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor < 0:
            print(f"    {sensor_name:28s} MISSING -- the grasp can never verify")
            continue
        kind = mujoco.mjtObj(int(model.sensor_objtype[sensor]))
        target = int(model.sensor_objid[sensor])
        print(
            f"    {sensor_name:28s} reads {kind.name} "
            f"{name_of(model, kind, target)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name="a3_libero_panda_single_cookie",
        hold_steps=40,
        cartesian=True,
        panda=True,
        matched_cameras=True,
        a3_single_cookie=True,
        a3_layout_offset_m=(0.020, -0.200),
    )
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    policy.reset()
    expert = policy.expert
    model, data = env.model, env.data
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()

    print("static: which hand do the left fingers belong to?")
    static_report(model, "panda scene")

    cookie = 0
    cookie_body = env._cookie_bodies[cookie]
    reported = False
    for step in range(1, args.steps + 1):
        action = expert.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        phase = expert.phase.name
        if phase != "VERIFY_GRASP" or reported:
            if terminated or truncated:
                print(f"  ended at step {step} terminated={terminated} truncated={truncated}")
                break
            continue
        reported = True
        print("\ndynamic: the jaws are commanded shut, phase VERIFY_GRASP")
        index = expert._current_cookie()
        contacts = env.privileged_left_finger_contacts(index)
        touches = env.privileged_left_touch_values()
        print(
            f"    cookie index {index}  contact(inner, outer) {contacts}  "
            f"touch {np.round(touches, 4)}  verified {expert._has_verified_grasp()}"
        )
        cookie_position = np.asarray(data.xpos[cookie_body], dtype=np.float64).copy()
        print(f"    Cookie centre        {np.round(cookie_position, 4)}")
        print(
            f"    Cookie half-extent   {np.round(env.COOKIE_HALF_SIZE, 4)}   "
            f"grasp offset above centre {expert._grasp_height_offset() * 1000:+.2f} mm"
        )
        pads = []
        for site_name in FINGERTIP_SITES:
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            pads.append(np.asarray(data.site_xpos[site], dtype=np.float64).copy())
        midpoint = 0.5 * (pads[0] + pads[1])
        separation = float(np.linalg.norm(pads[1] - pads[0]))
        axis = (pads[1] - pads[0]) / max(separation, 1e-12)
        rotation = np.asarray(data.xmat[cookie_body], dtype=np.float64).reshape(3, 3)
        extent = float(np.sum(np.abs(rotation.T @ axis) * env.COOKIE_HALF_SIZE))
        offset = float((cookie_position - midpoint) @ axis)
        print(
            f"    fingertip sites      {np.round(pads[0], 4)} and {np.round(pads[1], 4)}"
        )
        print(
            f"    pad separation       {separation * 1000:8.2f} mm"
            f"   Cookie half-width along it {extent * 1000:6.2f} mm"
            f"   centre offset {offset * 1000:+7.2f} mm"
        )
        print(
            f"    clearance per side   "
            f"{(separation / 2 - abs(offset) - extent) * 1000:+8.2f} mm"
            f"   (negative means the pads overlap the Cookie)"
        )
        print(
            f"    heights              pads z {midpoint[2] * 1000:.1f} mm"
            f"   Cookie centre z {cookie_position[2] * 1000:.1f} mm"
            f"   difference {(midpoint[2] - cookie_position[2]) * 1000:+.2f} mm"
        )
        print(
            f"    grasp target world   "
            f"{np.round(expert._grasp_target_world, 4) if expert._grasp_target_world is not None else None}"
        )
        print(
            f"    opening: measured {float(env.current_joint_action[7]):.4f}"
            f"   commanded {expert.closed_gripper:.4f}"
        )
        # The raw finger joint angles, because `opening` is a derived, polarity-flipped quantity
        # and the two numbers disagree by a factor of eight: commanded shut, measured 0.08.
        finger_joints = [
            float(data.qpos[int(model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ])])
            for name in ("L_finger_inner_joint", "L_finger_outer_joint")
        ]
        print(f"    finger qpos          {np.round(finger_joints, 5)} rad")
        # The raw contact list, because "no contact with the Cookie" and "the pads are not even
        # touching anything" are different findings and the privileged query conflates them.
        print("    colliding geometry at this instant:")
        for geom_name in FINGER_GEOMS:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            print(
                f"      {geom_name:22s} contype {int(model.geom_contype[geom])}"
                f" conaffinity {int(model.geom_conaffinity[geom])}"
                f" margin {model.geom_margin[geom]:.5f}"
            )
        print(f"      Cookie collision geoms {list(env._cookie_collision_geoms[0])}")
        for collision_geom in env._cookie_collision_geoms[0]:
            print(
                f"        geom {collision_geom} contype {int(model.geom_contype[collision_geom])}"
                f" conaffinity {int(model.geom_conaffinity[collision_geom])}"
            )
        interesting = set(env._cookie_collision_geoms[0]) | {
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)) for n in FINGER_GEOMS
        }
        found = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = (int(contact.geom1), int(contact.geom2))
            if interesting & set(pair):
                found += 1
                names = tuple(name_of(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in pair)
                print(f"      contact {names} dist {contact.dist:+.5f}")
        print(f"      {found} contact(s) involving a Cookie geom or a finger geom; "
              f"{data.ncon} contact(s) in total")
        for index_finger, name in enumerate(("L_finger_inner_joint", "L_finger_outer_joint")):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_position")
            print(
                f"    {name:22s} range {np.round(model.jnt_range[joint], 4)}"
                f"   ctrl {data.ctrl[actuator]:+.5f}"
                f"   ctrlrange {np.round(model.actuator_ctrlrange[actuator], 4)}"
            )
        for tag, height, width in (("front", 480, 480), ("left_wrist", 256, 256)):
            image = np.asarray(env.render_view(tag, height=height, width=width))
            imageio.imwrite(f"/tmp/panda_grasp_{tag}.png", image)
        print("    wrote /tmp/panda_grasp_front.png and /tmp/panda_grasp_left_wrist.png")
        env.close()
        return 0
    print("  the phase VERIFY_GRASP was never reached")
    env.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
