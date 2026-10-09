"""Why the descent onto the second Cookie pushes it over instead of straddling it.

The second transfer's DESCEND is where Stage 2 dies: one jaw touches the Cookie
and the other does not, and the Cookie is tipped 12.7 degrees in a single step,
47 two steps later and flat on the floor by the fourth.  A centred descent either
touches both pads or neither, so the question is how far off centre it is and
whether the clearance is enough to absorb that.

This prints, for every step of the descent onto the watched Cookie: where the
Cookie is, where the tool is, the separation of the two pads, and the Cookie's
offset from the middle of that separation along the closing axis.  The offset is
what has to stay inside the clearance, and the clearance is
``(pad separation - Cookie thickness) / 2``.

Usage:

    python scripts/probe_two_cookie_grasp.py --seed 0
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

LAYOUT = {
    "matched_cameras": True,
    "a3_single_cookie": True,
    "a3_layout_offset_m": (0.020, -0.200),
    "a3_cookie_positions_m": ((0.1194, 0.04875), (0.1194, 0.11941667)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--watch", type=int, default=1)
    parser.add_argument("--rows", type=int, default=60)
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

    inner = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "L_finger_inner_pad")
    outer = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "L_finger_outer_pad")
    cookie = args.watch

    print(
        "  step  phase           tool xyz (mm)          cookie xyz (mm)          tilt"
        "  pads (mm)  cookie offset (mm)  clearance (mm)  tooltilt  jaws"
    )

    shown = 0
    dumped = False
    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

        phase = expert.phase.name
        index = expert._current_cookie()
        if index != cookie or phase not in ("APPROACH", "ALIGN", "DESCEND", "CLOSE"):
            continue

        tool = np.asarray(env.data.site_xpos[env.ik.left_site], dtype=np.float64) * 1000.0
        centre = env.data.xpos[env._cookie_bodies[cookie]].copy() * 1000.0
        matrix = env._cookie_rotation(cookie)
        tilt = np.degrees(np.arccos(np.clip(abs(matrix[2, 2]), -1.0, 1.0)))
        # Yaw as well as tilt.  A Cookie turned about the *vertical* axis keeps its thin axis
        # vertical, so its tilt stays 0.1 degrees and a tilt-only check calls it untouched -- while
        # its silhouette along the jaws' closing axis goes from 6.4 mm to 50 mm, which is what puts
        # the pads inside it.
        yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))

        inner_point = np.asarray(env.data.site_xpos[inner], dtype=np.float64) * 1000.0
        outer_point = np.asarray(env.data.site_xpos[outer], dtype=np.float64) * 1000.0
        separation = float(np.linalg.norm(inner_point - outer_point))
        along = (outer_point - inner_point) / max(separation, 1e-9)
        # The Cookie's offset from the middle of the two pads, along the closing axis.
        offset = float(np.dot(centre - 0.5 * (inner_point + outer_point), along))
        thickness = 1000.0 * 2.0 * float(env.COOKIE_HALF_SIZE[1])
        clearance = 0.5 * (separation - thickness)
        jaws = env.privileged_left_finger_contacts(cookie)
        # A vertical pad cannot touch a 6.33 mm Cookie from 16 mm away.  One that is *tilted* can:
        # its inner-lower corner sweeps in by roughly (half the pad) times the sine of the tilt, so
        # the tilt of the tool's own z axis is what decides whether the first contact is a face or a
        # corner.  `_solve_l` only checks the position residual, so an orientation the arm cannot
        # reach still passes planning -- this is where that would show.
        tool_z = np.asarray(env.data.site_xmat[env.ik.left_site]).reshape(3, 3)[:, 2]
        tool_tilt = float(np.degrees(np.arccos(np.clip(abs(tool_z[2]), -1.0, 1.0))))

        print(
            f"  {step:4d}  {phase:13s} [{tool[0]:6.1f} {tool[1]:6.1f} {tool[2]:6.1f}]"
            f"  [{centre[0]:6.1f} {centre[1]:6.1f} {centre[2]:6.1f}]"
            f"  {tilt:5.1f}  {separation:9.2f}  {offset:+15.2f}  {clearance:+13.2f}"
            f"  {tool_tilt:6.1f}  yaw {yaw:8.2f}  {jaws}"
        )
        if any(jaws) and not dumped:
            dumped = True
            # Name the pair: `_jaw_geoms` walks the finger chain, so the first contact can be the
            # pad, the finger's own mesh, or something further down, and the distance says whether
            # it is a touch or a real push.
            print("  contacts involving the watched Cookie at the first touch:")
            cookie_geoms = set(env._cookie_collision_geoms[cookie])
            for contact_index in range(env.data.ncon):
                contact = env.data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if not pair & cookie_geoms:
                    continue
                names = [
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or "?"
                    for geom in (contact.geom1, contact.geom2)
                ]
                print(
                    f"    {names[0]} <-> {names[1]}  dist {contact.dist * 1000:+.4f} mm"
                    f"  pos {np.round(contact.pos * 1000, 2)}"
                )
            for jaw_name, geoms in (
                ("inner", env._left_jaw_geoms[0]),
                ("outer", env._left_jaw_geoms[1]),
            ):
                lowest = min(
                    float(env.data.geom_xpos[geom][2]) for geom in geoms
                ) * 1000.0
                print(
                    f"    {jaw_name} jaw chain {sorted(geoms)} lowest geom centre z {lowest:.2f} mm"
                )
            print(f"    Cookie top z {1000.0 * (env.data.xpos[env._cookie_bodies[cookie]][2] + 0.0125):.2f} mm")
        shown += 1
        if shown > args.rows:
            break
        if terminated or truncated:
            break

    print(f"\nfinal: phase={expert.phase.name} reason={expert.failure_reason}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
