"""Which Cookie is moved, by what, and when the second one is set down.

Stage 2 fails with "packed Cookie 0 disturbed": the first Cookie is grasped,
lifted and placed, and setting the second one down moves the first one 8.65 mm
and tips it 45 degrees.  The two slots are 56 mm apart -- 8 and 9 are the same
row of the target box's 2x5 grid -- so they should not touch, which means the
8.65 mm is coming from somewhere else and needs to be watched rather than
reasoned about.

This prints, every step: which Cookie the expert is transferring, what phase it
is in, where both Cookies are **in the target box's own frame**, how far each has
tipped, and which jaw chain is touching which Cookie.  The decision that fails is
`_packed_positions` with a 6 mm tolerance, so the moment the number crosses 6 mm
is the moment to explain.

Usage:

    python scripts/probe_two_cookie_disturbance.py --seed 0
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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

    print("counting indices: which Cookie the expert is on, and where both Cookies are")
    print("  step  phase            cookie  slot   cookie0 (box xyz mm)  tilt  cookie1 (box xyz mm)  tilt  jaws0  jaws1")

    # Watch the Cookie that is *not* being carried: the one still standing in the source bin is
    # the one the next transfer needs, and it is the one that falls over.
    target = args.watch
    history = []
    start_yaw = yaw_of(env, target)
    # Only the *increase* is reported, so the trace reads as the sequence of new records rather
    # than one line per step.
    baseline = [0.0]
    print(f"watching Cookie {target} (yaw {start_yaw:.2f} deg), reporting turns as they grow")
    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

        index = expert._current_cookie()
        phase = expert.phase.name
        rows = [box_frame(env, cookie) for cookie in range(2)]
        tilt = rows[target][1]
        yaw = yaw_of(env, target)
        if abs(yaw - start_yaw) > baseline[0]:
            baseline[0] = abs(yaw - start_yaw)
            print(
                f"  {step:4d}  {phase:16s} cookie {index}  the waiting Cookie {target} has turned"
                f" {yaw - start_yaw:+7.2f} deg from where it started"
            )
        if tilt > 1.0:
            site = np.asarray(env.data.site_xpos[env.ik.left_site], dtype=np.float64)
            jaws = [env.privileged_left_finger_contacts(cookie) for cookie in range(2)]
            print(
                f"  {step:4d}  {phase:16s} cookie {index}  tilt {tilt:6.1f}"
                f"  c0 {rows[0][0]} {rows[0][1]:4.1f}"
                f"  c1 {rows[1][0]} {rows[1][1]:4.1f}"
                f"  tool [{site[0]:6.3f} {site[1]:6.3f} {site[2]:6.3f}]"
                f"  jaws {jaws[0]} {jaws[1]}"
            )
            history.append(step)
            if len(history) > 12:
                break
        if terminated or truncated:
            break

    print(f"\nfinal: phase={expert.phase.name} reason={expert.failure_reason}")
    print(f"  success={info.get('success')}  steps={step}")
    for cookie in range(2):
        rows = box_frame(env, cookie)
        print(f"  cookie {cookie}: box frame {rows[0]}  tilt {rows[1]:.1f} deg")
    print("  packed positions the expert recorded (box frame, from the environment):")
    for cookie, packed in enumerate(expert._packed_positions):
        print(f"    {cookie}: {None if packed is None else np.round(np.asarray(packed) * 1000, 2)}")
    env.close()
    return 0


def yaw_of(env, cookie: int) -> float:
    """The Cookie's heading about the vertical axis, in degrees.

    This is the quantity that a tilt check misses: a Cookie turned about the vertical axis keeps
    its thin axis vertical, so it still reads as 0.1 degrees of tilt while its silhouette along
    the jaws' closing axis has gone from 6.4 mm to 50 mm.
    """

    matrix = env._cookie_rotation(cookie)
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def box_frame(env, cookie: int) -> tuple[str, float]:
    """The Cookie's position in the target box's frame, and how far it has tipped."""

    bin_position = env.data.xpos[env._target_bin_body].copy()
    bin_matrix = env.data.xmat[env._target_bin_body].reshape(3, 3).copy()
    cookie_position = env.data.xpos[env._cookie_bodies[cookie]].copy()
    cookie_matrix = env._cookie_rotation(cookie)
    relative = bin_matrix.T @ (cookie_position - bin_position)
    upright = float(np.abs(cookie_matrix[2, 2]))
    tilt = float(np.degrees(np.arccos(np.clip(upright, -1.0, 1.0))))
    position = np.round(relative * 1000.0, 1)
    return f"[{position[0]:6.1f} {position[1]:6.1f} {position[2]:6.1f}]", tilt


if __name__ == "__main__":
    raise SystemExit(main())
