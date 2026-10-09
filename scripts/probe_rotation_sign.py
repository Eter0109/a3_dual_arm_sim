"""Which way does a rotation command actually turn the arm?

The Panda's rotation error *grew* towards pi while its position converged (225 mm to 4.8 mm in 24
steps, rotation 2.20 rad to 3.11 rad), so the arm turned away from the goal until the shorter arc
flipped and it oscillated there.  The same code converges on A3's arm, so the question is not
whether the algebra is self-consistent -- it is which frame each controller reads the rotation
part of a `cartesian_delta` in, and that is a property of the controller, not of the algebra.

This drives one thing at a time.  It takes the arm's actual pose, picks a goal orientation a known
angle to one side of it, commands **only** a rotation (translation entry zero, so the controller
holds position), and prints how far the orientation actually moved and which way.  Comparing
"which way" with what `_goal_error` asked for is the whole measurement.

Run it for each arm:

    python scripts/probe_rotation_sign.py --panda
    python scripts/probe_rotation_sign.py
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene
from a3_dual_arm_sim.expert import _rotation_vector


def as_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).reshape(-1))
    return quaternion


def relative(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``first^-1 @ second`` as a rotation vector: the rotation taking first to second."""

    conjugate = np.r_[first[0], -first[1:4]]
    product = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(product, conjugate, second)
    return _rotation_vector(product)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panda", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--frame",
        choices=("world", "tool"),
        default="world",
        help="which frame the controller is assumed to read the rotation delta in",
    )
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name="a3_libero_panda_single_cookie",
        matched_cameras=True,
        a3_single_cookie=True,
        a3_layout_offset_m=(0.020, -0.200),
        panda=args.panda,
        cartesian=True,
    )
    env = scene.build_env(0.0, 0.0)
    env.reset(seed=0, options=None)
    model, data = env.model, env.data
    site = env.ik.left_site
    scale = env.config.cartesian_rotation_scale_rad
    print(
        f"panda={args.panda}  site "
        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site)}"
        f"  scale {scale} rad/step  frame={args.frame}"
        f"  config says cartesian_rotation_frame={env.config.cartesian_rotation_frame!r}"
    )

    start = as_quaternion(np.asarray(data.site_xmat[site]).reshape(3, 3).copy())
    step_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(step_quat, np.array([0.0, 1.0, 0.0]), 0.30)
    goal = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(goal, step_quat, start)

    print("  goal is 0.300 rad about +y from the start; |to goal| is frame-independent,")
    print("  so whether it falls or rises is the whole answer.")
    for index in range(args.steps):
        current = as_quaternion(np.asarray(data.site_xmat[site]).reshape(3, 3).copy())
        rotation = np.asarray(data.site_xmat[site]).reshape(3, 3).copy()
        local = relative(current, goal)
        asked = local if args.frame == "tool" else rotation @ local
        action = np.zeros(14, dtype=np.float64)
        action[3:6] = np.clip(asked / scale, -1.0, 1.0)
        before = np.linalg.norm(local)
        env.step(action)
        after = as_quaternion(np.asarray(data.site_xmat[site]).reshape(3, 3).copy())
        remaining = np.linalg.norm(relative(after, goal))
        if index < 6 or index % 4 == 3:
            print(
                f"  {index:3d}  |to goal| {before:.4f} -> {remaining:.4f}"
                f"   asked {np.round(asked, 3)}"
                f"   cmd {np.round(action[3:6], 3)}"
            )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
