"""How much reorientation does the expert demand, and is that the same question on both arms?

The expert's grasp orientation is a constant, `_target_quat_canonical`, and `_solve_l` -- which
decides whether a waypoint is reachable -- checks only the **position** residual.  So a waypoint
whose orientation the arm cannot reach is accepted as reachable, and the cartesian command then
asks for an orientation the arm cannot hold.  That is invisible on A3's arm, where the constant was
written and is attainable, and it would stall on any other arm.

This prints, at each arm's own start pose, the tool's orientation and the angle between it and the
constant -- and, more usefully, the same angle between the arm's own *grasp* stances, so the
question "is this constant a pose or a convention?" gets an answer rather than an argument.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene
from a3_dual_arm_sim.expert import _rotation_vector

LAYOUT = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
)

CANONICAL = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """Angle, in radians, between two orientations given as rotation matrices."""

    relative = first.T @ second
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(relative).reshape(-1))
    return float(np.linalg.norm(_rotation_vector(quaternion)))


for panda in (False, True):
    scene = libero_props_scene(
        task_name="a3_libero_panda_single_cookie", panda=panda, cartesian=True, **LAYOUT
    )
    env = scene.build_env(0.0, 0.0)
    env.reset(seed=0, options=None)
    model, data = env.model, env.data
    label = "panda" if panda else "a3"
    print(f"\n=== {label} ===")
    for name in ("L_eef", "gripper0_grip_site"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if index < 0:
            continue
        rotation = np.asarray(data.site_xmat[index]).reshape(3, 3).copy()
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[index])
        print(f"  {name:20s} on {body:22s}")
        print(f"    xmat {np.round(rotation, 3).tolist()}")
        print(f"    z axis (world) {np.round(rotation[:, 2], 3)}")
        print(f"    angle to the expert's canonical orientation "
              f"{np.degrees(angle_between(rotation, CANONICAL)):7.1f} deg")
    # The expert's own planned approach pose, i.e. what it will actually command.
    policy = scene.build_policy(env)
    policy.reset()
    expert = policy.expert
    expert.act()
    if "approach" in expert._waypoints:
        goal_position, goal_quaternion = expert._pose_of(expert._waypoints["approach"])
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, goal_quaternion)
        rotation = rotation.reshape(3, 3)
        current = np.asarray(data.site_xmat[env.ik.left_site]).reshape(3, 3).copy()
        print(f"  planned approach at {np.round(goal_position, 4)}, on "
              f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, env.ik.left_site)}")
        print(f"    its z axis (world) {np.round(rotation[:, 2], 3)}")
        print(f"    reorientation the command demands at step 0: "
              f"{np.degrees(angle_between(current, rotation)):7.1f} deg")
    env.close()
