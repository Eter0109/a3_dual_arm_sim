"""How thin an object can the Panda's gripper actually pinch?

The grasp reaches the right pose, the pads arrive either side of the Cookie, the jaws run to their
travel limit -- and the contact is a 0.03 mm graze carrying no measurable force, so
`_has_verified_grasp`'s 0.02 threshold is never met and the expert retries a grasp it is already
touching.  Either the pads are in the wrong place, or **this gripper cannot close onto a 6.4 mm
slab at all**, and those are different problems with different next steps.

This measures the second one directly: drive the fingers to their closed limit with nothing between
them and ask MuJoCo how far apart the two pads are.  If the answer is about 6.4 mm, the Cookie sits
at the gripper's minimum opening and no amount of path planning will produce a squeeze.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

FINGER_GEOMS = ("L_finger_inner_geom", "L_finger_outer_geom")
FINGER_JOINTS = ("L_finger_inner_joint", "L_finger_outer_joint")

scene = libero_props_scene(
    task_name="a3_libero_panda_single_cookie",
    cartesian=True,
    panda=True,
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
)
env = scene.build_env(0.0, 0.0)
env.reset(seed=0, options=None)
model, data = env.model, env.data

inner, outer = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FINGER_GEOMS)

print(f"Cookie thickness along its thin axis: {env.COOKIE_HALF_SIZE[1] * 2000:.2f} mm")
print()
print("  finger joint qpos     pad-to-pad distance between the two jaw collision meshes")
for opening in (0.04, 0.03, 0.02, 0.01, 0.005, 0.002, 0.0):
    for name, value in zip(FINGER_JOINTS, (opening, -opening), strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint])] = value
        data.qvel[int(model.jnt_dofadr[joint])] = 0.0
    mujoco.mj_forward(model, data)
    distance = float(mujoco.mj_geomDistance(model, data, inner, outer, 1.0, None))
    print(f"  {opening:+.4f} rad ({opening * 1000:+6.2f} mm jaw)   {distance * 1000:8.2f} mm")

env.close()
