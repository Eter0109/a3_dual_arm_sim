"""Which body do `L_eef` and robosuite's grasp site actually belong to?

If `L_eef` is on the *A3* arm and `LEFT_JOINTS` are the Panda's, then the expert's `_solve_l`
is solving for a site that does not move, and nothing downstream of it can be right.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene
from a3_dual_arm_sim.contracts import LEFT_JOINTS

scene = libero_props_scene(
    task_name="a3_libero_panda_single_cookie",
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
    panda=True,
    cartesian=True,
)
env = scene.build_env(0.0, 0.0)
env.reset(seed=0, options=None)
model, data = env.model, env.data


def site(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


names = ("L_eef", "R_eef", "gripper0_grip_site")
for name in names:
    index = site(name)
    if index < 0:
        print(f"{name:20s} MISSING")
        continue
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[index])
    local = np.asarray(data.site_xpos[index]) - np.asarray(data.xpos[model.site_bodyid[index]])
    print(
        f"{name:20s} body={body:26s} xpos={np.round(data.site_xpos[index], 4)}  "
        f"offset from body origin {np.round(local * 1000, 1)} mm"
    )

print(f"\nLEFT_JOINTS (the environment calls these the left arm): {LEFT_JOINTS}")
addresses = [
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
    for n in LEFT_JOINTS
]
for name in LEFT_JOINTS:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[joint])
    print(f"  {name:22s} body={body}")

print("\nmoving LEFT_JOINTS[0] by 0.5 rad:")
before = {n: np.asarray(data.site_xpos[site(n)]).copy() for n in names if site(n) >= 0}
data.qpos[addresses[0]] += 0.5
mujoco.mj_forward(model, data)
for name, was in before.items():
    now = np.asarray(data.site_xpos[site(name)])
    print(f"  {name:20s} moved {np.linalg.norm(now - was) * 1000:8.2f} mm")
env.close()
