"""Solve a Panda home pose whose hand sits above the Cookie, tool pointing down.

Written down as a constant afterwards, the way `LIBERO_PANDA_HOME_RAD` is: a measured
number with its provenance, not something recomputed at import.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np
from scipy.optimize import least_squares

from a3_dual_arm_sim.collection import libero_props_scene
from a3_dual_arm_sim.contracts import LEFT_JOINTS

# The registered layout: A3's arrangement slid by (0.02, -0.20), Cookie standing in the
# bin's near-left corner.
COOKIE_XY = (0.0994 + 0.020, 0.2311 - 0.200)
HOVER_HEIGHT_M = 0.120

scene = libero_props_scene(task_name="a3_probe_home_over_cookie", matched_cameras=True,
                           panda=True, a3_single_cookie=True,
                           a3_layout_offset_m=(0.020, -0.200))
env = scene.build_env(0.0, 0.0)
env.reset(seed=0, options=None)
model, data = env.model, env.data

joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_JOINTS]
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L_eef")
qpos_addr = [int(model.jnt_qposadr[j]) for j in joint_ids]

home = np.asarray(env.config.cookie_transfer.deployment_home[:7], dtype=np.float64)
target_pos = np.array((COOKIE_XY[0], COOKIE_XY[1], 0.7683 + HOVER_HEIGHT_M))

scratch = mujoco.MjData(model)
scratch.qpos[:] = data.qpos


def residuals(q):
    scratch.qpos[:] = data.qpos
    scratch.qpos[qpos_addr] = q
    mujoco.mj_forward(model, scratch)
    pos = np.asarray(scratch.site_xpos[site])
    rot = np.asarray(scratch.site_xmat[site]).reshape(3, 3)
    # Orientation residual: the tool axis (local z) must stay straight down, which is what
    # LIBERO's home has and what the wrist camera needs to look at the table.
    return np.r_[1000.0 * (pos - target_pos), 300.0 * (rot[:, 2] - np.array((0.0, 0.0, -1.0)))]


solution = least_squares(residuals, home, bounds=(-3.2, 3.2), xtol=1e-12, ftol=1e-12)
q = solution.x
scratch.qpos[:] = data.qpos
scratch.qpos[qpos_addr] = q
mujoco.mj_forward(model, scratch)
print(f"LIBERO's home       {np.round(home, 6)}")
print(f"  eef               {np.round(env.data.site_xpos[site], 4)}")
print(f"solved home         {tuple(np.round(q, 6))}")
print(f"  eef               {np.round(scratch.site_xpos[site], 4)}")
print(f"  target            {np.round(target_pos, 4)}   "
      f"residual {np.linalg.norm(scratch.site_xpos[site] - target_pos) * 1000:.2f} mm")
print(f"  tool axis         {np.round(np.asarray(scratch.site_xmat[site]).reshape(3, 3)[:, 2], 4)}")
print(f"  cost              {solution.cost:.6g}")
env.close()
