"""Is the solver's home an equilibrium, or is the arm grinding against something?

The least-squares solve constrained the eef site and nothing else.  So: step the physics with no
policy and no action, and see whether the joints hold the commanded pose, and whether anything
is in contact with the arm.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import LIBERO_PANDA_HOME_OVER_COOKIE_RAD, scene_by_name
from a3_dual_arm_sim.contracts import LEFT_JOINTS

env = scene_by_name("a3_probe_home_over_cookie").build_env(0.0, 0.0)
env.reset(seed=0, options=None)
model, data = env.model, env.data

addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
             for n in LEFT_JOINTS]
commanded = np.array(LIBERO_PANDA_HOME_OVER_COOKIE_RAD)
print(f"commanded        {np.round(commanded, 5)}")
print(f"at reset         {np.round(data.qpos[addresses], 5)}")
start = np.asarray(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L_eef")
                            if False else 0])  # unused, kept for symmetry

for _ in range(300):
    mujoco.mj_step(model, data)
held = np.asarray(data.qpos[addresses])
print(f"after 300 steps  {np.round(held, 5)}")
print(f"drift from commanded, per joint (mrad): {np.round((held - commanded) * 1000, 2)}")
print(f"largest drift {np.abs(held - commanded).max() * 1000:.2f} mrad")

eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L_eef")
print(f"eef now          {np.round(data.site_xpos[eef], 4)}")

print("\ncontacts involving the Panda, after settling:")
named = {i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
lo = min(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in
         ("robot0_base", "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
          "robot0_link5", "robot0_link6", "robot0_link7", "robot0_right_hand",
          "gripper0_leftfinger", "gripper0_rightfinger"))
found = 0
for index in range(data.ncon):
    contact = data.contact[index]
    a, b = named.get(int(model.geom_bodyid[contact.geom1])), \
        named.get(int(model.geom_bodyid[contact.geom2]))
    if a is not None and (a.startswith(("robot0", "gripper0")) or
                          (b and b.startswith(("robot0", "gripper0")))):
        print(f"  {a} <-> {b}   force {np.round(contact.frame[:3], 3)}")
        found += 1
print(f"  {found} contact(s)")
env.close()
