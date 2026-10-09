"""Which body carries each of the Panda hand's geoms and sites?

The rename in `model._add_libero_panda` names `L_finger_inner_geom` by looking for the first
geom under `gripper0_leftfinger` whose XML `contype` attribute is not "0", and robosuite sets
`contype` from `<default>` classes rather than per element -- so the attribute is absent, the
fallback "1" passes, and the *visual* mesh gets the name while the collision mesh keeps its own.
This prints the compiled truth so the choice can be made against it instead of against the XML.

Writes to a file because the terminal wraps and truncates long tables.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

OUT = "/tmp/hand_report.txt"

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
model = env.model

lines: list[str] = []


def body_name(index: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or f"<body {index}>"


lines.append("geoms on the hand and its jaws, with their compiled collision flags:")
for geom in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    body = body_name(int(model.geom_bodyid[geom]))
    if name is None or not (name.startswith(("gripper0", "L_finger")) or "finger" in body):
        continue
    lines.append(
        f"  {name:34s} body={body:28s} contype={int(model.geom_contype[geom])}"
        f" conaffinity={int(model.geom_conaffinity[geom])}"
        f" size={np.round(model.geom_size[geom], 5)}"
    )

lines.append("")
lines.append("bodies under the hand, with the geoms and sites each carries:")
for body in range(model.nbody):
    name = body_name(body)
    if not name.startswith(("gripper0", "robot0_right_hand")):
        continue
    geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"<geom {geom}>"
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) == body
    ]
    sites = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site) or f"<site {site}>"
        for site in range(model.nsite)
        if int(model.site_bodyid[site]) == body
    ]
    lines.append(f"  {name:28s} geoms={geoms} sites={sites}")

lines.append("")
lines.append("touch sensors and the body their site belongs to:")
for sensor in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor)
    if name is None or "touch" not in name:
        continue
    kind = mujoco.mjtObj(int(model.sensor_objtype[sensor]))
    target = int(model.sensor_objid[sensor])
    target_name = mujoco.mj_id2name(model, kind, target)
    line = (
        f"  {name:30s} type={mujoco.mjtSensor(int(model.sensor_type[sensor])).name}"
        f" objtype={kind.name} obj={target_name}"
    )
    if kind == mujoco.mjtObj.mjOBJ_SITE:
        line += f"  site body={body_name(int(model.site_bodyid[target]))}"
    lines.append(line)

with open(OUT, "w") as stream:
    stream.write("\n".join(lines) + "\n")
print(f"wrote {OUT}")
env.close()
