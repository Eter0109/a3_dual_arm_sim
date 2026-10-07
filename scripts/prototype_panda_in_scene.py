"""De-risking prototype: does robosuite's Panda compile inside this scene's model?

The swap is invasive enough that the first question is not "where does the code go" but
"does it compile at all".  Our model is assembled from a URDF with a virtual asset
dictionary and its own `<default>`, `<compiler>` and naming conventions; robosuite's Panda is
a 35 kB MJCF with 63 meshes, 50 materials and its own names.  A clash in any of those is a
compile error that says nothing about where the clash is.

So this does the smallest thing that can fail: write the scene's own XML out, move the
Panda's asset declarations and its `robot0_base` tree in beside ours, and compile.  It does
**not** remove the A3 arm, rename anything, or add actuators -- the Panda is a second,
inert robot here, and the only claim being tested is that the two can share one model.

Usage::

    python scripts/prototype_panda_in_scene.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

#: LIBERO's own base for `libero_spatial`, through the world offset this scene already uses,
#: so the arm stands where it stands in LIBERO and its reach covers the same objects.
LIBERO_WORLD_OFFSET_M = (0.2135, -0.0720, -0.1500)


def panda_tree() -> ET.Element:
    """robosuite's Panda with its gripper merged, as a `<mujoco>` root."""

    from robosuite.models.grippers import PandaGripper
    from robosuite.models.robots import Panda

    robot = Panda(idn=0)
    robot.merge(PandaGripper(idn=0), merge_body="robot0_right_hand")
    return ET.fromstring(robot.get_xml())


def main() -> int:
    import mujoco

    from a3_dual_arm_sim.config import load_config
    from a3_dual_arm_sim.model import write_generated_xml

    config = load_config(PROJECT / "configs/libero_props.yaml")
    scene_path = Path("/tmp/a3_scene.xml")
    write_generated_xml(scene_path, config, scene="cookie_transfer")
    ours = ET.parse(scene_path).getroot()
    print(f"our scene: {scene_path} ({scene_path.stat().st_size} bytes)")

    panda = panda_tree()
    print("panda 顶层块:", [child.tag for child in panda])
    print("panda 里带 class= 的元素:",
          sum(1 for element in panda.iter() if element.get("class")),
          "| 带 default= 的:", sum(1 for element in panda.iter() if element.get("default")))

    our_assets = ours.find("asset")
    clash = []
    existing = {element.get("name") for element in our_assets}
    for element in list(panda.find("asset")):
        name = element.get("name")
        if name in existing:
            clash.append((element.tag, name))
        our_assets.append(element)
    print(f"搬入 asset：{len(list(panda.find('asset')))} 项，重名 {len(clash)} 个"
          + (f" -> {clash[:6]}" if clash else ""))

    base = list(panda.find("worldbody"))[0]
    base.set("pos", " ".join(f"{v:.6f}" for v in LIBERO_WORLD_OFFSET_M))
    ours.find("worldbody").append(base)
    print(f"挂上 {base.get('name')} 于 pos={base.get('pos')}")

    # Our actuators stay ours; theirs are dropped.  The arm is inert in this prototype, and
    # an actuator block naming joints that are also driven by ours would be the one thing
    # that produces motion by accident.
    dropped = [child.tag for child in panda if child.tag in ("actuator", "sensor", "tendon", "equality")]
    print("未搬入:", dropped)

    combined = Path("/tmp/a3_scene_with_panda.xml")
    ET.ElementTree(ours).write(combined, encoding="unicode")
    print(f"写出 {combined} ({combined.stat().st_size} bytes)，编译中 ...")

    model = mujoco.MjModel.from_xml_path(str(combined))
    print(f"**编译通过**  nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
          f"nmesh={model.nmesh} nmat={model.nmat} nu={model.nu} ncam={model.ncam}")

    import numpy as np
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"  robot0_right_hand 世界位置 {np.round(data.xpos[hand], 4)}")
    print(f"  gripper0_grip_site 世界位置 {np.round(data.site_xpos[grip], 4)}")
    print(f"  （LIBERO 起始 grip site = [-0.2112 -0.0111 1.1742] + 偏移 = "
          f"{np.round(np.array([-0.2112, -0.0111, 1.1742]) + np.array(LIBERO_WORLD_OFFSET_M), 4)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
