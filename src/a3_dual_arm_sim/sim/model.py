# Explicit same-name imports preserve the legacy public API.
# ruff: noqa: PLC0414
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import mujoco

from a3_dual_arm_sim.config import SimConfig
from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
from a3_dual_arm_sim.paths import asset_root

from .model_types import JointSource as JointSource
from .model_types import ModelBundle as ModelBundle
from .object_builder import BIN_RGBA as BIN_RGBA
from .object_builder import _add_beveled_cookie_mesh as _add_beveled_cookie_mesh
from .object_builder import _add_cookie_scene as _add_cookie_scene
from .object_builder import _add_open_bin as _add_open_bin
from .robot_builder import ROBOTIQ_2F85_JAW_TRAVEL_M as ROBOTIQ_2F85_JAW_TRAVEL_M
from .robot_builder import ROBOTIQ_2F85_MAX_OPENING_M as ROBOTIQ_2F85_MAX_OPENING_M
from .robot_builder import _add_gripper as _add_gripper
from .robot_builder import _add_inertial as _add_inertial
from .robot_builder import _add_link_geometry as _add_link_geometry
from .urdf import _floats as _floats
from .urdf import _source as _source
from .urdf import _vec as _vec

SceneName = Literal["sandbox", "cookie_transfer"]


def build_model(config: SimConfig, *, scene: SceneName = "sandbox") -> ModelBundle:
    if scene not in ("sandbox", "cookie_transfer"):
        raise ValueError(f"unsupported scene: {scene}")
    urdf, source_joints = _source()
    links = {node.attrib["name"]: node for node in urdf.findall("link")}
    root = ET.Element("mujoco", model=f"a3_dual_arm_{scene}")
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        autolimits="true",
        balanceinertia="true",
    )
    option = ET.SubElement(
        root,
        "option",
        timestep=f"{1 / config.physics_hz:.10g}",
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="elliptic" if scene == "cookie_transfer" else "pyramidal",
        impratio="10" if scene == "cookie_transfer" else "1",
        noslip_iterations="5" if scene == "cookie_transfer" else "0",
    )
    if scene == "cookie_transfer":
        # Flat beveled meshes need a contact patch, not a single rocking point.
        ET.SubElement(option, "flag", multiccd="enable")
    ET.SubElement(root, "size", nconmax="400", njmax="1000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        offwidth=str(config.image_width),
        offheight=str(config.image_height),
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        limited="true",
        damping=str(config.joint_damping),
        armature=str(config.joint_armature),
    )
    if scene == "cookie_transfer":
        ET.SubElement(default, "geom", solref="0.04 0.85", solimp="0.75 0.97 0.002 0.5 2")
    else:
        ET.SubElement(default, "geom", solref="0.01 1", solimp="0.9 0.95 0.001")

    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.75 0.82 0.9",
        rgb2="0.12 0.16 0.24",
        width="512",
        height="3072",
    )
    ET.SubElement(assets, "material", name="floor", rgba="0.22 0.24 0.27 1", reflectance="0.05")
    ET.SubElement(assets, "material", name="table", rgba="0.52 0.34 0.18 1")
    for link_name in ("base_link", *(joint.child for joint in source_joints.values())):
        ET.SubElement(assets, "mesh", name=f"mesh_{link_name}", file=f"{link_name}.STL")
    ET.SubElement(
        assets,
        "mesh",
        name="robotiq_arg2f_85_base_link",
        file="robotiq_arg2f_85_base_link.stl",
    )
    for mesh_name in (
        "robotiq_arg2f_85_outer_knuckle_vis",
        "robotiq_arg2f_85_outer_finger_vis",
        "robotiq_arg2f_85_inner_finger_vis",
        "robotiq_arg2f_85_inner_knuckle_vis",
    ):
        ET.SubElement(
            assets,
            "mesh",
            name=mesh_name,
            file=f"{mesh_name}.stl",
            scale="0.001 0.001 0.001",
        )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.4 -0.8 2.4", dir="-0.2 0.3 -1", diffuse="0.9 0.9 0.9")
    ET.SubElement(world, "light", pos="-0.6 0.8 1.8", dir="0.2 -0.3 -1", diffuse="0.45 0.45 0.45")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="3 3 0.1",
        material="floor",
        contype="1",
        conaffinity="3",
    )
    ET.SubElement(
        world,
        "geom",
        name="table_top",
        type="box",
        pos="0.15 0 0.725",
        size="0.55 0.75 0.025",
        material="table",
        contype="1",
        conaffinity="3",
        friction="1 0.01 0.001",
    )
    base_height = config.cookie_transfer.base_height_m if scene == "cookie_transfer" else 0.98
    ET.SubElement(
        world,
        "geom",
        name="stand_mast",
        type="box",
        pos=f"-0.35 0 {base_height / 2}",
        size=f"0.040 0.040 {base_height / 2}",
        rgba="0.08 0.09 0.10 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        name="stand_bracket",
        type="box",
        pos=f"-0.35 0 {base_height - 0.07}",
        size="0.10 0.12 0.035",
        rgba="0.18 0.19 0.21 1",
        contype="0",
        conaffinity="0",
    )
    target = ET.SubElement(
        world,
        "body",
        name="workspace_target",
        pos=_vec(config.cameras.workspace_target_m),
    )
    ET.SubElement(target, "site", size="0.005", rgba="0 0 0 0")
    ET.SubElement(
        world,
        "camera",
        name="front",
        mode="targetbody",
        target="workspace_target",
        pos=_vec(config.cameras.front_position_m),
        fovy=f"{config.cameras.front_fovy_deg:.10g}",
    )

    base = ET.SubElement(world, "body", name="base_link", pos=f"-0.35 0 {base_height}")
    _add_inertial(base, links["base_link"])
    ET.SubElement(
        base,
        "geom",
        name="base_visual",
        type="mesh",
        mesh="mesh_base_link",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        base,
        "geom",
        name="base_collision",
        type="box",
        size="0.06 0.10 0.06",
        rgba="0.25 0.3 0.34 0.2",
        contype="2",
        conaffinity="1",
        group="3",
    )

    child_map = {
        joint.parent: joint for joint in source_joints.values() if joint.name in LEFT_JOINTS
    }
    child_map.update(
        {joint.parent: joint for joint in source_joints.values() if joint.name in RIGHT_JOINTS}
    )

    def add_chain(parent: ET.Element, joint: JointSource, side: str) -> None:
        body = ET.SubElement(parent, "body", name=joint.child, pos=_vec(joint.origin))
        ET.SubElement(
            body,
            "joint",
            name=joint.name,
            type="hinge",
            axis=_vec(joint.axis),
            range=_vec(joint.limits),
            damping=str(config.joint_damping),
            armature=str(config.joint_armature),
        )
        _add_inertial(body, links[joint.child])
        next_joint = child_map.get(joint.child)
        direction = 1.0 if side == "left" else -1.0
        length = abs(next_joint.origin[1]) if next_joint is not None else 0.0815
        if length < 0.02:
            length = 0.0815
        _add_link_geometry(body, joint.child, direction, length)
        if next_joint is None:
            _add_gripper(body, side, direction, config, cookie_scene=scene == "cookie_transfer")
        else:
            add_chain(body, next_joint, side)

    add_chain(base, source_joints[LEFT_JOINTS[0]], "left")
    add_chain(base, source_joints[RIGHT_JOINTS[0]], "right")

    if scene == "cookie_transfer":
        # Ideal model-based gravity compensation for the prototype arm servos.
        # Route compensation through actuators so joint force limits still apply.
        for body in base.iter("body"):
            body.set("gravcomp", "1")
        for joint in base.iter("joint"):
            joint.set("actuatorgravcomp", "true")
            source = source_joints.get(joint.attrib.get("name"))
            effort = source.effort if source is not None else 40
            joint.set("actuatorfrcrange", f"{-effort} {effort}")

    if scene == "cookie_transfer":
        _add_cookie_scene(root, world, config)
    else:
        colors = (
            (0.68, -0.22, 0.79, "0.85 0.15 0.12 1"),
            (0.72, 0.0, 0.79, "0.12 0.55 0.9 1"),
            (0.68, 0.22, 0.79, "0.18 0.75 0.3 1"),
        )
        for index, (x, y, z, rgba) in enumerate(colors):
            obj = ET.SubElement(world, "body", name=f"object_{index}", pos=f"{x} {y} {z}")
            ET.SubElement(obj, "freejoint", name=f"object_{index}_free")
            ET.SubElement(
                obj,
                "geom",
                name=f"object_{index}_geom",
                type="box",
                size="0.035 0.035 0.04",
                rgba=rgba,
                mass="0.10",
                contype="1",
                conaffinity="3",
                friction="1.5 0.02 0.001",
            )

    actuators = ET.SubElement(root, "actuator")
    for joint in source_joints.values():
        kp = 600.0 if joint.effort >= 60 else (400.0 if joint.effort >= 30 else 200.0)
        kv = 40.0 if joint.effort >= 60 else (30.0 if joint.effort >= 30 else 15.0)
        ET.SubElement(
            actuators,
            "position",
            name=f"{joint.name}_position",
            joint=joint.name,
            kp=str(kp),
            kv=str(kv),
            ctrlrange=_vec(joint.limits),
            forcerange=f"{-joint.effort} {joint.effort}",
        )
    for prefix in ("L", "R"):
        for finger in ("inner", "outer"):
            name = f"{prefix}_finger_{finger}"
            ET.SubElement(
                actuators,
                "position",
                name=f"{name}_position",
                joint=f"{name}_joint",
                kp=str(
                    config.cookie_transfer.right_gripper_kp
                    if prefix == "R"
                    else config.cookie_transfer.left_gripper_kp
                )
                if scene == "cookie_transfer"
                else "120",
                ctrlrange=f"0 {ROBOTIQ_2F85_JAW_TRAVEL_M}",
                forcerange="-40 40",
            )

    sensors = ET.SubElement(root, "sensor")
    for prefix in ("L", "R"):
        ET.SubElement(sensors, "force", name=f"{prefix}_wrist_force", site=f"{prefix}_wrist_ft")
        ET.SubElement(sensors, "torque", name=f"{prefix}_wrist_torque", site=f"{prefix}_wrist_ft")
        for finger in ("inner", "outer"):
            ET.SubElement(
                sensors,
                "touch",
                name=f"{prefix}_finger_{finger}_touch_sensor",
                site=f"{prefix}_finger_{finger}_touch",
            )

    xml = ET.tostring(root, encoding="unicode")
    mesh_dir = asset_root() / "meshes"
    binary_assets = {
        path.name: path.read_bytes() for path in mesh_dir.iterdir() if path.suffix.lower() == ".stl"
    }
    model = mujoco.MjModel.from_xml_string(xml, assets=binary_assets)
    return ModelBundle(model=model, xml=xml, source_joints=tuple(source_joints.values()))


def write_generated_xml(
    path: str | Path,
    config: SimConfig,
    *,
    scene: SceneName = "sandbox",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(build_model(config, scene=scene).xml)
    compiler = root.find("compiler")
    if compiler is not None:
        try:
            relative_meshes = os.path.relpath(asset_root() / "meshes", destination.parent)
            compiler.set("meshdir", Path(relative_meshes).as_posix())
        except ValueError:
            compiler.set("meshdir", (asset_root() / "meshes").as_posix())
    destination.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return destination
