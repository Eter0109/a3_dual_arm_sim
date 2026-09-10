from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import mujoco

from .config import SimConfig
from .contracts import ARM_JOINTS, LEFT_JOINTS, RIGHT_JOINTS
from .paths import asset_root


@dataclass(frozen=True)
class JointSource:
    name: str
    parent: str
    child: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits: tuple[float, float]
    effort: float
    velocity: float


@dataclass(frozen=True)
class ModelBundle:
    model: mujoco.MjModel
    xml: str
    source_joints: tuple[JointSource, ...]


SceneName = Literal["sandbox", "cookie_transfer"]

COOKIE_HALF_SIZE = (0.025, 0.0095, 0.025)
COOKIE_SOURCE_POSITIONS = tuple(
    [
        # Pick cookies (indices 0..9):
        (0.065, 0.340),  # 0
        (0.065, 0.370),  # 1
        (0.065, 0.400),  # 2
        (0.065, 0.430),  # 3
        (0.065, 0.460),  # 4
        (0.120, 0.340),  # 5
        (0.120, 0.370),  # 6
        (0.120, 0.400),  # 7
        (0.120, 0.430),  # 8
        (0.065, 0.490),  # 9
        # Remaining cookies in source bin (indices 10..29):
        (0.120, 0.460),  # 10
        (0.120, 0.490),  # 11
    ]
    + [
        (cx, cy)
        for cx in (0.175, 0.230, 0.285)
        for cy in (0.340, 0.370, 0.400, 0.430, 0.460, 0.490)
    ]
)
BIN_WALL_THICKNESS = 0.006
SOURCE_BIN_CENTER = (0.175, 0.415)
SOURCE_BIN_HALF_SIZE = (0.143, 0.093)
TARGET_BIN_CENTER = (0.120, 0.240)
TARGET_BIN_HALF_SIZE = (0.068, 0.075)
SOURCE_BIN_WALL_HEIGHT = 0.035
TARGET_BIN_WALL_HEIGHT = 0.025
TARGET_BIN_ATTACH_POS = (-0.284576, -0.120338, 0.077169)
TARGET_BIN_ATTACH_QUAT = (0.368302, 0.770142, -0.144659, 0.500309)
BIN_RGBA = "0.70 0.73 0.77 1"
ROBOTIQ_2F85_MAX_OPENING_M = 0.085
ROBOTIQ_2F85_JAW_TRAVEL_M = ROBOTIQ_2F85_MAX_OPENING_M / 2


def _floats(text: str | None, count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in (text or "").split())
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {text!r}")
    return values


def _source() -> tuple[ET.Element, dict[str, JointSource]]:
    root = ET.parse(asset_root() / "source" / "A3_mujoco.urdf").getroot()
    joints: dict[str, JointSource] = {}
    for node in root.findall("joint"):
        name = node.attrib["name"]
        if node.attrib.get("type") != "revolute":
            continue
        limit = node.find("limit")
        if limit is None:
            raise ValueError(f"joint {name} has no limit")
        joints[name] = JointSource(
            name=name,
            parent=node.find("parent").attrib["link"],  # type: ignore[union-attr]
            child=node.find("child").attrib["link"],  # type: ignore[union-attr]
            origin=_floats(node.find("origin").attrib.get("xyz"), 3),  # type: ignore[union-attr]
            axis=_floats(node.find("axis").attrib.get("xyz"), 3),  # type: ignore[union-attr]
            limits=(float(limit.attrib["lower"]), float(limit.attrib["upper"])),
            effort=float(limit.attrib["effort"]),
            velocity=float(limit.attrib["velocity"]),
        )
    if tuple(joints) != ARM_JOINTS:
        raise ValueError(f"unexpected A3 joint order: {tuple(joints)}")
    return root, joints


def _vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _add_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        return
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    tensor = inertial.find("inertia")
    if origin is None or mass is None or tensor is None:
        return
    ET.SubElement(
        body,
        "inertial",
        pos=origin.attrib.get("xyz", "0 0 0"),
        mass=mass.attrib["value"],
        fullinertia=" ".join(
            tensor.attrib[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    )


def _add_link_geometry(body: ET.Element, link_name: str, direction: float, length: float) -> None:
    ET.SubElement(
        body,
        "geom",
        name=f"{link_name}_visual",
        type="mesh",
        mesh=f"mesh_{link_name}",
        contype="0",
        conaffinity="0",
        group="2",
    )
    radius = 0.046 if "SHOULDER" in link_name else 0.038
    ET.SubElement(
        body,
        "geom",
        name=f"{link_name}_collision",
        type="capsule",
        fromto=f"0 0 0 0 {direction * length:.6f} 0",
        size=f"{radius:.5f}",
        rgba="0.2 0.35 0.8 0.12",
        contype="2",
        conaffinity="1",
        friction="0.8 0.01 0.001",
        group="3",
    )


def _add_gripper(parent: ET.Element, side: str, direction: float) -> None:
    prefix = side[0].upper()
    flange = ET.SubElement(
        parent,
        "body",
        name=f"{prefix}_flange",
        pos=f"0 {direction * 0.0815:.6f} 0",
    )
    ET.SubElement(
        flange, "geom", name=f"{prefix}_flange_geom", type="cylinder",
        size="0.035 0.018", euler="1.5707963268 0 0", rgba="0.25 0.25 0.28 1",
        contype="2", conaffinity="1",
    )
    ET.SubElement(flange, "site", name=f"{prefix}_eef", pos=f"0 {direction * 0.145:.6f} 0", size="0.008", rgba="0 0 0 0")
    ET.SubElement(flange, "site", name=f"{prefix}_wrist_ft", pos="0 0 0", size="0.012", rgba="0 0 0 0")
    ET.SubElement(
        flange, "camera", name=f"{side}_wrist",
        pos=f"0 {direction * 0.065:.6f} 0.095", mode="targetbody",
        target="workspace_target", fovy="70",
    )

    # Robotiq meshes use +z as the tool axis and +y across the jaws. Rotate the
    # source frame so +z follows the A3 terminal link and +y maps to A3 local +x.
    gripper = ET.SubElement(
        flange,
        "body",
        name=f"{prefix}_robotiq_2f85",
        xyaxes=f"0 0 {direction:.1f} 1 0 0",
    )
    ET.SubElement(
        gripper,
        "geom",
        name=f"{prefix}_2f85_base_visual",
        type="mesh",
        mesh="robotiq_arg2f_85_base_link",
        rgba="0.10 0.10 0.11 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(
        gripper,
        "geom",
        name=f"{prefix}_2f85_base_collision",
        type="cylinder",
        pos="0 0 0.045",
        size="0.043 0.045",
        rgba="0 0 0 0",
        contype="2",
        conaffinity="1",
        group="3",
    )

    jaw_specs = (
        ("inner", -0.0326011, 1.0, "0 0 0 1"),
        ("outer", 0.0326011, -1.0, "1 0 0 0"),
    )
    for finger_name, y, axis, assembly_quat in jaw_specs:
        moving = ET.SubElement(
            gripper,
            "body",
            name=f"{prefix}_finger_{finger_name}",
            pos=f"0 {y:.7f} 0.054904",
        )
        ET.SubElement(
            moving,
            "joint",
            name=f"{prefix}_finger_{finger_name}_joint",
            type="slide",
            axis=f"0 {axis:.1f} 0",
            range=f"0 {ROBOTIQ_2F85_JAW_TRAVEL_M}",
            damping="2",
            armature="0.005",
        )
        assembly = ET.SubElement(moving, "body", quat=assembly_quat)
        ET.SubElement(
            assembly,
            "geom",
            type="mesh",
            mesh="robotiq_arg2f_85_outer_knuckle_vis",
            rgba="0.78 0.81 0.92 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            assembly,
            "geom",
            type="mesh",
            pos="0 0.0315 -0.0041",
            mesh="robotiq_arg2f_85_outer_finger_vis",
            rgba="0.10 0.10 0.11 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        inner_finger = ET.SubElement(assembly, "body", pos="0 0.0376 0.043")
        ET.SubElement(
            inner_finger,
            "geom",
            type="mesh",
            mesh="robotiq_arg2f_85_inner_finger_vis",
            rgba="0.10 0.10 0.11 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            inner_finger,
            "geom",
            name=f"{prefix}_finger_{finger_name}_geom",
            type="box",
            pos="0 -0.0245203 0.03242",
            size="0.010 0.003175 0.01675",
            rgba="0.08 0.08 0.08 1",
            contype="2",
            conaffinity="1",
            friction="3.0 0.02 0.002",
            group="3",
        )
        ET.SubElement(
            inner_finger,
            "site",
            name=f"{prefix}_finger_{finger_name}_touch",
            type="box",
            pos="0 -0.0245203 0.03242",
            size="0.0105 0.0035 0.017",
            rgba="0 0 0 0",
        )

    for suffix, y, quat in (
        ("inner", -0.0127, "0 0 0 1"),
        ("outer", 0.0127, "1 0 0 0"),
    ):
        knuckle = ET.SubElement(
            gripper,
            "body",
            name=f"{prefix}_2f85_{suffix}_inner_knuckle",
            pos=f"0 {y:.4f} 0.06142",
            quat=quat,
        )
        ET.SubElement(
            knuckle,
            "geom",
            type="mesh",
            mesh="robotiq_arg2f_85_inner_knuckle_vis",
            rgba="0.10 0.10 0.11 1",
            contype="0",
            conaffinity="0",
            group="2",
        )


def _add_open_bin(
    parent: ET.Element,
    *,
    name: str,
    center: tuple[float, float],
    half_size: tuple[float, float],
    height: float,
    rgba: str,
    thickness: float = BIN_WALL_THICKNESS,
    floor_z: float = 0.753,
    wall_z: float | None = None,
) -> None:
    """Create a collision-enabled, open-topped box on the work surface."""
    x, y = center
    hx, hy = half_size
    if wall_z is None:
        wall_z = floor_z + height / 2
    collision_common = {
        "rgba": "0 0 0 0",
        "contype": "1",
        "conaffinity": "3",
        "friction": "1.2 0.02 0.001",
        "group": "3",
    }
    visual_common = {
        "rgba": rgba,
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    }
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_floor",
        type="box",
        pos=f"{x} {y} {floor_z}",
        size=f"{hx} {hy} {thickness / 2}",
        **collision_common,
    )
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_floor_visual",
        type="box",
        pos=f"{x} {y} {floor_z}",
        size=f"{hx} {hy} {thickness / 2}",
        **visual_common,
    )
    collision_walls = (
        # Extend adjacent walls by one thickness.  This makes the four corners
        # overlap instead of leaving diagonal gaps a small block can escape through.
        ("front", x + hx, y, thickness, hy + thickness),
        ("back", x - hx, y, thickness, hy + thickness),
        ("left", x, y + hy, hx + thickness, thickness),
        ("right", x, y - hy, hx + thickness, thickness),
    )
    for suffix, px, py, sx, sy in collision_walls:
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_{suffix}_wall",
            type="box",
            pos=f"{px} {py} {wall_z}",
            size=f"{sx} {sy} {height / 2}",
            **collision_common,
        )

    # Visible walls tile the outline without overlapping. Collision walls above
    # retain overlap at the corners, following robosuite's visual/collision split.
    visual_walls = (
        ("front", x + hx, y, thickness, hy - thickness),
        ("back", x - hx, y, thickness, hy - thickness),
        ("left", x, y + hy, hx + thickness, thickness),
        ("right", x, y - hy, hx + thickness, thickness),
    )
    for suffix, px, py, sx, sy in visual_walls:
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_{suffix}_wall_visual",
            type="box",
            pos=f"{px} {py} {wall_z}",
            size=f"{sx} {sy} {height / 2}",
            **visual_common,
        )


def _add_cookie_scene(root: ET.Element, world: ET.Element) -> None:
    # Source bin: on table
    _add_open_bin(
        world,
        name="source_bin",
        center=SOURCE_BIN_CENTER,
        half_size=SOURCE_BIN_HALF_SIZE,
        height=SOURCE_BIN_WALL_HEIGHT,
        rgba=BIN_RGBA,
        floor_z=0.753,
        wall_z=0.75 + SOURCE_BIN_WALL_HEIGHT / 2,
    )

    # Target bin: attached to right gripper with tilt
    r_robotiq = next(e for e in root.iter("body") if e.attrib.get("name") == "R_robotiq_2f85")
    target_bin = ET.SubElement(
        r_robotiq,
        "body",
        name="target_bin",
        pos=_vec(TARGET_BIN_ATTACH_POS),
        quat=_vec(TARGET_BIN_ATTACH_QUAT),
    )
    _add_open_bin(
        target_bin,
        name="target_bin",
        center=(0.0, 0.0),
        half_size=TARGET_BIN_HALF_SIZE,
        height=TARGET_BIN_WALL_HEIGHT,
        rgba=BIN_RGBA,
        floor_z=0.003,
        wall_z=TARGET_BIN_WALL_HEIGHT / 2,
    )

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for finger in ("inner", "outer"):
        ET.SubElement(
            contact,
            "exclude",
            body1="target_bin",
            body2=f"R_finger_{finger}",
        )

    for index, (x, y) in enumerate(COOKIE_SOURCE_POSITIONS):
        cookie = ET.SubElement(
            world, "body", name=f"cookie_{index}", pos=f"{x} {y} 0.781"
        )
        ET.SubElement(cookie, "freejoint", name=f"cookie_{index}_free")
        ET.SubElement(
            cookie,
            "geom",
            name=f"cookie_{index}_geom",
            type="box",
            size=_vec(COOKIE_HALF_SIZE),
            mass="0.035",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="3",
            friction="2.0 0.03 0.002",
            group="3",
        )
        visual_half_size = tuple(value - 0.0004 for value in COOKIE_HALF_SIZE)
        column, row = divmod(index, 6)
        cookie_rgba = (
            "1.0 0.78 0.20 1"
            if (column + row) % 2 == 0
            else "0.88 0.50 0.04 1"
        )
        ET.SubElement(
            cookie,
            "geom",
            name=f"cookie_{index}_visual",
            type="box",
            size=_vec(visual_half_size),
            mass="0.0001",
            rgba=cookie_rgba,
            contype="0",
            conaffinity="0",
            group="2",
        )


def build_model(config: SimConfig, *, scene: SceneName = "sandbox") -> ModelBundle:
    if scene not in ("sandbox", "cookie_transfer"):
        raise ValueError(f"unsupported scene: {scene}")
    urdf, source_joints = _source()
    links = {node.attrib["name"]: node for node in urdf.findall("link")}
    root = ET.Element("mujoco", model=f"a3_dual_arm_{scene}")
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="true", balanceinertia="true",
    )
    ET.SubElement(root, "option", timestep=f"{1 / config.physics_hz:.10g}", gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(root, "size", nconmax="400", njmax="1000")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(config.image_width), offheight=str(config.image_height))
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", limited="true", damping=str(config.joint_damping), armature=str(config.joint_armature))
    ET.SubElement(default, "geom", solref="0.01 1", solimp="0.9 0.95 0.001")

    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.75 0.82 0.9", rgb2="0.12 0.16 0.24", width="512", height="3072")
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
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", material="floor", contype="1", conaffinity="3")
    ET.SubElement(world, "geom", name="table_top", type="box", pos="0.15 0 0.725", size="0.55 0.75 0.025", material="table", contype="1", conaffinity="3", friction="1 0.01 0.001")
    ET.SubElement(world, "geom", name="stand_mast", type="box", pos="-0.35 0 0.49", size="0.040 0.040 0.49", rgba="0.08 0.09 0.10 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="stand_bracket", type="box", pos="-0.35 0 0.91", size="0.10 0.12 0.035", rgba="0.18 0.19 0.21 1", contype="0", conaffinity="0")
    target = ET.SubElement(world, "body", name="workspace_target", pos="0.28 0 0.80")
    ET.SubElement(target, "site", size="0.005", rgba="0 0 0 0")
    ET.SubElement(world, "camera", name="front", mode="targetbody", target="workspace_target", pos="1.35 -0.95 1.58", fovy="58")

    base = ET.SubElement(world, "body", name="base_link", pos="-0.35 0 0.98")
    _add_inertial(base, links["base_link"])
    ET.SubElement(base, "geom", name="base_visual", type="mesh", mesh="mesh_base_link", contype="0", conaffinity="0", group="2")
    ET.SubElement(base, "geom", name="base_collision", type="box", size="0.06 0.10 0.06", rgba="0.25 0.3 0.34 0.2", contype="2", conaffinity="1", group="3")

    child_map = {joint.parent: joint for joint in source_joints.values() if joint.name in LEFT_JOINTS}
    child_map.update({joint.parent: joint for joint in source_joints.values() if joint.name in RIGHT_JOINTS})

    def add_chain(parent: ET.Element, joint: JointSource, side: str) -> None:
        body = ET.SubElement(parent, "body", name=joint.child, pos=_vec(joint.origin))
        ET.SubElement(
            body, "joint", name=joint.name, type="hinge", axis=_vec(joint.axis),
            range=_vec(joint.limits), damping=str(config.joint_damping), armature=str(config.joint_armature),
        )
        _add_inertial(body, links[joint.child])
        next_joint = child_map.get(joint.child)
        direction = 1.0 if side == "left" else -1.0
        length = abs(next_joint.origin[1]) if next_joint is not None else 0.0815
        if length < 0.02:
            length = 0.0815
        _add_link_geometry(body, joint.child, direction, length)
        if next_joint is None:
            _add_gripper(body, side, direction)
        else:
            add_chain(body, next_joint, side)

    add_chain(base, source_joints[LEFT_JOINTS[0]], "left")
    add_chain(base, source_joints[RIGHT_JOINTS[0]], "right")

    if scene == "cookie_transfer":
        _add_cookie_scene(root, world)
    else:
        colors = ((0.68, -0.22, 0.79, "0.85 0.15 0.12 1"), (0.72, 0.0, 0.79, "0.12 0.55 0.9 1"), (0.68, 0.22, 0.79, "0.18 0.75 0.3 1"))
        for index, (x, y, z, rgba) in enumerate(colors):
            obj = ET.SubElement(world, "body", name=f"object_{index}", pos=f"{x} {y} {z}")
            ET.SubElement(obj, "freejoint", name=f"object_{index}_free")
            ET.SubElement(obj, "geom", name=f"object_{index}_geom", type="box", size="0.035 0.035 0.04", rgba=rgba, mass="0.10", contype="1", conaffinity="3", friction="1.5 0.02 0.001")

    actuators = ET.SubElement(root, "actuator")
    for joint in source_joints.values():
        kp = 600.0 if joint.effort >= 60 else (400.0 if joint.effort >= 30 else 200.0)
        ET.SubElement(
            actuators, "position", name=f"{joint.name}_position", joint=joint.name,
            kp=str(kp), ctrlrange=_vec(joint.limits), forcerange=f"{-joint.effort} {joint.effort}",
        )
    for prefix in ("L", "R"):
        for finger in ("inner", "outer"):
            name = f"{prefix}_finger_{finger}"
            ET.SubElement(actuators, "position", name=f"{name}_position", joint=f"{name}_joint", kp="120", ctrlrange=f"0 {ROBOTIQ_2F85_JAW_TRAVEL_M}", forcerange="-40 40")

    sensors = ET.SubElement(root, "sensor")
    for prefix in ("L", "R"):
        ET.SubElement(sensors, "force", name=f"{prefix}_wrist_force", site=f"{prefix}_wrist_ft")
        ET.SubElement(sensors, "torque", name=f"{prefix}_wrist_torque", site=f"{prefix}_wrist_ft")
        for finger in ("inner", "outer"):
            ET.SubElement(sensors, "touch", name=f"{prefix}_finger_{finger}_touch_sensor", site=f"{prefix}_finger_{finger}_touch")

    xml = ET.tostring(root, encoding="unicode")
    mesh_dir = asset_root() / "meshes"
    binary_assets = {
        path.name: path.read_bytes()
        for path in mesh_dir.iterdir()
        if path.suffix.lower() == ".stl"
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
        relative_meshes = os.path.relpath(asset_root() / "meshes", destination.parent)
        compiler.set("meshdir", Path(relative_meshes).as_posix())
    destination.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return destination
