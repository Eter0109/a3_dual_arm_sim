from __future__ import annotations

import math
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


def _add_gripper(
    parent: ET.Element,
    side: str,
    direction: float,
    config: SimConfig,
    *,
    cookie_scene: bool = False,
) -> None:
    prefix = side[0].upper()
    flange = ET.SubElement(
        parent,
        "body",
        name=f"{prefix}_flange",
        pos=f"0 {direction * 0.0815:.6f} 0",
    )
    ET.SubElement(
        flange,
        "geom",
        name=f"{prefix}_flange_geom",
        type="cylinder",
        size="0.035 0.018",
        euler="1.5707963268 0 0",
        rgba="0.25 0.25 0.28 1",
        contype="2",
        conaffinity="1",
    )
    ET.SubElement(
        flange,
        "site",
        name=f"{prefix}_eef",
        pos=f"0 {direction * 0.145:.6f} 0",
        size="0.008",
        rgba="0 0 0 0",
    )
    ET.SubElement(
        flange,
        "site",
        name=f"{prefix}_wrist_ft",
        pos="0 0 0",
        size="0.012",
        rgba="0 0 0 0",
    )
    if side == "left":
        wrist_camera_position = config.cameras.left_wrist_position_m
        wrist_target_position = config.cameras.left_wrist_target_m
    else:
        wrist_camera_position = config.cameras.right_wrist_position_m
        wrist_target_position = config.cameras.right_wrist_target_m

    wrist_target = ET.SubElement(
        flange,
        "body",
        name=f"{prefix}_wrist_camera_target",
        pos=_vec(wrist_target_position),
    )
    ET.SubElement(wrist_target, "site", size="0.003", rgba="0 0 0 0")
    ET.SubElement(
        flange,
        "camera",
        name=f"{side}_wrist",
        pos=_vec(wrist_camera_position),
        mode="targetbody",
        target=f"{prefix}_wrist_camera_target",
        fovy=f"{config.cameras.wrist_fovy_deg:.10g}",
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
    pad_half_thickness = (
        config.cookie_transfer.left_finger_pad_half_thickness_m
        if cookie_scene and side == "left"
        else 0.003175
    )
    if pad_half_thickness <= 0:
        raise ValueError("left finger pad half-thickness must be positive")
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
            size=f"0.010 {pad_half_thickness:.8f} 0.01675",
            rgba="0.08 0.08 0.08 1",
            contype="2",
            conaffinity="1",
            friction="3.0 0.02 0.002",
            condim="4" if cookie_scene else "3",
            group="3",
        )
        # The physical pad belongs to hidden collision group 3.  Render an
        # identical, visual-only pad so contact does not look like a gap
        # between the 2F-85 fingers and a Cookie.
        ET.SubElement(
            inner_finger,
            "geom",
            name=f"{prefix}_finger_{finger_name}_pad_visual",
            type="box",
            pos="0 -0.0245203 0.03242",
            size=f"0.010 {pad_half_thickness:.8f} 0.01675",
            rgba="0.08 0.08 0.08 1",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            inner_finger,
            "site",
            name=f"{prefix}_finger_{finger_name}_touch",
            type="box",
            pos="0 -0.0245203 0.03242",
            size=f"0.0105 {pad_half_thickness + 0.000325:.8f} 0.017",
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
    thickness: float,
    friction: tuple[float, ...],
    floor_z: float = 0.753,
    wall_z: float | None = None,
    material: str | None = None,
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
        "friction": _vec(friction),
        "group": "3",
    }
    # ``rgba`` on a geom overrides the material's own colour, so only one of the
    # two is ever set: passing both would silently discard the material.
    visual_common = {
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    }
    if material is None:
        visual_common["rgba"] = rgba
    else:
        visual_common["material"] = material
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


def _add_slot_grid(
    parent: ET.Element,
    *,
    name: str,
    slots: tuple[tuple[float, ...], ...],
    slot_half_size: tuple[float, ...],
    surface_z: float,
    materials: tuple[str, ...],
    height_m: float = 0.0004,
    sink_m: float = 0.0005,
) -> None:
    """Tile a bin floor with one flat tile per placement slot.

    The scene asks for ten upright Cookies in a 2x5 box, but nothing in the render
    said so: at this camera the bin is a plain rectangle about 90 by 60 pixels, and
    the ten slots inside it are an empty region.  Neither a policy reading "the 2x5
    box" nor a person watching the video had anything to ground the phrase in.

    The tiles are drawn from the same slot centres the success criterion uses
    (``target_slots_local_m``) and are exactly one Cookie's footprint, so the drawn
    grid cannot disagree with what the environment checks.  The 6 mm space between
    the two columns is not invented: it is the real gap between Cookie columns, and
    leaving it bare is what makes the box read as two columns rather than one wide
    field.

    Tiles alternate tone by *row*, not by row and column.  This is a resolution
    decision, not a taste one.  The five rows occupy 31.7 mm, which the video's front
    camera renders as roughly ten pixels: each row is about two pixels tall, so any
    drawn line between rows -- even a 2 mm one -- is sub-pixel and disappears.  Whole
    tiles in alternating tones survive at that size because the signal is carried by
    a two-pixel block of colour rather than by an edge, and the eye counts five bands
    where a lattice would have shown nothing.  The column split does not need the
    same treatment: its gap is six millimetres across, and it is a vertical edge
    running the full height of the region.

    Visual only.  ``contype``/``conaffinity`` are zero, so a tile cannot be touched
    and Cookie placement is unaffected, and the 0.4 mm the tiles stand proud of the
    floor is below the 1 mm clearance the expert keeps.

    Each tile is sunk 0.5 mm into the floor rather than rested on it.  A face exactly
    coplanar with the floor z-fights, and the artefact reads as a bright outline
    around the whole lattice; sinking costs nothing because the tiles are visual, and
    it leaves only the intended 0.4 mm of relief.
    """

    if not slots:
        raise ValueError(f"{name} slot grid needs at least one slot")
    if len(materials) < 2:
        raise ValueError(f"{name} slot grid needs two tones to alternate between")
    half_x, half_y = slot_half_size[0], slot_half_size[1]
    if half_x <= 0 or half_y <= 0:
        raise ValueError(f"{name} slot grid needs a positive slot footprint")
    half_height = (height_m + sink_m) / 2
    z = surface_z + height_m - half_height
    rows = sorted({slot[1] for slot in slots})
    columns = sorted({slot[0] for slot in slots})
    for slot in slots:
        row = rows.index(slot[1])
        column = columns.index(slot[0])
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_r{row}c{column}",
            type="box",
            pos=f"{slot[0]} {slot[1]} {z}",
            size=f"{half_x} {half_y} {half_height}",
            material=materials[row % len(materials)],
            contype="0",
            conaffinity="0",
            group="2",
        )


def _add_beveled_cookie_mesh(
    assets: ET.Element,
    name: str,
    half_size: tuple[float, ...],
    bevel: float,
    bottom_bevel: float | None = None,
) -> None:
    """Extrude an octagonal y-z profile along x for real sloped contacts."""
    if len(half_size) != 3 or any(value <= 0 for value in half_size):
        raise ValueError("cookie half-size must contain three positive values")
    hx, hy, hz = half_size
    if bottom_bevel is None:
        bottom_bevel = bevel
    if not 0 < bevel < min(hy, hz):
        raise ValueError("cookie edge bevel must be positive and smaller than y/z half-size")
    if not 0 < bottom_bevel < min(hy, hz):
        raise ValueError("cookie bottom bevel must be positive and smaller than y/z half-size")

    # Keep the overall x width unchanged. Four long x-direction edges are
    # cut away so a descending finger meets a ramp rather than a square corner.
    profile = (
        (-hy + bevel, hz),
        (hy - bevel, hz),
        (hy, hz - bevel),
        (hy, -hz + bottom_bevel),
        (hy - bottom_bevel, -hz),
        (-hy + bottom_bevel, -hz),
        (-hy, -hz + bottom_bevel),
        (-hy, hz - bevel),
    )
    vertices = tuple((x, y, z) for x in (-hx, hx) for y, z in profile)
    faces: list[tuple[int, int, int]] = []
    for index in range(1, 7):
        faces.extend(((0, index, index + 1), (8, 8 + index + 1, 8 + index)))
    for index in range(8):
        next_index = (index + 1) % 8
        faces.extend(
            (
                (index, 8 + index, 8 + next_index),
                (index, 8 + next_index, next_index),
            )
        )
    ET.SubElement(
        assets,
        "mesh",
        name=name,
        vertex=_vec(tuple(value for vertex in vertices for value in vertex)),
        face=" ".join(str(value) for triangle in faces for value in triangle),
    )


def _add_detailed_visuals(visual: ET.Element) -> None:
    """Tune lighting for shape readability, without touching geometry.

    The scene carries two directional lights, but MuJoCo also adds a headlight that
    follows the camera with a default diffuse of 0.4 -- stronger than the scene's own
    fill light.  A camera-locked light casts no visible shadow and shades every face
    it sees equally, so the two real lights end up contributing mostly a flat wash and
    a packed row of Cookie blades reads as one solid block.  Dimming the headlight
    restores directional shading, which is what makes edges visible.

    Only rendering changes: no geom, joint, actuator, contact parameter or mass is
    touched, and the visual-only Cookie mesh already carries
    ``contype="0" conaffinity="0"``.

    Measured, so the ineffective knobs are not carried as decoration:

    - headlight ``0.4`` -> ``0.16``: mean pixel change **6.80**, the whole point.
    - ``quality/shadowsize`` ``4096`` -> ``8192``: mean change **0.03**.  The default
      is already finer than the scene needs, so raising it buys nothing.
    - ``map/shadowclip`` ``1.0`` -> ``3.0``: mean change **0.00**.  No effect here.
    - There is no soft-shadow option: MuJoCo 3.3.7's ``mjtRndFlag`` has no
      ``mjRND_SOFTSHADOW``, so the hard shadow edge on the table cannot be smoothed
      from here.  It is visible in the videos and is not fixable by configuration.
    """

    ET.SubElement(
        visual,
        "headlight",
        ambient="0.08 0.08 0.09",
        diffuse="0.16 0.16 0.18",
        specular="0.12 0.12 0.12",
    )


def _add_cookie_scene(root: ET.Element, world: ET.Element, config: SimConfig) -> None:
    scene = config.cookie_transfer
    assets = root.find("asset")
    if assets is None:
        raise RuntimeError("model assets must exist before adding the Cookie scene")
    # Keep visual and collision envelopes coincident.  A smaller visual mesh
    # made a correctly contacting Cookie look detached from the finger pads.
    visual_half_size = scene.cookie_half_size_m
    bottom_bevel = (
        scene.cookie_edge_bevel_m
        if scene.cookie_bottom_edge_bevel_m is None
        else scene.cookie_bottom_edge_bevel_m
    )
    _add_beveled_cookie_mesh(
        assets, "cookie_collision_mesh", scene.cookie_half_size_m,
        scene.cookie_edge_bevel_m, bottom_bevel,
    )
    _add_beveled_cookie_mesh(
        assets, "cookie_visual_mesh", visual_half_size,
        scene.cookie_edge_bevel_m, bottom_bevel,
    )
    if scene.cookie_collision_mode not in ("mesh", "compound"):
        raise ValueError("cookie_collision_mode must be mesh or compound")
    if scene.cookie_collision_mode == "compound":
        hx, hy, hz = scene.cookie_half_size_m
        bevel = scene.cookie_edge_bevel_m
        # Exact non-overlapping decomposition of the same octagonal solid.
        # Flat side contacts use box-box collision; ramps remain real meshes.
        for label, sign, edge in (("top", 1, bevel), ("bottom", -1, bottom_bevel)):
            vertices = [
                (x, y, sign * z)
                for x in (-hx, hx)
                for z, width in ((hz - edge, hy), (hz, hy - edge))
                for y in (-width, width)
            ]
            ET.SubElement(
                assets, "mesh", name=f"cookie_{label}_mesh",
                vertex=_vec(tuple(v for vertex in vertices for v in vertex)),
            )
    # Source bin: on table.
    #
    # A mocap body, not a plain geom under `world`, so the whole bin can be moved
    # between episodes without being knocked around during one.  Mocap bodies are
    # positioned kinematically (`data.mocap_pos` / `mocap_quat`), act as
    # infinite-mass obstacles for everything else, and are unaffected by contacts
    # -- exactly the behaviour a heavy steel bin should have.  A free joint would
    # have made it a dynamic body the Cookies could shove, and rebuilding the
    # model per episode to move static geoms would cost seconds a rollout.
    #
    # The body sits at z=0 and the bin's z values stay absolute, so the geometry
    # is unchanged; only the parent frame differs.  `center` therefore becomes
    # local (0, 0) while the world position is the body's `pos`.
    source_center = scene.source_bin_center_m
    source_bin = ET.SubElement(
        world,
        "body",
        name="source_bin",
        mocap="true",
        pos=_vec((source_center[0], source_center[1], 0.0)),
    )
    # With a material the bin picks up a highlight along its rim; without one it
    # falls back to the flat inline colour the original renders used.
    bin_material = "bin" if config.visual_style == "detailed" else None

    def add_slot_grid(parent: ET.Element, bin_name: str) -> None:
        """Draw the 2x5 placement lattice on a target-style bin.

        Only the target bins get it.  The source bin holds eighty Cookies in a
        4x20 layout and is covered by them, so a lattice there would be invisible
        under the stack and would add eighty more geoms to every render for nothing.
        The source's count is carried by the task prompt instead.
        """

        if config.visual_style != "detailed":
            return
        _add_slot_grid(
            parent,
            name=bin_name,
            slots=scene.target_slots_local_m,
            slot_half_size=scene.cookie_half_size_m,
            # The floor geom is centred on ``floor_z`` with half the wall thickness,
            # so its top surface -- where a Cookie actually rests -- is that much
            # higher.  The tiles sit on it rather than inside it.
            surface_z=scene.target_floor_z_m + scene.bin_wall_thickness_m / 2,
            materials=("slot_even", "slot_odd"),
        )

    _add_open_bin(
        source_bin,
        name="source_bin",
        center=(0.0, 0.0),
        half_size=scene.source_bin_half_size_m,
        height=scene.source_bin_wall_height_m,
        rgba=BIN_RGBA,
        thickness=scene.bin_wall_thickness_m,
        friction=scene.bin_friction,
        floor_z=scene.source_floor_z_m,
        wall_z=scene.source_wall_base_z_m + scene.source_bin_wall_height_m / 2,
        material=bin_material,
    )

    # A free rigid box can only follow the gripper through physical contact.
    target_bin = ET.SubElement(
        world,
        "body",
        name="target_bin",
        pos=_vec(scene.target_bin_world_position_m),
    )
    ET.SubElement(target_bin, "freejoint", name="target_bin_free")
    ET.SubElement(
        target_bin, "inertial", pos="0 0 0.01", mass="0.10", diaginertia="0.0002 0.0002 0.0003"
    )
    _add_open_bin(
        target_bin,
        name="target_bin",
        center=(0.0, 0.0),
        half_size=scene.target_bin_half_size_m,
        height=scene.target_bin_wall_height_m,
        rgba=BIN_RGBA,
        thickness=scene.bin_wall_thickness_m,
        friction=scene.bin_friction,
        floor_z=scene.target_floor_z_m,
        wall_z=scene.target_bin_wall_height_m / 2,
        material=bin_material,
    )
    add_slot_grid(target_bin, "target_slot")
    if scene.spare_target_bin_world_position_m is not None:
        spare_bin = ET.SubElement(
            world,
            "body",
            name="spare_target_bin",
            pos=_vec(scene.spare_target_bin_world_position_m),
        )
        ET.SubElement(spare_bin, "freejoint", name="spare_target_bin_free")
        ET.SubElement(
            spare_bin,
            "inertial",
            pos="0 0 0.01",
            mass="0.10",
            diaginertia="0.0002 0.0002 0.0003",
        )
        _add_open_bin(
            spare_bin,
            name="spare_target_bin",
            center=(0.0, 0.0),
            half_size=scene.target_bin_half_size_m,
            height=scene.target_bin_wall_height_m,
            rgba=BIN_RGBA,
            thickness=scene.bin_wall_thickness_m,
            friction=scene.bin_friction,
            floor_z=scene.target_floor_z_m,
            wall_z=scene.target_bin_wall_height_m / 2,
            material=bin_material,
        )
        add_slot_grid(spare_bin, "spare_slot")

    columns = sorted({p[0] for p in scene.cookie_source_positions_m})
    rows = sorted({p[1] for p in scene.cookie_source_positions_m})
    for index, (x, y) in enumerate(scene.cookie_source_positions_m):
        cookie = ET.SubElement(
            world,
            "body",
            name=f"cookie_{index}",
            pos=f"{x} {y} {scene.cookie_model_z_m}",
        )
        ET.SubElement(cookie, "freejoint", name=f"cookie_{index}_free")
        collision = ET.SubElement(
            cookie,
            "geom",
            name=f"cookie_{index}_geom",
            type="mesh",
            mesh="cookie_collision_mesh",
            mass=f"{scene.cookie_mass_kg:.10g}",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="3",
            friction=_vec(scene.cookie_friction),
            group="3",
            # Override the scene's softer default floor contact; otherwise the
            # mixed time constant still lets the narrow beveled base sink.
            priority="1",
            # The beveled underside has a narrower floor contact patch than the
            # old box. Its former 0.04 s soft contact let a light Cookie sink
            # deep into the bin floor, so use a firmer contact response.
            solref=_vec(scene.cookie_solref),
            solimp=_vec(scene.cookie_solimp),
            # condim=6 enables full tangential + torsional friction so the gripper
            # keeps lateral purchase while extracting a cookie from the stack.
            condim="6",
        )
        if scene.cookie_collision_mode == "compound":
            hx, hy, hz = scene.cookie_half_size_m
            top_bevel = scene.cookie_edge_bevel_m
            core_half_height = hz - (top_bevel + bottom_bevel) / 2
            core_center_z = (bottom_bevel - top_bevel) / 2
            core_volume = 8 * hx * hy * core_half_height
            cap_volumes = {
                "top": 2 * hx * top_bevel * (2 * hy - top_bevel),
                "bottom": 2 * hx * bottom_bevel * (2 * hy - bottom_bevel),
            }
            volume = core_volume + sum(cap_volumes.values())
            collision.attrib.pop("mesh")
            collision.set("type", "box")
            collision.set("pos", _vec((0, 0, core_center_z)))
            collision.set("size", _vec((hx, hy, core_half_height)))
            collision.set("mass", str(scene.cookie_mass_kg * core_volume / volume))
            for label in ("top", "bottom"):
                attributes = dict(collision.attrib)
                attributes.pop("size")
                attributes.pop("pos")
                attributes.update(
                    name=f"cookie_{index}_{label}_geom",
                    type="mesh", mesh=f"cookie_{label}_mesh",
                    mass=str(scene.cookie_mass_kg * cap_volumes[label] / volume),
                )
                ET.SubElement(cookie, "geom", **attributes)
        column, row = columns.index(x), rows.index(y)
        cookie_rgba = "1.0 0.78 0.20 1" if (column + row) % 2 == 0 else "0.88 0.50 0.04 1"
        # In the detailed style the colour comes from a material rather than an
        # inline rgba, because a material is what carries specular and shininess;
        # pairs of adjacent Cookies are given opposite materials so the alternation
        # survives at grazing angles where a flat colour washes out.
        visual_attributes: dict[str, str] = {
            "name": f"cookie_{index}_visual",
            "type": "mesh",
            "mesh": "cookie_visual_mesh",
            "mass": "0.0001",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        }
        if config.visual_style == "detailed":
            visual_attributes["material"] = (
                "cookie_light" if (column + row) % 2 == 0 else "cookie_dark"
            )
        else:
            visual_attributes["rgba"] = cookie_rgba
        ET.SubElement(cookie, "geom", **visual_attributes)


def _add_libero_props(root: ET.Element, world: ET.Element, config: SimConfig) -> None:
    """Append the configured LIBERO objects to the scene as free bodies.

    A LIBERO object directory holds one self-contained MJCF: an ``<asset>`` block naming
    a ``.msh`` mesh (and a texture) and a ``<worldbody>`` block holding one body tree.
    Two things have to be true before that fragment can live inside this model.

    **Names must be made unique.**  LIBERO's objects are authored as if each were the
    only model in the world -- the plate and the bowl both name a texture ``texture.png``
    and a body ``object`` -- and MuJoCo treats a duplicate name as a compile error rather
    than as a shadow.  So every ``name`` is prefixed, and the ``mesh=`` / ``material=`` /
    ``texture=`` references that point at the renamed assets are rewritten with it.  Both
    halves are needed: prefixing the declarations alone leaves the references dangling,
    which MuJoCo reports as a missing asset rather than as a rename that only went
    half-way.

    **File paths must be absolute.**  These assets live in the LIBERO cache, not in this
    repository, and `build_model` compiles from a string with a virtual asset dictionary
    built out of ``assets/a3/meshes``.  An absolute path is looked up on the real
    filesystem even when that dictionary is non-empty, which is what makes the *written*
    XML (``write_generated_xml``, used by the inspection tools) resolve its props too;
    registering the bytes under invented names instead would leave the file on disk
    unreadable by anything but this one code path.

    **The body needs a free joint, inserted first.**  MJCF is order-sensitive inside a
    body -- joints before geoms -- and appending the joint after the fragment's own
    geoms produces "joint found after geom" rather than a body that falls.
    """

    props = config.cookie_transfer.libero_props
    if not props:
        return
    assets = root.find("asset")
    if assets is None:
        raise RuntimeError("model assets must exist before adding a LIBERO prop")

    for prop in props:
        directory = Path(prop.directory).expanduser()
        candidates = sorted(directory.glob("*.xml"))
        if len(candidates) != 1:
            raise ValueError(
                f"{prop.name}: {directory} holds {len(candidates)} XML files "
                f"({[path.name for path in candidates]}); a LIBERO object directory holds "
                "exactly one, and picking one of several silently would decide which "
                "object this scene gets"
            )
        source = ET.parse(candidates[0]).getroot()
        source_assets = source.find("asset")
        source_world = source.find("worldbody")
        if source_assets is None or source_world is None:
            raise ValueError(f"{prop.name}: {candidates[0]} has no <asset> or <worldbody>")

        prefix = f"libero_{prop.name}"
        # Declarations first, so the reference pass below has a complete map to consult.
        renames = {}
        for element in source_assets:
            old = element.get("name")
            if old:
                renames[old] = f"{prefix}_{old}"
            path = element.get("file")
            if path:
                absolute = (directory / path).resolve()
                if not absolute.is_file():
                    raise FileNotFoundError(
                        f"{prop.name}: {candidates[0]} names {path}, which is not at "
                        f"{absolute}"
                    )
                element.set("file", str(absolute))

        for element in source.iter():
            name = element.get("name")
            if name:
                element.set("name", f"{prefix}_{name}")
            for attribute in ("mesh", "material", "texture"):
                target = element.get(attribute)
                if target in renames:
                    element.set(attribute, renames[target])

        wrappers = list(source_world)
        if len(wrappers) != 1:
            raise ValueError(
                f"{prop.name}: {candidates[0]} has {len(wrappers)} bodies under "
                "<worldbody>; the loader was written for LIBERO's single-wrapper layout"
            )
        body = wrappers[0]
        # LIBERO's outer wrapper is usually anonymous, and an unnamed body cannot be
        # looked up by name -- which turns a missing prop into `mj_name2id` returning -1
        # and every later query silently reading the *last* body in the model instead.
        # Naming it here is what makes the prop addressable from a diagnostic.
        body.set("name", f"{prefix}_root")
        body.set("pos", _vec(tuple(float(v) for v in prop.position_m)))
        half = math.radians(float(prop.yaw_deg)) / 2.0
        body.set("quat", _vec((math.cos(half), 0.0, 0.0, math.sin(half))))
        body.insert(0, ET.Element("freejoint", name=f"{prefix}_free"))
        world.append(body)
        for element in list(source_assets):
            assets.append(element)


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
    if config.visual_style == "detailed":
        _add_detailed_visuals(visual)
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
    if config.visual_style == "detailed":
        # A dedicated material per role instead of the renderer's defaults, which
        # give every geom the same specular and shininess and therefore the same
        # flat highlight.  Blades packed 0.4 mm apart separate when their edges catch
        # light at different angles, so the specification that matters here is a
        # narrow, bright highlight rather than a broad matte one.
        #
        # Both Cookie tones are yellow, as the task prompt names them, and differ in
        # luminance by about 1.5:1.  Alternating two tones of one hue is what makes
        # the packing countable: twenty rows at a 6.3 mm pitch render to roughly two
        # pixels each, and a *colour* difference between neighbours survives that
        # where any drawn edge does not.  A hue difference was rejected for the same
        # reason -- at two pixels a viewer reads tone, not hue.
        ET.SubElement(
            assets,
            "material",
            name="cookie_light",
            rgba="1.00 0.87 0.12 1",
            specular="0.55",
            shininess="0.75",
            reflectance="0.02",
        )
        ET.SubElement(
            assets,
            "material",
            name="cookie_dark",
            rgba="0.76 0.56 0.00 1",
            specular="0.55",
            shininess="0.75",
            reflectance="0.02",
        )
        # The bin was given a material before any geom referenced it; it is now
        # actually bound through ``_add_open_bin``.  Bare metal: a light body makes
        # the dark slot tiles read as a recess rather than as paint on the floor.
        ET.SubElement(
            assets,
            "material",
            name="bin",
            rgba="0.72 0.74 0.78 1",
            specular="0.35",
            shininess="0.45",
        )
        # Slot tones are neutral and both far darker than the bin, so a filled slot
        # changes the pixel by a large step: yellow on near-black is a much easier
        # thing to read a progress count from than yellow on light grey.
        ET.SubElement(
            assets,
            "material",
            name="slot_even",
            rgba="0.36 0.41 0.50 1",
            specular="0.10",
            shininess="0.20",
        )
        ET.SubElement(
            assets,
            "material",
            name="slot_odd",
            rgba="0.13 0.16 0.21 1",
            specular="0.10",
            shininess="0.20",
        )
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
        # After the Cookie scene, because the props are passive scenery laid on top of it
        # and nothing in the scene builder may depend on them.
        _add_libero_props(root, world, config)
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
        # Velocity feedback on the arm position servos.  Without it a position
        # command that moves faster than the joint can track is answered by
        # overshoot and a sustained ring, because nothing but the joint's own
        # damping removes the energy.  A fast Cartesian servo *is* such a
        # command, so the ringing is what caps the rate rather than the
        # mechanism: measured on the approach transit, the plan reaches its goal
        # in six steps and the tool then swings between 1 mm and 15 mm away
        # indefinitely while the servo slows down to hide it.  ``kp / 14`` is
        # the ratio MuJoCo's own position actuator documents as critically
        # damped, and ``arm_actuator_damping`` scales it because the two-box
        # relay's right-arm controllers are tuned for the undamped response.
        kv = (
            40.0 if joint.effort >= 60 else (30.0 if joint.effort >= 30 else 15.0)
        ) * config.arm_actuator_damping
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
        relative_meshes = os.path.relpath(asset_root() / "meshes", destination.parent)
        compiler.set("meshdir", Path(relative_meshes).as_posix())
    destination.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return destination
