from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np

from a3_dual_arm_sim.config import SimConfig

from .urdf import _vec

ROBOTIQ_2F85_MAX_OPENING_M = 0.085
ROBOTIQ_2F85_JAW_TRAVEL_M = ROBOTIQ_2F85_MAX_OPENING_M / 2


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
    # Resolve the look-at direction once in the flange frame. targetbody would
    # continually reorient the camera instead of modelling a rigid mount.
    camera_z = np.asarray(wrist_camera_position) - np.asarray(wrist_target_position)
    if np.linalg.norm(camera_z) < 1e-9:
        raise ValueError("wrist camera position and target must differ")
    camera_z = camera_z / np.linalg.norm(camera_z)
    camera_x = np.array([direction, 0.0, 0.0])
    if abs(camera_x @ camera_z) > 0.99:
        camera_x = np.array([0.0, 0.0, 1.0])
    camera_x -= (camera_x @ camera_z) * camera_z
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    ET.SubElement(
        flange,
        "camera",
        name=f"{side}_wrist",
        pos=_vec(wrist_camera_position),
        mode="fixed",
        xyaxes=_vec(tuple(camera_x) + tuple(camera_y)),
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
