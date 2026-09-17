from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.contracts import ARM_JOINTS
from a3_dual_arm_sim.model import (
    ROBOTIQ_2F85_JAW_TRAVEL_M,
    ROBOTIQ_2F85_MAX_OPENING_M,
    build_model,
    write_generated_xml,
)
from a3_dual_arm_sim.paths import asset_root


def test_source_assets_exclude_invalid_terminal_meshes() -> None:
    meshes = {path.name for path in (asset_root() / "meshes").glob("*.STL")}
    assert "L_LAST_S.STL" not in meshes
    assert "R_LAST_S.STL" not in meshes
    assert "base_link.STL" in meshes
    assert len(meshes) == 15


def test_model_compiles_with_all_source_arm_joints_and_sensors() -> None:
    bundle = build_model(load_config())
    assert tuple(joint.name for joint in bundle.source_joints) == ARM_JOINTS
    assert bundle.model.nu == 18
    assert bundle.model.nsensor == 8
    assert bundle.model.nsensordata == 16
    for joint in bundle.source_joints:
        joint_id = mujoco.mj_name2id(
            bundle.model, mujoco.mjtObj.mjOBJ_JOINT, joint.name
        )
        assert joint_id >= 0
        assert tuple(bundle.model.jnt_range[joint_id]) == joint.limits


def test_robotiq_2f85_visuals_and_85mm_parallel_jaw_contract() -> None:
    bundle = build_model(load_config())
    model = bundle.model
    data = mujoco.MjData(model)
    for side in ("L", "R"):
        joint_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_finger_{finger}_joint"
            )
            for finger in ("inner", "outer")
        ]
        for joint_id in joint_ids:
            assert tuple(model.jnt_range[joint_id]) == (0.0, ROBOTIQ_2F85_JAW_TRAVEL_M)
        assert (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_2f85_base_visual"
            )
            >= 0
        )
        mujoco.mj_forward(model, data)
        pad_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_finger_{finger}_geom"
            )
            for finger in ("inner", "outer")
        ]
        center_distance = float(
            np.linalg.norm(data.geom_xpos[pad_ids[0]] - data.geom_xpos[pad_ids[1]])
        )
        clear_opening = center_distance - 2 * 0.003175
        assert abs(clear_opening - ROBOTIQ_2F85_MAX_OPENING_M) < 2e-4


def test_generated_xml_is_portable_from_models_directory(tmp_path: Path) -> None:
    destination = tmp_path / "a3_generated.xml"
    write_generated_xml(destination, load_config())
    model = mujoco.MjModel.from_xml_path(str(destination))
    assert model.nu == 18


def test_cookie_scene_and_camera_values_come_from_config() -> None:
    base = load_config()
    config = replace(
        base,
        cameras=replace(base.cameras, front_fovy_deg=61.0),
        cookie_transfer=replace(base.cookie_transfer, cookie_mass_kg=0.041),
    )
    root = ET.fromstring(build_model(config, scene="cookie_transfer").xml)

    front = root.find("./worldbody/camera[@name='front']")
    cookie = root.find("./worldbody/body[@name='cookie_0']/geom[@name='cookie_0_geom']")

    assert front is not None and front.attrib["fovy"] == "61"
    assert cookie is not None and cookie.attrib["mass"] == "0.041"
    np.testing.assert_allclose(
        np.fromstring(cookie.attrib["size"], sep=" "), [0.025, 0.0095 / 3, 0.0125]
    )
    assert base.cameras.calibration_status == "prototype_estimate"
    assert base.cookie_transfer.calibration_status == "prototype_estimate"


def test_wrist_cameras_are_symmetric() -> None:
    config = load_config()
    root = ET.fromstring(build_model(config, scene="cookie_transfer").xml)
    left_cam = root.find(".//camera[@name='left_wrist']")
    right_cam = root.find(".//camera[@name='right_wrist']")
    assert left_cam is not None
    assert right_cam is not None
    assert left_cam.attrib["pos"] == "0 0.045 0.085"
    assert right_cam.attrib["pos"] == "0 -0.045 -0.085"
    left_target = root.find(".//body[@name='L_wrist_camera_target']")
    right_target = root.find(".//body[@name='R_wrist_camera_target']")
    assert left_target is not None
    assert right_target is not None
    assert left_target.attrib["pos"] == "0 0.245 0.085"
    assert right_target.attrib["pos"] == "0 -0.245 -0.085"
