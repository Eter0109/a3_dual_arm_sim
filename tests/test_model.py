from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

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
        cookie_transfer=replace(
            base.cookie_transfer, cookie_mass_kg=0.041, cookie_edge_bevel_m=0.002
        ),
    )
    root = ET.fromstring(build_model(config, scene="cookie_transfer").xml)

    front = root.find("./worldbody/camera[@name='front']")
    cookie = root.find("./worldbody/body[@name='cookie_0']/geom[@name='cookie_0_geom']")
    visual = root.find("./worldbody/body[@name='cookie_0']/geom[@name='cookie_0_visual']")
    collision_mesh = root.find("./asset/mesh[@name='cookie_collision_mesh']")
    visual_mesh = root.find("./asset/mesh[@name='cookie_visual_mesh']")

    assert front is not None and front.attrib["fovy"] == "61"
    assert cookie is not None and cookie.attrib["mass"] == "0.041"
    assert cookie.attrib["type"] == "mesh"
    assert cookie.attrib["mesh"] == "cookie_collision_mesh"
    assert visual is not None and visual.attrib["mesh"] == "cookie_visual_mesh"
    assert collision_mesh is not None and visual_mesh is not None
    collision_vertices = np.fromstring(collision_mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
    visual_vertices = np.fromstring(visual_mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
    np.testing.assert_allclose(
        np.max(np.abs(collision_vertices), axis=0), [0.025, 0.0095 / 3, 0.0125]
    )
    top_vertices = collision_vertices[
        np.isclose(collision_vertices[:, 2], base.cookie_transfer.cookie_half_size_m[2])
    ]
    bottom_vertices = collision_vertices[
        np.isclose(collision_vertices[:, 2], -base.cookie_transfer.cookie_half_size_m[2])
    ]
    np.testing.assert_allclose(np.max(np.abs(top_vertices[:, 1])), 0.0095 / 3 - 0.002)
    np.testing.assert_allclose(np.max(np.abs(bottom_vertices[:, 1])), 0.0095 / 3 - 0.002)
    np.testing.assert_allclose(
        np.max(np.abs(visual_vertices), axis=0), np.max(np.abs(collision_vertices), axis=0)
    )
    assert base.cameras.calibration_status == "prototype_estimate"
    assert base.cookie_transfer.calibration_status == "prototype_estimate"


def test_gripper_visual_pads_match_hidden_collision_pads() -> None:
    root = ET.fromstring(build_model(load_config(), scene="cookie_transfer").xml)
    for side in ("L", "R"):
        for finger in ("inner", "outer"):
            collision = root.find(
                f".//geom[@name='{side}_finger_{finger}_geom']"
            )
            visual = root.find(
                f".//geom[@name='{side}_finger_{finger}_pad_visual']"
            )
            assert collision is not None and visual is not None
            assert visual.attrib["pos"] == collision.attrib["pos"]
            assert visual.attrib["size"] == collision.attrib["size"]
            assert visual.attrib["group"] == "2"
            assert visual.attrib["contype"] == "0"
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


# ---------------------------------------------------------------------------
# visual styles
# ---------------------------------------------------------------------------


def test_the_default_visual_style_is_plain() -> None:
    """The rendered pixels are a dataset contract, so the default must not drift.

    Every collected image and every comparison run so far was produced with the
    original appearance.  Changing the default would silently restyle the scene and
    make those numbers unreproducible.
    """

    assert load_config().visual_style == "plain"


def test_an_unknown_visual_style_is_refused() -> None:
    with pytest.raises(ValueError, match="visual_style must be one of"):
        replace(load_config(), visual_style="fancy")


def test_detailed_style_changes_only_appearance_not_physics() -> None:
    """The claim the whole feature rests on, checked field by field.

    A style that quietly altered a mass, a geom size or a contact bit would change
    every rollout and every behaviour metric with it -- and the change would look
    like a model difference, not like a rendering difference.

    The comparison is scoped to the *collision* geoms rather than to every geom.  An
    earlier version asserted ``plain.ngeom == detailed.ngeom``, which forbade exactly
    what the style exists to provide: appearance-only geometry, such as the target
    box's slot lattice.  What must not move is anything that can be contacted, and
    for those the count, names, size, pose, contact bits and friction are compared
    exactly.
    """

    config = load_config("configs/cookie_same_column.yaml")
    plain = build_model(replace(config, visual_style="plain"), scene="cookie_transfer").model
    detailed = build_model(
        replace(config, visual_style="detailed"), scene="cookie_transfer"
    ).model

    def colliding(model):
        return np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))

    def collision_names(model, ids):
        return {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(index)) for index in ids
        }

    plain_ids, detailed_ids = colliding(plain), colliding(detailed)
    assert plain_ids.size == detailed_ids.size, "same number of touchable geoms"
    assert collision_names(plain, plain_ids) == collision_names(detailed, detailed_ids)
    for name in ("geom_size", "geom_pos", "geom_friction"):
        assert np.array_equal(
            getattr(plain, name)[plain_ids], getattr(detailed, name)[detailed_ids]
        ), name
    for name in ("geom_contype", "geom_conaffinity"):
        assert np.array_equal(
            getattr(plain, name)[plain_ids], getattr(detailed, name)[detailed_ids]
        ), name

    # Everything the style adds on top must be inert, or it is not appearance.
    added = np.setdiff1d(np.arange(detailed.ngeom), detailed_ids)
    assert added.size > 0, "the detailed style is supposed to add visual geometry"
    assert not np.any(detailed.geom_contype[added] | detailed.geom_conaffinity[added])

    # Structure and dynamics.  These index joints and degrees of freedom rather than
    # geoms, so they are compared whole: no visual geom can reach them.
    assert (plain.nbody, plain.njnt, plain.nq) == (
        detailed.nbody,
        detailed.njnt,
        detailed.nq,
    )
    for name in ("body_mass", "body_inertia", "jnt_range", "dof_damping"):
        assert np.array_equal(getattr(plain, name), getattr(detailed, name)), name

    # Appearance actually differs, or the feature does nothing.
    assert not np.array_equal(plain.vis.headlight.diffuse, detailed.vis.headlight.diffuse)


def test_the_slot_lattice_tiles_exactly_one_cookie_per_configured_slot() -> None:
    """The drawn grid must be the environment's own slot list, not a lookalike.

    The lattice exists so the phrase "the 2 by 5 box" refers to something.  If it
    were laid out from the bin's dimensions instead of from ``target_slots_local_m``
    the two could drift, and a policy would be aiming at rectangles that are not the
    ones the success criterion measures.
    """

    config = load_config("configs/cookie_same_column.yaml")
    scene = config.cookie_transfer
    root = ET.fromstring(
        build_model(replace(config, visual_style="detailed"), scene="cookie_transfer").xml
    )
    tiles = root.findall(".//geom[@name]")
    tiles = [
        tile for tile in tiles if (tile.attrib.get("name") or "").startswith("target_slot_r")
    ]
    assert len(tiles) == len(scene.target_slots_local_m) == 10

    half_x, half_y = scene.cookie_half_size_m[0], scene.cookie_half_size_m[1]
    centres = set()
    for tile in tiles:
        x, y, _ = (float(value) for value in tile.attrib["pos"].split())
        centres.add((round(x, 6), round(y, 6)))
        size_x, size_y, size_z = (float(value) for value in tile.attrib["size"].split())
        # One Cookie's footprint, and thin enough never to be a wall.
        assert (size_x, size_y) == (half_x, half_y)
        assert size_z <= 0.001
        # Inert: a tile that could be touched would change the task.
        assert tile.attrib["contype"] == "0"
        assert tile.attrib["conaffinity"] == "0"
    assert centres == {
        (round(slot[0], 6), round(slot[1], 6)) for slot in scene.target_slots_local_m
    }


def test_the_plain_style_still_has_no_slot_lattice() -> None:
    """The original renders are a record; they are not retrofitted with the aid."""

    config = load_config("configs/cookie_same_column.yaml")
    root = ET.fromstring(
        build_model(replace(config, visual_style="plain"), scene="cookie_transfer").xml
    )
    assert [
        geom
        for geom in root.findall(".//geom[@name]")
        if (geom.attrib.get("name") or "").startswith("target_slot_r")
    ] == []
    assert root.find(".//material[@name='slot_even']") is None


def test_the_cookie_palette_is_the_two_yellows_the_prompt_names() -> None:
    """The prompt says "yellow", so the render has to be yellow.

    A colour word in an instruction only helps if what the camera shows agrees with
    it.  This pins the two tones to the same hue family and requires them to differ
    enough in luminance to separate at the two pixels a Cookie row occupies.
    """

    config = load_config("configs/cookie_same_column.yaml")
    root = ET.fromstring(
        build_model(replace(config, visual_style="detailed"), scene="cookie_transfer").xml
    )
    tones = {}
    for name in ("cookie_light", "cookie_dark"):
        material = root.find(f".//material[@name='{name}']")
        assert material is not None, name
        tones[name] = [float(value) for value in material.attrib["rgba"].split()]

    for name, (red, green, blue, alpha) in tones.items():
        assert alpha == 1.0, name
        # Yellow: red and green both dominate blue by a wide margin.
        assert red > 0.6 and green > 0.5 and blue < 0.2, name
        assert red >= green > blue, name

    def luminance(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    light, dark = luminance(tones["cookie_light"]), luminance(tones["cookie_dark"])
    assert light > dark, "the two tones must be ordered, not merely different"
    assert (light + 0.05) / (dark + 0.05) > 1.4, "tones are too close to read apart"


def test_the_cookie_visual_geom_never_enters_the_collision_pipeline() -> None:
    """Both styles must keep the Cookie's visual mesh out of contact.

    The visual and collision envelopes are deliberately coincident so contact looks
    right, which means the only thing stopping the visual mesh from also colliding is
    its contype/conaffinity.  Losing those would double every Cookie's contact
    geometry and change the grasp.
    """

    config = load_config("configs/cookie_same_column.yaml")
    for style in ("plain", "detailed"):
        model = build_model(replace(config, visual_style=style), scene="cookie_transfer").model
        visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_0_visual")
        assert visual >= 0, style
        assert model.geom_contype[visual] == 0, style
        assert model.geom_conaffinity[visual] == 0, style
        # And the collision geoms are still there beside it.
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_0_geom") >= 0, style


def test_detailed_style_gives_the_cookies_their_own_materials() -> None:
    """Packed blades separate when their edges catch light differently.

    A material is what carries specular and shininess, so giving the Cookies their
    own is what makes a 2.5 mm gap read as a gap rather than as one striped block.
    """

    config = load_config("configs/cookie_same_column.yaml")
    root = ET.fromstring(
        build_model(replace(config, visual_style="detailed"), scene="cookie_transfer").xml
    )
    names = {material.attrib["name"] for material in root.findall(".//material")}
    assert {"cookie_light", "cookie_dark"} <= names
    # The Cookies' visual geoms reference them rather than an inline rgba.
    for index in (0, 1):
        geom = root.find(f".//geom[@name='cookie_{index}_visual']")
        assert geom is not None
        assert geom.attrib["material"] in ("cookie_light", "cookie_dark")
        assert "rgba" not in geom.attrib


def test_plain_style_leaves_the_cookie_geoms_on_inline_colour() -> None:
    config = load_config("configs/cookie_same_column.yaml")
    root = ET.fromstring(
        build_model(replace(config, visual_style="plain"), scene="cookie_transfer").xml
    )
    assert root.find(".//material[@name='cookie_light']") is None
    geom = root.find(".//geom[@name='cookie_0_visual']")
    assert geom is not None
    assert "rgba" in geom.attrib


def test_the_detailed_style_does_not_carry_ineffective_renderer_knobs() -> None:
    """Only settings that demonstrably change pixels belong here.

    ``shadowsize`` and ``shadowclip`` were tried and measured: 0.03 and 0.00 mean
    pixel difference at this scene's scale.  Carrying them would suggest a
    correctness they do not have, so their absence is asserted.
    """

    config = load_config("configs/cookie_same_column.yaml")
    root = ET.fromstring(
        build_model(replace(config, visual_style="detailed"), scene="cookie_transfer").xml
    )
    assert root.find(".//quality") is None
    assert root.find(".//map") is None
    assert root.find(".//headlight") is not None


def test_the_camera_positions_survive_the_detailed_style() -> None:
    config = load_config("configs/cookie_same_column.yaml")
    for style in ("plain", "detailed"):
        root = ET.fromstring(
            build_model(replace(config, visual_style=style), scene="cookie_transfer").xml
        )
        assert root.find(".//camera[@name='front']") is not None, style
        assert root.find(".//camera[@name='left_wrist']") is not None, style
        assert root.find(".//camera[@name='right_wrist']") is not None, style
