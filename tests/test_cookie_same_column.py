from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.expert import CookiePhase

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cookie_same_column.yaml"


def test_batch_layout_has_uniform_small_gaps_not_finger_lanes():
    config = load_config(CONFIG)
    positions = np.asarray(config.cookie_transfer.cookie_source_positions_m)
    for column in positions.reshape(4, 20, 2):
        gaps = np.diff(column[:, 1]) - 2 * config.cookie_transfer.cookie_half_size_m[1]
        np.testing.assert_allclose(gaps, 0.0025, atol=1e-8)
        assert np.all(gaps < 0.00635)
    assert config.cookie_transfer.left_finger_pad_half_thickness_m == 0.0005
    assert config.cookie_transfer.cookie_edge_bevel_m == 0.0025
    assert config.cookie_transfer.cookie_bottom_edge_bevel_m == 0.001
    assert len(positions) == 80


def test_batch_thin_pad_is_left_only_and_configurable():
    from dataclasses import replace

    from a3_dual_arm_sim.model import build_model

    config = load_config(CONFIG)
    batch_model = build_model(config, scene="cookie_transfer").model
    assert batch_model.geom("L_finger_inner_geom").size[1] == pytest.approx(0.0005)
    assert batch_model.geom("R_finger_inner_geom").size[1] == pytest.approx(0.003175)

    default_config = replace(
        config,
        cookie_transfer=replace(
            config.cookie_transfer, left_finger_pad_half_thickness_m=0.003175
        ),
    )
    default_model = build_model(default_config, scene="cookie_transfer").model
    assert default_model.geom("L_finger_inner_geom").size[1] == pytest.approx(0.003175)


def test_next_batch_stays_in_same_column_despite_tilt():
    source = load_config(CONFIG).cookie_transfer.cookie_source_positions_m
    expert = object.__new__(A3SameColumnBatchExpert)
    expert.env = SimpleNamespace(
        SOURCE_POSITIONS=source, _cookie_bodies=tuple(range(80)),
        privileged_cookie_in_source=lambda i: True,
    )
    expert.data = SimpleNamespace(xmat=np.tile(np.eye(3).ravel(), (80, 1)))
    expert.completed_cookie_indices = []
    assert next(expert._candidate_batches()) == [0, 1, 2, 3, 4]
    expert.completed_cookie_indices = list(range(5))
    assert next(expert._candidate_batches()) == [5, 6, 7, 8, 9]
    expert.data.xmat[5, 8] = 0.0
    assert next(expert._candidate_batches()) == [5, 6, 7, 8, 9]
    expert.env.privileged_cookie_in_source = lambda i: i != 5
    assert next(expert._candidate_batches(), None) is None


def test_insertion_force_limit_ignores_one_spike_but_stops_sustained_jam():
    expert = object.__new__(A3SameColumnBatchExpert)
    expert._jam_count = 0
    spike = expert.MAX_INSERTION_FORCE_N + 1
    assert not expert._insertion_jammed([spike, 0])
    assert not expert._insertion_jammed([1, 1])
    assert not expert._insertion_jammed([spike, 0])
    assert not expert._insertion_jammed([spike, 0])
    assert expert._insertion_jammed([spike, 0])


def test_compound_collision_preserves_mass_and_inertia():
    from dataclasses import replace

    from a3_dual_arm_sim.model import build_model

    config = load_config(CONFIG)
    compound = build_model(config, scene="cookie_transfer").model
    mesh_config = replace(
        config, cookie_transfer=replace(config.cookie_transfer, cookie_collision_mode="mesh")
    )
    mesh = build_model(mesh_config, scene="cookie_transfer").model
    a, b = compound.body("cookie_0").id, mesh.body("cookie_0").id
    np.testing.assert_allclose(compound.body_mass[a], mesh.body_mass[b], rtol=1e-6)
    np.testing.assert_allclose(
        np.sort(compound.body_inertia[a]), np.sort(mesh.body_inertia[b]), rtol=1e-5
    )
    np.testing.assert_allclose(compound.body_ipos[a], mesh.body_ipos[b], atol=1e-9)


def test_batch_rejects_unreachable_table_box_before_grasping():
    from dataclasses import replace

    config = load_config(CONFIG)
    config = replace(
        config, cookie_transfer=replace(
            config.cookie_transfer, target_bin_world_position_m=(0.135, -0.015, 0.753)
        )
    )
    env = A3CookieTransferEnv(config, render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        with pytest.raises(RuntimeError, match="unreachable"):
            A3SameColumnBatchExpert(env).reset()
    finally:
        env.close()


def test_batch_preclose_happens_above_cookies_before_descent():
    env = A3CookieTransferEnv(CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        expert._select_batch()
        approach = None
        for _ in range(40):
            approach = expert.act()
            env.step(approach)
            if expert.phase is CookiePhase.PRE_CLOSE:
                break
        assert expert.phase is CookiePhase.PRE_CLOSE
        assert approach[6] == 1.0
        for _ in range(24):
            env.step(expert.act())
            if expert.phase is CookiePhase.DESCEND:
                break
        assert expert.phase is CookiePhase.DESCEND
        assert expert._opening < 1.0
    finally:
        env.close()


def test_batch_home_is_mirrored_and_near_first_approach():
    env = A3CookieTransferEnv(CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        np.testing.assert_allclose(env.DEPLOYMENT_HOME[:7], -env.DEPLOYMENT_HOME[8:15])
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        expert._select_batch()
        assert np.max(np.abs(expert._approach_q - env.DEPLOYMENT_HOME[:7])) < np.deg2rad(15)
    finally:
        env.close()


@pytest.mark.parametrize("missing", [None, 0, 2, 5])
def test_grasp_requires_whole_five_cookie_contact_chain(monkeypatch, missing):
    expert = object.__new__(A3SameColumnBatchExpert)
    expert.env = SimpleNamespace(_left_finger_geoms=(0, 6), _cookie_geoms=(1, 2, 3, 4, 5))
    expert.batch_indices = list(range(5))
    expert.model = None
    expert.max_pad_force = 0.0
    contacts = [SimpleNamespace(geom1=i, geom2=i + 1) for i in range(6) if i != missing]
    expert.data = SimpleNamespace(ncon=len(contacts), contact=contacts)

    def force(model, data, cid, wrench):
        wrench[0] = 0.5

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    connected, _ = expert._contact_chain()
    assert connected == (missing is None)


def test_contact_chain_includes_bevel_cap_contacts(monkeypatch):
    expert = object.__new__(A3SameColumnBatchExpert)
    expert.env = SimpleNamespace(
        _left_finger_geoms=(0, 6), _cookie_geoms=(1, 2, 3, 4, 5),
        _cookie_collision_geoms=tuple(frozenset((i, i + 6)) for i in range(1, 6)),
    )
    expert.batch_indices, expert.model, expert.max_pad_force = list(range(5)), None, 0.0
    nodes = (0, 7, 8, 9, 10, 11, 6)
    contacts = [SimpleNamespace(geom1=a, geom2=b) for a, b in pairwise(nodes)]
    expert.data = SimpleNamespace(ncon=len(contacts), contact=contacts)

    def force(model, data, cid, wrench):
        wrench[0] = 0.5

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    assert expert._contact_chain()[0]


def test_batch_success_rejects_held_or_overhanging_cookie(monkeypatch):
    env = A3CookieTransferEnv(
        CONFIG,
        render_cameras=False,
        task_config=CookieTransferTaskConfig(require_exact_slots=False, require_released=True),
    )
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        right_home = env.last_applied_action[8:15].copy()
        assert expert.box_support is None
        action = expert.act()
        assert len(action) == 14
        np.testing.assert_allclose(action[7:13], 0.0)
        assert action[13] == 1.0
        env.step(action)
        np.testing.assert_allclose(env.last_applied_action[8:15], right_home, atol=1e-5)
        p = env.privileged_target_slot_world(0, env._target_cookie_center_z)
        env.set_cookie_pose(0, tuple(p))
        assert env.privileged_cookie_in_target(0)
        monkeypatch.setattr(env, "privileged_left_finger_contacts", lambda index: (True, False))
        assert not env.privileged_cookie_in_target(0)
        monkeypatch.setattr(env, "privileged_left_finger_contacts", lambda index: (False, False))
        p[0] += 0.080
        env.set_cookie_pose(0, tuple(p))
        assert not env.privileged_cookie_in_target(0)
    finally:
        env.close()


def test_released_count_success_requires_ten_and_resets_when_one_leaves():
    env = A3CookieTransferEnv(
        CONFIG, render_cameras=False,
        task_config=CookieTransferTaskConfig(
            require_exact_slots=False, require_released=True, terminate_on_success=False,
            success_hold_steps=2,
        ),
    )
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        for i in range(10):
            env.set_cookie_pose(
                i, tuple(env.privileged_target_slot_world(i, env._target_cookie_center_z))
            )
        for _ in range(10):
            _, _, _, _, info = env.step(env.last_applied_action.copy())
        assert info["cookies_in_target"] == 10
        assert info["cookies_in_source"] == 70
        assert info["success"]
        env.set_cookie_pose(0, (0.4, -0.15, 0.9))
        _, _, _, _, info = env.step(env.last_applied_action.copy())
        assert not info["success"]
        assert info["cookies_in_target"] == 9
    finally:
        env.close()


@pytest.mark.parametrize("physics_hz", [1000, 2000])
def test_beveled_five_stack_resists_gripper_compression(physics_hz):
    """Isolated contact regression: gripping must not squeeze five meshes into one."""
    from copy import deepcopy
    from xml.etree import ElementTree as ET

    from a3_dual_arm_sim.model import build_model

    config = load_config(CONFIG)
    scene = config.cookie_transfer
    actual = ET.fromstring(build_model(config, scene="cookie_transfer").xml)
    root = ET.Element("mujoco")
    option = ET.SubElement(
        root,
        "option",
        timestep=str(1 / physics_hz),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        cone="elliptic",
        impratio="10",
        noslip_iterations="5",
    )
    ET.SubElement(option, "flag", multiccd="enable", nativeccd="enable")
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "geom",
        solref=" ".join(map(str, scene.cookie_solref)),
        solimp=" ".join(map(str, scene.cookie_solimp)),
        friction=" ".join(map(str, scene.cookie_friction)),
        condim="6",
    )
    asset = ET.SubElement(root, "asset")
    for mesh in actual.findall("./asset/mesh"):
        if mesh.get("name", "").startswith("cookie_"):
            asset.append(deepcopy(mesh))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="box", size="0.1 0.1 0.003", pos="0 0 -0.0155")
    actuator = ET.SubElement(root, "actuator")
    pitch = 2 * scene.cookie_half_size_m[1] + 0.0001
    for i in range(5):
        body = ET.SubElement(world, "body", name=f"c{i}", pos=f"0 {(i - 2) * pitch} 0")
        ET.SubElement(body, "freejoint")
        for geom in actual.findall("./worldbody/body[@name='cookie_0']/geom"):
            if geom.get("contype") == "1":
                collision = deepcopy(geom)
                collision.attrib.pop("name")
                body.append(collision)
    for i, sign in enumerate((-1, 1)):
        y = sign * (2 * pitch + scene.cookie_half_size_m[1] + 0.003175)
        body = ET.SubElement(world, "body", pos=f"0 {y} 0.010")
        ET.SubElement(
            body,
            "joint",
            name=f"jaw{i}",
            type="slide",
            axis=f"0 {-sign} 0",
            damping="2",
            armature="0.005",
        )
        ET.SubElement(
            body,
            "joint",
            name=f"lift{i}",
            type="slide",
            axis="0 0 1",
            damping="2",
            armature="0.005",
        )
        ET.SubElement(body, "geom", type="box", size="0.01 0.003175 0.01675", mass="0.02")
        ET.SubElement(actuator, "position", joint=f"jaw{i}", kp="600")
        ET.SubElement(actuator, "position", joint=f"lift{i}", kp="2000")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.ctrl[::2] = 0.003
    for _ in range(physics_hz):
        mujoco.mj_step(model, data)
    centers = np.array([data.body(f"c{i}").xpos for i in range(5)])
    assert np.all(np.diff(centers[:, 1]) > 0.0060), centers
    assert min(c.dist for c in data.contact) > -0.0005
    for step in range(physics_hz):
        data.ctrl[1::2] = min(0.080, step * 0.08 / physics_hz)
        mujoco.mj_step(model, data)
    lifted = [data.body(f"c{i}").xpos[2] for i in range(5)]
    assert min(lifted) > 0.06, lifted
