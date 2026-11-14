import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_2x10 import A3Cookie2x10Env
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_source_columns_have_clearance_and_exported_model_matches_config():
    config = load_config(ROOT / "configs/cookie_2x10.yaml").cookie_transfer
    original = load_config(ROOT / "configs/cookie_batch.yaml").cookie_transfer
    positions = np.asarray(config.cookie_source_positions_m)
    old_positions = np.asarray(original.cookie_source_positions_m)
    columns = np.unique(positions[:, 0])
    np.testing.assert_allclose(np.diff(columns), 0.0544, atol=1e-10)
    np.testing.assert_allclose(positions[:, 1], old_positions[:, 1], atol=1e-10)
    assert len(positions) == 80
    assert np.all(np.diff(columns) - 2 * config.cookie_half_size_m[0] >= 0.0044 - 1e-10)
    inner_x = config.source_bin_half_size_m[0] - config.bin_wall_thickness_m
    edge_x = max(abs(columns - config.source_bin_center_m[0])) + config.cookie_half_size_m[0]
    assert inner_x - edge_x >= 0.006 - 1e-10

    root = ET.parse(ROOT / "models/a3_cookie_transfer_2x10.xml").getroot()
    bin_body = root.find("./worldbody/body[@name='source_bin']")
    assert bin_body is not None
    np.testing.assert_allclose(
        [float(x) for x in bin_body.get("pos").split()][:2], config.source_bin_center_m
    )
    for index, expected in enumerate(positions):
        body = root.find(f"./worldbody/body[@name='cookie_{index}']")
        assert body is not None
        np.testing.assert_allclose([float(x) for x in body.get("pos").split()][:2], expected)


@pytest.mark.parametrize("profile", ["basic", "medium", "advanced"])
def test_configured_perturbation_preserves_cross_column_clearance(profile):
    config = load_config(ROOT / "configs/cookie_2x10.yaml").cookie_transfer
    settings, _ = load_cookie_2x10_settings()
    noise = settings["profiles"][profile]
    yaw = noise["cookie_yaw_noise_rad"]
    hx, hy = config.cookie_half_size_m[:2]
    # Maximum X projection within the configured yaw range, plus worst-case
    # independent translations in the two neighboring columns.
    max_extent_yaw = min(yaw, np.arctan2(hy, hx))
    extent = hx * np.cos(max_extent_yaw) + hy * np.sin(max_extent_yaw)
    columns = np.unique(np.asarray(config.cookie_source_positions_m)[:, 0])
    worst_gap = np.min(np.diff(columns)) - 2 * (extent + noise["cookie_noise_m"])
    assert worst_gap > A3Cookie2x10Env.MIN_SOURCE_COLUMN_CLEARANCE_M


def test_random_sampling_rejects_cross_column_interleaving_even_when_y_is_separate(monkeypatch):
    env = object.__new__(A3Cookie2x10Env)
    env.COOKIE_HALF_SIZE = np.array([0.025, 0.0031666666667, 0.0125])
    env.SOURCE_POSITIONS = ((0.0, 0.0), (0.054, 0.1))
    env._source_initial_rectangles = [(np.array([0.0, 0.0]), np.eye(2), 0.0)]
    env._source_pose_resamples = 0
    candidates = iter([
        ((0.051, 0.1, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.054, 0.1, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ])
    monkeypatch.setattr(A3CookieTransferEnv, "_sample_source_cookie_pose", lambda *args: next(candidates))
    position, _ = env._sample_source_cookie_pose(1, (0.054, 0.1, 0.0), True)
    assert position[0] == 0.054
    assert env._source_pose_resamples == 1


def test_fixed_invalid_layout_fails_instead_of_silently_accepting_interleaved_ends(monkeypatch):
    env = object.__new__(A3Cookie2x10Env)
    env.COOKIE_HALF_SIZE = np.array([0.025, 0.0031666666667, 0.0125])
    env.SOURCE_POSITIONS = ((0.0, 0.0), (0.051, 0.1))
    env._source_initial_rectangles = [(np.array([0.0, 0.0]), np.eye(2), 0.0)]
    env._source_pose_resamples = 0
    monkeypatch.setattr(
        A3CookieTransferEnv, "_sample_source_cookie_pose",
        lambda *args: ((0.051, 0.1, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )
    with pytest.raises(RuntimeError, match="cross-column clearance"):
        env._sample_source_cookie_pose(1, (0.051, 0.1, 0.0), False)


def test_insufficient_clearance_after_settling_fails_reset(monkeypatch):
    original_reset = A3CookieTransferEnv.reset

    def move_cookie_after_settling(env, *args, **kwargs):
        observation, info = original_reset(env, *args, **kwargs)
        address = int(env.model.jnt_qposadr[env._cookie_joints[20]])
        env.data.qpos[address] -= 0.004
        mujoco.mj_forward(env.model, env.data)
        return observation, info

    monkeypatch.setattr(A3CookieTransferEnv, "reset", move_cookie_after_settling)
    env = A3Cookie2x10Env(render_cameras=False)
    try:
        with pytest.raises(RuntimeError, match="settled source columns have insufficient clearance"):
            env.reset(seed=48868, options={"randomize_cookies": False})
    finally:
        env.close()


@pytest.mark.parametrize("profile,seed", [("basic", 48868), ("medium", 48869), ("advanced", 48870)])
def test_randomized_scene_records_actual_clearance_and_contains_all_cookies(profile, seed):
    benchmark = make_cookie_benchmark(task="2x10", profile=profile, source_column="1", first_grasp=5)
    env = benchmark.create_env(render_cameras=False)
    try:
        _, info = env.reset(seed=seed, options={
            "randomize_cookies": True,
            "randomize_boxes": True,
            "source_bin_noise_m": benchmark.source_bin_noise_m,
            "target_bin_noise_m": benchmark.target_bin_noise_m,
            "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
            "randomization_profile": profile,
            "randomization_settings": benchmark.randomization_settings,
        })
        assert info["cookies_in_source"] == 80
        assert info["source_initially_filled"]
        actual = env.randomization_metadata["source_column_clearances_m"]
        assert len(actual) == 3
        assert min(actual) >= env.MIN_SOURCE_COLUMN_CLEARANCE_M
        centers = env.cookie_positions
        rotations = env.data.xmat[list(env._cookie_bodies)].reshape(-1, 3, 3)
        extents = (np.abs(rotations) @ env.COOKIE_HALF_SIZE)[:, 0]
        for col in range(3):
            left = slice(col * 20, (col + 1) * 20)
            right = slice((col + 1) * 20, (col + 2) * 20)
            measured = np.min(centers[right, 0] - extents[right]) - np.max(centers[left, 0] + extents[left])
            assert actual[col] == pytest.approx(measured, abs=1e-12)
    finally:
        env.close()


def test_clearance_sampling_is_repeatable_for_the_same_seed():
    benchmark = make_cookie_benchmark(task="2x10", profile="basic", source_column="1", first_grasp=5)
    env = benchmark.create_env(render_cameras=False)
    try:
        options = {"randomize_cookies": True, "randomize_boxes": True,
                   "source_bin_noise_m": benchmark.source_bin_noise_m}
        env.reset(seed=48868, options=options)
        positions = env.cookie_positions.copy()
        quaternions = env.data.xquat[list(env._cookie_bodies)].copy()
        clearances = env.randomization_metadata["source_column_clearances_m"]
        env.reset(seed=48868, options=options)
        np.testing.assert_allclose(env.cookie_positions, positions, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            env.data.xquat[list(env._cookie_bodies)], quaternions, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            env.randomization_metadata["source_column_clearances_m"], clearances, rtol=0, atol=1e-14
        )
    finally:
        env.close()


def test_seed_48868_original_expert_first_grasp_does_not_carry_neighbor():
    benchmark = make_cookie_benchmark(
        task="2x10", first_grasp=5, profile="basic", source_column="1", max_steps=160
    )
    env = benchmark.create_env(render_cameras=False)
    try:
        result = benchmark.run_episode("variable_grasp", 48868, env=env)
        reports = result.randomization["grasp_reports"]
        assert reports and reports[0]["cookies"] == [0, 1, 2, 3, 4]
        assert reports[0]["lifted"] == 5
        assert result.cookies_in_source == 75
        initial = np.asarray(result.randomization["cookie_positions"])
        assert np.max(env.cookie_positions[20:, 2] - initial[20:, 2]) < 0.001
    finally:
        env.close()
