from itertools import combinations, product

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


def _benchmark(task="2x5", profile="basic", **scene):
    settings = load_randomization_config()
    settings["scene"] = scene
    return make_cookie_benchmark(
        task, profile=profile, randomization_settings=settings, success_hold_steps=2
    )


@pytest.mark.parametrize("task", ["2x5", "2x10"])
def test_four_box_placement_poses_fit_the_advanced_randomization_corners(task):
    benchmark = _benchmark(task, "advanced", target_box_count=4)
    env = benchmark.create_env()
    primary = env._target_bin_body
    try:
        env.reset(seed=4, options={"randomize_cookies": False, "randomize_boxes": False})
        factory = A3Cookie2x10Expert if task == "2x10" else A3SameColumnBatchExpert
        expert = factory(env)
        primary_joint = env._scene_randomizer.target_joints[0]
        primary_address = int(env.model.jnt_qposadr[primary_joint])
        initial = env.data.qpos[primary_address : primary_address + 3].copy()
        work = env._ik._work
        desired, actual, angle_error = np.empty(4), np.empty(4), np.empty(3)
        position_tol = getattr(expert, "_target_preflight_position_tol", .002)
        angle_tol = getattr(expert, "_target_preflight_rotation_tol", .035)
        noise = benchmark.target_bin_noise_m
        yaw_noise = benchmark.target_bin_yaw_noise_rad
        splits = (
            [Cookie2x10Plan.from_seed(first, 4) for first in range(10)]
            if task == "2x10" else [None]
        )
        for dx, dy, yaw in product((-noise, noise), (-noise, noise), (-yaw_noise, yaw_noise)):
            env.data.qpos[primary_address : primary_address + 3] = initial + [dx, dy, 0]
            env.data.qpos[primary_address + 3 : primary_address + 7] = (
                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
            )
            env._scene_randomizer.move_target_boxes()
            mujoco.mj_forward(env.model, env.data)
            for body in env.target_box_bodies:
                env._target_bin_body = body
                expert._tb_id = body
                for plan in splits:
                    batch_sizes = (5, 5)
                    if plan is not None:
                        expert.plan = plan
                        batch_sizes = plan.batch_sizes
                    for placement, size in enumerate(batch_sizes):
                        if not size:
                            continue
                        expert.batch_index = placement
                        for clearance in (0.0, .0425, .085):
                            position, rotation = expert._place_pose(clearance)
                            mujoco.mju_mat2Quat(desired, rotation.ravel())
                            work.qpos[expert._l_qpos] = env.solve_ik(
                                position, desired, expert.q_transit
                            )
                            mujoco.mj_kinematics(env.model, work)
                            site = expert._l_site
                            mujoco.mju_mat2Quat(actual, work.site_xmat[site])
                            mujoco.mju_subQuat(angle_error, desired, actual)
                            case = (body, plan, placement, clearance, dx, dy, yaw)
                            assert np.linalg.norm(work.site_xpos[site] - position) <= position_tol, case
                            assert np.linalg.norm(angle_error) <= angle_tol, case
    finally:
        env._target_bin_body = primary
        env.close()


def _separating_gap(p, rotation, extent, q, other_rotation, other_extent):
    axes = np.hstack((rotation, other_rotation))
    gap = np.abs((p - q) @ axes)
    gap -= extent @ np.abs(rotation.T @ axes)
    gap -= other_extent @ np.abs(other_rotation.T @ axes)
    return gap.max()


@pytest.mark.parametrize("task", ["2x5", "2x10"])
def test_four_boxes_clear_the_source_and_each_other_throughout_the_yaw_range(task):
    benchmark = _benchmark(task, "advanced", target_box_count=4)
    scene = benchmark.config.cookie_transfer
    poses = benchmark.scene_options.target_boxes(scene)
    target_half_size = np.array(scene.target_bin_half_size_m) + scene.bin_wall_thickness_m
    source_half_size = np.array(scene.source_bin_half_size_m) + scene.bin_wall_thickness_m
    # Relative translation between a target and the source can reach the
    # sum of their independent XY noise bounds.
    source_half_size += benchmark.target_bin_noise_m + benchmark.source_bin_noise_m
    source_xy = np.asarray(scene.source_bin_center_m)
    max_yaw = benchmark.target_bin_yaw_noise_rad
    target_radius = np.linalg.norm(target_half_size)
    yaw_sweep = 2 * target_radius * np.sin(max_yaw / 2)

    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array(((c, -s), (s, c)))

    for _, position, yaw in poses:
        xy = np.asarray(position[:2])
        nominal_rotation = rotation(yaw)
        gap = _separating_gap(
            xy, nominal_rotation, target_half_size, source_xy, np.eye(2), source_half_size
        )
        assert gap > yaw_sweep
        # The robot scene's table_top extends from (-.40, -.75) to (.70, .75).
        swept_extent = (
            np.abs(nominal_rotation) @ target_half_size
            + yaw_sweep + benchmark.target_bin_noise_m
        )
        assert np.all(xy - swept_extent >= (-.40, -.75))
        assert np.all(xy + swept_extent <= (.70, .75))

    yaw_samples = np.linspace(-max_yaw, max_yaw, 5)
    for (_, p, p_yaw), (_, q, q_yaw) in combinations(poses, 2):
        a, b = np.asarray(p[:2]), np.asarray(q[:2])
        min_gap = min(
            _separating_gap(
                a, rotation(p_yaw + yaw), target_half_size,
                b, rotation(q_yaw + yaw), target_half_size,
            )
            for yaw in yaw_samples
        )
        # Both boxes yaw by the same amount. Relative vertex projections
        # change only through the rotated separating axes between samples.
        nearest_sample_angle = max_yaw / 4
        assert min_gap > np.linalg.norm(a - b) * nearest_sample_angle


@pytest.mark.parametrize("task,box_count", [("2x5", 4), ("2x10", 3)])
@pytest.mark.parametrize("profile", ["basic", "medium", "advanced"])
def test_scene_options_combine_and_hold_at_each_randomization_level(task, box_count, profile):
    benchmark = _benchmark(
        task, profile, source_gap_groups=[2, 5], target_box_count=box_count,
        source_tilt_deg=8, cookie_crush_force_n=12,
    )
    env = benchmark.create_env()
    try:
        seed = 4
        _, initial = env.reset(seed=seed, options=benchmark.reset_options(seed))
        metadata = env.randomization_metadata["scene"]
        groups = metadata["gap_groups"]
        assert len(groups) == (5 if task == "2x5" else 4)
        missing = {i for group in groups for i in group}
        assert len(missing) == 5 * len(groups)
        for group in groups:
            assert len({env.SOURCE_POSITIONS[i][0] for i in group}) == 1
            assert list(group) == list(range(group[0], group[0] + 5))
        for i in missing:
            geoms = env._scene_randomizer._intact_geoms[i]
            assert not env.model.geom_contype[geoms].any()
            assert not env.model.geom_rgba[geoms, 3].any()

        tilted = metadata["tilted_cookie_ids"]
        assert len(tilted) == 5 and not missing.intersection(tilted)
        assert len(metadata["target_boxes"]) == box_count
        assert initial["cookies_in_source"] == 80 - len(missing)
        assert initial["required_cookies"] == benchmark.required_cookies
        _, again = env.reset(seed=seed, options=benchmark.reset_options(seed))
        repeated = env.randomization_metadata["scene"]
        assert repeated["gap_groups"] == groups
        assert repeated["tilted_cookie_ids"] == tilted
        assert repeated["requested_tilt_deg"] == metadata["requested_tilt_deg"]
        for actual, expected in zip(repeated["target_boxes"], metadata["target_boxes"], strict=True):
            np.testing.assert_allclose(actual["position"], expected["position"], atol=1e-7)
            np.testing.assert_allclose(actual["quaternion"], expected["quaternion"], atol=1e-6)
        assert again["cookies_in_source"] == initial["cookies_in_source"]

        for _ in range(8):
            _, _, terminated, _, info = env.step(env.DEPLOYMENT_HOME)
            assert not terminated
        z = env.data.xmat[[env._cookie_bodies[i] for i in tilted], 8]
        tilts_deg = np.rad2deg(np.arccos(np.clip(z, -1, 1)))
        np.testing.assert_allclose(tilts_deg, 8.0, atol=1.5)
        assert info["cookies_in_source"] == initial["cookies_in_source"]
        assert info["scenario_metrics"]["target_counts"] == [0] * box_count
        assert not info["scenario_metrics"]["crushed_cookie_ids"]
    finally:
        env.close()


def test_gap_range_respects_stock_without_making_fixed_requests_silent():
    three_boxes = _benchmark("2x10", source_gap_groups=[2, 5], target_box_count=3)
    assert three_boxes.scene_options.feasible_gap_range(three_boxes.config.cookie_transfer) == (2, 4)
    with pytest.raises(ValueError, match=r"needs 60 cookies"):
        _benchmark("2x10", source_gap_groups=5, target_box_count=3)
    with pytest.raises(ValueError, match=r"needs 80 cookies"):
        _benchmark("2x10", source_gap_groups=[2, 5], target_box_count=4)
    with pytest.raises(ValueError, match=r"source_gap_groups"):
        _benchmark(source_gap_groups=1)


def test_benchmark_reports_scene_inventory_and_max_score():
    benchmark = _benchmark("2x10", target_box_count=3, source_gap_groups=4)
    env = benchmark.create_env()
    action = env.DEPLOYMENT_HOME

    class HoldPolicy:
        finished = True

        def act(self, observation, task):
            assert "all 3 target boxes" in task
            return action

    try:
        result = benchmark.run_episode(HoldPolicy(), seed=4, env=env)
        assert (result.score, result.max_score) == (0, 60)
        assert result.cookies_in_source == 60
        assert result.scenario_metrics["expected_source_count"] == 0
        assert result.scenario_metrics["target_counts"] == [0, 0, 0]
        assert not result.success
    finally:
        env.close()


@pytest.mark.parametrize("task,box_count", [
    ("2x5", 2), ("2x5", 3), ("2x5", 4), ("2x10", 2), ("2x10", 3), ("2x10", 4),
])
def test_all_target_boxes_must_be_full(task, box_count):
    gap_count = 0 if (task, box_count) == ("2x10", 4) else 2
    benchmark = _benchmark(task, "advanced", target_box_count=box_count, source_gap_groups=gap_count)
    env = benchmark.create_env()
    try:
        env.reset(seed=17, options=benchmark.reset_options(17))
        active = iter(np.flatnonzero(env._scene_randomizer.active))
        capacity = len(env.TARGET_SLOTS_LOCAL)
        # Direct placements isolate the evaluator from controller performance.
        pending = []
        for body in env.target_box_bodies:
            for slot in range(capacity):
                center = env.privileged_target_slot_world(
                    slot, env._target_cookie_center_z + 0.001, target_body=body
                )
                # Leave a sub-millimeter gap in the evaluator-only fixture.
                # Tangent collision faces at exactly the nominal Cookie
                # pitch can push entire rows over on different MuJoCo builds;
                # the small spread remains within the same scored slots.
                center += env.data.xmat[body].reshape(3, 3)[:, 1] * (
                    0.08 * env.TARGET_SLOTS_LOCAL[slot][1]
                )
                pending.append((next(active), center, env.data.xquat[body].copy()))
        for placement in pending[:-1]:
            env.set_cookie_pose(*placement)
        for _ in range(5):
            _, _, _, _, info = env.step(env.DEPLOYMENT_HOME)
        assert info["scenario_metrics"]["target_counts"] == [capacity] * (box_count - 1) + [
            capacity - 1,
        ]
        assert not info["success"]

        env.set_cookie_pose(*pending[-1])
        for _ in range(5):
            _, _, _, _, info = env.step(env.DEPLOYMENT_HOME)
        assert info["scenario_metrics"]["target_filled"] == [True] * box_count
        assert info["required_cookies"] == capacity * box_count
        assert info["cookies_in_target"] == capacity * box_count
        assert info["scenario_metrics"]["target_score"] == capacity * box_count
        assert info["cookies_in_source"] == 80 - 5 * gap_count - capacity * box_count
        assert info["success"]
    finally:
        env.close()


def test_real_grip_load_crushes_geometry_and_reset_restores_it():
    benchmark = _benchmark(profile="advanced", cookie_crush_force_n=12)
    env = benchmark.create_env()
    try:
        env.reset(seed=4, options=benchmark.reset_options(4))
        model, data = env.model, env.data
        midpoint = data.geom_xpos[list(env._left_finger_geoms)].mean(axis=0)
        gripper = model.body("L_robotiq_2f85").id
        orientation = data.xquat[gripper].copy()
        jaw_axis = data.xmat[gripper].reshape(3, 3)[:, 1]
        thickness = 2 * env.COOKIE_HALF_SIZE[1]
        for i, offset in enumerate(np.arange(-2, 3) * thickness):
            env.set_cookie_pose(i, midpoint + jaw_axis * offset, orientation)
        # Use a floating grip fixture to measure the contact force with no floor load.
        bodies = list(env._cookie_bodies[:5])
        normal_gravity = model.body_gravcomp[bodies].copy()
        model.body_gravcomp[bodies] = 1.0
        mujoco.mj_forward(model, data)

        action = env.DEPLOYMENT_HOME.copy()
        last_intact_load = 0.0
        for opening in (0.7, 0.55, 0.4, 0.33, 0.28, 0.22, 0.15, 0.08, 0.0):
            action[7] = opening
            for _ in range(3):
                _, _, terminated, _, info = env.step(action)
                if terminated:
                    break
            if terminated:
                break
            last_intact_load = info["scenario_metrics"]["peak_cookie_grip_load_n"]

        assert 1.0 < last_intact_load < 12.0
        assert terminated and not info["success"] and info["safety_reason"] == "cookie_crushed"
        metrics = info["scenario_metrics"]
        assert metrics["peak_cookie_grip_load_n"] >= 12.0
        assert metrics["crushed_cookie_ids"]
        for i in metrics["crushed_cookie_ids"]:
            normal = model.geom(f"cookie_{i}_geom").id
            crushed = model.geom(f"cookie_{i}_crushed").id
            assert model.geom_size[crushed].min() < model.geom_size[normal].min()
            assert model.geom_contype[crushed] and model.geom_rgba[crushed, 3] == 1
            assert not model.geom_contype[normal]

        model.body_gravcomp[bodies] = normal_gravity
        _, info = env.reset(seed=4, options=benchmark.reset_options(4))
        assert info["cookies_in_source"] == 80
        assert not info["scenario_metrics"]["crushed_cookie_ids"]
        assert info["scenario_metrics"]["peak_cookie_grip_load_n"] == 0
        for i in metrics["crushed_cookie_ids"]:
            assert model.geom_contype[model.geom(f"cookie_{i}_geom").id]
            crushed = model.geom(f"cookie_{i}_crushed").id
            assert not model.geom_contype[crushed] and model.geom_rgba[crushed, 3] == 0
    finally:
        env.close()
