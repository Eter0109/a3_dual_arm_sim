"""Behavioral tests for the additional cookie benchmark tasks."""

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.tasks.cookie_scenarios import SCENARIO_TASKS
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


@pytest.mark.parametrize("task", SCENARIO_TASKS)
def test_initial_states_are_valid_reproducible_and_score_zero(task):
    benchmark = make_cookie_benchmark(task, max_steps=1)
    env = benchmark.create_env()
    options = {"randomize_boxes": False, "randomize_cookies": False}
    try:
        _, first = env.reset(seed=3, options=options)
        metadata = env.scenario_metadata.copy()
        positions = env.cookie_positions.copy()
        _, second = env.reset(seed=3, options=options)
        np.testing.assert_allclose(env.cookie_positions, positions, atol=1e-10)
        assert env.scenario_metadata == metadata
        assert first["scenario_metrics"] == second["scenario_metrics"]
        assert first["scenario_metrics"]["new_correct_cookies"] == 0
        assert first["scenario_metrics"]["cookies_outside_containers"] == 0
        assert sum(first["cookies_in_source_mask"]) == metadata["initial_source_count"]
        assert sum(first["cookies_in_target_mask"]) == sum(metadata["initial_target_counts"])
        for index in metadata["removed_cookie_ids"]:
            geoms = env._cookie_all_geoms[index]
            assert not env.model.geom_contype[geoms].any()
            assert not env.model.geom_conaffinity[geoms].any()
            assert not env.model.geom_rgba[geoms, 3].any()
        if task == "missing-columns":
            assert metadata["initial_source_count"] in (40, 60)
        if task == "partial-gaps":
            assert metadata["initial_source_count"] == 56
        if task == "cross-column":
            assert metadata["initial_source_count"] == 32
            assert metadata["remaining_per_column"] == 8
        if task == "tilted":
            assert any(
                metadata["initial_tilt_degrees"][i] >= 5 for i in metadata["tilted_cookie_ids"]
            )
        if task == "prefilled":
            assert metadata["initial_source_count"] == 75
            assert first["scenario_metrics"]["target_counts"] == [5]
        if task == "two-boxes":
            assert first["scenario_metrics"]["target_counts"] == [0, 0]
    finally:
        env.close()


def _settle(env):
    mujoco.mj_forward(env.model, env.data)
    for _ in range(200):
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


def _put(env, index, slot, body=None):
    previous = env._target_bin_body
    try:
        if body is not None:
            env._target_bin_body = body
        env._put_in_primary_slot(index, slot)
    finally:
        env._target_bin_body = previous


def _hold(env):
    info = {}
    for _ in range(env.task_config.success_hold_steps):
        _, _, _, _, info = env.step(env.DEPLOYMENT_HOME)
    return info


def test_prefilled_only_scores_new_cookies_and_requires_existing_cookies():
    env = make_cookie_benchmark("prefilled").create_env()
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        assert env._scenario_metrics()["new_correct_cookies"] == 0
        open_slots = sorted(set(range(10)) - set(env.scenario_metadata["prefilled_slot_ids"]))
        available = [i for i in range(80) if i not in env._prefilled][:5]
        for index, slot in zip(available, open_slots, strict=True):
            _put(env, index, slot)
        _settle(env)
        info = _hold(env)
        assert info["cookies_in_target"] == 5
        assert info["scenario_metrics"]["target_counts"] == [10]
        assert info["scenario_metrics"]["prefilled_remaining"] == 5
        assert info["success"]
        # Knocking out a prefilled cookie makes the task incomplete even if five
        # newly transferred cookies remain in the box.
        env.set_cookie_pose(min(env._prefilled), (1.0, 1.0, 0.77))
        _, _, _, _, info = env.step(env.DEPLOYMENT_HOME)
        assert info["cookies_in_target"] == 5
        assert not info["success"]
    finally:
        env.close()


def test_two_boxes_score_and_success_require_both_quotas():
    env = make_cookie_benchmark("two-boxes").create_env()
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        for index in range(10):
            _put(env, index, index)
        _settle(env)
        info = _hold(env)
        assert info["scenario_metrics"]["target_counts"] == [10, 0]
        assert info["cookies_in_target"] == 5
        assert not info["success"]
        for index, slot in zip(range(1, 10, 2), range(0, 10, 2), strict=True):
            _put(env, index, slot, env._spare_body)
        _settle(env)
        info = _hold(env)
        assert info["scenario_metrics"]["target_counts"] == [5, 5]
        assert info["cookies_in_target"] == 10
        assert info["success"]
    finally:
        env.close()


def test_two_full_boxes_have_independent_ten_cookie_goals():
    benchmark = make_cookie_benchmark("two-boxes", box_quota=10)
    env = benchmark.create_env()
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        for slot in range(10):
            _put(env, slot, slot)
        _settle(env)
        info = _hold(env)
        assert info["scenario_metrics"]["target_counts"] == [10, 0]
        assert info["cookies_in_target"] == 10
        assert not info["success"]
        for slot in range(10):
            _put(env, 10 + slot, slot, env._spare_body)
        _settle(env)
        info = _hold(env)
        assert benchmark.required_cookies == 20
        assert info["scenario_metrics"]["target_counts"] == [10, 10]
        assert info["cookies_in_target"] == 20
        assert info["cookies_in_source"] == 60
        assert info["success"]
    finally:
        env.close()


def test_two_box_positions_pass_existing_grasp_reachability_preflight():
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert

    env = make_cookie_benchmark("two-boxes").create_env()
    primary = env._target_bin_body
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        for body in (primary, env._spare_body):
            env._target_bin_body = body
            A3SameColumnBatchExpert(env)
    finally:
        env._target_bin_body = primary
        env.close()


@pytest.mark.parametrize("tolerance_y", [0.005, 0.0025])
def test_prefilled_cookies_settle_when_target_box_is_rotated(tolerance_y):
    env = make_cookie_benchmark(
        "prefilled", profile="advanced", slot_tolerance_m=(0.015, tolerance_y)
    ).create_env()
    try:
        _, info = env.reset(
            seed=1,
            options={
                "randomize_boxes": True,
                "randomize_cookies": True,
                "source_bin_noise_m": 0.02,
                "target_bin_noise_m": 0.02,
                "target_bin_yaw_noise_rad": 0.08,
                "randomization_profile": "advanced",
            },
        )
        assert info["scenario_metrics"]["target_counts"] == [5]
        assert info["scenario_metrics"]["cookies_outside_containers"] == 0
        assert sum(index >= 0 for index in info["target_slot_occupancy"]) == 5
    finally:
        env.close()


def test_fallen_variant_preserves_actual_inventory_and_pose():
    env = make_cookie_benchmark("tilted", tilt_level="fallen", profile="advanced").create_env()
    try:
        _, info = env.reset(
            seed=2,
            options={
                "randomize_boxes": True,
                "randomize_cookies": True,
                "source_bin_noise_m": 0.02,
                "target_bin_noise_m": 0.02,
                "target_bin_yaw_noise_rad": 0.08,
                "randomization_profile": "advanced",
            },
        )
        metadata = env.scenario_metadata
        assert metadata["tilt_level"] == "fallen"
        assert len(metadata["removed_cookie_ids"]) == 4
        assert metadata["initial_source_count"] == 76
        assert any(metadata["initial_tilt_degrees"][i] >= 60 for i in metadata["tilted_cookie_ids"])
        assert info["scenario_metrics"]["cookies_outside_containers"] == 0
    finally:
        env.close()


def test_fallen_variant_reaches_worker_processes():
    benchmark = make_cookie_benchmark("tilted", tilt_level="fallen", max_steps=1)
    result = benchmark.evaluate("expert", num_episodes=2, workers=2, verbose=False)
    for episode in result.episodes:
        scenario = episode.randomization["scenario"]
        assert scenario["tilt_level"] == "fallen"
        assert scenario["initial_source_count"] == 76
        assert "lying flat" in episode.randomization["task_instruction"]


def test_two_full_boxes_quota_reaches_worker_processes():
    benchmark = make_cookie_benchmark("two-boxes", box_quota=10, max_steps=1)
    result = benchmark.evaluate("expert", num_episodes=2, workers=2, verbose=False)
    for episode in result.episodes:
        assert episode.max_score == 20
        assert episode.randomization["scenario"]["target_quotas"] == [10, 10]
        assert "Transfer 20 cookies" in episode.randomization["task_instruction"]


def test_missing_columns_use_actual_inventory_for_success():
    env = make_cookie_benchmark("missing-columns").create_env()
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        chosen = np.flatnonzero(env._active)[:10]
        for slot, index in enumerate(chosen):
            _put(env, int(index), slot)
        _settle(env)
        info = _hold(env)
        assert info["cookies_in_source"] == env.initial_source_count - 10
        assert info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("task", SCENARIO_TASKS)
def test_benchmark_produces_the_existing_report_format(task):
    benchmark = make_cookie_benchmark(task, max_steps=1)
    result = benchmark.evaluate("expert", num_episodes=1, seed_start=0, verbose=False)
    report = result.to_dict()
    episode = report["episodes"][0]
    assert report["summary"]["total_episodes"] == 1
    assert episode["randomization"]["scenario"]["name"] == task
    assert episode["randomization"]["task_instruction"] == benchmark._episode_task(None, 0)
    assert episode["max_score"] == (5 if task == "prefilled" else 10)
    assert episode["scenario_metrics"]["new_correct_cookies"] == episode["score"]
    assert episode["steps"] == 1


def test_parallel_workers_preserve_scenario_and_seed_metadata():
    benchmark = make_cookie_benchmark("two-boxes", max_steps=1)
    serial = benchmark.evaluate("expert", num_episodes=2, seed_start=4, workers=1, verbose=False)
    parallel = benchmark.evaluate("expert", num_episodes=2, seed_start=4, workers=2, verbose=False)
    assert [e.seed for e in serial.episodes] == [e.seed for e in parallel.episodes]
    for first, second in zip(serial.episodes, parallel.episodes, strict=True):
        assert first.randomization["scenario"] == second.randomization["scenario"]
        assert first.scenario_metrics == second.scenario_metrics
