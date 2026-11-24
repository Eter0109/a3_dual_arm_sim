"""Course protocol regressions; physical expert trials are reported separately."""

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.tasks.cookie_scenarios import SCENARIO_2X10_TASKS
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
from a3_dual_arm_sim.workflows.course_protocol import COURSE_PROMPT, COURSE_TASKS


def options(benchmark):
    return {"randomize_boxes": True, "randomize_cookies": True,
            "source_bin_noise_m": benchmark.source_bin_noise_m,
            "target_bin_noise_m": benchmark.target_bin_noise_m,
            "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
            "randomization_profile": "basic", "randomization_settings": benchmark.randomization_settings}


def settle(env):
    mujoco.mj_forward(env.model, env.data)
    for _ in range(200):
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


def hold(env):
    info = None
    for _ in range(env.task_config.success_hold_steps + 10):
        _, _, _, _, info = env.step(env.DEPLOYMENT_HOME)
        if info["success"]:
            break
    return info


@pytest.mark.parametrize("task", COURSE_TASKS)
def test_course_prompt_and_protocol(task):
    benchmark = make_cookie_benchmark(task, profile="basic", max_steps=1)
    assert benchmark._episode_task(1, 13000) == COURSE_PROMPT
    assert benchmark._episode_task(4, 13008) == COURSE_PROMPT
    assert benchmark.protocol == "course-v1"


@pytest.mark.parametrize("task", SCENARIO_2X10_TASKS)
@pytest.mark.parametrize("seed", [13000, 13008, 13009])
def test_new_2x10_inventory_clearance_and_reset(task, seed):
    benchmark = make_cookie_benchmark(task, profile="basic", max_steps=1)
    env = benchmark.create_env(render_cameras=False)
    try:
        _, info = env.reset(seed=seed, options=options(benchmark))
        meta = env.scenario_metadata
        assert env.task_config.required_cookies == 20
        assert sum(info["cookies_in_source_mask"]) == meta["initial_source_count"]
        assert meta["target_quotas"] == [20]
        assert all(gap >= 0.003 - 1e-9
                   for gap in env.randomization_metadata["source_column_clearances_m"])
        env._check_container_contacts()
        for contact in env.data.contact:
            if contact.dist >= -1e-5:
                continue
            first = int(env.model.geom_bodyid[int(contact.geom1)])
            second = int(env.model.geom_bodyid[int(contact.geom2)])
            if first != second and first in env._cookie_bodies and second in env._cookie_bodies:
                assert not (env._active[env._cookie_bodies.index(first)]
                            and env._active[env._cookie_bodies.index(second)])
        camera = env._camera_ids["front"]
        camera_r = env.data.cam_xmat[camera].reshape(3, 3)
        box = env._target_bin_body
        box_r = env.data.xmat[box].reshape(3, 3)
        scene = env.config.cookie_transfer
        half_x, half_y = np.asarray(scene.target_bin_half_size_m) + scene.bin_wall_thickness_m
        tangent = np.tan(np.deg2rad(env.model.cam_fovy[camera] / 2))
        width, height = env.config.image_width, env.config.image_height
        for x in (-half_x, half_x):
            for y in (-half_y, half_y):
                point = env.data.xpos[box] + box_r @ np.array([x, y, scene.target_bin_wall_height_m])
                local = camera_r.T @ (point - env.data.cam_xpos[camera])
                depth = -local[2]
                u = (.5 + .5 * local[0] / (depth * tangent * width / height)) * width
                v = (.5 - .5 * local[1] / (depth * tangent)) * height
                assert depth > 0 and 0 <= u <= width and 0 <= v <= height
        if task == "2x10-missing-columns":
            assert meta["initial_source_count"] in (40, 60)
        if task == "2x10-partial-gaps":
            assert meta["initial_source_count"] == 56
        if task == "2x10-cross-column":
            assert meta["initial_source_count"] == 32 and meta["remaining_per_column"] == 8
        if task == "2x10-sparse-exact-twenty":
            assert sorted(meta["initial_column_counts"]) == [2, 4, 6, 8]
            assert meta["initial_source_count"] == 20
            assert benchmark.config.cookie_transfer.target_bin_half_size_m[1] == pytest.approx(.070)
        if task == "2x10-tilted":
            assert all(meta["initial_tilt_degrees"][i] >= 5 for i in meta["tilted_cookie_ids"])
            if seed == 13009:
                assert meta["scenario_attempt"] > 0
    finally:
        env.close()


def test_prefill_identity_and_outside_source_do_not_gate_course_success():
    benchmark = make_cookie_benchmark("prefilled", profile="basic", max_steps=1)
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        original = min(env._prefilled)
        slot = env.scenario_metadata["prefilled_slot_ids"][sorted(env._prefilled).index(original)]
        env.set_cookie_pose(original, (1.0, 1.0, 0.77))
        free_slots = sorted(set(range(10)) - set(env.scenario_metadata["prefilled_slot_ids"])) + [slot]
        available = [i for i in range(80) if i not in env._prefilled][:len(free_slots)]
        for index, target_slot in zip(available, free_slots, strict=True):
            env._put_in_primary_slot(index, target_slot)
        settle(env)
        info = hold(env)
        assert info["scenario_metrics"]["prefilled_remaining"] == 4
        assert info["scenario_metrics"]["cookies_outside_containers"] >= 1
        assert info["scenario_metrics"]["target_counts"] == [10]
        assert info["success"]
    finally:
        env.close()


def test_lean_source_is_not_a_success_gate():
    env = make_cookie_benchmark("whole-row-lean", profile="basic", max_steps=1).create_env(render_cameras=False)
    try:
        env.reset(seed=13000, options={"randomize_boxes": False, "randomize_cookies": False})
        selected = [i for i in range(80) if i not in env._eligible][:10]
        for slot, index in enumerate(selected):
            env._put_in_primary_slot(index, slot)
        settle(env)
        assert hold(env)["success"]
    finally:
        env.close()


def test_legacy_quota_prompts_remain_distinct():
    for task in ("2x10-four-source", "2x10-paired-columns"):
        benchmark = make_cookie_benchmark(task, profile="basic", max_steps=1)
        assert benchmark.protocol == "legacy"
        assert benchmark._episode_task(1, 13000) != COURSE_PROMPT


def test_expanded_2x10_accepts_both_randomization_files():
    from a3_dual_arm_sim.paths import resource_root

    for filename in ("randomization.yaml", "randomization_2x10.yaml"):
        benchmark = make_cookie_benchmark(
            "2x10-cross-column", randomization_config=resource_root() / "configs" / filename
        )
        assert benchmark.profile == "basic"
        assert benchmark._episode_task(1, 0) == COURSE_PROMPT


def test_worker_preserves_course_protocol():
    benchmark = make_cookie_benchmark("2x10-missing-columns", profile="basic", max_steps=1)
    # Use the old explicit controller for a one-step plumbing check only.
    result = benchmark.evaluate("same_column", num_episodes=2, workers=2, verbose=False)
    assert all(ep.randomization["task_instruction"] == COURSE_PROMPT for ep in result.episodes)
    assert all(ep.randomization["protocol"] == "course-v1" for ep in result.episodes)


def test_policy_act_receives_course_prompt():
    class TracePolicy:
        action_mode = "cartesian_delta"
        requires_cameras = False
        finished = False

        def reset(self, context):
            self.reset_prompt = context.task

        def act(self, observation, task):
            self.act_prompt = task
            return np.zeros(14)

    policy = TracePolicy()
    benchmark = make_cookie_benchmark("2x10-partial-gaps", max_steps=1)
    result = benchmark.evaluate(policy, num_episodes=1, workers=1, verbose=False)
    assert policy.reset_prompt == policy.act_prompt == COURSE_PROMPT
    assert result.episodes[0].randomization["task_instruction"] == COURSE_PROMPT


def test_course_suite_has_90_episodes_and_distinct_profile_contract(monkeypatch):
    from a3_dual_arm_sim.workflows import course_benchmark_cli

    calls = []

    class Result:
        def to_dict(self):
            return {"summary": {}, "episodes": [{"success": True} for _ in range(5)]}

    class Benchmark:
        config = type("Config", (), {"horizon": 12000})()

        def evaluate(self, policy, **kwargs):
            assert kwargs["num_episodes"] == 5
            return Result()

    def make(task, **kwargs):
        calls.append((task, kwargs))
        return Benchmark()

    monkeypatch.setattr(course_benchmark_cli, "make_cookie_benchmark", make)
    report = course_benchmark_cli.evaluate_course(policy="expert")
    assert report["status"] == "complete"
    assert report["base"] == {"completed": 70, "expected": 70, "successes": 70}
    assert report["bonus"] == {"completed": 20, "expected": 20, "successes": 20, "points": 20}
    assert len(calls) == 18
    assert all(kwargs["profile"] == ("advanced" if task in ("2x5", "2x10") else "basic")
               for task, kwargs in calls)
    assert all(kwargs["max_steps"] == (12000 if task.startswith("2x10") else 6000)
               for task, kwargs in calls)


def test_course_convenience_entrypoint_defaults_to_five_episodes():
    from a3_dual_arm_sim.workflows.benchmark import run_cookie_batch_benchmark

    class StopPolicy:
        action_mode = "cartesian_delta"
        requires_cameras = False
        finished = True

        def reset(self, context):
            self.prompt = context.task

        def act(self, observation, task):
            return np.zeros(14)

    benchmark = make_cookie_benchmark("2x10")
    assert benchmark.max_steps == benchmark.config.horizon == 12000
    result = run_cookie_batch_benchmark(StopPolicy(), task="2x10", verbose=False)
    assert result.total_episodes == 5
    assert all(episode.randomization["task_instruction"] == COURSE_PROMPT
               for episode in result.episodes)


def test_sparse_twenty_clears_blocker_before_rear_cookie():
    from a3_dual_arm_sim.controllers.course_sparse_expert import A3CourseSparseExpert

    benchmark = make_cookie_benchmark("2x10-sparse-exact-twenty", profile="basic", max_steps=1)
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=13008, options=options(benchmark))
        expert = A3CourseSparseExpert(env)
        selected = []
        for _ in range(3):
            assert expert._plan()
            cookie = expert._cookie
            slot = expert._slots[expert._slot_cursor]
            selected.append((cookie, slot))
            env._put_in_primary_slot(cookie, slot)
            expert._used.add(cookie)
            if cookie in expert._early_priority:
                expert._early_priority.remove(cookie)
            expert._slot_cursor += 1
            settle(env)
        assert selected == [(77, 19), (56, 0), (78, 2)]
    finally:
        env.close()


def test_sparse_twenty_course_expert_batches_all_source_columns():
    from a3_dual_arm_sim.controllers.course_expert import A3CourseSparse2x10Expert

    benchmark = make_cookie_benchmark("2x10-sparse-exact-twenty", profile="basic", max_steps=1)
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=13002, options=options(benchmark))
        adapter = benchmark._episode_policy("course_expert", None, 13002)
        adapter.bind_env(env)
        expert = adapter.expert
        assert isinstance(expert, A3CourseSparse2x10Expert)
        assert expert.plan.batch_sizes == (8, 2, 6, 4)
        counts = env.scenario_metadata["initial_column_counts"]
        for index, size in enumerate(expert.plan.batch_sizes):
            expert.batch_index = index
            batch = next(expert._candidate_batches())
            assert len(batch) == size
            assert counts[expert.selected_source_column_index] == size
    finally:
        env.close()


@pytest.mark.parametrize("task,required", [
    ("2x5", 10), ("2x10", 20), ("2x10-partial-gaps", 20),
])
def test_course_success_requires_balanced_released_and_stable_targets(monkeypatch, task, required):
    benchmark = make_cookie_benchmark(task, profile="basic", max_steps=1)
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=13000, options={"randomize_boxes": False, "randomize_cookies": False})
        active = getattr(env, "_active", np.ones(env.task_config.cookie_count, dtype=bool))
        chosen = np.flatnonzero(active)[:required]
        for slot, index in enumerate(chosen):
            position = env.privileged_target_slot_world(slot, env._target_cookie_center_z)
            env.set_cookie_pose(int(index), tuple(position))
        settle(env)
        assert hold(env)["success"]

        # Each injected condition isolates one real evaluator input. The poses
        # above are fixtures for scoring, not an expert or a physical rollout.
        with monkeypatch.context() as patch:
            patch.setattr(env, "target_column_counts", lambda mask: (required // 2 - 1, required // 2 + 1))
            assert not hold(env)["success"]
        with monkeypatch.context() as patch:
            patch.setattr(env, "privileged_left_finger_contacts", lambda index: (index == chosen[0], False))
            assert not hold(env)["success"]
        with monkeypatch.context() as patch:
            patch.setattr(env, "privileged_right_finger_contacts", lambda index: (index == chosen[0], False))
            assert not hold(env)["success"]
        with monkeypatch.context() as patch:
            speed = env._cookie_speed
            patch.setattr(env, "_cookie_speed", lambda index, **kwargs: (1.0, 0.0)
                          if index == chosen[0] else speed(index, **kwargs))
            assert not hold(env)["success"]
        assert hold(env)["success"]
    finally:
        env.close()
