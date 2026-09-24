from __future__ import annotations

import numpy as np

from a3_dual_arm_sim.benchmark import (
    BenchmarkPolicy,
    BenchmarkResult,
    CookieBatchBenchmark,
    DIVERSE_RANDOMIZATION,
    EpisodeScore,
    ExpertPolicyAdapter,
    column_task_instruction,
    make_policy_adapter,
)
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.policy import HoldPolicy
from a3_dual_arm_sim.recording import MemoryRecorder


def test_visual_policy_auto_enables_rgb_and_reuses_model_for_serial_episodes(monkeypatch):
    class Policy:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Env:
        def __init__(self, render_cameras):
            self.render_cameras = render_cameras
            self.closed = False

        def close(self):
            self.closed = True

    benchmark = CookieBatchBenchmark(max_steps=1)
    made_policies = []
    made_envs = []
    seen_policies = []

    def make_policy(_spec):
        instance = Policy()
        made_policies.append(instance)
        return instance

    def make_env(*, render_cameras=False):
        instance = Env(render_cameras)
        made_envs.append(instance)
        return instance

    def run_episode(policy, seed, episode_idx, env):
        seen_policies.append(policy)
        return EpisodeScore(episode=episode_idx, seed=seed, score=0)

    monkeypatch.setattr("a3_dual_arm_sim.benchmark.make_policy_adapter", make_policy)
    monkeypatch.setattr(benchmark, "create_env", make_env)
    monkeypatch.setattr(benchmark, "run_episode", run_episode)

    result = benchmark.evaluate("example.visual:make_policy", num_episodes=2, workers=1, verbose=False)
    assert len(made_policies) == 1
    assert seen_policies == made_policies * 2
    assert made_policies[0].closed
    assert len(made_envs) == 1
    assert made_envs[0].render_cameras and made_envs[0].closed
    assert result.to_dict()["summary"]["policy_rgb_cameras"] is True


def test_expert_benchmark_keeps_rgb_disabled(monkeypatch):
    benchmark = CookieBatchBenchmark(max_steps=1)
    seen = []

    class Env:
        def close(self):
            pass

    def make_env(*, render_cameras=False):
        seen.append(render_cameras)
        return Env()

    monkeypatch.setattr(benchmark, "create_env", make_env)
    monkeypatch.setattr(
        benchmark, "run_episode",
        lambda policy, seed, episode_idx, env: EpisodeScore(
            episode=episode_idx, seed=seed, score=0
        ),
    )
    benchmark.evaluate("same_column", num_episodes=1, workers=1, verbose=False)
    assert seen == [False]


def test_visual_policy_rejects_supplied_black_camera_environment():
    class Env:
        render_cameras = False

    class Policy:
        def act(self, observation, task=""):
            return np.zeros(16)

    benchmark = CookieBatchBenchmark()
    import pytest

    with pytest.raises(ValueError, match="render_cameras=False"):
        benchmark.run_episode(Policy(), seed=0, env=Env())


def test_benchmark_tracks_transient_cookie_extraction():
    class Policy:
        requires_camera_rendering = False

        def act(self, observation, task=""):
            return np.zeros(16)

    class Env:
        render_cameras = False
        action_mode = "joint_position"

        def __init__(self):
            self.steps = 0

        def reset(self, seed, options):
            self.steps = 0
            return {}, {"cookies_in_target": 0, "cookies_in_source": 80}

        def step(self, action):
            self.steps += 1
            source_counts = [80, 75, 80]
            return {}, 0.0, False, False, {
                "cookies_in_target": 0,
                "cookies_in_source": source_counts[self.steps - 1],
            }

    result = CookieBatchBenchmark(max_steps=3).run_episode(Policy(), seed=0, env=Env())
    assert result.cookies_in_source == 80
    assert result.min_cookies_in_source == 75
    assert result.max_cookies_in_target == 0


def test_diverse_profile_includes_geometry_and_appearance_randomization():
    assert DIVERSE_RANDOMIZATION == {
        "source_bin_noise_m": 0.020,
        "target_bin_noise_m": 0.020,
        "target_bin_yaw_noise_rad": 0.080,
        "cookie_noise_m": 0.0003,
        "cookie_yaw_noise_rad": 0.015,
        "camera_position_noise_m": 0.009,
        "camera_fovy_noise_deg": 2.0,
        "light_noise_fraction": 0.20,
        "color_noise_fraction": 0.15,
    }


def test_expert_preflight_rejection_counts_as_failed_episode(monkeypatch):
    class Env:
        action_mode = "joint_position"

        def reset(self, seed, options):
            return {}, {"cookies_in_target": 0, "cookies_in_source": 80}

    def reject(*_args, **_kwargs):
        raise RuntimeError("target unreachable with vertical grasp")

    monkeypatch.setattr("a3_dual_arm_sim.benchmark.make_policy_adapter", reject)
    result = CookieBatchBenchmark().run_episode("varied_column", seed=4, env=Env())
    assert not result.success
    assert result.score == 0
    assert result.cookies_in_source == 80
    assert result.failure_reason.startswith("expert_preflight:")


def test_benchmark_policy_protocol_conformance():
    class DummyPolicy:
        def act(self, observation, task=""):
            return np.zeros(14)

        def reset(self, context=None):
            pass

    dummy = DummyPolicy()
    assert isinstance(dummy, BenchmarkPolicy)


def test_make_policy_adapter_resolves_expert_names():
    adapter_same = make_policy_adapter("same_column")
    assert isinstance(adapter_same, ExpertPolicyAdapter)
    assert adapter_same.name == "same_column"

    adapter_cross = make_policy_adapter("cross_column")
    assert isinstance(adapter_cross, ExpertPolicyAdapter)
    assert adapter_cross.name == "cross_column"

    adapter_varied = make_policy_adapter("varied_column")
    assert isinstance(adapter_varied, ExpertPolicyAdapter)
    assert adapter_varied.name == "varied_column"

    for index in range(4):
        adapter_fixed = make_policy_adapter(f"column_{index}")
        assert isinstance(adapter_fixed, ExpertPolicyAdapter)
        assert adapter_fixed.name == f"column_{index}"

    hold = make_policy_adapter(HoldPolicy())
    assert isinstance(hold, HoldPolicy)


def test_box_randomization_options():
    benchmark = CookieBatchBenchmark(
        randomize_boxes=True,
        target_bin_noise_m=0.005,
        source_bin_noise_m=0.003,
        randomize_cookies=True,
    )
    env = benchmark.create_env()
    try:
        _, _info1 = env.reset(
            seed=1,
            options={
                "randomize_boxes": True,
                "target_bin_noise_m": 0.005,
                "source_bin_noise_m": 0.003,
            },
        )
        t_pos1 = env.data.xpos[env._target_bin_body].copy()
        s_pos1 = env.data.xpos[env._source_bin_body].copy()

        _, _info2 = env.reset(
            seed=2,
            options={
                "randomize_boxes": True,
                "target_bin_noise_m": 0.005,
                "source_bin_noise_m": 0.003,
            },
        )
        t_pos2 = env.data.xpos[env._target_bin_body].copy()
        s_pos2 = env.data.xpos[env._source_bin_body].copy()

        # Positions across different seeds should have slight random differences
        assert not np.allclose(t_pos1[:2], t_pos2[:2], atol=1e-5)
        assert not np.allclose(s_pos1[:2], s_pos2[:2], atol=1e-5)
    finally:
        env.close()


def test_diverse_appearance_is_seeded_and_restores_baseline():
    env = CookieBatchBenchmark().create_env(render_cameras=False)
    options = {
        "randomize_cookies": False,
        "camera_position_noise_m": 0.009,
        "camera_fovy_noise_deg": 2.0,
        "light_noise_fraction": 0.2,
        "color_noise_fraction": 0.15,
    }
    try:
        env.reset(seed=17, options=options)
        first = (
            env.model.cam_pos.copy(), env.model.cam_fovy.copy(),
            env.model.light_diffuse.copy(), env.model.geom_rgba.copy(),
        )
        env.reset(seed=17, options=options)
        second = (
            env.model.cam_pos.copy(), env.model.cam_fovy.copy(),
            env.model.light_diffuse.copy(), env.model.geom_rgba.copy(),
        )
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
        env.reset(seed=18, options=options)
        assert not np.array_equal(first[0], env.model.cam_pos)
        assert not np.array_equal(first[3], env.model.geom_rgba)
        env.reset(seed=19, options={"randomize_cookies": False})
        assert np.array_equal(env.model.cam_pos, env._base_cam_pos)
        assert np.array_equal(env.model.cam_fovy, env._base_cam_fovy)
        assert np.array_equal(env.model.light_diffuse, env._base_light_diffuse)
        assert np.array_equal(env.model.geom_rgba, env._base_geom_rgba)
    finally:
        env.close()


def test_varied_expert_changes_column_order_by_seed():
    env = CookieBatchBenchmark().create_env(render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        adapter = make_policy_adapter("varied_column", env=env)
        context = lambda seed: EpisodeContext(
            seed=seed, task="transfer 10 cookies into target box", action_mode=env.action_mode
        )
        adapter.reset(context(0))
        first = adapter.expert._column_order.copy()
        adapter.expert._held_rotation_batch = (40, 41, 42, 43, 44)
        adapter.reset(context(0))
        assert first == adapter.expert._column_order
        assert adapter.expert._held_rotation_batch is None
        orders = []
        for seed in range(8):
            adapter.reset(context(seed))
            orders.append(adapter.expert.selected_source_column_index)
        assert set(orders[:4]) == set(range(4))
        assert set(orders[4:]) == set(range(4))
        for index in range(4):
            fixed = make_policy_adapter(f"column_{index}", env=env)
            fixed.reset(context(17))
            assert fixed.expert.selected_source_column_index == index
            source_x = np.asarray(env.SOURCE_POSITIONS)[:, 0]
            assert fixed.expert._column_order == [sorted(set(source_x))[index]]
            assert f"column {index + 1}" in column_task_instruction(index)
    finally:
        env.close()


def test_wide_target_pose_passes_expert_workspace_preflight():
    benchmark = CookieBatchBenchmark(
        source_bin_noise_m=0.020,
        target_bin_noise_m=0.020,
        target_bin_yaw_noise_rad=0.080,
    )
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(
            seed=2,
            options={
                "randomize_boxes": True,
                "source_bin_noise_m": benchmark.source_bin_noise_m,
                "target_bin_noise_m": benchmark.target_bin_noise_m,
                "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
            },
        )
        assert make_policy_adapter("same_column", env=env).expert is not None
    finally:
        env.close()


def test_benchmark_score_and_summary_metrics():
    episodes = [
        EpisodeScore(episode=1, seed=0, score=10, success=True, steps=450, wall_seconds=60.0),
        EpisodeScore(episode=2, seed=1, score=10, success=True, steps=460, wall_seconds=62.0),
        EpisodeScore(episode=3, seed=2, score=5, success=False, steps=300, wall_seconds=40.0),
    ]
    res = BenchmarkResult(
        policy_name="test_policy",
        total_episodes=3,
        total_score=25,
        max_possible_score=30,
        mean_score=25 / 3,
        min_score=5,
        max_score=10,
        success_rate=2 / 3,
        mean_steps=403.33,
        mean_wall_seconds=54.0,
        score_distribution={10: 2, 5: 1},
        episodes=episodes,
    )
    d = res.to_dict()
    assert d["summary"]["total_score"] == 25
    assert d["summary"]["max_possible_score"] == 30
    assert d["summary"]["min_score"] == 5
    assert d["summary"]["max_score"] == 10
    assert len(d["episodes"]) == 3
    table = res.summary_table()
    assert "25 / 30" in table
    assert "test_policy" in table


def test_benchmark_recorder_stores_applied_joint_action():
    class Expert:
        selected_source_column_index = 2

    class Policy:
        action_mode = "cartesian_delta"
        name = "test_expert"
        finished = True
        expert = Expert()

        def reset(self, context=None):
            self.context = context

        def act(self, observation, task=""):
            return np.zeros(14)

    class Data:
        xpos = np.zeros((2, 3))

    class Env:
        action_mode = "joint_position"
        data = Data()
        _target_bin_body = 0
        _source_bin_body = 1

        def reset(self, seed, options):
            return {"marker": seed}, {}

        def step(self, action):
            applied = np.arange(16, dtype=np.float64)
            return (
                {},
                0.0,
                False,
                False,
                {
                    "applied_action": applied,
                    "success": True,
                    "cookies_in_target": 10,
                    "cookies_in_source": 70,
                    "safety_reason": None,
                },
            )

    recorder = MemoryRecorder()
    score = CookieBatchBenchmark(max_steps=1).run_episode(
        Policy(), seed=4, env=Env(), recorder=recorder
    )
    assert score.success
    assert score.max_cookies_in_target == 10
    assert score.min_cookies_in_source == 70
    assert score.source_column_index == 2
    assert score.source_column_number == 3
    assert "column 3" in score.task_instruction
    assert recorder.context.task == score.task_instruction
    assert len(recorder.episodes) == 1
    np.testing.assert_array_equal(recorder.episodes[0][0]["action"], np.arange(16))
