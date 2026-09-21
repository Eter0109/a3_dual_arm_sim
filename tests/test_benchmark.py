from __future__ import annotations

import numpy as np

from a3_dual_arm_sim.benchmark import (
    BenchmarkPolicy,
    BenchmarkResult,
    CookieBatchBenchmark,
    EpisodeScore,
    ExpertPolicyAdapter,
    make_policy_adapter,
)
from a3_dual_arm_sim.policy import HoldPolicy
from a3_dual_arm_sim.recording import MemoryRecorder


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
    class Policy:
        action_mode = "cartesian_delta"
        name = "test_expert"
        finished = True

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
    assert len(recorder.episodes) == 1
    np.testing.assert_array_equal(recorder.episodes[0][0]["action"], np.arange(16))
