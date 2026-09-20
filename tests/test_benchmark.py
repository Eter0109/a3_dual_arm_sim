from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from a3_dual_arm_sim.benchmark import (
    BenchmarkPolicy,
    BenchmarkResult,
    CookieBatchBenchmark,
    EpisodeScore,
    ExpertPolicyAdapter,
    make_policy_adapter,
)
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.policy import HoldPolicy


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
        _, info1 = env.reset(seed=1, options={
            "randomize_boxes": True,
            "target_bin_noise_m": 0.005,
            "source_bin_noise_m": 0.003,
        })
        t_pos1 = env.data.xpos[env._target_bin_body].copy()
        s_pos1 = env.data.xpos[env._source_bin_body].copy()

        _, info2 = env.reset(seed=2, options={
            "randomize_boxes": True,
            "target_bin_noise_m": 0.005,
            "source_bin_noise_m": 0.003,
        })
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
