from __future__ import annotations

import numpy as np

from a3_dual_arm_sim.data.recording import MemoryRecorder
from a3_dual_arm_sim.policies.base import HoldPolicy
from a3_dual_arm_sim.workflows.benchmark import (
    BenchmarkPolicy,
    BenchmarkResult,
    CookieBatchBenchmark,
    EpisodeScore,
    ExpertPolicyAdapter,
    make_policy_adapter,
)


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


def test_benchmark_terminates_after_five_consecutive_filled_steps():
    import mujoco

    benchmark = CookieBatchBenchmark()
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        for index, (x, y) in enumerate(env.TARGET_SLOT_CENTERS):
            env.set_cookie_pose(index, (float(x), float(y), 0.791))
        for _ in range(60):
            mujoco.mj_step(env.model, env.data)
        for expected in range(1, 6):
            _, _, terminated, _, info = env.step(env.DEPLOYMENT_HOME)
            assert info["success_hold_count"] == expected
            assert terminated == (expected == 5)
            assert info["success"] == (expected == 5)
    finally:
        env.close()


def test_benchmark_accepts_four_mm_slot_offset_with_strict_override_available():
    from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

    benchmark = CookieBatchBenchmark()
    env = benchmark.create_env(render_cameras=False)
    strict = A3CookieTransferEnv(render_cameras=False)
    try:
        for current in (env, strict):
            current.reset(seed=0, options={"randomize_cookies": False})
            for cookie_index in range(10):
                position = current.privileged_target_slot_world(
                    cookie_index, current._target_cookie_center_z
                )
                rotation = current.data.xmat[current._target_bin_body].reshape(3, 3)
                position += rotation @ np.array([0.0, 0.004, 0.0])
                current.set_cookie_pose(cookie_index, tuple(position))
            mask = (True,) * 10 + (False,) * 70
            slots = current._target_slot_occupancy(mask)
            assert all(index >= 0 for index in slots) == (current is env)
            if current is env:
                assert len(set(slots)) == 10
    finally:
        env.close()
        strict.close()
