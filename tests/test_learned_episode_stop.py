from types import SimpleNamespace

import numpy as np

from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


class StableSuccessEnv:
    action_mode = "joint_position"
    SOURCE_POSITIONS = tuple((float(c), float(r)) for c in range(4) for r in range(20))

    def __init__(self, source_count=60):
        self.randomization_metadata = {}
        self.step_count = 0
        self.source_count = source_count

    def reset(self, **kwargs):
        self.step_count = 0
        return {}, {}

    def step(self, action):
        self.step_count += 1
        return (
            {},
            0,
            False,
            False,
            {
                "success": self.step_count >= 20,
                "cookies_in_target": 20 if self.step_count >= 10 else 19,
                "cookies_in_source": self.source_count,
                "applied_action": action,
            },
        )


class LearnedPolicy:
    action_mode = "joint_position"

    def act(self, observation, task):
        return np.zeros(16)


def test_neural_grid_stops_at_stable_success_not_first_full_count():
    result = make_cookie_benchmark("2x10", max_steps=40).run_episode(
        LearnedPolicy(), 1, env=StableSuccessEnv()
    )
    assert result.steps == 20 and result.success


def test_wrong_source_count_cannot_trigger_successful_early_stop():
    result = make_cookie_benchmark("2x10", max_steps=40).run_episode(
        LearnedPolicy(), 1, env=StableSuccessEnv(source_count=59)
    )
    assert result.steps == 40 and not result.success


def test_expert_finishes_retreat_and_explicit_stop_opt_out_is_respected():
    policy = LearnedPolicy()
    policy.expert = SimpleNamespace(batch_reports=[])
    benchmark = make_cookie_benchmark("2x10", max_steps=25)
    assert benchmark.run_episode(policy, 1, env=StableSuccessEnv()).steps == 25
    benchmark.stop_on_success = False
    assert benchmark.run_episode(LearnedPolicy(), 1, env=StableSuccessEnv()).steps == 25
