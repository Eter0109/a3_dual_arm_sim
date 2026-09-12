"""Tests for the cookie collection driver.

The wrapper decides when an episode ends, which halves the recorded length, so
its stopping rule is worth pinning down without running the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from a3_dual_arm_sim.collection import CookieCollectionPolicy
from a3_dual_arm_sim.expert import CookiePhase


@dataclass
class _StubExpert:
    """Stands in for the scripted expert, exposing only what the wrapper reads."""

    phase: CookiePhase = CookiePhase.APPROACH
    failed: bool = False
    resets: int = 0
    calls: int = 0
    closed: bool = False
    actions: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.resets += 1

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        self.calls += 1
        action = np.zeros(16)
        self.actions.append(action)
        return action

    def close(self) -> None:
        self.closed = True


def _policy(expert: _StubExpert, hold_steps: int = 3) -> CookieCollectionPolicy:
    return CookieCollectionPolicy(expert=expert, hold_steps=hold_steps)  # type: ignore[arg-type]


def test_reset_clears_the_hold_counter_and_delegates() -> None:
    expert = _StubExpert()
    policy = _policy(expert, hold_steps=5)

    policy.act({})
    policy.reset()

    assert expert.resets == 1
    assert policy.stop_requested is False


def test_no_stop_request_while_the_expert_is_working() -> None:
    expert = _StubExpert(phase=CookiePhase.APPROACH)
    policy = _policy(expert)

    for _ in range(10):
        policy.act({})

    assert policy.stop_requested is False
    assert expert.calls == 10


def test_stop_is_requested_after_the_hold_window_following_done() -> None:
    expert = _StubExpert(phase=CookiePhase.DONE)
    policy = _policy(expert, hold_steps=3)

    policy.act({})
    policy.act({})
    assert policy.stop_requested is False
    policy.act({})
    assert policy.stop_requested is True


def test_hold_counter_only_counts_steps_spent_in_done() -> None:
    expert = _StubExpert(phase=CookiePhase.APPROACH)
    policy = _policy(expert, hold_steps=2)

    policy.act({})
    expert.phase = CookiePhase.DONE
    policy.act({})
    assert policy.stop_requested is False
    # A step elsewhere in the state machine must not advance the hold window.
    expert.phase = CookiePhase.RETRACT
    policy.act({})
    assert policy.stop_requested is False
    expert.phase = CookiePhase.DONE
    policy.act({})
    assert policy.stop_requested is True


def test_failure_requests_a_stop_immediately() -> None:
    expert = _StubExpert(phase=CookiePhase.FAILED, failed=True)
    policy = _policy(expert, hold_steps=10)

    policy.act({})

    assert policy.stop_requested is True


def test_action_mode_matches_the_environment_contract() -> None:
    assert _policy(_StubExpert()).action_mode == "joint_position"


def test_action_is_forwarded_unchanged() -> None:
    expert = _StubExpert()
    policy = _policy(expert)

    action = policy.act({"anything": np.zeros(1)})

    assert action.shape == (16,)
    assert np.array_equal(action, expert.actions[0])


def test_close_delegates_to_the_expert() -> None:
    expert = _StubExpert()
    _policy(expert).close()
    assert expert.closed is True
