from __future__ import annotations

import numpy as np
import pytest

from a3_dual_arm_sim.policies.act import (
    LEFT_ARM_ACTION_DIM,
    expand_left_arm_action,
    left_arm_hold_action,
    project_observation_value,
)
from a3_dual_arm_sim.config import CookieSceneConfig
from a3_dual_arm_sim.contracts import JOINT_ACTION_DIM


def test_hold_action_is_the_deployment_right_arm() -> None:
    expected = np.asarray(CookieSceneConfig().deployment_home[LEFT_ARM_ACTION_DIM:])
    np.testing.assert_allclose(left_arm_hold_action(), expected)


def test_expand_left_arm_action_pads_with_the_hold_pose() -> None:
    left = np.arange(LEFT_ARM_ACTION_DIM, dtype=np.float64) / 10
    action = expand_left_arm_action(left)
    assert action.shape == (JOINT_ACTION_DIM,)
    np.testing.assert_allclose(action[:LEFT_ARM_ACTION_DIM], left)
    np.testing.assert_allclose(action[LEFT_ARM_ACTION_DIM:], left_arm_hold_action())


def test_expand_left_arm_action_passes_bimanual_actions_through() -> None:
    value = np.linspace(-1.0, 1.0, JOINT_ACTION_DIM)
    np.testing.assert_allclose(expand_left_arm_action(value), value)


def test_expand_left_arm_action_rejects_other_dims_and_non_finite() -> None:
    with pytest.raises(ValueError, match="left-arm or 16D"):
        expand_left_arm_action(np.zeros(12))
    with pytest.raises(ValueError, match="NaN"):
        expand_left_arm_action(np.full(LEFT_ARM_ACTION_DIM, np.nan))
    with pytest.raises(ValueError, match="Right-arm hold"):
        expand_left_arm_action(np.zeros(LEFT_ARM_ACTION_DIM), hold=np.zeros(3))


def test_project_observation_value_trims_bimanual_state() -> None:
    state = np.arange(16, dtype=np.float32)
    projected = project_observation_value(state, LEFT_ARM_ACTION_DIM)
    assert projected.shape == (LEFT_ARM_ACTION_DIM,)
    np.testing.assert_allclose(projected, state[:LEFT_ARM_ACTION_DIM])

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    assert project_observation_value(image, None) is image
    assert project_observation_value(state, JOINT_ACTION_DIM) is state
