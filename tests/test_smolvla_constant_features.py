from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a3_dual_arm_sim.policies.smolvla import (
    SmolVLAPolicyPlugin,
    constant_feature_values,
    inactive_right_arm_target,
)


def test_exact_range_not_rounded_std_identifies_constants():
    mask, values = constant_feature_values({"min": [0, 1], "max": [2, 1], "std": [0.2, 0.003]}, 2)
    assert mask.tolist() == [False, True]
    assert values.tolist() == [0, 1]
    with pytest.raises(ValueError):
        constant_feature_values({"min": [0], "max": [0]}, 2)


def test_right_gripper_rounding_cannot_change_left_prediction():
    p = SmolVLAPolicyPlugin.__new__(SmolVLAPolicyPlugin)
    p._torch = torch
    p._device = torch.device("cpu")
    p._action_dim = 16
    p._input_keys = ("observation.state",)
    p._prepare_observation = lambda obs, *args: obs
    p._preprocessor = p._postprocessor = lambda batch: batch
    p._inactive_right_hold = np.ones(8, dtype=np.float32)
    seen = []

    def select(batch):
        state = batch["observation.state"]
        seen.append(state.copy())
        result = np.full(16, (state[15] - 1) / 1e-8, dtype=np.float32)
        result[0] += state[0]
        return torch.from_numpy(result).reshape(1, 16)

    p._policy = SimpleNamespace(select_action=select)
    state = np.ones(16, dtype=np.float32)
    baseline = p.predict_action({"observation.state": state}, "task")
    changed = state.copy()
    changed[15] = np.nextafter(np.float32(1), np.float32(2))
    before = changed.copy()
    result = p.predict_action({"observation.state": changed}, "task")
    assert np.array_equal(result, baseline)
    assert np.array_equal(changed, before)
    assert np.array_equal(seen[-1][8:], np.ones(8))
    changed[0] = 2
    assert p.predict_action({"observation.state": changed}, "task")[0] == 2
    p.ema_alpha = 0
    p._prev_action = None
    p.anchor_right_arm = p.gripper_sharpening = p.align_vertical = p.clamp_z = False
    p.env = None
    assert np.array_equal(p.act({"observation.state": changed}, "task")[8:], np.ones(8))
    assert np.array_equal(p.last_raw_action[8:], np.zeros(8))
    p._inactive_right_hold = None
    assert abs(p.predict_action({"observation.state": before}, "task")[0] - baseline[0]) > 10


@pytest.mark.parametrize("key,index", [("action", 8), ("action", 15), ("observation.state", 14)])
def test_any_demonstrated_right_motion_disables_auto_hold(key, index):
    stats = {
        name: {"min": [0] * 8 + [1] * 8, "max": [2] * 8 + [1] * 8}
        for name in ("observation.state", "action")
    }
    assert np.array_equal(inactive_right_arm_target(stats, 16), np.ones(8))
    stats[key]["max"][index] += 0.01
    assert inactive_right_arm_target(stats, 16) is None


def test_different_constant_poses_and_left_only_model_are_not_auto_held():
    stats = {
        name: {"min": [0] * 8 + [1] * 8, "max": [2] * 8 + [1] * 8}
        for name in ("observation.state", "action")
    }
    assert inactive_right_arm_target(stats, 8) is None
    stats["action"]["min"][8] = stats["action"]["max"][8] = 2
    assert inactive_right_arm_target(stats, 16) is None
