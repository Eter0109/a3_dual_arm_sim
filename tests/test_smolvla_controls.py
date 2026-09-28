from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.controllers.recovery import RecoveryPolicy
from a3_dual_arm_sim.policies.smolvla import (
    DEPLOYMENT_HOME_RIGHT,
    SmolVLAPolicyPlugin,
    validate_execution_horizon,
)


def plugin():
    p = SmolVLAPolicyPlugin.__new__(SmolVLAPolicyPlugin)
    p._torch = torch
    p._device = torch.device("cpu")
    p._input_keys = ("observation.state",)
    p._prepare_observation = lambda observation, *args: observation
    p._preprocessor = p._postprocessor = lambda x: x
    p._policy = SimpleNamespace(
        select_action=lambda _: torch.ones((1, 16)) * 0.3, reset=lambda: None
    )
    p._prev_action = None
    p.inference_seed = 17
    p.ema_alpha = 0
    p.anchor_right_arm = False
    p.gripper_sharpening = False
    p.align_vertical = False
    p.clamp_z = False
    p.env = None
    return p


def test_raw_actions_and_rule_switches():
    p = plugin()
    obs = {"observation.state": np.zeros(16)}
    assert np.allclose(p.act(obs, "task"), 0.3)
    p.anchor_right_arm = p.gripper_sharpening = True
    p.ema_alpha = 0.75
    p._prev_action = np.zeros(16)
    action = p.act(obs, "task")
    assert np.allclose(action[:7], 0.225)
    assert np.isclose(action[7], 0.2804)
    assert np.allclose(action[8:], DEPLOYMENT_HOME_RIGHT)
    assert np.allclose(p.last_raw_action, 0.3)


def test_left_only_ignores_right_state_and_holds_initial_right():
    p = plugin()
    p._action_dim = 8
    p._right_hold = None
    seen = []
    def select(batch):
        seen.append(np.asarray(batch['observation.state']).copy())
        return torch.ones((1, 8)) * 0.3
    p._policy.select_action = select
    state = np.arange(16, dtype=float)
    first = p.act({'observation.state': state}, 'task')
    changed = state.copy()
    changed[8:] += 50
    second = p.act({'observation.state': changed}, 'task')
    assert np.array_equal(seen[0], state[:8])
    assert np.array_equal(seen[0], seen[1])
    assert np.array_equal(first, second)
    assert np.array_equal(first[8:], state[8:])
    assert np.array_equal(state, np.arange(16))
    p.reset(EpisodeContext(seed=1, task='task', action_mode='joint_position'))
    assert p._right_hold is None


def test_reset_clears_history_and_reseeds():
    p = plugin()
    ctx = EpisodeContext(seed=1000, task="task", action_mode="joint_position")
    p.reset(ctx)
    first = torch.rand(8)
    p._prev_action = np.ones(16)
    p.last_raw_action = np.ones(16)
    p.reset(ctx)
    assert torch.equal(first, torch.rand(8))
    assert p._prev_action is None and p.last_raw_action is None


@pytest.mark.parametrize("height,corrected", [(0.7656, False), (0.74, True)])
def test_optional_height_floor_preserves_placement_and_works_without_rotation(height, corrected):
    from a3_dual_arm_sim.contracts import LEFT_JOINTS

    p = plugin()
    p.clamp_z = True
    p._target_quat_canonical = np.array([0.0, 1.0, 0.0, 0.0])
    work = SimpleNamespace(
        qpos=np.zeros(16),
        site_xpos=np.array([[0.0, 0.1, height]]),
        site_xmat=np.eye(3).reshape(1, 9),
    )
    calls = []

    def solve(pos, quat, **kwargs):
        calls.append((pos.copy(), quat.copy()))
        return np.full(7, 0.4)

    p.env = SimpleNamespace(
        _ik=SimpleNamespace(_work=work),
        _qpos_ids=dict(zip(LEFT_JOINTS, range(7))),
        data=SimpleNamespace(qpos=np.ones(16)),
        model=None,
        _eef_sites=[0],
        solve_ik=solve,
    )
    p._mujoco = SimpleNamespace(
        mj_kinematics=lambda *a: None,
        mju_mat2Quat=lambda out, mat: out.__setitem__(slice(None), [1.0, 0.0, 0.0, 0.0]),
    )
    result = p.act({"observation.state": np.zeros(16)}, "task")
    assert len(calls) == int(corrected)
    assert np.allclose(work.qpos[7:], 1)
    assert np.allclose(result[:7], 0.4 if corrected else 0.3)
    if corrected:
        assert calls[0][0][2] == 0.760
        assert np.allclose(calls[0][1], [1, 0, 0, 0])


@pytest.mark.parametrize("steps", [0, 17, -1, 1.5, True])
def test_invalid_execution_horizon(steps):
    with pytest.raises(ValueError):
        validate_execution_horizon(steps, 16)


def test_manual_takeover_marks_boundaries_and_keeps_action_interfaces():
    p = RecoveryPolicy(plugin())
    ctx = EpisodeContext(seed=1, task="task", action_mode="joint_position")
    p.reset(ctx)
    obs = {"observation.state": np.zeros(16)}
    assert p.act(obs, "task").shape == (16,)
    p.handle_key(ord("T"))
    assert p.act(obs, "task").shape == (14,)
    p.handle_key(ord("T"))
    assert p.act(obs, "task").shape == (16,)
    assert p.events == [{"step": 1, "controller": "human"}, {"step": 2, "controller": "model"}]
