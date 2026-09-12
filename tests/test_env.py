from __future__ import annotations

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
)
from a3_dual_arm_sim.env import A3DualArmEnv


def test_reset_returns_complete_fixed_observation_and_is_deterministic() -> None:
    env = A3DualArmEnv()
    try:
        first, _ = env.reset(seed=17)
        first_qpos = env.data.qpos.copy()
        second, _ = env.reset(seed=17)
        np.testing.assert_allclose(env.data.qpos, first_qpos)
        for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE):
            assert first[key].shape == (256, 256, 3)
            assert first[key].dtype == np.uint8
            np.testing.assert_allclose(first[key], second[key], atol=1)
        assert first[STATE].shape == (16,)
        assert first[VELOCITY].shape == (16,)
        assert first[EEF_POSE].shape == (14,)
        assert first[FORCE].shape == (18,)
    finally:
        env.close()


def test_joint_action_is_clipped_to_rate_and_joint_limits() -> None:
    env = A3DualArmEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        before = env.last_applied_action
        requested = np.full(16, 100.0)
        _, _, terminated, _, info = env.step(requested)
        applied = info["applied_action"]
        arm_delta = np.r_[applied[:7] - before[:7], applied[8:15] - before[8:15]]
        assert np.max(np.abs(arm_delta)) <= env.config.max_joint_step_rad + 1e-12
        assert applied[7] <= before[7] + env.config.max_gripper_step + 1e-12
        assert applied[15] <= before[15] + env.config.max_gripper_step + 1e-12
        assert not terminated
    finally:
        env.close()


def test_disabled_policy_cameras_return_black_frames_without_renderer() -> None:
    env = A3DualArmEnv(render_cameras=False)
    try:
        observation, info = env.reset(seed=0)
        assert not info["policy_camera_rendering"]
        assert env._renderer is None
        for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE):
            assert observation[key].shape == (256, 256, 3)
            assert observation[key].dtype == np.uint8
            assert not np.any(observation[key])
        observation, _, _, _, _ = env.step(env.current_joint_action)
        assert env._renderer is None
        assert not np.any(observation[FRONT_IMAGE])
    finally:
        env.close()


def test_cartesian_adapter_changes_selected_arm_and_stays_finite() -> None:
    env = A3DualArmEnv(action_mode="cartesian_delta", render_cameras=False)
    try:
        env.reset(seed=0)
        before = env.current_joint_action
        command = np.zeros(14)
        command[0] = 0.5
        command[6] = command[13] = 1.0
        canonical = env._ik.convert(env.data, command)
        assert np.linalg.norm(canonical[:7] - before[:7]) > 1e-5
        np.testing.assert_allclose(canonical[8:15], before[8:15], atol=1e-9)
        assert np.all(np.isfinite(canonical))
    finally:
        env.close()


@pytest.mark.parametrize(("side", "object_index", "force_slice"), [("L", 0, slice(12, 14)), ("R", 1, slice(14, 16))])
def test_each_gripper_reports_touch_and_actuator_force(
    side: str, object_index: int, force_slice: slice
) -> None:
    env = A3DualArmEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        env.model.opt.gravity[:] = 0.0
        touch_sites = [
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_finger_{finger}_touch")
            for finger in ("inner", "outer")
        ]
        center = np.mean(env.data.site_xpos[touch_sites], axis=0)
        object_joint = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{object_index}_free"
        )
        qpos_address = env.model.jnt_qposadr[object_joint]
        dof_address = env.model.jnt_dofadr[object_joint]
        env.data.qpos[qpos_address : qpos_address + 3] = center
        env.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        env.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
        command = env.current_joint_action
        command[7 if side == "L" else 15] = 0.0
        maximum_touch = np.zeros(2)
        observation = None
        for _ in range(30):
            observation, _, terminated, _, _ = env.step(command)
            assert not terminated
            maximum_touch = np.maximum(maximum_touch, observation[FORCE][force_slice])
        assert observation is not None
        assert np.all(maximum_touch > 0.0)
        actuator_force_index = 16 if side == "L" else 17
        assert abs(float(observation[FORCE][actuator_force_index])) > 1e-4
        assert np.all(np.isfinite(observation[FORCE][:12]))
    finally:
        env.close()


def test_emergency_stop_terminates_without_applying_requested_motion() -> None:
    env = A3DualArmEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        before = env.current_joint_action
        env.emergency_stop("test")
        _, _, terminated, _, info = env.step(np.zeros(16))
        assert terminated
        assert info["safety_reason"] == "test"
        np.testing.assert_allclose(info["applied_action"], before)
    finally:
        env.close()
