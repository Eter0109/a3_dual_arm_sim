from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import mujoco
import numpy as np
from scipy.optimize import least_squares

from a3_dual_arm_sim.contracts import LEFT_JOINTS, STATE, ActionMode, EpisodeContext
from a3_dual_arm_sim.tasks.grasp import A3GraspEnv

from .phases import GraspPhase


@dataclass
class A3GraspExpert:
    """Privileged joint-space expert with per-episode absolute IK waypoints."""

    env: A3GraspEnv
    action_mode: ActionMode = "joint_position"
    approach_height_m: float = 0.17
    grasp_eef_offset_m: float = -0.012
    lift_height_m: float = 0.18
    joint_tolerance_rad: float = 0.055
    transit_joint_step_rad: float = 0.025
    manipulation_joint_step_rad: float = 0.02
    closed_gripper_opening: float = 0.18
    close_steps: int = 60
    grasp_stable_steps: int = 8
    hold_steps: int = 12
    phase_timeout_steps: int = 180

    APPROACH_GUESS: ClassVar[np.ndarray] = np.asarray(
        [1.16, 1.06, -2.93, 0.056, 1.57, -0.55, -1.56], dtype=np.float64
    )
    GRASP_GUESS: ClassVar[np.ndarray] = np.asarray(
        [1.56, 1.02, -2.85, 0.0, 1.38, -0.545, -1.40], dtype=np.float64
    )

    def __post_init__(self) -> None:
        self.phase = GraspPhase.APPROACH
        self.phase_steps = 0
        self._grasp_stable_count = 0
        self._waypoints: dict[GraspPhase, np.ndarray] = {}
        self._qpos_ids = np.asarray(
            [self.env._qpos_ids[name] for name in LEFT_JOINTS], dtype=np.int32
        )
        self._joint_ranges = np.asarray(
            [self.env.model.jnt_range[self.env._joint_ids[name]] for name in LEFT_JOINTS]
        )
        self._site_id = self.env._eef_sites[0]
        target_rotation = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        self._target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(self._target_quaternion, target_rotation.reshape(-1))

    def reset(self, context: EpisodeContext) -> None:
        self.phase = GraspPhase.APPROACH
        self.phase_steps = 0
        self._grasp_stable_count = 0
        object_position = self.env.target_position
        approach_target = object_position + np.asarray([0.0, 0.0, self.approach_height_m])
        grasp_target = object_position + np.asarray([0.0, 0.0, self.grasp_eef_offset_m])
        lift_target = grasp_target.copy()
        lift_target[2] = self.env.initial_object_z + self.lift_height_m + self.grasp_eef_offset_m
        approach = self._solve(approach_target, self.APPROACH_GUESS)
        grasp = self._solve(grasp_target, self.GRASP_GUESS)
        lift = self._solve(lift_target, approach)
        self._waypoints = {
            GraspPhase.APPROACH: approach,
            GraspPhase.DESCEND: grasp,
            GraspPhase.CLOSE: grasp,
            GraspPhase.LIFT: lift,
            GraspPhase.HOLD: lift,
            GraspPhase.DONE: lift,
        }

    def _solve(self, target_position: np.ndarray, initial: np.ndarray) -> np.ndarray:
        work = mujoco.MjData(self.env.model)
        work.qpos[:] = self.env.data.qpos
        work.qvel[:] = 0.0

        def residual(qpos: np.ndarray) -> np.ndarray:
            work.qpos[self._qpos_ids] = qpos
            mujoco.mj_forward(self.env.model, work)
            current_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(current_quaternion, work.site_xmat[self._site_id])
            rotation_error = np.empty(3, dtype=np.float64)
            mujoco.mju_subQuat(rotation_error, self._target_quaternion, current_quaternion)
            return np.r_[
                work.site_xpos[self._site_id] - target_position,
                0.03 * rotation_error,
            ]

        result = least_squares(
            residual,
            np.clip(initial, self._joint_ranges[:, 0], self._joint_ranges[:, 1]),
            bounds=(self._joint_ranges[:, 0] + 1e-4, self._joint_ranges[:, 1] - 1e-4),
            max_nfev=600,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        position_error = np.linalg.norm(residual(result.x)[:3])
        if not result.success or position_error > 0.012:
            raise RuntimeError(
                f"grasp expert IK failed: success={result.success} "
                f"position_error={position_error:.5f}"
            )
        return np.asarray(result.x, dtype=np.float64)

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        self.phase_steps += 1
        state = np.asarray(observation[STATE], dtype=np.float64)
        action = state.copy()
        desired = self._waypoints[self.phase]
        command_step = (
            self.transit_joint_step_rad
            if self.phase is GraspPhase.APPROACH
            else self.manipulation_joint_step_rad
        )
        previous_command = self.env.last_applied_action[:7]
        action[:7] = previous_command + np.clip(
            desired - previous_command, -command_step, command_step
        )
        action[15] = 1.0
        action[7] = (
            1.0
            if self.phase in (GraspPhase.APPROACH, GraspPhase.DESCEND)
            else self.closed_gripper_opening
        )

        if self.phase is GraspPhase.APPROACH and self._reached(state[:7]):
            self._advance(GraspPhase.DESCEND)
        elif self.phase is GraspPhase.DESCEND and self._reached(state[:7]):
            self._advance(GraspPhase.CLOSE)
        elif self.phase is GraspPhase.CLOSE:
            stable = self.env.is_grasped()
            self._grasp_stable_count = self._grasp_stable_count + 1 if stable else 0
            if (
                self._grasp_stable_count >= self.grasp_stable_steps
                or self.phase_steps >= self.close_steps
            ):
                self._advance(GraspPhase.LIFT)
        elif self.phase is GraspPhase.LIFT:
            if self.env.success_hold_count > 0:
                self._advance(GraspPhase.HOLD)
            elif self.phase_steps >= self.phase_timeout_steps:
                self._advance(GraspPhase.DONE)
        elif self.phase is GraspPhase.HOLD and (
            self.env.success_hold_count >= self.env.task_config.success_hold_steps
            or self.phase_steps >= self.hold_steps
        ):
            self._advance(GraspPhase.DONE)

        if self.phase_steps >= self.phase_timeout_steps and self.phase in (
            GraspPhase.APPROACH,
            GraspPhase.DESCEND,
        ):
            self._advance(GraspPhase.DONE)
        return action

    def _reached(self, joints: np.ndarray) -> bool:
        return bool(
            np.max(np.abs(joints - self._waypoints[self.phase])) <= self.joint_tolerance_rad
        )

    def _advance(self, phase: GraspPhase) -> None:
        self.phase = phase
        self.phase_steps = 0

    def close(self) -> None:
        return None
