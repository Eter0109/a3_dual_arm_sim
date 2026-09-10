from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .contracts import LEFT_JOINTS, RIGHT_JOINTS, STATE, ActionMode, EpisodeContext
from .cookie_transfer import A3CookieTransferEnv
from .grasp import A3GraspEnv


class GraspPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    HOLD = auto()
    DONE = auto()


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
        lift_target[2] = (
            self.env.initial_object_z + self.lift_height_m + self.grasp_eef_offset_m
        )
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
            mujoco.mju_subQuat(
                rotation_error, self._target_quaternion, current_quaternion
            )
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
            np.max(np.abs(joints - self._waypoints[self.phase]))
            <= self.joint_tolerance_rad
        )

    def _advance(self, phase: GraspPhase) -> None:
        self.phase = phase
        self.phase_steps = 0

    def close(self) -> None:
        return None


@dataclass
class A3CookieTransferExpert:
    """Privileged joint-space expert transferring 10 cookies from source to tilted target bin."""

    env: A3CookieTransferEnv
    action_mode: ActionMode = "joint_position"
    max_joint_step_rad: float = 0.040
    max_gripper_step: float = 0.06

    def __post_init__(self) -> None:
        self.model = self.env.model
        self.data = self.env.data
        self._l_qpos = np.asarray(
            [self.env._qpos_ids[name] for name in LEFT_JOINTS], dtype=np.int32
        )
        self._r_qpos = np.asarray(
            [self.env._qpos_ids[name] for name in RIGHT_JOINTS], dtype=np.int32
        )
        self._l_site = self.env._eef_sites[0]
        self._tb_id = self.env._target_bin_body
        self._l_ranges = np.asarray(
            [self.model.jnt_range[self.env._joint_ids[name]] for name in LEFT_JOINTS]
        )

        rot_canonical = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
        )
        self._target_quat_canonical = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(self._target_quat_canonical, rot_canonical.reshape(-1))

        theta = np.deg2rad(10.0)
        R_world_target = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ]
        )
        rot_tilted = R_world_target @ rot_canonical
        self._target_quat_tilted = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(self._target_quat_tilted, rot_tilted.reshape(-1))

        self.q_branch_seed = np.array(
            [1.28, 1.119, -1.127, -0.999, -0.252, 0.478, -1.535]
        )
        self.q_r_hold = np.array(
            [-1.805, -1.499, 1.543, 0.54, -2.247, -0.058, 0.041]
        )
        self.actions: list[np.ndarray] = []
        self.step_idx: int = 0
        self.q_transit = self.q_branch_seed.copy()

    def _solve_l(self, pos: np.ndarray, quat: np.ndarray, q0: np.ndarray) -> np.ndarray:
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos[:]

        def residual(q: np.ndarray) -> np.ndarray:
            work.qpos[self._l_qpos] = q
            mujoco.mj_forward(self.model, work)
            cur_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(cur_quat, work.site_xmat[self._l_site])
            err = np.empty(3, dtype=np.float64)
            mujoco.mju_subQuat(err, quat, cur_quat)
            return np.r_[work.site_xpos[self._l_site] - pos, 0.3 * err, 0.005 * (q - q0)]

        res = least_squares(
            residual,
            np.clip(q0, self._l_ranges[:, 0] + 1e-3, self._l_ranges[:, 1] - 1e-3),
            bounds=(self._l_ranges[:, 0] + 1e-4, self._l_ranges[:, 1] - 1e-4),
            max_nfev=500,
        )
        return np.asarray(res.x, dtype=np.float64)

    def reset(self, context: EpisodeContext | None = None) -> None:
        self.actions = []
        self.step_idx = 0
        q_transit = self._solve_l(
            np.array([0.10, 0.30, 0.89]), self._target_quat_canonical, self.q_branch_seed
        )
        tb_pos = self.data.xpos[self._tb_id].copy()
        tb_mat = self.data.xmat[self._tb_id].reshape(3, 3).copy()

        transfer_pairs = [
            (0, -0.028, -0.048),
            (5, +0.028, -0.048),
            (1, -0.028, -0.024),
            (6, +0.028, -0.024),
            (2, -0.028,  0.000),
            (7, +0.028,  0.000),
            (3, -0.028, +0.024),
            (8, +0.028, +0.024),
            (9, +0.028, +0.048),
            (4, -0.028, +0.048),
        ]

        def add_segment(
            q_start: np.ndarray,
            q_end: np.ndarray,
            g_start: float,
            g_end: float,
            max_dq: float = 0.040,
            min_steps: int = 12,
        ) -> None:
            dq = float(np.max(np.abs(q_end - q_start)))
            dg = abs(g_end - g_start)
            steps = max(int(np.ceil(dq / max_dq)), int(np.ceil(dg / self.max_gripper_step)), min_steps)
            for a in np.linspace(0, 1, steps)[1:]:
                q_l = (1 - a) * q_start + a * q_end
                g_l = (1 - a) * g_start + a * g_end
                action = np.zeros(16, dtype=np.float64)
                action[:7] = q_l
                action[7] = g_l
                action[8:15] = self.q_r_hold
                action[15] = 0.6
                self.actions.append(action)

        def add_hold(q: np.ndarray, g: float, steps: int = 8) -> None:
            for _ in range(steps):
                action = np.zeros(16, dtype=np.float64)
                action[:7] = q
                action[7] = g
                action[8:15] = self.q_r_hold
                action[15] = 0.6
                self.actions.append(action)

        cur_q = q_transit
        cur_g = 0.28

        for cookie_idx, sx, sy in transfer_pairs:
            cid = self.env._cookie_bodies[cookie_idx]
            p_c = self.data.xpos[cid].copy()

            q_app = self._solve_l(
                np.array([p_c[0], p_c[1], 0.86]), self._target_quat_canonical, self.q_branch_seed
            )
            q_grasp = self._solve_l(
                np.array([p_c[0], p_c[1], p_c[2] - 0.008]), self._target_quat_canonical, q_app
            )
            q_lift = self._solve_l(
                np.array([p_c[0], p_c[1], 0.86]), self._target_quat_canonical, q_grasp
            )

            slot_place = tb_pos + tb_mat @ np.array([sx, sy, 0.020])
            slot_pre = slot_place + tb_mat @ np.array([0.0, 0.0, 0.06])

            q_pre = self._solve_l(slot_pre, self._target_quat_tilted, self.q_branch_seed)
            q_place = self._solve_l(slot_place, self._target_quat_tilted, q_pre)

            # Approach & Pick
            add_segment(cur_q, q_app, cur_g, 0.28)
            add_segment(q_app, q_grasp, 0.28, 0.28)
            add_segment(q_grasp, q_grasp, 0.28, 0.10, min_steps=12)
            add_hold(q_grasp, 0.10, steps=8)
            add_segment(q_grasp, q_lift, 0.10, 0.10)

            # Transit & Place
            add_segment(q_lift, q_transit, 0.10, 0.10)
            add_segment(q_transit, q_pre, 0.10, 0.10)
            add_segment(q_pre, q_place, 0.10, 0.10)

            # Release
            add_segment(q_place, q_place, 0.10, 0.25, min_steps=10)
            add_hold(q_place, 0.25, steps=5)

            # Retract
            add_segment(q_place, q_pre, 0.25, 0.25)
            add_segment(q_pre, q_transit, 0.25, 0.28)

            cur_q = q_transit
            cur_g = 0.28

        self.q_transit = q_transit

    def act(self, observation: dict[str, Any] | None = None, task: str = "") -> np.ndarray:
        if self.step_idx < len(self.actions):
            act = self.actions[self.step_idx]
            self.step_idx += 1
            return act
        hold_act = np.zeros(16, dtype=np.float64)
        hold_act[:7] = self.q_transit
        hold_act[7] = 0.28
        hold_act[8:15] = self.q_r_hold
        hold_act[15] = 0.6
        return hold_act

    def close(self) -> None:
        return None

