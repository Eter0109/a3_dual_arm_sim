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


class CookiePhase(Enum):
    SELECT_COOKIE = auto()
    APPROACH = auto()
    ALIGN = auto()
    DESCEND = auto()
    CLOSE = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    VERIFY_LIFT = auto()
    MOVE_TO_SLOT = auto()
    DESCEND_TO_PLACE = auto()
    OPEN = auto()
    VERIFY_RELEASE = auto()
    RETRACT = auto()
    DONE = auto()
    FAILED = auto()


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
    """Feedback-driven privileged expert for the ten-Cookie transfer task.

    This is intentionally not a policy: it reads simulator-only truth and follows
    explicit state-machine rules.  Those privileged values are never exposed in
    the learning observation.
    """

    env: A3CookieTransferEnv
    action_mode: ActionMode = "joint_position"
    max_joint_step_rad: float = 0.040
    max_gripper_step: float = 0.060
    joint_tolerance_rad: float = 0.055
    open_gripper: float = 0.28
    closed_gripper: float = 0.10
    release_gripper: float = 0.25
    close_steps: int = 18
    grasp_stable_steps: int = 8
    release_stable_steps: int = 6
    settle_steps: int = 3
    phase_timeout_steps: int = 180
    max_retries: int = 2

    TOOL_SLOT_TARGETS_LOCAL: ClassVar[np.ndarray] = np.asarray(
        [
            (-0.028, -0.048),
            (+0.028, -0.048),
            (-0.028, -0.024),
            (+0.028, -0.024),
            (-0.028, 0.000),
            (+0.028, 0.000),
            (-0.028, +0.024),
            (+0.028, +0.024),
            (-0.028, +0.048),
            (+0.028, +0.048),
        ],
        dtype=np.float64,
    )

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
        self.q_transit = self.q_branch_seed.copy()
        self.phase = CookiePhase.SELECT_COOKIE
        self.phase_steps = 0
        self.current_cookie_index: int | None = None
        self.target_slot_index: int | None = None
        self.retry_counts = [0] * self.env.task_config.required_cookies
        self.completed_cookie_indices: list[int] = []
        self.skipped_cookie_indices: list[int] = []
        self.transition_history: list[CookiePhase] = []
        self.failure_reason: str | None = None
        self.failure_phase: str | None = None
        self.retry_reasons: list[str] = []
        self._waypoints: dict[str, np.ndarray] = {}
        self._grasp_stable_count = 0
        self._release_stable_count = 0
        self._settle_count = 0
        self._move_stage = "transit"
        self._retract_stage = "preplace"
        self._motion_actions: list[np.ndarray] = []
        self._motion_key: tuple[bytes, float, int] | None = None
        self._motion_done = False
        self._recovering = False
        self._initial_cookie_positions = np.asarray(
            [
                self.env.privileged_cookie_position(index)
                for index in range(self.env.task_config.cookie_count)
            ],
            dtype=np.float64,
        )
        self._initial_target_position = self.data.xpos[self._tb_id].copy()
        self._initial_target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
        self._cookie_preference = (0, 5, 1, 6, 2, 7, 3, 8, 9, 4) + tuple(
            range(10, self.env.task_config.cookie_count)
        )
        self._slot_order = (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)

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
        position_error = float(np.linalg.norm(residual(res.x)[:3]))
        if not res.success or position_error > 0.016:
            raise RuntimeError(
                "cookie expert IK failed: "
                f"success={res.success} position_error={position_error:.5f}"
            )
        return np.asarray(res.x, dtype=np.float64)

    def reset(self, context: EpisodeContext | None = None) -> None:
        del context
        self.q_transit = self._solve_l(
            np.array([0.10, 0.30, 0.89]), self._target_quat_canonical, self.q_branch_seed
        )
        self.phase = CookiePhase.SELECT_COOKIE
        self.phase_steps = 0
        self.current_cookie_index = None
        self.target_slot_index = None
        self.retry_counts = [0] * self.env.task_config.required_cookies
        self.completed_cookie_indices = []
        self.skipped_cookie_indices = []
        self.transition_history = [self.phase]
        self.failure_reason = None
        self.failure_phase = None
        self.retry_reasons = []
        self._waypoints = {}
        self._grasp_stable_count = 0
        self._release_stable_count = 0
        self._settle_count = 0
        self._move_stage = "transit"
        self._retract_stage = "preplace"
        self._reset_motion()
        self._recovering = False
        self._initial_cookie_positions = np.asarray(
            [
                self.env.privileged_cookie_position(index)
                for index in range(self.env.task_config.cookie_count)
            ],
            dtype=np.float64,
        )
        self._initial_target_position = self.data.xpos[self._tb_id].copy()
        self._initial_target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()

    def act(self, observation: dict[str, Any] | None = None, task: str = "") -> np.ndarray:
        del observation, task
        # SELECT_COOKIE has no physical action. Resolve it immediately so every
        # returned command belongs to a meaningful motion/verification phase.
        if self.phase is CookiePhase.SELECT_COOKIE:
            self._select_cookie()

        self.phase_steps += 1
        if self.phase in (CookiePhase.DONE, CookiePhase.FAILED):
            return self._hold_command(self.q_transit, self.open_gripper)

        if self.phase is CookiePhase.APPROACH:
            target = self.q_transit if self._recovering else self._waypoints["approach"]
            action = self._trajectory_command(target, self.open_gripper)
            if self._motion_done:
                if self._recovering:
                    self._recovering = False
                    self._reset_motion()
                else:
                    self._advance(CookiePhase.ALIGN)
            return action

        if self.phase is CookiePhase.ALIGN:
            self._advance(CookiePhase.DESCEND)

        if self.phase is CookiePhase.DESCEND:
            action = self._trajectory_command(
                self._waypoints["grasp"], self.open_gripper
            )
            if self._motion_done:
                self._advance(CookiePhase.CLOSE)
            return action

        if self.phase is CookiePhase.CLOSE:
            action = self._trajectory_command(
                self._waypoints["grasp"], self.closed_gripper
            )
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_GRASP)
            return action

        if self.phase is CookiePhase.VERIFY_GRASP:
            action = self._hold_command(
                self._waypoints["grasp"], self.closed_gripper
            )
            grasped = self._has_verified_grasp()
            self._grasp_stable_count = self._grasp_stable_count + 1 if grasped else 0
            if (
                self._grasp_stable_count >= self.grasp_stable_steps
                or self.phase_steps >= 8
            ):
                self._advance(CookiePhase.LIFT)
            return action

        if self.phase is CookiePhase.LIFT:
            action = self._trajectory_command(
                self._waypoints["lift"], self.closed_gripper
            )
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_LIFT)
            elif self._cookie_dropped():
                self._retry("Cookie dropped during lift")
            elif self._timed_out():
                self._retry("lift timed out")
            return action

        if self.phase is CookiePhase.VERIFY_LIFT:
            if self._has_verified_lift():
                self._advance(CookiePhase.MOVE_TO_SLOT)
            else:
                action = self._hold_command(
                    self._waypoints["lift"], self.closed_gripper
                )
                if self._timed_out(30):
                    self._retry("Cookie did not follow the gripper")
                return action

        if self.phase is CookiePhase.MOVE_TO_SLOT:
            target = (
                self.q_transit
                if self._move_stage == "transit"
                else self._waypoints["preplace"]
            )
            action = self._trajectory_command(target, self.closed_gripper)
            if self._cookie_dropped():
                self._retry("Cookie dropped during transfer")
            elif self._motion_done:
                if self._move_stage == "transit":
                    self._move_stage = "preplace"
                    self.phase_steps = 0
                    self._reset_motion()
                else:
                    self._advance(CookiePhase.DESCEND_TO_PLACE)
            return action

        if self.phase is CookiePhase.DESCEND_TO_PLACE:
            action = self._trajectory_command(
                self._waypoints["place"], self.closed_gripper
            )
            if self._motion_done:
                self._advance(CookiePhase.OPEN)
            elif self._cookie_dropped():
                self._retry("Cookie released before reaching its slot")
            elif self._timed_out():
                self._retry("placement descent timed out")
            return action

        if self.phase is CookiePhase.OPEN:
            action = self._trajectory_command(
                self._waypoints["place"], self.release_gripper, min_steps=10
            )
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_RELEASE)
            return action

        if self.phase is CookiePhase.VERIFY_RELEASE:
            action = self._hold_command(
                self._waypoints["place"], self.release_gripper
            )
            contacts = self.env.privileged_left_finger_contacts(
                self._current_cookie()
            )
            # A released thin Cookie may still brush one finger while it settles;
            # only bilateral contact means it is still pinched by the gripper.
            released = not all(contacts)
            self._release_stable_count = (
                self._release_stable_count + 1 if released else 0
            )
            if self.phase_steps >= 5:
                self._advance(CookiePhase.RETRACT)
            return action

        if self.phase is CookiePhase.RETRACT:
            target = (
                self._waypoints["preplace"]
                if self._retract_stage == "preplace"
                else self.q_transit
            )
            gripper = (
                self.release_gripper
                if self._retract_stage == "preplace"
                else self.open_gripper
            )
            action = self._trajectory_command(target, gripper)
            # Packing is sequential: later Cookies compact earlier ones into the
            # final 2x5 slots. At this intermediate point require stable target-
            # bin containment; the environment performs strict slot validation
            # on the finished arrangement.
            placed = self.env.privileged_cookie_in_target(self._current_cookie())
            self._settle_count = self._settle_count + 1 if placed else 0
            if self._motion_done:
                if self._retract_stage == "preplace":
                    self._retract_stage = "transit"
                    self.phase_steps = 0
                    self._reset_motion()
                else:
                    if not self.env.privileged_cookie_in_target_region(
                        self._current_cookie()
                    ):
                        self._retry("Cookie left the target bin after release")
                        return action
                    if self._current_cookie() not in self.completed_cookie_indices:
                        self.completed_cookie_indices.append(self._current_cookie())
                    self._advance(CookiePhase.SELECT_COOKIE)
            return action

        raise RuntimeError(f"unhandled Cookie expert phase: {self.phase}")

    @property
    def finished(self) -> bool:
        return self.phase in (CookiePhase.DONE, CookiePhase.FAILED)

    @property
    def failed(self) -> bool:
        return self.phase is CookiePhase.FAILED

    @property
    def status(self) -> str:
        cookie = "-" if self.current_cookie_index is None else str(self.current_cookie_index)
        slot = "-" if self.target_slot_index is None else str(self.target_slot_index)
        return f"phase={self.phase.name} cookie={cookie} slot={slot}"

    def _select_cookie(self) -> None:
        transfer_index = len(self.completed_cookie_indices)
        if transfer_index >= self.env.task_config.required_cookies:
            self.current_cookie_index = None
            self.target_slot_index = None
            self._advance(CookiePhase.DONE)
            return
        self.target_slot_index = self._slot_order[transfer_index]
        last_error: RuntimeError | None = None
        for cookie_index in self._cookie_preference:
            if cookie_index in self.completed_cookie_indices:
                continue
            if cookie_index in self.skipped_cookie_indices:
                continue
            if not self.env.privileged_cookie_in_source(cookie_index):
                continue
            self.current_cookie_index = cookie_index
            try:
                self._plan_current_cookie()
            except RuntimeError as exc:
                last_error = exc
                continue
            self._advance(CookiePhase.APPROACH)
            return
        self._fail(
            "no reachable Cookie remains in the source bin"
            + (f": {last_error}" if last_error is not None else "")
        )

    def _plan_current_cookie(self) -> None:
        # Preserve the proven example trajectory for each first attempt.  If a
        # verification fails, re-plan from the live simulator state instead.
        first_attempt = self.retry_counts[self._current_slot()] == 0
        cookie_position = (
            self._initial_cookie_positions[self._current_cookie()].copy()
            if first_attempt
            else self.env.privileged_cookie_position(self._current_cookie())
        )
        approach_position = np.array(
            [cookie_position[0], cookie_position[1], 0.86], dtype=np.float64
        )
        grasp_position = cookie_position.copy()
        grasp_position[2] -= 0.008
        lift_position = approach_position.copy()
        target_position = (
            self._initial_target_position.copy()
            if first_attempt
            else self.data.xpos[self._tb_id].copy()
        )
        target_rotation = (
            self._initial_target_rotation.copy()
            if first_attempt
            else self.data.xmat[self._tb_id].reshape(3, 3).copy()
        )
        tool_xy = self.TOOL_SLOT_TARGETS_LOCAL[self._current_slot()]
        slot_place = target_position + target_rotation @ np.asarray(
            [tool_xy[0], tool_xy[1], 0.020]
        )
        slot_pre = target_position + target_rotation @ np.asarray(
            [tool_xy[0], tool_xy[1], 0.080]
        )
        q_approach = self._solve_l(
            approach_position, self._target_quat_canonical, self.q_branch_seed
        )
        q_grasp = self._solve_l(
            grasp_position, self._target_quat_canonical, q_approach
        )
        q_lift = self._solve_l(lift_position, self._target_quat_canonical, q_grasp)
        q_pre = self._solve_l(
            slot_pre, self._target_quat_tilted, self.q_branch_seed
        )
        q_place = self._solve_l(slot_place, self._target_quat_tilted, q_pre)
        self._waypoints = {
            "approach": q_approach,
            "grasp": q_grasp,
            "lift": q_lift,
            "preplace": q_pre,
            "place": q_place,
        }

    def _trajectory_command(
        self, target: np.ndarray, gripper: float, *, min_steps: int = 12
    ) -> np.ndarray:
        key = (target.tobytes(), float(gripper), min_steps)
        if self._motion_key != key:
            start = self.env.last_applied_action.copy()
            joint_delta = float(np.max(np.abs(target - start[:7])))
            gripper_delta = abs(gripper - float(start[7]))
            points = max(
                int(np.ceil((joint_delta - 1e-9) / self.max_joint_step_rad)),
                int(np.ceil((gripper_delta - 1e-9) / self.max_gripper_step)),
                min_steps,
            )
            self._motion_actions = []
            for alpha in np.linspace(0.0, 1.0, points)[1:]:
                action = np.zeros(16, dtype=np.float64)
                action[:7] = (1.0 - alpha) * start[:7] + alpha * target
                action[7] = (1.0 - alpha) * start[7] + alpha * gripper
                action[8:15] = self.q_r_hold
                action[15] = 0.6
                self._motion_actions.append(action)
            self._motion_key = key
            self._motion_done = not self._motion_actions
        action = (
            self._motion_actions.pop(0)
            if self._motion_actions
            else self._hold_command(target, gripper)
        )
        self._motion_done = not self._motion_actions
        return action

    def _hold_command(self, target: np.ndarray, gripper: float) -> np.ndarray:
        action = np.zeros(16, dtype=np.float64)
        action[:7] = target
        action[7] = gripper
        action[8:15] = self.q_r_hold
        action[15] = 0.6
        return action

    def _reset_motion(self) -> None:
        self._motion_actions = []
        self._motion_key = None
        self._motion_done = False

    def _reached(self, target: np.ndarray, *, tolerance: float | None = None) -> bool:
        joints = self.env.current_joint_action[:7]
        return bool(
            np.max(np.abs(joints - target))
            <= (self.joint_tolerance_rad if tolerance is None else tolerance)
        )

    def _has_verified_grasp(self) -> bool:
        contacts = self.env.privileged_left_finger_contacts(self._current_cookie())
        touches = self.env.privileged_left_touch_values()
        return all(contacts) and all(value >= 0.02 for value in touches)

    def _has_verified_lift(self) -> bool:
        cookie_position = self.env.privileged_cookie_position(self._current_cookie())
        eef_position = self.data.site_xpos[self._l_site]
        follows_gripper = np.linalg.norm(cookie_position - eef_position) <= 0.065
        lifted = cookie_position[2] >= self.env.SOURCE_FLOOR_TOP_Z + 0.055
        return bool(lifted and follows_gripper)

    def _cookie_dropped(self) -> bool:
        cookie_position = self.env.privileged_cookie_position(self._current_cookie())
        eef_position = self.data.site_xpos[self._l_site]
        # Contact flags can flicker while a thin object remains securely held.
        # Separation from the end effector is the robust transfer-time signal.
        separated = np.linalg.norm(cookie_position - eef_position) > 0.085
        fell_back = cookie_position[2] <= self.env.SOURCE_FLOOR_TOP_Z + 0.025
        return bool(self.phase_steps > 5 and (separated or fell_back))

    def _retry(self, reason: str) -> None:
        self.retry_reasons.append(
            f"{self.phase.name}: Cookie {self._current_cookie()}, "
            f"slot {self._current_slot()}: {reason}"
        )
        slot = self._current_slot()
        self.retry_counts[slot] += 1
        if self.retry_counts[slot] > self.max_retries:
            self.skipped_cookie_indices.append(self._current_cookie())
            self._advance(CookiePhase.SELECT_COOKIE)
            return
        self._grasp_stable_count = 0
        self._release_stable_count = 0
        self._settle_count = 0
        self._move_stage = "transit"
        self._recovering = True
        try:
            self._plan_current_cookie()
        except RuntimeError as exc:
            self._fail(str(exc))
            return
        self._advance(CookiePhase.APPROACH)

    def _advance(self, phase: CookiePhase) -> None:
        self.phase = phase
        self.phase_steps = 0
        self._reset_motion()
        self.transition_history.append(phase)
        if phase is CookiePhase.MOVE_TO_SLOT:
            self._move_stage = "transit"
        elif phase is CookiePhase.VERIFY_GRASP:
            self._grasp_stable_count = 0
        elif phase is CookiePhase.VERIFY_RELEASE:
            self._release_stable_count = 0
        elif phase is CookiePhase.RETRACT:
            self._settle_count = 0
            self._retract_stage = "preplace"

    def _fail(self, reason: str) -> None:
        self.failure_phase = self.phase.name
        self.failure_reason = reason
        self._advance(CookiePhase.FAILED)

    def _timed_out(self, limit: int | None = None) -> bool:
        return self.phase_steps >= (limit or self.phase_timeout_steps)

    def _current_cookie(self) -> int:
        if self.current_cookie_index is None:
            raise RuntimeError("Cookie expert has no active Cookie")
        return self.current_cookie_index

    def _current_slot(self) -> int:
        if self.target_slot_index is None:
            raise RuntimeError("Cookie expert has no active target slot")
        return self.target_slot_index

    def close(self) -> None:
        return None


def make_cookie_transfer_expert(env: A3CookieTransferEnv) -> A3CookieTransferExpert:
    """CLI factory: ``--policy a3_dual_arm_sim.expert:make_cookie_transfer_expert``."""

    return A3CookieTransferExpert(env)
