from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .box_support import BoxSupportController
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
    SUPPORT_BOX = auto()
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


# How far below the Cookie's top face the jaws are meant to pinch.  Only this
# depth is a real design choice; the rest of the offset follows
# ``scene.cookie_half_size_m[2]`` (see ``_pinch_offset_from_cookie_centre``).
_PINCH_DEPTH_BELOW_TOP_M = 0.002


def _pinch_offset_from_cookie_centre(half_size_z: float) -> float:
    """Offset from a Cookie's centre up to the height where the jaws pinch.

    The jaws grip just below the Cookie's top face, so the offset has to be
    derived from ``scene.cookie_half_size_m[2]`` -- it is *not* a free tuning
    knob.  Whenever that half-size changes, this offset changes with it; keeping
    them in sync is what makes the tool reach the Cookie instead of closing on
    empty space above it.

    This used to be the bare literal ``0.023``.  That was only correct while the
    Cookie was 50 mm tall (half-size 25 mm, so the jaws sat 2 mm below the top
    face).  When the Cookie was shortened to 25 mm tall the same literal aimed
    the jaws ~10.5 mm *above* the top face, i.e. at empty space rather than at
    the Cookie.
    """
    return float(half_size_z) - _PINCH_DEPTH_BELOW_TOP_M


@dataclass
class A3CookieTransferExpert:
    """Feedback-driven privileged expert for the ten-Cookie transfer task.

    This is intentionally not a policy: it reads simulator-only truth and follows
    explicit state-machine rules.  Those privileged values are never exposed in
    the learning observation.
    """

    env: A3CookieTransferEnv
    action_mode: ActionMode = "joint_position"
    max_joint_step_rad: float = 0.020
    max_gripper_step: float = 0.060
    joint_tolerance_rad: float = 0.055
    open_gripper: float = 0.28
    closed_gripper: float = 0.0
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

        self.q_branch_seed = np.array([1.28, 1.119, -1.127, -0.999, -0.252, 0.478, -1.535])
        self.q_r_hold = np.array([-1.805, -1.499, 1.543, 0.54, -2.247, -0.058, 0.041])
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
        self._cookie_preference = tuple(
            sorted(
                range(self.env.task_config.cookie_count),
                key=lambda i: (self.env.SOURCE_POSITIONS[i][1], self.env.SOURCE_POSITIONS[i][0]),
            )
        )
        self._slot_order = (8, 9, 6, 7, 4, 5, 2, 3, 0, 1)

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
            return np.r_[work.site_xpos[self._l_site] - pos, 0.12 * err, 0.001 * (q - q0)]

        res = least_squares(
            residual,
            np.clip(q0, self._l_ranges[:, 0] + 1e-3, self._l_ranges[:, 1] - 1e-3),
            bounds=(self._l_ranges[:, 0] + 1e-4, self._l_ranges[:, 1] - 1e-4),
            max_nfev=500,
        )
        position_error = float(np.linalg.norm(residual(res.x)[:3]))
        if position_error > 0.012:
            rng = np.random.default_rng(0)
            for seed in [
                self.data.qpos[self._l_qpos],
                self.env.DEPLOYMENT_HOME[:7],
                *rng.uniform(self._l_ranges[:, 0], self._l_ranges[:, 1], (4, 7)),
            ]:
                candidate = least_squares(
                    residual,
                    np.clip(seed, self._l_ranges[:, 0] + 1e-3, self._l_ranges[:, 1] - 1e-3),
                    bounds=(self._l_ranges[:, 0] + 1e-4, self._l_ranges[:, 1] - 1e-4),
                    max_nfev=300,
                )
                if np.linalg.norm(candidate.fun[:6]) < np.linalg.norm(res.fun[:6]):
                    res = candidate
                position_error = float(np.linalg.norm(residual(res.x)[:3]))
                if position_error < 0.012:
                    break
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
        self._packed_positions: dict[int, np.ndarray] = {}
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
        self.box_support = BoxSupportController(self.env)
        self.phase = CookiePhase.SUPPORT_BOX
        self.transition_history = [self.phase]
        self.right_gripper = 1.0

    def act(self, observation: dict[str, Any] | None = None, task: str = "") -> np.ndarray:
        del observation, task
        if self.phase is CookiePhase.SUPPORT_BOX:
            action = self.box_support.act()
            if self.box_support.failed:
                self._fail(self.box_support.failed)
            elif self.box_support.ready:
                self.q_r_hold = action[8:15].copy()
                self.right_gripper = action[15]
                self._initial_target_position = self.data.xpos[self._tb_id].copy()
                self._initial_target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
                rot_canonical = np.empty(9)
                mujoco.mju_quat2Mat(rot_canonical, self._target_quat_canonical)
                mujoco.mju_mat2Quat(
                    self._target_quat_tilted,
                    (self._initial_target_rotation @ rot_canonical.reshape(3, 3)).ravel(),
                )
                self._advance(CookiePhase.SELECT_COOKIE)
            return action
        if self.phase not in (CookiePhase.DONE, CookiePhase.FAILED) and not np.all(
            self.box_support.contact_forces() > 0.1
        ):
            self._fail("right gripper lost box support")
        if self.phase not in (CookiePhase.DONE, CookiePhase.FAILED):
            for index, placed_position in self._packed_positions.items():
                displacement = np.linalg.norm(
                    self.env.privileged_cookie_target_position(index) - placed_position
                )
                rotation = (
                    self.data.xmat[self._tb_id].reshape(3, 3).T
                    @ self.data.xmat[self.env._cookie_bodies[index]].reshape(3, 3)
                )
                if displacement > 0.006 or abs(rotation[2, 2]) < np.cos(np.deg2rad(12)):
                    self._fail(f"packed Cookie {index} disturbed; holding position")
                    break
        # SELECT_COOKIE has no physical action. Resolve it immediately so every
        # returned command belongs to a meaningful motion/verification phase.
        if self.phase is CookiePhase.SELECT_COOKIE:
            self._select_cookie()

        self.phase_steps += 1
        if self.phase is CookiePhase.FAILED:
            return self.env.last_applied_action.copy()
        if self.phase is CookiePhase.DONE:
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
            action = self._trajectory_command(self._waypoints["grasp"], self.open_gripper)
            if self._motion_done:
                self._advance(CookiePhase.CLOSE)
            return action

        if self.phase is CookiePhase.CLOSE:
            action = self._trajectory_command(self._waypoints["grasp"], self.closed_gripper)
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_GRASP)
            return action

        if self.phase is CookiePhase.VERIFY_GRASP:
            action = self._hold_command(self._waypoints["grasp"], self.closed_gripper)
            grasped = self._has_verified_grasp()
            self._grasp_stable_count = self._grasp_stable_count + 1 if grasped else 0
            if self._grasp_stable_count >= self.grasp_stable_steps:
                self._advance(CookiePhase.LIFT)
            elif self._timed_out(30):
                self._retry("bilateral cookie grasp was not verified")
            return action

        if self.phase is CookiePhase.LIFT:
            action = self._trajectory_command(self._waypoints["lift"], self.closed_gripper)
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_LIFT)
            elif self._cookie_dropped():
                self._retry("Cookie dropped during lift")
            elif self._timed_out():
                self._retry("lift timed out")
            return action

        if self.phase is CookiePhase.VERIFY_LIFT:
            if self._has_verified_lift():
                try:
                    self._plan_place_from_grasp()
                except RuntimeError as exc:
                    self._fail(f"held Cookie placement IK: {exc}")
                    return self._hold_command(self._waypoints["lift"], self.closed_gripper)
                self._advance(CookiePhase.MOVE_TO_SLOT)
            else:
                action = self._hold_command(self._waypoints["lift"], self.closed_gripper)
                if self._timed_out(30):
                    self._retry("Cookie did not follow the gripper")
                return action

        if self.phase is CookiePhase.MOVE_TO_SLOT:
            target = (
                self.q_transit if self._move_stage == "transit" else self._waypoints["preplace"]
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
            action = self.env.last_applied_action.copy()
            action[7] = self.closed_gripper
            action[8:15] = self.q_r_hold
            action[15] = self.right_gripper
            local_position = self.env.privileged_cookie_target_position(self._current_cookie())
            target_xy = np.asarray(self.env.TARGET_SLOTS_LOCAL[self._current_slot()])
            body = self.env._cookie_bodies[self._current_cookie()]
            box_rotation = self.data.xmat[self._tb_id].reshape(3, 3)
            cookie_rotation = self.data.xmat[body].reshape(3, 3)
            relative_rotation = box_rotation.T @ cookie_rotation
            position_error = box_rotation @ (
                np.r_[target_xy, self._placement_height] - local_position
            )
            rotation_error = (
                sum(np.cross(cookie_rotation[:, i], box_rotation[:, i]) for i in range(3)) * 0.5
            )
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._l_site)
            dofs = [self.env._dof_ids[name] for name in LEFT_JOINTS]
            jac = np.vstack((jacp[:, dofs], jacr[:, dofs]))
            dq = jac.T @ np.linalg.solve(
                jac @ jac.T + 0.02**2 * np.eye(6), np.r_[position_error, rotation_error] * 0.15
            )
            action[:7] = self.data.qpos[self._l_qpos] + np.clip(dq, -0.008, 0.008)
            self._waypoints["place"] = action[:7].copy()
            aligned = (
                np.linalg.norm(local_position[:2] - target_xy) < 0.004
                and abs(relative_rotation[2, 2]) > np.cos(np.deg2rad(8))
                and abs(local_position[2] - self._placement_height) < 0.004
            )
            self._settle_count = self._settle_count + 1 if aligned else 0
            if self._settle_count >= 8:
                self._advance(CookiePhase.OPEN)
            elif self._cookie_dropped():
                self._fail("Cookie released before reaching its slot; holding position")
                return self.env.last_applied_action.copy()
            elif self._timed_out():
                self._fail(
                    "placement descent timed out; holding position to protect packed Cookies"
                )
                return self.env.last_applied_action.copy()
            return action

        if self.phase is CookiePhase.OPEN:
            action = self._trajectory_command(
                self._waypoints["place"], self.release_gripper, min_steps=10
            )
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_RELEASE)
            return action

        if self.phase is CookiePhase.VERIFY_RELEASE:
            action = self._hold_command(self._waypoints["place"], self.release_gripper)
            contacts = self.env.privileged_left_finger_contacts(self._current_cookie())
            # A released thin Cookie may still brush one finger while it settles;
            # only bilateral contact means it is still pinched by the gripper.
            released = not all(contacts)
            self._release_stable_count = self._release_stable_count + 1 if released else 0
            if self.phase_steps >= 5:
                self._advance(CookiePhase.RETRACT)
            return action

        if self.phase is CookiePhase.RETRACT:
            target = (
                self._waypoints["preplace"] if self._retract_stage == "preplace" else self.q_transit
            )
            gripper = (
                self.release_gripper if self._retract_stage == "preplace" else self.open_gripper
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
                    if self._settle_count < 20:
                        if self._timed_out(100):
                            self._fail(
                                "released Cookie did not remain upright and stable in target"
                            )
                        return action
                    if self._current_cookie() not in self.completed_cookie_indices:
                        self.completed_cookie_indices.append(self._current_cookie())
                        self._packed_positions[self._current_cookie()] = (
                            self.env.privileged_cookie_target_position(self._current_cookie()).copy()
                        )
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
            cookie_rotation = self.data.xmat[self.env._cookie_bodies[cookie_index]].reshape(3, 3)
            if abs(cookie_rotation[2, 2]) < np.cos(np.deg2rad(12)):
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

    def _grasp_height_offset(self) -> float:
        """Height above the Cookie centre at which the jaws should close.

        Read from ``COOKIE_HALF_SIZE`` on every call instead of cached at
        construction time, because the environment's scene config -- and hence
        the Cookie size -- is what this must track.  Both ``grasp_position`` and
        the preliminary reachability ``tool_height`` in ``_plan_current_cookie``
        must use this same value.
        """
        return _pinch_offset_from_cookie_centre(self.env.COOKIE_HALF_SIZE[2])

    def _plan_current_cookie(self) -> None:
        cookie_position = self.env.privileged_cookie_position(self._current_cookie()).copy()
        approach_position = np.array(
            [cookie_position[0], cookie_position[1], 0.86], dtype=np.float64
        )
        grasp_position = cookie_position.copy()
        # Pinch the upper edge: deep grasps collide with adjacent upright
        # cookies when lowering into tightly spaced rows.  The offset is derived
        # from the Cookie height, not hard-coded, so it stays correct when
        # ``COOKIE_HALF_SIZE[2]`` changes; see ``_pinch_offset_from_cookie_centre``.
        grasp_position[2] += self._grasp_height_offset()
        lift_position = approach_position.copy()
        target_position = self.data.xpos[self._tb_id].copy()
        target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
        canonical_rotation = np.empty(9)
        mujoco.mju_quat2Mat(canonical_rotation, self._target_quat_canonical)
        mujoco.mju_mat2Quat(
            self._target_quat_tilted, (target_rotation @ canonical_rotation.reshape(3, 3)).ravel()
        )
        tool_xy = self.env.TARGET_SLOTS_LOCAL[self._current_slot()]
        # Preliminary reachability must use the same top-edge grasp height as
        # the post-lift, measured-transform planner below, so it has to take the
        # grasp offset from the same source rather than repeating a literal.
        tool_height = (
            self.env.config.cookie_transfer.target_floor_z_m
            + self.env.config.cookie_transfer.bin_wall_thickness_m
            + self.env.COOKIE_HALF_SIZE[2]
            + 0.008
            + self._grasp_height_offset()
        )
        slot_place = target_position + target_rotation @ np.asarray(
            [tool_xy[0], tool_xy[1], tool_height]
        )
        slot_pre = slot_place + target_rotation[:, 2] * 0.07
        q_approach = self._solve_l(
            approach_position, self._target_quat_canonical, self.q_branch_seed
        )
        q_grasp = self._solve_l(grasp_position, self._target_quat_canonical, q_approach)
        q_lift = self._solve_l(lift_position, self._target_quat_canonical, q_grasp)
        place_quat = self._target_quat_tilted.copy()
        try:
            q_pre = self._solve_l(slot_pre, place_quat, self.q_branch_seed)
        except RuntimeError:
            # A rectangular cookie and parallel jaws permit a half-turn about
            # the tool axis, which can avoid a wrist-limit IK branch.
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, place_quat)
            mujoco.mju_mat2Quat(
                place_quat, (rotation.reshape(3, 3) @ np.diag([-1.0, 1.0, -1.0])).ravel()
            )
            q_pre = self._solve_l(slot_pre, place_quat, self.q_branch_seed)
        q_place = self._solve_l(slot_place, place_quat, q_pre)
        self._waypoints = {
            "approach": q_approach,
            "grasp": q_grasp,
            "lift": q_lift,
            "preplace": q_pre,
            "place": q_place,
        }

    def _plan_place_from_grasp(self) -> None:
        """Account for the measured cookie-to-tool transform after pickup."""
        tool_rotation = self.data.site_xmat[self._l_site].reshape(3, 3)
        cookie_body = self.env._cookie_bodies[self._current_cookie()]
        cookie_rotation = self.data.xmat[cookie_body].reshape(3, 3)
        relative_rotation = tool_rotation.T @ cookie_rotation
        relative_position = tool_rotation.T @ (
            self.data.xpos[cookie_body] - self.data.site_xpos[self._l_site]
        )
        box_rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        xy = self.env.TARGET_SLOTS_LOCAL[self._current_slot()]
        floor = (
            self.env.config.cookie_transfer.target_floor_z_m
            + self.env.config.cookie_transfer.bin_wall_thickness_m
        )
        object_z = floor + self.env.COOKIE_HALF_SIZE[2] + 0.008
        self._placement_height = object_z
        target_cookie = self.data.xpos[self._tb_id] + box_rotation @ np.r_[xy, object_z]
        target_tool_rotation = box_rotation @ relative_rotation.T
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, target_tool_rotation.ravel())
        target_tool = target_cookie - target_tool_rotation @ relative_position
        pre_tool = target_tool + box_rotation[:, 2] * 0.07
        pre = self._solve_l(pre_tool, quat, self.data.qpos[self._l_qpos])
        place = self._solve_l(target_tool, quat, pre)
        self._waypoints["preplace"] = pre
        self._waypoints["place"] = place

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
                action[15] = self.right_gripper
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
        action[15] = self.right_gripper
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
