"""Two-box batch task with measured, contact-only right-arm box exchange.

The proven single-box batch expert is reused unchanged for each fill. This
module owns the separate handoff: filled A is physically pushed forward,
empty B is grasped and slid into the station, then filling repeats.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .same_column_batch_expert import A3SameColumnBatchExpert
from .box_support import BoxSupportController
from .cookie_transfer import A3CookieTransferEnv
from .expert import CookiePhase


class RightBoxPushController:
    """Push one free box in +Y using the closed right finger pads."""

    PUSH_SPEED_M_PER_STEP = 0.0008
    FORCE_LIMIT_N = 80.0

    def __init__(self, env: A3CookieTransferEnv, box_id: int, destination_y: float):
        self.env = env
        self.data = env.data
        self.box_id = box_id
        self.destination_y = float(destination_y)
        self.initial_position = self.data.xpos[box_id].copy()
        self.helper = BoxSupportController(env)
        self.helper.bin_id = box_id
        self.home = env.last_applied_action[8:15].copy()
        self.quat = np.empty(4)
        mujoco.mju_mat2Quat(
            self.quat, self.data.site_xmat[self.helper.site].copy()
        )
        self.x = float(self.initial_position[0])
        self.start_y = float(
            self.initial_position[1]
            - env.config.cookie_transfer.target_bin_half_size_m[1]
            - 0.027
        )
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.stable = 0
        self.failed: str | None = None
        self.done = False
        self.peak_force_n = 0.0
        self._force_over_count = 0
        self._push_goal_y = self.start_y
        self._retract_y = None
        self._target_q = None

    @property
    def box_position(self) -> np.ndarray:
        return self.data.xpos[self.box_id].copy()

    @property
    def box_yaw_rad(self) -> float:
        rotation = self.data.xmat[self.box_id].reshape(3, 3)
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0
        self.stable = 0
        self._target_q = None

    def _command(self, target_q: np.ndarray, opening: float) -> np.ndarray:
        action = self.env.last_applied_action.copy()
        action[8:15] += np.clip(target_q - action[8:15], -0.018, 0.018)
        action[15] = opening
        return action

    def _pose_command(self, y: float, z: float, opening: float) -> tuple[np.ndarray, bool]:
        if self._target_q is None:
            self._target_q = self.helper.solve(np.array([self.x, y, z]), self.quat)
        action = self._command(self._target_q, opening)
        site_position = self.data.site_xpos[self.helper.site]
        reached = (
            np.linalg.norm(site_position - np.array([self.x, y, z])) < 0.004
            and abs(self.env.current_joint_action[15] - opening) < 0.08
        )
        return action, reached

    def act(self) -> np.ndarray:
        if self.failed or self.done:
            return self.env.last_applied_action.copy()
        self.phase_steps += 1
        forces = self.helper.contact_forces()
        self.peak_force_n = max(self.peak_force_n, float(np.max(forces)))
        self._force_over_count = (
            self._force_over_count + 1
            if np.max(forces) > self.FORCE_LIMIT_N else 0
        )
        if self._force_over_count >= 3:
            self.failed = f"right fingers overloaded while pushing: {forces.tolist()} N"
            return self.env.last_applied_action.copy()
        if abs(self.box_position[0] - self.initial_position[0]) > 0.025:
            self.failed = "box drifted sideways during right-arm push"
            return self.env.last_applied_action.copy()
        up = self.data.xmat[self.box_id].reshape(3, 3)[2, 2]
        if abs(float(up)) < math.cos(math.radians(12)):
            self.failed = "box tilted during right-arm push"
            return self.env.last_applied_action.copy()
        phase_timeout = {
            "APPROACH": 220, "DESCEND": 220, "PUSH": 420,
            "RETRACT": 180, "HOME": 220,
        }[self.phase]
        if self.phase_steps > phase_timeout:
            self.failed = (
                f"right-arm {self.phase} timeout; box_y={self.box_position[1]:.4f}, "
                f"target_y={self.destination_y:.4f}"
            )
            return self.env.last_applied_action.copy()
        try:
            if self.phase == "APPROACH":
                action, reached = self._pose_command(self.start_y, 0.840, 0.0)
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._advance("DESCEND")
                return action
            if self.phase == "DESCEND":
                action, reached = self._pose_command(self.start_y, 0.755, 0.0)
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._push_goal_y = float(self.data.site_xpos[self.helper.site, 1])
                    self._advance("PUSH")
                return action
            if self.phase == "PUSH":
                if self.box_position[1] >= self.destination_y - 0.0025:
                    self._retract_y = float(self.data.site_xpos[self.helper.site, 1] - 0.035)
                    self._advance("RETRACT")
                    return self.env.last_applied_action.copy()
                # Compensate yaw while continuing the physical push.
                self.x = float(np.clip(
                    self.box_position[0] - 0.08 * self.box_yaw_rad,
                    self.initial_position[0] - 0.010,
                    self.initial_position[0] + 0.010,
                ))
                # Advance the commanded contact point slowly. A blocked pad
                # cannot accumulate a large unseen penetration target.
                measured_y = float(self.data.site_xpos[self.helper.site, 1])
                self._push_goal_y = min(
                    self._push_goal_y + self.PUSH_SPEED_M_PER_STEP,
                    measured_y + 0.018,
                )
                self._target_q = None
                if (
                    self.phase_steps >= 100
                    and self.box_position[1] - self.initial_position[1] < 0.005
                ):
                    self.failed = "right pad contacted but did not move the box"
                    return self.env.last_applied_action.copy()
                return self._pose_command(self._push_goal_y, 0.755, 0.0)[0]
            if self.phase == "RETRACT":
                assert self._retract_y is not None
                action, reached = self._pose_command(self._retract_y, 0.755, 0.0)
                # The retreat only needs to break contact; it need not settle
                # to the exact millimetre before the arm returns home.
                reached = reached or (
                    self.data.site_xpos[self.helper.site, 1] <= self._retract_y + 0.003
                    and abs(self.data.site_xpos[self.helper.site, 2] - 0.755) < 0.010
                )
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._advance("HOME")
                return action
            action = self._command(self.home, 1.0)
            reached = (
                np.max(np.abs(self.data.qpos[self.helper.qids] - self.home)) < 0.035
                and self.env.current_joint_action[15] > 0.95
            )
            self.stable = self.stable + 1 if reached else 0
            if self.stable >= 5:
                self.done = True
            return action
        except RuntimeError as exc:
            self.failed = str(exc)
            return self.env.last_applied_action.copy()


class RightBoxCarryController:
    """Pinch an empty box's rear wall and slide it into the filling station."""

    MOVE_SPEED_M_PER_STEP = 0.0008
    GRASP_OPENING = 0.10
    FORCE_LIMIT_N = 80.0

    def __init__(self, env: A3CookieTransferEnv, box_id: int, destination_y: float):
        self.env = env
        self.data = env.data
        self.box_id = box_id
        self.destination_y = float(destination_y)
        self.initial_position = self.data.xpos[box_id].copy()
        self.helper = BoxSupportController(env)
        self.helper.bin_id = box_id
        self.home = env.last_applied_action[8:15].copy()
        rotation = self.data.site_xmat[self.helper.site].reshape(3, 3).copy()
        self.quat = np.empty(4)
        mujoco.mju_mat2Quat(self.quat, rotation.ravel())
        pad_center = self.data.geom_xpos[self.helper.fingers].mean(axis=0)
        pad_offset = pad_center - self.data.site_xpos[self.helper.site]
        rear_wall_y = (
            self.initial_position[1]
            - env.config.cookie_transfer.target_bin_half_size_m[1]
        )
        self.anchor = np.array(
            [self.initial_position[0], rear_wall_y, 0.780]
        ) - pad_offset
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.stable = 0
        self.failed: str | None = None
        self.done = False
        self.peak_force_n = 0.0
        self._target_q = None
        self._move_goal_y = float(self.anchor[1])
        self._grasped = False

    @property
    def box_position(self) -> np.ndarray:
        return self.data.xpos[self.box_id].copy()

    @property
    def box_yaw_rad(self) -> float:
        rotation = self.data.xmat[self.box_id].reshape(3, 3)
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0
        self.stable = 0
        self._target_q = None

    def _command(self, target_q: np.ndarray, opening: float) -> np.ndarray:
        action = self.env.last_applied_action.copy()
        action[8:15] += np.clip(target_q - action[8:15], -0.018, 0.018)
        action[15] = opening
        return action

    def _pose_command(self, target: np.ndarray, opening: float) -> tuple[np.ndarray, bool]:
        if self._target_q is None:
            self._target_q = self.helper.solve(target, self.quat)
        action = self._command(self._target_q, opening)
        reached = (
            np.linalg.norm(self.data.site_xpos[self.helper.site] - target) < 0.005
            and abs(self.env.current_joint_action[15] - opening) < 0.08
        )
        return action, reached

    def act(self) -> np.ndarray:
        if self.failed or self.done:
            return self.env.last_applied_action.copy()
        self.phase_steps += 1
        forces = self.helper.contact_forces()
        self.peak_force_n = max(self.peak_force_n, float(np.max(forces)))
        if np.max(forces) > self.FORCE_LIMIT_N:
            self.failed = f"right fingers overloaded while carrying: {forces.tolist()} N"
            return self.env.last_applied_action.copy()
        if self._grasped:
            if abs(self.box_position[0] - self.initial_position[0]) > 0.015:
                self.failed = "empty box drifted sideways during carry"
                return self.env.last_applied_action.copy()
            if abs(self.box_yaw_rad) > math.radians(10):
                self.failed = "empty box rotated during carry"
                return self.env.last_applied_action.copy()
        timeout = {
            "APPROACH": 240, "DESCEND": 240, "CLOSE": 100,
            "MOVE": 450, "RELEASE": 100, "RETRACT": 200, "HOME": 240,
        }[self.phase]
        if self.phase_steps > timeout:
            self.failed = (
                f"right-arm carry {self.phase} timeout; "
                f"box_y={self.box_position[1]:.4f}, goal_y={self.destination_y:.4f}, "
                f"contacts={forces.tolist()}"
            )
            return self.env.last_applied_action.copy()
        try:
            if self.phase == "APPROACH":
                target = self.anchor + [0, 0, 0.080]
                action, reached = self._pose_command(target, 1.0)
            elif self.phase == "DESCEND":
                # A direct joint-space jump to the low pose sweeps the inner
                # finger through the bin floor before reaching the wall.
                # Keep each IK waypoint just below the measured wrist pose.
                target = self.anchor.copy()
                target[2] = max(
                    self.anchor[2],
                    float(self.data.site_xpos[self.helper.site, 2]) - 0.002,
                )
                self._target_q = None
                action, reached = self._pose_command(target, 1.0)
                reached = reached and target[2] <= self.anchor[2] + 0.001
            elif self.phase == "CLOSE":
                action, _ = self._pose_command(self.anchor, self.GRASP_OPENING)
                reached = bool(np.all(forces > 0.3))
            elif self.phase == "MOVE":
                if self.box_position[1] >= self.destination_y - 0.004:
                    self._advance("RELEASE")
                    return self.env.last_applied_action.copy()
                if np.min(forces) < 0.08:
                    self.stable += 1
                    if self.stable > 8:
                        self.failed = f"empty box grip lost during carry: {forces.tolist()} N"
                        return self.env.last_applied_action.copy()
                else:
                    self.stable = 0
                measured_y = float(self.data.site_xpos[self.helper.site, 1])
                self._move_goal_y = min(
                    self._move_goal_y + self.MOVE_SPEED_M_PER_STEP,
                    measured_y + 0.015,
                )
                target = self.anchor.copy()
                target[1] = self._move_goal_y
                self._target_q = None
                return self._pose_command(target, self.GRASP_OPENING)[0]
            elif self.phase == "RELEASE":
                action = self._command(self.data.qpos[self.helper.qids], 1.0)
                reached = bool(self.env.current_joint_action[15] > 0.95)
            elif self.phase == "RETRACT":
                target = self.data.site_xpos[self.helper.site].copy()
                target[2] = 0.850
                action, reached = self._pose_command(target, 1.0)
            else:
                action = self._command(self.home, 1.0)
                reached = bool(
                    np.max(np.abs(self.data.qpos[self.helper.qids] - self.home)) < 0.035
                )
            self.stable = self.stable + 1 if reached else 0
            if self.stable >= 5:
                next_phase = {
                    "APPROACH": "DESCEND", "DESCEND": "CLOSE",
                    "CLOSE": "MOVE", "RELEASE": "RETRACT",
                    "RETRACT": "HOME", "HOME": "DONE",
                }[self.phase]
                if next_phase == "DONE":
                    self.done = True
                else:
                    if next_phase == "MOVE":
                        self._grasped = True
                        self._move_goal_y = float(self.data.site_xpos[self.helper.site, 1])
                    self._advance(next_phase)
            return action
        except RuntimeError as exc:
            self.failed = str(exc)
            return self.env.last_applied_action.copy()


class TwoBoxBatchExpert:
    """Four five-cookie batches with a physical right-arm box exchange."""

    def __init__(self, env: A3CookieTransferEnv):
        self.env = env
        self.box_a = env.model.body("target_bin").id
        self.box_b = env.model.body("spare_target_bin").id
        self.station = np.asarray(
            env.config.cookie_transfer.target_bin_world_position_m[:2],
            dtype=np.float64,
        )
        self.fill: A3SameColumnBatchExpert | None = None
        self.pusher: RightBoxPushController | RightBoxCarryController | None = None
        self.stage = "FILL_A"
        self.failed: str | None = None
        self.done = False
        self.fill_reports: list[dict] = []
        self.push_reports: list[dict] = []
        self._settled_steps = 0
        self._a_indices: list[int] = []
        self._b_indices: list[int] = []

    def reset(self) -> None:
        self.env._target_bin_body = self.box_a
        self.stage = "FILL_A"
        self.failed = None
        self.done = False
        self.fill_reports = []
        self.push_reports = []
        self._a_indices = []
        self._b_indices = []
        self._settled_steps = 0
        self.pusher = None
        self.fill = A3SameColumnBatchExpert(self.env)
        self.fill.reset()

    @property
    def status(self) -> str:
        if self.stage.startswith("FILL") and self.fill is not None:
            return f"stage={self.stage} {self.fill.status}"
        if self.stage in ("PUSH_A", "CARRY_B") and self.pusher is not None:
            return (
                f"stage={self.stage} phase={self.pusher.phase} "
                f"box_y={self.pusher.box_position[1]:.3f} "
                f"goal_y={self.pusher.destination_y:.3f}"
            )
        return f"stage={self.stage}"

    def _count(self, box_id: int, indices: list[int]) -> int:
        old = self.env._target_bin_body
        try:
            self.env._target_bin_body = box_id
            return sum(self.env.privileged_cookie_in_target(i) for i in indices)
        finally:
            self.env._target_bin_body = old

    def counts(self) -> tuple[int, int]:
        all_indices = list(range(len(self.env._cookie_bodies)))
        return self._count(self.box_a, all_indices), self._count(
            self.box_b, all_indices
        )

    def _new_pusher(self, box_id: int, destination_y: float, stage: str) -> None:
        if box_id == self.box_b:
            self.pusher = RightBoxCarryController(self.env, box_id, destination_y)
        else:
            self.pusher = RightBoxPushController(self.env, box_id, destination_y)
        self.stage = stage

    def act(self) -> np.ndarray:
        if self.failed or self.done:
            return self.env.last_applied_action.copy()
        if self.stage in ("FILL_A", "FILL_B"):
            assert self.fill is not None
            action = self.fill.act()
            if self.fill.failed:
                self.failed = f"{self.stage}: {self.fill.failure_reason}"
            elif self.fill.finished and self.fill.phase is CookiePhase.DONE:
                box = "A" if self.stage == "FILL_A" else "B"
                indices = list(self.fill.completed_cookie_indices)
                self.fill_reports.append(
                    {"box": box, "cookie_ids": indices, "batches": self.fill.batch_reports}
                )
                if box == "A":
                    self._a_indices = indices
                    self._new_pusher(self.box_a, self.station[1] + 0.090, "PUSH_A")
                else:
                    self._b_indices = indices
                    self.stage = "VERIFY_BOTH"
                    self._settled_steps = 0
            return action
        if self.stage in ("PUSH_A", "CARRY_B"):
            assert self.pusher is not None
            action = self.pusher.act()
            if self.pusher.failed:
                self.failed = f"{self.stage}: {self.pusher.failed}"
            elif self.pusher.done:
                self.push_reports.append(
                    {
                        "box": "A" if self.stage == "PUSH_A" else "B",
                        "method": "push" if self.stage == "PUSH_A" else "grasp_and_slide",
                        "from_y_m": float(self.pusher.initial_position[1]),
                        "to_y_m": float(self.pusher.box_position[1]),
                        "final_yaw_deg": math.degrees(self.pusher.box_yaw_rad),
                        "peak_finger_force_n": self.pusher.peak_force_n,
                    }
                )
                self.stage = "VERIFY_A" if self.stage == "PUSH_A" else "VERIFY_B"
                self._settled_steps = 0
            return action
        if self.stage == "VERIFY_A":
            position = self.env.data.xpos[self.box_a]
            good = (
                position[1] >= self.station[1] + 0.085
                and self._count(self.box_a, self._a_indices) == 10
            )
            self._settled_steps = self._settled_steps + 1 if good else 0
            if self._settled_steps >= 15:
                self._new_pusher(self.box_b, self.station[1], "CARRY_B")
            elif self._settled_steps == 0 and self.pusher is not None:
                if self.pusher.done and abs(position[1] - self.station[1]) < 0.050:
                    self.failed = "filled box A did not clear the filling station"
            return self.env.last_applied_action.copy()
        if self.stage == "VERIFY_B":
            position = self.env.data.xpos[self.box_b]
            good = (
                np.linalg.norm(position[:2] - self.station) < 0.008
                and self._count(self.box_a, self._a_indices) == 10
            )
            self._settled_steps = self._settled_steps + 1 if good else 0
            if self._settled_steps >= 15:
                self.env._target_bin_body = self.box_b
                self.fill = A3SameColumnBatchExpert(self.env)
                try:
                    self.fill.reset()
                except RuntimeError as exc:
                    self.failed = f"FILL_B setup: {exc}"
                else:
                    self.stage = "FILL_B"
            elif self._settled_steps == 0 and self.pusher is not None:
                if self.pusher.done and np.linalg.norm(position[:2] - self.station) > 0.012:
                    self.failed = "empty box B did not reach the filling station"
            return self.env.last_applied_action.copy()
        if self.stage == "VERIFY_BOTH":
            a, b = self.counts()
            source = sum(
                self.env.privileged_cookie_in_source(i)
                for i in range(len(self.env._cookie_bodies))
            )
            good = (
                a == b == 10
                and len(self._a_indices) == len(self._b_indices) == 10
                and self._count(self.box_a, self._a_indices) == 10
                and self._count(self.box_b, self._b_indices) == 10
                and source == len(self.env._cookie_bodies) - 20
                and self.env.data.xpos[self.box_a, 1] >= self.station[1] + 0.085
                and np.linalg.norm(self.env.data.xpos[self.box_b, :2] - self.station) < 0.008
            )
            self._settled_steps = self._settled_steps + 1 if good else 0
            if self._settled_steps >= 20:
                self.stage = "DONE"
                self.done = True
            elif self._settled_steps == 0:
                self.failed = (
                    f"dual-box final criteria failed: A={a}, B={b}, source={source}"
                )
            return self.env.last_applied_action.copy()
        self.failed = f"unknown dual-box stage {self.stage}"
        return self.env.last_applied_action.copy()
