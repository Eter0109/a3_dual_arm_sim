from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import SimConfig
from .contracts import ARM_JOINTS, ActionMode
from .env import A3DualArmEnv


@dataclass(frozen=True)
class CookieTransferTaskConfig:
    cookie_count: int = 30
    required_cookies: int = 10
    position_noise_m: float = 0.0
    yaw_noise_rad: float = 0.0
    success_hold_steps: int = 20
    max_linear_speed_m_s: float = 0.06
    max_angular_speed_rad_s: float = 0.30
    max_tilt_rad: float = np.deg2rad(15.0)
    terminate_on_success: bool = True

    def __post_init__(self) -> None:
        if self.required_cookies != 10:
            raise ValueError("the 2x5 target contract requires exactly 10 cookies")


class A3CookieTransferEnv(A3DualArmEnv):
    """Video-inspired task: transfer packaged cookies from a large bin to a small bin."""

    TARGET_SLOT_CENTERS = np.asarray(
        [
            (0.0911, 0.1816),
            (0.1441, 0.1808),
            (0.0916, 0.2002),
            (0.1445, 0.1994),
            (0.0920, 0.2188),
            (0.1450, 0.2180),
            (0.0925, 0.2374),
            (0.1454, 0.2366),
            (0.0934, 0.2746),
            (0.1463, 0.2738),
        ],
        dtype=np.float64,
    )
    CONTACT_CONTAINMENT_TOLERANCE_M = 0.004
    WALL_CONTACT_TOLERANCE_M = 0.026
    TARGET_FLOOR_TOP_Z = 0.775

    def __init__(
        self,
        config: SimConfig | str | Path | None = None,
        *,
        task_config: CookieTransferTaskConfig | None = None,
        action_mode: ActionMode = "joint_position",
        render_mode: str | None = None,
        render_cameras: bool = True,
    ) -> None:
        self.task_config = task_config or CookieTransferTaskConfig()
        if config is None:
            config = SimConfig(horizon=2500)
        super().__init__(
            config,
            action_mode=action_mode,
            render_mode=render_mode,
            render_cameras=render_cameras,
            scene="cookie_transfer",
        )
        scene_config = self.config.cookie_transfer
        if self.task_config.cookie_count != len(scene_config.cookie_source_positions_m):
            raise ValueError(
                "task cookie_count must match configured cookie source positions"
            )
        self.SOURCE_POSITIONS = scene_config.cookie_source_positions_m
        self.SOURCE_CENTER = np.asarray(
            scene_config.source_bin_center_m, dtype=np.float64
        )
        self.SOURCE_INNER_HALF_SIZE = (
            np.asarray(scene_config.source_bin_half_size_m, dtype=np.float64)
            - scene_config.bin_wall_thickness_m
        )
        self.TARGET_CENTER = np.zeros(2, dtype=np.float64)
        self.TARGET_INNER_HALF_SIZE = (
            np.asarray(scene_config.target_bin_half_size_m, dtype=np.float64)
            - scene_config.bin_wall_thickness_m
        )
        self.COOKIE_HALF_SIZE = np.asarray(
            scene_config.cookie_half_size_m, dtype=np.float64
        )
        self.TARGET_SLOTS_LOCAL = scene_config.target_slots_local_m
        self.TARGET_SLOT_TOLERANCE = np.asarray(
            scene_config.target_slot_tolerance_m, dtype=np.float64
        )
        self.SOURCE_WALL_TOP_Z = (
            scene_config.source_wall_base_z_m + scene_config.source_bin_wall_height_m
        )
        self.TARGET_WALL_HEIGHT = scene_config.target_bin_wall_height_m
        self.SOURCE_FLOOR_TOP_Z = scene_config.source_floor_z_m
        self.COOKIE_RESET_Z = scene_config.cookie_reset_z_m
        self.DEPLOYMENT_HOME = np.asarray(
            scene_config.deployment_home, dtype=np.float64
        )
        self._target_bin_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "target_bin")
        self._cookie_bodies = tuple(
            self._id(mujoco.mjtObj.mjOBJ_BODY, f"cookie_{index}")
            for index in range(self.task_config.cookie_count)
        )
        self._cookie_joints = tuple(
            self._id(mujoco.mjtObj.mjOBJ_JOINT, f"cookie_{index}_free")
            for index in range(self.task_config.cookie_count)
        )
        self._cookie_geoms = tuple(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"cookie_{index}_geom")
            for index in range(self.task_config.cookie_count)
        )
        self._left_finger_geoms = tuple(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"L_finger_{finger}_geom")
            for finger in ("inner", "outer")
        )
        self._success_hold_count = 0
        self._source_initially_filled = False

    @property
    def cookie_positions(self) -> np.ndarray:
        return np.asarray(
            [self.data.xpos[body_id].copy() for body_id in self._cookie_bodies]
        )

    @property
    def success_hold_count(self) -> int:
        return self._success_hold_count

    def privileged_cookie_position(self, index: int) -> np.ndarray:
        """Return simulator-truth position for expert control, never policy input."""
        return self.data.xpos[self._cookie_bodies[index]].copy()

    def privileged_cookie_target_position(self, index: int) -> np.ndarray:
        """Return Cookie position in the moving target-bin frame."""
        target_position = self.data.xpos[self._target_bin_body]
        target_rotation = self.data.xmat[self._target_bin_body].reshape(3, 3)
        return target_rotation.T @ (
            self.data.xpos[self._cookie_bodies[index]] - target_position
        )

    def privileged_target_slot_world(self, slot_index: int, z: float) -> np.ndarray:
        """Convert a configured target slot into a simulator-truth world point."""
        target_position = self.data.xpos[self._target_bin_body]
        target_rotation = self.data.xmat[self._target_bin_body].reshape(3, 3)
        slot_xy = self.TARGET_SLOTS_LOCAL[slot_index]
        return target_position + target_rotation @ np.asarray(
            [slot_xy[0], slot_xy[1], z], dtype=np.float64
        )

    def privileged_left_finger_contacts(self, index: int) -> tuple[bool, bool]:
        """Report target-Cookie contact for each left finger from MuJoCo contacts."""
        cookie_geom = self._cookie_geoms[index]
        contacts = [False, False]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for finger_index, finger_geom in enumerate(self._left_finger_geoms):
                if pair == {finger_geom, cookie_geom}:
                    contacts[finger_index] = True
        return contacts[0], contacts[1]

    def privileged_left_touch_values(self) -> tuple[float, float]:
        """Return left fingertip touch signals for expert-only grasp verification."""
        return (
            float(self._sensor("L_finger_inner_touch_sensor")[0]),
            float(self._sensor("L_finger_outer_touch_sensor")[0]),
        )

    def privileged_cookie_in_source(self, index: int) -> bool:
        return self._cookie_inside_source(index)

    def privileged_cookie_in_target(self, index: int) -> bool:
        return self._cookie_inside_target(index)

    def privileged_cookie_in_target_region(self, index: int) -> bool:
        """Check target-bin geometry without requiring the Cookie to be settled."""
        local_position = self.privileged_cookie_target_position(index)
        return bool(
            abs(local_position[0])
            <= self.TARGET_INNER_HALF_SIZE[0] + self.CONTACT_CONTAINMENT_TOLERANCE_M
            and abs(local_position[1])
            <= self.TARGET_INNER_HALF_SIZE[1] + self.CONTACT_CONTAINMENT_TOLERANCE_M
            and -0.005 <= local_position[2]
            <= self.TARGET_WALL_HEIGHT
            + self.CONTACT_CONTAINMENT_TOLERANCE_M
            + 0.035
        )

    def privileged_cookie_in_slot(self, index: int, slot_index: int) -> bool:
        if not self._cookie_inside_target(index):
            return False
        local_position = self.privileged_cookie_target_position(index)
        target_xy = np.asarray(self.TARGET_SLOTS_LOCAL[slot_index], dtype=np.float64)
        return bool(
            np.all(
                np.abs(local_position[:2] - target_xy)
                <= self.TARGET_SLOT_TOLERANCE
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = super().reset(
            seed=seed,
            options={"randomize_objects": False},
        )
        arm_values = np.r_[self.DEPLOYMENT_HOME[:7], self.DEPLOYMENT_HOME[8:15]]
        for name, value in zip(ARM_JOINTS, arm_values, strict=True):
            self.data.qpos[self._qpos_ids[name]] = value
            self.data.qvel[self._dof_ids[name]] = 0.0
        for side in ("L", "R"):
            for joint_id in self._finger_joints[side]:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = 0.0
                self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
        self._last_applied_action = self.DEPLOYMENT_HOME.copy()
        self._apply_controls(self.DEPLOYMENT_HOME)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        randomize = (options or {}).get("randomize_cookies", True)
        for index, (base_x, base_y) in enumerate(self.SOURCE_POSITIONS):
            if randomize:
                dx, dy = self.np_random.uniform(
                    -self.task_config.position_noise_m,
                    self.task_config.position_noise_m,
                    size=2,
                )
                yaw = float(
                    self.np_random.uniform(
                        -self.task_config.yaw_noise_rad,
                        self.task_config.yaw_noise_rad,
                    )
                )
            else:
                dx = dy = yaw = 0.0
            quaternion = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
            self.set_cookie_pose(
                index, (base_x + dx, base_y + dy, self.COOKIE_RESET_Z), quaternion
            )
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self._success_hold_count = 0
        source_mask = tuple(
            self._cookie_inside_source(index)
            for index in range(self.task_config.cookie_count)
        )
        self._source_initially_filled = all(
            source_mask
        ) and self._collection_touches_all_walls(
            source_mask,
            self.SOURCE_CENTER,
            self.SOURCE_INNER_HALF_SIZE,
        )
        observation = self._observation()
        info.update(
            success=False,
            cookies_in_target=0,
            cookies_in_source=sum(source_mask),
            required_cookies=self.task_config.required_cookies,
            cookies_in_target_mask=[False] * self.task_config.cookie_count,
            cookies_in_source_mask=source_mask,
            target_slot_occupancy=[-1] * self.task_config.required_cookies,
            source_initially_filled=self._source_initially_filled,
        )
        return observation, info

    def set_cookie_pose(
        self,
        index: int,
        position: tuple[float, float, float],
        quaternion: tuple[float, float, float, float] | None = None,
    ) -> None:
        joint_id = self._cookie_joints[index]
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        pos = np.asarray(position, dtype=np.float64)
        if hasattr(self, "_target_bin_body"):
            tb_pos = self.data.xpos[self._target_bin_body]
            tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
            p_rel = tb_mat.T @ (pos - tb_pos)
            if (
                abs(p_rel[0]) <= self.TARGET_INNER_HALF_SIZE[0] + 0.02
                and abs(p_rel[1]) <= self.TARGET_INNER_HALF_SIZE[1] + 0.02
            ):
                p_rel[2] = 0.031
                pos = tb_pos + tb_mat @ p_rel
                if quaternion is None:
                    target_q = np.empty(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(target_q, tb_mat.reshape(-1))
                    quaternion = tuple(target_q)
        if quaternion is None:
            quaternion = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[qpos_address : qpos_address + 3] = pos
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _cookie_speed(
        self, index: int, *, relative_to_target: bool = False
    ) -> tuple[float, float]:
        dof_address = int(self.model.jnt_dofadr[self._cookie_joints[index]])
        velocity = self.data.qvel[dof_address : dof_address + 6].copy()
        if relative_to_target and hasattr(self, "_target_bin_body"):
            tb_cvel = self.data.cvel[self._target_bin_body]
            r = (
                self.data.xpos[self._cookie_bodies[index]]
                - self.data.xpos[self._target_bin_body]
            )
            v_expected = tb_cvel[3:] + np.cross(tb_cvel[:3], r)
            rel_v = velocity[:3] - v_expected
            rel_w = velocity[3:] - tb_cvel[:3]
            return float(np.linalg.norm(rel_v)), float(np.linalg.norm(rel_w))
        return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))

    def _cookie_region_status(
        self,
        index: int,
        center: np.ndarray,
        inner_half_size: np.ndarray,
        wall_top_z: float,
        *,
        floor_top_z: float = 0.753,
        require_upright: bool,
        require_settled: bool,
    ) -> bool:
        position = self.data.xpos[self._cookie_bodies[index]]
        rotation = self.data.geom_xmat[self._cookie_geoms[index]].reshape(3, 3)
        world_half_extent = np.abs(rotation) @ self.COOKIE_HALF_SIZE
        lower_xy = center - inner_half_size
        upper_xy = center + inner_half_size
        footprint_inside = bool(
            np.all(
                position[:2] - world_half_extent[:2]
                >= lower_xy - self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
            and np.all(
                position[:2] + world_half_extent[:2]
                <= upper_xy + self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
        )
        vertically_inside = bool(
            position[2] - world_half_extent[2] >= floor_top_z - 0.003
            and position[2] - world_half_extent[2]
            <= wall_top_z + self.CONTACT_CONTAINMENT_TOLERANCE_M
        )
        upright = bool(
            abs(float(rotation[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
        )
        linear_speed, angular_speed = self._cookie_speed(index)
        settled = (
            linear_speed <= self.task_config.max_linear_speed_m_s
            and angular_speed <= self.task_config.max_angular_speed_rad_s
        )
        return (
            footprint_inside
            and vertically_inside
            and (upright or not require_upright)
            and (settled or not require_settled)
        )

    def _cookie_inside_target(self, index: int) -> bool:
        tb_pos = self.data.xpos[self._target_bin_body]
        tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
        c_pos = self.data.xpos[self._cookie_bodies[index]]
        c_mat = self.data.geom_xmat[self._cookie_geoms[index]].reshape(3, 3)

        p_rel = tb_mat.T @ (c_pos - tb_pos)
        r_rel = tb_mat.T @ c_mat

        footprint_inside = bool(
            abs(p_rel[0])
            <= self.TARGET_INNER_HALF_SIZE[0] + self.CONTACT_CONTAINMENT_TOLERANCE_M
            and abs(p_rel[1])
            <= self.TARGET_INNER_HALF_SIZE[1] + self.CONTACT_CONTAINMENT_TOLERANCE_M
        )
        vertically_inside = bool(
            p_rel[2] >= -0.005
            and p_rel[2]
            <= self.TARGET_WALL_HEIGHT + self.CONTACT_CONTAINMENT_TOLERANCE_M + 0.035
        )
        upright = bool(
            abs(float(r_rel[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
            or abs(float(c_mat[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
        )
        linear_speed, angular_speed = self._cookie_speed(index, relative_to_target=True)
        settled = (
            linear_speed <= max(self.task_config.max_linear_speed_m_s, 0.09)
            and angular_speed <= self.task_config.max_angular_speed_rad_s
        )
        return footprint_inside and vertically_inside and upright and settled

    def _cookie_inside_source(self, index: int) -> bool:
        return self._cookie_region_status(
            index,
            self.SOURCE_CENTER,
            self.SOURCE_INNER_HALF_SIZE,
            self.SOURCE_WALL_TOP_Z,
            floor_top_z=self.SOURCE_FLOOR_TOP_Z,
            require_upright=False,
            require_settled=False,
        )

    def _target_slot_occupancy(self, in_target: tuple[bool, ...]) -> tuple[int, ...]:
        tb_pos = self.data.xpos[self._target_bin_body]
        tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
        occupancy = [-1] * len(self.TARGET_SLOTS_LOCAL)
        for cookie_index, eligible in enumerate(in_target):
            if not eligible:
                continue
            c_pos = self.data.xpos[self._cookie_bodies[cookie_index]]
            p_rel = tb_mat.T @ (c_pos - tb_pos)
            distances = np.abs(np.asarray(self.TARGET_SLOTS_LOCAL) - p_rel[:2])
            matches = np.flatnonzero(
                np.all(distances <= self.TARGET_SLOT_TOLERANCE, axis=1)
            )
            if len(matches) == 1 and occupancy[int(matches[0])] < 0:
                occupancy[int(matches[0])] = cookie_index
        return tuple(occupancy)

    def _collection_touches_all_walls(
        self,
        included: tuple[bool, ...],
        center: np.ndarray,
        inner_half_size: np.ndarray,
        *,
        is_target: bool = False,
    ) -> bool:
        edges: list[tuple[np.ndarray, np.ndarray]] = []
        if is_target:
            tb_pos = self.data.xpos[self._target_bin_body]
            tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
            for index, use_cookie in enumerate(included):
                if not use_cookie:
                    continue
                c_pos = self.data.xpos[self._cookie_bodies[index]]
                p_rel = tb_mat.T @ (c_pos - tb_pos)
                r_rel = tb_mat.T @ self.data.geom_xmat[
                    self._cookie_geoms[index]
                ].reshape(3, 3)
                half_extent = (np.abs(r_rel) @ self.COOKIE_HALF_SIZE)[:2]
                edges.append((p_rel[:2] - half_extent, p_rel[:2] + half_extent))
            inner_lower = -inner_half_size
            inner_upper = inner_half_size
        else:
            for index, use_cookie in enumerate(included):
                if not use_cookie:
                    continue
                position = self.data.xpos[self._cookie_bodies[index]][:2]
                rotation = self.data.geom_xmat[self._cookie_geoms[index]].reshape(3, 3)
                half_extent = (np.abs(rotation) @ self.COOKIE_HALF_SIZE)[:2]
                edges.append((position - half_extent, position + half_extent))
            inner_lower = center - inner_half_size
            inner_upper = center + inner_half_size

        if not edges:
            return False
        lower = np.min(np.asarray([edge[0] for edge in edges]), axis=0)
        upper = np.max(np.asarray([edge[1] for edge in edges]), axis=0)
        return bool(
            np.all(lower <= inner_lower + self.WALL_CONTACT_TOLERANCE_M)
            and np.all(upper >= inner_upper - self.WALL_CONTACT_TOLERANCE_M)
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, _, safety_terminated, truncated, info = super().step(action)
        in_target = tuple(
            self._cookie_inside_target(index)
            for index in range(self.task_config.cookie_count)
        )
        in_source = tuple(
            self._cookie_inside_source(index)
            for index in range(self.task_config.cookie_count)
        )
        target_count = sum(in_target)
        source_count = sum(in_source)
        occupancy = self._target_slot_occupancy(in_target)
        target_touches_all_walls = self._collection_touches_all_walls(
            in_target,
            self.TARGET_CENTER,
            self.TARGET_INNER_HALF_SIZE,
            is_target=True,
        )
        exact_fill = (
            target_count == self.task_config.required_cookies
            and source_count
            == self.task_config.cookie_count - self.task_config.required_cookies
            and all(index >= 0 for index in occupancy)
            and target_touches_all_walls
        )
        self._success_hold_count = self._success_hold_count + 1 if exact_fill else 0
        success = self._success_hold_count >= self.task_config.success_hold_steps
        terminated = safety_terminated or (
            success and self.task_config.terminate_on_success
        )
        reward = min(target_count / self.task_config.required_cookies, 1.0)
        info.update(
            success=success,
            cookies_in_target=target_count,
            cookies_in_source=source_count,
            required_cookies=self.task_config.required_cookies,
            cookies_in_target_mask=in_target,
            cookies_in_source_mask=in_source,
            target_slot_occupancy=occupancy,
            target_touches_all_walls=target_touches_all_walls,
            source_initially_filled=self._source_initially_filled,
            exact_2x5_fill=exact_fill,
            success_hold_count=self._success_hold_count,
            required_success_hold_steps=self.task_config.success_hold_steps,
            cookie_positions=self.cookie_positions,
        )
        return observation, reward, terminated, truncated and not terminated, info
