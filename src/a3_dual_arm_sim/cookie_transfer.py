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
    cookie_count: int = 80
    required_cookies: int = 10
    position_noise_m: float = 0.0
    yaw_noise_rad: float = 0.0
    success_hold_steps: int = 20
    max_linear_speed_m_s: float = 0.06
    max_angular_speed_rad_s: float = 0.30
    max_tilt_rad: float = np.deg2rad(15.0)
    terminate_on_success: bool = True
    require_exact_slots: bool = True
    require_released: bool = False

    def __post_init__(self) -> None:
        if self.required_cookies != 10:
            raise ValueError("the 2x5 target contract requires exactly 10 cookies")


class A3CookieTransferEnv(A3DualArmEnv):
    """Video-inspired task: transfer packaged cookies from a large bin to a small bin."""

    @property
    def TARGET_SLOT_CENTERS(self) -> np.ndarray:
        return np.asarray(
            [
                self.privileged_target_slot_world(i, self._target_cookie_center_z)[:2]
                for i in range(len(self.TARGET_SLOTS_LOCAL))
            ]
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
            config = SimConfig(horizon=6000)
        super().__init__(
            config,
            action_mode=action_mode,
            render_mode=render_mode,
            render_cameras=render_cameras,
            scene="cookie_transfer",
        )
        scene_config = self.config.cookie_transfer
        if task_config is None:
            self.task_config = CookieTransferTaskConfig(
                cookie_count=len(scene_config.cookie_source_positions_m)
            )
        if self.task_config.cookie_count != len(scene_config.cookie_source_positions_m):
            raise ValueError("task cookie_count must match configured cookie source positions")
        self.SOURCE_POSITIONS = scene_config.cookie_source_positions_m
        self.SOURCE_CENTER = np.asarray(scene_config.source_bin_center_m, dtype=np.float64)
        self.SOURCE_INNER_HALF_SIZE = (
            np.asarray(scene_config.source_bin_half_size_m, dtype=np.float64)
            - scene_config.bin_wall_thickness_m
        )
        self.TARGET_CENTER = np.zeros(2, dtype=np.float64)
        self.TARGET_INNER_HALF_SIZE = (
            np.asarray(scene_config.target_bin_half_size_m, dtype=np.float64)
            - scene_config.bin_wall_thickness_m
        )
        self.COOKIE_HALF_SIZE = np.asarray(scene_config.cookie_half_size_m, dtype=np.float64)
        self.TARGET_SLOTS_LOCAL = scene_config.target_slots_local_m
        self.TARGET_SLOT_TOLERANCE = np.asarray(
            scene_config.target_slot_tolerance_m, dtype=np.float64
        )
        self.SOURCE_WALL_TOP_Z = (
            scene_config.source_wall_base_z_m + scene_config.source_bin_wall_height_m
        )
        self.TARGET_WALL_HEIGHT = scene_config.target_bin_wall_height_m
        self.SOURCE_FLOOR_TOP_Z = (
            scene_config.source_floor_z_m + scene_config.bin_wall_thickness_m / 2
        )
        self._target_cookie_center_z = (
            scene_config.target_floor_z_m
            + scene_config.bin_wall_thickness_m / 2
            + scene_config.cookie_half_size_m[2]
        )
        self.COOKIE_RESET_Z = scene_config.cookie_reset_z_m
        self.DEPLOYMENT_HOME = np.asarray(scene_config.deployment_home, dtype=np.float64)
        self._target_bin_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "target_bin")
        self._source_bin_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "source_bin")
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
        self._cookie_collision_geoms = tuple(
            frozenset(
                geom for geom in range(self.model.ngeom)
                if self.model.geom_bodyid[geom] == body
                and self.model.geom_contype[geom] != 0
            )
            for body in self._cookie_bodies
        )
        self._left_finger_geoms = tuple(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"L_finger_{finger}_geom")
            for finger in ("inner", "outer")
        )
        self._success_hold_count = 0
        self._source_initially_filled = False
        # MuJoCo resetData resets qpos, not MjModel appearance or camera fields.
        # Keep pristine values so every episode starts from the same visual base.
        self._base_cam_pos = self.model.cam_pos.copy()
        self._base_cam_fovy = self.model.cam_fovy.copy()
        self._base_light_diffuse = self.model.light_diffuse.copy()
        self._base_geom_rgba = self.model.geom_rgba.copy()
        self._base_mat_rgba = self.model.mat_rgba.copy()
        self._appearance_geom_ids = tuple(
            geom for geom in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom))
            and name.endswith("_visual")
            and (name.startswith("cookie_") or "bin_" in name)
        )

    def _randomize_appearance(self, options: dict[str, Any]) -> None:
        camera_noise = float(options.get("camera_position_noise_m", 0.0))
        fovy_noise = float(options.get("camera_fovy_noise_deg", 0.0))
        light_noise = float(options.get("light_noise_fraction", 0.0))
        color_noise = float(options.get("color_noise_fraction", 0.0))
        if min(camera_noise, fovy_noise, light_noise, color_noise) < 0:
            raise ValueError("appearance randomization ranges must be nonnegative")

        self.model.cam_pos[:] = self._base_cam_pos
        self.model.cam_fovy[:] = self._base_cam_fovy
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.geom_rgba[:] = self._base_geom_rgba
        self.model.mat_rgba[:] = self._base_mat_rgba
        if camera_noise:
            self.model.cam_pos[:] += self.np_random.uniform(
                -camera_noise, camera_noise, size=self.model.cam_pos.shape
            )
        if fovy_noise:
            self.model.cam_fovy[:] += self.np_random.uniform(
                -fovy_noise, fovy_noise, size=self.model.cam_fovy.shape
            )
        if light_noise:
            scale = self.np_random.uniform(
                1 - light_noise, 1 + light_noise, size=(self.model.nlight, 1)
            )
            self.model.light_diffuse[:] = np.clip(self._base_light_diffuse * scale, 0, 1)
        if color_noise:
            tint = self.np_random.uniform(1 - color_noise, 1 + color_noise, size=3)
            for geom in self._appearance_geom_ids:
                local_tint = self.np_random.uniform(
                    1 - color_noise / 4, 1 + color_noise / 4, size=3
                )
                self.model.geom_rgba[geom, :3] = np.clip(
                    self._base_geom_rgba[geom, :3] * tint * local_tint, 0, 1
                )
            self.model.mat_rgba[:, :3] = np.clip(
                self._base_mat_rgba[:, :3]
                * self.np_random.uniform(1 - color_noise, 1 + color_noise, size=(self.model.nmat, 1)),
                0, 1,
            )

    @property
    def cookie_positions(self) -> np.ndarray:
        return np.asarray([self.data.xpos[body_id].copy() for body_id in self._cookie_bodies])

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
        return target_rotation.T @ (self.data.xpos[self._cookie_bodies[index]] - target_position)

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
        cookie_geoms = self._cookie_collision_geoms[index]
        contacts = [False, False]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for finger_index, finger_geom in enumerate(self._left_finger_geoms):
                if finger_geom in pair and bool(pair & cookie_geoms):
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
            and -0.005
            <= local_position[2]
            <= self.TARGET_WALL_HEIGHT + self.CONTACT_CONTAINMENT_TOLERANCE_M + 0.035
        )

    def privileged_cookie_in_slot(self, index: int, slot_index: int) -> bool:
        if not self._cookie_inside_target(index):
            return False
        local_position = self.privileged_cookie_target_position(index)
        target_xy = np.asarray(self.TARGET_SLOTS_LOCAL[slot_index], dtype=np.float64)
        return bool(np.all(np.abs(local_position[:2] - target_xy) <= self.TARGET_SLOT_TOLERANCE))

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
        options = options or {}
        randomize_cookies = options.get("randomize_cookies", True)
        randomize_boxes = options.get("randomize_boxes", False)
        randomize_source = options.get("randomize_source_bin", randomize_boxes)
        randomize_target = options.get("randomize_target_bin", randomize_boxes)

        sdx, sdy = 0.0, 0.0
        if hasattr(self, "_source_bin_body"):
            if randomize_source:
                source_noise = options.get("source_bin_noise_m", 0.001)
                if source_noise > 0:
                    sdx, sdy = self.np_random.uniform(-source_noise, source_noise, size=2)
            self.model.body_pos[self._source_bin_body, 0] = self.config.cookie_transfer.source_bin_center_m[0] + sdx
            self.model.body_pos[self._source_bin_body, 1] = self.config.cookie_transfer.source_bin_center_m[1] + sdy

        if hasattr(self, "_target_bin_body") and randomize_target:
            j_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "target_bin_free")
            qpos_adr = int(self.model.jnt_qposadr[j_id])
            dof_adr = int(self.model.jnt_dofadr[j_id])
            target_noise = options.get("target_bin_noise_m", 0.002)
            target_yaw_noise = options.get("target_bin_yaw_noise_rad", 0.015)
            tdx, tdy = self.np_random.uniform(-target_noise, target_noise, size=2)
            tyaw = float(self.np_random.uniform(-target_yaw_noise, target_yaw_noise))
            self.data.qpos[qpos_adr] += tdx
            self.data.qpos[qpos_adr + 1] += tdy
            tquat = np.array([np.cos(tyaw / 2), 0.0, 0.0, np.sin(tyaw / 2)], dtype=np.float64)
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = tquat
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

        for index, (base_x, base_y) in enumerate(self.SOURCE_POSITIONS):
            if randomize_cookies:
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
            self.set_cookie_pose(index, (base_x + sdx + dx, base_y + sdy + dy, self.COOKIE_RESET_Z), quaternion)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self._randomize_appearance(options)
        self.data.time = 0.0
        self._success_hold_count = 0
        source_center = (
            self.data.xpos[self._source_bin_body][:2]
            if hasattr(self, "_source_bin_body")
            else self.SOURCE_CENTER
        )
        source_mask = tuple(
            self._cookie_inside_source(index) for index in range(self.task_config.cookie_count)
        )
        self._source_initially_filled = all(source_mask) and self._collection_touches_all_walls(
            source_mask,
            source_center,
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
                p_rel[2] = self._target_cookie_center_z
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

    def _cookie_speed(self, index: int, *, relative_to_target: bool = False) -> tuple[float, float]:
        dof_address = int(self.model.jnt_dofadr[self._cookie_joints[index]])
        velocity = self.data.qvel[dof_address : dof_address + 6].copy()
        if relative_to_target and hasattr(self, "_target_bin_body"):
            tb_cvel = self.data.cvel[self._target_bin_body]
            r = self.data.xpos[self._cookie_bodies[index]] - self.data.xpos[self._target_bin_body]
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
        rotation = self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)
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
        upright = bool(abs(float(rotation[2, 2])) >= np.cos(self.task_config.max_tilt_rad))
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
        c_mat = self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)

        p_rel = tb_mat.T @ (c_pos - tb_pos)
        r_rel = tb_mat.T @ c_mat
        half_extent = np.abs(r_rel) @ self.COOKIE_HALF_SIZE

        footprint_inside = bool(
            np.all(
                np.abs(p_rel[:2]) + half_extent[:2]
                <= self.TARGET_INNER_HALF_SIZE + self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
        )
        floor_top = (
            self.config.cookie_transfer.target_floor_z_m
            + self.config.cookie_transfer.bin_wall_thickness_m / 2
        )
        vertically_inside = bool(
            p_rel[2] - half_extent[2] >= floor_top - 0.003
            and p_rel[2] - half_extent[2] < self.TARGET_WALL_HEIGHT
        )
        if self.task_config.require_released:
            vertically_inside = vertically_inside and p_rel[2] - half_extent[2] <= floor_top + 0.004
        if not footprint_inside or not vertically_inside:
            return False
        upright = bool(
            abs(float(r_rel[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
            or abs(float(c_mat[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
        )
        linear_speed, angular_speed = self._cookie_speed(index, relative_to_target=True)
        settled = (
            linear_speed <= max(self.task_config.max_linear_speed_m_s, 0.09)
            and angular_speed <= self.task_config.max_angular_speed_rad_s
        )
        released = (
            not self.task_config.require_released
            or not any(self.privileged_left_finger_contacts(index))
        )
        return footprint_inside and vertically_inside and upright and settled and released

    def _cookie_inside_source(self, index: int) -> bool:
        source_center = (
            self.data.xpos[self._source_bin_body][:2]
            if hasattr(self, "_source_bin_body")
            else self.SOURCE_CENTER
        )
        return self._cookie_region_status(
            index,
            source_center,
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
            matches = np.flatnonzero(np.all(distances <= self.TARGET_SLOT_TOLERANCE, axis=1))
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
                r_rel = tb_mat.T @ self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)
                half_extent = (np.abs(r_rel) @ self.COOKIE_HALF_SIZE)[:2]
                edges.append((p_rel[:2] - half_extent, p_rel[:2] + half_extent))
            inner_lower = -inner_half_size
            inner_upper = inner_half_size
        else:
            for index, use_cookie in enumerate(included):
                if not use_cookie:
                    continue
                position = self.data.xpos[self._cookie_bodies[index]][:2]
                rotation = self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)
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

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, _, safety_terminated, truncated, info = super().step(action)
        in_target = tuple(
            self._cookie_inside_target(index) for index in range(self.task_config.cookie_count)
        )
        in_source = tuple(
            self._cookie_inside_source(index) for index in range(self.task_config.cookie_count)
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
            and source_count == self.task_config.cookie_count - self.task_config.required_cookies
            and all(index >= 0 for index in occupancy)
            and target_touches_all_walls
        )
        count_fill = (
            target_count == self.task_config.required_cookies
            and source_count == self.task_config.cookie_count - self.task_config.required_cookies
        )
        task_filled = exact_fill if self.task_config.require_exact_slots else count_fill
        self._success_hold_count = self._success_hold_count + 1 if task_filled else 0
        success = self._success_hold_count >= self.task_config.success_hold_steps
        terminated = safety_terminated or (success and self.task_config.terminate_on_success)
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
            count_fill=count_fill,
            success_criterion="exact_slots" if self.task_config.require_exact_slots else "released_count",
            success_hold_count=self._success_hold_count,
            required_success_hold_steps=self.task_config.success_hold_steps,
            cookie_positions=self.cookie_positions,
        )
        return observation, reward, terminated, truncated and not terminated, info
