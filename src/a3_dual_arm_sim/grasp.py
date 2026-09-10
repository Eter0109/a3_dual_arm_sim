from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import SimConfig
from .contracts import ActionMode
from .env import A3DualArmEnv


@dataclass(frozen=True)
class GraspTaskConfig:
    object_x_m: float = 0.12
    object_y_m: float = 0.48
    position_noise_m: float = 0.025
    required_lift_m: float = 0.08
    success_hold_steps: int = 20
    touch_threshold_n: float = 0.02
    max_linear_speed_m_s: float = 0.08
    max_angular_speed_rad_s: float = 0.25
    max_grasp_center_error_m: float = 0.045
    terminate_on_success: bool = True


class A3GraspEnv(A3DualArmEnv):
    """Left-arm cube grasp/lift task; privileged task state stays out of policy observations."""

    def __init__(
        self,
        config: SimConfig | str | Path | None = None,
        *,
        task_config: GraspTaskConfig | None = None,
        action_mode: ActionMode = "joint_position",
        render_mode: str | None = None,
        render_cameras: bool = True,
    ) -> None:
        super().__init__(
            config,
            action_mode=action_mode,
            render_mode=render_mode,
            render_cameras=render_cameras,
        )
        self.task_config = task_config or GraspTaskConfig()
        self._object_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "object_0")
        self._object_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "object_0_free")
        self._object_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "object_0_geom")
        self._table_geom = self._id(mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self._finger_geoms = tuple(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"L_finger_{finger}_geom")
            for finger in ("inner", "outer")
        )
        self._initial_object_z = 0.79
        self._success_hold_count = 0
        self._ever_grasped = False

    @property
    def target_position(self) -> np.ndarray:
        return self.data.xpos[self._object_body].copy()

    @property
    def target_quaternion(self) -> np.ndarray:
        return self.data.xquat[self._object_body].copy()

    @property
    def initial_object_z(self) -> float:
        return self._initial_object_z

    @property
    def success_hold_count(self) -> int:
        return self._success_hold_count

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
        noise = self.task_config.position_noise_m
        x = self.task_config.object_x_m + self.np_random.uniform(-noise, noise)
        y = self.task_config.object_y_m + self.np_random.uniform(-noise, noise)
        self._set_free_object(0, (x, y, 0.79))
        self._set_free_object(1, (0.48, -0.48, 0.79))
        self._set_free_object(2, (0.68, -0.48, 0.79))
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self._initial_object_z = float(self.target_position[2])
        self._success_hold_count = 0
        self._ever_grasped = False
        observation = self._observation()
        info.update(
            success=False,
            grasped=False,
            ever_grasped=False,
            lift_m=0.0,
            target_position=self.target_position,
        )
        return observation, info

    def _set_free_object(self, index: int, position: tuple[float, float, float]) -> None:
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"object_{index}_free")
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_address : qpos_address + 3] = position
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _finger_object_contacts(self) -> tuple[bool, bool]:
        contacts = [False, False]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for finger_index, finger_geom in enumerate(self._finger_geoms):
                if pair == {finger_geom, self._object_geom}:
                    contacts[finger_index] = True
        return contacts[0], contacts[1]

    def is_grasped(self) -> bool:
        contacts = self._finger_object_contacts()
        touches = (
            float(self._sensor("L_finger_inner_touch_sensor")[0]),
            float(self._sensor("L_finger_outer_touch_sensor")[0]),
        )
        return all(contacts) and all(
            value >= self.task_config.touch_threshold_n for value in touches
        )

    def _object_touches_table(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if {int(contact.geom1), int(contact.geom2)} == {
                self._object_geom,
                self._table_geom,
            }:
                return True
        return False

    def _grasp_center_error(self) -> float:
        finger_center = 0.5 * (
            self.data.geom_xpos[self._finger_geoms[0]]
            + self.data.geom_xpos[self._finger_geoms[1]]
        )
        return float(np.linalg.norm(self.target_position - finger_center))

    def _object_speeds(self) -> tuple[float, float]:
        dof_address = int(self.model.jnt_dofadr[self._object_joint])
        linear = float(np.linalg.norm(self.data.qvel[dof_address : dof_address + 3]))
        angular = float(np.linalg.norm(self.data.qvel[dof_address + 3 : dof_address + 6]))
        return linear, angular

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, _, safety_terminated, truncated, info = super().step(action)
        grasped = self.is_grasped()
        self._ever_grasped = self._ever_grasped or grasped
        lift_m = float(self.target_position[2] - self._initial_object_z)
        linear_speed, angular_speed = self._object_speeds()
        grasp_center_error = self._grasp_center_error()
        touches_table = self._object_touches_table()
        stable_grasp = (
            grasped
            and not touches_table
            and lift_m >= self.task_config.required_lift_m
            and linear_speed <= self.task_config.max_linear_speed_m_s
            and angular_speed <= self.task_config.max_angular_speed_rad_s
            and grasp_center_error <= self.task_config.max_grasp_center_error_m
        )
        self._success_hold_count = self._success_hold_count + 1 if stable_grasp else 0
        success = self._success_hold_count >= self.task_config.success_hold_steps
        terminated = safety_terminated or (
            success and self.task_config.terminate_on_success
        )
        progress = float(np.clip(lift_m / self.task_config.required_lift_m, 0.0, 1.0))
        reward = 1.0 if success else 0.25 * float(grasped) + 0.5 * progress
        info.update(
            {
                "success": success,
                "grasped": grasped,
                "ever_grasped": self._ever_grasped,
                "lift_m": lift_m,
                "target_lifted": lift_m >= self.task_config.required_lift_m,
                "stable_grasp": stable_grasp,
                "object_touches_table": touches_table,
                "object_linear_speed_m_s": linear_speed,
                "object_angular_speed_rad_s": angular_speed,
                "grasp_center_error_m": grasp_center_error,
                "success_hold_count": self._success_hold_count,
                "finger_object_contacts": self._finger_object_contacts(),
                "target_position": self.target_position,
            }
        )
        return observation, reward, terminated, truncated and not terminated, info
