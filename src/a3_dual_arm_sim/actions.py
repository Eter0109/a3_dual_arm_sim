from __future__ import annotations

import mujoco
import numpy as np

from .config import SimConfig
from .contracts import LEFT_JOINTS, RIGHT_JOINTS, validate_action


class CartesianDeltaAdapter:
    """Map normalized bimanual Cartesian deltas to canonical joint targets."""

    def __init__(self, model: mujoco.MjModel, config: SimConfig) -> None:
        self.model = model
        self.config = config
        self._arms = (LEFT_JOINTS, RIGHT_JOINTS)
        self._sites = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "L_eef"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "R_eef"),
        )
        self._qpos_ids = tuple(
            tuple(model.jnt_qposadr[self._joint_id(name)] for name in arm)
            for arm in self._arms
        )
        self._dof_ids = tuple(
            tuple(model.jnt_dofadr[self._joint_id(name)] for name in arm)
            for arm in self._arms
        )
        self._ranges = tuple(
            np.asarray([model.jnt_range[self._joint_id(name)] for name in arm])
            for arm in self._arms
        )

    def _joint_id(self, name: str) -> int:
        result = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if result < 0:
            raise ValueError(f"model is missing joint {name}")
        return result

    def convert(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        command = validate_action(action, "cartesian_delta").reshape(2, 7)
        work = mujoco.MjData(self.model)
        work.qpos[:] = data.qpos
        work.qvel[:] = data.qvel
        mujoco.mj_forward(self.model, work)
        targets: list[np.ndarray] = []
        for arm_index in range(2):
            site_id = self._sites[arm_index]
            target_position = work.site_xpos[site_id].copy()
            target_position += (
                command[arm_index, :3] * self.config.cartesian_translation_scale_m
            )
            target_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(target_quaternion, work.site_xmat[site_id])
            mujoco.mju_quatIntegrate(
                target_quaternion,
                command[arm_index, 3:6],
                self.config.cartesian_rotation_scale_rad,
            )
            qids = np.asarray(self._qpos_ids[arm_index], dtype=np.int32)
            dids = np.asarray(self._dof_ids[arm_index], dtype=np.int32)
            limits = self._ranges[arm_index]
            for _ in range(self.config.ik_iterations):
                mujoco.mj_forward(self.model, work)
                position_error = target_position - work.site_xpos[site_id]
                current_quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(current_quaternion, work.site_xmat[site_id])
                rotation_error = np.empty(3, dtype=np.float64)
                mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
                error = np.r_[position_error, rotation_error]
                if np.linalg.norm(error) < 1e-5:
                    break
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, work, jacp, jacr, site_id)
                jacobian = np.vstack((jacp[:, dids], jacr[:, dids]))
                regularizer = np.eye(6) * self.config.ik_damping**2
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + regularizer, error
                )
                center = np.mean(limits, axis=1)
                delta += 0.01 * (center - work.qpos[qids])
                delta = np.clip(delta, -0.04, 0.04)
                work.qpos[qids] = np.clip(work.qpos[qids] + delta, limits[:, 0], limits[:, 1])
            targets.append(work.qpos[qids].copy())
        return np.ascontiguousarray(
            np.r_[targets[0], (command[0, 6] + 1.0) / 2.0,
                  targets[1], (command[1, 6] + 1.0) / 2.0],
            dtype=np.float64,
        )

