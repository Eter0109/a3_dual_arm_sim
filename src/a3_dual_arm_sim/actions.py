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
            qids = np.asarray(self._qpos_ids[arm_index], dtype=np.int32)
            initial_q = work.qpos[qids].copy()
            if not np.any(command[arm_index, :6]):
                targets.append(initial_q)
                continue

            site_id = self._sites[arm_index]
            target_position = work.site_xpos[site_id].copy()
            target_position += (
                command[arm_index, :3] * self.config.cartesian_translation_scale_m
            )
            target_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(target_quaternion, work.site_xmat[site_id])
            has_rot = bool(np.any(np.abs(command[arm_index, 3:6]) > 1e-4))
            if has_rot:
                mujoco.mju_quatIntegrate(
                    target_quaternion,
                    command[arm_index, 3:6],
                    self.config.cartesian_rotation_scale_rad,
                )
            limits = self._ranges[arm_index]
            rot_weight = 0.10 if has_rot else 0.03

            def residual(qpos: np.ndarray) -> np.ndarray:
                work.qpos[qids] = qpos
                mujoco.mj_forward(self.model, work)
                current_quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(current_quaternion, work.site_xmat[site_id])
                rotation_error = np.empty(3, dtype=np.float64)
                mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
                p_err = work.site_xpos[site_id] - target_position
                reg_err = 0.005 * (qpos - initial_q)
                return np.r_[p_err, rot_weight * rotation_error, reg_err]

            from scipy.optimize import least_squares

            res = least_squares(
                residual,
                np.clip(initial_q, limits[:, 0] + 1e-4, limits[:, 1] - 1e-4),
                bounds=(limits[:, 0] + 1e-4, limits[:, 1] - 1e-4),
                max_nfev=35,
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
            )
            targets.append(np.asarray(res.x, dtype=np.float64))
        return np.ascontiguousarray(
            np.r_[targets[0], (command[0, 6] + 1.0) / 2.0,
                  targets[1], (command[1, 6] + 1.0) / 2.0],
            dtype=np.float64,
        )

