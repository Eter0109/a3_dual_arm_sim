"""Contact-verified right-arm box pickup; no attachments or pose teleportation."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .contracts import RIGHT_JOINTS
from .cookie_transfer import A3CookieTransferEnv


class BoxSupportController:
    def __init__(self, env: A3CookieTransferEnv) -> None:
        self.env = env
        self.model, self.data = env.model, env.data
        self.qids = np.array([env._qpos_ids[n] for n in RIGHT_JOINTS])
        self.ranges = np.array([self.model.jnt_range[env._joint_ids[n]] for n in RIGHT_JOINTS])
        self.site = env._eef_sites[1]
        self.bin_id = env._target_bin_body
        self.fingers = [self.model.geom(f"R_finger_{n}_geom").id for n in ("inner", "outer")]
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self.steps = self.stable = 0
        self.failed = None
        self.ready = False
        self.initial_height = float(self.data.xpos[self.bin_id, 2])
        self.home = self.env.last_applied_action.copy()
        self.command = self.home.copy()
        self.quat = np.empty(4)
        mujoco.mju_mat2Quat(
            self.quat, np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).ravel()
        )
        # Approach the rim from outside so the gripper housing clears the box.
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.quat)
        angle = np.deg2rad(self.env.config.cookie_transfer.right_grasp_pitch_deg)
        outward = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        mujoco.mju_mat2Quat(self.quat, (outward @ rotation.reshape(3, 3)).ravel())
        scene = self.env.config.cookie_transfer
        self.edge = self.data.xpos[self.bin_id].copy()
        self.edge[1] -= scene.target_bin_half_size_m[1] - scene.bin_wall_thickness_m / 2
        self.edge[2] += scene.target_bin_wall_height_m - 0.013
        self.target = None

    def contact_forces(self) -> np.ndarray:
        forces = np.zeros(2)
        for contact_id in range(self.data.ncon):
            c = self.data.contact[contact_id]
            for i, finger in enumerate(self.fingers):
                other = c.geom2 if c.geom1 == finger else c.geom1 if c.geom2 == finger else -1
                if other >= 0 and self.model.geom_bodyid[other] == self.bin_id:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, contact_id, wrench)
                    forces[i] += max(0.0, wrench[0])
        return forces

    def solve(self, position: np.ndarray, quat: np.ndarray) -> np.ndarray:
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos

        def residual(q):
            work.qpos[self.qids] = q
            mujoco.mj_kinematics(self.model, work)
            current = np.empty(4)
            mujoco.mju_mat2Quat(current, work.site_xmat[self.site])
            error = np.empty(3)
            mujoco.mju_subQuat(error, quat, current)
            return np.r_[work.site_xpos[self.site] - position, 0.1 * error]

        result = least_squares(
            residual,
            np.clip(self.data.qpos[self.qids], self.ranges[:, 0] + 1e-5, self.ranges[:, 1] - 1e-5),
            bounds=(self.ranges[:, 0] + 1e-6, self.ranges[:, 1] - 1e-6),
            max_nfev=500,
        )
        if np.linalg.norm(result.fun[:3]) > 0.012 or np.linalg.norm(result.fun[3:]) > 0.04:
            raise RuntimeError(f"right box IK unreachable: error={np.linalg.norm(result.fun):.4f}")
        return result.x

    def act(self) -> np.ndarray:
        if self.failed or self.ready:
            return self.command.copy()
        self.steps += 1
        if self.steps > 240:
            self.failed = (
                f"right box {self.phase} timeout; contacts={self.contact_forces().tolist()}"
            )
            return self.command.copy()
        if self.target is None:
            pos, quat = self.edge.copy(), self.quat.copy()
            if self.phase == "approach":
                pos[2] += 0.06
            elif self.phase == "lift":
                pos[2] += self.env.config.cookie_transfer.box_lift_m
            elif self.phase == "tilt":
                pos[2] += self.env.config.cookie_transfer.box_lift_m
                mujoco.mju_quatIntegrate(
                    quat,
                    np.array([1.0, 0.0, 0.0]),
                    np.deg2rad(self.env.config.cookie_transfer.box_tilt_deg),
                )
            try:
                self.target = self.solve(pos, quat)
            except RuntimeError as exc:
                self.failed = str(exc)
                return self.command.copy()
        self.command[8:15] += np.clip(self.target - self.command[8:15], -0.025, 0.025)
        self.command[15] = (
            max(0.0, self.command[15] - 0.025) if self.phase in ("close", "lift", "tilt") else 0.7
        )
        reached = np.max(np.abs(self.data.qpos[self.qids] - self.target)) < 0.045
        contact = bool(np.all(self.contact_forces() > 0.1))
        condition = reached
        if self.phase == "close":
            condition = contact
        if self.phase in ("lift", "tilt"):
            condition = (
                reached and contact and self.data.xpos[self.bin_id, 2] > self.initial_height + 0.007
            )
        self.stable = self.stable + 1 if condition else 0
        if self.stable >= 10:
            phases = ["approach", "descend", "close", "lift", "tilt"]
            if self.phase == "tilt":
                tilt = np.arccos(np.clip(self.data.xmat[self.bin_id].reshape(3, 3)[2, 2], -1, 1))
                self.ready = np.deg2rad(3) < tilt < np.deg2rad(18)
            else:
                self.phase = phases[phases.index(self.phase) + 1]
                self.target = None
                self.steps = self.stable = 0
        return self.command.copy()
