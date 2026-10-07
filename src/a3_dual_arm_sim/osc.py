"""robosuite's operational-space control, for the scene that uses robosuite's robot.

Why this exists
---------------
The A3 scene drives its arm with joint-position actuators and solves the incoming Cartesian
delta into joint targets.  That works for the arm those gains were tuned on and does not
survive the swap to LIBERO's Panda, measurably, both ways round: with the A3's torque limits
the arm has 40 percent of its rated torque and never arrives (0 of 40 episodes got a
fingertip within 56 mm of the bowl, against 39 of 40 on the A3), and with robosuite's real
limits the A3's gains no longer hold it (34.6 degrees off home after a reset's settling
steps, against 3.75).  Neither is a tuning problem: **LIBERO does not drive its Panda with a
joint-position servo at all.**  It configures robosuite's `OSC_POSE`, which holds a *goal
pose* in task space and applies joint **torques** to reach it.

What this is, and what it is not
--------------------------------
It is the law robosuite applies, with robosuite's own numbers read out of the controller
LIBERO configures -- `kp = 150`, `damping_ratio = 1`, `output_max = [0.05 x3, 0.5 x3]`,
`control_delta = True`, `uncouple_pos_ori = True`:

    tau = J^T ( kp * e_pos + kp * e_rot - kd * J qvel ) + qfrc_bias

with `kd = 2 sqrt(kp) damping_ratio`, `e_rot` the axis-angle of `R_goal R_current^T`, and the
goal orientation composed as `R_err R_current` -- the world-frame composition robosuite's
`set_goal_orientation` performs, and the same one `CartesianDeltaAdapter` implements.

It is **not** robosuite's dynamically consistent inverse: robosuite multiplies the task force
through `Lambda = (J M^-1 J^T)^-1`, and this uses the plain Jacobian transpose.  Same control
law, same gains, same goals, same `control_delta` semantics; the difference is a weighting
inside the loop, and it is stated here rather than hidden.  `ramp_ratio` (robosuite's soft
start) is not implemented either.

The goal is set from the **current** measurement every control step, which is what
`control_delta = True` means: a delta is a request relative to where the arm is, not a
running sum of requests, so a missed command is not accumulated forever.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .config import SimConfig
from .contracts import LEFT_JOINTS, RIGHT_JOINTS

#: robosuite's `OSC_POSE` gains, from the controller LIBERO configures for its Panda.
KP = 150.0
DAMPING_RATIO = 1.0


class OperationalSpaceAdapter:
    """Normalized OSC deltas in, joint torques for the left arm out.

    The returned vector is fifteen-wide in the same canonical layout the rest of this
    repository speaks -- left joints, left gripper, right joints, right gripper -- with the
    one difference that the left arm's seven entries are **torques**, because that arm's
    actuators are motors.  `A3DualArmEnv._apply_controls` writes them to `ctrl` either way,
    and the motors' own control ranges are what bounds them.
    """

    def __init__(self, model: mujoco.MjModel, config: SimConfig, *, site: str = "L_eef") -> None:
        self.model = model
        self.config = config
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if site_id < 0:
            raise ValueError(f"the operational space controller needs a site named {site}")
        self._site = site_id
        self._left_qpos = tuple(self._qpos_address(name) for name in LEFT_JOINTS)
        self._left_dof = tuple(self._dof_address(name) for name in LEFT_JOINTS)
        self._right_qpos = tuple(self._qpos_address(name) for name in RIGHT_JOINTS)
        self._limits = np.asarray(
            [
                model.actuator_ctrlrange[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
                ]
                for name in LEFT_JOINTS
            ],
            dtype=np.float64,
        )
        self.kp = KP
        self.kd = 2.0 * math.sqrt(KP) * DAMPING_RATIO
        self._jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        self._jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    def _qpos_address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"the operational space controller needs a joint named {name}")
        return int(self.model.jnt_qposadr[joint])

    def _dof_address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[joint])

    def convert(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        delta = np.asarray(action, dtype=np.float64).reshape(2, 7)[0]
        position = np.asarray(data.site_xpos[self._site], dtype=np.float64).copy()
        rotation = np.asarray(data.site_xmat[self._site], dtype=np.float64).reshape(3, 3).copy()

        goal_position = position + delta[:3] * self.config.cartesian_translation_scale_m
        goal_rotation = rotation.copy()
        if np.any(np.abs(delta[3:6]) > 1e-4):
            # `mju_quatIntegrate` composes about the *tool's* axes; expressing the delta in
            # those axes makes it `exp(delta) @ q`, which is the world-frame composition
            # robosuite's `set_goal_orientation` performs.  The same conversion, and for the
            # same reason, as `CartesianDeltaAdapter`.
            local = rotation.T @ (delta[3:6] * self.config.cartesian_rotation_scale_rad)
            goal_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(goal_quaternion, rotation.reshape(9))
            mujoco.mju_quatIntegrate(goal_quaternion, local, 1.0)
            # `mju_quat2Mat` wants the *result* flat: a 3x3 view is rejected outright,
            # which it was -- the first rotation command a policy ever sent killed the
            # run at its first step, and the teardown noise in the log made it look like
            # a rendering failure.
            flat = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(flat, goal_quaternion)
            goal_rotation = flat.reshape(3, 3)

        mujoco.mj_jacSite(
            self.model, data, self._jacobian_position, self._jacobian_rotation, self._site
        )
        jacobian = np.vstack(
            [
                self._jacobian_position[:, self._left_dof],
                self._jacobian_rotation[:, self._left_dof],
            ]
        )
        current_quaternion = np.empty(4, dtype=np.float64)
        goal_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(current_quaternion, rotation.reshape(9))
        mujoco.mju_mat2Quat(goal_quaternion, goal_rotation.reshape(9))
        rotation_error = np.empty(3, dtype=np.float64)
        mujoco.mju_subQuat(rotation_error, goal_quaternion, current_quaternion)

        velocity = data.qvel[np.asarray(self._left_dof, dtype=np.int32)]
        task_velocity = jacobian @ velocity
        error = np.r_[
            self.kp * (goal_position - position),
            self.kp * rotation_error,
        ]
        # **The dynamically consistent inverse, which is what makes the gains mean what
        # robosuite says they mean.**  The plain Jacobian transpose was tried first and is not
        # equivalent: a full +x command moved the tool **124 mm** where robosuite's
        # `output_max` is 50, because an unweighted task force on a light arm is an
        # acceleration the gains were never scaled for.  robosuite multiplies the task force
        # through `Lambda = (J M^-1 J^T)^-1`, so the loop behaves as the second-order system
        # `kp` and `damping_ratio` describe whatever the arm's inertia is.
        dofs = np.asarray(self._left_dof, dtype=np.int32)
        mass = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, mass, data.qM)
        arm_mass = mass[np.ix_(dofs, dofs)]
        lam = jacobian @ np.linalg.inv(arm_mass) @ jacobian.T
        lam = np.linalg.inv(lam + 1e-9 * np.eye(6))
        torque = jacobian.T @ (lam @ (error - self.kd * task_velocity))
        # Gravity and Coriolis, so the loop only has to do the tracking.  This is the term
        # the joint-position servo got for free from being a servo.
        torque = torque + data.qfrc_bias[dofs]
        torque = np.clip(torque, self._limits[:, 0], self._limits[:, 1])

        right = np.asarray([data.qpos[address] for address in self._right_qpos], dtype=np.float64)
        return np.ascontiguousarray(
            np.r_[torque, (delta[6] + 1.0) / 2.0, right, 1.0], dtype=np.float64
        )

    def hold(self, data: mujoco.MjData) -> np.ndarray:
        """The torques that hold the arm where it already is, for a reset's settling steps.

        A reset writes the deployment home's **joint angles** into `ctrl`, which is right for
        the position actuators this harness was built around and wrong for torque ones: the
        arm was handed -2.45 N.m where zero was wanted and finished the hundred settling
        steps **19.4 degrees** from LIBERO's home -- before the first policy step, in the
        scene whose whole purpose is that the arm starts where LIBERO's starts.

        What a torque-driven arm wants there is its bias force, which is gravity and
        Coriolis, and nothing else: that is what makes it stay put.
        """

        dofs = np.asarray(self._left_dof, dtype=np.int32)
        torque = np.clip(data.qfrc_bias[dofs], self._limits[:, 0], self._limits[:, 1])
        right = np.asarray([data.qpos[address] for address in self._right_qpos], dtype=np.float64)
        return np.ascontiguousarray(np.r_[torque, 1.0, right, 1.0], dtype=np.float64)
