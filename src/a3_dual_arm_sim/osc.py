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
It is robosuite's law with robosuite's own numbers, read back off the controller LIBERO
configures -- `kp = 150` (six-vector), `kd = 2 sqrt(kp) damping_ratio` = 24.4949,
`output_max = [0.05 x3, 0.5 x3]`, `control_delta = True`, `uncoupling = True`,
`impedance_mode = "fixed"`, no interpolation:

    desired_force  = kp * (goal_pos - pos) - kd * J_pos qvel
    desired_torque = kp * ori_error       - kd * J_ori qvel
    wrench         = [ (J_pos M^-1 J_pos^T)^-1 desired_force ;
                       (J_ori M^-1 J_ori^T)^-1 desired_torque ]
    tau            = J^T wrench + qfrc_bias + N^T M (kp_n (q0 - q) - kd_n qvel)

with `ori_error` the axis-angle of `R_goal R_current^T`, the goal orientation composed as
`R_err R_current` (the world-frame composition `set_goal_orientation` performs, and the same
one `CartesianDeltaAdapter` implements), and `N = I - Jbar J` the dynamic nullspace projector.

The position and orientation halves are weighted **separately** because that is what
`uncouple_pos_ori = True` means; a single 6x6 `Lambda` was tried and is measurably not the
same loop.  The nullspace term is the third piece robosuite adds and the first version of
this file did not, and the reference arm's step response is not reproducible without it.
`ramp_ratio` (robosuite's soft start) does not exist in the version LIBERO pins.

The goal is set from the **current** measurement every control step, which is what
`control_delta = True` means: a delta is a request relative to where the arm is, not a
running sum of requests, so a missed command is not accumulated forever.
"""

from __future__ import annotations

import math
import os

import mujoco
import numpy as np

from .config import SimConfig
from .contracts import LEFT_JOINTS, RIGHT_JOINTS

#: robosuite's `OSC_POSE` gains, from the controller LIBERO configures for its Panda.
KP = 150.0
DAMPING_RATIO = 1.0

#: The nullspace posture goal, and its gains.
#:
#: robosuite adds a third term this file used to leave out: joint torques that pull the arm
#: towards a reference posture with the gains `joint_kp = 10`, `joint_kv = 2 sqrt(10)`,
#: projected into the task's null space.  Measured on the live controller LIBERO builds at
#: reset, `initial_joint` is the constant below -- it is set when the controller is
#: constructed (`self.initial_joint = self.joint_pos`) and nothing in robosuite or LIBERO
#: ever calls `update_initial_joints`, so it stays the model's own home rather than the
#: episode's start pose.
#:
#: It matters because the arm is redundant: seven joints for a six-dimensional task leaves
#: one direction in which the posture can drift without moving the tool, and without this
#: term nothing pulls it back.  It is also part of why the reference arm's step response is
#: the shape it is, so it is copied rather than approximated.
INITIAL_JOINT_RAD = (0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, 0.7854)
JOINT_KP = 10.0

#: How often the torque is recomputed for a latched goal.  This is robosuite's structure --
#: a goal that moves at the control rate, a torque that is re-solved every physics step --
#: and it is what the arm was tuned against.  The knob exists so that the alternative can be
#: measured rather than argued about: `once` computes a torque when the goal is set and
#: holds it for the whole control step, which is what this harness did before, and which
#: lets the arm overshoot the goal and stop further inside the bowl than the goal asked for.
#:
#: Measured, that overshoot is not a defect of the old loop so much as a source of depth the
#: policy does not command: `once` put the fingertip 2.3 mm from the bowl's axis (the
#: reference's own 1.0 mm, and three placements in forty episodes), while `always` stops at
#: 25.9 mm and never places.  An environment variable rather than a config key because it is
#: a property of the experiment, not of the scene, and nothing that ships should set it.
RETORQUE_EVERY_SUBSTEP = os.environ.get("A3_OSC_RETORQUE", "always").strip().lower() != "once"


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
        self._actuators = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            for name in LEFT_JOINTS
        )
        self._limits = np.asarray(
            [model.actuator_ctrlrange[actuator] for actuator in self._actuators],
            dtype=np.float64,
        )
        self.kp = KP
        self.kd = 2.0 * math.sqrt(KP) * DAMPING_RATIO
        self._initial_joint = np.asarray(INITIAL_JOINT_RAD, dtype=np.float64)
        # The latched goal.  robosuite's `SingleArm.control` sets the goal **once per policy
        # step** and then calls `run_controller` on **every physics substep**:
        #
        #     if policy_step:
        #         self.controller.set_goal(arm_action)
        #     torques = self.controller.run_controller()
        #
        # so the arm is inside a 500 Hz feedback loop during the twenty-five substeps of a
        # control step, with a goal that only moves at 20 Hz.  Computing the torque once and
        # holding it for the whole control step -- which is what this harness did -- is a
        # different, open-loop-between-steps system, and measurably a faster one: 3.9 mm on
        # the first step against the reference's 2.8, 64 mm against 52 over ten.
        self._goal_position: np.ndarray | None = None
        self._goal_rotation: np.ndarray | None = None
        self._goal_gripper = 0.5
        if self._initial_joint.shape != (len(LEFT_JOINTS),):
            raise RuntimeError(
                f"the nullspace goal needs one angle per arm joint, got "
                f"{self._initial_joint.shape}"
            )
        self._jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        self._jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    @property
    def rotation_frame(self) -> str:
        """Which frame this controller reads a `cartesian_delta`'s rotation part in.

        **Measured, because the code reads the other way.**  `convert` takes
        ``rotation.T @ (delta * scale)`` and hands it to `mju_quatIntegrate`, which composes
        about the tool's axes -- so the arithmetic says a caller should send a *world*-frame
        rotation.  It does not.  Driving one arm towards a goal a known angle away and watching
        whether the distance to that goal falls or rises, for both conventions and both arms
        (`scripts/probe_rotation_sign.py`):

            caller assumes   A3 arm (CartesianDeltaAdapter)   Panda (this controller)
            world            0.300 -> 0.033 rad  converges    0.300 -> 0.683  diverges
            tool             0.300 -> 0.807 rad  diverges     0.300 -> 0.090  converges

        So the two controllers disagree, and there is no derivation to appeal to -- which is why
        this is a property of the controller rather than a constant in the caller.  Sending the
        wrong one does not produce a small error: the arm turns *away* until the two orientations
        are near antipodal, then the shorter arc flips and it oscillates there, which is how a
        Panda rollout spends 900 steps in APPROACH with a rotation error parked at 3.1 rad.
        """

        return "tool"

    def _qpos_address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"the operational space controller needs a joint named {name}")
        return int(self.model.jnt_qposadr[joint])

    @property
    def left_site(self) -> int:
        """The site whose pose the goal is expressed in.

        Public because a caller that wants to command this controller has to build its delta at
        **this** point, and the answer is not `L_eef`.  The two are rigidly attached to the same
        hand but they are not the same point, so a delta computed at one of them moves the other
        by that delta and leaves it short of its own goal by the offset between them -- which is
        81 mm here, and which the arm would hold as a steady error rather than report.
        """

        return int(self._site)

    def _dof_address(self, name: str) -> int:
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[joint])

    def convert(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        # Make `data` describe the state the arm is actually in before reading the pose the
        # goal is built from.  robosuite's `set_goal` opens with `self.update()`, which calls
        # `sim.forward()` for exactly this reason.  Without it the goal is latched from
        # whatever `site_xpos` last held -- and a reset that writes `qpos` and goes straight
        # into settling leaves it holding the *previous* scene's pose: measured, the arm
        # drove **9.1 degrees** away from the home it had just been given, and sat there.
        mujoco.mj_forward(self.model, data)
        delta = np.asarray(action, dtype=np.float64).reshape(2, 7)[0]
        position = np.asarray(data.site_xpos[self._site], dtype=np.float64).copy()
        rotation = np.asarray(data.site_xmat[self._site], dtype=np.float64).reshape(3, 3).copy()

        self._goal_gripper = (float(delta[6]) + 1.0) / 2.0
        self._goal_position = position + delta[:3] * self.config.cartesian_translation_scale_m
        self._goal_rotation = rotation.copy()
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
            self._goal_rotation = flat.reshape(3, 3)

        return self._control(data)

    def retorque(self, data: mujoco.MjData) -> None:
        """Recompute the arm's torques for the latched goal and write them to `ctrl`.

        Call this before **every** physics substep of a control step, which is what
        robosuite does: the goal is held, the loop is not.  The caller is responsible for
        making `data` current first (`mj_forward`), for the same reason robosuite calls
        `sim.forward()` at the top of each substep.
        """

        if self._goal_position is None:
            raise RuntimeError("retorque needs a goal; call convert() first")
        for actuator, torque in zip(self._actuators, self._arm_torque(data), strict=True):
            data.ctrl[actuator] = torque

    def _control(self, data: mujoco.MjData) -> np.ndarray:
        torque = self._arm_torque(data)
        right = np.asarray([data.qpos[address] for address in self._right_qpos], dtype=np.float64)
        return np.ascontiguousarray(
            np.r_[torque, self._goal_gripper, right, 1.0], dtype=np.float64
        )

    def _arm_torque(self, data: mujoco.MjData) -> np.ndarray:
        assert self._goal_position is not None and self._goal_rotation is not None
        position = np.asarray(data.site_xpos[self._site], dtype=np.float64).copy()
        rotation = np.asarray(data.site_xmat[self._site], dtype=np.float64).reshape(3, 3).copy()
        goal_position, goal_rotation = self._goal_position, self._goal_rotation
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
        position_velocity = self._jacobian_position[:, self._left_dof] @ velocity
        rotation_velocity = self._jacobian_rotation[:, self._left_dof] @ velocity
        # robosuite's `run_controller`, term for term:
        #
        #     desired_force  = kp * position_error  + kd * (-ee_pos_vel)
        #     desired_torque = kp * ori_error       + kd * (-ee_ori_vel)
        #
        # with `kp` and `kd` six-vectors and `kd = 2 sqrt(kp) damping_ratio`.
        desired_force = self.kp * (goal_position - position) - self.kd * position_velocity
        desired_torque = self.kp * rotation_error - self.kd * rotation_velocity

        dofs = np.asarray(self._left_dof, dtype=np.int32)
        mass = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, mass, data.qM)
        arm_mass = mass[np.ix_(dofs, dofs)]
        arm_mass_inv = np.linalg.inv(arm_mass)
        # **`uncouple_pos_ori=True`, which is what LIBERO configures.**  robosuite weights
        # the position force through `Lambda_pos = (J_pos M^-1 J_pos^T)^-1` and the
        # orientation torque through `Lambda_ori` *separately*, not through one 6x6
        # `Lambda_full`.  Measured, this is the difference between the arm delivering 7.6 mm
        # of a 25 mm command per control step and 6.0 mm; the coupling between the position
        # and orientation halves is what the reference arm does not have.
        lambda_pos = np.linalg.pinv(
            self._jacobian_position[:, self._left_dof] @ arm_mass_inv
            @ self._jacobian_position[:, self._left_dof].T
        )
        lambda_ori = np.linalg.pinv(
            self._jacobian_rotation[:, self._left_dof] @ arm_mass_inv
            @ self._jacobian_rotation[:, self._left_dof].T
        )
        decoupled = np.r_[lambda_pos @ desired_force, lambda_ori @ desired_torque]
        torque = jacobian.T @ decoupled
        # Gravity and Coriolis, so the loop only has to do the tracking.  This is the term
        # the joint-position servo got for free from being a servo.
        torque = torque + data.qfrc_bias[dofs]
        # The nullspace term.  `nullspace_matrix = I - Jbar J`, with
        # `Jbar = M^-1 J^T Lambda_full` -- the full 6x6 Lambda, unlike the task term above,
        # because this is the projector and not the control.
        lambda_full = np.linalg.pinv(jacobian @ arm_mass_inv @ jacobian.T)
        jacobian_bar = arm_mass_inv @ jacobian.T @ lambda_full
        nullspace = np.eye(len(dofs)) - jacobian_bar @ jacobian
        posture = arm_mass @ (
            JOINT_KP
            * (self._initial_joint - data.qpos[np.asarray(self._left_qpos, dtype=np.int32)])
            - 2.0 * math.sqrt(JOINT_KP) * velocity
        )
        torque = torque + nullspace.T @ posture
        return np.clip(torque, self._limits[:, 0], self._limits[:, 1])

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
