from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .actions import CartesianDeltaAdapter
from .config import SimConfig, load_config
from .osc import OperationalSpaceAdapter
from . import osc as retorque
from .contracts import (
    ARM_JOINTS,
    CARTESIAN_ACTION_DIM,
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    JOINT_ACTION_DIM,
    LEFT_JOINTS,
    LEFT_WRIST_IMAGE,
    RIGHT_JOINTS,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    ActionMode,
    validate_action,
    validate_observation,
)
from .model import (
    ROBOTIQ_2F85_JAW_TRAVEL_M,
    ModelBundle,
    SceneName,
    build_model,
)


class A3DualArmEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Functional A3 tabletop sandbox with a policy-neutral API."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }
    # Robotiq 2F-85: each simplified jaw travels 42.5 mm for an 85 mm opening.
    GRIPPER_RANGE_M = ROBOTIQ_2F85_JAW_TRAVEL_M

    def __init__(
        self,
        config: SimConfig | str | Path | None = None,
        *,
        action_mode: ActionMode = "joint_position",
        render_mode: str | None = None,
        render_cameras: bool = True,
        scene: SceneName = "sandbox",
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, SimConfig) else load_config(config)
        if action_mode not in ("joint_position", "cartesian_delta"):
            raise ValueError(f"unsupported action mode: {action_mode}")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render mode: {render_mode}")
        self.action_mode = action_mode
        self.render_mode = render_mode
        # Keep the observation schema fixed in fast debug mode: disabled policy
        # cameras produce black frames without constructing an offscreen GL context.
        self.render_cameras = render_cameras
        self.scene = scene
        bundle: ModelBundle = build_model(self.config, scene=scene)
        self.model = bundle.model
        self.model_xml = bundle.xml
        self.source_joints = bundle.source_joints
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._view_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._viewer: Any = None
        self._key_callback: Any = None
        self._step_count = 0
        self._max_recorded_gripper_force = 0.0
        self._safety_stop = False
        self._safety_reason: str | None = None
        self._last_applied_action = np.zeros(JOINT_ACTION_DIM, dtype=np.float64)
        self._joint_ids = {name: self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS}
        self._qpos_ids = {
            name: int(self.model.jnt_qposadr[jid]) for name, jid in self._joint_ids.items()
        }
        self._dof_ids = {
            name: int(self.model.jnt_dofadr[jid]) for name, jid in self._joint_ids.items()
        }
        def _arm_actuator(name: str) -> int:
            """``{name}_position``, or ``{name}_motor`` where the arm is torque driven.

            A scene that uses robosuite's robot is driven by operational-space control, so its
            actuators are motors with the robot's own torque ranges; see
            `osc.OperationalSpaceAdapter`.  `_apply_controls` writes `ctrl` either way, and the
            actuator's own control range is what bounds the value.
            """

            for suffix in ("_position", "_motor"):
                identifier = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}{suffix}"
                )
                if identifier >= 0:
                    return identifier
            raise ValueError(f"model has no position or motor actuator for {name}")

        self._actuator_ids = {name: _arm_actuator(name) for name in ARM_JOINTS}
        self._finger_joints = {
            side: tuple(
                self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_finger_{finger}_joint")
                for finger in ("inner", "outer")
            )
            for side in ("L", "R")
        }
        self._finger_actuators = {
            side: tuple(
                self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger_{finger}_position")
                for finger in ("inner", "outer")
            )
            for side in ("L", "R")
        }
        # Finger travel and drive sign, **read from the model rather than assumed**.  The A3
        # gripper's jaws share a travel and are driven with the same sign, because their
        # joint axes are opposed.  robosuite's Panda gripper gives both jaws the *same* axis
        # and two mirrored ranges -- `[-0.04, 0]` and `[0, 0.04]` -- so its jaws need
        # opposite signs, and its state has to read the same way the A3's does or the
        # opening a policy sees would depend on which robot is driving.
        self._finger_travel = {}
        self._finger_sign = {}
        for side in ("L", "R"):
            self._finger_travel[side] = max(
                abs(float(self.model.jnt_range[joint_id][1]))
                for joint_id in self._finger_joints[side]
            )
            self._finger_sign[side] = tuple(
                # The sign of the joint coordinate at the **travel** end of its range: the
                # end with the larger magnitude.  Not `upper >= 0`, which is what this was,
                # and which silently gave the Panda's outer jaw +1 -- its range is
                # [-0.04, 0], so its upper bound *is* zero.  The jaw was then commanded out
                # of its own control range, clipped to zero, and never moved: one jaw of a
                # two-jaw gripper, with the state reporting half an opening (0.486 against
                # LIBERO's 0.97 at the same pose) because only one of the two `|q|` was ever
                # nonzero.
                1.0
                if abs(float(self.model.jnt_range[joint_id][1]))
                >= abs(float(self.model.jnt_range[joint_id][0]))
                else -1.0
                for joint_id in self._finger_joints[side]
            )
        # Which end of the travel is "open" is a property of the mechanism, and the two
        # grippers in this scene disagree about it.  The A3's jaws read **zero when they are
        # apart** and close towards their travel; robosuite's Panda reads zero when they are
        # *together* and opens to +/-0.04.  Each is self-consistent, so the formula below is
        # right for exactly one of them -- and it cannot be inferred from the ranges, because
        # the A3's joints are one-signed ([0, 0.0425]) while the Panda's are mirrored
        # ([0, 0.04] and [-0.04, 0]).  So it is stated, not guessed.
        #
        # Getting it wrong is not cosmetic: `opening` is what the eight-wide state reports,
        # and with the Panda's jaws held closed the state told the policy they were open --
        # the exact opposite of LIBERO's `robot0_gripper_qpos`, which is [0.0387, -0.0387]
        # at the start of every one of its episodes.
        panda_left = getattr(self.config, "left_arm", "a3") == "panda"
        self._finger_open_at_zero = {"L": not panda_left, "R": True}
        if os.environ.get("A3_GRIPPER_LEGACY", "0").strip() == "1":
            # Ablation: the three defects this harness had before they were found.  The
            # outer jaw's sign came from `upper >= 0` on a range of `[-0.04, 0]`, so it was
            # commanded out of its own control range and never moved; the opening read with
            # the A3's polarity; and every reset started with the jaws shut.
            self._finger_sign["L"] = tuple(
                1.0 if float(self.model.jnt_range[j][1]) >= 0.0 else -1.0
                for j in self._finger_joints["L"]
            )
            self._finger_open_at_zero["L"] = True
        self._eef_sites = (
            self._id(mujoco.mjtObj.mjOBJ_SITE, "L_eef"),
            self._id(mujoco.mjtObj.mjOBJ_SITE, "R_eef"),
        )
        self._camera_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("front", "left_wrist", "right_wrist")
        }
        if getattr(self.config, "left_arm", "a3") == "panda":
            # LIBERO drives this arm with robosuite's operational-space controller, so this
            # scene does too -- see `osc.OperationalSpaceAdapter` for why a joint-position
            # servo does not survive the swap.  The site is **robosuite's own grasp site**,
            # which is the frame its controller tracks, and not `L_eef`, which carries the
            # two frames LIBERO's *state* is reported in.
            self._ik = OperationalSpaceAdapter(
                self.model, self.config, site="gripper0_grip_site"
            )
        else:
            self._ik = CartesianDeltaAdapter(self.model, self.config)
        self._arm_ranges = np.asarray(
            [self.model.jnt_range[self._joint_ids[name]] for name in ARM_JOINTS],
            dtype=np.float64,
        )
        joint_low = np.r_[self._arm_ranges[:7, 0], 0.0, self._arm_ranges[7:, 0], 0.0]
        joint_high = np.r_[self._arm_ranges[:7, 1], 1.0, self._arm_ranges[7:, 1], 1.0]
        if self.action_mode == "joint_position":
            self.action_space = spaces.Box(joint_low, joint_high, dtype=np.float64)
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (CARTESIAN_ACTION_DIM,), dtype=np.float64)
        image_shape = (self.config.image_height, self.config.image_width, 3)
        self.observation_space = spaces.Dict(
            {
                FRONT_IMAGE: spaces.Box(0, 255, image_shape, dtype=np.uint8),
                LEFT_WRIST_IMAGE: spaces.Box(0, 255, image_shape, dtype=np.uint8),
                RIGHT_WRIST_IMAGE: spaces.Box(0, 255, image_shape, dtype=np.uint8),
                STATE: spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
                VELOCITY: spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
                EEF_POSE: spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
                FORCE: spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),
                "time": spaces.Box(0.0, np.inf, (), dtype=np.float64),
                "safety_stop": spaces.Discrete(2),
            }
        )
        self.metadata = dict(self.metadata, render_fps=self.config.control_hz)

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = mujoco.mj_name2id(self.model, object_type, name)
        if result < 0:
            raise ValueError(f"model is missing {object_type.name} {name}")
        return result

    def _id_or_none(self, object_type: mujoco.mjtObj, name: str) -> int:
        """Look up an optional model element, returning -1 when it is absent.

        Scenes are built from shared code, so some elements only exist for some
        configs (the spare box, for one).  Callers that can work without an
        element should not have to know which configs include it.
        """

        return mujoco.mj_name2id(self.model, object_type, name)

    @property
    def last_applied_action(self) -> np.ndarray:
        return self._last_applied_action.copy()

    @property
    def canonical_action_mode(self) -> str:
        """What this environment's own command vector actually *is*, as a name.

        ``step`` produces two arrays: the one the caller asked for
        (``info["requested_action"]``) and the one that reached the actuators
        (``info["applied_action"]``).  They are the same shape, and they mean the
        same thing only while the arm is a joint-servo arm.  Under
        `OperationalSpaceAdapter` the left arm's seven entries are the **torques** the
        controller solved for the latched goal -- `_limit_joint_action` says so and
        deliberately neither clips nor rate-limits them -- and there is no name for
        that among the action spaces, because it is not one.

        So this raises instead of returning a label.  A recorder that stores
        ``applied_action`` has to ask, and the answer for a torque-driven arm is "store
        the requested action instead", which is the space the policy emits and the one
        a fine-tune can use.  Returning ``"joint_position"`` here is what would put
        Newton-metres in a field a reader takes for radians.
        """

        if isinstance(self._ik, OperationalSpaceAdapter):
            # `noqa: TRY004` -- the rule reads an `isinstance` guard followed by `ValueError`
            # as a type check wanting `TypeError`.  This is not a type check: a torque-driven
            # arm is a legitimate configuration, and the error is that *this question* has no
            # answer for it, which is what every other contract violation in this package
            # raises (`ContractError` is a `ValueError`).
            raise ValueError(  # noqa: TRY004
                "this scene's left arm is driven by operational-space control, so the "
                "applied action is joint torque and belongs to no action space; record "
                "`requested_action` (the space the policy emits) rather than "
                "`applied_action`"
            )
        return "joint_position"

    @property
    def current_joint_action(self) -> np.ndarray:
        return self._current_joint_action()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._safety_stop = False
        self._safety_reason = None
        homes = dict(zip(LEFT_JOINTS, self.config.home.left, strict=True))
        homes.update(dict(zip(RIGHT_JOINTS, self.config.home.right, strict=True)))
        for name, value in homes.items():
            joint_id = self._joint_ids[name]
            limits = self.model.jnt_range[joint_id]
            self.data.qpos[self._qpos_ids[name]] = np.clip(value, limits[0], limits[1])
        for side, opening in zip(("L", "R"), self.config.home.grippers, strict=True):
            physical = self._finger_travel_fraction(side, opening) * self._finger_travel[side]
            for sign, joint_id in zip(
                self._finger_sign[side], self._finger_joints[side], strict=True
            ):
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = sign * physical
        self._randomize_objects(enabled=(options or {}).get("randomize_objects", True))
        mujoco.mj_forward(self.model, self.data)
        home_action = np.r_[
            self.config.home.left,
            self.config.home.grippers[0],
            self.config.home.right,
            self.config.home.grippers[1],
        ]
        self._apply_controls(home_action)
        self.settle()
        self.data.time = 0.0
        self._max_recorded_gripper_force = 0.0
        self._last_applied_action = np.asarray(home_action, dtype=np.float64)
        if self.render_mode == "human":
            self._ensure_viewer()
        observation = self._observation()
        return observation, {
            "seed": seed,
            "action_mode": self.action_mode,
            "scene": self.scene,
            "policy_camera_rendering": self.render_cameras,
        }

    def _randomize_objects(self, *, enabled: bool) -> None:
        if not enabled:
            return
        noise = self.config.object_position_noise_m
        for index in range(3):
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"object_{index}_free")
            address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[address] += self.np_random.uniform(-noise, noise)
            self.data.qpos[address + 1] += self.np_random.uniform(-noise, noise)

    def emergency_stop(self, reason: str = "user emergency stop") -> None:
        self._safety_stop = True
        self._safety_reason = reason

    def set_key_callback(self, callback: Any) -> None:
        if self._viewer is not None:
            raise RuntimeError("key callback must be set before opening the viewer")
        self._key_callback = callback

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        requested = validate_action(action, self.action_mode)
        if self._safety_stop:
            applied = self._current_joint_action()
            # Nothing was commanded, so what the controllers were told to track is what they are
            # already tracking.  A stopped arm's correction is zero, which is what the
            # comparison should read rather than a missing value.
            commanded = applied.copy()
        else:
            canonical = (
                requested
                if self.action_mode == "joint_position"
                else self._ik.convert(self.data, requested)
            )
            if self.action_mode == "cartesian_delta":
                # An idle arm holds its commanded pose, not its gravity-deflected
                # measurement. Re-targeting the measurement integrates sag forever.
                for offset, start in ((0, 0), (7, 8)):
                    if not np.any(requested[offset:offset + 6]):
                        canonical[start:start + 7] = self._last_applied_action[start:start + 7]
            #: The targets the controllers are about to be given, one step before the
            #: environment's own limits are applied to them.  Recorded because
            #: ``applied_action`` is always a sixteen-wide joint vector while
            #: ``requested_action`` is whatever the *policy* emitted -- fourteen Cartesian
            #: values for this scene -- and the two cannot be subtracted.  This one is in the
            #: environment's own space and is what "how far did the environment have to move
            #: the command" is actually a question about.  On a joint-target arm it is equal
            #: to ``requested_action`` by construction, so nothing about those runs changes.
            commanded = canonical.copy()
            applied = self._limit_joint_action(canonical)
            self._apply_controls(applied)
            for _ in range(self.config.substeps):
                self._refresh_operational_space()
                mujoco.mj_step(self.model, self.data)
                if not np.all(np.isfinite(self.data.qpos)) or not np.all(
                    np.isfinite(self.data.qvel)
                ):
                    self.emergency_stop("non-finite simulator state")
                    break
                if np.max(np.abs(self.data.qvel)) > 80.0:
                    self.emergency_stop("joint velocity exceeded safety threshold")
                    break
        self._last_applied_action = applied.copy()
        self._step_count += 1
        terminated = self._safety_stop
        truncated = self._step_count >= self.config.horizon and not terminated
        if self.render_mode == "human" and self._viewer is not None:
            self._update_viewer_force_hud()
            self._viewer.sync()
        observation = self._observation()
        info = {
            "applied_action": applied.copy(),
            "requested_action": requested.copy(),
            "commanded_action": commanded.copy(),
            "safety_reason": self._safety_reason,
            "step": self._step_count,
        }
        return observation, 0.0, terminated, truncated, info

    def _limit_joint_action(self, canonical: np.ndarray) -> np.ndarray:
        command = validate_action(canonical, "joint_position").copy()
        # Rate-limit the commanded trajectory, not the measured state. Limiting against
        # measured qpos caps the position error (and therefore actuator torque), which can
        # make gravity compensation impossible for the shoulder joints.
        current = self._last_applied_action
        arm_command = np.r_[command[:7], command[8:15]]
        arm_current = np.r_[current[:7], current[8:15]]
        if isinstance(self._ik, OperationalSpaceAdapter):
            # The left arm's seven entries are **torques**, not joint targets: there is no
            # trajectory to rate-limit and no joint range to clip them to, and doing either
            # would destroy them.  Their motors' own control ranges bound them, which is where
            # robosuite's torque limits live.  The right arm is still a joint-target arm here,
            # and it is inert.
            right = np.clip(arm_command[7:], self._arm_ranges[7:, 0], self._arm_ranges[7:, 1])
            return np.r_[arm_command[:7], command[7], right, command[15]]
        arm_command = np.clip(arm_command, self._arm_ranges[:, 0], self._arm_ranges[:, 1])
        if self.action_mode == "cartesian_delta":
            arm_delta = arm_command - arm_current
            max_delta = float(np.max(np.abs(arm_delta)))
            if max_delta > self.config.max_joint_step_rad:
                arm_delta = arm_delta * (self.config.max_joint_step_rad / max_delta)
            arm_command = np.clip(
                arm_current + arm_delta,
                self._arm_ranges[:, 0],
                self._arm_ranges[:, 1],
            )
        else:
            arm_command = np.clip(
                arm_command,
                arm_current - self.config.max_joint_step_rad,
                arm_current + self.config.max_joint_step_rad,
            )
        command[:7] = arm_command[:7]
        command[8:15] = arm_command[7:]
        for index in (7, 15):
            command[index] = np.clip(command[index], 0.0, 1.0)
            command[index] = np.clip(
                command[index],
                current[index] - self.config.max_gripper_step,
                current[index] + self.config.max_gripper_step,
            )
        return command

    def settle(self, steps: int = 100) -> None:
        """Run ``steps`` of physics while **holding** the pose, as a reset does.

        A torque-driven arm cannot be left to a one-shot bias force.  Balancing gravity exactly
        is a *neutral* equilibrium with no restoring term -- it neither holds the pose nor
        returns to it, so any numerical perturbation is free to grow.  Measured at LIBERO's
        home: the arm ended 8.48 degrees out with the bias force, and **44.7** when that force
        was recomputed from the pose it was actually holding, which is the wrong direction for
        a better force and the right one for no force at all.

        What holds a pose is a controller.  So the operational-space loop is closed here, one
        control step at a time, with a zero delta -- which asks for exactly "stay where you
        are".  A position-servo arm needs none of this: its servo is already a restoring term.
        """

        close_the_loop = isinstance(self._ik, OperationalSpaceAdapter)
        zero = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
        for _ in range(steps):
            if close_the_loop:
                # Latch the goal at the current pose, then hold it for a whole control
                # step's worth of substeps -- the same shape as `step`, so the settling is
                # the same loop the policy will drive.
                self._ik.convert(self.data, zero)
                for _ in range(self.config.substeps):
                    self._refresh_operational_space()
                    mujoco.mj_step(self.model, self.data)
                continue
            mujoco.mj_step(self.model, self.data)

    @property
    def ik(self):
        """The left arm's Cartesian controller, so the bridge can share this one.

        One controller and therefore one goal.  The substep loop re-computes the arm's
        torques from **this** adapter's latched goal every 2 ms; a bridge holding its own
        adapter would latch a goal nobody reads, and every command it converted would be
        overwritten before the first physics step.
        """

        return self._ik

    @ik.setter
    def ik(self, value) -> None:
        self._ik = value

    def _refresh_operational_space(self) -> None:
        """Recompute the operational-space torque for the goal this control step latched.

        robosuite's `SingleArm.control` sets the goal **once per policy step** and then calls
        `run_controller` on **every physics substep**:

            if policy_step:
                self.controller.set_goal(arm_action)
            torques = self.controller.run_controller()

        so its arm sits in a 500 Hz feedback loop for the twenty-five substeps of a control
        step, with a goal that only moves at 20 Hz.  Computing the torque once and holding it
        for the whole control step -- what this harness did until now -- is a different
        system, and measurably a quicker one: 3.9 mm on the first step of a 25 mm command
        against the reference's 2.8, and 64 mm against 52 over ten steps.

        `mj_forward` first: after `mj_step` the derived quantities still describe the state
        at the *start* of that step, and the torque has to be computed from the state the arm
        is actually in.  robosuite does exactly this, with `sim.forward()` at the top of each
        substep.
        """

        if not isinstance(self._ik, OperationalSpaceAdapter):
            return
        if not retorque.RETORQUE_EVERY_SUBSTEP:
            # The isolation experiment: torque solved once per control step, at the moment
            # the goal is latched, and held for the substeps that follow.
            return
        mujoco.mj_forward(self.model, self.data)
        self._ik.retorque(self.data)

    def _apply_controls(self, canonical: np.ndarray) -> None:
        arm_values = np.r_[canonical[:7], canonical[8:15]]
        for name, value in zip(ARM_JOINTS, arm_values, strict=True):
            self.data.ctrl[self._actuator_ids[name]] = value
        for side, opening in zip(("L", "R"), (canonical[7], canonical[15]), strict=True):
            value = self._finger_travel_fraction(side, opening) * self._finger_travel[side]
            for sign, actuator_id in zip(
                self._finger_sign[side], self._finger_actuators[side], strict=True
            ):
                self.data.ctrl[actuator_id] = sign * value

    def _finger_travel_fraction(self, side: str, opening: float) -> float:
        """The fraction of the jaw travel that ``opening`` commands.

        One definition, shared by the reset, the servo and the state read, so the polarity
        in `_finger_open_at_zero` cannot drift between the three places it matters.
        """
        value = float(np.clip(opening, 0.0, 1.0))
        return (1.0 - value) if self._finger_open_at_zero[side] else value

    def _current_joint_action(self) -> np.ndarray:
        left = [self.data.qpos[self._qpos_ids[name]] for name in LEFT_JOINTS]
        right = [self.data.qpos[self._qpos_ids[name]] for name in RIGHT_JOINTS]
        grips = [self._gripper_state(side)[0] for side in ("L", "R")]
        return np.asarray([*left, grips[0], *right, grips[1]], dtype=np.float64)

    def _gripper_state(self, side: str) -> tuple[float, float]:
        positions = []
        velocities = []
        for joint_id in self._finger_joints[side]:
            positions.append(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            velocities.append(self.data.qvel[self.model.jnt_dofadr[joint_id]])
        travel = self._finger_travel[side]
        signs = self._finger_sign[side]
        fraction = np.mean([abs(p) for p in positions]) / travel
        drift = np.mean([sign * v for sign, v in zip(signs, velocities, strict=True)]) / travel
        if self._finger_open_at_zero[side]:
            opening, rate = 1.0 - fraction, -drift
        else:
            opening, rate = fraction, drift
        return (float(opening), float(rate))

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[address : address + dimension].copy()

    def _observation(self) -> dict[str, Any]:
        left_state = [self.data.qpos[self._qpos_ids[name]] for name in LEFT_JOINTS]
        right_state = [self.data.qpos[self._qpos_ids[name]] for name in RIGHT_JOINTS]
        left_velocity = [self.data.qvel[self._dof_ids[name]] for name in LEFT_JOINTS]
        right_velocity = [self.data.qvel[self._dof_ids[name]] for name in RIGHT_JOINTS]
        left_grip = self._gripper_state("L")
        right_grip = self._gripper_state("R")
        poses = []
        for site_id in self._eef_sites:
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[site_id])
            poses.extend(self.data.site_xpos[site_id])
            poses.extend(quaternion)
        force = np.r_[
            self._sensor("L_wrist_force"),
            self._sensor("L_wrist_torque"),
            self._sensor("R_wrist_force"),
            self._sensor("R_wrist_torque"),
            self._sensor("L_finger_inner_touch_sensor"),
            self._sensor("L_finger_outer_touch_sensor"),
            self._sensor("R_finger_inner_touch_sensor"),
            self._sensor("R_finger_outer_touch_sensor"),
            sum(self.data.actuator_force[list(self._finger_actuators["L"])]),
            sum(self.data.actuator_force[list(self._finger_actuators["R"])]),
        ]
        observation: dict[str, Any] = {
            FRONT_IMAGE: self._render_camera("front"),
            LEFT_WRIST_IMAGE: self._render_camera("left_wrist"),
            RIGHT_WRIST_IMAGE: self._render_camera("right_wrist"),
            STATE: np.asarray(
                [*left_state, left_grip[0], *right_state, right_grip[0]], dtype=np.float32
            ),
            VELOCITY: np.asarray(
                [*left_velocity, left_grip[1], *right_velocity, right_grip[1]], dtype=np.float32
            ),
            EEF_POSE: np.asarray(poses, dtype=np.float32),
            FORCE: np.asarray(force, dtype=np.float32),
            "time": float(self.data.time),
            "safety_stop": bool(self._safety_stop),
        }
        validate_observation(observation)
        return observation

    def _reference_orientation(self, image: np.ndarray) -> np.ndarray:
        """Mirror a camera image the way the reference stack does.

        LeRobot wraps LIBERO's ``OffScreenRenderEnv`` and rotates every camera
        image 180 degrees before the policy sees it -- ``lerobot/envs/libero.py``
        line 227: ``image = image[::-1, ::-1]  # flip both H and W``.  LIBERO's own
        buffer is bottom-up (``obs["agentview_image"]`` is its ``sim.render`` byte
        for byte -- measured MSE 0.0 against all four orientations), so that
        ``[::-1]`` restores the upright orientation and the ``[:, ::-1]`` is a
        horizontal mirror *on top of* it.  The checkpoint was both trained and
        evaluated through that wrapper, so the image it expects is the horizontal
        mirror of the true camera view: measured against a frame of the official
        eval video, ``raw[::-1, ::-1]`` scores MSE 505 where ``raw`` scores 6190.

        ``mujoco.Renderer`` already hands back the upright image, so only the
        second of those two flips has to happen here.  Feeding the unmirrored view
        shows the policy a scene with left and right swapped, which is worth about
        twice the lateral offset of whatever it has been asked to reach.
        """
        # **Opt-in per scene, because the two halves of this repository live on opposite
        # conventions.**  Every A3 dataset and every checkpoint fine-tuned on one was recorded
        # through this environment with the images unmirrored; every LIBERO checkpoint was
        # trained on a mirrored view.  Applying the mirror globally broke the first group to
        # fix the second -- measured against `datasets/a3_cookie_adv_1000_merged`, whose front
        # camera puts the source bin on the right, and a mirrored render puts it on the left.
        mirror = self.config.cameras.reference_mirror
        if os.environ.get("A3_IMAGE_MIRROR", "").strip() == "0":
            mirror = False
        elif os.environ.get("A3_IMAGE_MIRROR", "").strip() == "1":
            mirror = True
        if not mirror:
            return np.ascontiguousarray(image)
        return np.ascontiguousarray(image[:, ::-1])

    def _render_camera(self, name: str) -> np.ndarray:
        if not self.render_cameras:
            return np.zeros((self.config.image_height, self.config.image_width, 3), dtype=np.uint8)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.image_height,
                width=self.config.image_width,
            )
        self._renderer.update_scene(self.data, camera=self._camera_ids[name])
        self._apply_render_flags(self._renderer.scene)
        return self._reference_orientation(
            np.ascontiguousarray(self._renderer.render(), dtype=np.uint8)
        )

    def _apply_render_flags(self, scene: Any) -> None:
        """Force the configured light passes on every frame.

        The renderer reuses one persistent scene, so the flags are re-applied
        here rather than once at construction to stay correct regardless of how
        ``update_scene`` leaves them.
        """
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.config.render_shadows)
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(self.config.render_reflections)

    def use_fast_render(self) -> None:
        """Drop the per-light shadow and reflection passes for speed.

        Software rasterizers spend most of their frame time generating shadow
        and reflection maps, whose cost does not depend on camera resolution.
        Disabling both keeps geometry, materials, and colours intact at roughly
        four times the frame rate.
        """
        self.config = replace(self.config, render_shadows=False, render_reflections=False)

    def render_view(self, camera: str, *, height: int, width: int) -> np.ndarray:
        """Render one scene camera at an arbitrary resolution.

        The policy cameras are pinned to the resolution the checkpoint was
        trained on, so a larger frame for a human viewer needs its own renderer.
        Renderers are cached per resolution and reused across steps, and the view
        inherits the same light passes as the policy cameras so a recording shows
        exactly the scene the dataset recorded.

        Unlike :meth:`_render_camera` this ignores ``render_cameras``: the caller
        is explicitly asking for pixels.
        """
        if camera not in self._camera_ids:
            raise ValueError(f"unknown camera {camera!r}; available: {sorted(self._camera_ids)}")
        if height < 1 or width < 1:
            raise ValueError("view dimensions must be positive")
        # MuJoCo refuses to allocate an offscreen render larger than the model's
        # framebuffer, which the scene XML sizes to the training resolution.
        # Growing it is harmless for the policy cameras: each renderer keeps its
        # own height and width, so they still produce training-sized images.
        global_settings = self.model.vis.global_
        global_settings.offwidth = max(global_settings.offwidth, width)
        global_settings.offheight = max(global_settings.offheight, height)
        renderer = self._view_renderers.get((height, width))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._view_renderers[(height, width)] = renderer
        renderer.update_scene(self.data, camera=self._camera_ids[camera])
        self._apply_render_flags(renderer.scene)
        # Same mirror as the policy cameras: a human comparing a recording against
        # LIBERO's own eval videos is looking at the pixels the policy was served.
        return self._reference_orientation(
            np.ascontiguousarray(renderer.render(), dtype=np.uint8)
        )

    def _ensure_viewer(self) -> None:
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._key_callback
            )
            # Keep the useful front-camera viewpoint, but use a free camera so
            # mouse rotation/pan/zoom work immediately. Policy cameras stay fixed.
            self._configure_viewer_camera(self._viewer.cam)

    def _configure_viewer_camera(self, camera: mujoco.MjvCamera) -> None:
        target = np.asarray(self.config.cameras.workspace_target_m, dtype=np.float64)
        offset = np.asarray(self.config.cameras.front_position_m) - target
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
        camera.lookat[:] = target
        camera.distance = float(np.linalg.norm(offset))
        camera.azimuth = float(np.rad2deg(np.arctan2(-offset[1], -offset[0])))
        camera.elevation = -float(np.rad2deg(np.arctan2(offset[2], np.linalg.norm(offset[:2]))))

    def _update_viewer_force_hud(self) -> None:
        if self._viewer is None:
            return
        try:
            inner_f = float(self._sensor("L_finger_inner_touch_sensor")[0])
            outer_f = float(self._sensor("L_finger_outer_touch_sensor")[0])
            max_f = max(inner_f, outer_f)
            self._max_recorded_gripper_force = max(self._max_recorded_gripper_force, max_f)

            # 在屏幕内部以紧凑小字 HUD 叠加显示最大值与历史峰值
            if hasattr(self._viewer, "set_texts"):
                text1 = "MAX GRIPPER FORCE:\nPEAK RECORDED:"
                text2 = f"{max_f:.2f} N\n{self._max_recorded_gripper_force:.2f} N"
                try:
                    self._viewer.set_texts([
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            text1,
                            text2,
                        )
                    ])
                except Exception:
                    pass

            # 窗口标题栏同步显示当前最大值与峰值
            title_text = (
                f"MuJoCo : a3_cookie_transfer | "
                f"Max Force: {max_f:.2f} N (Peak: {self._max_recorded_gripper_force:.2f} N)"
            )
            try:
                self._viewer.title = title_text
            except Exception:
                pass

            window = getattr(self._viewer, "_window", None)
            if window is not None:
                try:
                    import glfw
                    glfw.set_window_title(window, title_text)
                except Exception:
                    pass
        except Exception:
            pass

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._ensure_viewer()
            self._update_viewer_force_hud()
            self._viewer.sync()
            return None
        return self._render_camera("front")

    def close(self) -> None:
        # A human viewer and the offscreen camera renderer own separate OpenGL
        # contexts.  Destroy the viewer first; GLFW may otherwise tear down shared
        # process state while the visible context is still live (exit-time SIGSEGV).
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        for renderer in self._view_renderers.values():
            renderer.close()
        self._view_renderers.clear()
