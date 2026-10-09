from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .box_support import BoxSupportController
from .contracts import (
    CARTESIAN_ACTION_DIM,
    LEFT_JOINTS,
    RIGHT_JOINTS,
    STATE,
    ActionMode,
    EpisodeContext,
)
from .cookie_transfer import A3CookieTransferEnv
from .grasp import A3GraspEnv


def _rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """A unit quaternion as a rotation vector, in radians, along the shorter arc.

    ``q`` and ``-q`` are the same rotation, and taking the shorter arc is not an optimisation:
    a quaternion with a negative real part turns a 0.2 rad correction into a 6.1 rad one, in the
    other direction.  Four vectors of a pose error are a place where that is invisible until the
    arm spins.
    """

    w = float(quaternion[0])
    vector = np.asarray(quaternion[1:4], dtype=np.float64)
    if w < 0.0:
        w, vector = -w, -vector
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.ascontiguousarray(vector * (2.0 * np.arctan2(norm, w) / norm))


class GraspPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    HOLD = auto()
    DONE = auto()


class CookiePhase(Enum):
    SUPPORT_BOX = auto()
    SELECT_COOKIE = auto()
    APPROACH = auto()
    PRE_CLOSE = auto()
    ALIGN = auto()
    DESCEND = auto()
    CLOSE = auto()
    VERIFY_GRASP = auto()
    LIFT = auto()
    VERIFY_LIFT = auto()
    MOVE_TO_SLOT = auto()
    DESCEND_TO_PLACE = auto()
    OPEN = auto()
    VERIFY_RELEASE = auto()
    RETRACT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class A3GraspExpert:
    """Privileged joint-space expert with per-episode absolute IK waypoints."""

    env: A3GraspEnv
    action_mode: ActionMode = "joint_position"
    approach_height_m: float = 0.17
    grasp_eef_offset_m: float = -0.012
    lift_height_m: float = 0.18
    joint_tolerance_rad: float = 0.055
    transit_joint_step_rad: float = 0.025
    manipulation_joint_step_rad: float = 0.02
    closed_gripper_opening: float = 0.18
    close_steps: int = 60
    grasp_stable_steps: int = 8
    hold_steps: int = 12
    phase_timeout_steps: int = 180

    APPROACH_GUESS: ClassVar[np.ndarray] = np.asarray(
        [1.16, 1.06, -2.93, 0.056, 1.57, -0.55, -1.56], dtype=np.float64
    )
    GRASP_GUESS: ClassVar[np.ndarray] = np.asarray(
        [1.56, 1.02, -2.85, 0.0, 1.38, -0.545, -1.40], dtype=np.float64
    )

    def __post_init__(self) -> None:
        self.phase = GraspPhase.APPROACH
        self.phase_steps = 0
        self._grasp_stable_count = 0
        self._waypoints: dict[GraspPhase, np.ndarray] = {}
        self._qpos_ids = np.asarray(
            [self.env._qpos_ids[name] for name in LEFT_JOINTS], dtype=np.int32
        )
        self._joint_ranges = np.asarray(
            [self.env.model.jnt_range[self.env._joint_ids[name]] for name in LEFT_JOINTS]
        )
        self._site_id = self.env._eef_sites[0]
        target_rotation = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        self._target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(self._target_quaternion, target_rotation.reshape(-1))

    def reset(self, context: EpisodeContext) -> None:
        self.phase = GraspPhase.APPROACH
        self.phase_steps = 0
        self._grasp_stable_count = 0
        object_position = self.env.target_position
        approach_target = object_position + np.asarray([0.0, 0.0, self.approach_height_m])
        grasp_target = object_position + np.asarray([0.0, 0.0, self.grasp_eef_offset_m])
        lift_target = grasp_target.copy()
        lift_target[2] = self.env.initial_object_z + self.lift_height_m + self.grasp_eef_offset_m
        approach = self._solve(approach_target, self.APPROACH_GUESS)
        grasp = self._solve(grasp_target, self.GRASP_GUESS)
        lift = self._solve(lift_target, approach)
        self._waypoints = {
            GraspPhase.APPROACH: approach,
            GraspPhase.DESCEND: grasp,
            GraspPhase.CLOSE: grasp,
            GraspPhase.LIFT: lift,
            GraspPhase.HOLD: lift,
            GraspPhase.DONE: lift,
        }

    def _solve(self, target_position: np.ndarray, initial: np.ndarray) -> np.ndarray:
        work = mujoco.MjData(self.env.model)
        work.qpos[:] = self.env.data.qpos
        work.qvel[:] = 0.0

        def residual(qpos: np.ndarray) -> np.ndarray:
            work.qpos[self._qpos_ids] = qpos
            mujoco.mj_forward(self.env.model, work)
            current_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(current_quaternion, work.site_xmat[self._site_id])
            rotation_error = np.empty(3, dtype=np.float64)
            mujoco.mju_subQuat(rotation_error, self._target_quaternion, current_quaternion)
            return np.r_[
                work.site_xpos[self._site_id] - target_position,
                0.03 * rotation_error,
            ]

        result = least_squares(
            residual,
            np.clip(initial, self._joint_ranges[:, 0], self._joint_ranges[:, 1]),
            bounds=(self._joint_ranges[:, 0] + 1e-4, self._joint_ranges[:, 1] - 1e-4),
            max_nfev=600,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        position_error = np.linalg.norm(residual(result.x)[:3])
        if not result.success or position_error > 0.012:
            # `result.success` is the *optimizer's* convergence flag, not the
            # outcome: a converged solve can still land short of the target, and a
            # message reading "IK failed: success=True" sent readers looking for a
            # contradiction instead of at the residual.
            reasons = []
            if not result.success:
                reasons.append("the solver did not converge")
            if position_error > 0.012:
                reasons.append(
                    f"the closest pose is {position_error * 1000:.2f} mm from the "
                    f"target (tolerance 12.00 mm)"
                )
            raise RuntimeError("grasp expert IK failed: " + " and ".join(reasons))
        return np.asarray(result.x, dtype=np.float64)

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        self.phase_steps += 1
        state = np.asarray(observation[STATE], dtype=np.float64)
        action = state.copy()
        desired = self._waypoints[self.phase]
        command_step = (
            self.transit_joint_step_rad
            if self.phase is GraspPhase.APPROACH
            else self.manipulation_joint_step_rad
        )
        previous_command = self.env.last_applied_action[:7]
        action[:7] = previous_command + np.clip(
            desired - previous_command, -command_step, command_step
        )
        action[15] = 1.0
        action[7] = (
            1.0
            if self.phase in (GraspPhase.APPROACH, GraspPhase.DESCEND)
            else self.closed_gripper_opening
        )

        if self.phase is GraspPhase.APPROACH and self._reached(state[:7]):
            self._advance(GraspPhase.DESCEND)
        elif self.phase is GraspPhase.DESCEND and self._reached(state[:7]):
            self._advance(GraspPhase.CLOSE)
        elif self.phase is GraspPhase.CLOSE:
            stable = self.env.is_grasped()
            self._grasp_stable_count = self._grasp_stable_count + 1 if stable else 0
            if (
                self._grasp_stable_count >= self.grasp_stable_steps
                or self.phase_steps >= self.close_steps
            ):
                self._advance(GraspPhase.LIFT)
        elif self.phase is GraspPhase.LIFT:
            if self.env.success_hold_count > 0:
                self._advance(GraspPhase.HOLD)
            elif self.phase_steps >= self.phase_timeout_steps:
                self._advance(GraspPhase.DONE)
        elif self.phase is GraspPhase.HOLD and (
            self.env.success_hold_count >= self.env.task_config.success_hold_steps
            or self.phase_steps >= self.hold_steps
        ):
            self._advance(GraspPhase.DONE)

        if self.phase_steps >= self.phase_timeout_steps and self.phase in (
            GraspPhase.APPROACH,
            GraspPhase.DESCEND,
        ):
            self._advance(GraspPhase.DONE)
        return action

    def _reached(self, joints: np.ndarray) -> bool:
        return bool(
            np.max(np.abs(joints - self._waypoints[self.phase])) <= self.joint_tolerance_rad
        )

    def _advance(self, phase: GraspPhase) -> None:
        self.phase = phase
        self.phase_steps = 0

    def close(self) -> None:
        return None


# How far below the Cookie's top face the jaws are meant to pinch.  Only this
# depth is a real design choice; the rest of the offset follows
# ``scene.cookie_half_size_m[2]`` (see ``_pinch_offset_from_cookie_centre``).
_PINCH_DEPTH_BELOW_TOP_M = 0.002


def _pinch_offset_from_cookie_centre(half_size_z: float) -> float:
    """Offset from a Cookie's centre up to the height where the jaws pinch.

    The jaws grip just below the Cookie's top face, so the offset has to be
    derived from ``scene.cookie_half_size_m[2]`` -- it is *not* a free tuning
    knob.  Whenever that half-size changes, this offset changes with it; keeping
    them in sync is what makes the tool reach the Cookie instead of closing on
    empty space above it.

    This used to be the bare literal ``0.023``.  That was only correct while the
    Cookie was 50 mm tall (half-size 25 mm, so the jaws sat 2 mm below the top
    face).  When the Cookie was shortened to 25 mm tall the same literal aimed
    the jaws ~10.5 mm *above* the top face, i.e. at empty space rather than at
    the Cookie.
    """
    return float(half_size_z) - _PINCH_DEPTH_BELOW_TOP_M


@dataclass
class A3CookieTransferExpert:
    """Feedback-driven privileged expert for the ten-Cookie transfer task.

    This is intentionally not a policy: it reads simulator-only truth and follows
    explicit state-machine rules.  Those privileged values are never exposed in
    the learning observation.
    """

    env: A3CookieTransferEnv
    action_mode: ActionMode = "joint_position"
    max_joint_step_rad: float = 0.020
    max_gripper_step: float = 0.060    #: Largest joint step the placement descent may take.  A safety limit rather than a
    #: motion plan: the descent lowers a held Cookie into a 6.7 mm cell with neighbours on
    #: both sides, and the failure mode of overshooting is knocking the row over.
    #:
    #: It is also, measured, most of why the demonstrations dwell.  `0.008` is a **sixth** of
    #: the environment's own `max_joint_step_rad` (0.05), `DESCEND_TO_PLACE` is **27%** of
    #: every episode, and 100% of its steps come in under a fifth of the cap -- so this one
    #: clamp is 27 points of the 57% that makes the action distribution look stationary
    #: (`scripts/probe_expert_dwell.py`).  Raising it trades placement safety for a training
    #: signal, which is the experiment; the default is the verified value.
    place_step_limit_rad: float = 0.008
    #: Floor on how many steps `_trajectory_command` spends covering a joint delta.  The
    #: default 12 means a small delta -- anything under 0.24 rad, which every grasp descent
    #: is -- is interpolated into twelfth-sized steps, i.e. ~7.5% of the cap, and DESCEND is
    #: 20% of every episode.  Unlike `max_joint_step_rad`, which caps a step, this one sets
    #: the step, and only for the phases whose delta is small.
    descend_min_steps: int = 12
    joint_tolerance_rad: float = 0.055
    # IK acceptance.  The old thresholds (retry above 12 mm, accept up to 16 mm)
    # were loose enough to silently accept a solution that missed the commanded
    # point by 9 mm -- which put the jaw pads 5 mm above the Cookie with nothing
    # touching.  A pinch needs the pads within a millimetre of the planned height,
    # and a good solution exists (the same waypoint solves to 0.001 mm from a
    # different seed), so retry well before that instead of accepting the miss.
    ik_retry_tolerance_m: float = 0.002
    ik_max_tolerance_m: float = 0.004
    # How close the arm has to get before a descend is considered finished.
    #
    # This only has to catch the *transient*: ``_motion_done`` flips as soon as the
    # interpolated command queue empties, with the joints still chasing the last
    # command, and the expert used to leave DESCEND right then.  It must be LOOSER
    # than the servos' steady-state droop, measured at 0.0068 rad -- asking for
    # better than the servos can hold means the check never passes at all and the
    # refinement below never runs.  Accuracy comes from the tool-space correction
    # instead, which measures and cancels that droop.
    arrival_tolerance_rad: float = 0.010
    # Tool-space tolerance for the descent, and how many times to re-aim for it.
    # The position servos settle ~4 mm below the commanded height at this reach
    # even though the joints sit within 0.007 rad of the command.  That is enough
    # to hold the pads above the planned pinch height, where they close on air, and
    # joint-space arrival cannot see it -- so DESCEND closes the loop on the
    # measured tool position and re-aims, overshooting by exactly the measured
    # error.
    tool_tolerance_m: float = 0.001
    max_descent_aims: int = 4
    # Transit / align / retract: jaws well clear of the row so they cannot catch.
    open_gripper: float = 0.28
    # Descending: the pads slide down the flare between the target Cookie and its
    # two neighbours, so the opening is set from the chamfered top face rather
    # than left wide.  At 0.28 the jaws were 23.8 mm apart -- three and a half row
    # pitches -- which put the pads over Cookies two places away instead of in the
    # two gaps flanking the target.
    grasp_clearance_m: float = 0.0004
    closed_gripper: float = 0.0
    release_gripper: float = 0.25
    close_steps: int = 18
    grasp_stable_steps: int = 8
    release_stable_steps: int = 6
    settle_steps: int = 3
    phase_timeout_steps: int = 180
    max_retries: int = 2
    #: Height of the transit/approach waypoint above the table, in metres.
    #:
    #: It was a literal `0.86` in `_plan_current_cookie`, which is a number about **A3's arm**:
    #: its shoulder geometry puts a comfortable overhead pose there.  LIBERO's Panda reaches the
    #: same table from a different base with a different workspace, and the two probes that put
    #: its fingertip within 10.9-28 mm of the Cookie did so at the Cookie's own height (0.768 m),
    #: not 92 mm above it.  Measured on the Panda in the collection layout, the approach command
    #: stalls 38 mm short in y and 0.225 rad off in orientation with the command at 78% of full
    #: scale -- not saturated, which is what reaching a limit looks like -- so the height is the
    #: thing to vary, and it is a field so that varying it is a measurement rather than an edit.
    approach_height_m: float = 0.86
    #: How the arm is commanded, and therefore which action space this expert produces.
    #:
    #: ``"joint"`` is the original: the planner's joint targets are emitted directly, which is the
    #: arm's own space for A3's position-servo arms and is what every scene used until now.
    #:
    #: ``"cartesian"`` converts each joint target into the **pose it implies** and emits a
    #: `cartesian_delta` instead.  That is not a preference.  LIBERO's Panda is driven by
    #: `OperationalSpaceAdapter`, which takes a Cartesian goal and solves for torque, so on that
    #: arm the joint targets are written to motor actuators as Newton-metres.  Measured, same
    #: scene and same layout and only the arm swapped: A3's arm solves it **3 of 3 in 143 steps**,
    #: the Panda **0 of 3 in 640**.
    #:
    #: **It has to live here rather than in a wrapper around the expert, and that is a
    #: measurement rather than a taste.**  `_trajectory_command` both interpolates the motion and
    #: decides how many steps it should take, from `env.last_applied_action` -- which under
    #: operational-space control is a torque vector.  A wrapper that converted the finished joint
    #: command would still have had the expert choose its step count from torques and interpolate
    #: through poses that no arm could ever be at.
    command_space: str = "joint"
    #: How close the controlled site has to be to its goal before a cartesian move counts as
    #: finished.  `_motion_done` is what ends APPROACH, LIFT, MOVE_TO_SLOT and RETRACT, so a
    #: tolerance the controller cannot hold leaves those phases to their timeouts, while too
    #: loose a one ends them before the arm has gone anywhere.
    #:
    #: **Set from what the controllers were measured to hold, not from what a grasp would like.**
    #: Driving towards a waypoint and watching the residual settle (`scripts/probe_cartesian_convergence.py`):
    #:
    #:     A3's arm, closing on a grasped Cookie     floor about 2.0-2.3 mm, creeping
    #:     the Panda, descending towards the pinch     floor about 7.0 mm, creeping
    #:
    #: Both creep *towards* the goal rather than stopping, at roughly 0.1 mm per twenty steps, so
    #: this is a slow approach to a limit rather than a ballistics error -- and both are above the
    #: 2 mm this used to be, which is why nothing ever arrived.  It matters which side of the
    #: floor this sits on, because a phase whose arrival never fires has no escape at all (see
    #: the timeouts in `act`), so too tight a value is not a small error, it is a hang.
    #:
    #: This is a *coarse* arrival on purpose.  Precision is done by the loops that were built for
    #: it: `_aim_descent` re-aims until the tool's **measured** position is within `tool_tolerance_m`
    #: (1 mm), the placement descent settles on the **Cookie's** measured pose against the slot
    #: (4 mm), and the grasp is confirmed by contact.  None of those read this.
    cartesian_arrival_tolerance_m: float = 0.008
    cartesian_arrival_tolerance_rad: float = 0.02

    TOOL_SLOT_TARGETS_LOCAL: ClassVar[np.ndarray] = np.asarray(
        [
            (-0.028, -0.048),
            (+0.028, -0.048),
            (-0.028, -0.024),
            (+0.028, -0.024),
            (-0.028, 0.000),
            (+0.028, 0.000),
            (-0.028, +0.024),
            (+0.028, +0.024),
            (-0.028, +0.048),
            (+0.028, +0.048),
        ],
        dtype=np.float64,
    )

    def __post_init__(self) -> None:
        if self.command_space not in ("joint", "cartesian"):
            raise ValueError(
                f"command_space must be 'joint' or 'cartesian', got {self.command_space!r}"
            )
        if self.command_space == "cartesian" and self.action_mode != "cartesian_delta":
            # The two are one decision.  A scene builds this expert and also decides the action
            # space its environment accepts, and a mismatch between them fails on the first step
            # with a shape error that names neither.
            raise ValueError(
                f"command_space='cartesian' needs action_mode='cartesian_delta', "
                f"not {self.action_mode!r}"
            )
        self.model = self.env.model
        self.data = self.env.data
        self._l_qpos = np.asarray(
            [self.env._qpos_ids[name] for name in LEFT_JOINTS], dtype=np.int32
        )
        # The site the **controller in use** drives, which is not `L_eef`: the operational-space
        # adapter tracks robosuite's own grasp site.  A delta computed at the wrong one of the two
        # moves the other by that delta and leaves its own goal short by the offset between them,
        # held as steady error rather than reported.
        self._command_site = self.env.ik.left_site
        # One scratch state for forward kinematics, beside `_solve_l`'s own.  Only the cartesian
        # path allocates it, because only it needs the pose a joint target implies.
        self._scratch = mujoco.MjData(self.model) if self.command_space == "cartesian" else None
        self._r_qpos = np.asarray(
            [self.env._qpos_ids[name] for name in RIGHT_JOINTS], dtype=np.int32
        )
        self._l_site = self.env._eef_sites[0]
        self._tb_id = self.env._target_bin_body
        self._l_ranges = np.asarray(
            [self.model.jnt_range[self.env._joint_ids[name]] for name in LEFT_JOINTS]
        )

        # **The grasp orientation, read off the arm rather than written as a constant.**  It used
        # to be the literal `[[0,0,1],[-1,0,0],[0,-1,0]]`, and that matrix is *exactly* A3's
        # `L_eef` at its own start pose -- measured, 0.0 degrees from it.  So the constant was
        # never a statement about the task ("grasp from above"); it was a statement about one
        # robot's flange frame, and the same numbers mean a physically different tool pose on any
        # other arm: LIBERO's Panda at its start has its tool pointing **down** (z = -0.998) and
        # is **121.7 degrees** away from that matrix, so the expert asked it for a horizontal
        # wrist and then could not understand why it never arrived.
        #
        # Deriving it from the arm's own start pose is what makes the statement portable, and it
        # is also what makes it *verifiable*: on A3 it reproduces the old constant bit for bit,
        # so nothing about A3's behaviour can change, while the Panda gets the orientation it
        # actually starts in -- which is the top-down grasp both arms' homes are built for.
        #
        # The reachability oracle below (`_solve_l`) checks only the **position** residual, so a
        # waypoint whose orientation the arm cannot hold is accepted as reachable.  That hole is
        # why this was invisible: on A3 the constant is attainable by construction, and there is
        # no second arm to notice it on.
        home = np.asarray(self.env.DEPLOYMENT_HOME[:7], dtype=np.float64)
        reference = mujoco.MjData(self.model)
        reference.qpos[:] = self.data.qpos
        reference.qpos[self._l_qpos] = home
        mujoco.mj_kinematics(self.model, reference)
        rot_canonical = np.asarray(reference.site_xmat[self._l_site]).reshape(3, 3).copy()
        self._target_quat_canonical = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            self._target_quat_canonical, np.ascontiguousarray(rot_canonical).reshape(-1)
        )

        theta = np.deg2rad(10.0)
        R_world_target = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ]
        )
        rot_tilted = R_world_target @ rot_canonical
        self._target_quat_tilted = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(self._target_quat_tilted, rot_tilted.reshape(-1))

        self.q_branch_seed = np.array([1.28, 1.119, -1.127, -0.999, -0.252, 0.478, -1.535])
        self.q_r_hold = np.array([-1.805, -1.499, 1.543, 0.54, -2.247, -0.058, 0.041])
        self.q_transit = self.q_branch_seed.copy()
        self.phase = CookiePhase.SELECT_COOKIE
        self.phase_steps = 0
        self.current_cookie_index: int | None = None
        self.target_slot_index: int | None = None
        # Sized by the **cells** it is indexed by, not by `required_cookies`.  `_retry` looks
        # up `retry_counts[self._current_slot()]`, and `_current_slot()` is a
        # `target_slot_index` into `TARGET_SLOTS_LOCAL`, so the two dimensions only agree by
        # coincidence -- the ten-Cookie scene wants ten Cookies and the target box has ten
        # cells, which is why this went unnoticed.  The one-Cookie scene wants one, so the
        # list had length 1 while a slot could be anything in 0..9.  The `IndexError`
        # therefore fires only on a *retry*, i.e. only after the expert has already failed a
        # grasp -- and in 400 collected episodes on this scene it never had.  A faster
        # descent does fail occasionally, and that is how it surfaced.
        self.retry_counts = [0] * len(self.env.TARGET_SLOTS_LOCAL)
        self.completed_cookie_indices: list[int] = []
        self.skipped_cookie_indices: list[int] = []
        self.transition_history: list[CookiePhase] = []
        self.failure_reason: str | None = None
        self.failure_phase: str | None = None
        self.retry_reasons: list[str] = []
        self._waypoints: dict[str, np.ndarray] = {}
        self._grasp_target_world: np.ndarray | None = None
        self._descent_aims = 0
        self._grasp_stable_count = 0
        self._release_stable_count = 0
        self._settle_count = 0
        self._move_stage = "transit"
        self._retract_stage = "preplace"
        self._motion_actions: list[np.ndarray] = []
        self._motion_key: tuple[bytes, float, int] | None = None
        self._motion_done = False
        self._recovering = False
        self._initial_cookie_positions = np.asarray(
            [
                self.env.privileged_cookie_position(index)
                for index in range(self.env.task_config.cookie_count)
            ],
            dtype=np.float64,
        )
        self._initial_target_position = self.data.xpos[self._tb_id].copy()
        self._initial_target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
        self._cookie_preference = tuple(
            sorted(
                range(self.env.task_config.cookie_count),
                key=lambda i: (self.env.SOURCE_POSITIONS[i][1], self.env.SOURCE_POSITIONS[i][0]),
            )
        )
        self._slot_order = (8, 9, 6, 7, 4, 5, 2, 3, 0, 1)

    def _solve_l(self, pos: np.ndarray, quat: np.ndarray, q0: np.ndarray) -> np.ndarray:
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos[:]

        def residual(q: np.ndarray) -> np.ndarray:
            work.qpos[self._l_qpos] = q
            mujoco.mj_kinematics(self.model, work)
            cur_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(cur_quat, work.site_xmat[self._l_site])
            err = np.empty(3, dtype=np.float64)
            mujoco.mju_subQuat(err, quat, cur_quat)
            return np.r_[work.site_xpos[self._l_site] - pos, 0.12 * err, 0.001 * (q - q0)]

        res = least_squares(
            residual,
            np.clip(q0, self._l_ranges[:, 0] + 1e-3, self._l_ranges[:, 1] - 1e-3),
            bounds=(self._l_ranges[:, 0] + 1e-4, self._l_ranges[:, 1] - 1e-4),
            max_nfev=500,
        )
        position_error = float(np.linalg.norm(residual(res.x)[:3]))
        if position_error > self.ik_retry_tolerance_m:
            rng = np.random.default_rng(0)
            for seed in [
                self.data.qpos[self._l_qpos],
                self.env.DEPLOYMENT_HOME[:7],
                *rng.uniform(self._l_ranges[:, 0], self._l_ranges[:, 1], (4, 7)),
            ]:
                candidate = least_squares(
                    residual,
                    np.clip(seed, self._l_ranges[:, 0] + 1e-3, self._l_ranges[:, 1] - 1e-3),
                    bounds=(self._l_ranges[:, 0] + 1e-4, self._l_ranges[:, 1] - 1e-4),
                    max_nfev=300,
                )
                if np.linalg.norm(candidate.fun[:6]) < np.linalg.norm(res.fun[:6]):
                    res = candidate
                position_error = float(np.linalg.norm(residual(res.x)[:3]))
                if position_error < self.ik_retry_tolerance_m:
                    break
        if not res.success or position_error > self.ik_max_tolerance_m:
            # As above: `res.success` says the optimizer converged, not that the
            # pose is close enough.  Name the condition that actually failed, so a
            # rejected episode reads as "1.27 mm short" rather than as a paradox.
            reasons = []
            if not res.success:
                reasons.append("the solver did not converge")
            if position_error > self.ik_max_tolerance_m:
                reasons.append(
                    f"the closest pose is {position_error * 1000:.2f} mm from the "
                    f"target (tolerance {self.ik_max_tolerance_m * 1000:.2f} mm)"
                )
            raise RuntimeError("cookie expert IK failed: " + " and ".join(reasons))
        return np.asarray(res.x, dtype=np.float64)

    def reset(self, context: EpisodeContext | None = None) -> None:
        del context
        self.q_transit = self._solve_l(
            np.array([0.10, 0.30, 0.89]), self._target_quat_canonical, self.q_branch_seed
        )
        self.phase = CookiePhase.SELECT_COOKIE
        self.phase_steps = 0
        self.current_cookie_index = None
        self.target_slot_index = None
        # See the matching line in `__post_init__`: sized by cells, not by cookies.
        self.retry_counts = [0] * len(self.env.TARGET_SLOTS_LOCAL)
        self.completed_cookie_indices = []
        self._packed_positions: dict[int, np.ndarray] = {}
        self.skipped_cookie_indices = []
        self.transition_history = [self.phase]
        self.failure_reason = None
        self.failure_phase = None
        self.retry_reasons = []
        self._waypoints = {}
        self._grasp_target_world = None
        self._descent_aims = 0
        self._grasp_stable_count = 0
        self._release_stable_count = 0
        self._settle_count = 0
        self._move_stage = "transit"
        self._retract_stage = "preplace"
        self._reset_motion()
        self._recovering = False
        self._initial_cookie_positions = np.asarray(
            [
                self.env.privileged_cookie_position(index)
                for index in range(self.env.task_config.cookie_count)
            ],
            dtype=np.float64,
        )
        self._initial_target_position = self.data.xpos[self._tb_id].copy()
        self._initial_target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
        self.box_support = BoxSupportController(self.env)
        self.phase = CookiePhase.SUPPORT_BOX
        self.transition_history = [self.phase]
        self.right_gripper = 1.0

    def act(self, observation: dict[str, Any] | None = None, task: str = "") -> np.ndarray:
        del observation, task
        if self.phase is CookiePhase.SUPPORT_BOX:
            action = self.box_support.act()
            if self.box_support.failed:
                self._fail(self.box_support.failed)
            elif self.box_support.ready:
                self.q_r_hold = action[8:15].copy()
                self.right_gripper = action[15]
                self._initial_target_position = self.data.xpos[self._tb_id].copy()
                self._initial_target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
                rot_canonical = np.empty(9)
                mujoco.mju_quat2Mat(rot_canonical, self._target_quat_canonical)
                mujoco.mju_mat2Quat(
                    self._target_quat_tilted,
                    (self._initial_target_rotation @ rot_canonical.reshape(3, 3)).ravel(),
                )
                self._advance(CookiePhase.SELECT_COOKIE)
            return action
        if self.phase not in (CookiePhase.DONE, CookiePhase.FAILED) and not np.all(
            self.box_support.contact_forces() > 0.1
        ):
            self._fail("right gripper lost box support")
        if self.phase not in (CookiePhase.DONE, CookiePhase.FAILED):
            for index, placed_position in self._packed_positions.items():
                displacement = np.linalg.norm(
                    self.env.privileged_cookie_target_position(index) - placed_position
                )
                rotation = (
                    self.data.xmat[self._tb_id].reshape(3, 3).T
                    @ self.data.xmat[self.env._cookie_bodies[index]].reshape(3, 3)
                )
                if displacement > 0.006 or abs(rotation[2, 2]) < np.cos(np.deg2rad(12)):
                    self._fail(f"packed Cookie {index} disturbed; holding position")
                    break
        # SELECT_COOKIE has no physical action. Resolve it immediately so every
        # returned command belongs to a meaningful motion/verification phase.
        if self.phase is CookiePhase.SELECT_COOKIE:
            self._select_cookie()

        self.phase_steps += 1
        if self.phase is CookiePhase.FAILED:
            return self._stay_command()
        if self.phase is CookiePhase.DONE:
            return self._hold_command(self.q_transit, self.open_gripper)

        if self.phase is CookiePhase.APPROACH:
            target = self.q_transit if self._recovering else self._waypoints["approach"]
            action = self._trajectory_command(target, self.open_gripper)
            if self._motion_done:
                if self._recovering:
                    self._recovering = False
                    self._reset_motion()
                else:
                    self._advance(CookiePhase.ALIGN)
            elif self._timed_out():
                # **A phase whose arrival never fires has no escape unless the timeout stands on
                # its own.**  In the joint path that could not happen: `_motion_done` means "the
                # interpolated queue is empty", which is true after a fixed number of steps
                # whatever the arm does.  In Cartesian space it means "the arm is at the goal",
                # which a real controller can fail to reach -- and then this phase, which had no
                # timeout at all, waits forever: measured, a Panda sitting in APPROACH for its
                # whole 900-step budget with `reason=None` and no retries.
                self._retry("approach timed out without arriving")
            return action

        if self.phase is CookiePhase.ALIGN:
            self._advance(CookiePhase.DESCEND)

        if self.phase is CookiePhase.DESCEND:
            action = self._trajectory_command(
                self._waypoints["grasp"],
                self._descent_gripper(),
                min_steps=self.descend_min_steps,
            )
            if self._motion_done and self._reached(
                self._waypoints["grasp"], tolerance=self.arrival_tolerance_rad
            ):
                if self._aim_descent():
                    self._advance(CookiePhase.CLOSE)
            elif self._timed_out(120):
                # Something below is holding the pads up (a neighbour's corner, a
                # wall, or the servos bottomed out).  Close anyway rather than spin
                # here, and let VERIFY_GRASP judge whether a Cookie was caught.
                #
                # The `_motion_done` that used to gate this was a second reason the Cartesian
                # path could not escape: it is exactly the condition that fails when a controller
                # cannot reach the goal, so requiring it here made the timeout unreachable.
                self._advance(CookiePhase.CLOSE)
            return action

        if self.phase is CookiePhase.CLOSE:
            action = self._trajectory_command(self._waypoints["grasp"], self.closed_gripper)
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_GRASP)
            elif self._timed_out():
                # As in DESCEND: let VERIFY_GRASP judge whether a Cookie was caught, rather than
                # spinning in a phase whose only exit is "arrived".
                self._advance(CookiePhase.VERIFY_GRASP)
            return action

        if self.phase is CookiePhase.VERIFY_GRASP:
            action = self._hold_command(self._waypoints["grasp"], self.closed_gripper)
            grasped = self._has_verified_grasp()
            self._grasp_stable_count = self._grasp_stable_count + 1 if grasped else 0
            if self._grasp_stable_count >= self.grasp_stable_steps:
                self._advance(CookiePhase.LIFT)
            elif self._timed_out(30):
                self._retry("bilateral cookie grasp was not verified")
            return action

        if self.phase is CookiePhase.LIFT:
            action = self._trajectory_command(self._waypoints["lift"], self.closed_gripper)
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_LIFT)
            elif self._cookie_dropped():
                self._retry("Cookie dropped during lift")
            elif self._timed_out():
                self._retry("lift timed out")
            return action

        if self.phase is CookiePhase.VERIFY_LIFT:
            if self._has_verified_lift():
                try:
                    self._plan_place_from_grasp()
                except RuntimeError as exc:
                    self._fail(f"held Cookie placement IK: {exc}")
                    return self._hold_command(self._waypoints["lift"], self.closed_gripper)
                self._advance(CookiePhase.MOVE_TO_SLOT)
            else:
                action = self._hold_command(self._waypoints["lift"], self.closed_gripper)
                if self._timed_out(30):
                    self._retry("Cookie did not follow the gripper")
                return action

        if self.phase is CookiePhase.MOVE_TO_SLOT:
            target = (
                self.q_transit if self._move_stage == "transit" else self._waypoints["preplace"]
            )
            action = self._trajectory_command(target, self.closed_gripper)
            if self._cookie_dropped():
                self._retry("Cookie dropped during transfer")
            elif self._motion_done:
                if self._move_stage == "transit":
                    self._move_stage = "preplace"
                    self.phase_steps = 0
                    self._reset_motion()
                else:
                    self._advance(CookiePhase.DESCEND_TO_PLACE)
            elif self._timed_out():
                self._retry("transfer timed out without arriving")
            return action

        if self.phase is CookiePhase.DESCEND_TO_PLACE:
            local_position = self.env.privileged_cookie_target_position(self._current_cookie())
            target_xy = np.asarray(self.env.TARGET_SLOTS_LOCAL[self._current_slot()])
            body = self.env._cookie_bodies[self._current_cookie()]
            box_rotation = self.data.xmat[self._tb_id].reshape(3, 3)
            cookie_rotation = self.data.xmat[body].reshape(3, 3)
            relative_rotation = box_rotation.T @ cookie_rotation
            if self.command_space == "cartesian":
                # `_plan_place_from_grasp` already solved the tool pose that puts the *held*
                # Cookie at the slot, from the measured Cookie-to-tool transform, and the arm's
                # own controller closes that loop -- so this phase commands the pose and judges
                # the result from the Cookie's measured pose, which is what the joint version's
                # Jacobian servo is doing the long way round.
                #
                # The judgement below is shared, and that is deliberate: it is the part that
                # knows what "placed" means.
                action = self._cartesian_command(self._waypoints["place"], self.closed_gripper)
            else:
                action = self._stay_command()
                action[7] = self.closed_gripper
                action[8:15] = self.q_r_hold
                action[15] = self.right_gripper
                position_error = box_rotation @ (
                    np.r_[target_xy, self._placement_height] - local_position
                )
                rotation_error = (
                    sum(np.cross(cookie_rotation[:, i], box_rotation[:, i]) for i in range(3))
                    * 0.5
                )
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._l_site)
                dofs = [self.env._dof_ids[name] for name in LEFT_JOINTS]
                jac = np.vstack((jacp[:, dofs], jacr[:, dofs]))
                dq = jac.T @ np.linalg.solve(
                    jac @ jac.T + 0.02**2 * np.eye(6),
                    np.r_[position_error, rotation_error] * 0.15,
                )
                limit = self.place_step_limit_rad
                action[:7] = self.data.qpos[self._l_qpos] + np.clip(dq, -limit, limit)
                self._waypoints["place"] = action[:7].copy()
            aligned = (
                np.linalg.norm(local_position[:2] - target_xy) < 0.004
                and abs(relative_rotation[2, 2]) > np.cos(np.deg2rad(8))
                and abs(local_position[2] - self._placement_height) < 0.004
            )
            self._settle_count = self._settle_count + 1 if aligned else 0
            if self._settle_count >= 8:
                if self.command_space == "cartesian":
                    # Hold where the descent actually finished.  The settle test judges the
                    # *Cookie's* pose, so the tool is where it needed to be and not necessarily
                    # where it was planned to be -- and the joint path makes the same substitution
                    # above, for the same reason.
                    self._waypoints["place"] = self.data.qpos[self._l_qpos].copy()
                self._advance(CookiePhase.OPEN)
            elif self._cookie_dropped():
                self._fail("Cookie released before reaching its slot; holding position")
                return self._stay_command()
            elif self._timed_out():
                self._fail(
                    "placement descent timed out; holding position to protect packed Cookies"
                )
                return self._stay_command()
            return action

        if self.phase is CookiePhase.OPEN:
            action = self._trajectory_command(
                self._waypoints["place"], self.release_gripper, min_steps=10
            )
            if self._motion_done:
                self._advance(CookiePhase.VERIFY_RELEASE)
            elif self._timed_out():
                # VERIFY_RELEASE already waits a fixed five steps before RETRACT, so advancing
                # here gives the jaws at least that much travel; see the timeouts in APPROACH for
                # why every "arrived"-gated phase needs one.
                self._advance(CookiePhase.VERIFY_RELEASE)
            return action

        if self.phase is CookiePhase.VERIFY_RELEASE:
            action = self._hold_command(self._waypoints["place"], self.release_gripper)
            contacts = self.env.privileged_left_finger_contacts(self._current_cookie())
            # A released thin Cookie may still brush one finger while it settles;
            # only bilateral contact means it is still pinched by the gripper.
            released = not all(contacts)
            self._release_stable_count = self._release_stable_count + 1 if released else 0
            if self.phase_steps >= 5:
                self._advance(CookiePhase.RETRACT)
            return action

        if self.phase is CookiePhase.RETRACT:
            target = (
                self._waypoints["preplace"] if self._retract_stage == "preplace" else self.q_transit
            )
            gripper = (
                self.release_gripper if self._retract_stage == "preplace" else self.open_gripper
            )
            action = self._trajectory_command(target, gripper)
            # Packing is sequential: later Cookies compact earlier ones into the
            # final 2x5 slots. At this intermediate point require stable target-
            # bin containment; the environment performs strict slot validation
            # on the finished arrangement.
            placed = self.env.privileged_cookie_in_target(self._current_cookie())
            self._settle_count = self._settle_count + 1 if placed else 0
            if self._motion_done:
                if self._retract_stage == "preplace":
                    self._retract_stage = "transit"
                    self.phase_steps = 0
                    self._reset_motion()
                else:
                    if self._settle_count < 20:
                        if self._timed_out(100):
                            self._fail(
                                "released Cookie did not remain upright and stable in target"
                            )
                        return action
                    if self._current_cookie() not in self.completed_cookie_indices:
                        self.completed_cookie_indices.append(self._current_cookie())
                        self._packed_positions[self._current_cookie()] = (
                            self.env.privileged_cookie_target_position(self._current_cookie()).copy()
                        )
                    self._advance(CookiePhase.SELECT_COOKIE)
            return action

        raise RuntimeError(f"unhandled Cookie expert phase: {self.phase}")

    @property
    def finished(self) -> bool:
        return self.phase in (CookiePhase.DONE, CookiePhase.FAILED)

    @property
    def failed(self) -> bool:
        return self.phase is CookiePhase.FAILED

    @property
    def status(self) -> str:
        cookie = "-" if self.current_cookie_index is None else str(self.current_cookie_index)
        slot = "-" if self.target_slot_index is None else str(self.target_slot_index)
        return f"phase={self.phase.name} cookie={cookie} slot={slot}"

    def _select_cookie(self) -> None:
        transfer_index = len(self.completed_cookie_indices)
        if transfer_index >= self.env.task_config.required_cookies:
            self.current_cookie_index = None
            self.target_slot_index = None
            self._advance(CookiePhase.DONE)
            return
        self.target_slot_index = self._slot_order[transfer_index]
        last_error: RuntimeError | None = None
        for cookie_index in self._cookie_preference:
            if cookie_index in self.completed_cookie_indices:
                continue
            if cookie_index in self.skipped_cookie_indices:
                continue
            if not self.env.privileged_cookie_in_source(cookie_index):
                continue
            cookie_rotation = self.data.xmat[self.env._cookie_bodies[cookie_index]].reshape(3, 3)
            if abs(cookie_rotation[2, 2]) < np.cos(np.deg2rad(12)):
                continue
            self.current_cookie_index = cookie_index
            try:
                self._plan_current_cookie()
            except RuntimeError as exc:
                last_error = exc
                continue
            self._advance(CookiePhase.APPROACH)
            return
        self._fail(
            "no reachable Cookie remains in the source bin"
            + (f": {last_error}" if last_error is not None else "")
        )

    def _grasp_height_offset(self) -> float:
        """Height above the Cookie centre at which the jaws should close.

        Read from ``COOKIE_HALF_SIZE`` on every call instead of cached at
        construction time, because the environment's scene config -- and hence
        the Cookie size -- is what this must track.  Both ``grasp_position`` and
        the preliminary reachability ``tool_height`` in ``_plan_current_cookie``
        must use this same value.
        """
        return _pinch_offset_from_cookie_centre(self.env.COOKIE_HALF_SIZE[2])

    def _aim_descent(self) -> bool:
        """Re-aim the grasp waypoint until the tool is actually there.

        Returns True once the measured tool position is within
        ``tool_tolerance_m`` of the commanded point, or once the retry budget is
        spent.  The correction is the measured error itself: the servos droop by a
        steady-state amount that joint-space arrival cannot detect, so the command
        has to overshoot by exactly that much.
        """
        if self._grasp_target_world is None:
            return True
        site = self.data.site_xpos[self._l_site]
        error = self._grasp_target_world - site
        if float(np.linalg.norm(error)) <= self.tool_tolerance_m:
            return True
        if self._descent_aims >= self.max_descent_aims:
            return True
        self._descent_aims += 1
        try:
            self._waypoints["grasp"] = self._solve_l(
                self._grasp_target_world + error,
                self._target_quat_canonical,
                self.data.qpos[self._l_qpos],
            )
        except RuntimeError:
            return True
        # Force ``_trajectory_command`` to rebuild its interpolation toward the
        # corrected waypoint instead of reusing the previous one.
        self._motion_key = None
        return False

    def _descent_gripper(self) -> float:
        """Jaw opening that puts both pads in the grooves either side of the target.

        The pads must descend into the two gaps, so where they sit matters as much
        as how wide the opening is.  Centring each pad in its groove -- rather than
        merely clearing the target Cookie's top face -- is what buys the tolerance
        to descend without the arm drifting onto a neighbour.

        Both grooves together span ``pitch - cookie_thickness`` and the pair is
        centred on the target, so a pad centred in each has its inner face at
        ``(pitch - pad_thickness) / 2`` from the target's centre.  The face-to-face
        aperture is therefore

            aperture = pitch - pad_thickness

        Aperture is not a knob.  Merely clearing the target's top face left the pad
        centre 0.2 mm from the face edge while the measured drift is 0.6 mm, so a
        pad rode over the top face and the descent stopped short with the jaws
        closing on air; the centred value has over a millimetre of play.

        The formula only means anything when a pad actually fits in a groove.  A
        row gap smaller than the pad cannot be pinched Cookie-by-Cookie at all --
        that is the case the batch expert exists for -- so fall back to clearing
        the top face rather than commanding an opening that would drive the pads
        into the target.

        Converting aperture into a command: the jaws close symmetrically, so the
        face-to-face gap is ``2 * GRIPPER_RANGE_M * opening``.
        """
        pad = float(getattr(self.env, "PAD_THICKNESS_M", 2 * 2 * 0.003175))
        pitch = float(getattr(self.env, "COOKIE_PITCH_Y", 0.0))
        thickness = 2.0 * float(self.env.COOKIE_HALF_SIZE[1])
        aperture = pitch - pad
        if aperture <= thickness:  # no groove wide enough for a pad
            aperture = thickness + self.grasp_clearance_m
        return float(np.clip(aperture / (2.0 * self.env.GRIPPER_RANGE_M), 0.0, 1.0))

    def _plan_current_cookie(self) -> None:
        cookie_position = self.env.privileged_cookie_position(self._current_cookie()).copy()
        approach_position = np.array(
            [cookie_position[0], cookie_position[1], self.approach_height_m], dtype=np.float64
        )
        grasp_position = cookie_position.copy()
        # Pinch the upper edge: deep grasps collide with adjacent upright
        # cookies when lowering into tightly spaced rows.  The offset is derived
        # from the Cookie height, not hard-coded, so it stays correct when
        # ``COOKIE_HALF_SIZE[2]`` changes; see ``_pinch_offset_from_cookie_centre``.
        grasp_position[2] += self._grasp_height_offset()
        # Where the tool should end up, kept so DESCEND can close the loop on it.
        self._grasp_target_world = grasp_position.copy()
        self._descent_aims = 0
        lift_position = approach_position.copy()
        target_position = self.data.xpos[self._tb_id].copy()
        target_rotation = self.data.xmat[self._tb_id].reshape(3, 3).copy()
        canonical_rotation = np.empty(9)
        mujoco.mju_quat2Mat(canonical_rotation, self._target_quat_canonical)
        mujoco.mju_mat2Quat(
            self._target_quat_tilted, (target_rotation @ canonical_rotation.reshape(3, 3)).ravel()
        )
        tool_xy = self.env.TARGET_SLOTS_LOCAL[self._current_slot()]
        # Preliminary reachability must use the same top-edge grasp height as
        # the post-lift, measured-transform planner below, so it has to take the
        # grasp offset from the same source rather than repeating a literal.
        tool_height = (
            self.env.config.cookie_transfer.target_floor_z_m
            + self.env.config.cookie_transfer.bin_wall_thickness_m
            + self.env.COOKIE_HALF_SIZE[2]
            + 0.008
            + self._grasp_height_offset()
        )
        slot_place = target_position + target_rotation @ np.asarray(
            [tool_xy[0], tool_xy[1], tool_height]
        )
        slot_pre = slot_place + target_rotation[:, 2] * 0.07
        q_approach = self._solve_l(
            approach_position, self._target_quat_canonical, self.q_branch_seed
        )
        q_grasp = self._solve_l(grasp_position, self._target_quat_canonical, q_approach)
        q_lift = self._solve_l(lift_position, self._target_quat_canonical, q_grasp)
        place_quat = self._target_quat_tilted.copy()
        try:
            q_pre = self._solve_l(slot_pre, place_quat, self.q_branch_seed)
        except RuntimeError:
            # A rectangular cookie and parallel jaws permit a half-turn about
            # the tool axis, which can avoid a wrist-limit IK branch.
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, place_quat)
            mujoco.mju_mat2Quat(
                place_quat, (rotation.reshape(3, 3) @ np.diag([-1.0, 1.0, -1.0])).ravel()
            )
            q_pre = self._solve_l(slot_pre, place_quat, self.q_branch_seed)
        q_place = self._solve_l(slot_place, place_quat, q_pre)
        self._waypoints = {
            "approach": q_approach,
            "grasp": q_grasp,
            "lift": q_lift,
            "preplace": q_pre,
            "place": q_place,
        }

    def _plan_place_from_grasp(self) -> None:
        """Account for the measured cookie-to-tool transform after pickup."""
        tool_rotation = self.data.site_xmat[self._l_site].reshape(3, 3)
        cookie_body = self.env._cookie_bodies[self._current_cookie()]
        cookie_rotation = self.data.xmat[cookie_body].reshape(3, 3)
        relative_rotation = tool_rotation.T @ cookie_rotation
        relative_position = tool_rotation.T @ (
            self.data.xpos[cookie_body] - self.data.site_xpos[self._l_site]
        )
        box_rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        xy = self.env.TARGET_SLOTS_LOCAL[self._current_slot()]
        floor = (
            self.env.config.cookie_transfer.target_floor_z_m
            + self.env.config.cookie_transfer.bin_wall_thickness_m
        )
        object_z = floor + self.env.COOKIE_HALF_SIZE[2] + 0.008
        self._placement_height = object_z
        target_cookie = self.data.xpos[self._tb_id] + box_rotation @ np.r_[xy, object_z]
        target_tool_rotation = box_rotation @ relative_rotation.T
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, target_tool_rotation.ravel())
        target_tool = target_cookie - target_tool_rotation @ relative_position
        pre_tool = target_tool + box_rotation[:, 2] * 0.07
        pre = self._solve_l(pre_tool, quat, self.data.qpos[self._l_qpos])
        place = self._solve_l(target_tool, quat, pre)
        self._waypoints["preplace"] = pre
        self._waypoints["place"] = place

    def _trajectory_command(
        self, target: np.ndarray, gripper: float, *, min_steps: int = 12
    ) -> np.ndarray:
        if self.command_space == "cartesian":
            # `min_steps` is deliberately unused.  A floor on a step count is a statement about a
            # precomputed queue: how many commands to spread a joint delta over.  There is no
            # queue here -- see `_cartesian_command` for why the arm's own controller makes one
            # unnecessary -- and inventing a floor would only make the arm dwell.
            return self._cartesian_command(target, gripper)
        key = (target.tobytes(), float(gripper), min_steps)
        if self._motion_key != key:
            start = self.env.last_applied_action.copy()
            joint_delta = float(np.max(np.abs(target - start[:7])))
            gripper_delta = abs(gripper - float(start[7]))
            points = max(
                int(np.ceil((joint_delta - 1e-9) / self.max_joint_step_rad)),
                int(np.ceil((gripper_delta - 1e-9) / self.max_gripper_step)),
                min_steps,
            )
            self._motion_actions = []
            for alpha in np.linspace(0.0, 1.0, points)[1:]:
                action = np.zeros(16, dtype=np.float64)
                action[:7] = (1.0 - alpha) * start[:7] + alpha * target
                action[7] = (1.0 - alpha) * start[7] + alpha * gripper
                action[8:15] = self.q_r_hold
                action[15] = self.right_gripper
                self._motion_actions.append(action)
            self._motion_key = key
            self._motion_done = not self._motion_actions
        action = (
            self._motion_actions.pop(0)
            if self._motion_actions
            else self._hold_command(target, gripper)
        )
        self._motion_done = not self._motion_actions
        return action

    def _hold_command(self, target: np.ndarray, gripper: float) -> np.ndarray:
        if self.command_space == "cartesian":
            # "Hold" and "move towards" are the same operation in Cartesian space: the command is
            # a delta from where the arm is, so a goal the arm is already at produces a zero
            # delta.  That is why this needs no separate branch beyond the conversion.
            return self._cartesian_command(target, gripper)
        action = np.zeros(16, dtype=np.float64)
        action[:7] = target
        action[7] = gripper
        action[8:15] = self.q_r_hold
        action[15] = self.right_gripper
        return action

    def _site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the controlled site is now, as a position and a unit quaternion."""

        position = np.asarray(self.data.site_xpos[self._command_site], dtype=np.float64).copy()
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self._command_site])
        return position, quaternion

    def _pose_of(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The controlled site's pose if the left arm were at ``joints``.

        Forward kinematics on a scratch state, the same way `_solve_l` moves a copy of the
        model's `qpos` to evaluate a candidate.  **Exact rather than approximate**: a joint
        target from this planner came *from* an IK solve against a pose, so the pose it means is
        recoverable rather than merely close by.
        """

        assert self._scratch is not None
        self._scratch.qpos[:] = self.data.qpos
        self._scratch.qpos[self._l_qpos] = joints
        mujoco.mj_kinematics(self.model, self._scratch)
        position = np.asarray(
            self._scratch.site_xpos[self._command_site], dtype=np.float64
        ).copy()
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self._scratch.site_xmat[self._command_site])
        return position, quaternion

    def _goal_error(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position error, and rotation error in the frame **the controller reads**.

        The pose error itself is a well-defined thing, and so is the rotation vector that takes
        the current orientation to the goal's: ``log(q_current^-1 @ q_goal)``, expressed in the
        tool's own axes.  What is *not* well defined is which frame the controller wants it in,
        and the two controllers in this repository want different ones -- measured, see
        `OperationalSpaceAdapter.rotation_frame`.  So the tool-frame vector is what is computed
        here and the conversion to world is applied only for a controller that asks for it.  A
        caller that guessed wrong sends the arm away from its goal, so this asks rather than
        assumes.

        The conjugate is written out rather than taken from `mju_negQuat`: that function negates
        all four components, which is the same rotation under the double cover and not the
        conjugate, which negates only the vector part.
        """

        goal_position, goal_quaternion = self._pose_of(target)
        current_position, current_quaternion = self._site_pose()
        conjugate = np.r_[current_quaternion[0], -current_quaternion[1:4]]
        relative = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(relative, conjugate, goal_quaternion)
        rotation_vector = _rotation_vector(relative)
        if self.env.ik.rotation_frame == "world":
            rotation = np.asarray(self.data.site_xmat[self._command_site], dtype=np.float64)
            rotation_vector = rotation.reshape(3, 3) @ rotation_vector
        return goal_position - current_position, rotation_vector

    def _cartesian_command(self, target: np.ndarray, gripper: float) -> np.ndarray:
        """One ``cartesian_delta`` step towards the pose the joint target ``target`` implies.

        No interpolated queue and no step count, because the operational-space controller holds a
        goal as a **delta from where the arm is now**: commanding the whole remaining error,
        clipped to the action's own range, is already a closed loop that decelerates into the goal
        of its own accord.  What the joint path spends a queue buying -- covering a delta without
        exceeding a rate limit -- the arm's controller does per step, so `_motion_done` here is
        read off the measured pose rather than off an exhausted list.

        The right arm is left at zeros rather than at `q_r_hold`.  A zero translation and rotation
        for an arm means "stay where you are commanded to be" in both adapters -- the environment's
        own `step` substitutes the last *commanded* pose for exactly that reason -- so the idle arm
        holds the pose the scene gave it, which is what `q_r_hold` is for.
        """

        position_error, rotation_error = self._goal_error(target)
        action = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
        action[0:3] = np.clip(
            position_error / self.env.config.cartesian_translation_scale_m, -1.0, 1.0
        )
        action[3:6] = np.clip(
            rotation_error / self.env.config.cartesian_rotation_scale_rad, -1.0, 1.0
        )
        action[6] = float(np.clip(2.0 * float(gripper) - 1.0, -1.0, 1.0))
        self._motion_done = bool(
            float(np.linalg.norm(position_error)) <= self.cartesian_arrival_tolerance_m
            and float(np.linalg.norm(rotation_error)) <= self.cartesian_arrival_tolerance_rad
        )
        return action

    def _stay_command(self) -> np.ndarray:
        """A command that moves nothing, expressed in whichever space this expert speaks.

        The joint path already had one -- repeat the last applied command -- and for a
        position-servo arm that holds the pose, because the command *is* the pose.  It is returned
        **verbatim**, gripper entry included, because that is what "hold" means when a Cookie is
        in the jaws: the entry already carries the opening that was commanded.

        In Cartesian space the same idea is a **zero delta**, because the command is relative to
        where the arm is.  `env.last_applied_action` there is the *canonical* vector: 16 wide, and
        on the Panda a torque vector, so returning it as an action is a shape error at best and a
        full-speed lunge in an arbitrary direction at worst.  Its gripper entry therefore has to be
        built from the **measured** opening rather than left at zero -- a delta's gripper entry of
        zero means "half open", which is a real command and would let go of the Cookie this exists
        to keep hold of.
        """

        if self.command_space == "cartesian":
            opening = float(np.clip(self.env.current_joint_action[7], 0.0, 1.0))
            action = np.zeros(CARTESIAN_ACTION_DIM, dtype=np.float64)
            action[6] = float(np.clip(2.0 * opening - 1.0, -1.0, 1.0))
            return action
        return self.env.last_applied_action.copy()

    def _reset_motion(self) -> None:
        self._motion_actions = []
        self._motion_key = None
        self._motion_done = False

    def _reached(self, target: np.ndarray, *, tolerance: float | None = None) -> bool:
        if self.command_space == "cartesian":
            position_error, rotation_error = self._goal_error(target)
            del tolerance
            return bool(
                float(np.linalg.norm(position_error)) <= self.cartesian_arrival_tolerance_m
                and float(np.linalg.norm(rotation_error)) <= self.cartesian_arrival_tolerance_rad
            )
        joints = self.env.current_joint_action[:7]
        return bool(
            np.max(np.abs(joints - target))
            <= (self.joint_tolerance_rad if tolerance is None else tolerance)
        )

    def _has_verified_grasp(self) -> bool:
        contacts = self.env.privileged_left_finger_contacts(self._current_cookie())
        touches = self.env.privileged_left_touch_values()
        return all(contacts) and all(value >= 0.02 for value in touches)

    def _has_verified_lift(self) -> bool:
        cookie_position = self.env.privileged_cookie_position(self._current_cookie())
        eef_position = self.data.site_xpos[self._l_site]
        follows_gripper = np.linalg.norm(cookie_position - eef_position) <= 0.065
        lifted = cookie_position[2] >= self.env.SOURCE_FLOOR_TOP_Z + 0.055
        return bool(lifted and follows_gripper)

    def _cookie_rest_height(self) -> float:
        """World z of an upright Cookie standing on the source bin floor.

        This is the reference for "is the Cookie back on the bin floor?", so it
        is derived from ``COOKIE_HALF_SIZE`` for the same reason the grasp height
        is: it is a function of the Cookie's size, not a free tuning knob.  A
        Cookie that has shrunk must not have to rise the old, taller Cookie's
        height before a drop is noticed.
        """
        return self.env.SOURCE_FLOOR_TOP_Z + self.env.COOKIE_HALF_SIZE[2]

    def _cookie_dropped(self) -> bool:
        cookie_position = self.env.privileged_cookie_position(self._current_cookie())
        eef_position = self.data.site_xpos[self._l_site]
        # Contact flags can flicker while a thin object remains securely held.
        # Separation from the end effector is the robust transfer-time signal.
        separated = np.linalg.norm(cookie_position - eef_position) > 0.085
        # ``0.025`` used to stand here.  It was only correct while the Cookie was
        # 50 mm tall, because it is the Cookie's *half height* -- the height of an
        # upright Cookie's centre above the bin floor, i.e. exactly
        # ``COOKIE_HALF_SIZE[2]``.  Once the Cookie was shortened to 25 mm, the
        # stale literal sat 12.5 mm above the real rest height, so any Cookie
        # that rose by less than that was still reported as back on the floor.
        # Derive it instead.
        fell_back = cookie_position[2] <= self._cookie_rest_height()
        return bool(self.phase_steps > 5 and (separated or fell_back))

    def _retry(self, reason: str) -> None:
        self.retry_reasons.append(
            f"{self.phase.name}: Cookie {self._current_cookie()}, "
            f"slot {self._current_slot()}: {reason}"
        )
        slot = self._current_slot()
        self.retry_counts[slot] += 1
        if self.retry_counts[slot] > self.max_retries:
            self.skipped_cookie_indices.append(self._current_cookie())
            self._advance(CookiePhase.SELECT_COOKIE)
            return
        self._grasp_stable_count = 0
        self._release_stable_count = 0
        self._settle_count = 0
        self._move_stage = "transit"
        self._recovering = True
        try:
            self._plan_current_cookie()
        except RuntimeError as exc:
            self._fail(str(exc))
            return
        self._advance(CookiePhase.APPROACH)

    def _advance(self, phase: CookiePhase) -> None:
        self.phase = phase
        self.phase_steps = 0
        self._reset_motion()
        self.transition_history.append(phase)
        if phase is CookiePhase.MOVE_TO_SLOT:
            self._move_stage = "transit"
        elif phase is CookiePhase.VERIFY_GRASP:
            self._grasp_stable_count = 0
        elif phase is CookiePhase.VERIFY_RELEASE:
            self._release_stable_count = 0
        elif phase is CookiePhase.RETRACT:
            self._settle_count = 0
            self._retract_stage = "preplace"

    def _fail(self, reason: str) -> None:
        self.failure_phase = self.phase.name
        self.failure_reason = reason
        self._advance(CookiePhase.FAILED)

    def _timed_out(self, limit: int | None = None) -> bool:
        return self.phase_steps >= (limit or self.phase_timeout_steps)

    def _current_cookie(self) -> int:
        if self.current_cookie_index is None:
            raise RuntimeError("Cookie expert has no active Cookie")
        return self.current_cookie_index

    def _current_slot(self) -> int:
        if self.target_slot_index is None:
            raise RuntimeError("Cookie expert has no active target slot")
        return self.target_slot_index

    def close(self) -> None:
        return None


def make_cookie_transfer_expert(env: A3CookieTransferEnv) -> A3CookieTransferExpert:
    """CLI factory: ``--policy a3_dual_arm_sim.expert:make_cookie_transfer_expert``."""

    return A3CookieTransferExpert(env)
