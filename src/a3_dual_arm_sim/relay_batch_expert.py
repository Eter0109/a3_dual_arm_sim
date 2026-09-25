"""Fill a lane of boxes by relay, with a measured contact-only box exchange.

The proven single-box batch expert is reused unchanged for each fill.  This module owns
the separate handoff: a filled box is physically pushed out of the station, the next
empty one is grasped by its rear wall and slid into the station, and the fill repeats.

A *lane* rather than a pair, because the push is what generalises: pushing the station
box out shoves the parked line along by contact, so the pads never have to reach past
the box they are pushing.  See :class:`RelayBatchExpert` for what that costs and why the
parked boxes are reported rather than graded.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .box_support import BoxSupportController
from .cookie_transfer import A3CookieTransferEnv
from .expert import CookiePhase
from .same_column_batch_expert import A3SameColumnBatchExpert


class RightBoxPushController:
    """Push one free box in +Y using the closed right finger pads."""

    PUSH_SPEED_M_PER_STEP = 0.0008
    FORCE_LIMIT_N = 80.0

    # How the pads answer a yawed box: for a yaw of ``yaw`` the commanded
    # contact x moves back by ``YAW_COMPENSATION_M_PER_RAD * yaw``, clipped to
    # ``YAW_COMPENSATION_CLIP_M`` either side of where the push started.  Both
    # are named because the pair is what the push's accuracy lives and dies by --
    # see the class docstring of :mod:`a3_dual_arm_sim.batch_profile` for the
    # measurement that says so.
    YAW_COMPENSATION_M_PER_RAD = 0.08
    YAW_COMPENSATION_CLIP_M = 0.010

    def __init__(self, env: A3CookieTransferEnv, box_id: int, destination_y: float):
        self.env = env
        self.data = env.data
        self.box_id = box_id
        self.destination_y = float(destination_y)
        self.initial_position = self.data.xpos[box_id].copy()
        self.helper = BoxSupportController(env)
        self.helper.bin_id = box_id
        self.home = env.last_applied_action[8:15].copy()
        self.quat = np.empty(4)
        mujoco.mju_mat2Quat(self.quat, self.data.site_xmat[self.helper.site].copy())
        self.x = float(self.initial_position[0])
        self.start_y = float(
            self.initial_position[1] - env.config.cookie_transfer.target_bin_half_size_m[1] - 0.027
        )
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.stable = 0
        self.failed: str | None = None
        self.done = False
        self.peak_force_n = 0.0
        self._force_over_count = 0
        self._push_goal_y = self.start_y
        self._retract_y = None
        self._target_q = None

    @property
    def box_position(self) -> np.ndarray:
        return self.data.xpos[self.box_id].copy()

    @property
    def box_yaw_rad(self) -> float:
        rotation = self.data.xmat[self.box_id].reshape(3, 3)
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0
        self.stable = 0
        self._target_q = None

    def _command(self, target_q: np.ndarray, opening: float) -> np.ndarray:
        action = self.env.last_applied_action.copy()
        action[8:15] += np.clip(target_q - action[8:15], -0.018, 0.018)
        action[15] = opening
        return action

    def _pose_command(self, y: float, z: float, opening: float) -> tuple[np.ndarray, bool]:
        if self._target_q is None:
            self._target_q = self.helper.solve(np.array([self.x, y, z]), self.quat)
        action = self._command(self._target_q, opening)
        site_position = self.data.site_xpos[self.helper.site]
        reached = (
            np.linalg.norm(site_position - np.array([self.x, y, z])) < 0.004
            and abs(self.env.current_joint_action[15] - opening) < 0.08
        )
        return action, reached

    def act(self) -> np.ndarray:
        if self.failed or self.done:
            return self.env.last_applied_action.copy()
        self.phase_steps += 1
        forces = self.helper.contact_forces()
        self.peak_force_n = max(self.peak_force_n, float(np.max(forces)))
        self._force_over_count = (
            self._force_over_count + 1 if np.max(forces) > self.FORCE_LIMIT_N else 0
        )
        if self._force_over_count >= 3:
            self.failed = f"right fingers overloaded while pushing: {forces.tolist()} N"
            return self.env.last_applied_action.copy()
        if abs(self.box_position[0] - self.initial_position[0]) > 0.025:
            self.failed = "box drifted sideways during right-arm push"
            return self.env.last_applied_action.copy()
        up = self.data.xmat[self.box_id].reshape(3, 3)[2, 2]
        if abs(float(up)) < math.cos(math.radians(12)):
            self.failed = "box tilted during right-arm push"
            return self.env.last_applied_action.copy()
        phase_timeout = {
            "APPROACH": 220,
            "DESCEND": 220,
            "PUSH": 420,
            "RETRACT": 180,
            "HOME": 220,
        }[self.phase]
        if self.phase_steps > phase_timeout:
            self.failed = (
                f"right-arm {self.phase} timeout; box_y={self.box_position[1]:.4f}, "
                f"target_y={self.destination_y:.4f}"
            )
            return self.env.last_applied_action.copy()
        try:
            if self.phase == "APPROACH":
                action, reached = self._pose_command(self.start_y, 0.840, 0.0)
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._advance("DESCEND")
                return action
            if self.phase == "DESCEND":
                action, reached = self._pose_command(self.start_y, 0.755, 0.0)
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._push_goal_y = float(self.data.site_xpos[self.helper.site, 1])
                    self._advance("PUSH")
                return action
            if self.phase == "PUSH":
                if self.box_position[1] >= self.destination_y - 0.0025:
                    self._retract_y = float(self.data.site_xpos[self.helper.site, 1] - 0.035)
                    self._advance("RETRACT")
                    return self.env.last_applied_action.copy()
                # Compensate yaw while continuing the physical push.
                self.x = float(
                    np.clip(
                        self.box_position[0] - self.YAW_COMPENSATION_M_PER_RAD * self.box_yaw_rad,
                        self.initial_position[0] - self.YAW_COMPENSATION_CLIP_M,
                        self.initial_position[0] + self.YAW_COMPENSATION_CLIP_M,
                    )
                )
                # Advance the commanded contact point slowly. A blocked pad
                # cannot accumulate a large unseen penetration target.
                measured_y = float(self.data.site_xpos[self.helper.site, 1])
                self._push_goal_y = min(
                    self._push_goal_y + self.PUSH_SPEED_M_PER_STEP,
                    measured_y + 0.018,
                )
                self._target_q = None
                if (
                    self.phase_steps >= 100
                    and self.box_position[1] - self.initial_position[1] < 0.005
                ):
                    self.failed = "right pad contacted but did not move the box"
                    return self.env.last_applied_action.copy()
                return self._pose_command(self._push_goal_y, 0.755, 0.0)[0]
            if self.phase == "RETRACT":
                assert self._retract_y is not None
                action, reached = self._pose_command(self._retract_y, 0.755, 0.0)
                # The retreat only needs to break contact; it need not settle
                # to the exact millimetre before the arm returns home.
                reached = reached or (
                    self.data.site_xpos[self.helper.site, 1] <= self._retract_y + 0.003
                    and abs(self.data.site_xpos[self.helper.site, 2] - 0.755) < 0.010
                )
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._advance("HOME")
                return action
            action = self._command(self.home, 1.0)
            reached = (
                np.max(np.abs(self.data.qpos[self.helper.qids] - self.home)) < 0.035
                and self.env.current_joint_action[15] > 0.95
            )
            self.stable = self.stable + 1 if reached else 0
            if self.stable >= 5:
                self.done = True
            return action
        except RuntimeError as exc:
            self.failed = str(exc)
            return self.env.last_applied_action.copy()


class RightBoxCarryController:
    """Pinch an empty box's rear wall and slide it into the filling station."""

    MOVE_SPEED_M_PER_STEP = 0.0008
    #: Finger opening of the pinch that carries the box, on the same 0..1 scale
    #: as a joint target (0 is closed).  Tighter than it looks: the rear wall is a
    #: few millimetres thick, so what holds the box against twisting is the pinch
    #: force, not the geometry.  This value goes with the undamped servo the relay
    #: is calibrated for (see ``arm_actuator_damping``): an arm that holds its
    #: command more rigidly (actuator damping, which the fast fill needs) lets the
    #: same box twist past the 10 deg guard, and needs a tighter 0.085.
    GRASP_OPENING = 0.10
    FORCE_LIMIT_N = 80.0

    #: How close to the station the box must be before the carry lets go.
    #: This is the landing's accuracy, and it has to be inside the window the
    #: *next* stage accepts: the fill that follows refuses a box whose placement
    #: pose its IK cannot reach, which is about 2 mm of box offset.  It used to be
    #: 4 mm, so the carry deliberately let go up to 4 mm short and the fill's
    #: reachability pre-check then refused the result -- measured, a relay run
    #: ended at "FILL_B setup: target column 2 unreachable ... position error
    #: 2.28 mm" with the box 3.4 mm from the station.
    RELEASE_WITHIN_M = 0.0015
    #: How far the slide goal may lead the pads' measured y, so a box that will not
    #: slide is pushed with a bounded offset rather than an ever-growing one.
    LEAD_M = 0.015

    def __init__(self, env: A3CookieTransferEnv, box_id: int, destination_y: float):
        self.env = env
        self.data = env.data
        self.box_id = box_id
        self.destination_y = float(destination_y)
        self.initial_position = self.data.xpos[box_id].copy()
        self.helper = BoxSupportController(env)
        self.helper.bin_id = box_id
        self.home = env.last_applied_action[8:15].copy()
        rotation = self.data.site_xmat[self.helper.site].reshape(3, 3).copy()
        self.quat = np.empty(4)
        mujoco.mju_mat2Quat(self.quat, rotation.ravel())
        pad_center = self.data.geom_xpos[self.helper.fingers].mean(axis=0)
        pad_offset = pad_center - self.data.site_xpos[self.helper.site]
        self._pad_offset = pad_offset
        rear_wall_y = (
            self.initial_position[1] - env.config.cookie_transfer.target_bin_half_size_m[1]
        )
        self.anchor = np.array([self.initial_position[0], rear_wall_y, 0.780]) - pad_offset
        self.phase = "APPROACH"
        self.phase_steps = 0
        self.stable = 0
        self.failed: str | None = None
        self.done = False
        self.peak_force_n = 0.0
        self._target_q = None
        self._move_goal_y = float(self.anchor[1])
        self._grasped = False
        #: The box's yaw when the carry started.  The scene draws the spare box's
        #: yaw (up to 5.7 deg), so the guard below has to ask what the carry did
        #: to the box, not how the box happens to be standing: against an absolute
        #: 10 deg, a box that started at 5.7 deg and turned 4.3 deg was reported
        #: as "empty box rotated during carry".
        self._initial_yaw = self.box_yaw_rad

    @property
    def box_position(self) -> np.ndarray:
        return self.data.xpos[self.box_id].copy()

    @property
    def box_yaw_rad(self) -> float:
        rotation = self.data.xmat[self.box_id].reshape(3, 3)
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.phase_steps = 0
        self.stable = 0
        self._target_q = None

    def _command(self, target_q: np.ndarray, opening: float) -> np.ndarray:
        action = self.env.last_applied_action.copy()
        action[8:15] += np.clip(target_q - action[8:15], -0.018, 0.018)
        action[15] = opening
        return action

    def _pose_command(self, target: np.ndarray, opening: float) -> tuple[np.ndarray, bool]:
        if self._target_q is None:
            self._target_q = self.helper.solve(target, self.quat)
        action = self._command(self._target_q, opening)
        reached = (
            np.linalg.norm(self.data.site_xpos[self.helper.site] - target) < 0.005
            and abs(self.env.current_joint_action[15] - opening) < 0.08
        )
        return action, reached

    def act(self) -> np.ndarray:
        if self.failed or self.done:
            return self.env.last_applied_action.copy()
        self.phase_steps += 1
        forces = self.helper.contact_forces()
        self.peak_force_n = max(self.peak_force_n, float(np.max(forces)))
        if np.max(forces) > self.FORCE_LIMIT_N:
            self.failed = f"right fingers overloaded while carrying: {forces.tolist()} N"
            return self.env.last_applied_action.copy()
        if self._grasped:
            if abs(self.box_position[0] - self.initial_position[0]) > 0.015:
                self.failed = "empty box drifted sideways during carry"
                return self.env.last_applied_action.copy()
            if abs(self.box_yaw_rad - self._initial_yaw) > math.radians(10):
                self.failed = "empty box rotated during carry"
                return self.env.last_applied_action.copy()
        timeout = {
            "APPROACH": 240,
            "DESCEND": 240,
            "CLOSE": 100,
            "MOVE": 450,
            "RELEASE": 100,
            "RETRACT": 200,
            "HOME": 240,
        }[self.phase]
        if self.phase_steps > timeout:
            self.failed = (
                f"right-arm carry {self.phase} timeout; "
                f"box_y={self.box_position[1]:.4f}, goal_y={self.destination_y:.4f}, "
                f"contacts={forces.tolist()}"
            )
            return self.env.last_applied_action.copy()
        try:
            if self.phase == "APPROACH":
                target = self.anchor + [0, 0, 0.080]
                action, reached = self._pose_command(target, 1.0)
            elif self.phase == "DESCEND":
                # A direct joint-space jump to the low pose sweeps the inner
                # finger through the bin floor before reaching the wall.
                # Keep each IK waypoint just below the measured wrist pose.
                target = self.anchor.copy()
                target[2] = max(
                    self.anchor[2],
                    float(self.data.site_xpos[self.helper.site, 2]) - 0.002,
                )
                self._target_q = None
                action, reached = self._pose_command(target, 1.0)
                reached = reached and target[2] <= self.anchor[2] + 0.001
            elif self.phase == "CLOSE":
                action, _ = self._pose_command(self.anchor, self.GRASP_OPENING)
                reached = bool(np.all(forces > 0.3))
            elif self.phase == "MOVE":
                if self.box_position[1] >= self.destination_y - self.RELEASE_WITHIN_M:
                    self._advance("RELEASE")
                    return self.env.last_applied_action.copy()
                # "Grip lost" means the box is no longer held, and the test for
                # that is that *neither* pad reads a force.  Measured: the rear
                # wall is thin and the jaws are wide relative to it, so the two
                # pads do not share the load evenly -- the inner pad carries
                # 1.1-2.6 N while the outer one hovers between 0.04 and 0.2 N.
                # Reading a single pad therefore sits right on its own
                # threshold, and an arm that tracks its command more precisely
                # (actuator `kv` damping) holds that pad lower more often
                # without the box ever moving relative to the jaws.  The sum is
                # what says the box is still held.
                if np.max(forces) < 0.08:
                    self.stable += 1
                    if self.stable > 8:
                        self.failed = f"empty box grip lost during carry: {forces.tolist()} N"
                        return self.env.last_applied_action.copy()
                else:
                    self.stable = 0
                measured_y = float(self.data.site_xpos[self.helper.site, 1])
                self._move_goal_y = min(
                    self._move_goal_y + self.MOVE_SPEED_M_PER_STEP,
                    measured_y + self.LEAD_M,
                )
                # The pinch holds the box, so the pads and the box are one body
                # and the slide stays on the line the grip was closed on.  Steering
                # it sideways was tried three ways and every one trades the two
                # errors against each other rather than removing either: measured
                # from the relay's own state, a 0.08 m/rad yaw term left the box
                # 0.27 mm from the station but 5.41 deg off, and a rate-limited
                # lateral loop on the station's x left it 6.4 mm out with 8.4 deg
                # of yaw.  A pinch moves the box *with* the pads, so a sideways
                # command is a sideways displacement, and the torque that produces
                # is what the next fill's reachability pre-check reads.
                target = self.anchor.copy()
                target[1] = self._move_goal_y
                self._target_q = None
                return self._pose_command(target, self.GRASP_OPENING)[0]
            elif self.phase == "RELEASE":
                action = self._command(self.data.qpos[self.helper.qids], 1.0)
                reached = bool(self.env.current_joint_action[15] > 0.95)
            elif self.phase == "RETRACT":
                target = self.data.site_xpos[self.helper.site].copy()
                target[2] = 0.850
                action, reached = self._pose_command(target, 1.0)
            else:
                action = self._command(self.home, 1.0)
                reached = bool(np.max(np.abs(self.data.qpos[self.helper.qids] - self.home)) < 0.035)
            self.stable = self.stable + 1 if reached else 0
            if self.stable >= 5:
                next_phase = {
                    "APPROACH": "DESCEND",
                    "DESCEND": "CLOSE",
                    "CLOSE": "MOVE",
                    "RELEASE": "RETRACT",
                    "RETRACT": "HOME",
                    "HOME": "DONE",
                }[self.phase]
                if next_phase == "DONE":
                    self.done = True
                else:
                    if next_phase == "MOVE":
                        self._grasped = True
                        self._move_goal_y = float(self.data.site_xpos[self.helper.site, 1])
                    self._advance(next_phase)
            return action
        except RuntimeError as exc:
            self.failed = str(exc)
            return self.env.last_applied_action.copy()


class RelayBatchExpert:
    """Fill N boxes in a lane, exchanging each full one for the next empty one.

    One box sits at the *station* and is filled where it stands.  When it is full it is
    pushed one push-distance further out, the next box in the queue is pinched by its
    rear wall and slid into the station, and the fill repeats.  The **last** box is not
    pushed: there is nothing left to make room for, and pushing a filled box for no
    reason is a way to spill it.

    For two boxes this is the shipped relay exactly -- same order, same numbers, one
    push at 90 mm and one carry -- which is why its measured yield still applies to it.

    **A push shoves the whole line.**  A box pushed out of the station arrives at
    ``station + push_distance``, and a box already parked there is shoved along by
    contact, so the line advances by one push distance per fill and the pads never touch
    a parked box.  Pad travel is therefore the same whatever the box count, which is
    what the measured envelope requires: the pads reach 99 mm at the shipped station, so
    they can only ever follow the box they are pushing -- and the box they shove along
    ends up where its own contact put it, not where a controller aimed it.

    That is also why the parked line's *x* and yaw are only reported and not graded: a
    box moved by contact is not under control, so a bound on it would be a bound on
    something nothing is steering.  What is graded is what the next fill needs -- the
    station box's placement, which the carry delivers.
    """

    #: How long a verify stage may wait for the carry's result to settle before the
    #: state it sees is taken as final.  Both verify stages used to fail on a
    #: threshold *looser* than the one they passed on, which left a band where
    #: neither branch fired and the stage parked until the horizon ran out:
    #:
    #:   the push verify   pass at y >= station + 85 mm, fail only below station + 50 mm
    #:                     -> parked for a box that overshot 50..85 mm
    #:   the carry verify  pass within 8 mm, fail only beyond 12 mm
    #:                     -> parked for a box that stopped 8..12 mm out
    #:
    #: A band like that reads as "still running" rather than as a failure, and was
    #: observed as a run reaching the 12000-step horizon where a successful one
    #: takes 4798.  Waiting a bounded time instead of testing a second threshold
    #: closes the gap by construction: every state either settles or times out.
    VERIFY_SETTLE_STEPS = 200

    #: How clear of the station a pushed box has to be before the next one may be
    #: carried in.  It is the fill's own window that sets it: a box closer than this
    #: leaves the placement poses of the box behind it out of reach.
    PUSHED_CLEAR_M = 0.085
    #: How close to the station a carried box has to arrive.  Not the accuracy the
    #: carry is asked for -- that is its own 1.5 mm release threshold -- but the bound
    #: the *next fill* needs, since the fill's reachability pre-check refuses a box
    #: whose placement pose its IK cannot solve at about 2 mm of offset.
    ARRIVED_WITHIN_M = 0.008

    def __init__(self, env: A3CookieTransferEnv):
        self.env = env
        scene = env.config.cookie_transfer
        #: Every box in the lane, station first.  The station box is filled where it
        #: stands; each one behind it is carried in when its turn comes.
        self.boxes: tuple[int, ...] = env.target_bin_bodies
        self.names: tuple[str, ...] = scene.target_bin_body_names
        self.station = np.asarray(scene.target_bin_world_position_m[:2], dtype=np.float64)
        #: How far each push moves a filled box out of the station, and therefore the
        #: spacing the parked line compacts to.
        self.push_distance = float(env.config.relay_push_distance_m)
        self.fill: A3SameColumnBatchExpert | None = None
        self.pusher: RightBoxPushController | RightBoxCarryController | None = None
        #: Which box is at the station.  The lane is worked from the station backwards,
        #: so this counts up as the queue advances.
        self.index = 0
        self.stage = "FILL"
        self.failed: str | None = None
        self.done = False
        self.fill_reports: list[dict] = []
        self.push_reports: list[dict] = []
        #: Which Cookies each box holds, by box index.  A list rather than a pair,
        #: because the count is the thing that generalises.
        self.indices: list[list[int]] = [[] for _ in self.boxes]
        self._settled_steps = 0
        #: Steps spent in the current verify stage, so it can time out rather than
        #: wait forever when a result is short of the pass condition but not short
        #: enough to trip a failure threshold.
        self._verify_steps = 0

    def reset(self) -> None:
        self.stage = "FILL"
        self.failed = None
        self.done = False
        self.index = 0
        self.fill_reports = []
        self.push_reports = []
        self.indices = [[] for _ in self.boxes]
        self._settled_steps = 0
        self._verify_steps = 0
        self.pusher = None
        self.env._target_bin_body = self.boxes[0]
        self.fill = A3SameColumnBatchExpert(self.env)
        self.fill.reset()

    @property
    def status(self) -> str:
        where = f"box={self.index + 1}/{len(self.boxes)}"
        if self.stage == "FILL" and self.fill is not None:
            return f"stage={self.stage} {where} {self.fill.status}"
        if self.stage in ("PUSH", "CARRY") and self.pusher is not None:
            return (
                f"stage={self.stage} {where} phase={self.pusher.phase} "
                f"box_y={self.pusher.box_position[1]:.3f} "
                f"goal_y={self.pusher.destination_y:.3f}"
            )
        return f"stage={self.stage} {where}"

    @property
    def box_a(self) -> int:
        """The station box's body id -- what a two-box scene calls box A."""

        return self.boxes[0]

    @property
    def box_b(self) -> int:
        """The box behind the station -- what a two-box scene calls box B.

        These two exist because the shipped two-box dataset's summary keys and its tests
        name the first two boxes that way.  A lane of more than two boxes has no "box B"
        in that sense, and reports every box through :meth:`all_counts`; a single-box
        scene has no second box at all, so this raises rather than inventing an index.
        """

        if len(self.boxes) < 2:
            raise IndexError(
                f"this scene has {len(self.boxes)} box, so there is no box B; "
                f"use `boxes` for the lane"
            )
        return self.boxes[1]

    def _counts_in_box(self, box_id: int, indices: list[int]) -> tuple[int, int]:
        """How many of ``indices`` the box holds, and how many it holds *placed*.

        Two numbers, because one cannot say why a box reports fewer than ten.
        The first is geometric: the Cookie's centre is inside the box.  The
        second is the scene's own contract -- ``privileged_cookie_in_target``,
        which additionally wants the Cookie settled, released, and (unless the
        scene says otherwise) still upright.  A run that failed the push verify with
        "7/10 Cookies aboard" was holding all ten: three had leaned about 24
        degrees during the push, so the box was full and the count was about
        posture.  Reporting both makes a rejected episode say which it was.
        """

        old = self.env._target_bin_body
        try:
            self.env._target_bin_body = box_id
            contained = sum(self.env.privileged_cookie_in_target_region(i) for i in indices)
            placed = sum(self.env.privileged_cookie_in_target(i) for i in indices)
        finally:
            self.env._target_bin_body = old
        return contained, placed

    def _count(self, box_id: int, indices: list[int]) -> int:
        """How many of ``indices`` the box holds under the scene's contract."""

        return self._counts_in_box(box_id, indices)[1]

    def all_counts(self) -> list[int]:
        """How many Cookies each box holds, station first."""

        everything = list(range(len(self.env._cookie_bodies)))
        return [self._count(body, everything) for body in self.boxes]

    def counts(self) -> tuple[int, int]:
        """The first two boxes' counts, which is what a two-box summary states.

        Kept as a pair rather than replaced by :meth:`all_counts` because the collection
        summary's ``box_a_cookie_count`` / ``box_b_cookie_count`` are a dataset contract,
        and a lane longer than two boxes reports the rest through ``all_counts``.
        """

        every = self.all_counts()
        return (
            every[0] if every else 0,
            every[1] if len(every) > 1 else 0,
        )

    def _new_pusher(self, box_index: int, destination_y: float, stage: str) -> None:
        """Start the controller that moves one box, and name the stage it is in.

        A *push* moves the station box out with the closed pads; a *carry* pinches a
        queue box's rear wall and slides it in.  Which is needed follows from what the
        move is for, not from which box it is, so the stage picks the controller.
        """

        box_id = self.boxes[box_index]
        if stage == "CARRY":
            self.pusher = RightBoxCarryController(self.env, box_id, destination_y)
        else:
            self.pusher = RightBoxPushController(self.env, box_id, destination_y)
        self.stage = stage

    def act(self) -> np.ndarray:
        if self.failed or self.done:
            return self.env.last_applied_action.copy()
        if self.stage == "FILL":
            assert self.fill is not None
            action = self.fill.act()
            if self.fill.failed:
                self.failed = f"{self.stage} box {self.index + 1} ({self.names[self.index]}): {self.fill.failure_reason}"
            elif self.fill.finished and self.fill.phase is CookiePhase.DONE:
                indices = list(self.fill.completed_cookie_indices)
                self.indices[self.index] = indices
                self.fill_reports.append(
                    {
                        "box": self.names[self.index],
                        "cookie_ids": indices,
                        "batches": self.fill.batch_reports,
                    }
                )
                if self.index == len(self.boxes) - 1:
                    # Nothing left to make room for, so the last box stays where it is.
                    self.stage = "VERIFY_ALL"
                    self._settled_steps = 0
                    self._verify_steps = 0
                else:
                    self._new_pusher(self.index, self.station[1] + self.push_distance, "PUSH")
            return action
        if self.stage in ("PUSH", "CARRY"):
            assert self.pusher is not None
            action = self.pusher.act()
            if self.pusher.failed:
                self.failed = (
                    f"{self.stage} box {self.index + 1} ({self.names[self.index]}): "
                    f"{self.pusher.failed}"
                )
            elif self.pusher.done:
                self.push_reports.append(
                    {
                        "box": self.names[self.index],
                        "method": "push" if self.stage == "PUSH" else "grasp_and_slide",
                        "from_y_m": float(self.pusher.initial_position[1]),
                        "to_y_m": float(self.pusher.box_position[1]),
                        "final_yaw_deg": math.degrees(self.pusher.box_yaw_rad),
                        "peak_finger_force_n": self.pusher.peak_force_n,
                    }
                )
                self.stage = "VERIFY_PUSH" if self.stage == "PUSH" else "VERIFY_CARRY"
                self._settled_steps = 0
                self._verify_steps = 0
            return action
        if self.stage == "VERIFY_PUSH":
            body = self.boxes[self.index]
            position = self.env.data.xpos[body]
            clearance_mm = float(position[1] - self.station[1]) * 1000.0
            contained, placed = self._counts_in_box(body, self.indices[self.index])
            good = position[1] >= self.station[1] + self.PUSHED_CLEAR_M and contained == 10
            self._settled_steps = self._settled_steps + 1 if good else 0
            self._verify_steps += 1
            if self._settled_steps >= 15:
                self._new_pusher(self.index + 1, self.station[1], "CARRY")
            elif self._verify_steps >= self.VERIFY_SETTLE_STEPS:
                self.failed = (
                    f"filled box {self.index + 1} ({self.names[self.index]}) did not "
                    f"clear the filling station: {clearance_mm:.1f} mm clear of it "
                    f"(needs {self.PUSHED_CLEAR_M * 1000:.1f}) with {contained}/10 "
                    f"Cookies in the box ({placed}/10 of them placed flush)"
                )
            return self.env.last_applied_action.copy()
        if self.stage == "VERIFY_CARRY":
            body = self.boxes[self.index + 1]
            offset = self.env.data.xpos[body, :2] - self.station
            distance_mm = float(np.linalg.norm(offset)) * 1000.0
            contained, placed = self._counts_in_box(
                self.boxes[self.index], self.indices[self.index]
            )
            good = distance_mm < self.ARRIVED_WITHIN_M * 1000.0 and contained == 10
            self._settled_steps = self._settled_steps + 1 if good else 0
            self._verify_steps += 1
            if self._settled_steps >= 15:
                self.index += 1
                self.env._target_bin_body = self.boxes[self.index]
                self.fill = A3SameColumnBatchExpert(self.env)
                try:
                    self.fill.reset()
                except RuntimeError as exc:
                    self.failed = f"FILL setup for box {self.index + 1}: {exc}"
                else:
                    self.stage = "FILL"
            elif self._verify_steps >= self.VERIFY_SETTLE_STEPS:
                self.failed = (
                    f"box {self.index + 2} ({self.names[self.index + 1]}) did not reach "
                    f"the filling station: {distance_mm:.1f} mm from it (needs under "
                    f"{self.ARRIVED_WITHIN_M * 1000:.1f}) at x={offset[0] * 1000:+.1f} "
                    f"y={offset[1] * 1000:+.1f} mm, with {contained}/10 Cookies in box "
                    f"{self.index + 1} ({placed}/10 of them placed flush)"
                )
            return self.env.last_applied_action.copy()
        if self.stage == "VERIFY_ALL":
            # Every box is graded at once, and each is graded by what its position
            # makes meaningful: the *last* one is at the station and has to be there,
            # and every box behind it was pushed out and has to be clear of it.  For two
            # boxes that is the shipped criterion exactly -- one parked box clear by
            # 85 mm, one box at the station within 8 mm -- and for a longer lane it is
            # the same two statements applied to more boxes.
            #
            # It used to demand that *every* box past the station sit within 8 mm of it,
            # which only the last one ever does: measured on a three-box lane with box 2
            # at the station and boxes 1 and 0 parked at 90 and 180 mm, the middle box
            # was 90 mm out and the criterion refused a lane laid out exactly as
            # intended.
            #
            # The parked boxes' x and yaw are deliberately not graded -- they were moved
            # by contact, not by a controller -- but they are reported (see
            # `push_reports`).
            source = sum(
                self.env.privileged_cookie_in_source(i) for i in range(len(self.env._cookie_bodies))
            )
            per_box = self.env.batch_plan.capacity
            counts = [
                self._counts_in_box(body, ids)
                for body, ids in zip(self.boxes, self.indices, strict=True)
            ]
            clearances_mm = [
                float(self.env.data.xpos[body, 1] - self.station[1]) * 1000.0
                for body in self.boxes[:-1]
            ]
            last_offset = self.env.data.xpos[self.boxes[-1], :2] - self.station
            last_distance_mm = float(np.linalg.norm(last_offset)) * 1000.0
            expected_source = len(self.env._cookie_bodies) - per_box * len(self.boxes)
            good = (
                all(len(ids) == per_box for ids in self.indices)
                and all(contained == per_box for contained, _ in counts)
                and source == expected_source
                and all(clearance >= self.PUSHED_CLEAR_M * 1000.0 for clearance in clearances_mm)
                and last_distance_mm < self.ARRIVED_WITHIN_M * 1000.0
            )
            self._settled_steps = self._settled_steps + 1 if good else 0
            self._verify_steps += 1
            if self._settled_steps >= 20:
                self.stage = "DONE"
                self.done = True
            elif self._verify_steps >= self.VERIFY_SETTLE_STEPS:
                # The last check used to fail on its *first* bad step, which makes
                # it a race against whatever the previous stage left still moving:
                # a box a tenth of a millimetre outside its tolerance at the moment
                # the fill ended rejected the episode, and the faster motions make
                # that window narrower rather than wider.  A bounded wait keeps the
                # criterion strict without racing the settling.
                self.failed = (
                    f"lane final criteria failed: "
                    f"{self._describe_boxes(counts, clearances_mm, last_distance_mm, per_box)}, "
                    f"source={source} (needs {expected_source})"
                )
            return self.env.last_applied_action.copy()
        self.failed = f"unknown relay stage {self.stage}"
        return self.env.last_applied_action.copy()

    def _describe_boxes(
        self,
        counts: list[tuple[int, int]],
        clearances_mm: list[float],
        last_distance_mm: float,
        per_box: int,
    ) -> str:
        """One clause per box, for a failure that has to say which box was wrong.

        A lane of four boxes reporting only "the final criteria failed" would leave the
        reader to guess which of four positions or eight counts was the problem, and the
        whole reason this expert carries its own counts is that a rejected episode has to
        be diagnosable from its message.
        """

        clauses = []
        for index, (name, (contained, placed)) in enumerate(zip(self.names, counts, strict=True)):
            if index == len(self.boxes) - 1:
                where = (
                    f"{last_distance_mm:.1f} mm from the station (needs under "
                    f"{self.ARRIVED_WITHIN_M * 1000:.1f})"
                )
            else:
                where = (
                    f"{clearances_mm[index]:.1f} mm clear (needs {self.PUSHED_CLEAR_M * 1000:.1f})"
                )
            clauses.append(f"{name}={contained}/{per_box} in the box ({placed} flush) at {where}")
        return ", ".join(clauses)
