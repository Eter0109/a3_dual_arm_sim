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

    #: Jaw closure the push commands while it is in contact.  At 0.0 the jaws are
    #: jammed shut on the box's rear wall: fast (measured 0.60 mm/step, 150 steps for
    #: the 90 mm push) but the box walks sideways 16 to 23 mm and its yaw reaches 2.8
    #: to 9 deg, because a pad that only touches a 6 mm wall has almost nothing to
    #: resist a moment with.  Widening the closure makes the pads pass the wall and
    #: bear on the box's floor plate instead, which is a friction drive: the box then
    #: walks only 3.9 mm with 1.2 deg of yaw, but at 0.17 mm/step -- 3.5x slower, so
    #: the 90 mm push needs about 530 steps against the fast fill's 528.  Neither end
    #: is free, so this is a knob rather than a constant to be sure of; see
    #: `docs/configurable-scenes.md`.
    PUSH_OPENING = 0.0

    #: How far the box may be walked sideways before the push is called a failure,
    #: signed: negative is away from the arm, positive is towards it.  The two sides
    #: are **not** symmetric, and a single 25 mm bound was wrong in both directions --
    #: it failed pushes that walked the box the harmless way while tolerating three
    #: times as much in the direction that actually hurts.
    #:
    #: Measured against the fill's own pre-check (`p8_offset_tolerance.py`), because the
    #: next stage is what the bound has to serve: a box shifted in x shifts every
    #: placement pose with it, so the offset eats the fill's x reach.
    #:
    #:   -x   free out to 45 mm: the fill's worst residual is 0.03 mm there, *better*
    #:        than at 0, because moving the box away from the arm puts its slots
    #:        further inside the workspace
    #:   +x   about 7 mm: 8 mm measures 2.10 mm of residual against the 2.0 mm window
    #:
    #: and the push walks the box in -x, which is why the shipped 12 mm and the damped
    #: 23 mm are both harmless to the fill.
    SIDEWAYS_LIMIT_AWAY_M = 0.045
    SIDEWAYS_LIMIT_TOWARDS_M = 0.007

    #: The box's yaw the push may leave behind, which is the quantity that actually
    #: decides whether the next fill can place: a slot 28 mm out from the box's centre
    #: moves ``28 * sin(yaw)`` in y, so the fill's 2.0 mm window is crossed at about
    #: 4 degrees.  Measured the same way (`p8_yaw_tolerance.py`): 4.0 deg gives
    #: 1.19 mm / 1.66 deg and passes, 5.0 deg gives 1.44 mm / 2.05 deg and is over.
    #:
    #: This is the guard that protects the next stage, and the sideways bound above is
    #: the one that catches a runaway.  Without it the push could leave a box the fill
    #: cannot use and report success, which is exactly what a 6.2 deg yaw did.
    #:
    #: It bounds the yaw the push *added*, not the box's absolute yaw, for the same
    #: reason the carry's rotation guard does: the scene draws the station box's yaw
    #: (up to 4.5 deg here), and an absolute bound would read "started at 4.5 deg" as
    #: "the push turned it 4.5 deg".
    PUSHED_YAW_LIMIT_RAD = math.radians(4.0)

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
        #: The box's yaw when the push started.  The yaw guard below bounds what the
        #: push *added*, not the box's absolute yaw, because the scene draws it.
        self.initial_yaw = self.box_yaw_rad

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
        # The box's own yaw, which the next fill's pre-check reads.  Checked before the
        # sideways bound because it is the one with a measured window behind it: the
        # fill places into the box's frame, so a yawed box moves its own slots.
        turned = self.box_yaw_rad - self.initial_yaw
        if abs(turned) > self.PUSHED_YAW_LIMIT_RAD:
            self.failed = (
                f"box yawed {math.degrees(turned):.2f} deg during right-arm push "
                f"(from {math.degrees(self.initial_yaw):.2f}), past the "
                f"{math.degrees(self.PUSHED_YAW_LIMIT_RAD):.1f} deg the next fill can "
                f"place into"
            )
            return self.env.last_applied_action.copy()
        drift = float(self.box_position[0] - self.initial_position[0])
        limit = self.SIDEWAYS_LIMIT_TOWARDS_M if drift > 0.0 else self.SIDEWAYS_LIMIT_AWAY_M
        if abs(drift) > limit:
            self.failed = (
                f"box drifted {drift * 1e3:+.2f} mm sideways during right-arm push, "
                f"past the {limit * 1e3:.1f} mm the next fill can absorb in that "
                f"direction"
            )
            return self.env.last_applied_action.copy()
        up = self.data.xmat[self.box_id].reshape(3, 3)[2, 2]
        if abs(float(up)) < math.cos(math.radians(12)):
            self.failed = "box tilted during right-arm push"
            return self.env.last_applied_action.copy()
        phase_timeout = {
            "APPROACH": 220,
            "DESCEND": 220,
            # Derived from the travel and the commanded speed rather than fixed at
            # 420 steps, which is only right for the shipped speed: a push whose pads
            # bear on the box's floor plate advances at 0.17 mm/step and needs about
            # 530, so a flat budget rejects it as a timeout rather than as a failure.
            # Same reasoning as `BoxSupportController._travel_budget`.
            "PUSH": max(
                240,
                round(
                    (self.destination_y - (self.initial_position[1] - 0.034) + 0.020)
                    / self.PUSH_SPEED_M_PER_STEP
                )
                + 120,
            ),
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
                action, reached = self._pose_command(self.start_y, 0.840, self.PUSH_OPENING)
                self.stable = self.stable + 1 if reached else 0
                if self.stable >= 5:
                    self._advance("DESCEND")
                return action
            if self.phase == "DESCEND":
                action, reached = self._pose_command(self.start_y, 0.755, self.PUSH_OPENING)
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
                return self._pose_command(self._push_goal_y, 0.755, self.PUSH_OPENING)[0]
            if self.phase == "RETRACT":
                assert self._retract_y is not None
                action, reached = self._pose_command(self._retract_y, 0.755, self.PUSH_OPENING)
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

    #: Height the box is carried at, above the height the grip was closed on.
    #:
    #: The hypothesis this exists to test: while the box slides on its own floor, the
    #: table's friction is a second constraint on it, and a pinch that does not sit
    #: straight on the wall converts its own torque into a *yaw through that friction*.
    #: Lift it and the grip is the only thing holding the box, so the same torque has
    #: only the box's inertia to argue with.  `box_lift_m` exists for the fill and
    #: defaults to 30 mm there; the shipped relay sets it to zero.
    #:
    #: Zero is the shipped behaviour exactly -- with no lift there is no LOWER phase and
    #: the action stream is unchanged -- so this is a knob that can be measured rather
    #: than a change that has to be justified.
    LIFT_M = 0.0
    #: Slide speed, as the distance the pads' target y advances per step.
    MOVE_SPEED_M_PER_STEP = 0.0008
    #: Finger opening of the pinch that carries the box, as the normalised 0..1 joint
    #: target the action carries.  **0 closes the jaws and 1 opens them**, which is the
    #: same direction as a joint target, and `A3DualArmEnv._apply_controls` is why: it
    #: maps the action to `(1 - opening) * 0.0425`, so the two finger slides travel
    #: *further* as the number falls.
    #:
    #: A comment here once claimed the opposite and the claim was copied into a commit
    #: before it was checked against the geometry, so it is worth stating what the
    #: measurement is.  Driving the finger actuators with a box at the carry's own
    #: anchor (`q6_open_map.py`, `q7_pinch_raw.py`) gives, with the wall 12 mm thick:
    #:
    #:   opening   face gap   bite/side   pad forces
    #:     0.000     6.88 mm    2.56 mm   [6.97, 7.05] N     <- as hard as the jaws go
    #:     0.085     9.28 mm    1.36 mm   [2.36, 1.98] N
    #:     0.100     9.83 mm    1.09 mm   [1.41, 1.49] N     <- the shipped value
    #:     0.130    11.32 mm    0.34 mm   [0.40, 0.44] N     <- below CLOSE's 0.3 N test
    #:     0.160    13.58 mm   no contact  [0.25, 0.05] N
    #:
    #: The old formula `gap = 0.01 + 85 * opening` is right and still holds for the
    #: commanded gap; what it does not describe is the *achieved* one, which for a tight
    #: enough command is set by the wall rather than by the command.
    #:
    #: The shipped 0.10 asks for a 9.83 mm gap against a 12 mm wall: about a millimetre
    #: of bite and 1.4 N.  **It cannot be derived from the wall**, which is worth stating
    #: because that is what it looks like it should be, and two measurements say no:
    #:
    #:   - the servo is compliant, so the command does not set the bite.  Asking for a
    #:     0.01 mm gap achieves 6.88 mm (2.56 mm of bite) and asking for 7.24 mm achieves
    #:     9.28 mm (1.36 mm) -- only about a third of the command becomes bite, so a bite
    #:     target cannot be commanded at all;
    #:   - the grip strength is a *trade*, and the fill needs the other side of it.
    #:     Measured in the pinned test's own scene (`q10_test_pose.py`), where the box's
    #:     delivered pose is what the next fill's pre-check reads:
    #:
    #:       opening   delivered yaw   arrival error   fill pre-check
    #:         0.000      4.56 deg        1.12 mm      REFUSED (2.06 deg)
    #:         0.050      3.92 deg        1.54 mm      accepted
    #:         0.060     13.22 deg       38.03 mm      slipped out of the grip
    #:         0.070      1.64 deg        1.98 mm      accepted
    #:         0.100      1.15 deg        3.14 mm      accepted
    #:
    #:     A hard pinch tracks the pads better -- the box arrives 1.12 mm from the
    #:     station against 3.14 mm -- and it also *transmits* the pads' orientation
    #:     error, which at the station is 4.279 deg, so the box leaves 4.56 deg yawed
    #:     and the fill refuses it.  A light pinch lets the box slip relative to the
    #:     pads, so it keeps its own orientation and the yaw stays at 1.15 deg.  The
    #:     fill's pre-check needs both, and the shipped value sits on the yaw-favouring
    #:     side, which is the side that matters.
    #:
    #: The 0.060 row is why this is not tuned finer: the response is not monotone, and
    #: just below the shipped value the box slips out of the grip entirely.  The shipped
    #: 0.10 is the furthest from that cliff.
    GRASP_OPENING: float | None = 0.10
    FORCE_LIMIT_N = 80.0

    #: The measured map from the normalised opening to the *commanded* face gap, in
    #: metres: ``gap = 0.00001 + 0.085 * opening``.
    OPENING_GAP_BASE_M = 0.00001
    OPENING_GAP_RANGE_M = 0.085
    #: Least bite that still means the pads are pressing into the wall rather than
    #: resting on its face, used by the geometric contact tests below.
    GRASP_MIN_BITE_M = 0.0003

    #: The orientation the grip is commanded to hold, **derived rather than copied**.
    #:
    #: This used to be whatever the arm happened to be doing when the carry was
    #: constructed.  Worth stating that the derived constant here is the *same*
    #: orientation that copied value happens to hold at the deployment home -- the two
    #: families differ by a 180 deg yaw about the vertical, which changes nothing about
    #: the pose, so this half of the change is behaviour-neutral and the interesting
    #: variable is the pitch below.  What it buys is that the grip no longer depends on
    #: where the arm happened to be, so a scene that moves the home pose moves the grip
    #: with it.
    #:
    #: A pinch makes the box a slave to the pads' *achieved* orientation, and the
    #: achieved orientation of the copied pose is not the commanded one: at the station
    #: it misses by 4.279 deg with ``R_WRIST_P`` and ``R_SHOULDER_R`` on their bounds.
    #:
    #: **That error is real but it is not what the delivered yaw consists of**, which
    #: took a sweep to establish and is worth flagging before anyone optimises it again.
    #: Decomposed at the station (`q9_error_components.py`), and compared against the
    #: yaw the box is actually handed in a rollout (`q8_bench.py`):
    #:
    #:   pitch   err deg   yaw part   pitch part   delivered |yaw| mean
    #:     0.0     4.279     -3.916      -1.700        6.03 deg over 4 seeds
    #:     1.0     3.434     -3.147      -1.353        6.65 deg over 6 seeds
    #:     2.0     2.586     -2.374      -1.009        6.59 deg over 4 seeds
    #:     6.0     0.000      0.000       0.000        jams at 31-56 mm
    #:
    #: The orientation error can be driven to zero and the yaw the box leaves with does
    #: not move, so the box is not tracking the pads' orientation; what it tracks is the
    #: contact, and the pads' orientation is one input to that rather than the answer.
    #:
    #: Three axes were swept for a family whose warm pose *is* achievable, all at the
    #: carry's own anchor and all converged (`q2_reachable_grip.py`, `q3_...`):
    #:
    #:   - **yaw** about the world's vertical axis: reachable from 9 deg on (0.779 deg
    #:     residual), but it turns the pad faces out of the wall's plane and the
    #:     readiness is a lie -- rolled out, the box is handed 9.28 deg and 11.60 deg at
    #:     9 and 15 deg of yaw against 6.69 deg at zero, and two runs in three fail;
    #:   - **roll** about the pads' own face normal: exhausted at 3.84 deg, and it is
    #:     the one axis the grip is indifferent to, so there was nothing to win;
    #:   - **pitch** about the world's x axis: the station comes back exactly, and it is
    #:     the axis ``right_grasp_pitch_deg`` already exists for -- the fill derives its
    #:     orientation the same way, a world-x rotation on the left.
    #:
    #: Along the carry's own path, worst residual over 17 points from the rear wall to
    #: the station:
    #:
    #:   pitch   0.0    3.0    4.0    5.0    6.0    8.0   10.0   12.0   15.0
    #:   angle   4.279  1.738  0.889  0.040  0.000  0.000  0.001  0.000  0.000  (deg)
    #:   pinned  WRIST_P      WRIST_P       WRIST_P   --     --     --     --
    #:
    #: So 6 deg is where the wrist comes off its bound and the orientation becomes
    #: exact, and everything above it holds with *slack*.  **It is still not usable**,
    #: which is the whole reason this knob ships at zero: reachability is a joint-space
    #: fact, and the grip is a contact fact.  Rolled out with the derived pinch
    #: (`q8_bench.py`), the pitched grip jams the slide instead of the wrist:
    #:
    #:   pitch    seeds   finished   travel        delivered |yaw|
    #:     0.0       4        4/4     159-196 mm    6.03 deg mean
    #:     4.0       1        0/1      63 mm       jams, "rotated during carry"
    #:     6.0       4        0/4      31-56 mm    10.25 deg mean
    #:
    #: The pitched pads bite the wall on one z-edge ahead of the other, and a hard
    #: pinch turns that asymmetry into a couple the box cannot resist.  So the pitch
    #: buys an exact orientation and loses the grip, exactly as yaw did.
    GRASP_ROTATION = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    #: Pitch applied to :attr:`GRASP_ROTATION`, about the world's x axis, left-handed
    #: on the rotation as ``right_grasp_pitch_deg`` is on the fill's.  **Zero is the
    #: measured default**, and the docstring above says why the reachable values are not
    #: the right ones.  The band 1-3 deg is untested and is the one thing that might still
    #: shave the 4.98 deg the pads hold at zero without jamming the slide.
    GRASP_PITCH_RAD = 0.0

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
        self.quat = self.grasp_quat()
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
        #: The wall the pads must straddle, and the opening that straddles it.  Both
        #: are scene properties, so they are read here rather than assumed: the config
        #: states a wall *half*-thickness (`_add_open_bin` puts it on the size), which is
        #: why the shipped 6 mm is a 12 mm wall.
        self.wall_thickness_m = 2.0 * float(env.config.cookie_transfer.bin_wall_thickness_m)
        # The command is the shipped one; see :attr:`GRASP_OPENING` for why it is not
        # derived from the wall.  What *is* derived is the test for whether the grip took:
        # the pads cannot be closer together than the wall is thick unless they are
        # pressing into it, so :meth:`pinching_wall` reads the geometry rather than the
        # servo's force constant.
        self.opening = (
            float(self.GRASP_OPENING) if self.GRASP_OPENING is not None else 0.0
        )

    def opening_for_gap(self, gap_m: float) -> float:
        """The normalised opening whose *commanded* face gap is ``gap_m``.

        Inverts the measured map, so a reader can see what a command asks for.  It is
        deliberately not used to choose the grip: see :attr:`GRASP_OPENING` for why the
        command is closed instead.
        """

        return (gap_m - self.OPENING_GAP_BASE_M) / self.OPENING_GAP_RANGE_M

    def commanded_gap_m(self) -> float:
        """The face gap the current opening *asks* for, from the measured map."""

        return self.OPENING_GAP_BASE_M + self.OPENING_GAP_RANGE_M * self.opening

    def pad_gap_m(self) -> float:
        """The gap between the two pads' wall-facing faces, as measured.

        Projecting the pads' centres on the axis they close along and subtracting
        their own half-thicknesses, so this is geometry rather than the commanded
        number -- which is the point: the commanded number is what the servo was told,
        while this is where the jaws actually ended up.
        """

        axis = self.data.site_xmat[self.helper.site].reshape(3, 3)[:, 0]
        centres = np.array([self.data.geom_xpos[g] for g in self.helper.fingers])
        half = sum(
            float(self.env.model.geom_size[g][1]) for g in self.helper.fingers
        ) / 2.0
        return abs(float((centres[1] - centres[0]) @ axis)) - 2.0 * half

    def relative_yaw_rad(self) -> float:
        """How far the box is yawed relative to the direction the pads close on."""

        box_axis = self.data.xmat[self.box_id].reshape(3, 3)[:, 1]
        grip_axis = self.data.site_xmat[self.helper.site].reshape(3, 3)[:, 0]
        return math.acos(min(1.0, abs(float(box_axis @ grip_axis))))

    def wall_extent_along_grip_m(self) -> float:
        """The wall's thickness as seen along the direction the pads close on.

        The pads close along the site's x axis and the box can be yawed relative to the
        tool -- the carry's own guard allows 10 deg -- so the slab they straddle is
        thicker along that axis than the wall is: a 12 mm wall at 10 deg of relative yaw
        presents 12.19 mm.  Comparing the pad gap against the unrotated thickness reads
        that as the grip having opened, and it did exactly that in a relay run: the
        guard fired at a pad gap of 12.19 mm against a 12.00 mm wall while the inner pad
        was reading 7.34 N, which is a box being held.
        """

        box_axis = self.data.xmat[self.box_id].reshape(3, 3)[:, 1]
        grip_axis = self.data.site_xmat[self.helper.site].reshape(3, 3)[:, 0]
        cosine = abs(float(box_axis @ grip_axis))
        # A relative yaw past 60 deg is not a grip at all, so the clamp only keeps the
        # division finite; it is not a tolerance.
        return self.wall_thickness_m / max(cosine, 0.5)

    def grip_margins_m(self) -> tuple[float, float]:
        """How far each pad's face is inside the wall face it should be behind.

        Positive means inside.  Returned separately rather than as a boolean so a
        failure can say *which* pad came off, which is the difference between a box
        that slipped along the wall's normal and one that was never gripped.
        """

        axis = self.data.site_xmat[self.helper.site].reshape(3, 3)[:, 0]
        centres = np.array([self.data.geom_xpos[g] for g in self.helper.fingers])
        thickness = sum(
            float(self.env.model.geom_size[g][1]) for g in self.helper.fingers
        ) / 2.0
        pad_centre = float(centres.mean(axis=0) @ axis)
        pad_half = abs(float((centres[1] - centres[0]) @ axis)) / 2.0
        inner_face = pad_centre - pad_half + thickness
        outer_face = pad_centre + pad_half - thickness

        box_axis = self.data.xmat[self.box_id].reshape(3, 3)[:, 1]
        half_y = float(self.env.config.cookie_transfer.target_bin_half_size_m[1])
        wall_centre = float((self.data.xpos[self.box_id] - box_axis * half_y) @ axis)
        wall_half = self.wall_extent_along_grip_m() / 2.0
        return (
            inner_face - (wall_centre - wall_half),
            (wall_centre + wall_half) - outer_face,
        )

    def pinching_wall(self) -> bool:
        """Whether the jaws are inside the wall's thickness.

        A geometric test for "the grip is holding", in place of reading the pads'
        forces.  The two pads cannot be closer together than the wall is thick unless
        they are pressing into it, so this is a fact about the grip rather than about
        the servo's force constant -- and unlike the force test it does not depend on
        the load being shared evenly, which measured it is not.

        **It is deliberately not the stricter "both pads are inside the wall" test**,
        which is the more physical question and the wrong one.  Measured over 8 seeds
        (`q13`), that version fails every run after 46-160 mm of travel while the pads
        read a 9.2-9.3 mm gap against a 12 mm wall and the box arrives within 1.3 deg
        of its start: a box being slid lags the pads along the wall's normal by about
        the bite depth, so one pad is always just clear of its wall face and the
        stricter test reads a working grip as a lost one.  The gap is the quantity that
        stays meaningful while the box is moving.

        The wall's thickness is projected onto the axis the pads close along, because
        the box can be yawed relative to the tool -- the carry's own guard allows 10 deg
        -- and a 12 mm wall at 10 deg presents 12.19 mm along that axis.
        """

        return self.pad_gap_m() < self.wall_extent_along_grip_m() - 2.0 * self.GRASP_MIN_BITE_M

    def grip_lost(self) -> bool:
        """Whether the jaws have come off the wall they were pinching.

        The complement of :meth:`pinching_wall`, and the replacement for the old
        ``max(forces) < 0.08`` test, which sat right on one pad's own resting level:
        the outer pad measures 0.04-0.2 N against the inner pad's 1.1-2.6 N, so an arm
        that tracks its command more precisely -- actuator damping, which the fast
        profile needs -- held that pad low without the box ever moving.
        """

        return not self.pinching_wall()

    @property
    def box_position(self) -> np.ndarray:
        return self.data.xpos[self.box_id].copy()

    @property
    def box_yaw_rad(self) -> float:
        rotation = self.data.xmat[self.box_id].reshape(3, 3)
        return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

    @classmethod
    def grasp_rotation(cls) -> np.ndarray:
        """The commanded grip rotation: the derived constant, pitched.

        A class method rather than arithmetic in ``__init__`` because the bench that
        measures the carry has to build the pose the controller will build, and a
        second copy of the arithmetic would be a second thing to keep in step.
        """

        base = np.array(cls.GRASP_ROTATION, dtype=float)
        angle = float(cls.GRASP_PITCH_RAD)
        if angle == 0.0:
            return base
        cosine, sine = math.cos(angle), math.sin(angle)
        about_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
        )
        return about_x @ base

    @classmethod
    def grasp_quat(cls) -> np.ndarray:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, cls.grasp_rotation().ravel())
        return quat

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
            "LOWER": 240,
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
                action, _ = self._pose_command(self.anchor, self.opening)
                # Geometric rather than "both pads read force": see `pinching_wall`.
                reached = self.pinching_wall()
            elif self.phase == "MOVE":
                if self.box_position[1] >= self.destination_y - self.RELEASE_WITHIN_M:
                    self._advance("LOWER" if self.LIFT_M else "RELEASE")
                    return self.env.last_applied_action.copy()
                # "Grip lost" is now a geometric question -- have the jaws come off the
                # wall -- rather than a force threshold.  The old test read a single
                # pad's resting level (measured: the outer pad hovers between 0.04 and
                # 0.2 N against the inner pad's 1.1-2.6 N) and so could not tell a box
                # that had been let go from one an arm was simply holding more
                # precisely.
                if self.grip_lost():
                    self.stable += 1
                    if self.stable > 8:
                        inner_margin, outer_margin = self.grip_margins_m()
                        self.failed = (
                            f"empty box grip lost during carry: pad gap "
                            f"{self.pad_gap_m() * 1e3:.2f} mm against a "
                            f"{self.wall_extent_along_grip_m() * 1e3:.2f} mm wall "
                            f"(thickness {self.wall_thickness_m * 1e3:.2f} mm at "
                            f"{math.degrees(self.relative_yaw_rad()):.1f} deg of relative "
                            f"yaw); inner pad {inner_margin * 1e3:+.2f} mm and outer pad "
                            f"{outer_margin * 1e3:+.2f} mm inside their wall faces "
                            f"(positive is inside); forces "
                            f"{[round(float(v), 3) for v in forces]} N"
                        )
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
                target[2] += self.LIFT_M
                self._target_q = None
                return self._pose_command(target, self.opening)[0]
            elif self.phase == "LOWER":
                # Put the box back down on the height the grip was closed on, so the
                # release happens where the fill expects the box to be standing.
                action, reached = self._pose_command(self.anchor, self.opening)
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
                    "LOWER": "RELEASE",
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
