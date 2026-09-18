"""Test-only recovery: stand a dominoed source row back up after a batch leaves it.

A batch of five is grasped by driving the closed jaws into the two outermost gaps
of five touching Cookies, then squeezing.  That squeeze pushes the neighbours that
were left behind: at seed 0, after the first batch of five leaves column 0, the
fifteen Cookies still in the bin lean *backward* by 21.5 .. 44.2 deg in a domino
chain.  They are stable (the friction cone holds ``tan 44.2 < 0.8``) but they are
tilted past the batch expert's upright test, so the column can never supply
another batch while they stay that way.

This expert adds one recovery stroke after each batch that leaves damage behind.
It is a *thin blade* move: the two pads closed together are a 20.0 x 6.35 x 33.5
mm plate, and the space a batch of five just vacated is wide open, so the tool can
descend behind the back-most leaning Cookie and creep forward into it.

Which surface the blade meets decides whether the Cookie straightens or launches,
and the difference is 4 mm:

    push the top back corner (45 deg chamfer)  -> reaction is forward *and down*,
                                                  the Cookie pivots upright
    push 4 mm lower (flat back face)           -> reaction is forward *and up*,
                                                  the Cookie lifts off the floor

So the blade is aimed at the core's top corner -- ``cookie_half_size_m[2]`` minus
``cookie_edge_bevel_m``, which is exactly where the compound collision core ends
and its chamfer cap begins -- and driven at a regulated 2.5 N.  A stiff position
servo would report nothing until the pad is already buried, and the ~7 N that
takes lifts the blade onto the top face instead of pushing the Cookie over.

Measured on the post-batch state (seed 0): one stroke rights 13 of the 15 leaning
Cookies.  The two that do not come back are the front-most pair, which have the
whole vacated column in front of them and tip past vertical once the chain
unfolds; they end up flat, and no later stroke can reach them because a flat
Cookie has no back face left to push.  That is why the recovery reports what it
recovered instead of retrying, and why it runs as its own expert rather than as
another batch phase: ``A3CookieBatchExpert`` keeps its exact behaviour, this class
only adds a stroke in front of its ``SELECT_COOKIE``.
"""

from __future__ import annotations

from enum import Enum, auto

import mujoco
import numpy as np

from .batch_expert import A3CookieBatchExpert
from .expert import CookiePhase


class RecoveryStage(Enum):
    IDLE = auto()
    TRAVEL = auto()
    DESCEND = auto()
    PLOUGH = auto()
    RETRACT = auto()


class A3CookieRecoveryExpert(A3CookieBatchExpert):
    """Batch expert plus one plough stroke over the Cookies a batch left leaning.

    Only ``SELECT_COOKIE`` is intercepted, and only when the source bin actually
    holds a leaning Cookie, so a scene the batch expert already handles is never
    touched: with ``--recover`` off, or on an undamaged column, the action stream
    is identical to ``A3CookieBatchExpert``.
    """

    #: A Cookie counts as leaning once it tilts past this far from vertical.
    UPRIGHT_DEG = 10.0
    #: Abort the stroke if the pads push this hard; well past the 2.5 N that
    #: rights a Cookie, so a normal stroke never trips it.
    FORCE_LIMIT_N = 6.0
    #: Regulated push force along -y.  2 N rights the row; the margin below the
    #: 6 N guard leaves room for the chain to gather.
    PLOUGH_FORCE_N = 2.5
    #: How far the blade creeps into the row.  The emptying batch leaves ~44 mm.
    PLOUGH_TRAVEL_M = 0.055
    #: Control steps the whole row must stay upright before the stroke is called
    #: done.  One step is not enough: a Cookie crossing vertical on its way down
    #: reads as upright for a step, which is how the first prototype stopped with
    #: the row still falling.
    SETTLE_STEPS = 25
    #: Aim this far above the core's top corner, and start this far behind it.
    BLADE_DROP_M = -0.0015
    BLADE_BACK_OFFSET_M = 0.0025
    #: Clearance height for the travel move over the bin.
    TRAVEL_CLEARANCE_M = 0.120
    #: Rate limits, in m per control step, matching the batch expert's descent.
    TRAVEL_SPEED_M = 0.0015
    DESCEND_SPEED_M = 0.0007
    PLOUGH_SPEED_M = 0.0005
    STAGE_LIMITS = {"TRAVEL": 900, "DESCEND": 700, "PLOUGH": 900, "RETRACT": 500}

    def reset(self, context=None):
        super().reset(context)
        self._stage = RecoveryStage.IDLE
        self._stage_steps = 0
        self._plough_goal = None
        self._plough_advanced = 0.0
        self._plough_stall = 0
        self._upright_streak = 0
        self._overhead = None
        self._down = None
        self._retract_site = None
        self._servo_reached = False
        self._aborted: str | None = None
        self._recovery_target_index = None
        self._recovery_leaners: list[int] = []
        self._tilt_before: dict[int, float] = {}
        # Which batch index a stroke was last started for.  One stroke per batch
        # is deliberate: the Cookies a stroke cannot reach are flat, and a flat
        # Cookie has no back face for the blade to push.
        self._recovered_for_batch = -1
        self.recovery_reports: list[dict] = []
        # Blade = both pads closed.  Read the sizes instead of trusting the
        # comment: the pads are the only colliding part of the fingers.
        pad = self.env._left_finger_geoms[0]
        self._blade_half_y = 2.0 * float(self.model.geom_size[pad][1])
        self._blade_half_z = float(self.model.geom_size[pad][2])
        self._source_positions = np.asarray(self.env.SOURCE_POSITIONS)
        # The compound collision Cookie is a box core with chamfered caps, and
        # the push has to land on the corner where the core ends.
        self._core_half_z = float(self.env.COOKIE_HALF_SIZE[2]) - float(
            self.env.config.cookie_transfer.cookie_edge_bevel_m
        )

    # ------------------------------------------------------------------ recovery
    def _tilt_deg(self, index: int) -> float:
        rotation = self.data.xmat[self.env._cookie_bodies[index]].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))

    def leaning_cookies(self) -> list[int]:
        """Source-bin Cookies that are neither upright nor already finished."""
        return [
            index
            for index in range(len(self.env.SOURCE_POSITIONS))
            if self.env.privileged_cookie_in_source(index)
            and self._tilt_deg(index) > self.UPRIGHT_DEG
        ]

    def _recovery_target(self) -> int | None:
        """The back-most leaning Cookie, or None when there is nothing to do.

        The blade descends behind it, so the back-most one is the only reachable
        entry point into the chain: every other leaner has a neighbour behind it.
        """
        if self._recovered_for_batch == self.batch_index:
            return None
        leaning = self.leaning_cookies()
        if not leaning:
            return None
        return max(leaning, key=lambda index: self.env.privileged_cookie_position(index)[1])

    def _blade_force(self) -> float:
        """Normal force the two pads put into whatever they touch, in newtons."""
        total = 0.0
        wrench = np.zeros(6)
        pads = set(self.env._left_finger_geoms)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if int(contact.geom1) in pads or int(contact.geom2) in pads:
                mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
                total += abs(float(wrench[0]))
        return total

    def _blade_targets(self, target: int) -> tuple[np.ndarray, np.ndarray]:
        """Overhead and in-place site poses for the stroke against ``target``."""
        position = self.env.privileged_cookie_position(target)
        rotation = self.data.xmat[self.env._cookie_bodies[target]].reshape(3, 3)
        corner = position + rotation @ np.array([0.0, self.env.COOKIE_HALF_SIZE[1], self._core_half_z])
        blade_bottom = corner[2] + self.BLADE_DROP_M
        centre_z = blade_bottom + self._blade_half_z
        # The blade sits just behind the corner and drops onto it, so the first
        # thing it meets is the chamfer, not the flat face underneath.
        centre_y = corner[1] + self.BLADE_BACK_OFFSET_M + self._blade_half_y
        # The stroke runs along the Cookie's own column, which the source layout
        # gives directly; neighbouring columns are 50.4 mm away and stay clear.
        column_x = float(self._source_positions[target, 0])
        pad_mid = [column_x, centre_y, centre_z]
        site = np.asarray(pad_mid) - np.asarray(self._canonical) @ np.asarray(self._pad_offset)
        overhead = site + np.array([0.0, 0.0, self.TRAVEL_CLEARANCE_M])
        return overhead, site

    def _begin_recovery(self, target: int) -> None:
        self._overhead, self._down = self._blade_targets(target)
        self._recovery_target_index = target
        self._recovery_leaners = self.leaning_cookies()
        self._tilt_before = {index: self._tilt_deg(index) for index in self._recovery_leaners}
        self._recovered_for_batch = self.batch_index
        self._aborted = None
        self._servo_pos = None
        self._stage_steps = 0
        self._stage = RecoveryStage.TRAVEL

    @property
    def recovery_status(self) -> str:
        return (
            f"RECOVER {self._stage.name} step={self._stage_steps} "
            f"target={self._recovery_target_index} "
            f"leaners={len(self.leaning_cookies())}"
        )

    def _advance_stage(self, stage: RecoveryStage) -> None:
        self._stage = stage
        self._stage_steps = 0
        self._servo_pos = None
        if stage is RecoveryStage.PLOUGH:
            # The creep drives its own virtual target; hand it the measured pose
            # so it starts exactly where the blade landed.
            self._servo_pos = self.data.site_xpos[self._l_site].copy()
            self._plough_goal = self._servo_pos.copy()
            self._plough_advanced = 0.0
            self._plough_stall = 0
            self._upright_streak = 0
        if stage is RecoveryStage.RETRACT:
            self._retract_site = self.data.site_xpos[self._l_site].copy() + [0.0, 0.0, 0.100]

    def _recovery_step(self) -> np.ndarray:
        self._stage_steps += 1
        if self._stage_steps > self.STAGE_LIMITS[self._stage.name]:
            return self._finish_recovery(f"{self._stage.name} stage did not finish")

        if self._stage is RecoveryStage.PLOUGH:
            return self._plough_step()

        goal = {
            RecoveryStage.TRAVEL: self._overhead,
            RecoveryStage.DESCEND: self._down,
            RecoveryStage.RETRACT: self._retract_site,
        }[self._stage]
        speed = (
            self.DESCEND_SPEED_M
            if self._stage is RecoveryStage.DESCEND
            else self.TRAVEL_SPEED_M
        )
        action = self._recovery_servo(goal, speed)
        if action is None:
            return self._finish_recovery(self._aborted)
        force = self._blade_force()
        if force > self.FORCE_LIMIT_N:
            return self._finish_recovery(f"force guard tripped at {force:.2f} N")
        if self._servo_reached:
            if self._stage is RecoveryStage.TRAVEL:
                self._advance_stage(RecoveryStage.DESCEND)
            elif self._stage is RecoveryStage.DESCEND:
                self._advance_stage(RecoveryStage.PLOUGH)
            else:                              # RETRACT is done when it is back up
                return self._finish_recovery(None)
        return action

    def _recovery_servo(self, goal: np.ndarray, speed: float) -> np.ndarray | None:
        """``_servo`` with the IK failure kept inside the stroke.

        The batch expert lets a ``RuntimeError`` from the solver end the episode.
        A recovery stroke is an add-on, so a pose the solver refuses must abort
        the stroke and leave the source bin to the batch expert, not kill a run
        that would have succeeded without it.
        """
        try:
            action, self._servo_reached = self._servo(goal, self._canonical, 0.0, speed=speed)
        except RuntimeError as exc:
            self._aborted = f"solver refused the stroke pose: {exc}"
            return None
        return action

    def _plough_step(self) -> np.ndarray:
        """Creep -y under a measured force limit until the row stands up."""
        force = self._blade_force()
        if force <= self.PLOUGH_FORCE_N and self._plough_advanced < self.PLOUGH_TRAVEL_M:
            self._plough_goal[1] -= self.PLOUGH_SPEED_M
            self._plough_advanced += self.PLOUGH_SPEED_M
            self._plough_stall = 0
        else:
            self._plough_stall += 1
        action = self._recovery_servo(self._plough_goal, 0.0)
        if action is None:
            return self._finish_recovery(self._aborted)
        self._plough_goal = self._servo_pos.copy()
        self._upright_streak = (
            self._upright_streak + 1 if not self.leaning_cookies() else 0
        )
        if self._upright_streak >= self.SETTLE_STEPS:
            self._advance_stage(RecoveryStage.RETRACT)
        elif self._plough_advanced >= self.PLOUGH_TRAVEL_M or self._plough_stall > 80:
            self._advance_stage(RecoveryStage.RETRACT)
        return action

    def _finish_recovery(self, aborted: str | None) -> np.ndarray:
        stood = [
            index
            for index in self._recovery_leaners
            if self._tilt_deg(index) <= self.UPRIGHT_DEG
        ]
        self.recovery_reports.append(
            {
                "target": self._recovery_target_index,
                "leaning_before": len(self._recovery_leaners),
                "stood_up": len(stood),
                # Before and after for the ones that did not come back: the
                # front-most pair of a long chain can be pushed *past* vertical
                # by the unfold, so a stroke may leave a Cookie worse than it
                # found it and the report has to say so.
                "still_leaning": {
                    str(index): {
                        "before_deg": round(self._tilt_before[index], 1),
                        "after_deg": round(self._tilt_deg(index), 1),
                    }
                    for index in self._recovery_leaners
                    if index not in stood
                },
                "plough_advanced_m": round(self._plough_advanced, 5),
                "aborted": aborted,
            }
        )
        self._stage = RecoveryStage.IDLE
        self._stage_steps = 0
        self._servo_pos = None
        self._recovery_target_index = None
        return self.env.last_applied_action.copy()

    # ------------------------------------------------------------------ hook
    def _act_step(self, observation=None, task=""):
        if not self.finished and self._stage is RecoveryStage.IDLE:
            if self.phase is CookiePhase.SELECT_COOKIE:
                target = self._recovery_target()
                if target is not None:
                    self._begin_recovery(target)
        if self._stage is not RecoveryStage.IDLE:
            return self._recovery_step()
        return super()._act_step(observation, task)
