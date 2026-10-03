"""One Cookie, grasped on its own, with the target box left on the table.

Why this exists
---------------
:class:`~a3_dual_arm_sim.expert.A3CookieTransferExpert` is *already* a complete single-Cookie
state machine -- ``SELECT_COOKIE``, ``APPROACH``, ``ALIGN``, ``DESCEND``, ``CLOSE``,
``VERIFY_GRASP``, ``LIFT``, ``VERIFY_LIFT``, ``MOVE_TO_SLOT``, ``DESCEND_TO_PLACE``, ``OPEN``,
``VERIFY_RELEASE``, ``RETRACT`` -- and it is the only one in the repository.  What it also is,
is a **held-box** workflow: it opens in ``SUPPORT_BOX``, where the right arm picks the target
box up and carries it, and ``act()`` re-checks ``box_support.contact_forces()`` every step so
that a dropped box ends the episode.

That workflow no longer completes.  Measured on the one-Cookie scene:

    step   0  phase=SUPPORT_BOX
    step 200  phase=SUPPORT_BOX
    failed: right box approach timeout; contacts=[0.0, 0.0]

which is the same failure three pre-existing tests record.  It is about the right arm reaching
the tabletop box, not about the Cookies, so it would have happened with eighty pieces too.

This subclass removes exactly that assumption and nothing else.  The box stays where the scene
put it, the right arm holds its start pose, and the base class's own state machine runs
unchanged -- the phases, the IK, the placement geometry and the retry logic are all reused
rather than re-derived.  That is the whole point: a scene whose scripted expert is a *copy*
would only ever demonstrate that the copy works.

Nor is any new placement geometry needed.  With the box static, the target pose *is* the box's
pose at reset, which ``_initial_target_position``/``_initial_target_rotation`` already hold,
and the two orientations ``SUPPORT_BOX`` would have computed are derived here from the same
quaternions by the same two lines.

The batch experts arrived at the same arrangement from the other direction; see the comment in
:meth:`~a3_dual_arm_sim.batch_expert.A3CookieBatchExpert.reset` about two vertical 2F85
housings not fitting in the box opening.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .expert import A3CookieTransferExpert, CookiePhase


class _BoxOnTable:
    """Stand-in for ``BoxSupportController`` when nothing is being carried.

    Two members are all the base class reads outside ``SUPPORT_BOX``: ``failed``, and
    ``contact_forces()``, which ``act()`` compares against 0.1 to notice a dropped box.  With
    the box never picked up there is nothing to drop, so this reports support.  Returning
    zeros instead would fail every episode on its first step with "right gripper lost box
    support" -- a message about a box this workflow is defined by not holding.
    """

    failed: str | None = None
    ready: bool = False

    def contact_forces(self) -> np.ndarray:
        return np.ones(2, dtype=np.float64)

    def act(self) -> np.ndarray:
        raise AssertionError(
            "SUPPORT_BOX must not be entered when the box rests on the table; "
            "A3SinglePieceExpert.reset advances past it"
        )


class A3SinglePieceExpert(A3CookieTransferExpert):
    """The base expert's single-Cookie workflow, with the box left on the table.

    Intended for a scene whose source bin holds one Cookie.  Nothing here requires that -- the
    workflow it inherits transfers pieces one at a time and stops after
    ``required_cookies`` -- but the scene it was written for is
    :func:`~a3_dual_arm_sim.collection.cookie_single_piece_scene`, where it is the reference
    that says the scene can be solved at all.
    """

    def reset(self, context=None) -> None:
        super().reset(context)
        # `super().reset` finishes in SUPPORT_BOX holding a live `BoxSupportController`; undo
        # both.  Everything else it established -- the transit pose, the Cookie preference
        # order, the slot order, the initial Cookie and box poses -- is exactly what this
        # workflow needs, which is why this is four lines rather than a new state machine.
        self.box_support = _BoxOnTable()  # type: ignore[assignment]
        # Hold the right arm where the episode started, and leave its gripper closed.  Both
        # are what `_hold_command` emits for joints 8-15 on every step, so the arm is
        # commanded to a fixed pose rather than merely receiving no commands.
        self.q_r_hold = self.env.last_applied_action[8:15].copy()
        self.right_gripper = 1.0
        # SUPPORT_BOX computes these once the box is up.  The box never moves here, so the
        # pose `super().reset` captured is the one to derive them from -- the same two lines
        # that SUPPORT_BOX runs, with the same inputs.
        canonical = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(canonical, self._target_quat_canonical)
        mujoco.mju_mat2Quat(
            self._target_quat_tilted,
            (self._initial_target_rotation @ canonical.reshape(3, 3)).ravel(),
        )
        self._advance(CookiePhase.SELECT_COOKIE)

    def _plan_place_from_grasp(self) -> None:
        """Plan the placement into whichever slot the arm can actually reach.

        The inherited version plans for the single slot ``_slot_order`` named back at selection
        time, and a slot that is out of reach ends the episode.  Measured on this scene with
        the Cookie already in the gripper: slot 8, the far row, put the placement pose 4.53 mm
        past the IK's 4 mm acceptance, and the run failed with "held Cookie placement IK"
        immediately after a clean grasp and lift.

        Failing there is right for a ten-piece fill: the order *is* the fill strategy, reaching
        for an earlier row while a later one is still empty means something has moved, and
        silently reordering would hide it.  With one piece there is no order to preserve and
        no second Cookie to fall back on, so the slot is disposable -- and which of ten
        identical 6.7 mm cells the piece lands in is not part of the task.  Trying them in turn
        keeps the verdict about whether the Cookie can be placed *somewhere* in the box.

        Re-planning is safe because the inherited planner is pure with respect to the robot: it
        reads the live tool, Cookie and box poses and writes only ``_waypoints``,
        ``_placement_height`` and the slot index, none of which the following phases read until
        the plan is complete.
        """

        last_error: RuntimeError | None = None
        for slot in self._slot_order:
            self.target_slot_index = slot
            try:
                super()._plan_place_from_grasp()
            except RuntimeError as exc:
                last_error = exc
                continue
            return
        raise RuntimeError(
            f"no target slot is reachable with the Cookie in hand: {last_error}"
        )
