"""Contact-driven five-Cookie grasp through bevels; privileged, not a VLA policy.

Objects are never attached, teleported, or moved by this controller. Each batch
must form a pad--Cookie contact chain, lift together, and be released into the box.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .arm_servo import CommandedPose
from .batch_plan import BatchPlan
from .batch_profile import PROFILES, BatchProfile
from .contracts import LEFT_JOINTS
from .expert import A3CookieTransferExpert, CookiePhase


class A3CookieBatchExpert(A3CookieTransferExpert):
    MAX_INSERTION_FORCE_N = 8.0

    #: How many Cookies one grasp takes.  A class attribute so that a controller
    #: driven without ``reset`` still answers -- which is how the unit tests exercise
    #: one method at a time, via ``object.__new__`` -- and ``reset`` replaces it with
    #: the scene's ``batch_expert_per_grasp``.  Five is the shipped value, so such a
    #: test measures the shipped configuration rather than a placeholder.
    per_grasp: int = 5

    #: Norm of the left arm's joint velocity below which a move counts as
    #: arrived.  A phase that ends while the arm is still swinging starts the
    #: next one from a moving target, which is what a fast rate would amplify.
    SERVO_SETTLED_QVEL = 0.25
    #: How far short of the retreat height the tool may stop and still count as
    #: clear of the box.  See ``_retreat_clear``.
    RETREAT_HEIGHT_TOLERANCE_M = 0.002
    #: Steps a release stage may wait for the batch to settle before it gives up
    #: and says why.  ``VERIFY_RELEASE`` used to have a fail branch only for "a
    #: Cookie is missing", so a batch that was all present but never settled --
    #: or, before ``_release_settled``, never showed a clean pad -- waited out the
    #: whole phase limit instead.  A bounded wait keeps every state either a pass
    #: or a named failure.
    RELEASE_SETTLE_STEPS = 200
    #: How far the plan may run ahead of the arm before it is pulled back, in the
    #: baseline profile only.  Measuring from the *commanded* pose removes the
    #: need for it (see ``arm_servo``), but the baseline plans from the measured
    #: pose, where a blocked move would otherwise wind the plan into a spring.
    SERVO_ANTIWINDUP_M = 0.008
    #: Arrival tolerances.  Position and orientation are the same in both
    #: profiles; ``_validate_target_workspace`` ties its reachability bounds to
    #: them, since a pose the servo cannot report arrival on is a pose the run
    #: cannot finish.
    SERVO_ARRIVAL_M = 0.0015
    SERVO_ARRIVAL_RAD = 0.045

    def reset(self, context=None):
        super().reset(context)
        self.profile: BatchProfile = PROFILES[self.env.config.batch_expert_profile]
        # How many Cookies a grasp takes, and which slots each grasp fills.  Both
        # come from the scene: the size from the config, the plan from the env's own
        # slot lattice (see `A3CookieTransferEnv.batch_plan`).
        self.per_grasp: int = self.env.config.batch_expert_per_grasp
        self.plan: BatchPlan = self.env.batch_plan
        #: Index of the last batch, which is the one the scene's success test
        #: watches, so it is the one that has to hold for the full success window.
        self._final_batch = len(self.plan.groups) - 1
        self.batch_index = 0
        self.batch_indices = []
        self.batch_reports = []
        self._canonical = np.empty(9)
        mujoco.mju_quat2Mat(self._canonical, self._target_quat_canonical)
        self._canonical = self._canonical.reshape(3, 3)
        site_r = self.data.site_xmat[self._l_site].reshape(3, 3)
        midpoint = self.data.geom_xpos[list(self.env._left_finger_geoms)].mean(axis=0)
        self._pad_offset = site_r.T @ (midpoint - self.data.site_xpos[self._l_site])
        self._stable = 0
        self._jam_count = 0
        self._opening = 1.0
        self._preclose_stable = 0
        self._grip_reference = None
        # The servo plans from the left arm's commanded pose, and the reach test
        # needs the norm of its joint velocity.  Both are built once rather than
        # per step: the servo runs on every step of every phase.
        self._servo_plan = CommandedPose(
            self.model, self.data, self._l_site, self._l_qpos, slice(0, 7)
        )
        self._l_dofs = np.asarray(
            [self.env._dof_ids[name] for name in LEFT_JOINTS], dtype=np.int32
        )
        self.max_pad_force = 0.0
        self.max_contact_penetration_m = 0.0
        # Two vertical 2F85 housings cannot share this small box opening.
        # Keep the free target box resting on the table and the right arm clear.
        # The original single-Cookie expert retains its held-box workflow.
        self.q_r_hold = self.env.last_applied_action[8:15].copy()
        self.right_gripper = 1.0
        self.box_support = None
        self.phase = CookiePhase.SELECT_COOKIE
        self.transition_history = [self.phase]
        self._group_offset = self._pad_offset - self._canonical.T @ np.array([0, 0, 0.010])
        self._validate_target_workspace()

    def _validate_target_workspace(self):
        """Reject unreachable table layouts before starting a physical grasp.

        One pose per *batch*, not per column: a batch is placed into its own run of
        slots, so a plan with two batches in one column has two poses to check and
        they are not the same pose.  For the shipped one-batch-per-column plan this
        is the two columns it has always been, at both clearances -- which is why
        the anchor holds here as well as in the plan itself.
        """

        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        for group in self.plan.groups:
            for clearance in (0.0, self.profile.transport_clearance_m):
                position, rotation = self._place_pose(clearance, group)
                target_q, actual_q, error = np.empty(4), np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(target_q, rotation.ravel())
                try:
                    q = self._solve_l(position, target_q, self.q_transit)
                except RuntimeError as exc:
                    # The IK refuses a solution that misses by more than its own
                    # tolerance, which for this pose means the box is out of reach.
                    # Report it as such rather than letting the raw solver message
                    # escape: this check exists to say "unreachable".
                    raise RuntimeError(
                        f"target column {group.column + 1} "
                        f"(batch {group.index + 1} of {len(self.plan.groups)}) "
                        f"unreachable with vertical grasp: {exc}; "
                        f"reposition the tabletop target box"
                    ) from exc
                work.qpos[self._l_qpos] = q
                mujoco.mj_kinematics(self.model, work)
                mujoco.mju_mat2Quat(actual_q, work.site_xmat[self._l_site])
                mujoco.mju_subQuat(error, target_q, actual_q)
                distance = np.linalg.norm(work.site_xpos[self._l_site] - position)
                # The bounds the shipped expert used, kept as the profile's.
                # Tried tightening them to the placement servo's arrival
                # tolerances (1.5 mm, 0.045 rad) on the reasoning that a pose the
                # servo cannot report arrival on is one the run cannot finish:
                # that refused a two-box layout at 1.53 mm which the baseline
                # accepts and completes, because the *measured* tool position
                # depends on gravity and contact as well as on the IK solution,
                # so an IK miss is not the same as a missed arrival.  Tried
                # widening to ``_solve_l``'s own 4 mm acceptance instead, and a
                # layout that missed by 2.06 mm got past the check only to spend
                # 421 steps in ``DESCEND_TO_PLACE`` without arriving.  So the
                # window is real and these are its measured edges.
                if (
                    distance > self.profile.reach_position_m
                    or np.linalg.norm(error) > self.profile.reach_angle_rad
                ):
                    raise RuntimeError(
                        f"target column {group.column + 1} "
                        f"(batch {group.index + 1} of {len(self.plan.groups)}) "
                        f"unreachable with vertical grasp: "
                        f"position error {distance * 1000:.2f} mm, "
                        f"angle error {np.rad2deg(np.linalg.norm(error)):.2f} deg; "
                        "reposition the tabletop target box"
                    )

    @property
    def status(self):
        return (
            f"phase={self.phase.name} "
            f"batch={min(self.batch_index + 1, len(self.plan.groups))}/{len(self.plan.groups)} "
            f"cookies={self.batch_indices} "
            f"confirmed={len(self.completed_cookie_indices)}/{self.plan.capacity}"
        )

    def _advance(self, phase):
        super()._advance(phase)
        self._stable = 0
        if phase is not CookiePhase.PRE_CLOSE:
            self._preclose_stable = 0
        self._jam_count = 0
        self._servo_pos = None

    def _positions(self):
        return self.env.cookie_positions[self.batch_indices]

    def _contact_chain(self):
        fingers = tuple(self.env._left_finger_geoms)
        # Collapse all collision pieces of a Cookie into one graph node.
        groups = getattr(
            self.env, "_cookie_collision_geoms",
            tuple(frozenset((geom,)) for geom in self.env._cookie_geoms),
        )
        mapping = {
            geom: self.env._cookie_geoms[i]
            for i in self.batch_indices for geom in groups[i]
        }
        cookies = set(mapping.values())
        nodes = cookies | set(fingers)
        graph = {node: set() for node in nodes}
        forces = np.zeros(2)
        total_forces = np.zeros(2)
        for cid in range(self.data.ncon):
            contact = self.data.contact[cid]
            a = mapping.get(int(contact.geom1), int(contact.geom1))
            b = mapping.get(int(contact.geom2), int(contact.geom2))
            chain_contact = a in nodes and b in nodes
            pad_contact = a in fingers or b in fingers
            if not chain_contact and not pad_contact:
                continue
            self.max_contact_penetration_m = max(
                getattr(self, "max_contact_penetration_m", 0.0),
                max(0.0, -getattr(contact, "dist", 0.0)),
            )
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, cid, wrench)
            for j, finger in enumerate(fingers):
                if finger in (a, b):
                    total_forces[j] += max(0.0, wrench[0])
            if not chain_contact:
                continue
            if wrench[0] < 0.005:
                continue
            graph[a].add(b)
            graph[b].add(a)
            for j, finger in enumerate(fingers):
                if finger in (a, b):
                    forces[j] += wrench[0]
        seen, stack = set(), [fingers[0]]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node] - seen)
        self._total_pad_forces = total_forces
        self.max_pad_force = max(self.max_pad_force, float(max(total_forces)))
        return nodes <= seen and bool(np.all(forces > 0.12)), forces

    def _insertion_jammed(self, forces, limit=None):
        """Whether the pads have been over ``limit`` newtons for three cycles.

        ``limit`` lets a variant tighten the threshold for one specific move
        (the same-column expert does this for its base-push, which meets the
        Cookie edges side-on and would otherwise read as a normal insertion).
        """
        threshold = self.MAX_INSERTION_FORCE_N if limit is None else limit
        self._jam_count = self._jam_count + 1 if max(forces) > threshold else 0
        return self._jam_count >= 3

    def _release_settled(self) -> bool:
        """Whether the just-placed batch may be counted as released.

        The scene's own containment test decides this, not a contact query.
        ``privileged_cookie_in_target`` already requires each Cookie to be
        settled and -- whenever the scene sets ``require_released`` -- to be down
        on the box floor, and a Cookie in a pad's grip is in neither state.  So
        the separate "no finger contact on this batch" test the expert used to
        add could only ever be satisfied *later* than the scene's own answer, and
        there is a state where it is never satisfied at all: a Cookie that comes
        to rest leaning against a retracted pad keeps a contact alive
        indefinitely.  ``all_inside`` was then true while ``released`` was false,
        so neither the pass nor the fail branch could fire and the phase parked
        until its own timeout -- measured under the fast profile, 420 steps of a
        two-box fill, reported only as a bare "batch phase timeout".

        When a scene does *not* check release the contact query is what keeps
        this honest, because without ``require_released`` a Cookie merely
        *dangling* inside the box satisfies the containment geometry.  So it is
        kept for that case only.
        """

        if self.env.task_config.require_released:
            return True
        return not any(
            any(self.env.privileged_left_finger_contacts(i))
            for i in self.batch_indices
        )

    def _retreat_clear(self) -> bool:
        """Whether a retreat has lifted the tool to the height it commanded.

        A retreat's job is to get the jaws up and out of the box, and that is a
        statement about *height*: the retreat target is the placement pose plus
        ``retract_m_per_step``'s full distance straight up, so reaching its z
        means the tool is clear of the rim.  It used to end only on the servo's
        own arrival test, which compares the *measured* tool pose against the
        target in all three axes -- and a retreat that is dragging the box it
        just filled cannot close the horizontal part of that gap: measured under
        the fast profile, a two-box fill's second retreat had already lifted the
        full 85 mm, with both pads clear of the rim, yet it sat 14 mm short
        horizontally and oscillated at a four-step period for the whole 421-step
        phase budget, pulling the box 5 mm sideways while it did.  The height is
        what the phase is for; the horizontal residual is something the next
        phase re-targets anyway.

        This cannot fire early.  The height *is* the retreat distance, so a tool
        that satisfies it has risen the whole way; an earlier version tested
        instead whether the finger geoms' centres had passed the rim, which is
        true while the pads are still inside the box -- the following approach
        then swept sideways through the walls and threw the box 60 mm.
        """

        return bool(
            self.data.site_xpos[self._l_site, 2]
            >= self._retract_pos[2] - self.RETREAT_HEIGHT_TOLERANCE_M
        )

    def _servo(self, position, rotation, opening, speed=0.0015):
        """Rate-limited Cartesian setpoint, advancing at most ``speed`` a step.

        ``rotation`` may be a quaternion or a rotation matrix; the call sites have
        one or the other to hand and the two differ only in how the target
        quaternion is built.  What makes a fast ``speed`` safe is in
        :mod:`a3_dual_arm_sim.arm_servo`, and which mechanism drives the move at
        all is ``self.profile``:

        * ``commanded_reference`` plans from the pose the arm was *told* to hold,
          so the plan cannot wind up on a blocked move and needs no clamp;
        * otherwise the plan is built from the measured pose and clamped within
          ``SERVO_ANTIWINDUP_M`` of it, which is what the baseline does.
        """
        rot = np.asarray(rotation, dtype=np.float64).ravel()
        if self.profile.commanded_reference:
            setpoint = self._servo_plan.setpoint(
                self.env.last_applied_action,
                position,
                speed,
            )
        else:
            current = self.data.site_xpos[self._l_site].copy()
            if self._servo_pos is None:
                self._servo_pos = current.copy()
            delta = np.asarray(position) - self._servo_pos
            self._servo_pos += delta * min(
                1.0,
                speed / max(float(np.linalg.norm(delta)), 1e-9),
            )
            tracking = self._servo_pos - current
            if np.linalg.norm(tracking) > self.SERVO_ANTIWINDUP_M:
                self._servo_pos = current + tracking * (
                    self.SERVO_ANTIWINDUP_M / np.linalg.norm(tracking)
                )
            setpoint = self._servo_pos
        target_q, actual_q, error_r = np.empty(4), np.empty(4), np.empty(3)
        if rot.size == 4:
            target_q[:] = rot
        else:
            mujoco.mju_mat2Quat(target_q, rot)
        mujoco.mju_mat2Quat(actual_q, self.data.site_xmat[self._l_site])
        mujoco.mju_subQuat(error_r, target_q, actual_q)
        error_r = self.data.site_xmat[self._l_site].reshape(3, 3) @ error_r
        q = self._solve_l(setpoint, target_q, self.env.last_applied_action[:7])
        reached = (
            np.linalg.norm(position - self.data.site_xpos[self._l_site])
            < self.SERVO_ARRIVAL_M
            and np.linalg.norm(error_r) < self.SERVO_ARRIVAL_RAD
            and float(np.linalg.norm(self.data.qvel[self._l_dofs]))
            < self.SERVO_SETTLED_QVEL
        )
        return self._hold_command(q, opening), reached

    def _candidate_batches(self):
        """Prefer exposed ends; do not force an insertion through fallen Cookies."""
        source = np.asarray(self.env.SOURCE_POSITIONS)
        size = self._next_batch_size()
        for x in sorted(set(source[:, 0])):
            indices = np.flatnonzero(np.isclose(source[:, 0], x))
            remaining = sorted(
                (i for i in indices.tolist() if i not in self.completed_cookie_indices),
                key=lambda i: source[i, 1],
            )
            if len(remaining) < size:
                continue
            for batch in (remaining[:size], remaining[-size:]):
                if all(
                    self.env.privileged_cookie_in_source(i)
                    and abs(self.data.xmat[self.env._cookie_bodies[i], 8])
                    >= np.cos(np.deg2rad(10))
                    for i in batch
                ):
                    yield batch

    def _next_batch_size(self) -> int:
        """How many Cookies the next grasp takes.

        The plan's own size for this batch, not ``per_grasp``: a capacity that does
        not divide by the column count gives a short batch (ten Cookies at three per
        grasp is ``[3, 2, 3, 2]``), and grasping three for a two-slot batch would
        leave one Cookie with nowhere to go.  Falls back to the class attribute
        ``per_grasp`` when a controller is driven without ``reset``, which is how the
        unit tests exercise one method at a time.
        """

        plan = getattr(self, "plan", None)
        if plan is None:
            return self.per_grasp
        index = min(self.batch_index, len(plan.groups) - 1)
        return plan.groups[index].size

    #: Narrowest the jaws may close to while squeezing a batch, and the opening at or
    #: below which the batch counts as held.  Both are fractions of the gripper's
    #: stroke, and both were measured on the five-Cookie batch.
    #:
    #: A batch's width is proportional to how many Cookies are in it, so a fraction
    #: fixed in the stroke is only right for the size it was measured at -- and below
    #: five it is not merely imprecise, it is unusable.  Measured on a three-Cookie
    #: fill: the shipped 0.28 floor is 23.8 mm of jaw gap, *wider* than the 17.7 mm
    #: batch, so CLOSE opens the jaws from 26.6 mm to 30.2 mm and reads 0.00 N of pad
    #: force for the whole phase.  Every fill at 1, 2 and 3 Cookies per grasp failed
    #: that way, at its first batch, on every seed.
    CLOSE_FLOOR_AT_SHIPPED_GRASP = 0.28
    CLOSE_HELD_AT_SHIPPED_GRASP = 0.33
    #: The grasp size those two were measured at.  At that size the scaling below is
    #: exactly 1.0, so the shipped behaviour is reproduced bit for bit.
    SHIPPED_GRASP_SIZE = 5

    def _close_floor(self) -> float:
        """The narrowest the jaws may close to while squeezing *this* batch."""

        return self.CLOSE_FLOOR_AT_SHIPPED_GRASP * (
            self._next_batch_size() / self.SHIPPED_GRASP_SIZE
        )

    def _close_held(self) -> float:
        """The opening at or below which *this* batch counts as held."""

        return self.CLOSE_HELD_AT_SHIPPED_GRASP * (
            self._next_batch_size() / self.SHIPPED_GRASP_SIZE
        )

    def _is_later_batch(self) -> bool:
        """Whether this is not the first grasp of the fill.

        The batch expert's tuning is split between the first grasp and the rest, and
        it used to say so as ``batch_index == 0`` in one place and ``batch_index ==
        1`` in several others -- which for the shipped two-batch plan are the same
        two statements, so naming it keeps the anchor exact while saying what the
        split is actually about.

        Why the first batch is different: it is taken from the *exposed* end of the
        column, so its approach comes in from outside the row and its insertion has
        the emptied space ahead of it.  Every later batch descends past Cookies the
        earlier ones already took, which is what the same-column expert's neighbour
        logic, rim-contact stop and tightened closing width all exist for.

        It is deliberately not "has a following row", which every batch has.
        Whether every later batch of a four-batch plan wants all of that treatment is
        untested -- the anchor pins the two-batch case only -- so a smaller grasp
        size measures it rather than assuming it.
        """

        return getattr(self, "batch_index", 0) > 0

    def _select_batch(self):
        source = np.asarray(self.env.SOURCE_POSITIONS)
        self.batch_indices = next(self._candidate_batches(), [])
        expected = self._next_batch_size()
        if len(self.batch_indices) != expected:
            raise RuntimeError(
                f"no exposed, upright batch of {expected} Cookies available"
            )
        self.current_cookie_index = self.batch_indices[0]
        self.target_slot_index = self.batch_index
        positions = self._positions()
        self._pick_center = positions.mean(axis=0)
        # The batch's own measured spacing, which is what the expert has always used
        # -- and it is *not* the layout's nominal pitch: the configs write their row
        # positions to seven decimals, so the differences between the written values
        # are not all equal (they alternate 0.0088334 and 0.0088333 here) and their
        # median differs from `rows[1] - rows[0]` in the eighth significant figure.
        # That is 5 nm, but it is a real difference rather than rounding, and keeping
        # the old expression is what makes "no behaviour change" a measurement.
        #
        # A one-Cookie batch has no spacing to measure -- `np.median` of an empty
        # difference is NaN -- so it falls back to the layout's pitch, which is the
        # only spacing such a batch has.
        if len(self.batch_indices) > 1:
            pitch = float(np.median(np.diff(source[self.batch_indices, 1])))
        else:
            pitch = float(self.env.COOKIE_PITCH_Y)
        # Pad centres land in the two boundary gaps. Their flat bottoms first
        # meet the sloped bevel, then push neighbours apart during slow descent.
        pad_thickness = 0.00635
        self._insert_opening = float(
            np.clip((expected * pitch - pad_thickness) / 0.085, 0, 1)
        )
        self._opening = self._insert_opening
        self._pick_eef = self._pick_center + [0, 0, 0.010] - self._canonical @ self._pad_offset
        self._high_eef = self._pick_eef + [0, 0, self.profile.grasp_clearance_m]
        if self.profile.joint_transit:
            # Seed from the measured deployment pose so the first motion stays on
            # the nearby mirrored IK branch instead of taking the transit branch
            # with a large wrist/elbow rotation.
            self._approach_q = self._solve_l(
                self._high_eef,
                self._target_quat_canonical,
                self.env.current_joint_action[:7],
            )
        self._advance(CookiePhase.APPROACH)

    def _place_pose(self, clearance=0.0, group=None):
        """The tool pose that places ``group``'s batch, ``clearance`` above the box.

        Aimed at the mean of the slots *that batch* fills, rather than at the
        column's centre: a plan with two batches in one column has two different
        poses, and using the column's centre for both would drop the second batch in
        the wrong place.  For the shipped one-batch-per-column plan the two are the
        same point, because a batch that fills a whole column has that column's
        centre as its own mean -- which is why this change is invisible to the
        shipped scenes and shows up only in the plan's own tests.
        """

        group = self.plan.groups[self.batch_index] if group is None else group
        rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        batch_slots = slots[np.asarray(group.slot_indices)]
        center = np.array(
            [
                # Every slot of a column shares its x, so reading the first one is
                # exact -- and it is the same number the old per-column formula took
                # out of `np.unique`.  A *mean* of five identical values is not
                # bit-identical to the value, and that one ulp propagates through the
                # IK into every action of the episode, which would make the anchor a
                # tolerance rather than an equality.
                batch_slots[0, 0],
                # The y does have to be averaged, and this is the same mean the old
                # formula took: the same slots, in the same lattice order.
                batch_slots[:, 1].mean(),
                self.env._target_cookie_center_z + 0.002 + clearance,
            ]
        )
        center = self.data.xpos[self._tb_id] + rotation @ center
        tool_r = rotation @ self._canonical
        return center - tool_r @ self._group_offset, tool_r

    def act(self, observation=None, task=""):
        try:
            return self._act_step(observation, task)
        except RuntimeError as exc:
            self._fail(str(exc))
            return self.env.last_applied_action.copy()

    def _act_step(self, observation=None, task=""):
        del observation, task
        if self.finished:
            return self.env.last_applied_action.copy()
        if self.phase is CookiePhase.SELECT_COOKIE:
            self._select_batch()
        self.phase_steps += 1
        self._contact_chain()
        if self.phase_steps > 420:
            # Named, because the phase is what says which move failed: a timeout
            # in DESCEND means the insertion never seated, in CLOSE that the
            # jaws closed on nothing, in LIFT that the batch was not held.  The
            # pad forces alone cannot tell those apart -- all three read zero on
            # a grasp that never formed.
            self._fail(
                f"batch {self.batch_index + 1} {self.phase.name} timed out after "
                f"{self.phase_steps} steps; "
                f"pad forces={self._contact_chain()[1].tolist()} N, "
                f"opening={self._opening:.3f}"
            )
            return self.env.last_applied_action.copy()
        if self.phase is CookiePhase.APPROACH:
            # Travel with the jaws fully open.  The target batch width is set
            # only once the tool is steady above the Cookies.
            if self.profile.joint_transit:
                action = self._trajectory_command(self._approach_q, 1.0)
                if self._motion_done and self._reached(
                    self._approach_q, tolerance=0.025
                ):
                    self._advance(CookiePhase.PRE_CLOSE)
                return action
            action, reached = self._servo(
                self._high_eef, self._canonical, 1.0, speed=self.profile.transit_m_per_step
            )
            if reached:
                self._advance(CookiePhase.PRE_CLOSE)
            return action
        if self.phase is CookiePhase.PRE_CLOSE:
            if self.profile.joint_transit:
                action = self._hold_command(self._approach_q, self._opening)
            else:
                action, _ = self._servo(
                    self._high_eef,
                    self._canonical,
                    self._opening,
                    speed=self.profile.hold_m_per_step,
                )
            actual_opening = float(self.env.current_joint_action[7])
            if abs(actual_opening - self._opening) <= self.profile.preclose_tolerance:
                self._preclose_stable += 1
            else:
                self._preclose_stable = 0
            if self._preclose_stable >= self.profile.preclose_stable_steps:
                self._advance(CookiePhase.DESCEND)
            elif self.phase_steps >= 120:
                self._fail(
                    "pre-close did not reach target opening; "
                    f"target={self._opening * 0.085 * 1000:.1f} mm "
                    f"actual={actual_opening * 0.085 * 1000:.1f} mm"
                )
            return action
        if self.phase is CookiePhase.DESCEND:
            if self._insertion_jammed(self._total_pad_forces):
                self._fail(f"insertion blocked; pad forces={self._total_pad_forces.tolist()}")
                return self.env.last_applied_action.copy()
            insertion_distance = float(
                np.linalg.norm(self._pick_eef - self.data.site_xpos[self._l_site])
            )
            action, reached = self._servo(
                self._pick_eef,
                self._canonical,
                self._opening,
                speed=(
                    self.profile.descend_far_m_per_step
                    if insertion_distance > self.profile.near_m
                    else self.profile.descend_near_m_per_step
                ),
            )
            if reached:
                self._advance(CookiePhase.CLOSE)
            return action
        if self.phase is CookiePhase.CLOSE:
            chain, forces = self._contact_chain()
            # Close fast until the jaws touch anything, then at a rate the five
            # Cookies can absorb.  The first half of the travel is free.
            close_rate = (
                self.profile.close_free_m_per_step
                if (not chain or min(forces) < 0.5)
                else self.profile.close_contact_m_per_step
            )
            if not chain or min(forces) < 2.5:
                self._opening = max(self._close_floor(), self._opening - close_rate)
            self._stable = self._stable + 1 if chain and min(forces) >= 2.0 else 0
            action, _ = self._servo(
                self._pick_eef, self._canonical, self._opening, speed=0.0008
            )
            if self._stable >= self.profile.close_stable_steps:
                self._grip_reference = self._positions().copy()
                self._lift_start_eef = self.data.site_xpos[self._l_site].copy()
                self._advance(CookiePhase.LIFT)
            return action
        if self.phase is CookiePhase.LIFT:
            _, forces = self._contact_chain()
            if min(forces) < 2.0:
                self._opening = max(self._close_floor(), self._opening - 0.001)
            lifted_so_far = float(
                np.linalg.norm(self.data.site_xpos[self._l_site] - self._lift_start_eef)
            )
            action, reached = self._servo(
                self._high_eef,
                self._canonical,
                self._opening,
                speed=(
                    self.profile.lift_near_m_per_step
                    if lifted_so_far < self.profile.near_m
                    else self.profile.lift_far_m_per_step
                ),
            )
            lifted = self._positions()[:, 2] - self._grip_reference[:, 2]
            tool_rise = self.data.site_xpos[self._l_site, 2] - self._lift_start_eef[2]
            if tool_rise > 0.015 and np.any(tool_rise - lifted > 0.016):
                self._fail("Cookie dropped during lift")
                return self.env.last_applied_action.copy()
            if reached:
                if not np.all(lifted > 0.070):
                    self._fail(f"not all five lifted: {lifted.tolist()}")
                else:
                    r = self.data.site_xmat[self._l_site].reshape(3, 3)
                    self._group_offset = r.T @ (
                        self._positions().mean(axis=0) - self.data.site_xpos[self._l_site]
                    )
                    self._held_offsets = (self._positions() - self.data.site_xpos[self._l_site]) @ r
                    self.batch_reports.append(
                        {
                            "cookies": list(self.batch_indices),
                            "lifted": len(self.batch_indices),
                            "lift_heights_m": lifted.tolist(),
                            "released": False,
                        }
                    )
                    self._advance(CookiePhase.MOVE_TO_SLOT)
            return action
        if self.phase in (CookiePhase.MOVE_TO_SLOT, CookiePhase.DESCEND_TO_PLACE):
            r = self.data.site_xmat[self._l_site].reshape(3, 3)
            offsets = (self._positions() - self.data.site_xpos[self._l_site]) @ r
            if np.max(np.linalg.norm(offsets - self._held_offsets, axis=1)) > 0.016:
                self._fail("Cookie slipped out of batch during transport")
                return self.env.last_applied_action.copy()
            moving = self.phase is CookiePhase.MOVE_TO_SLOT
            clearance = self.profile.transport_clearance_m if moving else 0.0
            pos, rotation = self._place_pose(clearance)
            if moving:
                place_speed = self.profile.place_free_m_per_step
            else:
                to_place = float(np.linalg.norm(pos - self.data.site_xpos[self._l_site]))
                place_speed = (
                    self.profile.place_far_m_per_step
                    if to_place > self.profile.near_m
                    else self.profile.place_near_m_per_step
                )
            action, reached = self._servo(
                pos,
                rotation,
                self._opening,
                speed=place_speed,
            )
            if reached:
                self._advance(CookiePhase.DESCEND_TO_PLACE if moving else CookiePhase.OPEN)
            return action
        if self.phase is CookiePhase.OPEN:
            # The jaws are done with the batch; opening them is not a contact
            # move, so it does not need to be slow, only synchronized with the
            # servo that holds the placement pose.
            self._opening = min(0.49, self._opening + self.profile.open_m_per_step)
            pos, rotation = self._place_pose()
            action, _ = self._servo(pos, rotation, self._opening, speed=0.0010)
            if self._opening >= 0.49 and self.phase_steps >= 8:
                self._retract_pos = (
                    self.data.site_xpos[self._l_site].copy() + rotation[:, 1] * -0.085
                )
                self._retract_rotation = rotation.copy()
                self._advance(CookiePhase.RETRACT)
            return action
        if self.phase is CookiePhase.RETRACT:
            action, reached = self._servo(
                self._retract_pos, self._retract_rotation, self._opening,
                speed=self.profile.retract_m_per_step,
            )
            if reached or self._retreat_clear():
                self._advance(CookiePhase.VERIFY_RELEASE)
            return action
        if self.phase is CookiePhase.VERIFY_RELEASE:
            all_inside = all(
                self.env.privileged_cookie_in_target(i)
                for i in self.completed_cookie_indices + self.batch_indices
            )
            released = self._release_settled()
            self._stable = self._stable + 1 if all_inside and released else 0
            if self.phase_steps >= 80 and not all_inside:
                bad = [
                    i for i in self.completed_cookie_indices + self.batch_indices
                    if not self.env.privileged_cookie_in_target(i)
                ]
                self._fail(f"released Cookies not upright, settled and contained: {bad}")
                return self.env.last_applied_action.copy()
            if self.phase_steps >= self.RELEASE_SETTLE_STEPS and self._stable == 0:
                self._fail(
                    f"batch {self.batch_index + 1} never settled in the box: "
                    f"all contained={all_inside}, pad still on the batch={not released}"
                )
                return self.env.last_applied_action.copy()
            # The first batch only has to be released; the last is the one the
            # scene's success test watches, so it is the one that has to hold for
            # the full success window.
            required_stable = (
                self.env.task_config.success_hold_steps
                if self.batch_index == self._final_batch
                else self.profile.release_stable_first
            )
            if self._stable >= required_stable:
                self.batch_reports[-1]["released"] = True
                self.completed_cookie_indices.extend(self.batch_indices)
                self.batch_index += 1
                self._advance(
                    CookiePhase.DONE
                    if self.batch_index > self._final_batch
                    else CookiePhase.SELECT_COOKIE
                )
            return self.env.last_applied_action.copy()
        self._fail("unsupported batch phase")
        return self.env.last_applied_action.copy()
