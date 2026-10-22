"""Contact-driven five-Cookie grasp through bevels; privileged, not a VLA policy.

Objects are never attached, teleported, or moved by this controller. Each batch
must form a pad--Cookie contact chain, lift together, and be released into the box.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .arm_servo import CommandedPose
from .batch_profile import PROFILES, BatchProfile
from .contracts import LEFT_JOINTS
from .expert import A3CookieTransferExpert, CookiePhase


class A3CookieBatchExpert(A3CookieTransferExpert):
    MAX_INSERTION_FORCE_N = 8.0

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

    #: Kept between the lowest **colliding** point on either jaw and the level the batch of
    #: Cookies stands on, so that seating the pads cannot drive the hand into the bin floor.
    #: See `_jaw_lowest_offset` for why this is measured per hand instead of assumed.
    JAW_FLOOR_CLEARANCE_M = 0.001

    def reset(self, context=None):
        super().reset(context)
        self.profile: BatchProfile = PROFILES[self.env.config.batch_expert_profile]
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
        # **The second-batch condition this episode runs under.**  Drawn by the
        # environment's reset from `second_batch_modes`, by seed; the expert applies
        # it because the batch selection -- which five, which column, which end --
        # is the expert's own state, not the environment's.  `full` is the no-op and
        # the default, so a scene that has never heard of the field is byte-identical
        # to what it was.
        mode = getattr(self.env, "second_batch_mode", "full")
        if mode in ("prefill", "prefill_perturb"):
            self._prefill_first_batch(hard=(mode == "prefill_perturb"))
        elif mode == "adopt":
            # The DAgger-shaped entry: continue from whatever is already in the box.  See
            # `_adopt_current_state` for why nothing is moved here.
            self._adopt_current_state()

    def _prefill_first_batch(self, *, hard: bool) -> None:
        """Start the episode *at* the second batch: five Cookies already in the box.

        Why this exists: a batch collection's episodes are otherwise all "start empty,
        do batch 0, do batch 1", so a policy trained on them meets the second batch
        only ever from the one state the *expert's* first batch leaves behind -- five
        perfectly upright Cookies at the slot centres, which is a state a policy never
        produces.  Its own first batch leaves five imperfectly-placed ones, and from
        there it is off the distribution it was trained on, which is the classic
        covariate-shift hole and the shape of every "first five land, second five
        don't" result so far.

        So this takes the five the first batch *would have* taken (the same
        `_candidate_batches` first yield, so the remaining second batch is the same
        move it always is), teleports them into the first column's slots with drawn
        imperfections -- leaning, jittered, yawed, each by a different amount -- and
        seeds the bookkeeping so the state machine opens on the second batch:

        ``completed_cookie_indices`` carries the five (so `_candidate_batches` skips
        them), ``batch_index`` is 1 (so `_select_batch` aims the *other* column, and
        `_place_pose` reads it), and ``batch_reports`` carries a released report for
        the batch that "already happened" (the report the LIFT phase would have
        written; VERIFY_RELEASE only ever reads the last one, which batch 1's own
        LIFT will have become by then).

        The arm stays at its home: a real second batch starts from wherever RETRACT
        left the tool, near the box at transit height, and this starts from home --
        which is *also* a state a redeployed policy starts from, and the one a
        mid-episode handoff has to survive.  Both are progress states; neither is the
        one the baseline trains.

        ``hard`` is the ``prefill_perturb`` draw: the same placement with wider
        maxima, so the worst-case second batch -- progress that is both partial *and*
        messy -- is in the data too.  The maxima are bounded by the packing (see
        `place_cookie_in_slot`): the column's five slots are 6.33 mm apart in y
        against a 6.33 mm-thick Cookie, so the lean and the yaw both cost lateral
        room the group absorbs by shifting within the box's ~14 mm of margin.
        """

        batch = next(self._candidate_batches(), [])
        if len(batch) != 5:
            raise RuntimeError(
                "no exposed, upright five-Cookie batch available to prefill from"
            )
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        column_x = np.unique(slots[:, 0])[0]
        column = [
            index
            for index in range(len(slots))
            if np.isclose(slots[index, 0], column_x)
        ]
        column.sort(key=lambda index: slots[index, 1])
        # **One draw, shared by the whole group.**  The lean and the yaw are drawn once
        # here and passed to every Cookie of the five, because the group has zero
        # clearance cookie-to-cookie (6.33 mm slots, 6.33 mm Cookies) and an
        # independently drawn lean towards a neighbour interpenetrates it -- measured,
        # the first version's independently drawn leans ended with the group's two end
        # Cookies outside the box.  A coherent lean is a parallelogram: no pair
        # interpenetrates, and it is also the realistic shape, a group that settled
        # together leans together.  The per-Cookie jitter stays individual -- it is
        # sub-millimetre, and it is where the variety of an imperfect placement lives.
        rng = self.env.np_random
        if hard:
            max_lean, max_yaw, max_jitter = np.deg2rad(6.0), np.deg2rad(1.5), 0.0003
        else:
            max_lean, max_yaw, max_jitter = np.deg2rad(4.0), np.deg2rad(1.0), 0.0002
        lean = float(rng.uniform(0.0, max_lean))
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        yaw = float(rng.uniform(-max_yaw, max_yaw))
        for cookie, slot in zip(batch, column, strict=True):
            self.env.place_cookie_in_slot(
                cookie,
                slot,
                lean_azimuth_rad=azimuth,
                lean_rad=lean,
                yaw_rad=yaw,
                max_jitter_m=float(max_jitter),
            )
        self.completed_cookie_indices = list(batch)
        self.batch_index = 1
        self.batch_reports.append(
            {
                "cookies": list(batch),
                "lifted": 5,
                "lift_heights_m": [0.0] * 5,
                "released": True,
                "prefilled": True,
            }
        )

    def _adopt_current_state(self) -> None:
        """Continue from whatever is already in the box, moving nothing.

        `_prefill_first_batch` teleports five Cookies into the slots because the alternative
        was starting from an empty box.  This is the other case: the Cookies are already in
        the box, put there by a *policy*, and they are not where the expert would have put
        them (measured on the 4.05 checkpoint: a coherent block 2-10 mm off the slot column,
        4 % of Cookies within 2 mm -- see `docs/pi05-training-history.md` 9.25).  The point
        of taking over here is to demonstrate recovery from exactly that state, so nothing
        is moved and no slot is re-derived:

        * ``completed_cookie_indices`` becomes the Cookies the judge already counts inside
          the target, so `_candidate_batches` skips them and the expert asks for a *new*
          five from the source;
        * ``batch_index`` becomes 1 when five are there, which is what makes `_place_pose`
          aim at the other column;
        * a released report is appended for the batch that "already happened", because
          VERIFY_RELEASE only ever reads the most recent one.

        **The slot targets are deliberately not shifted by the block's offset.**  The slots
        are the box's geometry; the Cookies are what is misplaced.  Following the Cookies
        would teach the second batch to aim at the mistake, and the recovery the policy needs
        is the opposite: put the next five where the slots actually are.
        """
        in_target = tuple(
            self.env._cookie_inside_target(index)  # noqa: SLF001 - the judge, not a policy input
            for index in range(len(self.env.cookie_positions))
        )
        completed = [index for index, inside in enumerate(in_target) if inside]
        self.completed_cookie_indices = completed
        self.batch_indices = []
        self.adopted_cookie_count = len(completed)
        if len(completed) >= 5:
            self.batch_index = 1
        self.adopted_slot_column = self._free_slot_column(in_target)
        self.batch_reports.append(
            {
                "cookies": list(completed),
                "lifted": len(completed),
                "lift_heights_m": [0.0] * len(completed),
                "released": True,
                "adopted": True,
            }
        )

    def _free_slot_column(self, in_target: tuple[bool, ...]) -> int:
        """The slot column that is emptiest right now, so an adopted box is not re-filled.

        `batch_index` is the wrong answer here, and the gate is what showed it: with the
        policy's five already in one column, `batch_index = 1` sends the expert at the *other*
        column by index -- which is right only if the policy happened to use column 0.  On the
        seeds where the policy filled column 1 instead, the expert descended straight onto the
        full column and stopped at 39.5 / 39.6 N with the pads on nothing, three times out of
        four at `execute: 48`.

        Occupancy is the thing that is actually known at takeover, so it is what decides.  A
        tie goes to `batch_index`, which keeps the empty-box path byte-identical.
        """
        occupancy = self.env._target_slot_occupancy(in_target)  # noqa: SLF001 - the judge
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        columns = np.unique(slots[:, 0])
        taken = [
            sum(1 for index, cookie in enumerate(occupancy)
                if cookie >= 0 and np.isclose(slots[index, 0], column))
            for column in columns
        ]
        fewest = min(taken)
        candidates = [column for column, count in enumerate(taken) if count == fewest]
        preferred = self.batch_index % len(columns)
        return preferred if preferred in candidates else candidates[0]

    def _validate_target_workspace(self):
        """Reject unreachable table layouts before starting a physical grasp."""
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        for column in range(2):
            for clearance in (0.0, self.profile.transport_clearance_m):
                position, rotation = self._place_pose(clearance, column)
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
                        f"target column {column + 1} unreachable with vertical grasp: "
                        f"{exc}; reposition the tabletop target box"
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
                        f"target column {column + 1} unreachable with vertical grasp: "
                        f"position error {distance * 1000:.2f} mm, "
                        f"angle error {np.rad2deg(np.linalg.norm(error)):.2f} deg; "
                        "reposition the tabletop target box"
                    )

    @property
    def status(self):
        return (
            f"phase={self.phase.name} batch={min(self.batch_index + 1, 2)}/2 "
            f"cookies={self.batch_indices} confirmed={len(self.completed_cookie_indices)}/10"
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
        # **A jaw is a chain of bodies, and the contact does not have to land on the geom the jaw
        # is named after.**  `fingers` is one geom per jaw -- the one on the finger *body* -- and
        # the graph used to be built from those geoms directly, so a contact anywhere else on the
        # chain was invisible twice over: it added nothing to `total_forces`, and it could not
        # join a Cookie to the hand.  On LIBERO's Panda the part that meets a Cookie is
        # `gripper0_finger{1,2}_pad_collision`, a box on the *pad* body, and the measurement is
        # this: with both pads in contact --
        #
        #     gripper0_finger1_pad_collision <-> cookie_0_geom  dist -0.02 mm  |fn| 1.42 N
        #     gripper0_finger2_pad_collision <-> cookie_4_geom  dist -0.00 mm  |fn| 0.37 N
        #
        # -- the expert read `pad forces=[0.0, 0.0] N`, so `CLOSE` never saw the 2.0 N it waits
        # for, drove its command to 0.000 and failed after 421 steps.  `env._left_jaw_geoms` is
        # the tree walk that already exists for exactly this reason, and routing every jaw geom
        # through it is what makes the two hands measure the same way.  A3 is unaffected in
        # principle -- its pads *are* the named geoms, so the mapping is the identity on the
        # contacts it makes -- and that is checked by re-running `a3_cookie_batch` rather than
        # assumed.
        jaw_of_geom = {
            int(geom): index
            for index, jaw in enumerate(getattr(self.env, "_left_jaw_geoms", ()))
            for geom in jaw
        }

        def node_of(raw: int) -> int:
            """The graph node a contact's geom belongs to: a jaw's named geom, or a Cookie."""
            if raw in jaw_of_geom:
                return int(fingers[jaw_of_geom[raw]])
            return int(mapping.get(raw, raw))

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
            a = node_of(int(contact.geom1))
            b = node_of(int(contact.geom2))
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

        **``commanded_reference`` needs a joint-target arm, and only half of that
        condition used to be checked.**  It reads the commanded pose out of
        ``env.last_applied_action``, which under `OperationalSpaceAdapter` holds the
        controller's solved **torques** and not a pose -- so on the Panda this branch
        built a setpoint out of Newtons.  `applied_action_is_joint_target` is the
        environment's own answer to "is that vector a pose", and where it is not, the
        measured-pose clamp below is the right mechanism rather than a fallback: an
        operational-space goal is latched and *can* wind up against a blocked move,
        which is the thing the clamp exists to prevent.
        """
        rot = np.asarray(rotation, dtype=np.float64).ravel()
        if self.profile.commanded_reference and self.env.applied_action_is_joint_target:
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
            if self._clamp_setpoint and np.linalg.norm(tracking) > self.SERVO_ANTIWINDUP_M:
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
        q = self._solve_l(setpoint, target_q, self._left_joint_feedback())
        reached = (
            np.linalg.norm(position - self.data.site_xpos[self._l_site])
            < self.SERVO_ARRIVAL_M
            and np.linalg.norm(error_r) < self.SERVO_ARRIVAL_RAD
            and float(np.linalg.norm(self.data.qvel[self._l_dofs]))
            < self.SERVO_SETTLED_QVEL
        )
        return self._hold_command(q, opening), reached

    @property
    def _clamp_setpoint(self) -> bool:
        """Whether the plan has to be pulled back to within ``SERVO_ANTIWINDUP_M`` of the arm.

        Only for a **joint-target** arm, where the command *is* the pose: there a blocked move
        leaves the commanded pose ahead of the arm and the servo goes on pushing, which is the
        windup this clamp exists to remove.

        A torque arm driven through `OperationalSpaceAdapter` is the opposite case.  Its goal is
        re-derived from the **measured** state at the start of every control step, so a blocked
        move leaves nothing latched to unwind -- and the clamp is then not merely unnecessary,
        it throttles every transit.  Measured on the dense Panda scene: with the clamp the
        `APPROACH` servo asks for 15 mm/step and the tool covers **1.14 mm/step**, because the
        plan can never sit more than 8 mm ahead of an arm that is tracking at about a
        millimetre; the 882 mm transit therefore needs some 780 steps and the phase's 420-step
        limit expires with **405 mm** still to go.
        """

        return self.env.applied_action_is_joint_target

    def _preclose_arrived(self, actual_opening: float) -> bool:
        """Whether the jaws are at the pre-close opening, to within a **distance**.

        `preclose_tolerance` is metres on the finger and is converted through this hand's own
        travel here, because an "opening value" is a different distance on a different
        gripper.  Measured on the dense Panda scene: commanded to a 43.6 mm aperture the jaws
        settle at 45.7 mm and stay there, `qvel` 0, no joint at its limit -- a 1.0 mm droop
        from a load this gripper's `kp` does not remove.  In opening units that is 0.025, so
        the old 0.008 asked for 0.32 mm, four times tighter than the 1.5 mm
        ``SERVO_ARRIVAL_M`` the same expert accepts for every other move, and `PRE_CLOSE`
        could never pass on that hand.
        """

        travel = float(self.env._finger_travel["L"])
        return abs(actual_opening - self._opening) * travel <= self.profile.preclose_tolerance

    #: How far a source Cookie may sit from the point it was placed at and still be grasped as
    #: part of its row.  A row grasp puts one fingertip in each end groove, and A3's fingertip is
    #: 1.00 mm wide in a 2.50 mm groove -- 0.75 mm of slack a side -- so a Cookie that has been
    #: shoved more than about a millimetre is no longer part of the row the pads close on.
    SOURCE_IN_PLACE_TOLERANCE_M = 0.001

    def _graspable_in_place(self, index: int, source: np.ndarray, actual: np.ndarray) -> bool:
        """Is Cookie ``index`` still standing where the row grasp assumes it is?"""
        if index in self.completed_cookie_indices:
            return False
        if not self.env.privileged_cookie_in_source(index):
            return False
        if abs(self.data.xmat[self.env._cookie_bodies[index], 8]) < np.cos(np.deg2rad(10)):
            return False
        return (
            float(np.linalg.norm(actual[index][:2] - source[index][:2]))
            <= self.SOURCE_IN_PLACE_TOLERANCE_M
        )

    def _candidate_batches(self):
        """Prefer exposed ends; do not force an insertion through fallen Cookies.

        **Consecutive Cookies, not merely the first five of the ones that remain.**  A column is
        a row at one pitch, and the grasp closes round five neighbours; if the five handed to
        `_select_batch` are not neighbours the pads meet each other in the gaps.  Collection only
        ever consumed a column from one end, so index order and physical adjacency were the same
        thing and the difference could not show.  A *policy* takes its first batch from wherever
        it likes, and leaves a hole: measured on the 4.05 checkpoint, seed 0 had taken Cookies
        1-5 of a twenty-Cookie column, so the remaining 0 and 6-19 sorted to `[0, 6, 7, 8, 9]` --
        a 31 mm gap between the first two -- and CLOSE timed out with both pads at 8 N and the
        fingers fully shut.  Seeds 2 and 7 failed the same way.

        So a run has to be consecutive *and* still in place: each Cookie within
        `SOURCE_IN_PLACE_TOLERANCE_M` of where it was put, upright, and inside the source bin.
        Undisturbed rows pass unchanged, which is why this is a filter rather than a new
        behaviour.
        """
        source = np.asarray(self.env.SOURCE_POSITIONS)
        actual = np.asarray(self.env.cookie_positions)
        for x in sorted(set(source[:, 0])):
            indices = np.flatnonzero(np.isclose(source[:, 0], x))
            order = sorted(indices.tolist(), key=lambda i: source[i, 1])
            runs: list[list[int]] = []
            run: list[int] = []
            for index in order:
                if self._graspable_in_place(index, source, actual):
                    run.append(index)
                else:
                    if len(run) >= 5:
                        runs.append(run)
                    run = []
            if len(run) >= 5:
                runs.append(run)
            for whole in runs:
                yield whole[:5]
                if len(whole) > 5:
                    yield whole[-5:]

    def _jaw_lowest_offset(self) -> float:
        """Signed z, in the **tool** frame, of the lowest colliding point on either jaw.

        `_select_batch` seats the pads by putting the two finger geoms' midpoint 10 mm above the
        batch's centre, which is a statement about the *pads*, and on A3 the pads are also the
        lowest thing on the hand: measured, a pad centre 10 mm above a Cookie's centre leaves the
        pad's bottom (half-thickness 16.75 mm) **5.8 mm above the level the Cookies stand on**, so
        no descent this expert makes can reach the floor.

        On LIBERO's Panda that is not true, and the target is not a near miss: the finger
        **body's** collision mesh (`L_finger_{inner,outer}_geom`) hangs well below the pad box
        that gives the hand its thickness.  Measured on a five-Cookie column with free ends,
        where nothing else could obstruct the descent: the same target drove both jaws into the
        source bin's floor, `source_bin_floor <-> L_finger_{inner,outer}_geom` at **14.0 N and
        14.0 N** with `dist -0.56/-0.58 mm`, which `_insertion_jammed` reads as an insertion that
        cannot proceed -- a failure with nothing to do with the Cookies.

        So the depth is clamped by a measurement of this hand, taken once per episode in the pose
        the hand is actually in.  A3 is unaffected: the clamp only ever *raises* a target, and
        its pad bottom is already clear of the floor.
        """

        canonical = np.asarray(self._canonical, dtype=np.float64)
        site_rotation = np.asarray(
            self.data.site_xmat[self._l_site], dtype=np.float64
        ).reshape(3, 3)
        site_position = np.asarray(self.data.site_xpos[self._l_site], dtype=np.float64)
        lowest = 0.0
        for jaw in self.env._left_jaw_geoms:
            for geom in jaw:
                # Only what can collide: the visual meshes hang around the finger and would
                # otherwise set a depth that no contact ever justifies.
                if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                    continue
                points = self._geom_vertices(geom)
                # `R^T (p - c)` is the offset in the tool frame as the hand is *now*; the descent
                # drives the tool to `_canonical`, so that offset rotates by `_canonical` to give
                # the world offset it will have when the pads are seated.
                tool_offsets = (points - site_position) @ site_rotation
                lowest = min(lowest, float(np.min((tool_offsets @ canonical.T)[:, 2])))
        return lowest

    def _geom_vertices(self, geom: int) -> np.ndarray:
        """World coordinates of a geom's extreme points: its mesh's vertices, or its box corners.

        The two shapes the two jaw pads come in.  Anything else falls back to the bounding sphere
        `geom_rbound` gives, which is conservative in the direction this is used for -- it can
        only make the clamp raise the target further.
        """

        if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(self.model.geom_dataid[geom])
            start = int(self.model.mesh_vertadr[mesh])
            count = int(self.model.mesh_vertnum[mesh])
            local = np.asarray(self.model.mesh_vert[start : start + count], dtype=np.float64)
            rotation = np.asarray(self.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
            return local @ rotation.T + np.asarray(self.data.geom_xpos[geom], dtype=np.float64)
        if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            half = np.asarray(self.model.geom_size[geom][:3], dtype=np.float64)
            local = np.array(
                [
                    [sx * half[0], sy * half[1], sz * half[2]]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ]
            )
            rotation = np.asarray(self.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
            return local @ rotation.T + np.asarray(self.data.geom_xpos[geom], dtype=np.float64)
        centre = np.asarray(self.data.geom_xpos[geom], dtype=np.float64)
        radius = float(self.model.geom_rbound[geom])
        return centre + np.array([[0.0, 0.0, -radius], [0.0, 0.0, radius]])

    def _select_batch(self):
        source = np.asarray(self.env.SOURCE_POSITIONS)
        self.batch_indices = next(self._candidate_batches(), [])
        if len(self.batch_indices) != 5:
            raise RuntimeError("no exposed, upright five-Cookie batch available")
        self.current_cookie_index = self.batch_indices[0]
        #: In `adopt` mode the column is chosen from occupancy at reset, not from the index;
        #: outside it there is no occupancy to read and the index is the whole story.
        self.target_slot_index = getattr(self, "adopted_slot_column", self.batch_index)
        positions = self._positions()
        self._pick_center = positions.mean(axis=0)
        pitch = float(np.median(np.diff(source[self.batch_indices, 1])))
        self._insert_opening = self._insertion_opening(pitch)
        self._opening = self._insert_opening
        #: Whether these fingertips have to stand **outside** the row because they cannot fit a
        #: groove.  Decided by geometry here, once, because two different behaviours hang on
        #: it: a fingertip in a groove has to stay centred in it, and one outside the row has
        #: to open when it touches something early.
        self._finger_stands_outside = float(self.env.PAD_THICKNESS_M) >= (
            pitch - 2.0 * float(np.asarray(self.env.COOKIE_HALF_SIZE).ravel()[1])
        )
        # The nominal grasp pose, kept separate from `_pick_eef` because `_align_insertion`
        # rewrites `_pick_eef` every step of `DESCEND`.  Applying a correction to the
        # *previous* output instead of to the nominal one integrates it: the arm lags the
        # target by about the correction itself, so the same offset is measured again next
        # step and added again.  Measured on `a3_cookie_batch`: the corrected target ran away
        # at 0.8 mm per step and `DESCEND` jammed at 27.4 N after 48 steps.
        self._pick_eef_nominal = (
            self._pick_center + [0, 0, 0.010] - self._canonical @ self._pad_offset
        )
        # **And then no deeper than this hand's own lowest point allows.**  See
        # `_jaw_lowest_offset`: the target above describes where the pads go, and on LIBERO's
        # Panda the finger body reaches ~13 mm below them, so the same target put both jaws
        # 0.6 mm into the source bin's floor at 14 N apiece and read as a blocked insertion.
        # Raising the target costs nothing that matters -- the pads still cover 0.7617-0.7777 of
        # a Cookie standing from 0.756 to 0.781 -- and it only ever raises.
        standing = float(self._pick_center[2]) - float(
            np.asarray(self.env.COOKIE_HALF_SIZE).ravel()[2]
        )
        deepest = standing + self.JAW_FLOOR_CLEARANCE_M - self._jaw_lowest_offset()
        self._pick_eef_nominal[2] = max(self._pick_eef_nominal[2], deepest)
        self._pick_eef = self._pick_eef_nominal.copy()
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

    def _pad_separation(self, opening: float) -> float:
        """The two fingertip centres' separation the **model** would hold at ``opening``.

        Kinematics on a scratch state, driven through the environment's own finger encoding.
        **The polarity has to come from there rather than from an assumption**: on A3's
        gripper a finger joint at 0 is fully *open* (42.5 mm of travel) and on LIBERO's Panda
        it is fully *closed*, so "set the joints to zero and call that closed" measures
        91.36 mm on one hand and 23.92 mm on the other -- and the second is right only by
        coincidence.

        What this reports is what the fingers *would* do, which is not what they do: on the
        dense Panda scene the commanded aperture is 68.8 mm and the pads sit at 70.8 mm.  See
        `_align_insertion` for the loop that closes that gap.
        """

        model = self.env.model
        finger_geoms = list(self.env._left_finger_geoms)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = self.env.data.qpos
        travel = (
            self.env._finger_travel_fraction("L", opening) * self.env._finger_travel["L"]
        )
        for sign, joint in zip(
            self.env._finger_sign["L"], self.env._finger_joints["L"], strict=True
        ):
            scratch.qpos[model.jnt_qposadr[joint]] = sign * travel
        mujoco.mj_kinematics(model, scratch)
        return float(
            np.linalg.norm(scratch.geom_xpos[finger_geoms[0]] - scratch.geom_xpos[finger_geoms[1]])
        )

    def _align_insertion(self) -> None:
        """Trim the jaws and the tool onto the row the pads are actually meeting.

        Two measured errors, both of which put a fingertip through the end Cookie rather than
        onto it, and both of which are invisible to the commanded pose:

        * **the jaws do not reach the aperture they are given.**  Commanded 0.561 (68.8 mm of
          separation) the pads sat at **70.8 mm**, because the fingers hold a ~1 mm droop
          under a load this gripper's `kp` does not remove.  2.0 mm too wide is 1.0 mm per
          side, and there is only 0.2 mm of room.
        * **the pair is not centred on the row.**  The pads' midpoint came out **0.8 mm** off
          the five Cookies' centre, so even a correct width would be 0.8 mm out on one side.

        Together they produced 0.2 mm of clearance on one side and **1.8 mm of interference on
        the other**, which is the 73-130 N `_insertion_jammed` reports.  Both are corrected
        from measurements rather than from constants, so they hold on either hand: the opening
        is trimmed until the pads that exist span what the model asks for, and the tool target
        is shifted along the jaw axis until the midpoint is on the row.
        """

        finger_geoms = list(self.env._left_finger_geoms)
        if len(finger_geoms) != 2:
            return
        first = np.asarray(self.data.geom_xpos[finger_geoms[0]], dtype=np.float64).copy()
        second = np.asarray(self.data.geom_xpos[finger_geoms[1]], dtype=np.float64).copy()
        # **Width, and only while the pads are free.**  Once they touch, the separation they
        # hold is the Cookies' business rather than the hand's, so trimming on it would walk
        # the command closed into them instead of correcting anything -- and the descent is
        # supposed to *reach* them, not to be finished before it starts.
        #
        # The correction is towards the **wanted opening read off the measured separation**,
        # not towards "the measurement agrees with the model".  Those are not the same target
        # and only one of them is reachable: the jaws sit a fixed ~1 mm short of whatever they
        # are told under load, so `achieved - model(command)` is a constant +2.0 mm that no
        # command can remove.  Correcting that made the command walk 0.025 per step until the
        # jaws were 26 mm apart inside a 41.7 mm row -- measured, the pads' separation fell
        # 66 -> 26 mm over the descent and `DESCEND` jammed at 43.9 N.  Reading the wanted
        # opening out of the measurement instead has a fixed point at the droop it actually
        # has, which is what makes it converge.
        if not np.any(np.asarray(self._total_pad_forces) > 0.01):
            achieved = float(np.linalg.norm(second - first))
            closed = self._pad_separation(0.0)
            per_unit = self._pad_separation(1.0) - closed
            if abs(per_unit) > 1e-9:
                measured_opening = (achieved - closed) / per_unit
                step = float(np.clip(self._insert_opening - measured_opening, -0.02, 0.02))
                self._opening = float(np.clip(self._opening + step, 0.0, 1.0))
        # **Opening when a fingertip touches something mid-descent was tried here, and it is
        # the wrong way round.**  The contact is on `L_finger_outer_geom`, the *finger body's*
        # mesh, and the thing it meets is not in the batch at all: measured on step 18 of
        # `DESCEND`, `cookie_5` and `cookie_6` -- the next two pieces of the same twenty-long
        # column -- at `dist -0.36 mm` and 64/83 N.  Widening moves the finger *further into
        # them*: it took the sidestep from 65 N to 87 N and the run died at the same step.  The
        # real number is that this jaw's contact face is 8.3 mm from its own centre (26.40 mm on
        # the finger-body mesh, 7.50 mm on `gripper0_finger1_pad_collision`; the two agree to
        # the same 0.8 mm the centring loop removes), so the hand is **~16.6 mm thick** and the
        # groove between two packed Cookies is **2.50 mm**.  It cannot enter the row at all --
        # which is a fact about the fingertips rather than about this loop, and is why
        # `a3_libero_panda_dense_batch` cannot be solved five at a time on LIBERO's Panda.
        #
        # Centring, always, and rebuilt from the nominal pose -- never from the previous
        # step's output, which is what made this accumulate.
        axis = np.asarray(self._canonical, dtype=np.float64)[:, 1]
        offset = float((self._pick_center - 0.5 * (first + second)) @ axis)
        self._pick_eef = self._pick_eef_nominal + offset * axis

    def _insertion_opening(self, pitch: float) -> float:
        """The jaw command that lays the two fingertips on the row's **outer faces**.

        The command is not a distance, and treating it as one is the bug this replaces.
        Measured on the dense Panda scene by sweeping it and reading the two finger
        geometries' separation:

            opening 0.000 ->  24.92 mm      0.400 ->  55.38 mm      0.800 ->  87.38 mm
            opening 0.200 ->  39.39 mm      0.508 ->  64.05 mm      1.000 -> 103.38 mm

        which is exactly affine -- ``separation = 2 * |finger y| + 2 * jaw_travel * opening``
        -- and **both** terms come from the model rather than from a constant.  On this
        scene the fingertips sit 23.92 mm apart at zero travel and travel 2 x 40 mm, so the
        map is ``23.92 + 80.0 * opening`` mm.  The old code commanded
        ``(5 * pitch - pad) / 0.085``: no constant term at all, and 85 mm instead of
        ``2 * GRIPPER_RANGE_M`` = 80.  Compounded, it missed the wanted separation by
        **5.4 mm -- 2.7 mm of interference on each side**.

        The hardcoded 6.35 mm it subtracted was also not a pad thickness: it is numerically
        the **Cookie** thickness (6.33 mm), and the quantity the five-Cookie row needs taken
        off is exactly that.

        Note that the fingertips are 27.13 mm thick here (`PAD_THICKNESS_M`, read from the
        model) against a 2.5 mm row gap: a fingertip cannot enter the groove at all, and
        `left_finger_pad_half_thickness_m` does not change that, because it only reaches
        A3's own Robotiq pads (see `model._add_gripper`).  That is what makes this a
        contact-chain grasp over the row's ends rather than an insertion into the gaps.
        Where a pad *is* thinner than the row gap -- A3's own gripper, at 1.00 mm against
        2.50 mm -- the documented placement stands and the pad centres go **into** the two
        grooves, which wants the wider separation `5 * pitch`.
        """

        cookie_thickness = 2.0 * float(np.asarray(self.env.COOKIE_HALF_SIZE).ravel()[1])
        finger_thickness = float(self.env.PAD_THICKNESS_M)
        closed = self._pad_separation(0.0)
        per_unit = self._pad_separation(1.0) - closed
        groove = pitch - cookie_thickness
        # **The row's outer faces are `4 * pitch + cookie_thickness` apart, not
        # `5 * pitch - cookie_thickness`.**  Five Cookies stand at `c, c+p, ... c+4p`, so the
        # span from the first one's outer face to the last one's is `4p + t`; `5p - t` is a
        # different quantity that happens to be close, and using it put the two fingertips
        # **0.9 mm inside the end Cookies** -- measured, the pads settled at y 0.0110 and
        # 0.0780 against a row whose outer faces are at 0.0245 and 0.0661, an outer pad force
        # of **125.6 N in a single step** and `cookie_6` gone at **160.8 m/s**.
        #
        # Which of the two placements is meant is decided by whether a fingertip fits a groove
        # at all, not by which span is larger:
        #
        # * it fits (A3's own gripper, 1.00 mm into 2.50 mm) -- the pads are **centred in the
        #   two grooves**, `5 * pitch` apart, which already leaves `(groove - finger) / 2` of
        #   room on each side and needs no allowance;
        # * it does not (LIBERO's Panda, 27.13 mm against 2.50 mm) -- they stand **on the
        #   row's outer faces**, and there the allowance matters: aiming at exactly the outer
        #   face is contact by construction, and the measured lateral error there is 0.8 mm,
        #   so `DESCEND` jammed at 20-130 N before it had finished descending.
        if finger_thickness < groove:
            span = 5.0 * pitch
        else:
            span = (
                4.0 * pitch + cookie_thickness + finger_thickness
                + 2.0 * self.profile.insert_clearance_m
            )
        return float(np.clip((span - closed) / per_unit, 0.0, 1.0))

    def _place_pose(self, clearance=0.0, column_index=None):
        rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        column_x = np.unique(slots[:, 0])[
            self.batch_index if column_index is None else column_index
        ]
        center = np.array(
            [
                column_x,
                slots[np.isclose(slots[:, 0], column_x), 1].mean(),
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
            return self._stay_command()

    def _act_step(self, observation=None, task=""):
        del observation, task
        if self.finished:
            return self._stay_command()
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
            return self._stay_command()
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
            if self._preclose_arrived(actual_opening):
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
                return self._stay_command()
            self._align_insertion()
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
                # **No position floor here.**  The loop's entry condition is a *force* and its
                # exit is a force, but it used to stop command from going below 0.28 -- and
                # 0.28 is where the force is capped, because a blocked finger pushes with
                # `kp * (command - achieved)`.  Measured on `a3_cookie_batch`: the row is
                # squeezed from a 35.3 mm span to 25.3 mm, the pads stop at 0.308 of travel,
                # the command sits on the 0.28 floor and the chain reads **1.44 N** -- which is
                # exactly `1200 N/m * 1.2 mm`, the whole 0.028 of remaining error.  The 2.0 N
                # the phase asks for needs 1.67 mm more, i.e. a command of 0.269, so the floor
                # made its own success criterion unreachable.  The closed end of the jaw is
                # 0.0, and the loop is already bounded where it should be: it stops closing at
                # 2.5 N and `_insertion_jammed` fails the run at 8 N.
                self._opening = max(0.0, self._opening - close_rate)
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
                # Same reasoning as `CLOSE`: a position floor on a force-driven squeeze caps
                # the force at `kp * (floor - achieved)` and can never reach its own target.
                self._opening = max(0.0, self._opening - 0.001)
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
                return self._stay_command()
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
                            "lifted": 5,
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
                return self._stay_command()
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
                return self._stay_command()
            if self.phase_steps >= self.RELEASE_SETTLE_STEPS and self._stable == 0:
                self._fail(
                    f"batch {self.batch_index + 1} never settled in the box: "
                    f"all contained={all_inside}, pad still on the batch={not released}"
                )
                return self._stay_command()
            # The first batch only has to be released; the second is the one the
            # scene's success test watches, so it is the one that has to hold for
            # the full success window.
            required_stable = (
                self.env.task_config.success_hold_steps
                if self.batch_index == 1
                else self.profile.release_stable_first
            )
            if self._stable >= required_stable:
                self.batch_reports[-1]["released"] = True
                self.completed_cookie_indices.extend(self.batch_indices)
                self.batch_index += 1
                # **The inter-batch perturbation, at the one moment it belongs.**  The
                # five are released and settled; the second batch is about to be
                # selected.  Between those two lines the first batch is as good as it
                # will ever be -- five upright at the slot centres -- and nudging them
                # *here* is what makes the second batch an insertion next to imperfect
                # neighbours rather than the same insertion the baseline trains.  The
                # maxima are the modest ones (see `_prefill_first_batch` for the hard
                # draw): a lean up to 3 degrees, a yaw to 0.8, a shift to 0.15 mm --
                # enough to read, small enough that the group absorbs it by shifting
                # within the box's margin rather than by knocking a neighbour over
                # (the upright tolerance is 15 degrees, and the perturbation has to
                # stay a long way under it or the episode fails its own contract).
                if (
                    getattr(self.env, "second_batch_mode", "full") == "perturb"
                    and self.batch_index == 1
                ):
                    # One draw for the whole group, for the same reason the prefill
                    # draws once: the settled group has zero clearance
                    # cookie-to-cookie, and independently drawn leans
                    # interpenetrate the neighbours.  See
                    # `perturb_settled_cookies` and `_prefill_first_batch`.
                    rng = self.env.np_random
                    self.env.perturb_settled_cookies(
                        tuple(self.completed_cookie_indices[-5:]),
                        lean_azimuth_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
                        lean_rad=float(rng.uniform(0.0, np.deg2rad(3.0))),
                        yaw_rad=float(rng.uniform(-np.deg2rad(0.8), np.deg2rad(0.8))),
                        max_jitter_m=0.00015,
                    )
                self._advance(
                    CookiePhase.DONE if self.batch_index == 2 else CookiePhase.SELECT_COOKIE
                )
            return self._stay_command()
        self._fail("unsupported batch phase")
        return self._stay_command()
