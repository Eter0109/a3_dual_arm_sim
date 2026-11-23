"""Opt-in five-Cookie experts for the composable randomization scenes.

Only the simulator expert uses object geometry and contacts. No object poses,
collision flags, or physics state are changed by the controller. Older single
box experts and their variable-grasp instructions remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
from a3_dual_arm_sim.controllers.expert import CookiePhase
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert
from a3_dual_arm_sim.controllers.scene_motion import (
    plan_clearance_path, follow_clearance_path, insertion_reachable,
)
from scipy.spatial.transform import Rotation


class SceneBatchMixin:
    """Manage the source inventory, target boxes and bounded physical recovery."""

    MAX_GRASP_RETRIES = 2

    @property
    def _batches_per_box(self):
        return self.env.task_config.required_cookies // 5

    @property
    def num_batches(self):
        return self._batches_per_box * len(self.env.target_box_bodies)

    @property
    def _target_batch_index(self):
        return self.batch_index % self._batches_per_box

    @property
    def _target_box_index(self):
        return min(self.batch_index // self._batches_per_box, len(self.env.target_box_bodies) - 1)

    @property
    def _is_followup_batch(self):
        if not self._selected_group:
            return False
        start = self._selected_group[0]
        # A group after a genuine opening in *its own* column is a follow-up,
        # even if a gap was already present at reset. A new column isn't.
        return start % 20 != 0 and not self.env.privileged_cookie_in_source(start - 1)

    @property
    def status(self):
        return (
            f"phase={self.phase.name} grasp={min(self.batch_index + 1, self.num_batches)}"
            f"/{self.num_batches} target_box={self._target_box_index + 1} "
            f"cookies={self.batch_indices} confirmed={len(self.completed_cookie_indices)}"
            f"/{self.env.target_cookie_goal}"
        )

    def reset(self, context=None):
        self._tb_id = self.env.target_box_bodies[0]
        self._selected_group = None
        self._selection_global_batch = None
        self._assigned_target_bodies = {}
        self._scene_retries = 0
        self.scene_recovery_reports = []
        self.scene_motion_reports = []
        self._placement_flip = None
        self._placement_relief_deg = 0.0
        self._scene_held_rotations = None
        self._placement_offsets = None
        self._placement_recalibrated = False
        self._placement_realignments = 0
        self._placement_z_correction = 0.0
        self._release_from_geometry = False
        self._release_pose_cache = None
        self._travel_stage = None
        self._safe_transfer_path = None
        self._safe_approach_path = None
        self._safe_transfer_index = self._safe_approach_index = 0
        self._scene_ready = False
        super().reset(context)
        self._preferred_source_column = self.selected_source_column_index
        self._current_source_column = self._preferred_source_column
        self._starting_inventory = self.env.initial_cookie_count
        self._tilted_ids = frozenset(self.env._scene_randomizer.tilted_group)
        self._batch_pinch_height_m = 0.010
        r = self.data.site_xmat[self._l_site].reshape(3, 3)
        base = self.model.geom("L_2f85_base_collision").id
        self._base_offset_local = r.T @ (self.data.geom_xpos[base] - self.data.site_xpos[self._l_site])
        self._base_rotation_local = r.T @ self.data.geom_xmat[base].reshape(3, 3)
        self._base_size = self.model.geom_size[base].copy()
        self._scene_ready = True

    def _validate_target_workspace(self):
        original = self._tb_id
        try:
            for body in self.env.target_box_bodies:
                self._tb_id = body
                super()._validate_target_workspace()
        finally:
            self._tb_id = original

    def _candidate_batches(self):
        if self._selected_group is not None:
            if all(self.env.privileged_cookie_in_source(i) for i in self._selected_group):
                yield list(self._selected_group)
            return

        groups = self.env._scene_randomizer.groups
        completed = set(self.completed_cookie_indices)
        available = [
            group for group in groups
            if not completed.intersection(group)
            and all(self.env.privileged_cookie_in_source(i) for i in group)
        ]
        if not available:
            return
        # If the scene contains an intentional lean, transfer that group too;
        # don't call a scene compatible merely by choosing all upright Cookies.
        tilted = next((g for g in available if set(g) == self._tilted_ids), None)
        if tilted is not None:
            group = tilted
        else:
            group = min(available, key=lambda g: (
                g[0] // 20 != self._current_source_column,
                abs(g[0] // 20 - self._preferred_source_column),
                g[0],
            ))
        self._selected_group = tuple(group)
        self._current_source_column = self.selected_source_column_index = group[0] // 20
        yield list(group)

    def _select_batch(self):
        if self._selection_global_batch != self.batch_index:
            self._selection_global_batch = self.batch_index
            self._selected_group = None
            self._base_push_attempts = 0
            self._scene_retries = 0
            self._placement_flip = None
            self._placement_relief_deg = 0.0
            self._scene_held_rotations = None
            self._placement_offsets = None
            self._placement_recalibrated = False
            self._placement_realignments = 0
            self._placement_z_correction = 0.0
            self._release_from_geometry = False
            self._release_pose_cache = None
            self._travel_stage = None
            self._safe_transfer_path = None
            self._safe_approach_path = None
            self._safe_transfer_index = self._safe_approach_index = 0
            self._held_rotation_batch = None
            self._tb_id = self.env.target_box_bodies[self._target_box_index]
            orders = getattr(self.env, "demonstration_box_column_orders", None)
            if orders:
                self.env.demonstration_column_order = orders[self._target_box_index]
        # Resolve the source group before the parent chooses the wrist pitch,
        # grip limits, and whether to use the follow-up alignment behavior.
        candidate = next(self._candidate_batches(), None)
        if candidate is None:
            raise RuntimeError("no complete available five-Cookie group in the source inventory")
        self._max_lift_speed = 0.0012 if self.env.task_config.required_cookies == 20 else 0.0020
        self._transport_speed = 0.0042 if self.selected_source_column_index >= 2 else 0.0048
        self._batch_pinch_height_m = 0.010
        self.min_grip_opening, self.secure_grip_opening = 0.28, 0.33
        self.release_opening = 0.49
        self.target_grip_force, self.stable_grip_force = 1.3, 1.0
        super()._select_batch()

        threshold = self.env.scene_options.cookie_crush_force_n
        if threshold is not None:
            self.target_grip_force = min(self.target_grip_force, threshold * 0.50)
            self.stable_grip_force = min(self.stable_grip_force, threshold * 0.38)
        if self.phase is CookiePhase.APPROACH and self._scene_retries:
            # Recompute the contact pose from the actual Cookie positions and
            # make a small, bounded depth correction after an unsuccessful grip.
            depth = 0.0015 * self._scene_retries
            delta = self._grasp_rotation[:, 1] * depth
            self._pick_eef += delta
            self._high_eef += delta
            self._approach_eef += delta
        self._safe_approach_path = None
        self._safe_approach_index = 0

    def _is_intentional_lean(self):
        return bool(self._selected_group and set(self._selected_group) == self._tilted_ids)

    def _rear_neighbour_present(self, index):
        return self.env.privileged_cookie_in_source(index)

    def _align_grasp_requested(self):
        return (
            (self._is_intentional_lean() or self._scene_retries > 0)
            and self._grasp_tilt_deg > 2.5
        ) or super()._align_grasp_requested()

    def _anchor_aligned_gap(self):
        # A deliberate sloped group can have overlapping full-body projected
        # extents although the bevel leaves a usable top-entry gap.
        return not self._is_intentional_lean() and self._scene_retries == 0

    def _straighten_requested(self):
        return not self._is_intentional_lean() and super()._straighten_requested()

    def _grasp_pitch_for_column(self, column_rank, pitch_deg):
        if column_rank == 3:
            # Reuse the native far-column, base-clearance-aware IK search for
            # both box sizes, including groups newly exposed by a source gap.
            return A3Cookie2x10Expert._grasp_pitch_for_column(self, column_rank, pitch_deg)
        return super()._grasp_pitch_for_column(column_rank, pitch_deg)

    def _insertion_jammed(self, forces, limit=None):
        threshold = self.env.scene_options.cookie_crush_force_n
        if threshold is not None:
            limit = min(self.MAX_INSERTION_FORCE_N if limit is None else limit, threshold * 0.85)
        return super()._insertion_jammed(forces, limit)

    def _can_tighten_grip(self):
        threshold = self.env.scene_options.cookie_crush_force_n
        return threshold is None or max(self._total_pad_forces) < threshold * 0.70

    def _released_cookie_in_target(self, index):
        return self.env.privileged_cookie_in_target(
            index, target_body=self._assigned_target_bodies.get(index, self._tb_id)
        )

    def _complete_batch_report(self):
        super()._complete_batch_report()
        report = self.batch_reports[-1]
        report.update(
            target_box=self._target_box_index + 1,
            target_batch=self._target_batch_index + 1,
            source_column=self.selected_source_column_index + 1,
            planned_count=5,
            intentional_lean=self._is_intentional_lean(),
            grasp_retries=self._scene_retries,
            placement_yaw_flipped=self._placement_flip,
            placement_relief_deg=self._placement_relief_deg,
            placement_realignments=self._placement_realignments,
        )
        for index in self.batch_indices:
            self._assigned_target_bodies[index] = self._tb_id

    def _target_column(self):
        from .placement_order import ordered_column

        local = self._target_batch_index // 2 if self._batches_per_box == 4 else self._target_batch_index
        return ordered_column(self.env, local)

    def _needs_planned_transfer(self):
        return len(self.env.target_box_bodies) > 1 or self.selected_source_column_index >= 2

    def _fit_held_group(self, center, tool_rotation, clearance):
        """Compute a floor-supported fit from the measured stack, not one slab."""
        box_r = self.data.xmat[self._tb_id].reshape(3, 3)
        box_pos = self.data.xpos[self._tb_id]
        local_center = box_r.T @ (center - box_pos)
        if self._target_column() == 1:
            local_center[0] -= 0.002  # Recover clearance at the outer wall.
        local_tool = box_r.T @ tool_rotation
        offsets = self._held_offsets if self._placement_offsets is None else self._placement_offsets
        offset = (offsets - self._group_offset) @ local_tool.T
        local_rot = np.einsum("ij,njk->nik", local_tool, self._scene_held_rotations)
        extent = np.abs(local_rot) @ self.env.COOKIE_HALF_SIZE
        lo = np.min(offset - extent, axis=0)
        hi = np.max(offset + extent, axis=0)
        room = self.env.TARGET_INNER_HALF_SIZE - 0.001
        lower, upper = -room - lo[:2], room - hi[:2]

        # Use the actual previously deposited footprints to leave an insertion
        # gap in the other column or half-row. Never jump an occupied slot.
        safe_lower, safe_upper = lower.copy(), upper.copy()
        column = self._target_column()
        columns = np.unique(np.asarray(self.env.TARGET_SLOTS_LOCAL)[:, 0])
        for i, assigned in self._assigned_target_bodies.items():
            if assigned != self._tb_id:
                continue
            p = self.env.privileged_cookie_target_position(i, target_body=self._tb_id)
            r = box_r.T @ self.data.xmat[self.env._cookie_bodies[i]].reshape(3, 3)
            e = np.abs(r) @ self.env.COOKIE_HALF_SIZE
            prior_col = int(np.argmin(np.abs(columns - p[0])))
            if prior_col != column:
                if column == 1:
                    safe_lower[0] = max(safe_lower[0], p[0] + e[0] - lo[0] + 0.0003)
                else:
                    safe_upper[0] = min(safe_upper[0], p[0] - e[0] - hi[0] - 0.0003)
            elif self._batches_per_box == 4 and self._target_batch_index % 2 == 1:
                safe_lower[1] = max(safe_lower[1], p[1] + e[1] - lo[1] + 0.0003)
        for axis in range(2):
            if safe_lower[axis] <= safe_upper[axis]:
                if axis == 0 and self._target_batch_index >= self._batches_per_box // 2:
                    # Balance the remaining clearance to an already filled
                    # column and the outer wall instead of hugging either.
                    midpoint = (safe_lower[axis] + safe_upper[axis]) / 2
                    local_center[axis] = np.clip(midpoint, local_center[axis] - 0.004,
                                                local_center[axis] + 0.004)
                local_center[axis] = np.clip(local_center[axis], safe_lower[axis], safe_upper[axis])
            elif lower[axis] <= upper[axis]:
                local_center[axis] = np.clip(local_center[axis], lower[axis], upper[axis])
        cfg = self.env.config.cookie_transfer
        floor = cfg.target_floor_z_m + cfg.bin_wall_thickness_m / 2
        # The lowest actual held Cookie bottom approaches the floor before
        # the jaws open; this removes the unnecessary release drop height.
        local_center[2] = floor - lo[2] + 0.0007 + clearance
        if clearance == 0.0:
            local_center[2] += self._placement_z_correction
        return box_pos + box_r @ local_center

    def _capture_placement_geometry(self):
        actual_r = self.data.site_xmat[self._l_site].reshape(3, 3)
        actual_p = self.data.site_xpos[self._l_site]
        cookie_r = self.data.xmat[[self.env._cookie_bodies[i] for i in self.batch_indices]].reshape(-1, 3, 3)
        self._scene_held_rotations = np.einsum("ij,njk->nik", actual_r.T, cookie_r)
        self._placement_offsets = (self._positions() - actual_p) @ actual_r
        self._group_offset = self._placement_offsets.mean(axis=0)
        u, _, vt = np.linalg.svd(self._scene_held_rotations.mean(axis=0))
        u[:, -1] *= np.linalg.det(u @ vt)
        self._held_cookie_rotation_local = u @ vt
        self._held_rotation_batch = tuple(self.batch_indices)
        self._placement_recalibrated = True

    def _place_pose(self, clearance=0.0, column_index=None):
        planned_loaded = (
            column_index is None and self.batch_indices and self._needs_planned_transfer()
            and self.phase in (CookiePhase.MOVE_TO_SLOT, CookiePhase.DESCEND_TO_PLACE, CookiePhase.OPEN)
        )
        secondary_loaded = planned_loaded and self._target_box_index > 0
        if secondary_loaded and self.phase is CookiePhase.DESCEND_TO_PLACE and not self._placement_recalibrated:
            # The loaded stack may move millimeters relative to the fingers
            # during a long turn. Measure it again above the target instead
            # of reusing the grasp-time offset for precision insertion.
            self._capture_placement_geometry()
        if planned_loaded and self._scene_held_rotations is None:
            actual_r = self.data.site_xmat[self._l_site].reshape(3, 3)
            cookie_r = self.data.xmat[[self.env._cookie_bodies[i] for i in self.batch_indices]].reshape(-1, 3, 3)
            self._scene_held_rotations = np.einsum("ij,njk->nik", actual_r.T, cookie_r)
            # Orthogonal mean balances the full five-piece stack's upright
            # direction instead of forcing the middle Cookie to define it.
            if secondary_loaded:
                u, _, vt = np.linalg.svd(self._scene_held_rotations.mean(axis=0))
                u[:, -1] *= np.linalg.det(u @ vt)
                self._held_cookie_rotation_local = u @ vt
                self._held_rotation_batch = tuple(self.batch_indices)
        position, rotation = super()._place_pose(clearance, column_index)
        if not planned_loaded:
            if column_index is None and clearance == 0.0 and self._placement_z_correction:
                position = position + self.data.xmat[self._tb_id].reshape(3, 3)[:, 2] * self._placement_z_correction
            return position, rotation

        # A 180 degree yaw of a rectangular Cookie is physically equivalent.
        # On side-facing boxes, choose the reachable orientation with the
        # smaller rotation from the actual loaded wrist.
        center = position + rotation @ self._group_offset
        box_r = self.data.xmat[self._tb_id].reshape(3, 3)
        flip = box_r @ np.diag((-1.0, -1.0, 1.0)) @ box_r.T
        if self._placement_flip is None:
            actual_r = self.data.site_xmat[self._l_site].reshape(3, 3)
            options = []
            for sign, r in ((False, rotation), (True, flip @ rotation)):
                fitted = self._fit_held_group(center, r, clearance) if secondary_loaded else center
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, r.ravel())
                goal = fitted - r @ self._group_offset
                q = self.env.solve_ik(goal, quat, self.env.last_applied_action[:7])
                work = self.env._ik._work
                work.qpos[self._l_qpos] = q
                mujoco.mj_kinematics(self.model, work)
                pos_err = np.linalg.norm(work.site_xpos[self._l_site] - goal)
                achieved, error = np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(achieved, work.site_xmat[self._l_site])
                mujoco.mju_subQuat(error, quat, achieved)
                angle = np.arccos(np.clip((np.trace(actual_r.T @ r) - 1) / 2, -1, 1))
                options.append((pos_err > 0.004 or np.linalg.norm(error) > 0.08,
                                angle + pos_err * 100 + np.linalg.norm(error), sign))
            self._placement_flip = min(options)[2]
        if self._placement_flip:
            rotation = flip @ rotation
        if self._placement_relief_deg:
            relief = Rotation.from_euler("y", self._placement_relief_deg, degrees=True).as_matrix()
            rotation = box_r @ relief @ box_r.T @ rotation
        if secondary_loaded:
            center = self._fit_held_group(center, rotation, clearance)
        elif clearance == 0.0:
            center = center + box_r[:, 2] * self._placement_z_correction
        return center - rotation @ self._group_offset, rotation

    def _placement_reached(self, position, rotation, reached):
        # Preserve the ordinary first-box path unless it actually stalls.
        # The advanced source-gap case exposed the same height-tracking
        # problem on the primary tray, without any large-yaw transport.
        primary = self._target_box_index == 0
        if primary and self.phase_steps < 60 and not self._placement_z_correction and not self._placement_realignments:
            return reached
        # If all bottom surfaces are already supported, small elastic contact
        # error in the final commanded height must not force a harder press.
        box_r = self.data.xmat[self._tb_id].reshape(3, 3)
        pos = (self._positions() - self.data.xpos[self._tb_id]) @ box_r
        matrices = self.data.xmat[[self.env._cookie_bodies[i] for i in self.batch_indices]].reshape(-1, 3, 3)
        extent = np.abs(np.einsum("ij,njk->nik", box_r.T, matrices)) @ self.env.COOKIE_HALF_SIZE
        cfg = self.env.config.cookie_transfer
        bottom = pos[:, 2] - extent[:, 2] - cfg.target_floor_z_m - cfg.bin_wall_thickness_m / 2
        actual = self.data.site_xpos[self._l_site]
        actual_r = self.data.site_xmat[self._l_site].reshape(3, 3)
        angle = np.arccos(np.clip((np.trace(actual_r.T @ rotation) - 1) / 2, -1, 1))
        footprint_ok = np.all(np.abs(pos[:, :2]) + extent[:, :2]
                              <= self.env.TARGET_INNER_HALF_SIZE + 0.0005)
        upright_ok = np.min(np.abs(matrices[:, 2, 2])) >= np.cos(np.deg2rad(14.0))
        columns = np.unique(np.asarray(self.env.TARGET_SLOTS_LOCAL)[:, 0])
        column_ok = abs(float(pos[:, 0].mean()) - columns[self._target_column()]) <= 0.010
        close = (np.linalg.norm((actual - position)[:2]) < 0.0025
                 and abs(actual[2] - position[2]) < 0.004 and angle < 0.06
                 and np.linalg.norm(self.data.qvel[:7]) < 0.25)
        if primary and self.phase_steps >= 60 and column_ok:
            close = close or (np.linalg.norm((actual - position)[:2]) < 0.005
                              and abs(actual[2] - position[2]) < 0.010 and angle < 0.15
                              and np.linalg.norm(self.data.qvel[:7]) < 0.25)
        supported = np.max(bottom) < 0.0035 and np.min(bottom) > -0.002
        if primary:
            # If the wrist has reached a kinematic limit, the actual Cookie
            # geometry is still the task goal. The advanced-gap failure had
            # all five slabs upright, correctly inside their slots, and only
            # ~3 mm above the floor, but the nominal wrist pose was not reached.
            # Reorienting a held group beside a completed row caused collisions.
            slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
            ys = slots[np.isclose(slots[:, 0], columns[self._target_column()]), 1]
            if self._batches_per_box == 4:
                ys = ys[list(self.plan.target_rows(self._target_batch_index))]
            row_tolerance = 0.005 if self._batches_per_box == 2 else 0.012
            row_ok = len(ys) == len(pos) and np.max(np.abs(np.sort(pos[:, 1]) - np.sort(ys))) <= row_tolerance
            stable_actual = np.linalg.norm(self.data.qvel[:7]) < 0.25
            if footprint_ok and upright_ok and column_ok and row_ok and stable_actual:
                if supported:
                    self._release_from_geometry = True
                    self.scene_motion_reports.append({
                        "batch": self.batch_index + 1, "kind": "primary_measured_release",
                        "bottom_clearance_m": bottom.tolist(), "column_center_m": float(pos[:, 0].mean()),
                    })
                    return True
                # The following bounded height feedback is allowed only
                # when the measured held group already fits the correct row.
                close = True
            else:
                if self.phase_steps >= 180:
                    self._fail("primary placement still lacks a supported, slot-aligned Cookie group")
                return False
        if footprint_ok and upright_ok and column_ok and supported and close:
            return True
        # The actuator/IK tracking error can leave an otherwise correctly
        # aligned stack several millimeters above the floor. Close the
        # measured vertical gap gradually while keeping the grip and XY fixed.
        # Raising and retrying the same nominal pose cannot fix this error.
        if (reached or close) and footprint_ok and upright_ok and column_ok:
            high = float(np.min(bottom))
            threshold = self.env.scene_options.cookie_crush_force_n
            safe_force = 6.0 if threshold is None else min(6.0, threshold * 0.65)
            if high > 0.002 and max(self._total_pad_forces) < safe_force and self._placement_z_correction > -0.008:
                correction = float(np.clip(high - 0.0008, 0.0005, 0.0025))
                self._placement_z_correction = max(-0.008, self._placement_z_correction - correction)
                self.scene_motion_reports.append({
                    "batch": self.batch_index + 1, "kind": "seat_height_feedback",
                    "measured_bottom_m": bottom.tolist(), "correction_m": self._placement_z_correction,
                })
                return False
        can_realign = reached or close or (self.phase_steps >= 100 and np.linalg.norm(actual - position) < 0.018)
        if primary:
            # Do not blindly reposition at the bottom of a partly filled
            # primary box. Preserve the previously released row on failure.
            return False
        if can_realign and self._placement_realignments < 2:
            self._placement_realignments += 1
            self._placement_z_correction = 0.0
            self._place_recover_pose = actual.copy() + [0, 0, 0.035]
            self._place_recover_rotation = actual_r.copy()
            self.scene_motion_reports.append({
                "batch": self.batch_index + 1, "kind": "placement_realign",
                "attempt": self._placement_realignments, "footprint_ok": bool(footprint_ok),
                "upright_ok": bool(upright_ok), "bottom_clearance_m": bottom.tolist(),
            })
            self._advance(CookiePhase.REPOSITION_PLACE)
        elif can_realign and self._placement_realignments >= 2:
            self._fail("placement geometry did not settle inside the target after two realignments")
        return False

    def _release_pose(self):
        if self._target_box_index == 0 and not self._placement_z_correction and not self._release_from_geometry:
            return super()._release_pose()
        if self._release_pose_cache is None:
            self._release_pose_cache = (self.data.site_xpos[self._l_site].copy(),
                                        self.data.site_xmat[self._l_site].reshape(3, 3).copy())
        return self._release_pose_cache

    def _release_hold_steps(self):
        return 15 if (self._target_box_index > 0 or self._placement_z_correction or self._release_from_geometry) else super()._release_hold_steps()

    def _servo_transport(self, position, rotation, opening, speed):
        if not self._needs_planned_transfer():
            return super()._servo_transport(position, rotation, opening, speed)
        if self._safe_transfer_path is None:
            original_flip = bool(self._placement_flip)
            # Preserve every strict-route success. Only after every original
            # orientation/route has failed try clearance-checked, less exact
            # *intermediate* Cartesian samples and an inward withdrawal.
            for relaxed in (False, True):
                for relief in (0.0, -6.0, 6.0, -10.0, 10.0):
                    for flip in (original_flip, not original_flip):
                        self._placement_flip, self._placement_relief_deg = flip, relief
                        wanted_p, wanted_r = self._place_pose(0.060)
                        cookie_r = np.einsum("ij,njk->nik", wanted_r, self._scene_held_rotations)
                        if np.min(np.abs(cookie_r[:, 2, 2])) < np.cos(np.deg2rad(14.5)):
                            continue
                        path = plan_clearance_path(self, wanted_p, wanted_r, loaded=True,
                                                   allow_intermediate_relax=relaxed)
                        if path is not None:
                            bottom_p, bottom_r = self._place_pose(0.0)
                            if not insertion_reachable(self, path.joints[-1], wanted_p, bottom_p, bottom_r):
                                continue
                            self._safe_transfer_path = path
                            position, rotation = wanted_p, wanted_r
                            break
                    if self._safe_transfer_path is not None:
                        break
                if self._safe_transfer_path is not None:
                    break
            if self._safe_transfer_path is None:
                self.scene_motion_reports.append({"batch": self.batch_index + 1, "kind": "loaded_path_rejected",
                                                  "checks": getattr(self, "_last_path_failures", [])})
                raise RuntimeError("no clearance-safe loaded path to the selected target")
            self.scene_motion_reports.append({
                "batch": self.batch_index + 1, "kind": "loaded_transfer",
                "route": path.route, "min_clearance_m": path.minimum_z - path.obstacle_z,
                "waypoints": len(path.joints), "yaw_flipped": self._placement_flip,
                "relief_deg": self._placement_relief_deg,
            })
        return self._follow_planned(position, rotation, opening, loaded=True)

    def _servo_approach(self, position, rotation, opening, speed):
        if self._target_box_index == 0:
            return super()._servo_approach(position, rotation, opening, speed)
        if self._safe_approach_path is None:
            path = plan_clearance_path(self, position, rotation, loaded=False)
            if path is None:
                path = plan_clearance_path(self, position, rotation, loaded=False, allow_intermediate_relax=True)
            if path is None:
                self.scene_motion_reports.append({"batch": self.batch_index + 1, "kind": "empty_path_rejected",
                                                  "checks": getattr(self, "_last_path_failures", [])})
                raise RuntimeError("no clearance-safe empty approach to the source group")
            self._safe_approach_path = path
            self.scene_motion_reports.append({
                "batch": self.batch_index + 1, "kind": "empty_approach", "route": path.route,
                "min_clearance_m": path.minimum_z - path.obstacle_z, "waypoints": len(path.joints),
            })
        return self._follow_planned(position, rotation, opening, loaded=False)

    def _follow_planned(self, position, rotation, opening, *, loaded):
        path = self._safe_transfer_path if loaded else self._safe_approach_path
        index = self._safe_transfer_index if loaded else self._safe_approach_index
        was_moving = index < len(path.joints)
        action, index = follow_clearance_path(self, path, index, opening, loaded=loaded)
        if loaded:
            self._safe_transfer_index = index
        else:
            self._safe_approach_index = index
        if was_moving:
            return action, False
        actual = self.data.site_xpos[self._l_site]
        actual_r = self.data.site_xmat[self._l_site].reshape(3, 3)
        angle = np.linalg.norm(Rotation.from_matrix(actual_r.T @ rotation).as_rotvec())
        error = np.linalg.norm(actual - position)
        reached = error < 0.004 and angle < 0.065 and np.linalg.norm(self.data.qvel[:7]) < 0.30
        if not reached and error < 0.012 and angle < 0.12:
            # The preplanned path has finished. A small final feedback motion
            # compensates gravity sag; large errors never bypass the path.
            return self._servo(position, rotation, opening, speed=0.0015)
        return action, reached

    def _approach_ready(self, target, rotation, reached):
        if self._target_box_index > 0 and self._safe_approach_path is not None:
            return self._safe_approach_index >= len(self._safe_approach_path.joints) and reached
        return super()._approach_ready(target, rotation, reached)

    def _approach_timeout_steps(self):
        return 400 if self._target_box_index > 0 else super()._approach_timeout_steps()

    def _batch_phase_limit(self):
        if self.phase is CookiePhase.MOVE_TO_SLOT and self._needs_planned_transfer():
            return 700
        return super()._batch_phase_limit()

    def _try_grasp_recovery(self, reason):
        if (
            not self._scene_ready or self._scene_retries >= self.MAX_GRASP_RETRIES
            or self.phase not in (CookiePhase.APPROACH, CookiePhase.PRE_CLOSE, CookiePhase.DESCEND,
                                  CookiePhase.CLOSE, CookiePhase.LIFT)
            or not self._selected_group
            or not all(self.env.privileged_cookie_in_source(i) for i in self._selected_group)
            or self.env._scene_randomizer.crushed_ids
        ):
            return False
        self._scene_retries += 1
        self.scene_recovery_reports.append({
            "batch": self.batch_index + 1, "attempt": self._scene_retries,
            "phase": self.phase.name, "reason": reason, "cookies": list(self._selected_group),
        })
        self._recover_pose = self.data.site_xpos[self._l_site].copy() + [0, 0, 0.065]
        self._recover_rotation = self.data.site_xmat[self._l_site].reshape(3, 3).copy()
        self._advance(CookiePhase.RECOVER_GRASP)
        return True

    def _fail(self, reason):
        if not self._try_grasp_recovery(reason):
            super()._fail(reason)

    def _act_step(self, observation=None, task=""):
        if self.phase is CookiePhase.REPOSITION_PLACE:
            self.phase_steps += 1
            current_r = self.data.site_xmat[self._l_site].reshape(3, 3)
            offsets = (self._positions() - self.data.site_xpos[self._l_site]) @ current_r
            if np.max(np.linalg.norm(offsets - self._held_offsets, axis=1)) > 0.016:
                self._fail("Cookie slipped during placement realignment")
                return self._hold_command(self._opening)
            action, reached = self._servo(self._place_recover_pose, self._place_recover_rotation,
                                          self._opening, speed=0.002)
            if self.phase_steps > 180:
                self._fail("placement realignment could not lift clear")
            elif reached:
                self._capture_placement_geometry()
                self._advance(CookiePhase.DESCEND_TO_PLACE)
            return action
        if self.phase is CookiePhase.CLOSE and self.phase_steps >= 120:
            self._try_grasp_recovery("no stable five-Cookie contact chain")
        if self.phase is CookiePhase.RECOVER_GRASP:
            self.phase_steps += 1
            self._opening = min(1.0, self._opening + 0.020)
            action, reached = self._servo(self._recover_pose, self._recover_rotation, self._opening, speed=0.0025)
            if self.phase_steps > 160:
                super()._fail("physical grasp recovery could not retract")
            elif reached and self._opening >= 0.99:
                self._advance(CookiePhase.SELECT_COOKIE)
            return action
        previous_count = len(self.batch_reports)
        action = super()._act_step(observation, task)
        if len(self.batch_reports) > previous_count and self.env.task_config.required_cookies == 10:
            # The measured stack width is preferable to a fixed release gap
            # once groups have been grasped at different angles/locations.
            A3Cookie2x10Expert._set_release_opening(self)
            if self._target_box_index > 0:
                self.release_opening = max(self.release_opening, 0.46)
        if self.phase is CookiePhase.DONE and self.env.task_config.required_cookies == 10:
            self._verify_complete_fill()
        return action

    def _verify_complete_fill(self):
        counts = [sum(self.env.privileged_cookie_in_target(i, target_body=body)
                      for i in range(len(self.env.SOURCE_POSITIONS)))
                  for body in self.env.target_box_bodies]
        remaining = sum(self.env.privileged_cookie_in_source(i) for i in range(len(self.env.SOURCE_POSITIONS)))
        expected = self._starting_inventory - self.env.target_cookie_goal
        if counts != [self.env.task_config.required_cookies] * len(counts) or remaining != expected:
            self._fail(f"scene fill integrity failed: target counts={counts}, source cookies={remaining}/{expected}")


@dataclass
class A3Scene2x5Expert(SceneBatchMixin, A3VariedColumnBatchExpert):
    requested_source_column_index: int | None = 0


@dataclass
class A3Scene2x10Expert(SceneBatchMixin, A3Cookie2x10Expert):
    """Repeat the native five/five placement plan for every active target box."""

    first_grasp: int | str = 5

    def reset(self, context=None):
        if self.first_grasp != 5:
            raise ValueError("the extended five-group scene expert requires first_grasp=5")
        super().reset(context)
