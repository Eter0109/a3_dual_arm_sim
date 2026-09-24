"""Contact-driven five-Cookie grasp through bevels; privileged, not a VLA policy.

Objects are never attached, teleported, or moved by this controller. Each batch
must form a pad--Cookie contact chain, lift together, and be released into the box.
Outputs normalized delta TCP actions directly to the simulation environment.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .batch_expert import A3CookieBatchExpert
from .expert import CookiePhase


class A3SameColumnBatchExpert(A3CookieBatchExpert):
    MAX_INSERTION_FORCE_N = 20.0
    MAX_RECOVERABLE_TILT_DEG = 65.0

    def reset(self, context=None):
        super().reset(context)
        self._base_push_attempts = 0
        self._rim_contact_stop = False

    def _insertion_jammed(self, forces, limit=None):
        threshold = self.MAX_INSERTION_FORCE_N if limit is None else limit
        self._jam_count = self._jam_count + 1 if max(forces) > threshold else 0
        return self._jam_count >= 3

    def _approach_ready(self, target, rotation, reached):
        if reached:
            return True
        if not getattr(self, "_wait_for_approach", False):
            return self.phase_steps >= 15
        position_error = np.linalg.norm(
            np.asarray(target) - self.data.site_xpos[self._l_site]
        )
        desired_quat = np.empty(4)
        mujoco.mju_mat2Quat(desired_quat, np.asarray(rotation).ravel())
        actual_quat = np.empty(4)
        mujoco.mju_mat2Quat(actual_quat, self.data.site_xmat[self._l_site])
        rotation_error = np.empty(3)
        mujoco.mju_subQuat(rotation_error, desired_quat, actual_quat)
        return position_error < 0.008 and np.linalg.norm(rotation_error) < 0.14

    def _candidate_batches(self):
        """Take the exposed end of the same source column for both batches."""
        source = np.asarray(self.env.SOURCE_POSITIONS)
        columns = getattr(self, "_column_order", sorted(set(source[:, 0])))
        if self.completed_cookie_indices:
            columns = [source[self.completed_cookie_indices[0], 0]]
        for x in columns:
            indices = np.flatnonzero(np.isclose(source[:, 0], x))
            remaining = sorted(
                (i for i in indices.tolist() if i not in self.completed_cookie_indices),
                key=lambda i: source[i, 1],
            )
            if len(remaining) < 5:
                continue
            batch = remaining[:5]
            if all(self.env.privileged_cookie_in_source(i) for i in batch):
                yield batch

    @property
    def _approach_q(self) -> np.ndarray:
        return self.env.solve_ik(
            self._approach_eef, self._grasp_quat, self.env.current_joint_action[:7]
        )

    @property
    def _push_q(self) -> np.ndarray:
        return self.env.solve_ik(
            self._push_high, self._target_quat_canonical, self.env.current_joint_action[:7]
        )

    def _select_batch(self):
        self._rim_contact_stop = False
        self.batch_indices = next(self._candidate_batches(), [])
        if len(self.batch_indices) != 5:
            raise RuntimeError("the next five Cookies in the source column are unavailable")
        self.current_cookie_index = self.batch_indices[0]
        self.target_slot_index = self.batch_index
        positions = self._positions()
        rotations = np.stack(
            [self.data.xmat[self.env._cookie_bodies[i]].reshape(3, 3)
             for i in self.batch_indices]
        )
        self._rear_clearance_m = float("inf")
        neighbour = None
        neighbour_rotation = None
        next_index = self.batch_indices[-1] + 1
        if (
            self.batch_index == 1
            and next_index < len(self.env.SOURCE_POSITIONS)
            and np.isclose(
                self.env.SOURCE_POSITIONS[next_index][0],
                self.env.SOURCE_POSITIONS[self.batch_indices[-1]][0],
            )
        ):
            neighbour = self.env.cookie_positions[next_index]
            neighbour_rotation = self.data.xmat[
                self.env._cookie_bodies[next_index]
            ].reshape(3, 3)
            target_rear = positions[-1, 1] + np.abs(rotations[-1, 1]) @ self.env.COOKIE_HALF_SIZE
            neighbour_front = neighbour[1] - np.abs(neighbour_rotation[1]) @ self.env.COOKIE_HALF_SIZE
            self._rear_clearance_m = float(neighbour_front - target_rear)
        up = rotations[:, :, 2].mean(axis=0)
        up /= np.linalg.norm(up)
        self._grasp_tilt_deg = float(np.rad2deg(np.arccos(np.clip(up[2], -1, 1))))
        if self._grasp_tilt_deg > self.MAX_RECOVERABLE_TILT_DEG:
            raise RuntimeError(
                f"same-column batch tilt {self._grasp_tilt_deg:.1f} deg exceeds "
                f"{self.MAX_RECOVERABLE_TILT_DEG:.0f} deg"
            )
        self._grasp_rotation = self._canonical.copy()
        self._aligned_grasp = (
            self.batch_index == 1
            and self._base_push_attempts >= 2
            and self._grasp_tilt_deg > 5
            and self._rear_clearance_m < 0.0025
        )
        if self._aligned_grasp:
            # The five slabs share a fall direction. Descend along that
            # direction so the far jaw slides back towards the emptied gap,
            # rather than driving vertically into the following Cookie.
            angle = float(np.clip(np.arctan2(up[1], up[2]), -0.6, 0.6))
            c, s = np.cos(angle), np.sin(angle)
            fall_rotation = np.array(
                [[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float64
            )
            self._grasp_rotation = fall_rotation @ self._canonical
        pitch_by_column = getattr(self, "_column_tool_pitch_deg", None)
        if pitch_by_column is not None:
            source = np.asarray(self.env.SOURCE_POSITIONS)
            columns = sorted(set(source[:, 0]))
            column_x = source[self.batch_indices[0], 0]
            column_rank = next(
                index for index, x in enumerate(columns) if np.isclose(x, column_x)
            )
            pitch = np.deg2rad(pitch_by_column[column_rank])
            c, s = np.cos(pitch), np.sin(pitch)
            # Rotate around the jaw axis. The pads still close along the
            # Cookie row while the wrist reaches farther source columns.
            reach_rotation = np.array(
                [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64
            )
            self._grasp_rotation = reach_rotation @ self._grasp_rotation
        self._grasp_quat = np.empty(4)
        mujoco.mju_mat2Quat(self._grasp_quat, self._grasp_rotation.ravel())
        jaw_axis = self._grasp_rotation[:, 0]
        projections = positions @ jaw_axis
        half_extents = np.einsum(
            "nji,j->ni", rotations, jaw_axis
        )
        projected_widths = np.abs(half_extents) @ self.env.COOKIE_HALF_SIZE
        lower = float(np.min(projections - projected_widths))
        upper = float(np.max(projections + projected_widths))
        self._aligned_gap_center = None
        if self._aligned_grasp and neighbour is not None:
            rear_axis = jaw_axis * np.sign((neighbour - positions[-1]) @ jaw_axis)
            self._rear_axis = rear_axis
            selected_half_width = float(
                np.abs(rotations[-1].T @ rear_axis) @ self.env.COOKIE_HALF_SIZE
            )
            neighbour_half_width = float(
                np.abs(neighbour_rotation.T @ rear_axis) @ self.env.COOKIE_HALF_SIZE
            )
            gap_start = float(positions[-1] @ rear_axis + selected_half_width)
            gap_end = float(neighbour @ rear_axis - neighbour_half_width)
            self._aligned_gap_width_m = gap_end - gap_start
            pad_thickness = 2 * self.env.config.cookie_transfer.left_finger_pad_half_thickness_m
            if self._aligned_gap_width_m <= pad_thickness:
                raise RuntimeError(
                    f"bevel gap {self._aligned_gap_width_m * 1000:.2f} mm is narrower "
                    f"than the {pad_thickness * 1000:.2f} mm fingertip"
                )
            self._aligned_gap_center = (gap_start + gap_end) / 2
        self._pick_center = positions.mean(axis=0)
        self._pick_center += ((lower + upper) / 2 - self._pick_center @ jaw_axis) * jaw_axis
        insertion_overlap = 0.0038
        if (
            self.batch_index == 1
            and self._base_push_attempts >= 2
            and self._rear_clearance_m < 0.0025
            and not self._aligned_grasp
        ):
            insertion_overlap = 0.009
            self._pick_center += 0.002 * jaw_axis
        self._insert_opening = float(np.clip((upper - lower - insertion_overlap) / 0.085, 0, 1))
        self._opening = self._insert_opening
        pad_height_axis = up if self._aligned_grasp else np.array([0.0, 0.0, 1.0])
        self._pick_eef = (
            self._pick_center + pad_height_axis * 0.010
            - self._grasp_rotation @ self._pad_offset
        )
        self._high_eef = self._pick_eef + [0, 0, 0.082]
        self._approach_eef = self._high_eef
        if self._aligned_grasp:
            self._approach_eef = self._pick_eef + self._grasp_rotation[:, 1] * -0.082

        if self.batch_index == 1 and self._grasp_tilt_deg > 10 and self._base_push_attempts < 2:
            base_y = positions[0, 1] - rotations[0, 1, 2] * self.env.COOKIE_HALF_SIZE[2]
            self._push_start = np.array(
                [positions[0, 0], base_y - 0.012, self.env.SOURCE_FLOOR_TOP_Z + 0.004]
            )
            self._push_end = self._push_start + [0, 0.016, 0]
            self._push_high = self._push_start + [0, 0, 0.075]
            self._opening = 0.10
            self._advance(CookiePhase.ALIGN)
            self._push_stage = "approach"
        else:
            self._advance(CookiePhase.APPROACH)

    def _act_step(self, observation=None, task=""):
        del observation, task
        if self.finished:
            return self._hold_command(self._opening)
        if self.phase is CookiePhase.SELECT_COOKIE:
            self._select_batch()
        self.phase_steps += 1
        self._contact_chain()
        phase_limit = 500 if self.phase is CookiePhase.ALIGN else 420
        if self.phase_steps > phase_limit:
            self._fail(f"batch phase timeout; contacts={self._contact_chain()[1].tolist()}")
            return self._hold_command(self._opening)

        if self.phase is CookiePhase.ALIGN:
            if self._push_stage == "approach":
                action, reached = self._servo(self._push_high, self._canonical, 1.0, speed=0.015)
                if self._approach_ready(self._push_high, self._canonical, reached):
                    self._push_stage = "preclose"
                elif self.phase_steps >= 120:
                    self._fail("could not reach the straightening approach pose")
                return action
            if self._push_stage == "preclose":
                action, _ = self._servo(self._push_high, self._canonical, self._opening, speed=0.001)
                if abs(float(self.env.current_joint_action[7]) - self._opening) < 0.006:
                    self._push_stage = "descend"
                    self._servo_pos = None
                return action
            if self._push_stage in ("descend", "sweep"):
                if self._insertion_jammed(self._total_pad_forces, limit=12.0):
                    self._fail("base straightening blocked by excessive pad force")
                    return self._hold_command(self._opening)
                target = self._push_start if self._push_stage == "descend" else self._push_end
                speed = 0.0018 if self._push_stage == "descend" else 0.0012
                action, reached = self._servo(target, self._canonical, self._opening, speed=speed)
                if reached:
                    if self._push_stage == "descend":
                        self._push_stage = "sweep"
                    else:
                        self._push_retract = self.data.site_xpos[self._l_site].copy()
                        self._push_retract[2] += 0.075
                        self._push_stage = "retract"
                    self._servo_pos = None
                return action
            action, reached = self._servo(
                self._push_retract, self._canonical, self._opening, speed=0.0030
            )
            if reached:
                self._base_push_attempts += 1
                self._advance(CookiePhase.SELECT_COOKIE)
            return action

        if self.phase is CookiePhase.APPROACH:
            action, reached = self._servo(self._approach_eef, self._grasp_rotation, 1.0, speed=0.015)
            if self._approach_ready(self._approach_eef, self._grasp_rotation, reached):
                self._advance(CookiePhase.PRE_CLOSE)
            elif self.phase_steps >= 120:
                self._fail("could not reach the grasp approach pose")
            return action

        if self.phase is CookiePhase.PRE_CLOSE:
            action, _ = self._servo(self._approach_eef, self._grasp_rotation, self._opening, speed=0.001)
            actual_opening = float(self.env.current_joint_action[7])
            if abs(actual_opening - self._opening) <= 0.008:
                self._preclose_stable += 1
            else:
                self._preclose_stable = 0
            if self._preclose_stable >= 2:
                if self._aligned_gap_center is not None:
                    fingers = list(self.env._left_finger_geoms)
                    pad_projection = self.data.geom_xpos[fingers] @ self._rear_axis
                    self._far_pad_geom = fingers[int(np.argmax(pad_projection))]
                    correction = self._aligned_gap_center - float(max(pad_projection))
                    self._pick_eef += correction * self._rear_axis
                    self._far_pad_anchor = self._aligned_gap_center
                self._advance(CookiePhase.DESCEND)
            elif self.phase_steps >= 120:
                self._fail(
                    "pre-close did not reach target opening; "
                    f"target={self._opening * 0.085 * 1000:.1f} mm "
                    f"actual={actual_opening * 0.085 * 1000:.1f} mm"
                )
            return action

        if self.phase is CookiePhase.DESCEND:
            if (
                self.batch_index == 1
                and self._base_push_attempts >= 2
                and max(self._total_pad_forces) > 2.0
                and any(
                    any(self.env.privileged_left_finger_contacts(i))
                    for i in self.batch_indices
                )
                and np.linalg.norm(
                    self._pick_eef - self.data.site_xpos[self._l_site]
                ) < 0.030
            ):
                self._rim_target_eef = self._pick_eef.copy()
                self._pick_eef = self.data.site_xpos[self._l_site].copy()
                self._rim_contact_stop = True
                self._advance(CookiePhase.CLOSE)
                return self._servo(
                    self._pick_eef, self._grasp_rotation, self._opening
                )[0]
            if self._insertion_jammed(self._total_pad_forces):
                self._fail(f"insertion blocked; pad forces={self._total_pad_forces.tolist()}")
                return self._hold_command(self._opening)
            dist = np.linalg.norm(self._pick_eef - self.data.site_xpos[self._l_site])
            descend_speed = 0.0035 if dist > 0.015 else 0.0010
            action, reached = self._servo(
                self._pick_eef, self._grasp_rotation, self._opening, speed=descend_speed
            )
            if (
                not reached
                and self.batch_index == 1
                and self.phase_steps >= 100
                and np.linalg.norm(self._pick_eef - self.data.site_xpos[self._l_site]) < 0.005
                and max(self._total_pad_forces) > 0.1
            ):
                self._pick_eef = self.data.site_xpos[self._l_site].copy()
                reached = True
            if reached:
                self._advance(CookiePhase.CLOSE)
            return action

        if self.phase is CookiePhase.CLOSE:
            if self._insertion_jammed(self._total_pad_forces, limit=20.0):
                self._fail("batch closure blocked by excessive pad force")
                return self._hold_command(self._opening)
            chain, forces = self._contact_chain()
            close_rate = 0.0060 if (not chain or min(forces) < 0.5) else 0.0030
            if (
                self.batch_index == 1 and self._opening > 0.33
            ) or not chain or min(forces) < 1.3:
                self._opening = max(0.28, self._opening - close_rate)
            secure_width = self.batch_index == 0 or self._opening <= 0.33
            enough_depth = (
                not self._rim_contact_stop
                or self.data.site_xpos[self._l_site, 2] <= self._rim_target_eef[2] + 0.008
            )
            self._stable = (
                self._stable + 1
                if secure_width and chain and min(forces) >= 1.0 and enough_depth
                else 0
            )
            if (
                self._rim_contact_stop
                and (not chain or not enough_depth)
                and self._opening <= 0.28
                and max(self._total_pad_forces) < (18.0 if self._aligned_grasp else 12.0)
            ):
                delta = self._rim_target_eef - self._pick_eef
                distance = np.linalg.norm(delta)
                if distance > 1e-9:
                    self._pick_eef += delta * min(1.0, 0.00025 / distance)
            if self._aligned_gap_center is not None:
                far_projection = float(
                    self.data.geom_xpos[self._far_pad_geom] @ self._rear_axis
                )
                correction = np.clip(
                    self._far_pad_anchor - far_projection, -0.0006, 0.0006
                )
                self._pick_eef += correction * self._rear_axis
            action, _ = self._servo(self._pick_eef, self._grasp_rotation, self._opening, speed=0.0008)
            if self._stable >= 3:
                self._grip_reference = self._positions().copy()
                self._lift_start_eef = self.data.site_xpos[self._l_site].copy()
                self._advance(CookiePhase.LIFT)
            return action

        if self.phase is CookiePhase.LIFT:
            _, forces = self._contact_chain()
            if min(forces) < 1.0:
                self._opening = max(0.28, self._opening - 0.001)
            dist_lifted = np.linalg.norm(self.data.site_xpos[self._l_site] - self._lift_start_eef)
            lift_speed = 0.0018 if dist_lifted < 0.015 else 0.0035
            lift_speed = min(lift_speed, getattr(self, "_max_lift_speed", lift_speed))
            action, reached = self._servo(
                self._high_eef, self._grasp_rotation, self._opening, speed=lift_speed
            )
            lifted = self._positions()[:, 2] - self._grip_reference[:, 2]
            tool_rise = self.data.site_xpos[self._l_site, 2] - self._lift_start_eef[2]
            if tool_rise > 0.015 and np.any(tool_rise - lifted > 0.016):
                self._fail("Cookie dropped during lift")
                return self._hold_command(self._opening)
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
                return self._hold_command(self._opening)
            moving = self.phase is CookiePhase.MOVE_TO_SLOT
            clearance = 0.060 if moving else 0.0
            pos, rotation = self._place_pose(clearance)
            if moving:
                servo_speed = getattr(self, "_transport_speed", 0.0050)
            else:
                dist_to_place = np.linalg.norm(pos - self.data.site_xpos[self._l_site])
                servo_speed = 0.0030 if dist_to_place > 0.015 else 0.0012
            action, reached = self._servo(
                pos,
                rotation,
                self._opening,
                speed=servo_speed,
            )
            if reached:
                self._advance(CookiePhase.DESCEND_TO_PLACE if moving else CookiePhase.OPEN)
            return action

        if self.phase is CookiePhase.OPEN:
            self._opening = min(0.49, self._opening + 0.015)
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
                self._retract_pos, self._retract_rotation, self._opening, speed=0.0050
            )
            if reached:
                self._advance(CookiePhase.VERIFY_RELEASE)
            return action

        if self.phase is CookiePhase.VERIFY_RELEASE:
            all_inside = all(
                self.env.privileged_cookie_in_target(i)
                for i in self.completed_cookie_indices + self.batch_indices
            )
            released = not any(
                any(self.env.privileged_left_finger_contacts(i)) for i in self.batch_indices
            )
            self._stable = self._stable + 1 if all_inside and released else 0
            if self.phase_steps >= 80 and not all_inside:
                bad = [
                    i for i in self.completed_cookie_indices + self.batch_indices
                    if not self.env.privileged_cookie_in_target(i)
                ]
                self._fail(f"released Cookies not upright, settled and contained: {bad}")
                return self._hold_command(self._opening)
            required_stable = (
                self.env.task_config.success_hold_steps
                if self.batch_index == 1
                else 6
            )
            if self._stable >= required_stable:
                self.batch_reports[-1]["released"] = True
                self.completed_cookie_indices.extend(self.batch_indices)
                self.batch_index += 1
                self._advance(
                    CookiePhase.DONE if self.batch_index == 2 else CookiePhase.SELECT_COOKIE
                )
            return self._hold_command(self._opening)

        self._fail("unsupported batch phase")
        return self._hold_command(self._opening)


class A3VariedColumnBatchExpert(A3SameColumnBatchExpert):
    """Same physical grasp, with a reproducible source-column choice per episode."""

    def _act_step(self, observation=None, task=""):
        previous_phase = self.phase
        action = super()._act_step(observation, task)
        if previous_phase is CookiePhase.OPEN and self.phase is CookiePhase.RETRACT:
            # The tilted place tool must lift clear of the small box before
            # translating away; retreating along its tilted axis can jam.
            self._retract_pos = self.data.site_xpos[self._l_site].copy()
            self._retract_pos[2] += 0.060
        return action

    def _place_pose(self, clearance=0.0, column_index=None):
        position, rotation = super()._place_pose(clearance, column_index)
        target_column = self.batch_index if column_index is None else column_index
        box_rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        if (
            column_index is None
            and self.phase in (CookiePhase.MOVE_TO_SLOT, CookiePhase.DESCEND_TO_PLACE, CookiePhase.OPEN)
            and self.batch_indices
        ):
            batch_key = tuple(self.batch_indices)
            if getattr(self, "_held_rotation_batch", None) != batch_key:
                tool_rotation = self.data.site_xmat[self._l_site].reshape(3, 3)
                middle_cookie = self.env._cookie_bodies[self.batch_indices[2]]
                cookie_rotation = self.data.xmat[middle_cookie].reshape(3, 3)
                self._held_cookie_rotation_local = tool_rotation.T @ cookie_rotation
                self._held_rotation_batch = batch_key
            # Preserve the measured Cookie-to-tool rotation. A tilted grasp
            # must not become a tilted Cookie when the arm reaches the box.
            batch_center = position + rotation @ self._group_offset
            cookie_allowance = (
                np.deg2rad(8.0)
                if self.selected_source_column_index == 3 else 0.0
            )
            c, s = np.cos(cookie_allowance), np.sin(cookie_allowance)
            reach_relief = np.array(
                [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64
            )
            rotation = box_rotation @ reach_relief @ self._held_cookie_rotation_local.T
            position = batch_center - rotation @ self._group_offset
        wall_clearance = 0.005 if target_column == 0 else 0.003
        position = position + box_rotation[:, 0] * wall_clearance
        return position, rotation

    def reset(self, context=None):
        # The 2.5 mm / 0.035 rad vertical IK probe is slightly conservative
        # for tilted multi-column grasps. Keep the legacy experts unchanged.
        self._target_preflight_position_tol = 0.0030
        self._target_preflight_rotation_tol = 0.050
        super().reset(context)
        self._held_rotation_batch = None
        self._wait_for_approach = True
        # The far column needs a steeper approach when the source bin moves
        # away under the diverse profile; -30 deg leaves up to 9 mm of
        # unachievable descent on held-out layouts.
        self._column_tool_pitch_deg = (0, -15, -35, -35)
        columns = sorted(set(np.asarray(self.env.SOURCE_POSITIONS)[:, 0]))
        requested = getattr(self, "requested_source_column_index", None)
        if requested is None:
            seed = 0 if context is None else context.seed
            # Every consecutive group of four seeds covers all columns once,
            # with a reproducibly shuffled order within each group.
            group, offset = divmod(seed, len(columns))
            selected = int(np.random.default_rng(group).permutation(len(columns))[offset])
        else:
            selected = int(requested)
            if not 0 <= selected < len(columns):
                raise ValueError(f"source column {selected} is outside 0..{len(columns) - 1}")
        self.selected_source_column_index = selected
        self._max_lift_speed = 0.0022 if selected == len(columns) - 1 else 0.0035
        self._transport_speed = (
            0.0030 if selected == len(columns) - 1
            else 0.0035 if selected == 1 else 0.0050
        )
        self._column_order = [columns[selected]]
