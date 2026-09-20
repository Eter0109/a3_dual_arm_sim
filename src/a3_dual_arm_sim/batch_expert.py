"""Contact-driven five-Cookie grasp through bevels; privileged, not a VLA policy.

Objects are never attached, teleported, or moved by this controller. Each batch
must form a pad--Cookie contact chain, lift together, and be released into the box.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .expert import A3CookieTransferExpert, CookiePhase


class A3CookieBatchExpert(A3CookieTransferExpert):
    MAX_INSERTION_FORCE_N = 8.0

    def __post_init__(self):
        super().__post_init__()
        self.reset()

    def reset(self, context=None):
        super().reset(context)
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
        self._servo_pos = None
        self._grip_reference = None
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
        """Reject unreachable table layouts before starting a physical grasp."""
        for column in range(2):
            for clearance in (0.0, 0.085):
                position, rotation = self._place_pose(clearance, column)
                target_q = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(target_q, rotation.ravel())
                q = self.env.solve_ik(position, target_q, self.q_transit)
                work = self.env._ik._work
                work.qpos[self._l_qpos] = q
                mujoco.mj_kinematics(self.model, work)
                actual_q, error = np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(actual_q, work.site_xmat[self._l_site])
                mujoco.mju_subQuat(error, target_q, actual_q)
                distance = np.linalg.norm(work.site_xpos[self._l_site] - position)
                if distance > 0.002 or np.linalg.norm(error) > 0.035:
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
        threshold = self.MAX_INSERTION_FORCE_N if limit is None else limit
        self._jam_count = self._jam_count + 1 if max(forces) > threshold else 0
        return self._jam_count >= 3

    def _hold_command(self, opening: float) -> np.ndarray:
        action = np.zeros(14, dtype=np.float64)
        action[6] = np.clip(2.0 * float(opening) - 1.0, -1.0, 1.0)
        action[13] = 1.0
        return action

    def _servo(self, position, rotation, opening, speed=0.0024):
        """Rate-limited Cartesian delta TCP command."""
        work = self.env._ik._work
        work.qpos[:] = self.data.qpos
        work.qpos[list(self.env._ik._qpos_ids[0])] = self.env.last_applied_action[:7]
        work.qpos[list(self.env._ik._qpos_ids[1])] = self.env.last_applied_action[8:15]
        mujoco.mj_kinematics(self.model, work)
        cmd_pos = work.site_xpos[self._l_site].copy()

        target_pos = np.asarray(position, dtype=np.float64)
        pos_err = target_pos - cmd_pos
        dist = np.linalg.norm(pos_err)

        # Soft deceleration ramp to prevent abrupt stopping impact
        decel_dist = 0.020
        if dist < decel_dist and speed > 0.0020:
            effective_speed = max(0.0012, speed * (dist / decel_dist))
        else:
            effective_speed = speed
        step_dist = min(dist, effective_speed)
        if dist > 1e-6:
            delta_p = (pos_err / dist) * (step_dist / self.env.config.cartesian_translation_scale_m)
        else:
            delta_p = np.zeros(3)
        delta_p = np.clip(delta_p, -1.0, 1.0)

        target_q = np.empty(4, dtype=np.float64)
        rot_arr = np.asarray(rotation, dtype=np.float64)
        if rot_arr.size == 4:
            target_q[:] = rot_arr.ravel()
        else:
            mujoco.mju_mat2Quat(target_q, rot_arr.ravel())
        cur_q = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(cur_q, work.site_xmat[self._l_site])

        rot_err = np.empty(3, dtype=np.float64)
        mujoco.mju_subQuat(rot_err, target_q, cur_q)
        rot_dist = np.linalg.norm(rot_err)

        max_rot = min(0.05, self.env.config.cartesian_rotation_scale_rad)
        step_rot = min(rot_dist, max_rot)
        if rot_dist > 1e-6:
            delta_r = (rot_err / rot_dist) * (step_rot / self.env.config.cartesian_rotation_scale_rad)
        else:
            delta_r = np.zeros(3)
        delta_r = np.clip(delta_r, -1.0, 1.0)

        delta_g = np.clip(2.0 * float(opening) - 1.0, -1.0, 1.0)

        action = np.zeros(14, dtype=np.float64)
        action[0:3] = delta_p
        action[3:6] = delta_r
        action[6] = delta_g
        action[13] = 1.0

        meas_pos = self.data.site_xpos[self._l_site]
        meas_q = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(meas_q, self.data.site_xmat[self._l_site])
        meas_rot_err = np.empty(3, dtype=np.float64)
        mujoco.mju_subQuat(meas_rot_err, target_q, meas_q)

        qvel_norm = float(np.linalg.norm(self.data.qvel[:7]))
        reached = (
            np.linalg.norm(target_pos - meas_pos) < 0.0025
            and np.linalg.norm(meas_rot_err) < 0.045
            and qvel_norm < 0.25
        )
        return action, reached

    def _candidate_batches(self):
        """Prefer exposed ends; do not force an insertion through fallen Cookies."""
        source = np.asarray(self.env.SOURCE_POSITIONS)
        for x in sorted(set(source[:, 0])):
            indices = np.flatnonzero(np.isclose(source[:, 0], x))
            remaining = sorted(
                (i for i in indices.tolist() if i not in self.completed_cookie_indices),
                key=lambda i: source[i, 1],
            )
            if len(remaining) < 5:
                continue
            for batch in (remaining[:5], remaining[-5:]):
                if all(
                    self.env.privileged_cookie_in_source(i)
                    and abs(self.data.xmat[self.env._cookie_bodies[i], 8])
                    >= np.cos(np.deg2rad(10))
                    for i in batch
                ):
                    yield batch

    @property
    def _approach_q(self) -> np.ndarray:
        return self.env.solve_ik(
            self._high_eef, self._target_quat_canonical, self.env.current_joint_action[:7]
        )

    def _select_batch(self):
        source = np.asarray(self.env.SOURCE_POSITIONS)
        self.batch_indices = next(self._candidate_batches(), [])
        if len(self.batch_indices) != 5:
            raise RuntimeError("no exposed, upright five-Cookie batch available")
        self.current_cookie_index = self.batch_indices[0]
        self.target_slot_index = self.batch_index
        positions = self._positions()
        self._pick_center = positions.mean(axis=0)
        pitch = float(np.median(np.diff(source[self.batch_indices, 1])))
        # Pad centres land in the two boundary gaps. Their flat bottoms first
        # meet the sloped bevel, then push neighbours apart during slow descent.
        pad_thickness = 0.00635
        self._insert_opening = float(np.clip((5 * pitch - pad_thickness) / 0.085, 0, 1))
        self._opening = self._insert_opening
        self._pick_eef = self._pick_center + [0, 0, 0.010] - self._canonical @ self._pad_offset
        self._high_eef = self._pick_eef + [0, 0, 0.082]
        self._advance(CookiePhase.APPROACH)

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
            return self._hold_command(self._opening)

    def _act_step(self, observation=None, task=""):
        del observation, task
        if self.finished:
            return self._hold_command(self._opening)
        if self.phase is CookiePhase.SELECT_COOKIE:
            self._select_batch()
        self.phase_steps += 1
        self._contact_chain()
        if self.phase_steps > 420:
            self._fail(f"batch phase timeout; contacts={self._contact_chain()[1].tolist()}")
            return self._hold_command(self._opening)
        if self.phase is CookiePhase.APPROACH:
            action, reached = self._servo(self._high_eef, self._canonical, 1.0, speed=0.015)
            if reached or self.phase_steps >= 15:
                self._advance(CookiePhase.PRE_CLOSE)
            return action
        if self.phase is CookiePhase.PRE_CLOSE:
            action, _ = self._servo(self._high_eef, self._canonical, self._opening, speed=0.001)
            actual_opening = float(self.env.current_joint_action[7])
            if abs(actual_opening - self._opening) <= 0.008:
                self._preclose_stable += 1
            else:
                self._preclose_stable = 0
            if self._preclose_stable >= 2:
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
                return self._hold_command(self._opening)
            dist = np.linalg.norm(self._pick_eef - self.data.site_xpos[self._l_site])
            descend_speed = 0.0035 if dist > 0.015 else 0.0010
            action, reached = self._servo(
                self._pick_eef, self._canonical, self._opening, speed=descend_speed
            )
            if reached:
                self._advance(CookiePhase.CLOSE)
            return action
        if self.phase is CookiePhase.CLOSE:
            chain, forces = self._contact_chain()
            close_rate = 0.0060 if (not chain or min(forces) < 0.5) else 0.0030
            if not chain or min(forces) < 1.3:
                self._opening = max(0.28, self._opening - close_rate)
            self._stable = self._stable + 1 if chain and min(forces) >= 1.0 else 0
            action, _ = self._servo(self._pick_eef, self._canonical, self._opening, speed=0.0008)
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
            action, reached = self._servo(
                self._high_eef, self._canonical, self._opening, speed=lift_speed
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
                servo_speed = 0.0050
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
