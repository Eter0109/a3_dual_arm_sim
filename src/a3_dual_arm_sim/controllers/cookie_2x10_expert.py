"""Variable-size, contact-driven expert for the separate 2x10 scene."""

from dataclasses import dataclass

import numpy as np

from a3_dual_arm_sim.controllers.expert import CookiePhase
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan, parse_first_grasp


@dataclass
class A3Cookie2x10Expert(A3VariedColumnBatchExpert):
    first_grasp: int | str = 5
    sampling_seed: int = 0
    requested_source_column_index: int | None = 0
    num_batches = 4

    def reset(self, context=None):
        if self.env.task_config.required_cookies != 20:
            raise ValueError("the variable-size expert requires the 2x10 environment")
        self.first_grasp = parse_first_grasp(self.first_grasp)
        self.plan = Cookie2x10Plan.from_seed(
            self.first_grasp, self.sampling_seed if context is None else context.seed
        )
        self._selection_batch_index = None
        super().reset(context)

    @property
    def batch_size(self):
        return self.plan.batch_sizes[self.batch_index] if self.batch_index < 4 else 0

    @property
    def _is_followup_batch(self):
        return bool(self.completed_cookie_indices)

    @property
    def status(self):
        return (
            f"phase={self.phase.name} grasp={min(self.batch_index + 1, 4)}/4 "
            f"plan={self.plan.batch_sizes} cookies={self.batch_indices} "
            f"confirmed={len(self.completed_cookie_indices)}/20"
        )

    def _select_batch(self):
        while self.batch_index < 4 and self.batch_size == 0:
            self.batch_reports.append(
                {
                    "grasp": self.batch_index + 1,
                    "planned_count": 0,
                    "cookies": [],
                    "lifted": 0,
                    "released": True,
                    "skipped": True,
                }
            )
            self.batch_index += 1
        if self.batch_index >= 4:
            self._advance(CookiePhase.DONE)
            return
        if self._selection_batch_index != self.batch_index:
            self._base_push_attempts = 0
            self._selection_batch_index = self.batch_index
        # Match the unchanged physical 85 mm gripper to the selected stack.
        # Five-cookie values are exactly the old limits; one-cookie stacks
        # can close farther, and ten-cookie stacks can release farther.
        thickness = 2 * float(self.env.COOKIE_HALF_SIZE[1])
        delta = (self.batch_size - 5) * thickness / 0.085
        self.min_grip_opening = max(0.0, 0.28 + delta)
        self.secure_grip_opening = max(0.0, 0.33 + delta)
        self.release_opening = min(1.0, max(0.18, 0.49 + delta))
        # With unchanged thin pads and jaw end stops a single 6.3 mm Cookie
        # develops about 0.58 N at full closure, below the old 1 N threshold.
        self.target_grip_force = 0.45 if self.batch_size == 1 else 1.3
        self.stable_grip_force = 0.35 if self.batch_size == 1 else 1.0
        super()._select_batch()

    def _place_pose(self, clearance=0.0, column_index=None):
        column = self.batch_index // 2 if column_index is None else column_index
        box_rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        column_x = np.unique(slots[:, 0])[column]
        rows = self.plan.target_rows(self.batch_index) if column_index is None else tuple(range(10))
        column_slots = slots[np.isclose(slots[:, 0], column_x)]
        if not rows:
            raise RuntimeError("zero-count grasp must be skipped before selecting a place pose")
        center = np.array(
            [
                column_x,
                column_slots[list(rows), 1].mean(),
                self.env._target_cookie_center_z + 0.002 + clearance,
            ]
        )
        if column_index is None and self.plan.first_grasp > 0:
            # Leave room for the inner pad to release the second group without
            # pushing the first group over. The larger box has this clearance.
            center[1] += -0.004 if self.batch_index % 2 == 0 else 0.004
        center = self.data.xpos[self._tb_id] + box_rotation @ center
        tool_rotation = box_rotation @ self._canonical
        if (
            column_index is None
            and self.phase
            in (CookiePhase.MOVE_TO_SLOT, CookiePhase.DESCEND_TO_PLACE, CookiePhase.OPEN)
            and self.batch_indices
        ):
            key = tuple(self.batch_indices)
            if getattr(self, "_held_rotation_batch", None) != key:
                measured_tool = self.data.site_xmat[self._l_site].reshape(3, 3)
                middle = self.env._cookie_bodies[self.batch_indices[len(self.batch_indices) // 2]]
                measured_cookie = self.data.xmat[middle].reshape(3, 3)
                self._held_cookie_rotation_local = measured_tool.T @ measured_cookie
                self._held_rotation_batch = key
            angle = np.deg2rad(8.0) if self.selected_source_column_index == 3 else 0.0
            c, s = np.cos(angle), np.sin(angle)
            relief = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            tool_rotation = box_rotation @ relief @ self._held_cookie_rotation_local.T
        center += box_rotation[:, 0] * (0.005 if column == 0 else 0.003)
        return center - tool_rotation @ self._group_offset, tool_rotation

    def _act_step(self, observation=None, task=""):
        previous_count = len(self.batch_reports)
        action = super()._act_step(observation, task)
        for report in self.batch_reports[previous_count:]:
            if not report.get("skipped"):
                report.update(
                    grasp=self.batch_index + 1, planned_count=self.batch_size, skipped=False
                )
                self._set_release_opening()
        if self.phase is CookiePhase.DONE:
            self._verify_complete_fill()
        return action

    def _verify_complete_fill(self):
        in_target = tuple(
            self.env.privileged_cookie_in_target(i) for i in range(len(self.env.SOURCE_POSITIONS))
        )
        columns = self.env.target_column_counts(in_target)
        remaining = sum(
            self.env.privileged_cookie_in_source(i) for i in range(len(self.env.SOURCE_POSITIONS))
        )
        expected = len(self.env.SOURCE_POSITIONS) - 20
        if columns != (10, 10) or remaining != expected:
            self._fail(
                f"2x10 fill integrity failed: target columns={columns}, "
                f"source cookies={remaining}/{expected}; inspect collateral movement"
            )

    def _set_release_opening(self):
        """Release with local clearance, rather than opening into the first group."""
        pads = self.data.geom_xpos[list(self.env._left_finger_geoms)]
        axis = pads[1] - pads[0]
        pad_distance = np.linalg.norm(axis)
        axis /= pad_distance
        half_pad = self.env.config.cookie_transfer.left_finger_pad_half_thickness_m
        # Thin pads retain the original jaw end stops: normalized zero does
        # not mean zero physical clearance between the two contact surfaces.
        gap_offset = pad_distance - 2 * half_pad - self.env.current_joint_action[7] * 0.085
        rotations = self.data.xmat[
            list(np.asarray(self.env._cookie_bodies)[self.batch_indices])
        ].reshape(-1, 3, 3)
        half_extents = np.abs(np.einsum("nji,j->ni", rotations, axis)) @ self.env.COOKIE_HALF_SIZE
        projections = self._positions() @ axis
        width = max(projections + half_extents) - min(projections - half_extents)
        self.release_opening = float(np.clip((width + 0.006 - gap_offset) / 0.085, 0, 1))
