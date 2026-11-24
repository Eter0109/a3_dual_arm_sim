"""Privileged physical experts for the course's variable-inventory scenes.

These controllers use simulator state and contact feedback; they do not represent
learned-policy results. Cookies remain dynamic rigid bodies throughout an episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cookie_2x10_expert import A3Cookie2x10Expert
from .phases import CookiePhase
from .same_column_batch_expert import A3VariedColumnBatchExpert


class _AvailableBatchMixin:
    """Choose contiguous, present, upright objects, including across columns."""

    @property
    def _is_followup_batch(self):
        if not self.completed_cookie_indices or not self.batch_indices:
            return False
        source = np.asarray(self.env.SOURCE_POSITIONS)
        return bool(np.isclose(source[self.completed_cookie_indices[-1], 0],
                               source[self.batch_indices[0], 0]))

    def _candidate_batches(self):
        source = np.asarray(self.env.SOURCE_POSITIONS)
        columns = np.unique(source[:, 0])
        for rank, x in enumerate(columns):
            indices = sorted(np.flatnonzero(np.isclose(source[:, 0], x)).tolist(),
                             key=lambda i: source[i, 1])
            runs, run = [], []
            for index in indices:
                present = (index not in self.completed_cookie_indices
                           and self.env.privileged_cookie_in_source(index)
                           and abs(self.data.xmat[self.env._cookie_bodies[index], 8])
                           >= np.cos(np.deg2rad(15)))
                if present:
                    run.append(index)
                else:
                    if run:
                        runs.append(run)
                    run = []
            if run:
                runs.append(run)
            for run in runs:
                if len(run) >= self.batch_size:
                    self.selected_source_column_index = rank
                    yield run[:self.batch_size]
                    if len(run) > self.batch_size:
                        yield run[-self.batch_size:]


class A3Course2x10Expert(_AvailableBatchMixin, A3Cookie2x10Expert):
    """Four five-cookie transfers, selecting from any available source column."""

    def _verify_complete_fill(self):
        mask = tuple(self.env.privileged_cookie_in_target(i)
                     for i in range(len(self.env.SOURCE_POSITIONS)))
        if self.env.target_column_counts(mask) != (10, 10):
            self._fail("the two target columns are not both filled")


@dataclass(frozen=True)
class _SparseTwentyBatchPlan:
    """Pair the four available source runs into two groups of ten."""

    first_grasp: int = 8
    batch_sizes: tuple[int, ...] = (8, 2, 6, 4)

    def target_rows(self, index):
        return ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9),
                (0, 1, 2, 3, 4, 5), (6, 7, 8, 9))[index]


class A3CourseSparse2x10Expert(A3Course2x10Expert):
    """Move the four sparse runs as 8+2 and 6+4 through physical contacts."""

    def reset(self, context=None):
        super().reset(context)
        counts = self.env.scenario_metadata["initial_column_counts"]
        if sorted(counts) != [2, 4, 6, 8]:
            raise ValueError("sparse 2x10 expert requires source counts 2, 4, 6, 8")
        self.plan = _SparseTwentyBatchPlan()
        self._source_order = tuple(counts.index(size) for size in self.plan.batch_sizes)

    def _candidate_batches(self):
        source = np.asarray(self.env.SOURCE_POSITIONS)
        columns = np.unique(source[:, 0])
        rank = self._source_order[self.batch_index]
        remaining = [int(i) for i in np.flatnonzero(np.isclose(source[:, 0], columns[rank]))
                     if int(i) not in self.completed_cookie_indices
                     and self.env.privileged_cookie_in_source(int(i))]
        remaining.sort(key=lambda i: source[i, 1])
        if len(remaining) == self.batch_size:
            self.selected_source_column_index = rank
            yield remaining


class A3Course2x5Expert(_AvailableBatchMixin, A3VariedColumnBatchExpert):
    """Fill 2x5 targets, including prefilled and two-target variants."""

    @property
    def batch_size(self):
        return len(self._course_plan[self.batch_index][2])

    def reset(self, context=None):
        env = self.env
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        columns = np.unique(slots[:, 0])
        if getattr(env, "_spare_body", None) is not None:
            target_columns = (0, 1) if env.box_quota == 10 else (0,)
            self._course_plan = [(body, col, tuple(range(col, 10, 2)))
                                 for body in (env._target_bin_body, env._spare_body)
                                 for col in target_columns]
        else:
            prefilled = set(getattr(env, "scenario_metadata", {}).get("prefilled_slot_ids", []))
            self._course_plan = []
            for column, x in enumerate(columns):
                free = tuple(i for i, slot in enumerate(slots)
                             if np.isclose(slot[0], x) and i not in prefilled)
                if free:
                    self._course_plan.append((env._target_bin_body, column, free))
        self.num_batches = len(self._course_plan)
        super().reset(context)

    def _select_batch(self):
        self._tb_id = self._course_plan[self.batch_index][0]
        delta = (self.batch_size - 5) * (2 * float(self.env.COOKIE_HALF_SIZE[1])) / 0.085
        self.min_grip_opening = max(0.0, 0.28 + delta)
        self.secure_grip_opening = max(0.0, 0.33 + delta)
        self.release_opening = min(1.0, max(0.18, 0.49 + delta))
        self.target_grip_force = 0.45 if self.batch_size == 1 else 1.3
        self.stable_grip_force = 0.35 if self.batch_size == 1 else 1.0
        super()._select_batch()

    def _place_pose(self, clearance=0.0, column_index=None):
        if column_index is not None:
            return super()._place_pose(clearance, column_index)
        body, column, indices = self._course_plan[self.batch_index]
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        box_rotation = self.data.xmat[body].reshape(3, 3)
        center = np.array([slots[indices[0], 0], slots[list(indices), 1].mean(),
                           self.env._target_cookie_center_z + 0.002 + clearance])
        if len(indices) < 5:
            column_y = slots[np.isclose(slots[:, 0], slots[indices[0], 0]), 1].mean()
            center[1] += 0.004 if center[1] >= column_y else -0.004
        center = self.data.xpos[body] + box_rotation @ center
        center += box_rotation[:, 0] * (0.005 if column == 0 else 0.003)
        rotation = box_rotation @ self._canonical
        if (self.phase in (CookiePhase.MOVE_TO_SLOT, CookiePhase.DESCEND_TO_PLACE, CookiePhase.OPEN)
                and self.batch_indices):
            key = tuple(self.batch_indices)
            if getattr(self, "_held_rotation_batch", None) != key:
                tool = self.data.site_xmat[self._l_site].reshape(3, 3)
                cookie = self.env._cookie_bodies[self.batch_indices[len(self.batch_indices) // 2]]
                self._held_cookie_rotation_local = tool.T @ self.data.xmat[cookie].reshape(3, 3)
                self._held_rotation_batch = key
            rotation = box_rotation @ self._held_cookie_rotation_local.T
        return center - rotation @ self._group_offset, rotation

    def _act_step(self, observation=None, task=""):
        if (self.phase is CookiePhase.VERIFY_RELEASE
                and getattr(self.env, "_spare_body", None) is not None):
            self.phase_steps += 1
            all_inside = all(self.env._inside_bin(i, self.env._target_bin_body)
                             or self.env._inside_bin(i, self.env._spare_body)
                             for i in self.completed_cookie_indices)
            all_inside &= all(self.env._inside_bin(i, self._tb_id) for i in self.batch_indices)
            released = not any(any(self.env.privileged_left_finger_contacts(i))
                               for i in self.batch_indices)
            self._stable = self._stable + 1 if all_inside and released else 0
            if self.phase_steps >= 80 and not all_inside:
                self._fail("released cookies are not stable in both target boxes")
            elif self._stable >= (self.env.task_config.success_hold_steps
                                  if self.batch_index == self.num_batches - 1 else 6):
                self.batch_reports[-1]["released"] = True
                self.completed_cookie_indices.extend(self.batch_indices)
                self.batch_index += 1
                self._advance(CookiePhase.DONE if self.batch_index == self.num_batches
                              else CookiePhase.SELECT_COOKIE)
            return self._hold_command(self._opening)
        return super()._act_step(observation, task)
