"""``adopt`` for the twenty-slot box: place five at a time into the rows that are free.

The batch scene's `adopt` answers one question -- *which column is empty* -- and that is
the whole story when a column is five slots.  The 2x10 target is two columns of **ten**,
and a policy does not fill columns: measured on v9's eight 2x10 episodes, the policy left
``(8, 3)``, ``(9, 8)``, ``(9, 9)``, ``(10, 4)`` ... i.e. both columns partly full, with
the two columns' totals differing by anything from 0 to 9.  Two consequences:

* **"a column with five free slots" is the wrong gate.**  It is satisfiable 5 times in 8
  on v9's end states, and the three that fail are the ones already at 19-20, so the gate
  costs nothing -- but it also *under*-describes what is possible, because a column with
  eight free slots can take one group of five and a column with ten can take two.
* **Which rows matters, not just which column.**  `_place_pose` carries the batch as a
  block and aims at the **mean y of the rows it is given**; the batch scene can pass the
  whole column because all five of its rows are free by construction.  Aimed at the
  centre of a half-full ten-row column it descends onto the Cookies the policy already
  put there.  So the plan has to name a *contiguous run of free rows*, and the group
  goes into that run.

Hence the entry here: read the judge's occupancy, cut each column into contiguous free
runs, and turn the longest runs into a sequence of groups of five.  Nothing is moved and
no slot is re-derived -- the recovery the policy needs is "put the next five where the
slots actually are", which is the opposite of following its own misplaced block.

**Why the batch is still exactly five.**  The 2x10 data's variable-size grasps
(``a3_2x10_n5`` and friends) are what teach a policy to take one Cookie or nine; they are
not what a *recovery* needs.  Every end state that wants help has a free run of at least
six rows (measured: 7, 7, 10, 6, 10 on the five recoverable seeds), so five always fits,
and the batch expert's insertion geometry -- ``_group_offset``, ``_insert_opening``, the
pad force chain -- is measured for five.  Reusing it unchanged is the point; a
variable-size grasp here would be a second, untested controller.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .batch_expert import A3CookieBatchExpert
from .expert import CookiePhase

#: Cookies per grasp.  Fixed, for the reasons in the module docstring.
BATCH = 5


class A3Cookie2x10RecoveryExpert(A3CookieBatchExpert):
    """Finish a twenty-slot box from whatever the policy left in it.

    Runs only in ``adopt`` mode (``env.second_batch_mode = "adopt"``), because that is the
    only mode in which an occupancy exists to plan from.  Everything the batch expert does
    after the plan -- the clearing lift, the adopted grasp, the row selection, the
    insertion, the release -- is inherited unchanged.
    """

    #: Rows of clearance wanted on the far side of a group, so the inner pad can open
    #: without pushing the neighbouring group over.  The 2x10 expert uses 4 mm for the same
    #: reason; this is applied on whichever side the free run *ends*, not on parity.
    GROUP_RELIEF_M = 0.004

    def reset(self, context: Any = None) -> None:
        mode = getattr(self.env, "second_batch_mode", "full")
        if mode != "adopt":
            raise ValueError(
                "A3Cookie2x10RecoveryExpert plans from occupancy and needs "
                f'second_batch_mode="adopt", got {mode!r}'
            )
        # The base class reads the same switch and would run the *two-column* adopt entry.
        # Masking it for the duration of `super().reset` is deliberate: that entry is
        # production-proven for the batch scene and must not be refactored to serve a
        # layout it was not measured on.
        self.env.second_batch_mode = "full"
        try:
            super().reset(context)
        finally:
            self.env.second_batch_mode = mode
        self._targets: list[tuple[int, tuple[int, ...]]] = []
        self._skipped_runs: list[dict[str, Any]] = []
        self.adopt_infeasible: str | None = None
        self._adopt_2x10()

    # -- the plan ----------------------------------------------------------
    def _column_rows(self, slots: np.ndarray, columns: np.ndarray) -> list[list[int]]:
        """For each column, the occupancy indices of its rows, ordered by y.

        The occupancy tuple is indexed by slot, and a slot's row is its rank within its
        column's y order -- not ``index // 2``.  The two agree for the shipped layout
        (slots are generated row-major) and would silently disagree for any other, so the
        row is derived from the geometry rather than from the file's line order.
        """

        rows: list[list[int]] = []
        for column in range(len(columns)):
            indices = [
                index for index in range(len(slots)) if np.isclose(slots[index, 0], columns[column])
            ]
            rows.append(sorted(indices, key=lambda index: float(slots[index, 1])))
        return rows

    def _plan_targets(self) -> list[tuple[int, tuple[int, ...]]]:
        """Greedy: fill the longest contiguous free run, five rows at a time.

        A run shorter than five is recorded rather than used -- those rows stay empty, and
        the report says so instead of the episode looking like it simply ran out of steps.
        Ties go to the column nearer the arm (lower index), which is the one the batch
        expert's own reach ranking prefers.
        """

        runs: list[tuple[int, tuple[int, ...]]] = []
        for column, indices in enumerate(self._column_rows_cache):
            free = [row for row, index in enumerate(indices) if self._occupancy[index] < 0]
            start = 0
            for position in range(1, len(free) + 1):
                if position == len(free) or free[position] != free[position - 1] + 1:
                    if free[start:position]:
                        runs.append((column, tuple(free[start:position])))
                    start = position
        # Longest first, then the column nearer the arm, then the lower rows -- the last
        # key only so the order is total and the plan is reproducible.
        runs.sort(key=lambda run: (-len(run[1]), run[0], run[1][0]))
        targets: list[tuple[int, tuple[int, ...]]] = []
        for column, rows in runs:
            whole = list(rows)
            while len(whole) >= BATCH:
                group = tuple(whole[:BATCH])
                # **A free run is a claim about slots; the corridor is a claim about the box.**
                # A policy that shoves Cookies in without slotting them leaves rows that read
                # free while a Cookie lies across them -- see `_corridor_blocked`.  Skipping
                # the group here costs one plan entry; not skipping it costs a 400-step
                # `LIFT` or `DESCEND_TO_PLACE` failure that names the expert.
                blocker = self._corridor_blocked(column, group)
                if blocker is None:
                    targets.append((column, group))
                else:
                    self._skipped_runs.append(
                        {
                            "column": column,
                            "rows": list(group),
                            "reason": f"Cookie {blocker} lies in the insertion corridor",
                        }
                    )
                whole = whole[BATCH:]
            if whole:
                self._skipped_runs.append(
                    {"column": column, "rows": list(whole), "reason": "shorter than one batch"}
                )
        return targets

    def _adopt_2x10(self) -> None:
        """Take over the box's occupancy without moving anything in it."""

        in_target = tuple(
            self.env._cookie_inside_target(index)  # noqa: SLF001 - the judge, not an input
            for index in range(len(self.env.cookie_positions))
        )
        completed = [index for index, inside in enumerate(in_target) if inside]
        self.completed_cookie_indices = completed
        self.batch_indices = []
        self.adopted_cookie_count = len(completed)

        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        columns = np.unique(slots[:, 0])
        self._column_rows_cache = self._column_rows(slots, columns)
        self._occupancy = self.env._target_slot_occupancy(in_target)  # noqa: SLF001 - the judge
        self._targets = self._plan_targets()
        self._targets_planned = list(self._targets)

        if not self._targets:
            # Cheap and named, rather than a 420-step DESCEND timeout that reports "a
            # placement went wrong" when nothing was attempted.  Same rule and same
            # reasoning as the batch expert's `adopt_infeasible`.
            self.adopt_infeasible = (
                "no column has five contiguous free rows "
                f"(occupancy: {list(self._occupancy)}, skipped runs: {self._skipped_runs})"
            )
            self._fail(self.adopt_infeasible)
            self.failure_phase = "ADOPT"
            return

        self.batch_index = 0
        self.target_column = self._targets[0][0]

        # **Leave the box before touching anything.**  Identical to the batch expert's
        # adopt, and for the same measured reason (9.39): held at the opening the hand
        # already has, a vertical retreat cannot sweep or squeeze the block.  Opening the
        # jaws here displaced 13 of 70 source Cookies.
        here = np.asarray(self.data.site_xpos[self._l_site], dtype=np.float64).copy()
        rotation = np.asarray(self.data.site_xmat[self._l_site], dtype=np.float64).reshape(3, 3)
        self._clear_lift = (
            here + np.array([0.0, 0.0, self.ADOPT_CLEAR_HEIGHT_M]),
            rotation.copy(),
        )
        # **A grasp already in the jaws is not something to throw away** -- a policy's
        # second batch often ends with five closed on and not lifted, and opening there is
        # a failure the expert would cause rather than inherit.
        self.adopted_grasp = self._adopt_grasped_batch()
        if self.adopted_grasp:
            self.right_gripper = float(np.clip(self.env.last_applied_action[15], 0.0, 1.0))
            self.batch_reports.append(
                {
                    "cookies": list(self.adopted_grasp),
                    "lifted": len(self.adopted_grasp),
                    "lift_heights_m": [0.0] * len(self.adopted_grasp),
                    "released": False,
                    "adopted_grasp": True,
                    "recovery_targets": [self._describe_target(0)],
                }
            )
            return
        self.adopt_start = os.environ.get("A3_ADOPT_START", "lift")
        if self.adopt_start == "park":
            self._adopt_park_left_arm()
        elif self.adopt_start == "lift":
            self._clear_stage = "lift"
            self._clear_opening = float(np.clip(self.env.current_joint_action[7], 0.0, 1.0))
        self.right_gripper = float(np.clip(self.env.last_applied_action[15], 0.0, 1.0))
        self.batch_reports.append(
            {
                "cookies": list(completed),
                "lifted": len(completed),
                "lift_heights_m": [0.0] * len(completed),
                "released": True,
                "adopted": True,
                "recovery_targets": [self._describe_target(i) for i in range(len(self._targets))],
                "skipped_runs": list(self._skipped_runs),
            }
        )

    def _describe_target(self, batch_index: int) -> dict[str, Any]:
        column, rows = self._targets[batch_index]
        return {"column": int(column), "rows": list(rows), "count": len(rows)}

    # -- the batch machine, extended to more than two ----------------------
    @property
    def _is_final_batch(self) -> bool:
        return self.batch_index + 1 >= len(self._targets)

    @property
    def _batches_complete(self) -> bool:
        return self.batch_index >= len(self._targets)

    def _on_batch_advanced(self) -> None:
        if self.batch_index < len(self._targets):
            self.target_column = self._targets[self.batch_index][0]

    @property
    def status(self) -> str:
        total = len(self._targets)
        return (
            f"phase={self.phase.name} batch={min(self.batch_index + 1, total)}/{total} "
            f"column={self.target_column} cookies={self.batch_indices} "
            f"confirmed={len(self.completed_cookie_indices)}"
        )

    # -- placement ---------------------------------------------------------
    def _place_pose(self, clearance: float = 0.0, column_index: int | None = None):
        """Aim at the mean y of the planned free rows, not the column's centre.

        ``column_index`` is passed by `_validate_target_workspace`, which checks every
        column's reachability before an episode starts; that caller wants the whole column,
        exactly as the batch expert does, so the planned rows are used only when the
        column was not named.
        """

        rotation = self.data.xmat[self._tb_id].reshape(3, 3)
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        columns = np.unique(slots[:, 0])
        planned = column_index is None and self.batch_index < len(self._targets)
        if column_index is None:
            column_index = (
                self.target_column if self.target_column is not None else self.batch_index
            )
        column_x = columns[column_index]
        column_slots = slots[np.isclose(slots[:, 0], column_x)]
        rows = self._targets[self.batch_index][1] if planned else tuple(range(len(column_slots)))
        centre = np.array(
            [
                column_x,
                column_slots[list(rows), 1].mean(),
                self.env._target_cookie_center_z + 0.002 + clearance,
            ]
        )
        if planned:
            # **Clearance goes on the side the group is not.**  The batch scene shifts by
            # parity because its groups alternate; a recovery group is placed next to one
            # the *policy* put down, and the pad on that side has to open into the gap
            # rather than against the neighbour.  Whichever end of the run touches an
            # occupied row is the end to lean away from.
            centre[1] += self._relief_sign(column_index, rows)
        centre = self.data.xpos[self._tb_id] + rotation @ centre
        tool_rotation = rotation @ self._canonical
        wall_clearance = 0.005 if column_index == 0 else 0.003
        centre += rotation[:, 0] * wall_clearance
        return centre - tool_rotation @ self._group_offset, tool_rotation

    def _relief_sign(self, column_index: int, rows: tuple[int, ...]) -> float:
        """Which way to lean so the inner pad opens into free space."""

        indices = self._column_rows_cache[column_index]
        low = indices[rows[0] - 1] if rows[0] > 0 else None
        high = indices[rows[-1] + 1] if rows[-1] + 1 < len(indices) else None
        low_taken = low is not None and self._occupancy[low] >= 0
        high_taken = high is not None and self._occupancy[high] >= 0
        if low_taken and not high_taken:
            return self.GROUP_RELIEF_M
        if high_taken and not low_taken:
            return -self.GROUP_RELIEF_M
        return 0.0
