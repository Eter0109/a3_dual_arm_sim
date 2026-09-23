"""How a target box's slots are grouped into grasps.

The batch expert takes a run of Cookies in one grasp and places the whole run into
one *column* of the target box, as consecutive slots.  That is a geometric
constraint rather than a preference: the jaws hold the batch in a straight line, so
a batch cannot span two columns without a sideways move in the middle of the
placement.  So the grouping is per column, and the capacity, the column count and
the grasp size determine it completely -- there is no free choice left in it.

That is what makes this derivable, and it is also why the obvious formula is
wrong.  ``[per_grasp] * (capacity // per_grasp) + [remainder]`` (the plan's first
version) spreads the remainder across the *whole box*: for ten Cookies in two
columns at three per grasp it gives ``[3, 3, 3, 1]``, and there is no way to place
that second batch of three, because column 0 has only two rows left and a batch
cannot continue into column 1.  The remainder is per column: the same box is
``[3, 2, 3, 2]``.

A box whose capacity is not a multiple of the column count gets one shorter
column rather than a shorter row -- ``capacity`` Cookies are placed and the
leftover slots simply stay empty -- so the lattice is ``ceil(capacity / columns)``
rows tall and the last column may not use its last row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchGroup:
    """One grasp: the target slots it fills, and where it is placed.

    ``rows`` and ``slot_indices`` are the same information in two forms: rows are
    the indices within the column (what the placement pose's y is computed from),
    and slot indices are positions in the flattened slot list, which is row-major
    (``slot = row * columns + column``), matching the order the shipped configs
    list their slots in.
    """

    #: Order in which the batches are grasped, 0-based.
    index: int
    #: Which target column this batch is placed into.
    column: int
    #: Slot rows within that column, ascending.
    rows: tuple[int, ...]
    #: Indices into the flattened slot list.
    slot_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class BatchPlan:
    """Every grasp a fill makes, derived from the capacity and the grasp size."""

    #: Rows in the target lattice: ``ceil(capacity / columns)``.
    rows: int
    columns: int
    per_grasp: int
    capacity: int
    groups: tuple[BatchGroup, ...]

    @classmethod
    def for_capacity(cls, capacity: int, per_grasp: int, columns: int) -> BatchPlan:
        if capacity < 1:
            raise ValueError(f"box capacity must be positive, got {capacity}")
        if per_grasp < 1:
            raise ValueError(f"per_grasp must be positive, got {per_grasp}")
        if columns < 1:
            raise ValueError(f"target columns must be positive, got {columns}")

        rows = math.ceil(capacity / columns)
        groups: list[BatchGroup] = []
        remaining = capacity
        for column in range(columns):
            # The last column takes whatever capacity is left over, so a capacity
            # that does not divide evenly gives one shorter column rather than a
            # row that is missing a slot.
            available = min(rows, remaining)
            remaining -= available
            sizes = [per_grasp] * (available // per_grasp)
            if available % per_grasp:
                sizes.append(available % per_grasp)
            start = 0
            for size in sizes:
                rows_used = tuple(range(start, start + size))
                groups.append(
                    BatchGroup(
                        index=len(groups),
                        column=column,
                        rows=rows_used,
                        slot_indices=tuple(row * columns + column for row in rows_used),
                    )
                )
                start += size
        return cls(
            rows=rows,
            columns=columns,
            per_grasp=per_grasp,
            capacity=capacity,
            groups=tuple(groups),
        )

    @property
    def placed(self) -> int:
        return sum(group.size for group in self.groups)

    def column_sizes(self, column: int) -> tuple[int, ...]:
        """The batch sizes of one column, in the order they are grasped."""

        return tuple(group.size for group in self.groups if group.column == column)

    def next_size(self, filled: int) -> int:
        """How many Cookies the next grasp should take, given ``filled`` already in.

        This is the running form of the same rule -- ``min(per_grasp, what is left
        of the current column)`` -- which is what lets a fill resume a box that
        arrived with Cookies already in it.  The static plan above is the special
        case ``filled == 0``.

        It assumes the Cookies already in the box occupy the leading groups in
        order, which is how a fill puts them there; a box filled from the far end
        would need the plan read backwards instead.
        """

        if not 0 <= filled < self.capacity:
            raise ValueError(f"filled must be in [0, {self.capacity}), got {filled}")
        consumed = 0
        for group in self.groups:
            if filled < consumed + group.size:
                return group.size - (filled - consumed)
            consumed += group.size
        raise ValueError(f"filled={filled} is past the plan's {self.placed} Cookies")
