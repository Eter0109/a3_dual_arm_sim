"""One five-object goal per target region; no two-batch-only plan validation."""
import numpy as np
from .contracts import BatchGoal


def target_regions(slots, center_z):
    slots = np.asarray(slots, dtype=float)
    regions = []
    for x in np.unique(slots[:, 0]):
        column = slots[np.isclose(slots[:, 0], x)]
        column = column[np.argsort(column[:, 1])]
        if len(column) % 5:
            raise ValueError("Each target column must contain complete five-cookie regions")
        for start in range(0, len(column), 5):
            xy = column[start:start + 5].copy()
            if len(column) > 5:
                # Preserve the current 2x10 expert's extra jaw-release gap.
                xy[:, 1] += -.004 if start == 0 else .004
            regions.append(np.c_[xy, np.full(5, center_z)])
    return tuple(regions)


class BatchPlanner:
    def __init__(self, env, *, source_column=1, require_geometry=False):
        self.require_geometry = require_geometry
        self.regions = target_regions(env.TARGET_SLOTS_LOCAL, env._target_cookie_center_z)
        source = np.asarray(env.SOURCE_POSITIONS)
        columns = np.unique(source[:, 0])
        if source_column != "auto" and not 1 <= source_column <= len(columns):
            raise ValueError("Source column does not exist")
        eligible = (range(len(source)) if source_column == "auto" else
                    np.flatnonzero(np.isclose(source[:, 0], columns[source_column - 1])).tolist())
        self.source_order = tuple(sorted(
            eligible, key=lambda i: (source[i, 0], source[i, 1])))
        self.allowed_ids = frozenset(self.source_order)
        self.column_orders = tuple(tuple(i for i in self.source_order if np.isclose(source[i, 0], x))
                                   for x in columns)
        if len(self.source_order) < len(self.regions) * 5:
            raise ValueError("Source column has too few objects")
        if len(self.regions) * 5 != env.task_config.required_cookies:
            raise ValueError("Target geometry and required count disagree")

    def next_goal(self, index, completed, scene):
        selected = None
        for column in self.column_orders:
            # Select the exposed end of each uninterrupted available run. A
            # missing object starts a new run, never a stitched five-object set.
            runs, run = [], []
            for i in column:
                if i in completed or not scene.cookies[i].in_source:
                    if run:
                        runs.append(run)
                    run = []
                else:
                    run.append(i)
            if run:
                runs.append(run)
            for run in runs:
                if len(run) < 5:
                    continue
                candidate = tuple(run[:5])
                if self.require_geometry:
                    from .grounding import validate_pick_group
                    try:
                        validate_pick_group(scene, candidate)
                    except ValueError:
                        continue
                selected = candidate
                break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError("The exposed next five cookies are unavailable; relocalization required")
        return BatchGoal(index, selected, self.regions[index])


def rectangle_overlap(center_a, rotation_a, half_a, center_b, rotation_b, half_b, margin=0.0):
    """2D SAT test, used only as an occupancy proxy, not a collision-free-path proof."""
    axes = np.r_[rotation_a[:2, :2].T, rotation_b[:2, :2].T]
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            continue
        axis = axis / norm
        ra = np.abs(axis @ rotation_a[:2, :]) @ half_a
        rb = np.abs(axis @ rotation_b[:2, :]) @ half_b
        if abs(axis @ (center_b[:2] - center_a[:2])) >= ra + rb + margin:
            return False
    return True
