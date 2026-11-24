"""Shared source-cookie clearance checks for expanded layouts.

The checks use X projections across columns, rather than only two-dimensional
intersection, because ends in different rows can interleave during a grasp.
"""

from itertools import pairwise

import numpy as np


class SourceColumnClearanceMixin:
    """Sample and validate source poses using the environment's RNG."""

    MIN_SOURCE_COLUMN_CLEARANCE_M = 0.003

    def _begin_source_clearance_sampling(self):
        self._source_initial_rectangles = []
        self._source_pose_resamples = 0

    def _sample_source_cookie_pose(self, index, base_position, randomize):
        half_size = self.COOKIE_HALF_SIZE[:2]
        nominal_x = self.SOURCE_POSITIONS[index][0]
        for _ in range(100 if randomize else 1):
            position, quaternion = super()._sample_source_cookie_pose(
                index, base_position, randomize
            )
            c, s = 1 - 2 * quaternion[3] ** 2, 2 * quaternion[0] * quaternion[3]
            rotation = np.array([[c, -s], [s, c]])
            center = np.asarray(position[:2])
            valid = True
            for other_center, other_rotation, other_nominal_x in self._source_initial_rectangles:
                if nominal_x != other_nominal_x:
                    direction = np.sign(nominal_x - other_nominal_x)
                    x_gap = (
                        direction * (center[0] - other_center[0])
                        - np.abs(rotation[0]) @ half_size
                        - np.abs(other_rotation[0]) @ half_size
                    )
                    if x_gap < self.MIN_SOURCE_COLUMN_CLEARANCE_M:
                        valid = False
                        break
                else:
                    axes = np.concatenate((rotation.T, other_rotation.T))
                    distance = np.abs(axes @ (center - other_center))
                    radii = (
                        np.abs(axes @ rotation) @ half_size
                        + np.abs(axes @ other_rotation) @ half_size
                    )
                    if np.all(distance < radii + 0.0001):
                        valid = False
                        break
            if valid:
                self._source_initial_rectangles.append((center, rotation, nominal_x))
                return position, quaternion
            self._source_pose_resamples += 1
        raise RuntimeError(
            f"cannot sample a clear source pose for Cookie {index} within the configured noise; "
            f"cross-column clearance must be at least {self.MIN_SOURCE_COLUMN_CLEARANCE_M * 1000:.1f} mm"
        )

    def _record_settled_source_clearances(self, source_mask):
        """Check only cookies still present in the source after scenario setup."""
        positions = self.cookie_positions
        rotations = self.data.xmat[list(self._cookie_bodies)].reshape(-1, 3, 3)
        x_extent = (np.abs(rotations) @ self.COOKIE_HALF_SIZE)[:, 0]
        nominal_x = np.asarray(self.SOURCE_POSITIONS)[:, 0]
        active = np.asarray(source_mask, dtype=bool)
        if active.shape != nominal_x.shape:
            raise ValueError("source clearance mask must match the cookie inventory")
        clearances = []
        column_pairs = []
        for column_index, (left, right) in enumerate(pairwise(np.unique(nominal_x))):
            left_indices = (nominal_x == left) & active
            right_indices = (nominal_x == right) & active
            if not left_indices.any() or not right_indices.any():
                continue
            clearances.append(float(
                np.min(positions[right_indices, 0] - x_extent[right_indices])
                - np.max(positions[left_indices, 0] + x_extent[left_indices])
            ))
            column_pairs.append([column_index + 1, column_index + 2])
        if clearances and min(clearances) < self.MIN_SOURCE_COLUMN_CLEARANCE_M - 1e-9:
            raise RuntimeError(
                "settled source columns have insufficient clearance: "
                f"{min(clearances) * 1000:.3f} mm; "
                f"at least {self.MIN_SOURCE_COLUMN_CLEARANCE_M * 1000:.3f} mm required"
            )
        self.randomization_metadata["source_pose_resamples"] = self._source_pose_resamples
        self.randomization_metadata["source_column_clearances_m"] = clearances
        self.randomization_metadata["source_column_clearance_pairs"] = column_pairs
