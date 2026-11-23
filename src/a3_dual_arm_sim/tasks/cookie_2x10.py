"""Independent 2x10 target task; the 2x5 environment remains the default."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig


@dataclass(frozen=True)
class Cookie2x10TaskConfig(CookieTransferTaskConfig):
    required_cookies: int = 20
    require_exact_slots: bool = False
    require_released: bool = True

    def __post_init__(self) -> None:
        if self.required_cookies != 20 or self.cookie_count < 20:
            raise ValueError("the 2x10 target requires 20 cookies and at least 20 source cookies")


class A3Cookie2x10Env(A3CookieTransferEnv):
    TASK_CONFIG_CLASS = Cookie2x10TaskConfig
    # The source columns must remain separable as the selected row is
    # compressed. A mere 2-D nonintersection test can accept interleaved
    # ends when their Y coordinates differ.
    MIN_SOURCE_COLUMN_CLEARANCE_M = 0.003

    def __init__(self, config=None, **kwargs):
        super().__init__(config or resource_root() / "configs/cookie_2x10.yaml", **kwargs)
        if self.config.cookie_transfer.target_rows != 10:
            raise ValueError("A3Cookie2x10Env requires the separate 2x10 scene configuration")

    def reset(self, *, seed=None, options=None):
        self._source_initial_rectangles = []
        self._source_pose_resamples = 0
        observation, info = super().reset(seed=seed, options=options)
        positions = self.cookie_positions
        rotations = self.data.xmat[list(self._cookie_bodies)].reshape(-1, 3, 3)
        x_extent = (np.abs(rotations) @ self.COOKIE_HALF_SIZE)[:, 0]
        nominal_x = np.asarray(self.SOURCE_POSITIONS)[:, 0]
        present = (
            self._scene_randomizer.active
            if self._scene_randomizer is not None
            else np.ones_like(nominal_x, dtype=bool)
        )
        columns = np.unique(nominal_x[present])
        clearances = []
        for left, right in pairwise(columns):
            left_indices = present & (nominal_x == left)
            right_indices = present & (nominal_x == right)
            clearances.append(float(
                np.min(positions[right_indices, 0] - x_extent[right_indices])
                - np.max(positions[left_indices, 0] + x_extent[left_indices])
            ))
        if clearances and min(clearances) < self.MIN_SOURCE_COLUMN_CLEARANCE_M - 1e-9:
            raise RuntimeError(
                "settled source columns have insufficient clearance: "
                f"{min(clearances) * 1000:.3f} mm; "
                f"at least {self.MIN_SOURCE_COLUMN_CLEARANCE_M * 1000:.3f} mm required"
            )
        if info["cookies_in_source"] != self.initial_cookie_count:
            raise RuntimeError("invalid 2x10 initial state: active cookies outside the source box")
        self.randomization_metadata["source_pose_resamples"] = self._source_pose_resamples
        self.randomization_metadata["source_column_clearances_m"] = clearances
        return observation, info

    def _sample_source_cookie_pose(self, index, base_position, randomize):
        # In different source columns, require a gap between the X projections
        # even if the cookies initially occupy different Y coordinates. This
        # leaves room for a group to be compressed without interleaving ends.
        # Within each column, retain the existing 2-D nonintersection check.
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

    def target_column_counts(
        self, in_target: tuple[bool, ...], *, target_body: int | None = None
    ) -> tuple[int, int]:
        columns = np.unique(np.asarray(self.TARGET_SLOTS_LOCAL)[:, 0])
        counts = [0, 0]
        for index, contained in enumerate(in_target):
            if contained:
                local_x = self.privileged_cookie_target_position(index, target_body=target_body)[0]
                counts[int(np.argmin(np.abs(columns - local_x)))] += 1
        return tuple(counts)

    def _target_fill_is_valid(
        self, in_target: tuple[bool, ...], *, target_body: int | None = None
    ) -> bool:
        return self.target_column_counts(in_target, target_body=target_body) == (10, 10)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["target_layout"] = "2x10"
        if "cookies_in_target_masks" in info:
            counts = [
                self.target_column_counts(mask, target_body=body)
                for mask, body in zip(info["cookies_in_target_masks"], self.target_box_bodies,
                                      strict=True)
            ]
            info["target_column_counts"] = counts[0]
            info["scenario_metrics"]["target_column_counts_by_box"] = counts
        else:
            info["target_column_counts"] = self.target_column_counts(info["cookies_in_target_mask"])
            if "scenario_metrics" in info:
                info["scenario_metrics"]["target_column_counts_by_box"] = [info["target_column_counts"]]
        return observation, reward, terminated, truncated, info
