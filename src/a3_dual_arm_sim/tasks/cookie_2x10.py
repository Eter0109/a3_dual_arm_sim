"""Independent 2x10 target task; the 2x5 environment remains the default."""

from dataclasses import dataclass

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

    def __init__(self, config=None, **kwargs):
        super().__init__(config or resource_root() / "configs/cookie_2x10.yaml", **kwargs)
        if self.config.cookie_transfer.target_rows != 10:
            raise ValueError("A3Cookie2x10Env requires the separate 2x10 scene configuration")

    def reset(self, *, seed=None, options=None):
        self._source_initial_rectangles = []
        self._source_pose_resamples = 0
        observation, info = super().reset(seed=seed, options=options)
        self.randomization_metadata["source_pose_resamples"] = self._source_pose_resamples
        return observation, info

    def _sample_source_cookie_pose(self, index, base_position, randomize):
        # Intersecting side edges can interlock the compound collision slabs
        # when the expert compresses a row. Reject these initial placements,
        # preserving the configured translation and yaw ranges.
        half_size = self.COOKIE_HALF_SIZE[:2]
        for _ in range(100):
            position, quaternion = super()._sample_source_cookie_pose(
                index, base_position, randomize
            )
            c, s = 1 - 2 * quaternion[3] ** 2, 2 * quaternion[0] * quaternion[3]
            rotation = np.array([[c, -s], [s, c]])
            center = np.asarray(position[:2])
            valid = True
            if randomize:
                for other_center, other_rotation in self._source_initial_rectangles:
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
                self._source_initial_rectangles.append((center, rotation))
                return position, quaternion
            self._source_pose_resamples += 1
        raise RuntimeError(
            f"cannot sample a clear source pose for Cookie {index} within the configured noise"
        )

    def target_column_counts(self, in_target: tuple[bool, ...]) -> tuple[int, int]:
        columns = np.unique(np.asarray(self.TARGET_SLOTS_LOCAL)[:, 0])
        counts = [0, 0]
        for index, contained in enumerate(in_target):
            if contained:
                local_x = self.privileged_cookie_target_position(index)[0]
                counts[int(np.argmin(np.abs(columns - local_x)))] += 1
        return tuple(counts)

    def _target_fill_is_valid(self, in_target: tuple[bool, ...]) -> bool:
        return self.target_column_counts(in_target) == (10, 10)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["target_layout"] = "2x10"
        info["target_column_counts"] = self.target_column_counts(info["cookies_in_target_mask"])
        return observation, reward, terminated, truncated, info
