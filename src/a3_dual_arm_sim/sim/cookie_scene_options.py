"""Opt-in scene variations shared by the 2x5 and 2x10 tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from a3_dual_arm_sim.config import CookieSceneConfig

COOKIES_PER_GROUP = 5
LEAN_SUPPORT_HALF_HEIGHT_M = 0.0015


@dataclass(frozen=True)
class CookieSceneOptions:
    source_gap_groups: int | tuple[int, int] = 0
    target_box_count: int = 1
    source_tilt_deg: float = 0.0
    cookie_crush_force_n: float | None = None

    def __post_init__(self) -> None:
        groups = self.source_gap_groups
        if type(groups) is int:
            groups = (groups, groups)
        if (
            not isinstance(groups, (list, tuple))
            or len(groups) != 2
            or any(type(n) is not int for n in groups)
            or not (tuple(groups) == (0, 0) or 2 <= groups[0] <= groups[1] <= 5)
        ):
            raise ValueError("scene.source_gap_groups must be 0, 2–5, or a range such as [2, 5]")
        object.__setattr__(self, "source_gap_groups", tuple(groups))
        if type(self.target_box_count) is not int or not 1 <= self.target_box_count <= 4:
            raise ValueError("scene.target_box_count must be an integer from 1 to 4")
        if not _finite_number(self.source_tilt_deg) or not 0 <= self.source_tilt_deg <= 8:
            raise ValueError("scene.source_tilt_deg must be between 0 and 8")
        force = self.cookie_crush_force_n
        if force is not None and (not _finite_number(force) or force <= 0):
            raise ValueError("scene.cookie_crush_force_n must be null or a positive force in N")

    @classmethod
    def from_settings(cls, settings: dict) -> CookieSceneOptions:
        values = settings.get("scene", {})
        if not isinstance(values, dict):
            raise ValueError("scene must be a mapping")  # noqa: TRY004
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid scene options: {exc}") from exc

    @property
    def enabled(self) -> bool:
        return bool(
            self.source_gap_groups[1]
            or self.target_box_count > 1
            or self.source_tilt_deg
            or self.cookie_crush_force_n is not None
        )

    def feasible_gap_range(self, scene: CookieSceneConfig) -> tuple[int, int]:
        stock = len(scene.cookie_source_positions_m)
        target = len(scene.target_slots_local_m) * self.target_box_count
        low, high = self.source_gap_groups
        high = min(high, (stock - target) // COOKIES_PER_GROUP)
        if high < low:
            raise ValueError(
                f"scene needs {target} cookies for {self.target_box_count} target boxes, "
                f"but {low} gap groups leave only {stock - low * COOKIES_PER_GROUP}"
            )
        return low, high

    def target_boxes(
        self, scene: CookieSceneConfig
    ) -> tuple[tuple[str, tuple[float, ...], float], ...]:
        """Nominal world poses for the standard left-arm tabletop workspace."""
        if self.target_box_count > 1 and scene.spare_target_bin_world_position_m is not None:
            raise ValueError("scene.target_box_count cannot be combined with the legacy spare box")
        x, y, z = scene.target_bin_world_position_m
        if self.target_box_count == 1:
            return (("target_bin", (x, y, z), 0.0),)

        # Offsets from the native target pose. Use the larger 2x10 footprint
        # and reserve room for the advanced profile's translation and yaw.
        offsets = (
            (-0.043, -0.008, 0),
            (-0.165, 0.319, -90),
            (-0.218, 0.473, -155),
            (-0.380, 0.453, 120),
        )
        return tuple(
            (
                "target_bin" if i == 0 else f"target_bin_{i + 1}",
                (x + dx, y + dy, z),
                math.radians(yaw),
            )
            for i, (dx, dy, yaw) in enumerate(offsets[: self.target_box_count])
        )


def _finite_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
