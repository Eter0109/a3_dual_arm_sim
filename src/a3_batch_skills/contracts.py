from dataclasses import dataclass
from enum import Enum
import numpy as np


class Phase(str, Enum):
    LOCATE = "LOCATE"
    ALIGN_PICK = "ALIGN_PICK_ENTRY"
    TRANSFER = "PICK_PLACE_FIVE"
    ALIGN = "PRECISE_PLACEMENT"
    VERIFY = "VERIFY_PLACEMENT"
    ADJUST = "ADJUST_PLACEMENT"
    RETREAT = "RETREAT"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int = 5
    placement_mode: str = "contained_unstacked"
    adjust_placement: bool = True
    align_policy_entry: bool = False
    floor_clearance_m: float = 0.004
    containment_tolerance_m: float = 0.004
    verify_steps: int = 20
    settle_timeout_steps: int = 60
    max_adjustments: int = 1
    max_batch_steps: int = 750
    placement_timeout_steps: int = 350
    takeover_distance_m: float = 0.10
    adjustment_max_offset_m: float = 0.015
    adjustment_max_tilt_deg: float = 25.0
    max_pad_force_n: float = 12.0
    source_column: int | str = 1
    policy_scope: str = "transfer"

    def __post_init__(self):
        if self.batch_size != 5 or not (self.source_column == "auto" or
                                       isinstance(self.source_column, int) and self.source_column >= 1):
            raise ValueError("Use five cookies and a positive source column")
        if self.placement_mode not in ("strict", "contained_unstacked"):
            raise ValueError("Unknown placement mode")
        if self.policy_scope not in ("pick", "transfer"):
            raise ValueError("policy_scope must be pick or transfer")
        if min(self.verify_steps, self.settle_timeout_steps, self.max_batch_steps,
               self.placement_timeout_steps) < 1 or self.max_adjustments < 0:
            raise ValueError("Invalid step limits")
        values = (self.floor_clearance_m, self.containment_tolerance_m,
                  self.adjustment_max_offset_m,
                  self.takeover_distance_m, self.max_pad_force_n)
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("Tolerances and limits must be finite and positive")
        if not 0 < self.adjustment_max_tilt_deg < 90:
            raise ValueError("Invalid adjustment tilt limit")


@dataclass(frozen=True)
class BatchGoal:
    index: int
    cookie_ids: tuple[int, ...]
    target_positions: np.ndarray  # target-box frame, one row per object

    def __post_init__(self):
        positions = np.asarray(self.target_positions, dtype=float).copy()
        if self.index < 0 or len(self.cookie_ids) != 5 or len(set(self.cookie_ids)) != 5:
            raise ValueError("A goal requires five distinct objects")
        if positions.shape != (5, 3) or not np.isfinite(positions).all():
            raise ValueError("Invalid target positions")
        positions.setflags(write=False)
        object.__setattr__(self, "target_positions", positions)

    @property
    def instruction(self):
        return "Transfer one batch of five cookies into the target box, release, and retreat."


@dataclass(frozen=True)
class CookieState:
    identifier: int
    position: np.ndarray
    rotation: np.ndarray
    linear_speed: float
    angular_speed: float
    in_source: bool
    finger_contact: bool


@dataclass(frozen=True)
class SceneState:
    cookies: dict[int, CookieState]
    box_position: np.ndarray
    box_rotation: np.ndarray
    tool_position: np.ndarray
    tool_rotation: np.ndarray
    half_size: np.ndarray
    inner_half_size: np.ndarray
    floor_top: float
    wall_height: float
    held_ids: frozenset[int]
    pad_forces: tuple[float, float]
    safety_stop: bool
    source: str

    def local(self, ids):
        return np.asarray([self.box_rotation.T @ (self.cookies[i].position - self.box_position)
                           for i in ids])

    def world(self, local):
        return self.box_position + self.box_rotation @ np.asarray(local)
