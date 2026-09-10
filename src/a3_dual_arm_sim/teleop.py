from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import ActionMode, EpisodeContext


@dataclass
class KeyboardTeleopPolicy:
    """One-shot Cartesian keyboard commands for MuJoCo's passive viewer callback."""

    action_mode: ActionMode = "cartesian_delta"
    selected: str = "left"
    grippers: list[float] = field(default_factory=lambda: [1.0, 1.0])
    recording: bool = True
    stop_requested: bool = False
    discard_requested: bool = False
    emergency_requested: bool = False

    def __post_init__(self) -> None:
        self._pending = np.zeros((2, 6), dtype=np.float64)

    def reset(self, context: EpisodeContext) -> None:
        self._pending.fill(0.0)
        self.stop_requested = False
        self.discard_requested = False
        self.emergency_requested = False

    def handle_key(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except (ValueError, OverflowError):
            return
        if key == "1":
            self.selected = "left"
            return
        if key == "2":
            self.selected = "right"
            return
        if key == "3":
            self.selected = "both"
            return
        targets = (0, 1) if self.selected == "both" else ((0,) if self.selected == "left" else (1,))
        motion = {
            "w": (0, 1.0), "s": (0, -1.0), "a": (1, 1.0), "d": (1, -1.0),
            "r": (2, 1.0), "f": (2, -1.0), "i": (3, 1.0), "k": (3, -1.0),
            "j": (4, 1.0), "l": (4, -1.0), "u": (5, 1.0), "o": (5, -1.0),
        }
        if key in motion:
            axis, value = motion[key]
            for target in targets:
                self._pending[target, axis] = value
        elif key in ("[", "]"):
            delta = -0.1 if key == "[" else 0.1
            for target in targets:
                self.grippers[target] = float(np.clip(self.grippers[target] + delta, -1.0, 1.0))
        elif key == "p":
            self.recording = not self.recording
        elif key == "q":
            self.stop_requested = True
        elif key == "x":
            self.discard_requested = True
            self.stop_requested = True
        elif key == " ":
            self.emergency_requested = True

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        action = np.r_[self._pending[0], self.grippers[0], self._pending[1], self.grippers[1]]
        self._pending.fill(0.0)
        return action

    def close(self) -> None:
        return None

