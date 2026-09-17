from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import ActionMode, EpisodeContext

MOTION_KEYS: dict[str, tuple[int, float]] = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "a": (1, 1.0),
    "d": (1, -1.0),
    "r": (2, 1.0),
    "f": (2, -1.0),
    "i": (3, 1.0),
    "k": (3, -1.0),
    "j": (4, 1.0),
    "l": (4, -1.0),
    "u": (5, 1.0),
    "o": (5, -1.0),
}


@dataclass
class KeyboardTeleopPolicy:
    """Thread-safe continuous Cartesian commands for the teleop control panel."""

    action_mode: ActionMode = "cartesian_delta"
    selected: str = "left"
    grippers: list[float] = field(default_factory=lambda: [1.0, 1.0])
    recording: bool = True
    recording_available: bool = True
    speed: float = 0.5
    gripper_step: float = 0.04
    stop_requested: bool = False
    discard_requested: bool = False
    emergency_requested: bool = False

    def __post_init__(self) -> None:
        self._pending = np.zeros((2, 6), dtype=np.float64)
        self._pressed: set[str] = set()
        self._lock = threading.RLock()

    def reset(self, context: EpisodeContext) -> None:
        with self._lock:
            self._pending.fill(0.0)
            self._pressed.clear()
            self.stop_requested = False
            self.discard_requested = False
            self.emergency_requested = False

    def _targets(self) -> tuple[int, ...]:
        if self.selected == "both":
            return (0, 1)
        return (0,) if self.selected == "left" else (1,)

    def select(self, selected: str) -> None:
        if selected not in ("left", "right", "both"):
            raise ValueError(f"unsupported arm selection: {selected}")
        with self._lock:
            self.selected = selected

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self.speed = float(np.clip(speed, 0.1, 1.0))

    def set_key_state(self, key: str, pressed: bool) -> None:
        """Track key/button state so motion continues until release."""
        key = key.lower()
        with self._lock:
            if key in MOTION_KEYS or key in ("[", "]"):
                if pressed:
                    self._pressed.add(key)
                else:
                    self._pressed.discard(key)
                return
            if pressed:
                self._handle_command_locked(key)

    def handle_key(self, keycode: int) -> None:
        """Compatibility input for both MuJoCo GLFW viewer and tests."""
        glfw_map = {
            265: "up",
            264: "down",
            263: "left",
            262: "right",
            266: "prior",
            267: "next",
            32: " ",
            256: "q",
        }
        if keycode in glfw_map:
            key = glfw_map[keycode]
        else:
            try:
                key = chr(keycode).lower()
            except (ValueError, OverflowError):
                return
        with self._lock:
            if key in MOTION_KEYS:
                axis, value = MOTION_KEYS[key]
                for target in self._targets():
                    self._pending[target, axis] = value
            elif key in ("[", "]"):
                self._adjust_grippers_locked(-0.1 if key == "[" else 0.1)
            else:
                self._handle_command_locked(key)

    def _handle_command_locked(self, key: str) -> None:
        if key == "1":
            self.selected = "left"
        elif key == "2":
            self.selected = "right"
        elif key == "3":
            self.selected = "both"
        elif key == "p" and self.recording_available:
            self.recording = not self.recording
        elif key == "q":
            self.stop_requested = True
        elif key == "x":
            self.discard_requested = True
            self.stop_requested = True
        elif key == " ":
            self.emergency_requested = True

    def _adjust_grippers_locked(self, delta: float) -> None:
        for target in self._targets():
            self.grippers[target] = float(
                np.clip(self.grippers[target] + delta, -1.0, 1.0)
            )

    def request_stop(self, *, discard: bool = False) -> None:
        with self._lock:
            self.discard_requested = discard
            self.stop_requested = True
            self._pressed.clear()

    def request_emergency_stop(self) -> None:
        with self._lock:
            self.emergency_requested = True
            self._pressed.clear()

    def toggle_recording(self) -> None:
        with self._lock:
            if self.recording_available:
                self.recording = not self.recording

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "selected": self.selected,
                "speed": self.speed,
                "recording": self.recording,
                "recording_available": self.recording_available,
                "emergency": self.emergency_requested,
                "grippers": tuple(self.grippers),
            }

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        with self._lock:
            motion = self._pending.copy()
            self._pending.fill(0.0)
            targets = self._targets()
            for key in self._pressed:
                if key not in MOTION_KEYS:
                    continue
                axis, value = MOTION_KEYS[key]
                for target in targets:
                    motion[target, axis] += value
            motion = np.clip(motion, -1.0, 1.0) * self.speed
            gripper_delta = 0.0
            if "[" in self._pressed:
                gripper_delta -= self.gripper_step * self.speed
            if "]" in self._pressed:
                gripper_delta += self.gripper_step * self.speed
            if gripper_delta:
                self._adjust_grippers_locked(gripper_delta)
            return np.asarray(
                [*motion[0], self.grippers[0], *motion[1], self.grippers[1]],
                dtype=np.float64,
            )

    def close(self) -> None:
        with self._lock:
            self._pressed.clear()
