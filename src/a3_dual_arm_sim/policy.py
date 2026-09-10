from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .contracts import STATE, ActionMode, EpisodeContext


@runtime_checkable
class Policy(Protocol):
    action_mode: ActionMode

    def reset(self, context: EpisodeContext) -> None: ...

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray: ...

    def close(self) -> None: ...


def load_policy(spec: str) -> Policy:
    """Load a zero-argument policy factory from ``module:factory``."""

    if ":" not in spec:
        raise ValueError("policy must use module:factory syntax")
    module_name, factory_name = spec.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    policy = factory()
    if not isinstance(policy, Policy):
        raise TypeError(f"{spec} did not create an A3 policy plugin")
    return policy


@dataclass
class HoldPolicy:
    """Keep the reset joint pose; useful for model and recorder smoke tests."""

    action_mode: ActionMode = "joint_position"
    _target: np.ndarray | None = None

    def reset(self, context: EpisodeContext) -> None:
        self._target = None

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        if self._target is None:
            self._target = np.asarray(observation[STATE], dtype=np.float64).copy()
        return self._target.copy()

    def close(self) -> None:
        return None


@dataclass
class SineJointPolicy:
    """Small deterministic joint trajectory demonstrating code-driven collection."""

    action_mode: ActionMode = "joint_position"
    amplitude_rad: float = 0.12
    period_steps: int = 120
    _step: int = 0
    _home: np.ndarray | None = None

    def reset(self, context: EpisodeContext) -> None:
        self._step = 0
        self._home = None

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        if self._home is None:
            self._home = np.asarray(observation[STATE], dtype=np.float64).copy()
        action = self._home.copy()
        phase = 2.0 * np.pi * self._step / self.period_steps
        action[0] += self.amplitude_rad * np.sin(phase)
        action[8] -= self.amplitude_rad * np.sin(phase)
        self._step += 1
        return action

    def close(self) -> None:
        return None


def make_hold_policy() -> HoldPolicy:
    return HoldPolicy()


def make_sine_policy() -> SineJointPolicy:
    return SineJointPolicy()

