from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class JointSource:
    name: str
    parent: str
    child: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits: tuple[float, float]
    effort: float
    velocity: float


@dataclass(frozen=True)
class ModelBundle:
    model: mujoco.MjModel
    xml: str
    source_joints: tuple[JointSource, ...]
