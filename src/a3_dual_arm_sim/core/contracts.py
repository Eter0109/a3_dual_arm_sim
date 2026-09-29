from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

LEFT_JOINTS = (
    "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P",
    "L_WRIST_R", "L_WRIST_P", "L_WRIST_Y",
)
RIGHT_JOINTS = (
    "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P",
    "R_WRIST_R", "R_WRIST_P", "R_WRIST_Y",
)
ARM_JOINTS = LEFT_JOINTS + RIGHT_JOINTS
JOINT_ACTION_DIM = 16
CARTESIAN_ACTION_DIM = 14
ActionMode = Literal["joint_position", "cartesian_delta"]

FRONT_IMAGE = "observation.images.front"
LEFT_WRIST_IMAGE = "observation.images.left_wrist"
RIGHT_WRIST_IMAGE = "observation.images.right_wrist"
STATE = "observation.state"
VELOCITY = "observation.velocity"
EEF_POSE = "observation.eef_pose"
FORCE = "observation.force"


@dataclass(frozen=True)
class EpisodeContext:
    seed: int
    task: str
    action_mode: ActionMode


class ContractError(ValueError):
    pass


def validate_action(action: Any, mode: ActionMode) -> NDArray[np.float64]:
    expected = JOINT_ACTION_DIM if mode == "joint_position" else CARTESIAN_ACTION_DIM
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (expected,):
        raise ContractError(f"{mode} action must have shape ({expected},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ContractError("action contains NaN or infinity")
    if mode == "cartesian_delta" and np.any(np.abs(value) > 1.0):
        raise ContractError("cartesian_delta action values must remain in [-1, 1]")
    return np.ascontiguousarray(value)


def validate_observation(observation: Mapping[str, Any]) -> None:
    shapes = {
        STATE: (16,), VELOCITY: (16,), EEF_POSE: (14,), FORCE: (18,),
    }
    for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE):
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ContractError(f"{key} must be uint8 HxWx3 RGB")
    for key, shape in shapes.items():
        value = np.asarray(observation[key])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ContractError(f"{key} must be a finite vector with shape {shape}")
    if not np.isfinite(float(observation["time"])):
        raise ContractError("time must be finite")
    if not isinstance(observation["safety_stop"], (bool, np.bool_)):
        raise ContractError("safety_stop must be boolean")
