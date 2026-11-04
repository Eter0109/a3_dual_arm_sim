"""Explicit execution-side inputs; no environment, score or contact graph.

Simulator adapters and future hardware drivers construct the same packet.
This is an API/data-flow boundary, not a security sandbox for arbitrary code.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np


def numeric_json(value):
    """Serialize numeric sensor diagnostics without silently hiding bad objects."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError('Unsupported sensor event value: '+type(value).__name__)

CAMERAS = ("front", "left_wrist", "right_wrist")
IMAGE_KEYS = tuple("observation.images." + name for name in CAMERAS)


class SensorInputError(ValueError):
    """Caller must hold/stop; never recover by querying simulation truth."""


def needs_sensor_refresh(step,phase_age,phase,previous_phase):
    # A transition can occur on an odd capture step. It still needs the first
    # stationary image; periodic frame skipping must not suppress calibration.
    return phase_age==0 or step%2==0 or phase!=previous_phase


def frozen_array(value, *, shape=None, dtype=None):
    result = np.asarray(value, dtype=dtype).copy()
    if shape is not None and result.shape != shape:
        raise SensorInputError(f"Expected shape {shape}, received {result.shape}")
    result.setflags(write=False)
    return result


def policy_observation(raw, *, gripper_position_measured):
    """Whitelist model inputs and keep the left arm's eight measured dimensions.

    A commanded gripper position is not a measured position. Hardware lacking
    opening feedback needs a separately validated estimator/model input change.
    """
    if not gripper_position_measured:
        raise SensorInputError("Measured gripper opening is unavailable")
    state = np.asarray(raw["observation.state"])
    if state.shape not in ((8,), (16,)) or not np.isfinite(state).all():
        raise SensorInputError("Expected finite left8 or legacy16 measured state")
    result = {"observation.state": frozen_array(state[:8], dtype=np.float32)}
    for key in IMAGE_KEYS:
        image = np.asarray(raw[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise SensorInputError(f"Expected RGB uint8 image: {key}")
        result[key] = frozen_array(image)
    return MappingProxyType(result)


@dataclass(frozen=True)
class ExecutionPacket:
    """One synchronized sample; wrist poses come from calibrated robot FK."""
    observation: object
    rgbd: object
    calibrations: object
    timestamp: float
    pad_midpoint_world: object = None
    jaw_gap_m: float | None = None

    @classmethod
    def create(cls, raw, rgbd, calibrations, *, timestamp,
               gripper_position_measured=False, pad_midpoint_world=None,
               jaw_gap_m=None):
        observation = policy_observation(raw,
            gripper_position_measured=gripper_position_measured)
        if len(rgbd.frames) != 3 or set(calibrations) != set(CAMERAS):
            raise SensorInputError("Exactly front/left_wrist/right_wrist RGB-D required")
        if not np.isfinite(timestamp):
            raise SensorInputError("Nonfinite robot timestamp")
        if "time" in raw and (not np.isfinite(raw["time"]) or abs(raw["time"]-timestamp) > 1e-6):
            raise SensorInputError("Robot observation timestamp does not match packet")
        if any(abs(f.timestamp-timestamp) > 1e-6 for f in rgbd.frames):
            raise SensorInputError("Unsynchronized RGB-D/robot samples")
        copied = {}
        for name, (position, rotation, fovy) in calibrations.items():
            position = frozen_array(position, shape=(3,), dtype=float)
            rotation = frozen_array(rotation, shape=(3, 3), dtype=float)
            if (not np.isfinite(position).all() or not np.isfinite(rotation).all()
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    or np.linalg.det(rotation) < .999 or not 0 < fovy < 180):
                raise SensorInputError("Invalid camera calibration")
            copied[name] = (position, rotation, float(fovy))
        pad = None if pad_midpoint_world is None else frozen_array(
            pad_midpoint_world, shape=(3,), dtype=float)
        if pad is not None and not np.isfinite(pad).all():
            raise SensorInputError("Nonfinite robot FK")
        if jaw_gap_m is not None and (not np.isfinite(jaw_gap_m) or jaw_gap_m < 0):
            raise SensorInputError("Invalid measured jaw opening")
        return cls(observation, rgbd, MappingProxyType(copied), float(timestamp),
                   pad, None if jaw_gap_m is None else float(jaw_gap_m))

    def validate_age(self, now, max_age_s=.1):
        if (not np.isfinite(now) or not np.isfinite(max_age_s) or max_age_s <= 0
                or now < self.timestamp-1e-6 or now-self.timestamp > max_age_s):
            raise SensorInputError("Stale/future sensor packet; hold and reacquire")


def act_from_packet(controller, packet, *, now, max_age_s=.1):
    """Only this whitelisted packet crosses into the visual batch controller."""
    packet.validate_age(now, max_age_s)
    action = controller.act(packet.observation, packet.calibrations,
        frame=packet.rgbd, pad_midpoint_world=packet.pad_midpoint_world,
        jaw_gap_m=packet.jaw_gap_m)
    if action is None:
        return None  # runner holds its last safe robot targets
    result = np.asarray(action, dtype=float)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise SensorInputError("Expected finite left-arm-only 8D action")
    return result.copy()
