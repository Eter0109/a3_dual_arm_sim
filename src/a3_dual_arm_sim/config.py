from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import default_config_path

# Upright pieces: 50 mm wide, 19/3 mm thick, 25 mm high. A 0.4 mm
# clearance avoids initial interpenetration while keeping the rows dense.
COOKIE_HALF_SIZE = (0.025, 0.0095 / 3, 0.0125)
COOKIE_EDGE_BEVEL_M = 0.0025
COOKIE_PITCH = (0.0504, 0.019 / 3 + 0.0004)
COOKIE_SOURCE_POSITIONS = tuple(
    (0.175 + (column - 1.5) * COOKIE_PITCH[0], 0.315 + (row - 9.5) * COOKIE_PITCH[1])
    for column in range(4)
    for row in range(20)
)
TARGET_SLOTS_LOCAL = tuple(
    ((column - 0.5) * COOKIE_PITCH[0], (row - 2) * COOKIE_PITCH[1])
    for row in range(5)
    for column in range(2)
)


@dataclass(frozen=True)
class CameraConfig:
    """Prototype camera geometry; replace it with calibrated poses later."""

    calibration_status: str = "prototype_estimate"
    workspace_target_m: tuple[float, ...] = (0.15, 0.34, 0.80)
    front_position_m: tuple[float, ...] = (1.30, -0.80, 1.55)
    front_fovy_deg: float = 58.0
    left_wrist_position_m: tuple[float, ...] = (0.0, 0.045, 0.085)
    left_wrist_target_m: tuple[float, ...] = (0.0, 0.245, 0.085)
    right_wrist_position_m: tuple[float, ...] = (0.0, -0.045, -0.085)
    right_wrist_target_m: tuple[float, ...] = (0.0, -0.245, -0.085)
    wrist_fovy_deg: float = 70.0


@dataclass(frozen=True)
class CookieSceneConfig:
    """Prototype cookie-transfer values copied from the working example."""

    calibration_status: str = "prototype_estimate"
    base_height_m: float = 1.10
    box_lift_m: float = 0.030
    box_tilt_deg: float = 8.0
    left_gripper_kp: float = 400.0
    right_gripper_kp: float = 2000.0
    right_grasp_pitch_deg: float = 0.0
    cookie_half_size_m: tuple[float, ...] = COOKIE_HALF_SIZE
    cookie_edge_bevel_m: float = COOKIE_EDGE_BEVEL_M
    cookie_collision_mode: str = "mesh"
    cookie_mass_kg: float = 0.035 / 6
    cookie_friction: tuple[float, ...] = (2.0, 0.03, 0.002)
    cookie_solref: tuple[float, ...] = (0.01, 1.0)
    cookie_solimp: tuple[float, ...] = (0.9, 0.95, 0.001, 0.5, 2.0)
    cookie_source_positions_m: tuple[tuple[float, ...], ...] = COOKIE_SOURCE_POSITIONS
    cookie_model_z_m: float = 0.7685
    cookie_reset_z_m: float = 0.7688
    bin_wall_thickness_m: float = 0.006
    bin_friction: tuple[float, ...] = (1.2, 0.02, 0.001)
    source_bin_center_m: tuple[float, ...] = (0.175, 0.315)
    source_bin_half_size_m: tuple[float, ...] = (0.1096, 0.07613333333333333)
    source_bin_wall_height_m: float = 0.036
    source_floor_z_m: float = 0.753
    source_wall_base_z_m: float = 0.750
    target_bin_half_size_m: tuple[float, ...] = (0.0562, 0.022633333333333333)
    target_bin_wall_height_m: float = 0.031
    # Match source columns 0/1 and the same row lattice, across the box gap.
    target_bin_world_position_m: tuple[float, ...] = (0.1246, -0.01156666666666667, 0.753)
    target_bin_attach_position_m: tuple[float, ...] = (-0.284576, -0.120338, 0.077169)
    target_bin_attach_quaternion: tuple[float, ...] = (
        0.368302,
        0.770142,
        -0.144659,
        0.500309,
    )
    target_floor_z_m: float = 0.003
    target_slots_local_m: tuple[tuple[float, ...], ...] = TARGET_SLOTS_LOCAL
    target_slot_tolerance_m: tuple[float, ...] = (0.010, 0.0025)
    deployment_home: tuple[float, ...] = (
        1.294467,
        0.986556,
        -1.192891,
        -1.005229,
        -0.336042,
        0.367167,
        -1.506819,
        1.0,
        -1.294467,
        -0.986556,
        1.192891,
        1.005229,
        0.336042,
        -0.367167,
        1.506819,
        1.0,
    )


@dataclass(frozen=True)
class HomeConfig:
    left: tuple[float, ...] = (0.0, 0.35, 0.0, -0.65, 0.0, 0.25, 0.0)
    right: tuple[float, ...] = (0.0, -0.35, 0.0, 0.65, 0.0, -0.25, 0.0)
    grippers: tuple[float, float] = (0.6, 0.6)


@dataclass(frozen=True)
class SimConfig:
    physics_hz: int = 500
    control_hz: int = 20
    horizon: int = 1000
    image_width: int = 256
    image_height: int = 256
    max_joint_step_rad: float = 0.05
    max_gripper_step: float = 0.08
    cartesian_translation_scale_m: float = 0.025
    cartesian_rotation_scale_rad: float = 0.10
    ik_damping: float = 0.05
    ik_iterations: int = 8
    joint_damping: float = 0.8
    joint_armature: float = 0.02
    object_position_noise_m: float = 0.025
    home: HomeConfig = field(default_factory=HomeConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    cookie_transfer: CookieSceneConfig = field(default_factory=CookieSceneConfig)

    def __post_init__(self) -> None:
        if self.physics_hz <= 0 or self.control_hz <= 0:
            raise ValueError("physics_hz and control_hz must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        if self.horizon < 1 or self.image_width < 1 or self.image_height < 1:
            raise ValueError("horizon and image dimensions must be positive")
        if len(self.home.left) != 7 or len(self.home.right) != 7:
            raise ValueError("each arm home pose must contain seven joints")
        if len(self.home.grippers) != 2:
            raise ValueError("home.grippers must contain left and right values")
        if (
            len(self.cameras.workspace_target_m) != 3
            or len(self.cameras.front_position_m) != 3
            or len(self.cameras.left_wrist_position_m) != 3
            or len(self.cameras.left_wrist_target_m) != 3
            or len(self.cameras.right_wrist_position_m) != 3
            or len(self.cameras.right_wrist_target_m) != 3
        ):
            raise ValueError("camera positions must contain three values")
        if len(self.cookie_transfer.cookie_source_positions_m) < 10:
            raise ValueError("cookie_transfer must contain at least 10 source positions")
        if len(self.cookie_transfer.target_slots_local_m) != 10:
            raise ValueError("cookie_transfer must contain exactly 10 target slots")
        if len(self.cookie_transfer.deployment_home) != 16:
            raise ValueError("cookie_transfer.deployment_home must contain 16 values")

    @property
    def substeps(self) -> int:
        return self.physics_hz // self.control_hz


def load_config(path: str | Path | None = None) -> SimConfig:
    source = Path(path) if path is not None else default_config_path()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    home_raw = raw.pop("home", {})
    camera_raw = raw.pop("cameras", {})
    cookie_raw = raw.pop("cookie_transfer", {})
    home = HomeConfig(
        left=tuple(float(v) for v in home_raw.get("left", HomeConfig.left)),
        right=tuple(float(v) for v in home_raw.get("right", HomeConfig.right)),
        grippers=tuple(float(v) for v in home_raw.get("grippers", HomeConfig.grippers)),
    )
    cameras = CameraConfig(
        calibration_status=str(camera_raw.get("calibration_status", "prototype_estimate")),
        workspace_target_m=_float_tuple(camera_raw, "workspace_target_m", (0.15, 0.34, 0.80)),
        front_position_m=_float_tuple(camera_raw, "front_position_m", (0.95, -0.55, 1.28)),
        front_fovy_deg=float(camera_raw.get("front_fovy_deg", 52.0)),
        left_wrist_position_m=_float_tuple(
            camera_raw, "left_wrist_position_m", (0.0, 0.045, 0.085)
        ),
        left_wrist_target_m=_float_tuple(camera_raw, "left_wrist_target_m", (0.0, 0.160, -0.010)),
        right_wrist_position_m=_float_tuple(
            camera_raw, "right_wrist_position_m", (0.0, -0.045, -0.085)
        ),
        right_wrist_target_m=_float_tuple(
            camera_raw, "right_wrist_target_m", (0.0, -0.160, 0.010)
        ),
        wrist_fovy_deg=float(camera_raw.get("wrist_fovy_deg", 70.0)),
    )
    defaults = CookieSceneConfig()
    cookie_transfer = CookieSceneConfig(
        calibration_status=str(cookie_raw.get("calibration_status", defaults.calibration_status)),
        cookie_half_size_m=_float_tuple(
            cookie_raw, "cookie_half_size_m", defaults.cookie_half_size_m
        ),
        cookie_edge_bevel_m=float(
            cookie_raw.get("cookie_edge_bevel_m", defaults.cookie_edge_bevel_m)
        ),
        cookie_mass_kg=float(cookie_raw.get("cookie_mass_kg", defaults.cookie_mass_kg)),
        cookie_collision_mode=str(
            cookie_raw.get("cookie_collision_mode", defaults.cookie_collision_mode)
        ),
        cookie_friction=_float_tuple(cookie_raw, "cookie_friction", defaults.cookie_friction),
        cookie_solref=_float_tuple(cookie_raw, "cookie_solref", defaults.cookie_solref),
        cookie_solimp=_float_tuple(cookie_raw, "cookie_solimp", defaults.cookie_solimp),
        cookie_source_positions_m=_nested_float_tuple(
            cookie_raw, "cookie_source_positions_m", defaults.cookie_source_positions_m
        ),
        cookie_model_z_m=float(cookie_raw.get("cookie_model_z_m", defaults.cookie_model_z_m)),
        cookie_reset_z_m=float(cookie_raw.get("cookie_reset_z_m", defaults.cookie_reset_z_m)),
        bin_wall_thickness_m=float(
            cookie_raw.get("bin_wall_thickness_m", defaults.bin_wall_thickness_m)
        ),
        bin_friction=_float_tuple(cookie_raw, "bin_friction", defaults.bin_friction),
        source_bin_center_m=_float_tuple(
            cookie_raw, "source_bin_center_m", defaults.source_bin_center_m
        ),
        source_bin_half_size_m=_float_tuple(
            cookie_raw, "source_bin_half_size_m", defaults.source_bin_half_size_m
        ),
        source_bin_wall_height_m=float(
            cookie_raw.get("source_bin_wall_height_m", defaults.source_bin_wall_height_m)
        ),
        source_floor_z_m=float(cookie_raw.get("source_floor_z_m", defaults.source_floor_z_m)),
        source_wall_base_z_m=float(
            cookie_raw.get("source_wall_base_z_m", defaults.source_wall_base_z_m)
        ),
        target_bin_half_size_m=_float_tuple(
            cookie_raw, "target_bin_half_size_m", defaults.target_bin_half_size_m
        ),
        target_bin_wall_height_m=float(
            cookie_raw.get("target_bin_wall_height_m", defaults.target_bin_wall_height_m)
        ),
        base_height_m=float(cookie_raw.get("base_height_m", defaults.base_height_m)),
        box_lift_m=float(cookie_raw.get("box_lift_m", defaults.box_lift_m)),
        box_tilt_deg=float(cookie_raw.get("box_tilt_deg", defaults.box_tilt_deg)),
        left_gripper_kp=float(cookie_raw.get("left_gripper_kp", defaults.left_gripper_kp)),
        right_gripper_kp=float(cookie_raw.get("right_gripper_kp", defaults.right_gripper_kp)),
        right_grasp_pitch_deg=float(cookie_raw.get("right_grasp_pitch_deg", defaults.right_grasp_pitch_deg)),
        target_bin_world_position_m=_float_tuple(
            cookie_raw,
            "target_bin_world_position_m",
            defaults.target_bin_world_position_m,
        ),
        target_bin_attach_position_m=_float_tuple(
            cookie_raw,
            "target_bin_attach_position_m",
            defaults.target_bin_attach_position_m,
        ),
        target_bin_attach_quaternion=_float_tuple(
            cookie_raw,
            "target_bin_attach_quaternion",
            defaults.target_bin_attach_quaternion,
        ),
        target_floor_z_m=float(cookie_raw.get("target_floor_z_m", defaults.target_floor_z_m)),
        target_slots_local_m=_nested_float_tuple(
            cookie_raw, "target_slots_local_m", defaults.target_slots_local_m
        ),
        target_slot_tolerance_m=_float_tuple(
            cookie_raw, "target_slot_tolerance_m", defaults.target_slot_tolerance_m
        ),
        deployment_home=_float_tuple(cookie_raw, "deployment_home", defaults.deployment_home),
    )
    return SimConfig(home=home, cameras=cameras, cookie_transfer=cookie_transfer, **raw)


def _float_tuple(values: dict[str, Any], key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values.get(key, default))


def _nested_float_tuple(
    values: dict[str, Any],
    key: str,
    default: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values.get(key, default))
