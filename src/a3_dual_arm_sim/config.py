from __future__ import annotations

import math
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
    left_finger_pad_half_thickness_m: float = 0.003175
    right_gripper_kp: float = 2000.0
    right_grasp_pitch_deg: float = 0.0
    cookie_half_size_m: tuple[float, ...] = COOKIE_HALF_SIZE
    cookie_edge_bevel_m: float = COOKIE_EDGE_BEVEL_M
    cookie_bottom_edge_bevel_m: float | None = None
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
    spare_target_bin_world_position_m: tuple[float, ...] | None = None
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
class SceneRandomization:
    """How far each thing in the scene may move between episodes.

    Every value is a **half-range**: the applied offset is drawn uniformly from
    ``[-value, +value]``.  All-zero means "the exact layout", which is what every
    scene did before this existed.

    This is *scene*-level variation, as opposed to the per-Cookie placement jitter
    in ``CookieTaskConfig``: the boxes move, and the Cookies move with the source
    box because they are laid out relative to it.  The distinction matters for
    what a policy can learn -- moving the arm, the source box and the target box
    changes the whole picture and the whole motion, while jittering one Cookie by a
    pixel does not.

    Why the source box has no yaw: the batch experts find a batch by grouping the
    configured Cookie layout by ``x`` (``np.isclose(source[:, 0], x)``), so a
    rotated source layout would silently stop matching its own rows.  Translating
    the box is free; rotating it would need that grouping rewritten.
    """

    #: Source box translation, half-range per axis.  The Cookies follow it.
    source_bin_xy_m: tuple[float, float] = (0.0, 0.0)
    #: Working box translation and yaw.  Its slots rotate with it, and the experts
    #: read the live box frame, so a modest yaw is safe.
    target_bin_xy_m: tuple[float, float] = (0.0, 0.0)
    target_bin_yaw_rad: float = 0.0
    #: Spare box translation and yaw.  Only present in the two-box scene.
    spare_bin_xy_m: tuple[float, float] = (0.0, 0.0)
    spare_bin_yaw_rad: float = 0.0
    #: Per-joint jitter on the deployment home, giving each episode a different
    #: starting pose.  This is the variation the collection docstring used to call
    #: out as missing: with it at zero the arm starts byte-identical every episode,
    #: so a policy can memorise one joint trajectory.
    arm_home_rad: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether any of the above is non-zero."""

        return any(
            (
                any(self.source_bin_xy_m),
                any(self.target_bin_xy_m),
                self.target_bin_yaw_rad,
                any(self.spare_bin_xy_m),
                self.spare_bin_yaw_rad,
                self.arm_home_rad,
            )
        )

    def __post_init__(self) -> None:
        if len(self.source_bin_xy_m) != 2 or len(self.target_bin_xy_m) != 2:
            raise ValueError("box translation half-ranges must contain two values")
        if len(self.spare_bin_xy_m) != 2:
            raise ValueError("spare_bin_xy_m must contain two values")
        for name in (
            "source_bin_xy_m",
            "target_bin_xy_m",
            "target_bin_yaw_rad",
            "spare_bin_xy_m",
            "spare_bin_yaw_rad",
            "arm_home_rad",
        ):
            values = getattr(self, name)
            values = values if isinstance(values, tuple) else (values,)
            if any(value < 0 or not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class SimConfig:
    physics_hz: int = 500
    control_hz: int = 20
    horizon: int = 1000
    image_width: int = 256
    image_height: int = 256
    # Per-light shadow and reflection passes dominate frame time on software
    # rasterizers (no GPU), and their cost is independent of camera resolution.
    # Disabling them trades lighting fidelity for roughly four times the frame
    # rate while keeping geometry, materials, and colours intact.  `env.use_fast_render()`
    # turns both off; see the `--fast-render` CLI flag.
    render_shadows: bool = True
    render_reflections: bool = True
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
    #: Per-episode scene variation.  All-zero keeps every scene exactly as it was.
    randomization: SceneRandomization = field(default_factory=SceneRandomization)

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
        spare_position = self.cookie_transfer.spare_target_bin_world_position_m
        if spare_position is not None and len(spare_position) != 3:
            raise ValueError("spare_target_bin_world_position_m must contain three values")
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
    randomization_raw = {
        # Defaults come from the dataclass, so a config only states what it wants
        # to vary and an absent section means "the exact layout".
        key: value
        for key, value in (raw.pop("randomization", {}) or {}).items()
    }
    defaults_randomization = SceneRandomization()
    randomization = SceneRandomization(
        source_bin_xy_m=_float_tuple(
            randomization_raw, "source_bin_xy_m", defaults_randomization.source_bin_xy_m
        ),
        target_bin_xy_m=_float_tuple(
            randomization_raw, "target_bin_xy_m", defaults_randomization.target_bin_xy_m
        ),
        target_bin_yaw_rad=float(
            randomization_raw.get("target_bin_yaw_rad", defaults_randomization.target_bin_yaw_rad)
        ),
        spare_bin_xy_m=_float_tuple(
            randomization_raw, "spare_bin_xy_m", defaults_randomization.spare_bin_xy_m
        ),
        spare_bin_yaw_rad=float(
            randomization_raw.get("spare_bin_yaw_rad", defaults_randomization.spare_bin_yaw_rad)
        ),
        arm_home_rad=float(
            randomization_raw.get("arm_home_rad", defaults_randomization.arm_home_rad)
        ),
    )
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
        cookie_bottom_edge_bevel_m=(
            None
            if cookie_raw.get("cookie_bottom_edge_bevel_m") is None
            else float(cookie_raw["cookie_bottom_edge_bevel_m"])
        ),
        cookie_mass_kg=float(cookie_raw.get("cookie_mass_kg", defaults.cookie_mass_kg)),
        left_finger_pad_half_thickness_m=float(
            cookie_raw.get(
                "left_finger_pad_half_thickness_m",
                defaults.left_finger_pad_half_thickness_m,
            )
        ),
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
        spare_target_bin_world_position_m=(
            _float_tuple(
                cookie_raw,
                "spare_target_bin_world_position_m",
                defaults.target_bin_world_position_m,
            )
            if cookie_raw.get("spare_target_bin_world_position_m") is not None
            else None
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
    return SimConfig(
        home=home,
        cameras=cameras,
        cookie_transfer=cookie_transfer,
        randomization=randomization,
        **raw,
    )


def _float_tuple(values: dict[str, Any], key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values.get(key, default))


def _nested_float_tuple(
    values: dict[str, Any],
    key: str,
    default: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values.get(key, default))
