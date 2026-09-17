from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .paths import default_config_path

# Upright pieces: 50 mm wide, 19/3 mm thick, 25 mm high.
COOKIE_HALF_SIZE = (0.025, 0.0095 / 3, 0.0125)

# Every one of the four corners that faces a neighbour in the row is chamfered,
# so the gap between two neighbours flares open into a V at the top.  A 6.35 mm
# thick jaw pad can then slide down that V instead of landing flat on a
# neighbour's top face, which is what the pad did before and why no Cookie could
# be pinched.  COOKIE_CHAMFER_M is (cut depth along y, cut span along z).
#
# The cut is symmetric top and bottom, so it also sets the *base* width:
#     2 * (COOKIE_HALF_SIZE[1] - COOKIE_CHAMFER_M[0])
# A deeper cut opens a longer V but leaves a narrower base to stand on.  At
# 2 mm the base is still 2.33 mm; past ~2.2 mm it becomes a knife edge.
#
# The span pulls two ways:
#   * it has to be long enough that the groove still clears the pad where the pad
#     bottom reaches (~4.1 mm below the top face), because the groove narrows with
#     depth:  groove(d) = gap + 2a * max(0, 1 - d / span);
#   * but shortening it is what lets the pads down and widens the face they bite.
#
# Measured, with the pads descending from the top and the target Cookie flanked by
# neighbours (b = span in mm, reach in mm below the top face, face = the widest
# Cookie surface a pad actually touched):
#
#     span  reach   face     outcome
#     10    0.02    3.93     no grip at all -- the pads jammed on a ledge before
#                           reaching the planned depth
#      8    5.12    4.73     held, reached full depth
#      6    2.57    3.93     lifted but the episode later failed
#
# So 8 mm: long enough to keep the groove clear, short enough that the staircase's
# first ledge sits below the pinch and the pads meet a full-thickness band.
# The span cannot exceed COOKIE_HALF_SIZE[2], where the top and bottom cuts meet.
COOKIE_CHAMFER_M = (0.002, 0.008)

# How the chamfer is turned into collision geometry.
#
# A convex hull gives the smooth shape but MuJoCo builds only ONE contact point
# against it (a box gets four), and that single point has to carry the whole load
# while providing no rotational restraint.  The solver's impulses then fling a
# Cookie at metres per second -- measured 86-133 m/s, tripping the joint-velocity
# safety stop during placement -- even though the actual penetration was under a
# millimetre.  Nothing about the contacts' stiffness fixed it.
#
# So collision is built from a stack of boxes instead: a full-thickness band in
# the middle plus `COOKIE_COLLISION_STEPS` steps per side approximating the taper.
# Boxes give four contact points per pair, which is what the solver wants, and
# geoms inside one body do not collide with each other, so the stack acts as a
# union.  Each step is sized to the *narrower* end of its slice, so the stack never
# reaches outside the designed hull.  The visual geom stays a smooth mesh.
#
# Five steps put the first ledge (the place a descending pad could catch) 7.5 mm
# below the top face, well past the ~4.1 mm a pinch reaches.
COOKIE_COLLISION_STEPS = 5

# Row pitch along y = full thickness + the nominal gap.  The gap is what the pad
# has to displace, so it must satisfy, with clearance c:
#     gap + 2 * COOKIE_CHAMFER_M[0] >= pad_thickness + c
#
# The binding number is not that one though -- it is the groove where the pad's
# *bottom* sits, ~4.1 mm below the top face, because the groove narrows with
# depth.  At gap 4 mm that came to 6.371 mm against the 6.35 mm pad: 21 um of
# clearance, so every descent had to wedge in hard, and the reaction visibly
# pushed the arm sideways (2.3 mm of drift).  At 5 mm there is 1.02 mm of
# clearance, which is what lets the pads drop in cleanly.  The cost is a longer
# row: 19 pitches + one Cookie is 221.7 mm instead of 202.7 mm, and the bins have
# to be that much wider.
COOKIE_GAP_Y_M = 0.005
COOKIE_PITCH = (0.0504, 2 * COOKIE_HALF_SIZE[1] + COOKIE_GAP_Y_M)

BIN_WALL_THICKNESS_M = 0.006
# Free space between the outermost Cookie's face and the inside of the bin wall.
# Pinching the end row needs the jaw pad to fit in there, so this has to exceed
# the pad thickness (2 x 3.175 mm) with room to spare -- otherwise the pad jams
# against the wall before it can straddle the Cookie.  The row is 19 pitches plus
# one Cookie wide, and the slack is per side.
COOKIE_BIN_SLACK_M = 0.0075
_COOKIE_ROW_HALF_Y = (19 * COOKIE_PITCH[1] + 2 * COOKIE_HALF_SIZE[1]) / 2
SOURCE_BIN_HALF_SIZE_M = (0.1096, _COOKIE_ROW_HALF_Y + COOKIE_BIN_SLACK_M + BIN_WALL_THICKNESS_M)

# Contact stiffness used by every geom in the cookie scene.
#
# MuJoCo builds far fewer contact points against a convex mesh than against a box
# (~1 instead of 4) and each of those is ~8x softer, so the stock compliance lets
# a Cookie sink ~13 mm into whatever it rests on -- straight through the 6 mm bin
# floor.  A shorter time constant stiffens the contact and brings the resting
# penetration back to well under a millimetre.
#
# It cannot be shortened freely: the time constant has to stay comfortably longer
# than the physics timestep (1/physics_hz = 2 ms) or the contact cannot be
# resolved and impulses explode.  At 1 ms -- half the timestep -- a Cookie being
# lowered into the target bin reached 86 m/s and tripped the joint-velocity safety
# stop.  5 ms is 2.5x the timestep and still ~5x stiffer than the original 40 ms.
#
# MuJoCo blends the two surfaces' parameters, so this has to be applied to *both*
# sides; putting it in the scene's `<default>` geom covers the Cookies, the bins,
# the floor and the jaw pads at once.
COOKIE_SCENE_SOLREF = "0.002 1"
COOKIE_SCENE_SOLIMP = "0.99 0.999 0.001 0.5 2"
COOKIE_SOURCE_POSITIONS = tuple(
    (0.175 + (column - 1.5) * COOKIE_PITCH[0], 0.315 + (row - 9.5) * COOKIE_PITCH[1])
    for column in range(4)
    for row in range(20)
)
_SOURCE_ROWS = 20
TARGET_SLOTS_LOCAL = tuple(
    ((column - 0.5) * COOKIE_PITCH[0], (row - 2) * COOKIE_PITCH[1])
    for row in range(5)
    for column in range(2)
)

# The target bin sits across the gap from the source bin with its slot rows on the
# same lattice as the source rows, so a Cookie lifted from a source row drops into
# the slot level with it.  The y offset is snapped onto that lattice instead of
# written as a literal: the old hard-coded -0.01156666666666667 was on-lattice for
# the 6.733 mm pitch only and drifted 1.07 mm off it once the row was widened.
#
# The snap is to a HALF pitch, not a whole one, because the two lattices are
# offset by half a step: source rows sit at `center + (row - 9.5) * pitch` while
# slot rows sit at `bin + (row - 2) * pitch`, so the two interleave.
# See test_dense_dimensions_capacity_walls_and_mirrored_home, which asserts this
# alignment for whatever pitch is configured.
_SOURCE_CENTER_Y = COOKIE_SOURCE_POSITIONS[0][1] + (_SOURCE_ROWS - 0.5) * COOKIE_PITCH[1]
_NOMINAL_TARGET_BIN_Y = -0.01156666666666667
_TARGET_BIN_WORLD_Y = _SOURCE_CENTER_Y + COOKIE_PITCH[1] * (
    round((_NOMINAL_TARGET_BIN_Y - _SOURCE_CENTER_Y) / COOKIE_PITCH[1] - 0.5) + 0.5
)
TARGET_BIN_WORLD_POSITION = (0.1246, _TARGET_BIN_WORLD_Y, 0.753)

# Slack the target bin leaves around its 2x5 slot lattice, per side and per axis.
#
# y needs real room: the jaw pads stay closed around a Cookie until it is released,
# so they descend past the wall with it.  Half a pad is 3.175 mm, and the old
# literal left the pads 3.175 mm INSIDE the wall -- a jam that a stiff contact
# turned into an 86 m/s impulse and a safety stop.  4.2 mm clears the pad with
# about a millimetre to spare.
# x only needs the Cookie to drop in, so a little under a millimetre is enough.
TARGET_BIN_SLACK_M = (0.0005, 0.0042)

# The target bin has to enclose the 2x5 slot lattice, the Cookies in it, the closed
# jaw pads, and a wall's thickness.  Derived rather than written down, because the
# slots are spaced by COOKIE_PITCH and so the bin grows with it.
COOKIE_TARGET_COLUMNS = 2
COOKIE_TARGET_ROWS = 5
_TARGET_SLOT_HALF = np.asarray(
    [
        (COOKIE_TARGET_COLUMNS - 1) / 2 * COOKIE_PITCH[0],
        (COOKIE_TARGET_ROWS - 1) / 2 * COOKIE_PITCH[1],
    ]
)
TARGET_BIN_HALF_SIZE_M = tuple(
    _TARGET_SLOT_HALF
    + np.asarray(COOKIE_HALF_SIZE[:2])
    + np.asarray(TARGET_BIN_SLACK_M)
    + BIN_WALL_THICKNESS_M
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
    cookie_chamfer_m: tuple[float, ...] = COOKIE_CHAMFER_M
    cookie_mass_kg: float = 0.035 / 6
    cookie_friction: tuple[float, ...] = (2.0, 0.03, 0.002)
    cookie_source_positions_m: tuple[tuple[float, ...], ...] = COOKIE_SOURCE_POSITIONS
    cookie_model_z_m: float = 0.7685
    cookie_reset_z_m: float = 0.7688
    bin_wall_thickness_m: float = BIN_WALL_THICKNESS_M
    bin_friction: tuple[float, ...] = (1.2, 0.02, 0.001)
    source_bin_center_m: tuple[float, ...] = (0.175, 0.315)
    # Holds the row plus COOKIE_BIN_SLACK_M per side, and the wall itself.
    source_bin_half_size_m: tuple[float, ...] = SOURCE_BIN_HALF_SIZE_M
    source_bin_wall_height_m: float = 0.036
    source_floor_z_m: float = 0.753
    source_wall_base_z_m: float = 0.750
    # Grows with COOKIE_PITCH and leaves TARGET_BIN_SLACK_M for the pads.
    target_bin_half_size_m: tuple[float, ...] = TARGET_BIN_HALF_SIZE_M
    target_bin_wall_height_m: float = 0.031
    # Match source columns 0/1 and the same row lattice, across the box gap.
    target_bin_world_position_m: tuple[float, ...] = TARGET_BIN_WORLD_POSITION
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
        1.046019,
        1.494815,
        -0.398286,
        -0.866287,
        1.894449,
        -0.248025,
        1.285323,
        1.0,
        -1.046019,
        -1.494815,
        0.398286,
        0.866287,
        -1.894449,
        0.248025,
        -1.285323,
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
    # Per-light shadow and reflection passes dominate frame time on software
    # rasterizers (no GPU), and their cost is independent of camera resolution.
    # Disabling them trades lighting fidelity for roughly four times the frame
    # rate while keeping geometry, materials, and colours intact.
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
        cookie_chamfer_m=_float_tuple(
            cookie_raw, "cookie_chamfer_m", defaults.cookie_chamfer_m
        ),
        cookie_mass_kg=float(cookie_raw.get("cookie_mass_kg", defaults.cookie_mass_kg)),
        cookie_friction=_float_tuple(cookie_raw, "cookie_friction", defaults.cookie_friction),
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
