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
    #: Positions of the tabletop boxes *beyond* the first two, in fill order, each
    #: one lane pitch further towards -y.  Empty by default, so a single-box or
    #: two-box scene states nothing here.
    #:
    #: The first two keep their own named keys rather than moving into a list: the
    #: shipped configs name them, the relay's controllers refer to box A and box B,
    #: and a two-box scene is the case every measurement was taken on.  A longer
    #: lane is a queue of identical boxes behind them, and it says so here.  Read
    #: the whole lane through :attr:`target_bin_world_positions_m`.
    queue_target_bin_world_positions_m: tuple[tuple[float, ...], ...] = ()
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

    @property
    def target_bin_world_positions_m(self) -> tuple[tuple[float, ...], ...]:
        """Every tabletop box's nominal world position, station first.

        The one list to read a lane from.  It is assembled rather than stored,
        because the first two boxes have their own keys for the reason given on
        :attr:`queue_target_bin_world_positions_m` -- so a scene states its boxes in
        whichever way matches how many it has, and everything downstream sees one
        list either way.
        """

        positions = [self.target_bin_world_position_m]
        if self.spare_target_bin_world_position_m is not None:
            positions.append(self.spare_target_bin_world_position_m)
        positions.extend(self.queue_target_bin_world_positions_m)
        return tuple(positions)

    @property
    def target_bin_count(self) -> int:
        """How many tabletop boxes this scene builds."""

        return len(self.target_bin_world_positions_m)

    @property
    def target_bin_body_names(self) -> tuple[str, ...]:
        """Their body names, in the same order.  See :func:`target_bin_body_name`."""

        return tuple(target_bin_body_name(index) for index in range(self.target_bin_count))

    @property
    def worst_nominal_box_gap_m(self) -> float:
        """The largest per-axis gap between any two tabletop boxes, as configured.

        Negative means they overlap on both axes.  Measured between *outer* extents
        -- the half size plus one wall thickness, the envelope
        ``A3CookieTransferEnv._boxes_are_clear`` uses -- because that is what
        ``min_box_clearance_m`` is documented against, and a check using the inner
        half size would pass a layout the draw check then refuses.  Getting this
        wrong is not hypothetical: the derived lane pitch is exactly
        ``box_extent_y + min_box_clearance``, so it measures as legal on the half
        sizes and illegal on the outer ones.

        The nominal layout has no yaw: a config states positions, not orientations.
        So this is the unrotated version of the draw check rather than a
        reimplementation of it.

        The source bin is not included.  Its position is derived from the lane's
        length (``SceneSpec.source_offset_m``) rather than stated in the config, so
        a config alone cannot say where it will be; the spec checks that pair.

        A single box has no pair to check and reports infinity.
        """

        half_x, half_y = self.target_bin_half_size_m[:2]
        outer_x = half_x + self.bin_wall_thickness_m
        outer_y = half_y + self.bin_wall_thickness_m
        positions = self.target_bin_world_positions_m
        worst = math.inf
        for index, centre in enumerate(positions):
            for other in positions[index + 1 :]:
                worst = min(
                    worst,
                    max(
                        abs(centre[0] - other[0]) - 2.0 * outer_x,
                        abs(centre[1] - other[1]) - 2.0 * outer_y,
                    ),
                )
        return worst


def target_bin_body_name(index: int) -> str:
    """The body name of the ``index``-th tabletop box.

    The first two keep the names they have always had, because the shipped configs,
    the tests and the relay's own controllers all refer to box A and box B by them.
    The queue beyond them is numbered, which is also what says how long a lane is:
    there is no separate count to keep in step.
    """

    if index == 0:
        return "target_bin"
    if index == 1:
        return "spare_target_bin"
    return f"queue_target_bin_{index}"


@dataclass(frozen=True)
class HomeConfig:
    left: tuple[float, ...] = (0.0, 0.35, 0.0, -0.65, 0.0, 0.25, 0.0)
    right: tuple[float, ...] = (0.0, -0.35, 0.0, 0.65, 0.0, -0.25, 0.0)
    grippers: tuple[float, float] = (0.6, 0.6)


@dataclass(frozen=True)
class AxisRange:
    """A uniform draw over ``[low, high]`` for one axis of one thing.

    Ranges are **asymmetric on purpose**.  The reachable region is not centred on
    the nominal pose, and it is not symmetric: measured on the same-column scene,
    the target box can move 30 mm away from the arm but 140 mm towards it, so a
    symmetric +/-30 mm range would throw away 80% of the space the arm can
    actually serve.  Stating both ends costs one more number and lets a range
    follow the workspace.

    ``(0.0, 0.0)`` means "do not move", which is what every scene did before
    randomization existed, and what makes an all-zero config byte-identical to the
    old behaviour.
    """

    low: float = 0.0
    high: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("range bounds must be finite")
        if self.high < self.low:
            raise ValueError(f"range high ({self.high}) is below low ({self.low})")
        if self.high == self.low and self.low != 0.0:
            raise ValueError(
                f"degenerate range ({self.low}, {self.high}) is ambiguous: it would "
                f"either mean 'never move' or 'always move by this much'. Write "
                f"[{self.low}, {self.low} + something] for a fixed offset, or "
                f"change the nominal pose instead."
            )

    @property
    def movable(self) -> bool:
        return self.high > self.low

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def half_span(self) -> float:
        return (self.high - self.low) / 2.0

    def draw(self, rng) -> float:
        """One offset, or exactly zero when the range is empty."""

        if not self.movable:
            return 0.0
        return float(rng.uniform(self.low, self.high))

    def maximum_magnitude(self) -> float:
        return max(abs(self.low), abs(self.high))


@dataclass(frozen=True)
class SceneRandomization:
    """How far each thing in the scene may move between episodes.

    Every value is an :class:`AxisRange`: a uniform draw over ``[low, high]``,
    where the offset is added to the nominal pose.  All-zero means "the exact
    layout", which is what every scene did before this existed, and keeps an
    episode byte-identical to one collected before the feature.

    This is *scene*-level variation, as opposed to the per-Cookie placement jitter
    in ``CookieTaskConfig``: the boxes move, and the Cookies move with the source
    box because they are laid out relative to it.  The distinction matters for what
    a policy can learn -- moving the arm, the source box and the target box changes
    the whole picture and the whole motion, while jittering one Cookie by a pixel
    does not.

    **Where the numbers come from.**  They are measured, not chosen: the arm's
    workspace sets a limit per direction, and the boxes set another between them.
    Measured reach on the same-column scene (offset at which the expert's own
    check starts refusing):

        target box   +x  30 mm   -x 140 mm   +y  90 mm   -y  55 mm
        source box   +x  80 mm   -x 170 mm   +y 230 mm   -y 245 mm

    and the source and target boxes are only 66.9 mm apart, so two ranges that
    move them towards each other are bounded by that, not by the arm.  Configured
    ranges stay well inside both, and :attr:`min_box_clearance_m` rejects a draw
    that would close the gap anyway -- the source box is a mocap body with infinite
    mass, so a collision would shove the target box out of the pose the expert
    planned for rather than being resolved between them.

    Yaw is available on all three boxes now, including the source.  It was
    previously excluded because the batch experts group the layout by ``x``
    (``np.isclose(source[:, 0], x)``); that grouping reads the *configured* layout
    rather than the live poses, so it still resolves rows after a rotation -- what
    needs checking is whether the insertion itself tolerates entering a rotated
    row, which is measured rather than assumed.
    """

    #: Source box translation, per axis in metres.  The Cookies follow it.
    source_bin_x_m: AxisRange = field(default_factory=AxisRange)
    source_bin_y_m: AxisRange = field(default_factory=AxisRange)
    #: Source box yaw in radians, about its own centre.
    source_bin_yaw_rad: AxisRange = field(default_factory=AxisRange)
    #: Working box translation and yaw.  Its slots rotate with it, and the experts
    #: read the live box frame.
    target_bin_x_m: AxisRange = field(default_factory=AxisRange)
    target_bin_y_m: AxisRange = field(default_factory=AxisRange)
    target_bin_yaw_rad: AxisRange = field(default_factory=AxisRange)
    #: Spare box translation and yaw.  Only present in the two-box scene.
    spare_bin_x_m: AxisRange = field(default_factory=AxisRange)
    spare_bin_y_m: AxisRange = field(default_factory=AxisRange)
    spare_bin_yaw_rad: AxisRange = field(default_factory=AxisRange)
    #: Per-joint jitter on the deployment home, giving each episode a different
    #: starting pose.  This is the variation the collection docstring used to call
    #: out as missing: with it at zero the arm starts byte-identical every episode,
    #: so a policy can memorise one joint trajectory.
    arm_home_rad: AxisRange = field(default_factory=AxisRange)
    #: Smallest gap allowed between any two boxes' outer extents.  A draw that
    #: would close it is discarded and redrawn; see :attr:`max_clearance_attempts`.
    min_box_clearance_m: float = 0.015
    #: How many draws to try before giving up.  Exhausting this is a configuration
    #: error -- the ranges are asking for a layout that cannot exist -- and is
    #: raised rather than silently falling back to a less varied episode.
    max_clearance_attempts: int = 200

    @property
    def enabled(self) -> bool:
        """Whether any range can actually move something."""

        return any(axis.movable for name, axis in self.axes().items())

    def axes(self) -> dict[str, AxisRange]:
        """Every ranged quantity by name, for iteration and validation."""

        return {
            "source_bin_x_m": self.source_bin_x_m,
            "source_bin_y_m": self.source_bin_y_m,
            "source_bin_yaw_rad": self.source_bin_yaw_rad,
            "target_bin_x_m": self.target_bin_x_m,
            "target_bin_y_m": self.target_bin_y_m,
            "target_bin_yaw_rad": self.target_bin_yaw_rad,
            "spare_bin_x_m": self.spare_bin_x_m,
            "spare_bin_y_m": self.spare_bin_y_m,
            "spare_bin_yaw_rad": self.spare_bin_yaw_rad,
            "arm_home_rad": self.arm_home_rad,
        }

    def __post_init__(self) -> None:
        if self.min_box_clearance_m < 0 or not math.isfinite(self.min_box_clearance_m):
            raise ValueError("min_box_clearance_m must be finite and non-negative")
        if self.max_clearance_attempts < 1:
            raise ValueError("max_clearance_attempts must be positive")


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
    # Velocity feedback on the arm position servos, as a fraction of the
    # critically damped ratio for the actuator's own gain (see ``build_model``).
    # A fast Cartesian servo needs it: without it a command the joint cannot
    # track is answered by a sustained ring, and the batch expert's fast transit
    # never satisfies its arrival test.  It is a config field rather than a
    # constant because the two-box relay's right-arm push and carry controllers
    # are calibrated against the *un*damped response -- measured, with damping on
    # they let the pinched box twist more than 10 deg against the pads, and the
    # relay's yield falls from 2/3 to 0/4.  So it is a property of the arm an
    # expert was tuned for, and the relay's config turns it off.
    arm_actuator_damping: float = 1.0
    # Which of the batch expert's validated tunings to run; see
    # ``batch_profile``.  "fast" is the one the README's speed section
    # documents; "baseline" is the slower tuning the two-box relay is
    # calibrated around.
    batch_expert_profile: str = "fast"
    #: How many Cookies one grasp takes.  The expert's batch size, and what the
    #: fill's plan is built from: this many Cookies are grasped in a line and placed
    #: into this many consecutive slots of *one* target column, so a batch cannot
    #: span two columns and the value is bounded above by a column's row count
    #: (``ceil(capacity / columns)``) as well as by the jaw travel
    #: (``per_grasp * source_row_pitch - pad_thickness <= 0.085``).
    #:
    #: **Five is currently the only value the grasp itself is verified at**, and the
    #: config refuses anything below it rather than collecting at a yield nobody has
    #: measured.  What was measured, at three seeds each on the same-column scene:
    #: five places ten Cookies in 483-543 steps, while one, two and three all fail at
    #: their *first* batch with the pads closed to the floor and at most 0.06 N of
    #: contact.  The plan is not the problem -- it derives correctly for every size
    #: from 1 to 10, and a test partitions all of them -- so what is missing is the
    #: grasp's own tuning: the pads descend onto the batch's outer Cookies and
    #: compress them, and how far they have to close to build force is a property of
    #: how many bevels are in the line.  Opening this range up means measuring that,
    #: not relaxing this check.
    batch_expert_per_grasp: int = 5
    #: The smallest grasp size whose grasp has been measured to work.  See
    #: :attr:`batch_expert_per_grasp` for the numbers behind it.
    min_verified_batch_expert_per_grasp: int = 5
    #: How far the relay's push moves a filled box out of the filling station, in
    #: metres.  It is the spacing the lane compacts to and therefore what the source
    #: bin's clearance is computed from (see ``SceneSpec.source_offset_m``), so the
    #: two have to agree.
    #:
    #: 90 mm is the relay's measured value.  It is deliberately *not* the derived
    #: ``lane_pitch`` (87 mm for this box) even though that would satisfy the 85 mm
    #: clearance the push is verified at: 90 is what the push and the carry were
    #: calibrated against, and changing it by 3 mm would invalidate a yield that took
    #: an 80-minute run to measure.  So it is a parameter with its measured default,
    #: like ``min_push_m``, rather than a formula.
    relay_push_distance_m: float = 0.090
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
        if self.batch_expert_per_grasp < 1:
            raise ValueError(
                f"batch_expert_per_grasp must be at least 1, got {self.batch_expert_per_grasp}"
            )
        if self.relay_push_distance_m <= 0:
            raise ValueError(
                f"relay_push_distance_m must be positive, got {self.relay_push_distance_m}"
            )
        # The plan supports any size; the *grasp* does not, and this refuses the gap
        # between them rather than collecting episodes nobody has measured.
        if self.batch_expert_per_grasp < self.min_verified_batch_expert_per_grasp:
            raise ValueError(
                f"batch_expert_per_grasp is {self.batch_expert_per_grasp}, but only "
                f"{self.min_verified_batch_expert_per_grasp} and above have a measured "
                f"grasp: at 1, 2 and 3 Cookies the fill closes the jaws to its floor "
                f"and reads at most 0.06 N of pad contact, so it never forms the "
                f"chain the grasp needs, while 5 places all ten Cookies in 483-543 "
                f"steps.  The plan derives correctly for every size from 1 to 10 -- "
                f"what is missing is the grasp's own tuning, which is a measurement "
                f"rather than a parameter"
            )
        # A batch is placed into one column, so it cannot be longer than that
        # column.  Checked here rather than in the expert because it is a property of
        # the box and the grasp size together, and it is knowable from the config
        # alone -- the same reason the nominal box gap is checked here.
        slots = self.cookie_transfer.target_slots_local_m
        if slots:
            columns = len({slot[0] for slot in slots})
            rows = math.ceil(len(slots) / columns)
            if self.batch_expert_per_grasp > rows:
                raise ValueError(
                    f"batch_expert_per_grasp is {self.batch_expert_per_grasp} but a "
                    f"target column holds {rows} rows ({len(slots)} slots over "
                    f"{columns} columns), so a batch could not be placed in one "
                    f"column; use at most {rows}"
                )
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
        for position in self.cookie_transfer.queue_target_bin_world_positions_m:
            if len(position) != 3:
                raise ValueError(
                    "every queue_target_bin_world_positions_m entry must contain "
                    f"three values, got {position!r}"
                )
        # The *nominal* layout has to be clear when the scene draws within a
        # clearance, because it is the layout a fixed-layout run uses and the layout
        # a lane's queue is placed at before anything moves it.  A config with no
        # ranges is exempt: nothing is drawn and nothing is checked, so its boxes
        # stay exactly where it put them -- which is how `cookie_two_box.yaml` sits
        # with its two boxes 12 mm apart against a 15 mm clearance it never draws
        # against.
        #
        # A tolerance, because a derived lane pitch is exactly
        # `box_extent_y + min_box_clearance` and float arithmetic puts that a few
        # ulps under.  A layout that equals the clearance is legal.
        if self.randomization.enabled:
            gap = self.cookie_transfer.worst_nominal_box_gap_m
            if gap < self.randomization.min_box_clearance_m - 1e-9:
                raise ValueError(
                    f"the nominal box layout leaves only {gap * 1000:.1f} mm between "
                    f"the closest pair of outer extents, against the "
                    f"{self.randomization.min_box_clearance_m * 1000:.1f} mm "
                    f"min_box_clearance_m this scene draws within; move the boxes "
                    f"further apart in the config, or lower min_box_clearance_m"
                )
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
    randomization_raw = dict(raw.pop("randomization", {}) or {})
    randomization = SceneRandomization(
        **{name: _axis_range(randomization_raw, name) for name in SceneRandomization().axes()},
        min_box_clearance_m=float(
            randomization_raw.get("min_box_clearance_m", SceneRandomization().min_box_clearance_m)
        ),
        max_clearance_attempts=int(
            randomization_raw.get(
                "max_clearance_attempts", SceneRandomization().max_clearance_attempts
            )
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
        right_wrist_target_m=_float_tuple(camera_raw, "right_wrist_target_m", (0.0, -0.160, 0.010)),
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
        right_grasp_pitch_deg=float(
            cookie_raw.get("right_grasp_pitch_deg", defaults.right_grasp_pitch_deg)
        ),
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
        queue_target_bin_world_positions_m=_nested_float_tuple(
            cookie_raw,
            "queue_target_bin_world_positions_m",
            defaults.queue_target_bin_world_positions_m,
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


def _axis_range(values: dict[str, Any], key: str) -> AxisRange:
    """Parse one randomization range, accepting either form.

    ``[low, high]`` is the explicit form and what the configs use.  A bare number
    is accepted as symmetric ``[-value, +value]`` because that is the natural thing
    to write for the arm and yaw ranges, where the reachable region *is* centred
    and a signed pair would be noise.
    """

    if key not in values:
        return AxisRange()
    raw = values[key]
    if isinstance(raw, (int, float)):
        return AxisRange(low=-abs(float(raw)), high=abs(float(raw)))
    pair = tuple(float(value) for value in raw)
    if len(pair) != 2:
        raise ValueError(f"randomization.{key} must be a number or [low, high], got {raw!r}")
    return AxisRange(low=pair[0], high=pair[1])


def _nested_float_tuple(
    values: dict[str, Any],
    key: str,
    default: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values.get(key, default))
