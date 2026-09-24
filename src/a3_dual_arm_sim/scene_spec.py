"""Derive a whole cookie scene's geometry from a handful of key parameters.

Two scenes exist today -- one box and two boxes -- and each is a hand-tuned config:
box sizes typed in as pairs of numbers, a batch size written as the literal ``5``,
a lane distance as ``0.090``, and randomization ranges whose bounds live in a
comment.  A third layout means a third copy of all of it.  This module is the
alternative: state the parameters, derive everything else, and *test the
derivation* instead of each layout.

The five key parameters are :attr:`SceneSpec.boxes`, :attr:`SceneSpec.per_grasp`,
:attr:`SceneSpec.box_capacity`, :attr:`SceneSpec.source_cookies` and the
``randomize`` flag the caller keeps beside them (its ranges are not derived yet;
see below).  Everything geometric follows, and the derivation is anchored: the
spec for ``(boxes=1, per_grasp=5, box_capacity=10, source_cookies=80)`` must
reproduce ``configs/cookie_same_column.yaml``'s ``cookie_transfer`` block, and the
two-box one must reproduce ``configs/cookie_two_box_batch.yaml``'s.  That anchor is
what keeps the formulas honest, and it is not a formality -- writing this module
corrected two things the plan had wrong:

* the source layout's row pitch is the Cookie's thickness plus a **2.5 mm** gap,
  not the ``0.019/3 + 0.0004`` the module constant ``COOKIE_PITCH`` says, because
  that gap is the lane the fingertip descends into and the shipped configs widened
  it when the Cookies got thinner;
* the batch plan is per *column*, not per box; see :mod:`a3_dual_arm_sim.batch_plan`.

**What is derived and what is chosen.**  Deriving everything would be a lie: some
numbers are clearances and margins that someone picked because they work, and some
are larger than the minimum they have to satisfy.  So the split is explicit.  The
derived quantities are forced by the parameters plus the Cookie's geometry, and
they are what the anchor reproduces.  The chosen ones are fields -- station pose,
margins, the lane pitch -- and for each of them the spec validates the *inequality*
it must satisfy and reports the number it missed by.  A field that is merely
recorded is honest; a field pretending to be a formula is not.

**What is not here yet.**  The randomization *ranges* are Phase 5: this module
exposes the bounds they must respect (:meth:`SceneSpec.reach_limits`) but does not
pick values inside them, so ``randomize`` is a caller's flag rather than a field
here.  A parameter nothing reads is worse than an absent one.

The measured numbers the validation uses all live in :class:`MeasuredEnvelopes`,
each with the sweep that produced it, so a bound can be re-taken rather than
trusted.  ``scripts/measure_scene_envelopes.py`` reproduces them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .batch_plan import BatchPlan
from .config import AxisRange

#: Height of the tabletop the boxes stand on, and therefore the z of every box's
#: body.  The table's own top face is at 0.750; the 3 mm is the boxes' floor
#: thickness, which is modelled inside each box body.
TABLE_Z_M = 0.753


@dataclass(frozen=True)
class CookieGeometry:
    """The Cookie's dimensions, and every clearance a layout is built from.

    Each number is either a property of the Cookie mesh or a clearance that was
    measured to work.  None of them is derived: they are the inputs the rest of the
    derivation is a function of.
    """

    #: Cookie half extents: 50 mm wide, 19/3 mm thick, 25 mm high.
    half_size_m: tuple[float, float, float] = (0.025, 0.0095 / 3, 0.0125)
    #: Gap between source columns, along x.  Just enough to avoid initial
    #: interpenetration: the jaws never enter this direction.
    source_column_clearance_m: float = 0.0004
    #: Gap between source rows, along y.  This is the lane the fingertip descends
    #: into, which is why the source pitch is *not* the Cookie's thickness.  The
    #: shipped configs use 2.5 mm (row pitch 0.0088333 against a 0.0063333
    #: thickness), and it is what the jaw-travel bound on ``per_grasp`` is
    #: computed from.
    source_row_clearance_m: float = 0.0025
    #: Gap between the two columns of slots inside a target box.
    target_column_gap_m: float = 0.006
    #: Clearance from the outermost slot to the box's inner wall, per axis.  The x
    #: one is 4 mm in both shipped configs.  The y one is the jaws' own clearance
    #: -- they descend into the box beside the batch -- and the shipped relay box
    #: uses the tightest value that leaves them room, 9.167 mm against a 6.35 mm
    #: pad; the single-box scene widens it to 13.167 mm because it has the room.
    target_x_margin_m: float = 0.004
    target_y_margin_m: float = 0.009166666666666667
    #: Clearance from the source grid to the bin's inner wall, per axis.  Chosen,
    #: and larger than the derived minimum: the shipped bin has 12 mm in x and
    #: 8 mm in y.
    source_bin_margin_m: tuple[float, float] = (0.012, 0.008)
    #: Thickness of one box wall, added to every half size to make an outer one.
    bin_wall_thickness_m: float = 0.006
    #: Thickness of one jaw pad along the closing axis, and the gripper's full
    #: stroke.  Both are properties of the 2F-85, and together they bound how many
    #: Cookies one grasp can span: the pads have to fit in the boundary gaps, so
    #: ``per_grasp * row_pitch - pad_thickness <= stroke``.
    pad_thickness_m: float = 0.00635
    jaw_travel_m: float = 0.085

    @property
    def thickness_m(self) -> float:
        return 2.0 * self.half_size_m[1]

    @property
    def width_m(self) -> float:
        return 2.0 * self.half_size_m[0]

    @property
    def source_column_pitch_m(self) -> float:
        return self.width_m + self.source_column_clearance_m

    @property
    def source_row_pitch_m(self) -> float:
        return self.thickness_m + self.source_row_clearance_m

    @property
    def target_slot_pitch_x_m(self) -> float:
        """Column spacing inside a target box.

        A whole number of the Cookie's width plus the column gap, so the two
        columns of slots are one Cookie wide with a gap between them.
        """

        return self.width_m + self.target_column_gap_m

    @property
    def target_slot_pitch_y_m(self) -> float:
        """Row spacing inside a target box: the Cookie's *nominal* thickness.

        No clearance, unlike the source layout.  The slots are where a batch comes
        to rest after the jaws have compressed it, and the shipped lattice has the
        five slabs touching at 0.0063333.
        """

        return self.thickness_m

    @property
    def max_per_grasp(self) -> int:
        """How many Cookies one grasp can span, from the jaw travel.

        The jaws enter the boundary gaps, so the opening they need is the batch's
        outer span less one pad: ``per_grasp * pitch - pad_thickness``, against a
        stroke of 0.085.  Measured against the shipped pitch this is 10, and the
        shipped ``per_grasp`` of 5 is well inside it.
        """

        return math.floor((self.jaw_travel_m + self.pad_thickness_m) / self.source_row_pitch_m)


@dataclass(frozen=True)
class MeasuredEnvelopes:
    """What the arm can actually reach, as measured, in the frames the checks use.

    Every number here came from a sweep, and each one is what bounds a different
    parameter.  They are tabulated rather than fitted on purpose: the sweeps were
    taken on a grid, and interpolating between grid points would claim a precision
    the measurement does not have.  A lookup therefore rounds *outwards*, and a
    station or column outside the swept range is refused rather than extrapolated.
    """

    #: How far the push and carry pads can reach in +y, by the station's x.  This
    #: is the envelope that bounds a *lane*'s ``box_capacity``: parking a filled box
    #: one lane pitch out puts the pads on its rear wall, at
    #: ``station_y + box_half_y + clearance``.  A single-box scene never pushes, so
    #: it is not bounded by this.
    #:
    #: Measured in 1 mm steps at z = 0.755 and 0.780 (identical grids).  It falls
    #: off a cliff past the shipped station: 99 mm at x = 0.075 but 80 at 0.090 and
    #: 68 at 0.110, because +y and +x compete for the same arm reach.  A lane
    #: therefore wants its station as far *in* as the fill's band allows -- 124 mm
    #: at x = 0.030 against 99 at 0.075 -- and a lane at the single-box scene's
    #: station (0.095) could hold only a six-Cookie box.
    pad_y_cap_by_station_x_m: tuple[tuple[float, float], ...] = (
        (0.030, 0.124),
        (0.040, 0.119),
        (0.050, 0.114),
        (0.060, 0.109),
        (0.075, 0.099),
        (0.090, 0.080),
        (0.100, 0.072),
        (0.110, 0.068),
        (0.120, 0.057),
    )
    #: The y at which the fill's own reachability pre-check starts accepting a
    #: station, by x.  Below it the placement pose's IK misses by more than its
    #: 2 mm / 0.035 rad rule.  The band is diagonal: a box further out in x has to
    #: be further out in y as well.
    #:
    #: Measured on a 2 mm grid, one env with the box moved by writing its free joint
    #: (about 65 ms a point, against a 5.4 s reset).  A coarse 30 mm grid was used
    #: first and overstated the floor by up to 20 mm -- which is only *conservative*,
    #: but it also made the table misleading to read and would have shrunk a derived
    #: range for no reason.
    #:
    #: It is a function of x alone, and in particular independent of how tall the
    #: box is: the placement pose is aimed at a *column's* centre, and a centred
    #: lattice has its column centres at the body's own y whatever the row count.
    #: So a tall box does not push this band outwards, which is what makes a
    #: capacity of 16 as feasible as a capacity of 10.
    station_min_y_by_x_m: tuple[tuple[float, float], ...] = (
        (0.030, 0.016),
        (0.050, 0.018),
        (0.060, 0.018),
        (0.075, 0.022),
        (0.085, 0.034),
        (0.090, 0.040),
        (0.100, 0.056),
        (0.110, 0.072),
    )
    #: How far out in +x a station may sit, as a function of how far out in +y it is.
    #: This is *the same boundary* as :attr:`station_min_y_by_x_m` read the other way
    #: round -- at x = 0.085 the floor is 0.034 and the boundary's +x limit at
    #: dy = +4 mm is +10 mm, which is the same point -- so it is not tabulated
    #: separately.  See :meth:`MeasuredEnvelopes.station_max_x`.
    #:
    #: It was tabulated separately, briefly, and the duplication was worse than
    #: redundant: a table in *offsets* from one station cannot be used at another, and
    #: the single-box scene's station at (0.095, 0.100) derived a +x range of -4 mm --
    #: a box forbidden from moving towards the arm at all -- from a window measured
    #: 60 mm away and 70 mm lower.
    #:
    #: How far in -x a station may sit, at the nominal y.  Two sweeps agree to within
    #: 10 mm and disagree about which is tighter, because the bound is a property of
    #: *where* the station is rather than a constant: measured here at the relay's
    #: station it is -150 mm (and -160 mm missed by 3.13 mm), while the single-box
    #: scene's own sweep reports -140 mm at its station.  The tighter of the two is
    #: used, so a derived range is safe at either.
    #:
    #: The shipped relay draws x up to +4 mm with y down to -6 mm, and the boundary
    #: there is about +2 mm: a *corner* of its range is outside the window, measured
    #: as "position error 2.27 mm, angle error 2.00 deg" against a 2.005 deg bound.
    #: That corner is a 2 mm by 2 mm triangle of a range 84 mm wide and 46 mm tall,
    #: so it costs about 1 episode in 2000 -- far below the resolution of the 3-seed
    #: measurement the shipped ranges were validated by, which is why it is recorded
    #: rather than "fixed" by narrowing a range whose yield has been measured.
    station_window_dx_min_m: float = -0.140
    #: How far out in +y a station may sit, at the nominal x.  The sweep's own edge:
    #: every step from +8 mm to +52 mm was accepted, so this is a floor on the real
    #: bound rather than the bound itself.  A lane does not use it -- its +y is set by
    #: the push it still has to perform -- but a single-box scene has no push and
    #: nothing else measured to bound it.
    station_window_dy_max_m: float = 0.052
    #: How far a station may yaw, at the nominal pose.  Also the sweep's edge: every
    #: step from 0 to +/-4 deg was accepted, so the real bound is wider.  What stops
    #: a yaw in practice is not the yaw alone but the yaw *combined* with an x and y
    #: offset, which is the same coupling as above.
    station_window_dyaw_rad: float = math.radians(4.0)
    #: A source column is usable if its x is at or below this.  The shipped layout
    #: puts its four columns at 0.0994 / 0.1498 / 0.2002 / 0.2506 and only the
    #: first two are: at 0.2002 a five-Cookie batch's approach pose misses by
    #: 7.24 mm against the 4 mm rule, and at 0.2506 by 19.12 mm.  Those two columns
    #: are decoration -- the expert's candidate rule skips them -- so the shipped
    #: 4 x 20 grid offers 40 graspable Cookies, not 80.
    source_usable_column_max_x_m: float = 0.150
    #: How far up in y a batch *centre* can be grasped, by column x.  Measured in
    #: 5 mm steps on the two usable columns: the near one to 0.480, the far one to
    #: 0.390.  This is what bounds the source grid's row count, and the shipped
    #: 20 rows sit exactly on the far column's limit.
    source_batch_centre_y_cap_by_x_m: tuple[tuple[float, float], ...] = (
        (0.0994, 0.480),
        (0.1498, 0.390),
    )
    #: How far the source bin can be offset in +y before something other than the
    #: arm stops it.  Measured: every batch position of both usable columns still
    #: solves at +340 mm, and the table ends at y = 0.750 against a bin 101 mm
    #: deep, so the bin's own depth is what sets 0.328.  Moving it out of a lane's
    #: way is therefore never what makes a layout fail.
    source_offset_cap_m: float = 0.328
    #: The right arm's reach in -y, checked past the sweep's edge.  A queue of six
    #: boxes fits, so the queue is never what bounds a layout either.
    right_arm_min_y_m: float = -0.520
    #: How far out in +x the fill can still place, from the shipped pose.  The
    #: outermost slot may sit 30 mm beyond the shipped 28 mm, which is what bounds
    #: a target box's column count.
    station_slot_x_reach_m: float = 0.030

    def _at_or_above(self, table: tuple[tuple[float, float], ...], x: float, what: str) -> float:
        """Look up a cap at the first tabulated x at or above ``x``.

        Rounding outwards, because a cap that falls with x would otherwise be
        optimistic between grid points.  Below the first entry there is nothing
        tighter to take, so the first entry's value is used -- which is
        conservative on both tables here, since the pad cap falls with x and the
        station band's floor rises with it.  Past the last entry the sweep has
        nothing to say and this refuses rather than extrapolating.
        """

        for entry_x, value in table:
            if x <= entry_x + 1e-12:
                return value
        raise ValueError(
            f"{what} is only measured up to x = {table[-1][0]:.3f}; "
            f"x = {x:.3f} is outside the sweep"
        )

    def pad_y_cap(self, station_x_m: float) -> float:
        return self._at_or_above(
            self.pad_y_cap_by_station_x_m, station_x_m, "the push pad envelope"
        )

    def station_min_y(self, station_x_m: float) -> float:
        return self._at_or_above(self.station_min_y_by_x_m, station_x_m, "the fill's station band")

    def source_batch_centre_y_cap(self, column_x_m: float) -> float:
        return self._at_or_above(
            self.source_batch_centre_y_cap_by_x_m,
            column_x_m,
            "the source grasp band",
        )

    def station_max_x(self, y_m: float, x_ceiling_m: float) -> float:
        """The largest x whose band floor is at or below ``y_m``.

        The band's boundary is one curve in (x, y), so this reads
        :attr:`station_min_y_by_x_m` the other way round rather than tabulating the
        same measurements twice.  ``x_ceiling_m`` caps the answer at something the
        sweep covered, because past the last entry there is nothing to say.
        """

        widest = self.station_min_y_by_x_m[0][0]
        for entry_x, floor in self.station_min_y_by_x_m:
            if floor > y_m + 1e-12:
                break
            widest = entry_x
        return min(widest, x_ceiling_m)


@dataclass(frozen=True)
class BoxRange:
    """One box's draw ranges, as offsets from its nominal pose."""

    x_m: AxisRange = field(default_factory=AxisRange)
    y_m: AxisRange = field(default_factory=AxisRange)
    yaw_rad: AxisRange = field(default_factory=AxisRange)

    @property
    def enabled(self) -> bool:
        return self.x_m.movable or self.y_m.movable or self.yaw_rad.movable


@dataclass(frozen=True)
class SceneRandomizationSpec:
    """How far each thing in the scene may move between episodes.

    Derived rather than hand-tuned, which is what the user asked for: *"how do we
    make sure that with n boxes the randomization does not let the boxes hit each
    other"*.  The answer has three parts and they are worth keeping apart, because
    only the first is a formula:

    1. **A pairwise clearance formula.**  Two neighbours ``gap`` apart, each drawn
       within its own range, must still leave ``min_box_clearance_m`` between their
       *outer* extents -- and a yawed box's outer extent is wider than its half size,
       so the formula uses the yawed one.  For symmetric ranges that is
       ``2r <= gap - yawed_extent - clearance``, and the station box's own range
       eats into the same budget, so ``r <= gap - yawed_extent - clearance -
       |station_y_low|``.  This is exact for a pair, and adjacent pairs are the
       binding ones.
    2. **The measured windows**, for what the formula cannot see: the fill's
       acceptance window bounds where a box may sit or yaw when it is *filled*, and
       the band floor bounds how far -y that is.
    3. **A gate**, because 1 and 2 are per-axis bounds measured with the other axes
       at nominal, and the windows are *coupled* -- the relay's shipped range has a
       corner outside its own window, which no per-axis bound predicts.  See
       :mod:`a3_dual_arm_sim.scene_spec_validate`.

    The distinction that makes the derivation tractable: **a box's y offset does not
    survive the carry, but its x and yaw do.**  The carry slides the queue's boxes to
    the station's y whatever y they started at, so the queue's y range is bounded only
    by the clearance formula and the arm's reach, while its x and yaw ranges are
    bounded by the fill's window -- because those are what arrive with it.
    """

    station: BoxRange = field(default_factory=BoxRange)
    queue: BoxRange = field(default_factory=BoxRange)
    source: BoxRange = field(default_factory=BoxRange)
    arm_home_rad: AxisRange = field(default_factory=AxisRange)
    min_box_clearance_m: float = 0.025
    max_clearance_attempts: int = 200

    def as_config(self) -> dict[str, Any]:
        """The top-level ``randomization`` section, in the config's own key names.

        The station's ranges keep the ``target_bin_*`` names and every box behind it
        shares the ``spare_bin_*`` ones, because those are roles rather than indices
        -- see ``A3CookieTransferEnv._apply_scene_randomization``.
        """

        def pair(axis: AxisRange) -> list[float]:
            return [axis.low, axis.high]

        return {
            "source_bin_x_m": pair(self.source.x_m),
            "source_bin_y_m": pair(self.source.y_m),
            "source_bin_yaw_rad": pair(self.source.yaw_rad),
            "target_bin_x_m": pair(self.station.x_m),
            "target_bin_y_m": pair(self.station.y_m),
            "target_bin_yaw_rad": pair(self.station.yaw_rad),
            "spare_bin_x_m": pair(self.queue.x_m),
            "spare_bin_y_m": pair(self.queue.y_m),
            "spare_bin_yaw_rad": pair(self.queue.yaw_rad),
            "arm_home_rad": pair(self.arm_home_rad),
            "min_box_clearance_m": self.min_box_clearance_m,
            "max_clearance_attempts": self.max_clearance_attempts,
        }


@dataclass(frozen=True)
class SceneSpec:
    """A whole cookie scene, stated as parameters.

    Build one, read its derived properties, and call :meth:`derive_config` to get
    the ``cookie_transfer`` block an environment can be built from.
    :meth:`validate` reports *every* parameter that is out of bounds rather than
    the first, so a caller sweeping for a workable layout sees all of them at once.
    """

    # ---- the key parameters
    #: How many boxes the run fills.  One is the single-box scene; two or more is a
    #: lane: the station box is filled, pushed one pitch further out, and the queue
    #: behind it advances.
    boxes: int = 1
    #: Cookies taken in one grasp.
    per_grasp: int = 5
    #: Cookies one box holds.
    box_capacity: int = 10
    #: Cookies the source bin is laid out with.  What matters is how many are
    #: *graspable*, which is measured rather than assumed; see
    #: :attr:`source_supply`.
    source_cookies: int = 80
    #: Columns of slots in a target box.  Two is the shipped value and the number
    #: the x reach allows; three fits with 2 mm to spare and makes a wider box.
    target_columns: int = 2
    #: Columns of Cookies in the source bin.  Four is the shipped value; only the
    #: two nearest the arm are reachable, and a derived layout should place its
    #: columns where they can be grasped rather than keep the decoration.
    source_columns: int = 4

    # ---- the station, and the lane behind it
    #: Where the working box sits.  Bounded below by the fill's own band and above
    #: by the push pads: every millimetre further out costs a millimetre of lane.
    station_x_m: float = 0.075
    station_y_m: float = 0.030
    #: Nominal centre of the source layout.
    source_center_m: tuple[float, float] = (0.175, 0.315)
    #: Smallest gap allowed between any two boxes' outer extents.  Sets the lane
    #: pitch when one is not given.
    min_box_clearance_m: float = 0.025
    #: Distance the station box is pushed out, and therefore the spacing between
    #: boxes in the lane.  ``None`` derives ``box_extent_y + min_box_clearance``.
    lane_pitch_m: float | None = None
    #: How far behind the station the queue's first box starts.  ``None`` derives
    #: :attr:`lane_pitch`, which is the spacing the line *compacts to* -- and which
    #: is therefore not usable as a layout spacing.  Three quantities are easy to
    #: confuse here and it is worth separating them:
    #:
    #: * :attr:`lane_pitch` is how far each fill *pushes* the line, and it is
    #:   ``box_extent_y + min_box_clearance``.  The source bin's clearance is a
    #:   function of this, not of the gap below: the boxes compact to it whatever
    #:   they started at.
    #: * this gap is the spacing the lane is *laid out* at.  It has to be strictly
    #:   more than the pitch, because the pitch is exactly the clearance between
    #:   *outer* extents: two boxes that far apart have no room to be drawn at all,
    #:   so the derived default is refused at load (see
    #:   ``CookieSceneConfig.worst_nominal_box_gap_m``).  Measured, a three-box lane
    #:   at the 87 mm pitch has 13 mm between neighbours against the 25 mm it draws
    #:   within.
    #: * the ranges then need room *on top of* this gap, which is why the shipped
    #:   relay's 160 mm is comfortable rather than arbitrary: the queue's own y range
    #:   can close 55 mm between two neighbours, and 160 - 74 = 86 mm of outer gap
    #:   absorbs that with 31 mm to spare.
    #:
    #: Deriving this from the ranges is Phase 5's job -- the ranges are not a spec
    #: field yet, and inventing the formula before the ranges exist would be
    #: guessing.  Until then a lane states its gap, and the two shipped two-box
    #: scenes state theirs as 160 mm because that is what they were built with.
    queue_gap_m: float | None = None
    #: How much push a lane's fill must still have left to perform, which is what
    #: bounds the station box's +y draw.  A box that starts most of the way to the
    #: parking position has no push left, and the push is the mechanism the whole
    #: relay is built on -- so this is a *workflow* bound rather than a geometric one,
    #: and it is a field because nothing about the arm measures it.  The shipped
    #: relay's +40 mm draw against a 90 mm push leaves 50 mm of push, which is where
    #: this default comes from.
    min_push_m: float = 0.050

    geometry: CookieGeometry = field(default_factory=CookieGeometry)
    envelopes: MeasuredEnvelopes = field(default_factory=MeasuredEnvelopes)

    # ------------------------------------------------------------------ derived
    @property
    def plan(self) -> BatchPlan:
        return BatchPlan.for_capacity(self.box_capacity, self.per_grasp, self.target_columns)

    @property
    def target_rows(self) -> int:
        return self.plan.rows

    @property
    def target_slots_local_m(self) -> tuple[tuple[float, float], ...]:
        """The slot lattice in the box's own frame, row-major.

        Row-major to match the shipped configs: for each row, ascending x.  A
        batch's ``slot_indices`` are indices into this list.
        """

        geometry = self.geometry
        offset = geometry.target_slot_pitch_x_m / 2.0
        pitch = geometry.target_slot_pitch_y_m
        half_rows = (self.target_rows - 1) / 2.0
        return tuple(
            (x, (row - half_rows) * pitch)
            for row in range(self.target_rows)
            for x in (-offset, offset)
        )

    @property
    def target_bin_half_size_m(self) -> tuple[float, float]:
        """Outer half size of a target box: the slots, the Cookie, a margin, a wall."""

        geometry = self.geometry
        slots = self.target_slots_local_m
        outer_x = max(abs(slot[0]) for slot in slots)
        outer_y = max(abs(slot[1]) for slot in slots)
        return (
            outer_x
            + geometry.half_size_m[0]
            + geometry.target_x_margin_m
            + geometry.bin_wall_thickness_m,
            outer_y
            + geometry.half_size_m[1]
            + geometry.target_y_margin_m
            + geometry.bin_wall_thickness_m,
        )

    @property
    def box_extent_y_m(self) -> float:
        return 2.0 * self.target_bin_half_size_m[1]

    @property
    def lane_pitch(self) -> float:
        """Spacing between boxes in the lane, and the station box's push distance."""

        if self.lane_pitch_m is not None:
            return self.lane_pitch_m
        return self.box_extent_y_m + self.min_box_clearance_m

    @property
    def queue_gap(self) -> float:
        return self.lane_pitch if self.queue_gap_m is None else self.queue_gap_m

    @property
    def box_positions_m(self) -> tuple[tuple[float, float], ...]:
        """Every box's nominal (x, y): the station first, then the queue.

        The queue runs towards -y, because the arm's +y reach is capped at about
        +0.100 and the lane has to grow somewhere.  ``box_positions_m[0]`` is the
        station, which is the box filled first.
        """

        return tuple(
            (self.station_x_m, self.station_y_m - self.queue_gap * index)
            for index in range(self.boxes)
        )

    @property
    def source_rows(self) -> int:
        return math.ceil(self.source_cookies / self.source_columns)

    @property
    def source_positions_m(self) -> tuple[tuple[float, float], ...]:
        """The source layout, row-major, centred on the nominal centre.

        The rows are generated in full -- ``source_columns * source_rows``
        positions, which may exceed ``source_cookies`` when the count does not
        divide evenly.  That is deliberate: the layout is a lattice, and a bin
        holding 78 Cookies still has 80 slots, two of them empty.
        """

        geometry = self.geometry
        column_pitch = geometry.source_column_pitch_m
        row_pitch = geometry.source_row_pitch_m
        center_x, center_y = self.source_center_m
        half_columns = (self.source_columns - 1) / 2.0
        half_rows = (self.source_rows - 1) / 2.0
        return tuple(
            (
                center_x + (column - half_columns) * column_pitch,
                center_y + (row - half_rows) * row_pitch,
            )
            for column in range(self.source_columns)
            for row in range(self.source_rows)
        )

    @property
    def source_bin_half_size_m(self) -> tuple[float, float]:
        """Outer half size of the source bin: the grid, the Cookie, a margin, a wall."""

        geometry = self.geometry
        positions = self.source_positions_m
        half_x = max(abs(x - self.source_center_m[0]) for x, _ in positions)
        half_y = max(abs(y - self.source_center_m[1]) for _, y in positions)
        margin_x, margin_y = geometry.source_bin_margin_m
        return (
            half_x + geometry.half_size_m[0] + margin_x + geometry.bin_wall_thickness_m,
            half_y + geometry.half_size_m[1] + margin_y + geometry.bin_wall_thickness_m,
        )

    @property
    def usable_source_columns(self) -> tuple[float, ...]:
        """The source columns the arm can grasp from, by their x."""

        return tuple(
            sorted(
                {
                    x
                    for x, _ in self.source_positions_m
                    if x <= self.envelopes.source_usable_column_max_x_m + 1e-12
                }
            )
        )

    def usable_source_rows(self, column_x_m: float) -> int:
        """How many rows of one column can be grasped, whole batches only.

        Walks the row count down until every batch of that column fits inside the
        measured band.  It is the *batch centre* that has to fit, not the rows
        themselves: a batch is grasped as a line, so the pose that has to be
        reachable is the one above its middle.
        """

        cap = self.envelopes.source_batch_centre_y_cap(column_x_m)
        geometry = self.geometry
        row_pitch = geometry.source_row_pitch_m
        _, center_y = self.source_center_m
        for rows in range(self.source_rows, 0, -1):
            half_rows = (rows - 1) / 2.0
            sizes = self._column_batch_sizes(rows)
            start = 0
            for size in sizes:
                centre = center_y + (start + (size - 1) / 2.0 - half_rows) * row_pitch
                if centre > cap + 1e-12:
                    break
                start += size
            else:
                return rows
        return 0

    def _column_batch_sizes(self, rows: int) -> tuple[int, ...]:
        """How one column of ``rows`` Cookies is split into grasps."""

        sizes = [self.per_grasp] * (rows // self.per_grasp)
        if rows % self.per_grasp:
            sizes.append(rows % self.per_grasp)
        return tuple(sizes)

    @property
    def source_supply(self) -> int:
        """Graspable Cookies: the usable columns times their usable rows.

        This is the number that bounds :attr:`boxes`, and it is measured rather
        than counted.  The shipped layout states 80 Cookies and supplies 40: two
        of its four columns are out of reach, and the shipped 20 rows are exactly
        what the far usable column's band allows.
        """

        columns = self.usable_source_columns
        if not columns:
            return 0
        return min(self.usable_source_rows(x) for x in columns) * len(columns)

    @property
    def max_box_capacity(self) -> int | None:
        """Largest capacity whose box still leaves a *lane*'s push pads room.

        A filled box is pushed one lane pitch further out and the pads follow it to
        its rear wall, so the taller the box the further the pads have to reach.
        ``None`` for a single box, which never pushes and so is not bounded by this
        at all.

        The bound moves a lot with the station: 16 at the shipped (0.075, 0.030),
        24 at (0.050, 0.030), 6 at the single-box scene's x of 0.095.  That is why
        the station's x is not free either.
        """

        if self.boxes < 2:
            return None
        geometry = self.geometry
        try:
            cap = self.envelopes.pad_y_cap(self.station_x_m)
        except ValueError:
            return None
        pitch = geometry.target_slot_pitch_y_m
        for rows in range(1, 200):
            half_y = (
                (rows - 1) / 2.0 * pitch
                + geometry.half_size_m[1]
                + geometry.target_y_margin_m
                + geometry.bin_wall_thickness_m
            )
            pad_y = self.station_y_m + half_y + self.min_box_clearance_m
            if pad_y > cap + 1e-12:
                return (rows - 1) * self.target_columns
        return None

    @property
    def max_boxes(self) -> int:
        """How many boxes the source bin can fill, from its graspable supply."""

        return self.source_supply // self.box_capacity

    @property
    def source_offset_m(self) -> float:
        """How far the source bin has to move in +y to clear the parked lane.

        The lane grows towards +y and the source bin is in the way, so the bin
        moves -- which is free, since the pick poses solve at every offset out to
        the measured cap.  Zero for the shipped two-box layout, 46 mm for three,
        133 for four.
        """

        parked_front = (
            self.station_y_m + self.lane_pitch * (self.boxes - 1) + self.box_extent_y_m / 2.0
        )
        source_front = self.source_center_m[1] - self.source_bin_half_size_m[1]
        return max(0.0, parked_front + self.min_box_clearance_m - source_front)

    # ------------------------------------------------------------- randomization
    @property
    def randomization(self) -> SceneRandomizationSpec:
        """The draw ranges this spec derives, as the formula's answer.

        Not a promise: these are per-axis bounds taken from measurements made with
        the other axes at nominal, and the acceptance windows are *coupled*.  What
        says whether a range is usable is the gate in
        :mod:`a3_dual_arm_sim.scene_spec_validate`, which draws layouts and runs the
        scene's own checks on each.  This is the starting point it validates.

        What each range comes from, and why the station and the queue differ:

        * the **station** box is filled where it stands, so its y is bounded below by
          the band floor and above by the push it still has to perform, and its x by
          the window at that y.  Those two are coupled -- the floor rises with x --
          so the pair is solved by a short fixpoint, which converges because the
          shipped numbers make it a one-step correction.
        * the **queue** boxes are *carried* to the station, and the carry slides them
          to the station's y whatever y they started at.  So their y is bounded only
          by the clearance formula, while their x and yaw are bounded by the fill's
          window -- because x and yaw are what arrive with them.
        * the **source** bin is bounded by its clearance to the station box, which is
          the pair the lane does not move.
        """

        envelopes = self.envelopes
        yaw_bound = AxisRange(
            low=-envelopes.station_window_dyaw_rad, high=envelopes.station_window_dyaw_rad
        )

        # --- the station
        # +y: a lane is bounded by the push it must still perform; a single box has
        # no push, so it is bounded by the window's own edge.
        if self.boxes >= 2:
            station_y_high = self.lane_pitch - self.min_push_m
        else:
            station_y_high = envelopes.station_window_dy_max_m
        # -y: the band floor.  The floor rises with x and x is drawn independently,
        # so the floor is taken at the far end of the x range -- and the x range's
        # far end depends on the floor.  Two monotone steps settle it: the shipped
        # numbers converge on the first.
        x_high = 0.0
        for _ in range(8):
            y_low = envelopes.station_min_y(self.station_x_m + x_high) - self.station_y_m
            settled = (
                envelopes.station_max_x(
                    self.station_y_m + y_low,
                    envelopes.station_min_y_by_x_m[-1][0],
                )
                - self.station_x_m
            )
            if abs(settled - x_high) < 1e-12:
                break
            x_high = settled
        station = BoxRange(
            x_m=AxisRange(low=envelopes.station_window_dx_min_m, high=x_high),
            y_m=AxisRange(low=y_low, high=station_y_high),
            yaw_rad=yaw_bound,
        )

        # --- the queue.  A single-box scene has no queue, so its ranges are empty:
        # the keys still exist in a config, and the env ignores them when there is no
        # box to move.
        #
        # The yawed extent is the honest width for the clearance formula: a box turned
        # by the yaw bound is wider than its half size, and the boxes are tens of
        # millimetres apart, so ignoring that would pass a draw that overlaps.  The
        # *box's* half size, not the Cookie's -- the first version of this used
        # `geometry.half_size_m` and got a 4 mm box instead of a 63 mm one, which made
        # the y range six times too wide.
        #
        # The budget can come out negative, and that is the Phase 2 finding restated:
        # a lane laid out at the *derived* pitch has no room for the yaw the fill
        # tolerates, because the pitch is exactly `box_extent_y + clearance` and the
        # yaw widens the extent.  It is reported by `validate` with the number rather
        # than clamped silently.
        box_half_x, box_half_y = self.target_bin_half_size_m
        yaw_magnitude = max(abs(yaw_bound.low), abs(yaw_bound.high))
        yawed_half_y = math.cos(yaw_magnitude) * box_half_y + math.sin(yaw_magnitude) * box_half_x
        budget = self.queue_gap - 2.0 * yawed_half_y - self.min_box_clearance_m
        if self.boxes >= 2:
            # The station's own range eats into the same budget, and a symmetric queue
            # range has to clear the box below it as well as the station above it.
            queue_y_half = max(0.0, min(budget - abs(y_low), budget / 2.0))
            queue = BoxRange(
                x_m=AxisRange(low=envelopes.station_window_dx_min_m, high=x_high),
                y_m=AxisRange(low=-queue_y_half, high=queue_y_half),
                yaw_rad=yaw_bound,
            )
        else:
            queue = BoxRange()

        # --- the source bin, against the station box: the pair the lane never moves.
        # One box's half size each, not two: the gap is between centres, and the
        # clearance is what is left after both outer extents are taken off it.
        source_gap = self.source_center_m[1] - self.station_y_m
        source_budget = (
            source_gap
            - self.source_bin_half_size_m[1]
            - self.target_bin_half_size_m[1]
            - self.min_box_clearance_m
        )
        # The station box can move up into the source, so what is left for the bin's
        # own downward draw is the budget less that.
        source_y_half = max(0.0, source_budget - station_y_high)
        source = BoxRange(
            y_m=AxisRange(low=-source_y_half, high=source_y_half),
            yaw_rad=AxisRange(low=-0.05, high=0.05),
        )

        return SceneRandomizationSpec(
            station=station,
            queue=queue,
            source=source,
            #: Kept at the measured value: the arm's start jitter is not bounded by
            #: any of the geometry above, only by the arm still being able to reach
            #: its first pose from wherever it starts.
            arm_home_rad=AxisRange(low=-0.04, high=0.04),
            min_box_clearance_m=self.min_box_clearance_m,
        )

    def derive_randomization(self) -> dict[str, Any]:
        """The ``randomization`` section for :meth:`randomization`."""

        return self.randomization.as_config()

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        """Refuse a spec that is outside the measured bounds, naming the number.

        Collects every problem rather than raising on the first: a caller choosing
        parameters is usually wrong in more than one way at once, and finding that
        out one round trip at a time is what makes a search slow.
        """

        geometry = self.geometry
        envelopes = self.envelopes
        problems: list[str] = []

        if self.boxes < 1:
            problems.append(f"boxes = {self.boxes}, needs at least 1")
        if self.target_columns < 1:
            problems.append(f"target_columns = {self.target_columns}, needs at least 1")
        if self.source_columns < 1:
            problems.append(f"source_columns = {self.source_columns}, needs at least 1")
        if self.source_cookies < 1:
            problems.append(f"source_cookies = {self.source_cookies}, needs at least 1")

        # The grasp size is bounded by the column it is placed into and by the jaw.
        if self.per_grasp < 1:
            problems.append(f"per_grasp = {self.per_grasp}, needs at least 1")
        elif self.per_grasp > self.target_rows:
            problems.append(
                f"per_grasp = {self.per_grasp} exceeds the {self.target_rows} rows a "
                f"target column holds (capacity {self.box_capacity} over "
                f"{self.target_columns} columns), so a batch could not be placed in "
                f"one column"
            )
        if self.per_grasp > geometry.max_per_grasp:
            problems.append(
                f"per_grasp = {self.per_grasp} exceeds the {geometry.max_per_grasp} "
                f"Cookies the jaws can span: the pads need "
                f"{self.per_grasp * geometry.source_row_pitch_m - geometry.pad_thickness_m:.4f} m "
                f"of opening against a {geometry.jaw_travel_m:.3f} m stroke"
            )

        # The box's shape is bounded by where the fill can still place.
        max_slot_x = max(abs(x) for x, _ in self.target_slots_local_m)
        allowed_slot_x = geometry.target_slot_pitch_x_m / 2.0 + envelopes.station_slot_x_reach_m
        if max_slot_x > allowed_slot_x + 1e-12:
            problems.append(
                f"the outermost target slot is {max_slot_x * 1000:.1f} mm out in x but "
                f"the fill reaches {allowed_slot_x * 1000:.1f} mm at station x = "
                f"{self.station_x_m:.3f}; use fewer target_columns"
            )

        # The station's own band, and the pads' envelope beyond it.
        try:
            min_y = envelopes.station_min_y(self.station_x_m)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if self.station_y_m < min_y - 1e-12:
                problems.append(
                    f"station y = {self.station_y_m * 1000:.1f} mm is inside the fill's "
                    f"reach at x = {self.station_x_m:.3f}, where it starts at "
                    f"{min_y * 1000:.1f} mm"
                )
        # Only a lane pushes a box, so only a lane is bounded by the pads.
        if self.boxes >= 2:
            try:
                cap = envelopes.pad_y_cap(self.station_x_m)
            except ValueError as exc:
                problems.append(str(exc))
            else:
                pad_y = self.station_y_m + self.box_extent_y_m / 2.0 + self.min_box_clearance_m
                if pad_y > cap + 1e-12:
                    problems.append(
                        f"a box of capacity {self.box_capacity} puts the push pads at "
                        f"{pad_y * 1000:.1f} mm, past the {cap * 1000:.1f} mm they reach "
                        f"at station x = {self.station_x_m:.3f}; the largest capacity "
                        f"here is {self.max_box_capacity}"
                    )

        # The lane, in both directions.
        if self.lane_pitch_m is not None and self.lane_pitch_m < self.box_extent_y_m + 1e-12:
            problems.append(
                f"lane_pitch = {self.lane_pitch_m * 1000:.1f} mm is less than the "
                f"{self.box_extent_y_m * 1000:.1f} mm box it has to fit"
            )
        if self.boxes > 1:
            last_y = self.box_positions_m[-1][1] - self.box_extent_y_m / 2.0
            if last_y < envelopes.right_arm_min_y_m:
                problems.append(
                    f"{self.boxes} boxes at a {self.queue_gap * 1000:.1f} mm queue gap "
                    f"put the last one's edge at {last_y * 1000:.1f} mm, past the "
                    f"{envelopes.right_arm_min_y_m * 1000:.1f} mm the arm reaches"
                )

        # The source bin: enough graspable Cookies, and room to move out of the way.
        supply = self.source_supply
        needed = self.boxes * self.box_capacity
        if supply < needed:
            problems.append(
                f"{self.boxes} boxes of {self.box_capacity} need {needed} graspable "
                f"Cookies but the source layout supplies {supply} "
                f"({len(self.usable_source_columns)} usable columns of "
                f"{self.source_columns}); the largest box count here is "
                f"{self.max_boxes}"
            )
        if self.source_offset_m > envelopes.source_offset_cap_m + 1e-12:
            problems.append(
                f"clearing the lane needs the source bin {self.source_offset_m * 1000:.1f} mm "
                f"out in +y, past the {envelopes.source_offset_cap_m * 1000:.1f} mm it "
                f"can move"
            )

        # A lane's queue has to have room for the yaw its boxes arrive with.  The
        # budget is the queue gap less both boxes' *yawed* outer extents and the
        # clearance, and it can come out negative -- which is the Phase 2 finding
        # restated: a lane laid out at the derived pitch has no room at all, because
        # the pitch is exactly `box_extent_y + clearance` and the yaw widens the
        # extent.  Reported rather than clamped, because a zero-width range is a
        # scene that cannot vary and would look like a working configuration.
        if self.boxes >= 2:
            yaw_bound = envelopes.station_window_dyaw_rad
            box_half_x, box_half_y = self.target_bin_half_size_m
            yawed_half_y = (
                math.cos(yaw_bound) * box_half_y + math.sin(yaw_bound) * box_half_x
            )
            budget = self.queue_gap - 2.0 * yawed_half_y - self.min_box_clearance_m
            if budget <= 0:
                problems.append(
                    f"a queue gap of {self.queue_gap * 1000:.1f} mm leaves nothing for "
                    f"the queue to move: two boxes turned by the {math.degrees(yaw_bound):.1f} "
                    f"deg the fill tolerates are {2.0 * yawed_half_y * 1000:.1f} mm wide, "
                    f"against the {self.min_box_clearance_m * 1000:.1f} mm clearance, so "
                    f"the gap has to exceed "
                    f"{(2.0 * yawed_half_y + self.min_box_clearance_m) * 1000:.1f} mm"
                )

        if problems:
            listed = "\n".join(f"  - {problem}" for problem in problems)
            raise ValueError(f"this scene spec is outside the measured bounds:\n{listed}")

    # ------------------------------------------------------------ what a scene needs
    def derive_config(self) -> dict[str, Any]:
        """The ``cookie_transfer`` keys this spec determines.

        Only the geometry.  Everything else in a shipped config -- the gains, the
        contact parameters, the wall heights -- is tuning that the spec records
        rather than derives, and it comes from ``CookieSceneConfig``'s defaults.

        The boxes are emitted in the schema's own split -- the first two by name,
        the rest as a queue -- so a lane of three states one key here and a lane of
        two states none, and both describe themselves the same way to the model.

        ``source_bin_center_m`` is emitted with the lane's offset already applied,
        because that is what the key *means*: it is the source bin's nominal
        position, and for a lane long enough to reach it that position has moved.
        The shipped two-box scenes derive an offset of zero, which is why their
        configs state the value they always had.
        """

        self.validate()
        source_center_x, source_center_y = self.source_center_m
        config: dict[str, Any] = {
            "source_bin_center_m": [source_center_x, source_center_y + self.source_offset_m],
            "source_bin_half_size_m": list(self.source_bin_half_size_m),
            "target_bin_half_size_m": list(self.target_bin_half_size_m),
            "target_bin_world_position_m": [
                self.station_x_m,
                self.station_y_m,
                TABLE_Z_M,
            ],
            "target_slots_local_m": [list(slot) for slot in self.target_slots_local_m],
            "cookie_source_positions_m": [list(position) for position in self.source_positions_m],
        }
        if self.boxes >= 2:
            config["spare_target_bin_world_position_m"] = [
                self.box_positions_m[1][0],
                self.box_positions_m[1][1],
                TABLE_Z_M,
            ]
        if self.boxes >= 3:
            config["queue_target_bin_world_positions_m"] = [
                [x, y, TABLE_Z_M] for x, y in self.box_positions_m[2:]
            ]
        return config

    def describe(self) -> str:
        """One line per derived quantity, for a log or a plan review."""

        capacity = self.max_box_capacity
        half_x, half_y = self.target_bin_half_size_m
        capacity_limit = capacity if capacity is not None else "unbounded (no push)"
        lines = [
            (
                f"boxes={self.boxes} per_grasp={self.per_grasp} "
                f"capacity={self.box_capacity} source={self.source_cookies}"
            ),
            (
                f"  target lattice   {self.target_columns} x {self.target_rows} "
                f"slots, box {half_x * 2000:.1f} x {half_y * 2000:.1f} mm"
            ),
            (
                f"  batch plan       {[group.size for group in self.plan.groups]} "
                f"({self.target_columns} columns)"
            ),
            (
                f"  source layout    {self.source_columns} x {self.source_rows} "
                f"= {self.source_columns * self.source_rows} slots, "
                f"{len(self.usable_source_columns)} columns usable"
            ),
            (
                f"  source supply    {self.source_supply} graspable, needs "
                f"{self.boxes * self.box_capacity}"
            ),
        ]
        if self.boxes >= 2:
            lines.append(
                f"  lane             pitch {self.lane_pitch * 1000:.1f} mm, "
                f"queue gap {self.queue_gap * 1000:.1f} mm, source offset "
                f"{self.source_offset_m * 1000:.1f} mm"
            )
        lines.append(
            f"  limits           per_grasp <= "
            f"{min(self.target_rows, self.geometry.max_per_grasp)}, "
            f"capacity <= {capacity_limit}, boxes <= {self.max_boxes}"
        )
        return "\n".join(lines)


def same_column_spec() -> SceneSpec:
    """The spec that reproduces ``configs/cookie_same_column.yaml``.

    Its station is further out than the relay's -- (0.095, 0.100) against
    (0.075, 0.030) -- and it has the room to keep the looser 35 mm box, so its
    y margin is the wider of the two shipped values.
    """

    return SceneSpec(
        boxes=1,
        station_x_m=0.095,
        station_y_m=0.100,
        min_box_clearance_m=0.020,
        geometry=replace(CookieGeometry(), target_y_margin_m=0.013166666666666667),
    )


def two_box_spec() -> SceneSpec:
    """The spec that reproduces ``configs/cookie_two_box_batch.yaml``.

    The queue gap is stated rather than derived: the lane pitch is the minimum that
    keeps the boxes clear (87 mm), and the shipped relay places its spare 160 mm
    back, which is a choice that gives the carry a longer slide.
    """

    return SceneSpec(
        boxes=2,
        station_x_m=0.075,
        station_y_m=0.030,
        queue_gap_m=0.160,
    )
