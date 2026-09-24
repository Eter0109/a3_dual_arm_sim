"""The scene derivation, anchored to the two configs it has to reproduce.

Every test here is either an anchor (the spec for a shipped scene must regenerate
that scene's ``cookie_transfer`` block) or a bound (a spec outside the measured
envelopes must be refused, with the number it missed by in the message).  Those are
the two ways a derivation can be wrong: it can disagree with the scene that is known
to work, or it can accept a scene that will not.

The bounds are not invented for the test -- they come from
``scripts/measure_scene_envelopes.py``'s sweeps, and each test names the measurement
it is pinning.  A bound that moves has to move because a sweep says so.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from a3_dual_arm_sim.batch_plan import BatchPlan
from a3_dual_arm_sim.scene_spec import (
    CookieGeometry,
    SceneSpec,
    same_column_spec,
    two_box_spec,
)

ROOT = Path(__file__).resolve().parents[1]

#: The keys the derivation is responsible for.  Anything else in a shipped config
#: is tuning -- gains, contact parameters, wall heights -- which the spec records
#: rather than derives, so that a new geometry key cannot quietly escape the anchor
#: by being added to a config and to nothing else.
DERIVED_KEYS = {
    "source_bin_center_m",
    "source_bin_half_size_m",
    "target_bin_half_size_m",
    "target_bin_world_position_m",
    "target_slots_local_m",
    "cookie_source_positions_m",
}
#: ... plus the ones a lane has and a single box does not: one named spare for the
#: second box, and a queue list for every box after it.
LANE_KEYS = ("spare_target_bin_world_position_m", "queue_target_bin_world_positions_m")

#: Tuning keys the shipped configs carry.  Listed rather than ignored so that
#: adding a key to a config is a deliberate act: this test fails until the list is
#: extended, which is the moment to ask whether the key is derived after all.
TUNING_KEYS = {
    "box_lift_m",
    "box_tilt_deg",
    "cookie_bottom_edge_bevel_m",
    "cookie_collision_mode",
    "cookie_edge_bevel_m",
    "cookie_friction",
    "cookie_solimp",
    "cookie_solref",
    "left_finger_pad_half_thickness_m",
    "left_gripper_kp",
    "right_gripper_kp",
    "source_bin_wall_height_m",
    "target_bin_wall_height_m",
}

#: The configs are written to ten significant decimals, so a derived value can only
#: be expected to match them to about 1e-10.
TOLERANCE = 1e-8


def _numbers(value: object) -> list[float]:
    """Flatten a config value to floats.

    The shipped configs write their slots as ``-.028``, which YAML 1.1 reads as a
    *string* -- a signed number needs a digit after the sign -- and
    ``config.load_config`` coerces them on the way in.  So a comparison has to do
    the same.
    """

    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_numbers(item))
        return out
    return [float(value)]  # type: ignore[arg-type]


def _lane_keys(spec: SceneSpec) -> set[str]:
    """The lane keys a spec of this many boxes has to emit.

    One named spare for the second box, and a queue list for every box after it --
    which is the schema's own split, so a two-box lane emits one and a three-box
    lane emits both.
    """

    keys = set()
    if spec.boxes >= 2:
        keys.add(LANE_KEYS[0])
    if spec.boxes >= 3:
        keys.add(LANE_KEYS[1])
    return keys


def _assert_reproduces(spec: SceneSpec, config_name: str) -> None:
    shipped = yaml.safe_load(
        (ROOT / "configs" / f"{config_name}.yaml").read_text(encoding="utf-8")
    )["cookie_transfer"]
    derived = spec.derive_config()
    expected = DERIVED_KEYS | _lane_keys(spec)
    assert set(derived) == expected, (
        "the derivation's key set changed; if a key is genuinely derived, add it "
        "to DERIVED_KEYS, and if it is not, take it out of derive_config"
    )
    assert set(shipped) == expected | TUNING_KEYS, (
        "the shipped config's keys changed; a new one is either derived (extend "
        "the spec and DERIVED_KEYS) or tuning (extend TUNING_KEYS)"
    )
    for key, want in derived.items():
        got = _numbers(shipped[key])
        flat = _numbers(want)
        assert len(flat) == len(got), f"{key}: {len(flat)} values against {len(got)}"
        worst = max(abs(a - b) for a, b in zip(flat, got, strict=True))
        assert worst <= TOLERANCE, (
            f"{key} differs from {config_name}.yaml by {worst:.3e}: "
            f"derived {[round(v, 7) for v in flat[:6]]} against "
            f"{[round(v, 7) for v in got[:6]]}"
        )


# --------------------------------------------------------------------- the anchors


def test_same_column_spec_reproduces_its_config():
    """The single-box anchor: one box, five at a time, ten per box, eighty source.

    This is the whole point of the derivation -- if a formula cannot reproduce a
    scene that is known to work, the formula is wrong and no amount of testing a
    third layout would have said so.
    """

    spec = same_column_spec()
    spec.validate()
    assert (spec.boxes, spec.per_grasp, spec.box_capacity, spec.source_cookies) == (
        1,
        5,
        10,
        80,
    )
    _assert_reproduces(spec, "cookie_same_column")


def test_two_box_spec_reproduces_its_config():
    """The lane anchor: the same box and grasp, one box queued behind the station."""

    spec = two_box_spec()
    spec.validate()
    _assert_reproduces(spec, "cookie_two_box_batch")


def test_the_two_anchors_differ_only_in_the_parameters_they_state():
    """The two scenes share a Cookie, a lattice and a source layout.

    Worth pinning because it is what makes a *third* scene cheap: the difference
    between the shipped scenes is a station pose, a box margin and a queue gap, not
    two unrelated derivations.
    """

    single = same_column_spec()
    lane = two_box_spec()
    assert single.target_slots_local_m == lane.target_slots_local_m
    assert single.source_positions_m == lane.source_positions_m
    assert single.target_bin_half_size_m[0] == lane.target_bin_half_size_m[0]
    # The relay's box is narrower, which is what shortens its lane pitch.
    assert lane.target_bin_half_size_m[1] < single.target_bin_half_size_m[1]
    # ... and the difference is entirely the y margin, not the lattice.
    assert single.geometry.target_y_margin_m > lane.geometry.target_y_margin_m


# ------------------------------------------------------------------- the envelopes


def test_the_source_supply_is_measured_not_counted():
    """Forty graspable Cookies out of eighty laid out, because of the reach map.

    The layout has four columns at 0.0994 / 0.1498 / 0.2002 / 0.2506 and only the
    first two are reachable: at 0.2002 a five-Cookie batch's approach pose misses
    by 7.24 mm against the 4 mm rule, and at 0.2506 by 19.12 mm.  Those two columns
    are decoration, and counting them is how a spec would advertise six boxes and
    refuse half of them at collection time.
    """

    spec = two_box_spec()
    assert spec.source_columns * spec.source_rows == spec.source_cookies == 80
    # The generated x is a float expression, so it differs from the config's
    # literal in the last bit.
    assert spec.usable_source_columns == pytest.approx((0.0994, 0.1498))
    assert 0.2002 not in spec.usable_source_columns
    # Both usable columns can be grasped to their last row -- and the shipped 20
    # rows sit exactly on the far column's measured cap of 0.390 m, which is why
    # 21 rows would not fit.
    assert spec.usable_source_rows(0.0994) == 20
    assert spec.usable_source_rows(0.1498) == 20
    assert spec.source_supply == 40
    assert spec.max_boxes == 4


def test_a_fifth_box_is_refused_for_supply_and_the_message_says_so():
    spec = replace(two_box_spec(), boxes=5)
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    message = str(excinfo.value)
    assert "need 50 graspable Cookies" in message
    assert "supplies 40" in message
    assert "largest box count here is 4" in message


def test_a_smaller_grasp_costs_usable_source_rows():
    """Grasping one at a time loses three rows of source, and that is not a bug.

    The reach map bounds a batch's *centre*, and a batch's centre is where its pose
    is aimed.  A one-Cookie batch's centre is the Cookie itself, which sits closer
    to the band's edge than a five-Cookie batch's centre does, so the last three
    rows of a twenty-row column fall outside it.  So ``per_grasp`` is not free: at
    the shipped source layout, dropping to one costs a whole box.
    """

    spec = replace(two_box_spec(), per_grasp=1)
    assert spec.usable_source_rows(0.1498) == 17
    assert spec.source_supply == 34
    assert spec.max_boxes == 3
    spec.validate()
    with pytest.raises(ValueError, match="largest box count here is 3"):
        replace(spec, boxes=4).validate()


def test_per_grasp_is_bounded_by_the_column_it_is_placed_into():
    """A batch goes into one column, so it cannot be longer than that column."""

    spec = replace(two_box_spec(), per_grasp=6)
    assert spec.target_rows == 5
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    message = str(excinfo.value)
    assert "per_grasp = 6 exceeds the 5 rows" in message
    assert "capacity 10 over 2 columns" in message


def test_per_grasp_is_bounded_by_the_jaw_travel():
    """The pads have to fit in the boundary gaps: ``K * pitch - pad <= stroke``.

    Isolated with a capacity large enough that the column bound does not fire
    first, so this pins the jaw bound and not the lattice one.  Measured against the
    shipped 2.5 mm row gap this is ten Cookies against an 85 mm stroke; eleven would
    need 90.8 mm.
    """

    spec = replace(two_box_spec(), box_capacity=24, per_grasp=11)
    assert spec.target_rows == 12
    assert spec.geometry.max_per_grasp == 10
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    assert "exceeds the 10 Cookies the jaws can span" in str(excinfo.value)
    assert "0.0908 m of opening against a 0.085 m stroke" in str(excinfo.value)


def test_box_capacity_is_bounded_by_the_push_envelope():
    """A lane's capacity is bounded by where the push pads can reach.

    Parking a filled box one lane pitch out puts the pads on its rear wall, so a
    taller box needs a longer reach.  At the shipped station that is 18 Cookies;
    the relay's 10 is comfortable.
    """

    spec = two_box_spec()
    assert spec.max_box_capacity == 18
    assert spec.box_capacity <= spec.max_box_capacity
    over = replace(spec, box_capacity=20)
    with pytest.raises(ValueError) as excinfo:
        over.validate()
    message = str(excinfo.value)
    assert "puts the push pads at" in message
    assert "largest capacity here is 18" in message


def test_a_single_box_is_not_bounded_by_the_push_envelope():
    """No push happens with one box, so the pads' reach is irrelevant to it.

    The single-box scene's station is at (0.095, 0.100), which no lane could use --
    the pads reach only 72 mm there against the 86 mm a ten-Cookie box needs -- and
    this is what keeps the two scenes' bounds from being applied to each other.
    """

    spec = same_column_spec()
    assert spec.max_box_capacity is None
    assert spec.envelopes.pad_y_cap(0.095) == 0.072
    assert spec.envelopes.pad_y_cap(0.095) < (
        spec.station_y_m + spec.box_extent_y_m / 2.0 + spec.min_box_clearance_m
    )
    spec.validate()


def test_the_lane_wants_its_station_as_far_in_as_the_band_allows():
    """+y and +x compete for the same reach, and it shows up as a capacity cliff.

    The pad envelope is flat while the station is in the fill band's flat part --
    34 Cookies at x = 0.030, 26 at 0.050, 18 at the shipped 0.075 -- and then falls
    off: a station at the single-box scene's x of 0.095 could hold nothing at all.
    """

    capacities = {
        x: replace(two_box_spec(), station_x_m=x).max_box_capacity
        for x in (0.030, 0.050, 0.075, 0.095)
    }
    assert capacities == {0.030: 34, 0.050: 26, 0.075: 18, 0.095: 0}
    # And the bound really is a capacity bound, not a station one: the shipped
    # station is a legal station for a ten-Cookie box, just not for a large one.
    two_box_spec().validate()
    with pytest.raises(ValueError, match="largest capacity here is 0"):
        replace(two_box_spec(), station_x_m=0.095, station_y_m=0.100).validate()


def test_a_station_below_the_measured_band_is_refused():
    """The fill's own pre-check is the authority, and the band's floor is measured."""

    spec = replace(two_box_spec(), station_y_m=0.020)
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    message = str(excinfo.value)
    assert "station y = 20.0 mm is inside the fill's reach" in message
    assert "it starts at 30.0 mm" in message


def test_an_unmeasured_station_is_refused_rather_than_extrapolated():
    """Past the sweep, the spec says so instead of guessing.

    The station band was swept to x = 0.110 and the pad envelope to 0.120, so a
    station further out than either is refused with the sweep's limit.  Guessing
    there would be the same class of mistake as copying the single-box scene's
    randomization ranges into the relay scene, which cost two of three seeds.
    """

    spec = replace(two_box_spec(), station_x_m=0.130, station_y_m=0.090)
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    message = str(excinfo.value)
    assert "the fill's station band is only measured up to x = 0.110" in message
    assert "the push pad envelope is only measured up to x = 0.120" in message


def test_the_measured_tables_round_outwards():
    """Between grid points the lookup takes the tighter entry, never the looser.

    The pad cap falls with x and the band's floor rises with it, so a station
    *between* two tabulated ones must be judged by the nearer-the-limit of the two
    -- otherwise a spec would pass on an interpolated value the sweep never
    produced.  Below the first entry there is nothing tighter to take, so the
    sweep's own value is used, which is conservative on both tables.
    """

    envelopes = two_box_spec().envelopes
    # Between 0.075 (99 mm) and 0.090 (80 mm): the tighter of the two.
    assert envelopes.pad_y_cap(0.080) == 0.080
    assert envelopes.pad_y_cap(0.075) == 0.099
    # Below the sweep's first entry, the first entry's value.
    assert envelopes.pad_y_cap(0.010) == 0.124
    assert envelopes.station_min_y(0.010) == 0.030
    assert envelopes.station_min_y(0.080) == 0.060


def test_every_problem_is_reported_at_once():
    """A caller choosing parameters is usually wrong in more than one way.

    Finding that out one round trip at a time is what makes a search slow, so the
    validation collects rather than raising on the first.
    """

    spec = replace(two_box_spec(), boxes=5, box_capacity=24, per_grasp=11, source_cookies=200)
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    message = str(excinfo.value)
    assert message.count("\n  - ") >= 3, message


# -------------------------------------------------------------------- the batch plan


def test_batch_plan_splits_per_column_not_per_box():
    """The plan's remainder is per column, because a batch cannot span two.

    Spreading it across the whole box gives ``[3, 3, 3, 1]`` for ten Cookies in two
    columns at three per grasp, and there is nowhere to place that second batch of
    three: column 0 has two rows left.  Per column it is ``[3, 2, 3, 2]``.
    """

    plan = BatchPlan.for_capacity(capacity=10, per_grasp=3, columns=2)
    assert [group.size for group in plan.groups] == [3, 2, 3, 2]
    assert plan.column_sizes(0) == (3, 2)
    assert plan.column_sizes(1) == (3, 2)
    # Every batch stays inside its own column.
    for group in plan.groups:
        assert {index % plan.columns for index in group.slot_indices} == {group.column}


def test_the_shipped_plan_is_one_batch_per_column():
    """Which is why the two shipped scenes are step-identical after the change."""

    plan = two_box_spec().plan
    assert plan.rows == 5
    assert [group.size for group in plan.groups] == [5, 5]
    assert plan.column_sizes(0) == (5,)
    assert plan.column_sizes(1) == (5,)
    assert plan.groups[0].slot_indices == (0, 2, 4, 6, 8)
    assert plan.groups[1].slot_indices == (1, 3, 5, 7, 9)


def test_batch_plan_partitions_the_capacity_exactly():
    """Every Cookie is placed once, whatever the capacity and grasp size are.

    Including capacities that do not divide by the column count, where the last
    column is the short one -- 15 over two columns is 8 and 7 rows, not 8 and 8.
    The slots are taken in *column-major* order, so a lattice with more cells than
    the capacity has its unused cells at the end of the last column, not spread
    through the box.
    """

    for capacity in range(1, 25):
        for per_grasp in range(1, 11):
            for columns in (1, 2, 3):
                plan = BatchPlan.for_capacity(capacity, per_grasp, columns)
                assert plan.placed == capacity
                slots = [index for group in plan.groups for index in group.slot_indices]
                # The first `capacity` cells of the lattice in column-major order.
                cells = [
                    row * columns + column for column in range(columns) for row in range(plan.rows)
                ][:capacity]
                assert sorted(slots) == sorted(cells), (capacity, per_grasp, columns)
                assert all(group.size <= per_grasp for group in plan.groups)
                assert all(group.size >= 1 for group in plan.groups)
    assert BatchPlan.for_capacity(15, 5, 2).column_sizes(0) == (5, 3)
    assert BatchPlan.for_capacity(15, 5, 2).column_sizes(1) == (5, 2)


def test_batch_plan_resumes_a_partial_box():
    """The running form of the same rule, which is what a partial box needs.

    ``min(per_grasp, what is left of the current column)`` -- and it assumes the
    Cookies already in the box occupy the leading groups in order, which is how a
    fill puts them there.
    """

    plan = BatchPlan.for_capacity(capacity=10, per_grasp=3, columns=2)
    assert [group.size for group in plan.groups] == [3, 2, 3, 2]
    # Each column is finished before the next is started, so the sizes repeat per
    # column rather than alternating across the box.
    assert [plan.next_size(filled) for filled in range(10)] == [
        3,
        2,
        1,
        2,
        1,
        3,
        2,
        1,
        2,
        1,
    ]
    # A box that arrives with seven already in takes two, then one.
    assert plan.next_size(7) == 1
    assert plan.next_size(8) == 2
    with pytest.raises(ValueError, match="filled must be in"):
        plan.next_size(10)


# ------------------------------------------------------------------------- the lane


def test_source_offset_grows_with_the_box_count():
    """The lane grows towards +y and the source bin has to get out of the way.

    Free, as it turns out: the pick poses solve at every offset out to 340 mm, so
    the bin is never what makes a layout fail.  The shipped two-box layout needs no
    offset at all, three boxes need 46 mm, four need 133 mm.
    """

    offsets = {
        boxes: replace(two_box_spec(), boxes=boxes).source_offset_m for boxes in (1, 2, 3, 4)
    }
    assert offsets[1] == offsets[2] == 0.0
    assert offsets[3] == pytest.approx(0.0461, abs=5e-4)
    assert offsets[4] == pytest.approx(0.1331, abs=5e-4)
    # All of them are inside the measured cap, so the bound never fires here.
    for offset in offsets.values():
        assert offset <= two_box_spec().envelopes.source_offset_cap_m


def test_the_queue_gap_is_a_choice_and_a_large_one_costs_boxes():
    """The lane pitch is the minimum that keeps boxes clear; the shipped gap is bigger.

    The relay places its spare 160 mm back against a derived pitch of 87 mm, which
    gives the carry a longer slide.  It also means five boxes at that gap reach past
    the arm, where five at the derived pitch would fit -- so the gap belongs in the
    spec rather than in a controller, and the refusal says which number was used.
    """

    derived = replace(two_box_spec(), queue_gap_m=None)
    assert derived.lane_pitch == pytest.approx(0.087)
    assert derived.queue_gap == derived.lane_pitch
    # At the derived pitch a fifth box still fits the arm's -y reach, so it is the
    # shipped gap -- not the lane -- that puts it out of reach.
    fifth = replace(derived, boxes=5)
    assert fifth.box_positions_m[-1][1] - fifth.box_extent_y_m / 2.0 >= (
        derived.envelopes.right_arm_min_y_m
    )
    with pytest.raises(ValueError) as excinfo:
        replace(two_box_spec(), boxes=5).validate()
    assert "at a 160.0 mm queue gap" in str(excinfo.value)
    assert "past the -520.0 mm the arm reaches" in str(excinfo.value)


def test_a_box_too_tall_for_its_lane_is_refused():
    """The other half of the lane check: how far a queue reaches in -y.

    Isolated from the supply bound by keeping the boxes at ten Cookies, so four of
    them are exactly what the source supplies, and by pushing the queue gap out to
    200 mm -- which is a legal-looking layout that simply runs off the end of the
    arm's reach.
    """

    spec = replace(two_box_spec(), boxes=4, queue_gap_m=0.200)
    assert spec.source_supply == spec.boxes * spec.box_capacity
    assert spec.max_box_capacity is not None and spec.box_capacity <= spec.max_box_capacity
    with pytest.raises(ValueError) as excinfo:
        spec.validate()
    message = str(excinfo.value)
    assert "4 boxes at a 200.0 mm queue gap" in message
    assert "past the -520.0 mm the arm reaches" in message


def test_the_derived_config_is_json_serialisable():
    """A spec has to be able to write itself out for a dataset summary."""

    import json

    payload = json.loads(json.dumps(two_box_spec().derive_config()))
    assert payload["target_bin_world_position_m"] == [0.075, 0.030, 0.753]
    assert len(payload["cookie_source_positions_m"]) == 80


def test_the_default_geometry_is_the_shipped_cookie():
    """The Cookie's own numbers, which every layout is built from."""

    geometry = CookieGeometry()
    assert geometry.half_size_m == (0.025, 0.0095 / 3, 0.0125)
    assert geometry.thickness_m == pytest.approx(0.006333333, abs=1e-9)
    assert geometry.source_row_pitch_m == pytest.approx(0.008833333, abs=1e-9)
    assert geometry.target_slot_pitch_y_m == pytest.approx(0.006333333, abs=1e-9)
    assert geometry.max_per_grasp == 10


def test_the_derivation_imports_without_mujoco():
    """A spec has to be readable where the simulator will not load.

    Same reason `tasks.py` carries no simulator import: a GPU-only worker may not
    expose the CPU instructions MuJoCo's wheels need, and a caller that only wants
    to *describe* a scene -- a dataset summary, a sweep's plan, a training audit --
    must not be blocked by that.  Checked in a fresh interpreter, because the import
    graph is the thing under test.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "import a3_dual_arm_sim.batch_plan, a3_dual_arm_sim.scene_spec;"
                "from a3_dual_arm_sim.scene_spec import two_box_spec;"
                "two_box_spec().validate();"
                "print('mujoco' in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
