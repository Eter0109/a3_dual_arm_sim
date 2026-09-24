"""The derived randomization, and the gate that decides whether it is usable.

Phase 5 of `docs/configurable-scenes.md`.  Two claims are under test and they are
different claims:

* the *formula* -- a pairwise clearance bound and the measured windows -- which is
  checked by arithmetic against the numbers the measurements produced; and
* the *gate*, which exists because the formula is per-axis and the windows are
  coupled.  The relay's shipped ranges have a corner outside their own window, and so
  do the ranges this derivation produces, which is the whole argument for having a gate
  rather than trusting the arithmetic.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.scene_spec import same_column_spec, two_box_spec
from a3_dual_arm_sim.scene_spec_validate import (
    check_corners,
    check_layout,
    check_randomization,
    fit_randomization,
)

ROOT = Path(__file__).resolve().parents[1]
SINGLE_CONFIG = ROOT / "configs" / "cookie_same_column.yaml"
RELAY_CONFIG = ROOT / "configs" / "cookie_two_box_batch.yaml"
THREE_BOX_CONFIG = ROOT / "configs" / "generated" / "cookie_three_box.yaml"


# ------------------------------------------------------------------ the formula


def test_the_derived_station_range_reproduces_the_shipped_relay():
    """The derivation agrees with the range a person tuned by hand, and corrects it.

    The station's y comes out at ``[-8, +37] mm`` against the shipped
    ``[-6, +40] mm``: the -y is the band floor at the station's x, and the +y is
    ``lane_pitch - min_push``, which is the push the relay still has to perform.  The
    x's high end is the correction -- ``0`` against the shipped ``+4`` -- and it is
    exactly the value that keeps the corner inside the window.
    """

    derived = two_box_spec().randomization.station
    shipped = load_config(RELAY_CONFIG).randomization
    assert derived.y_m.low == pytest.approx(-0.008, abs=1e-6)
    assert derived.y_m.high == pytest.approx(0.037, abs=1e-6)
    assert shipped.target_bin_y_m.low == pytest.approx(-0.006, abs=1e-6)
    assert shipped.target_bin_y_m.high == pytest.approx(0.040, abs=1e-6)
    # The correction: the shipped +4 mm x is a corner outside the window.
    assert derived.x_m.high == pytest.approx(0.0, abs=1e-6)
    assert shipped.target_bin_x_m.high > derived.x_m.high


def test_the_station_y_high_is_the_push_a_lane_still_has_to_perform():
    """+y is a *workflow* bound, and it moves with the pitch rather than being fixed.

    A box drawn most of the way to the parking position has no push left, so the
    bound is ``lane_pitch - min_push_m``.  The shipped relay's +40 mm against its own
    90 mm push is where the 50 mm default comes from; this spec's derived pitch is
    87 mm, so its own bound is 37 mm.
    """

    spec = two_box_spec()
    assert spec.randomization.station.y_m.high == pytest.approx(spec.lane_pitch - spec.min_push_m)
    wider = replace(spec, min_push_m=0.020)
    assert wider.randomization.station.y_m.high == pytest.approx(spec.lane_pitch - 0.020)
    # A single box has no push at all, so it falls back to the window's own edge.
    assert same_column_spec().randomization.station.y_m.high == pytest.approx(
        same_column_spec().envelopes.station_window_dy_max_m
    )


def test_the_queue_y_is_the_pairwise_clearance_formula():
    """The bound the user asked for, as arithmetic rather than a table.

    Two neighbours drawn within their own ranges must still leave the clearance
    between their *yawed* outer extents -- and a box turned by the yaw the fill
    tolerates is wider than its half size, which is why ignoring the yaw would pass a
    draw that overlaps.
    """

    import math

    spec = two_box_spec()
    ranges = spec.randomization
    box_half_x, box_half_y = spec.target_bin_half_size_m
    yaw = ranges.station.yaw_rad.high
    yawed_half_y = math.cos(yaw) * box_half_y + math.sin(yaw) * box_half_x
    budget = spec.queue_gap - 2.0 * yawed_half_y - spec.min_box_clearance_m
    expected = min(budget - abs(ranges.station.y_m.low), budget / 2.0)
    assert ranges.queue.y_m.high == pytest.approx(expected)
    assert ranges.queue.y_m.low == pytest.approx(-expected)
    # The yaw really does matter: without it the bound would be 5 mm wider.
    without_yaw = spec.queue_gap - 2.0 * box_half_y - spec.min_box_clearance_m
    assert without_yaw > 2.0 * expected


def test_a_lane_at_the_derived_pitch_has_no_queue_range_at_all():
    """The Phase 2 finding, now visible in the derivation rather than in a redraw.

    A lane laid out at ``box_extent_y + clearance`` has no room for the yaw its boxes
    arrive with, so the budget comes out negative.  It is reported by `validate` with
    the number rather than clamped to a zero range, because a scene that cannot vary
    would otherwise look like a working configuration.
    """

    tight = replace(two_box_spec(), boxes=3, queue_gap_m=None)
    assert tight.randomization.queue.y_m.high == 0.0
    with pytest.raises(ValueError) as excinfo:
        tight.validate()
    message = str(excinfo.value)
    assert "leaves nothing for the queue to move" in message
    assert "the gap has to exceed" in message


def test_a_single_box_scene_derives_no_queue_ranges():
    """There is no queue to move, so the ranges are empty rather than invented."""

    ranges = same_column_spec().randomization
    assert not ranges.queue.enabled
    assert ranges.station.enabled


def test_the_source_bin_is_bounded_by_its_clearance_to_the_station():
    """The pair a lane never moves: the bin and the station box, against each other."""

    spec = two_box_spec()
    ranges = spec.randomization
    budget = (
        spec.source_center_m[1]
        - spec.station_y_m
        - spec.source_bin_half_size_m[1]
        - spec.target_bin_half_size_m[1]
        - spec.min_box_clearance_m
    )
    assert ranges.source.y_m.high == pytest.approx(budget - ranges.station.y_m.high)
    assert ranges.source.y_m.high > 0.05, "the shipped config draws +/-50 mm here"


def test_the_derived_randomization_is_a_config_section():
    """It has to be writable, and in the config's own key names."""

    section = two_box_spec().derive_randomization()
    assert json.loads(json.dumps(section)) == section
    assert section["target_bin_y_m"] == pytest.approx([-0.008, 0.037])
    assert section["min_box_clearance_m"] == 0.025
    # Roles, not indices: the station keeps `target_bin_*` and everything behind it
    # shares `spare_bin_*`.
    assert "spare_bin_y_m" in section


# --------------------------------------------------------------------- the gate


def test_the_gate_accepts_every_shipped_config():
    """A config that collects must pass its own gate, or the gate is measuring wrong."""

    for config, draws in ((SINGLE_CONFIG, 3), (RELAY_CONFIG, 3), (THREE_BOX_CONFIG, 3)):
        failures = check_randomization(config, draws=draws)
        assert failures == [], f"{config.name}: {failures[0]}"


def test_the_gate_finds_the_relay_corner_the_shipped_range_has():
    """The measurement that justifies the gate, pinned as a reproducible layout.

    The shipped relay draws x up to +4 mm with y down to -6 mm, and the fill's
    pre-check refuses that combination: "position error 2.27 mm, angle error 2.00 deg"
    against a 2.005 deg bound.  No per-axis bound predicts it -- the +x reach measured
    at the nominal y is wider than that, and so is the -y reach at the nominal x.
    """

    failures = check_layout(RELAY_CONFIG, offset_m=(0.004, -0.006))
    assert len(failures) == 1
    assert failures[0].check == "fill reach"
    # The numbers from `scripts/measure_scene_envelopes.py`'s 2 mm sweep: the fill's
    # angle bound is 0.035 rad and this corner misses by 1.86 deg = 0.0325 rad, so it
    # is inside the angle bound and refused on the *position* -- 2.11 mm against
    # 2.00.  Both are close, which is why the window is a band rather than a box.
    assert "position error 2.11 mm" in failures[0].detail
    assert "angle error 1.86 deg" in failures[0].detail
    # The other three corners of the shipped range are fine, so it really is a corner.
    for offset in ((-0.080, -0.006), (-0.080, 0.040), (0.004, 0.040)):
        assert check_layout(RELAY_CONFIG, offset_m=offset) == [], offset


def test_the_derived_station_x_keeps_the_corner_inside_the_window():
    """What the derivation is for: the same corner, moved to where it is legal."""

    derived = two_box_spec().randomization.station
    for offset in (
        (derived.x_m.low, derived.y_m.low),
        (derived.x_m.high, derived.y_m.low),
        (derived.x_m.high, derived.y_m.high),
    ):
        assert check_layout(RELAY_CONFIG, offset_m=offset) == [], offset


def test_the_gate_checks_corners_because_draws_do_not_reach_them():
    """The finding that shaped the gate, pinned as a measurement.

    The relay's derived ranges have four failing corners, and random draws do not
    reach them: with six draws the sweep reports nothing while the corners report
    four.  A shrink loop driven by draws alone therefore returned the failing range
    unchanged -- which is exactly what the first version of this did.
    """

    ranges = two_box_spec().randomization
    corner_failures = check_corners(RELAY_CONFIG, ranges)
    assert len(corner_failures) >= 1
    assert {failure.check for failure in corner_failures} == {"fill reach"}
    # Draws alone see nothing, which is the point.
    assert check_randomization(RELAY_CONFIG, draws=6) == []


def test_the_gate_names_the_box_and_carries_the_numbers():
    """A rejected range has to say which box and by how much."""

    ranges = replace(
        two_box_spec().randomization,
        station=replace(
            two_box_spec().randomization.station,
            y_m=load_config(RELAY_CONFIG).randomization.target_bin_y_m.__class__(0.0, 0.200),
        ),
    )
    failures = check_corners(RELAY_CONFIG, ranges)
    assert failures, "a station 200 mm out in +y has to be refused"
    assert failures[0].box == "target_bin"
    assert "unreachable" in failures[0].detail
    assert "layout" in dir(failures[0]) and failures[0].layout


def test_fitting_shrinks_until_the_corners_pass():
    """The loop between the formula and the gate, and what it costs in range.

    It converges to ``x[-79, 0] y[-5, +21] mm`` -- whose x low end lands within a
    millimetre of the range a person tuned by hand, which is a pleasant check that the
    loop is not merely shrinking towards nothing.
    """

    fitted, failures = fit_randomization(two_box_spec(), base_config=RELAY_CONFIG, draws=3)
    assert failures == []
    assert check_corners(RELAY_CONFIG, fitted) == []
    assert fitted.station.x_m.low == pytest.approx(-0.079, abs=0.01)
    assert fitted.station.x_m.high == 0.0
    # It did not shrink everything to nothing: both ranges still have real room, and
    # the fitted -x lands within a millimetre and a quarter of the range a person
    # tuned by hand (-80 mm) -- close enough to say the loop is not merely shrinking
    # towards nothing, and not equal, because the loop does not know that number.
    assert fitted.queue.y_m.high > 0.015
    assert fitted.station.y_m.high > 0.015
    shipped = load_config(RELAY_CONFIG).randomization.target_bin_x_m
    assert fitted.station.x_m.low == pytest.approx(shipped.low, abs=0.002)


def test_fitting_refuses_a_shrink_factor_that_is_not_a_shrink():
    with pytest.raises(ValueError, match="shrink must be in"):
        fit_randomization(two_box_spec(), base_config=RELAY_CONFIG, shrink=1.0)
