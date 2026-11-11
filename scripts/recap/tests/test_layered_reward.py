"""Tests for the layered task potential.

The properties here are the ones the reward's correctness rests on: nesting, monotonicity, and
the specific preference the success condition encodes (a seated Cookie must beat a dropped one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layered_reward import (  # noqa: E402
    DEFAULT_WEIGHTS,
    FrameSignals,
    LayerWeights,
    episode_rewards,
    frame_signals,
    potential,
)


def after(cookies: int = 0, slots: list[int] | None = None, walls: bool = False, source: int = 80):
    """A trace `after` block, with `slots` as the occupancy list the environment writes."""
    if slots is None:
        slots = [-1] * 20
    return {
        "cookies_in_target": cookies,
        "cookies_in_source": source,
        "target_slot_occupancy": slots,
        "target_touches_all_walls": walls,
    }


def frame(**kwargs):
    return {"after": after(**kwargs)}


def test_frame_signals_counts_occupied_slots():
    occupancy = [-1, 3, -1, 7] + [-1] * 16
    signals = frame_signals(after(cookies=4, slots=occupancy))
    assert signals.in_box == 4
    assert signals.slotted == 2
    assert signals.walls == 0


def test_frame_signals_tolerates_missing_occupancy():
    """The first frames of a recording have no occupancy yet; that is not evidence of placement."""
    signals = frame_signals({"cookies_in_target": 2, "target_slot_occupancy": None})
    assert signals.in_box == 2
    assert signals.slotted == 0


def test_a_seated_cookie_beats_a_merely_dropped_one():
    """The whole point of the change: precision must be worth something.

    Under the old count-only reward these two frames scored identically.
    """
    dropped = FrameSignals(in_box=1, slotted=0, walls=0, source=80)
    seated = FrameSignals(in_box=1, slotted=1, walls=0, source=80)
    assert potential(seated) > potential(dropped)


def test_walls_is_worth_its_own_layer():
    """`walls` separates 31.2 % of episodes from 19.5 %, so it must move the potential."""
    without = FrameSignals(in_box=20, slotted=20, walls=0, source=60)
    with_walls = FrameSignals(in_box=20, slotted=20, walls=1, source=60)
    assert potential(with_walls) > potential(without)


def test_slotting_credit_survives_an_overhanging_cookie():
    """`slotted > in_box` is a real state, not a contradiction.

    Measured: 355 of 400 episodes contain frames where more Cookies are seated than are fully
    inside, because a seated test compares the centre against a 10 mm x 2.5 mm tolerance while
    containment requires the whole oriented box to fit.  The potential must stay finite and must
    not go backwards when that happens.
    """
    frames = [
        frame(cookies=3, slots=[0, 1, 2] + [-1] * 17),
        frame(cookies=2, slots=[0, 1, 2, 3] + [-1] * 16),  # tilted: loses in_box, keeps a slot
        frame(cookies=3, slots=[0, 1, 2, 3] + [-1] * 16),
    ]
    result = episode_rewards(frames)
    assert all(r >= 0 for r in result["rewards"]), result["rewards"]
    assert result["slotted"][1] > result["in_box"][1]
    # The potential is monotone by construction, so the intermediate state cannot subtract.
    assert result["potential"][1] >= result["potential"][0]


def test_nesting_is_not_assumed_by_the_weights():
    """Seating a Cookie must not cost more than delivering it, but the two are independent."""
    base = potential(FrameSignals(in_box=0, slotted=0, walls=0, source=0))
    one_in_box = potential(FrameSignals(in_box=1, slotted=0, walls=0, source=0))
    one_seated = potential(FrameSignals(in_box=1, slotted=1, walls=0, source=0))
    step_in = one_in_box - base
    step_slot = one_seated - one_in_box
    assert step_in == pytest.approx(DEFAULT_WEIGHTS.in_box)
    assert step_slot == pytest.approx(DEFAULT_WEIGHTS.slotted)
    assert step_slot < step_in, "seating a Cookie must not cost more than delivering it"


def test_rewards_are_never_negative_despite_flicker():
    """A Cookie regrasped out of the box must not be punished.

    Measured: the count falls on 1.40 % of transitions and occupancy on 1.18 %, so this is a
    real case rather than a hypothetical.
    """
    frames = [
        frame(cookies=0),
        frame(cookies=1),
        frame(cookies=0),          # knocked back out for a frame
        frame(cookies=2),
        frame(cookies=1),
        frame(cookies=3),
    ]
    rewards = episode_rewards(frames)["rewards"]
    assert all(r >= 0 for r in rewards), rewards
    # The potential reached is the peak, so the rewards must sum to it.
    assert sum(rewards) == pytest.approx(episode_rewards(frames)["potential"][-1])


def test_rewards_sum_to_the_final_potential():
    frames = [
        frame(cookies=0, source=80),
        frame(cookies=5, slots=[0, 1, 2, 3, 4] + [-1] * 15, source=75),
        frame(cookies=10, slots=list(range(10)) + [-1] * 10, source=70),
        frame(cookies=20, slots=list(range(20)), walls=True, source=60),
    ]
    result = episode_rewards(frames)
    assert sum(result["rewards"]) == pytest.approx(result["potential"][-1])
    expected = potential(FrameSignals(in_box=20, slotted=20, walls=1, source=80))
    assert result["potential"][-1] == pytest.approx(expected)


def test_source_term_is_not_in_the_potential():
    """The source box starts full in every episode and only empties, so it carries no gradient.

    Measured: `cookies_in_source` is 80 at frame 0 of all 400 episodes, so after the running
    maximum it is the constant 80 everywhere and would add a constant offset that cancels out of
    every advantage.  It is therefore excluded from the potential and reported as `dropped`
    instead -- keeping it as a term would look like it did something.
    """
    intact = FrameSignals(in_box=10, slotted=10, walls=0, source=70)
    spilled = FrameSignals(in_box=10, slotted=10, walls=0, source=60)
    assert potential(intact) == pytest.approx(potential(spilled))


def test_dropped_cookies_are_reported_even_though_they_are_not_penalised():
    """Damage must be visible somewhere, even though it cannot be a monotone reward term."""
    frames = [
        frame(cookies=0, source=80),
        frame(cookies=1, source=79),   # intended transfer: nothing lost
        frame(cookies=1, source=78),   # one Cookie left both boxes
    ]
    dropped = episode_rewards(frames)["dropped"]
    assert dropped == [0, 0, 1]


def test_empty_trace_is_handled():
    result = episode_rewards([])
    assert result["rewards"] == []
    assert result["potential"] == []


def test_weights_are_configurable_in_one_place():
    """A future run must be able to change the preference without editing logic."""
    seating_only = LayerWeights(in_box=0.0, slotted=1.0, walls=0.0)
    rewards = episode_rewards([frame(cookies=1), frame(cookies=1, slots=[0] + [-1] * 19)],
                              seating_only)["rewards"]
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(1.0), "only the slotting step should pay under this weight"


def test_walls_layer_is_binary_not_cumulative():
    """A Cookie that stays touching the walls must not pay again every frame."""
    frames = [frame(cookies=20, slots=list(range(20)), walls=True) for _ in range(5)]
    rewards = episode_rewards(frames)["rewards"]
    assert rewards[0] > 0
    assert all(r == pytest.approx(0.0) for r in rewards[1:]), rewards
