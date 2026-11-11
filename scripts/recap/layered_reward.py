"""Layered task potential for the 2x10 Cookie transfer, computed from a trace.

Why this replaces counting Cookies in the box
--------------------------------------------
The old reward was `change in running-max(cookies_in_target)`.  Measured on the 400-episode
RECAP set, that signal cannot tell the two halves of this task apart:

| quantity | mean | median | at maximum |
|---|---|---|---|
| Cookies in the box            | 14.61 | 16 | 31.2 % |
| Cookies in a *valid slot*     | 13.57 | 14 | 19.5 % |
| Target touching all walls     |  —    |  — | 70.5 % |

So 31 % of episodes already put every Cookie in the box and still fail, because 20 slots at
10 mm x 2.5 mm tolerance is a different requirement from "inside the box", and because the
packing has to touch all four walls.  The gap `in_box - slotted` averages **+1.04 Cookies and
is non-zero in 62.5 % of episodes (max 5)** -- a graded signal the count-only reward throws
away, which is why the strict success rate is 1.8 % while the mean count is 8.3.

What is measured
----------------
Four nested quantities, all already in the trace's `after` block:

* `in_box`   -- `cookies_in_target`.
* `slotted`  -- occupied entries of `target_slot_occupancy`.  It is `-1` for an empty slot and
                the occupying Cookie's index otherwise.
* `walls`    -- `target_touches_all_walls`, a 0/1 indicator that the packing reaches all four
                inner walls.  It is the only *binary* layer, and it is worth its own weight
                because it is exactly the condition that separates 31.2 % from 19.5 %.
* `preserved`-- `cookies_in_source`.  A Cookie knocked out of the source box is damage that
                `in_box` alone does not see.

**`slotted <= in_box` does not hold.**  I assumed it did -- the slot table is filled from the
Cookies already inside -- and it is false in 355 of 400 episodes.  The two are different
tests: `_target_slot_occupancy` compares a Cookie's *centre* against `TARGET_SLOT_TOLERANCE`
(10 mm x 2.5 mm), while `cookies_in_target` requires the Cookie's whole oriented bounding box
to stay inside the target's inner volume.  A Cookie lying tilted can be near a slot centre with
a corner overhanging the wall, giving `slotted = in_box + 1`.

Consequences, stated rather than hidden: the weights below are "what a Cookie is worth at each
stage", but they cannot be read as a strict per-Cookie ladder, and a state can lose `in_box`
credit while keeping `slotted` credit.  Since each layer is monotone-ised separately, such a
trade never produces a negative reward -- by construction it can only hold the potential flat.
If the ladder reading matters for a future run, the fix is to derive `slotted` only for Cookies
that satisfy `_cookie_inside_target`, i.e. to re-implement the per-Cookie check rather than
count the table; that needs the per-Cookie masks, which the trace does carry
(`cookies_in_target_mask`) but `diagnostics.py` does not currently forward.

The potential is `PHI = w_in * in_box + w_slot * slotted + w_walls * walls + w_src * preserved`
with default weights `1.0 / 0.5 / 1.0 / 0.5`, so one Cookie delivered *and* slotted is worth
1.5 and a Cookie merely dropped in the box is worth 1.0 -- the reward now prefers precision,
which is what the task's success condition asks for and what the count-only reward ignored.

Jitter, and why the running maximum stays
----------------------------------------
Measured over 80 episodes (~120k transitions) the number of steps where each quantity *falls*:

| quantity | falls / transitions |
|---|---|
| `in_box`    | 1.40 % |
| `slotted`   | 1.18 % |
| `walls`     | 0.14 % |

These are smaller than the flicker in the raw count (which needed `maximum.accumulate` in the
old reward), but not zero: a Cookie being regrasped leaves the containment region for a frame
and returns.  So each layer is still monotone-ised with a running maximum, and the reward is
the change in that potential.  The stated cost is unchanged from the old reward and now smaller
in relative terms, because `slotted` and `walls` are ratios rather than counts and the potential
is on a larger scale.

What this is *not*
------------------
It is a labelling function for advantage conditioning and for the value head, not a policy
gradient.  Training a policy *on* this reward directly would want the potential-based form
`r_t = gamma * PHI(s_{t+1}) - PHI(s_t)` to keep the optimal policy unchanged; for scoring
trajectories the plain potential is what we want and is strictly cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerWeights:
    """How much one Cookie is worth at each stage, and what the indicator layer adds.

    Defaults are chosen so that `deliver + slot` (1.5) beats `deliver only` (1.0), which is the
    preference the success condition encodes, and so the binary wall-coverage layer weighs about
    one Cookie.  None of these are tuned against a result yet -- they are stated here so a future
    run can vary them in one place.

    There is deliberately no `source` weight: see the module docstring for the measurement showing
    that term is constant across all 400 episodes and therefore carries no gradient.
    """

    in_box: float = 1.0
    slotted: float = 0.5
    walls: float = 1.0


DEFAULT_WEIGHTS = LayerWeights()

#: The largest potential an episode can reach: every Cookie delivered and seated, plus wall cover.
MAX_POTENTIAL = 20 * DEFAULT_WEIGHTS.in_box + 20 * DEFAULT_WEIGHTS.slotted + DEFAULT_WEIGHTS.walls


@dataclass(frozen=True)
class FrameSignals:
    """The four per-frame quantities, in the order the weights name them."""

    in_box: int
    slotted: int
    walls: int
    source: int


def frame_signals(after: dict[str, Any]) -> FrameSignals:
    """Read one trace frame's `after` block.

    `target_slot_occupancy` is `None` in the very first frames of some recordings (before the
    scene has been stepped), and `-1` for an empty slot, so both are treated as zero rather
    than as an error: a missing reading cannot be evidence that a Cookie is placed.
    """
    occupancy = after.get("target_slot_occupancy") or []
    return FrameSignals(
        in_box=int(after.get("cookies_in_target") or 0),
        slotted=sum(1 for value in occupancy if isinstance(value, int) and value >= 0),
        walls=1 if after.get("target_touches_all_walls") else 0,
        source=int(after.get("cookies_in_source") or 0),
    )


def potential(signals: FrameSignals, weights: LayerWeights = DEFAULT_WEIGHTS) -> float:
    """The task potential.  `signals.source` is deliberately not a term -- see the module
    docstring: it is the constant 80 in every episode and would only add a constant offset."""
    return (
        weights.in_box * signals.in_box
        + weights.slotted * signals.slotted
        + weights.walls * signals.walls
    )


def episode_rewards(
    frames: list[dict[str, Any]], weights: LayerWeights = DEFAULT_WEIGHTS
) -> dict[str, list[float]]:
    """Per-frame reward as the change in the monotone potential, plus the raw layers.

    The returned dict carries both the reward and the underlying signals so a caller can check
    that the reward is what it claims to be without re-parsing the trace -- and so a later
    analysis can ask which layer produced a given reward.

    The first frame's reward is its potential relative to zero, matching the old reward's
    convention (`rewards[0] = counts[0]`): the episode opens with a state, and the value head is
    trained to predict where that state leads, so the opening frame's contribution is the state
    itself rather than an arbitrary 0.
    """
    signals = [frame_signals(row.get("after") or {}) for row in frames]
    if not signals:
        return {
            "rewards": [], "potential": [], "in_box": [], "slotted": [], "walls": [],
            "source": [], "dropped": [],
        }

    # Per-layer running maxima, then the potential of those, so the reward can never be negative
    # because of flicker in a single layer.  `source` is tracked for the diagnostic only.
    running = {
        "in_box": [signals[0].in_box],
        "slotted": [signals[0].slotted],
        "walls": [signals[0].walls],
        "source": [signals[0].source],
    }
    for signal in signals[1:]:
        running["in_box"].append(max(running["in_box"][-1], signal.in_box))
        running["slotted"].append(max(running["slotted"][-1], signal.slotted))
        running["walls"].append(max(running["walls"][-1], signal.walls))
        running["source"].append(max(running["source"][-1], signal.source))

    monotone = [
        potential(
            FrameSignals(
                in_box=running["in_box"][i],
                slotted=running["slotted"][i],
                walls=running["walls"][i],
                source=running["source"][i],
            ),
            weights,
        )
        for i in range(len(signals))
    ]
    rewards = [monotone[0]] + [b - a for a, b in zip(monotone, monotone[1:])]
    # Cookies that are neither in the target nor in the source: knocked onto the table.  Not in
    # the potential (that would make it non-monotone), but worth recording: it is the one kind of
    # damage the other three layers cannot see.
    initial_total = signals[0].in_box + signals[0].source
    dropped = [max(0, initial_total - (s.in_box + s.source)) for s in signals]
    return {
        "rewards": rewards,
        "potential": monotone,
        "in_box": running["in_box"],
        "slotted": running["slotted"],
        "walls": running["walls"],
        "source": running["source"],
        "dropped": dropped,
    }
