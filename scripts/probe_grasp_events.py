#!/usr/bin/env python3
"""Did the policy grab and fail to lift, or grab nothing at all?

The question this answers, stated exactly: at each **close event** in a recorded rollout,
were there Cookies between the pads, and if so did they come up with the hand?

Why it is not answerable from the existing logs.  `9.56` counted jaw open-close cycles and
found the policy stops cycling after the first minute, which is a statement about *attempts*.
It says nothing about what was in the jaws at each one, so "the policy never retries" and
"the policy retries but always on air" are indistinguishable in it -- and they are different
failures with different remedies.

The operational definition used here, and why:

* a **close event** is the gripper channel falling below `CLOSED` after having been above
  `OPEN`, with `SETTLE` steps of confirmation so a two-step dip does not count;
* a Cookie is **held** when, over the `LIFT_WINDOW` steps after the close, its horizontal
  displacement tracks the effector's -- same direction and at least `TRACK_FRACTION` of the
  magnitude.  Using co-motion rather than a jaw-geometry test is deliberate: the trace does
  not record the finger bodies, so "is it between the pads" can only be approximated, while
  "did it move with the hand" is exactly what a grasp means;
* a held Cookie has been **lifted** when its height rises more than `LIFT_RISE_M` above where
  it started.

Each event therefore lands in one of three classes, which is the distinction the question
asks for:

* `grab_lift`   -- closed on Cookies and carried them up;
* `grab_no_lift`-- closed on Cookies and they did **not** come up (the "half-held, will not
  lift" case);
* `empty`       -- closed on nothing (the "hovering with an empty gripper" case).

Usage:

    python scripts/probe_grasp_events.py --tasks 2x10 2x5
    python scripts/probe_grasp_events.py --tasks 2x10 --seeds 0 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Where the recorded rollouts live (one `seed_N` directory each, with `trace.jsonl`).
DIAG = Path("/root/autodl-tmp/diag/v9_run")

#: Gripper channel in the 16-wide applied action: left at 7, right at 15.  The left arm is
#: the one that transfers; the right one only holds its gripper shut.
LEFT_GRIPPER = 7

#: A close is this crossing.  The jaws read ~0.49 fully open and reach 0.0 shut; 0.35 is
#: where the pads are within a Cookie's thickness of each other.
OPEN_ABOVE = 0.45
CLOSED_BELOW = 0.30
#: Steps the channel has to stay shut for the crossing to count.
SETTLE = 3
#: Steps after the close over which co-motion is measured.
LIFT_WINDOW = 25
#: A Cookie counts as held when it moves at least this share of the hand's own motion.
TRACK_FRACTION = 0.35
#: ... and as lifted when it rises this far above its pre-close height.  Measured on the
#: recorded rollouts: a real grasp lifts the row ~12 mm, and standing Cookies sit at z =
#: 0.7685, so 8 mm separates "came up" from "stayed put" with room either side.
LIFT_RISE_M = 0.008
#: Cookies further than this from the effector are never candidates.  Calibrated from the
#: traces rather than guessed: a held row spans 5-18 mm from the effector, its immediate
#: neighbours sit at 26-35 mm, and the next column is 50 mm away, so 45 mm takes the row and
#: the two neighbours a pad could catch, and nothing else.
REACH_M = 0.025
#: Below this much hand motion a co-motion test measures noise, so such an event is judged on
#: height alone.  The policy's own idle wiggle is ~1.6 mm of travel, which is under it.
MIN_HAND_MOVE_M = 0.004
#: How close the effector has to come to a Cookie's own height for the close to count as an
#: attempt at *that* Cookie.  Calibrated from the traces and it is the most informative
#: number in this file: a real grasp closes with the pads 12 mm above the row's centres,
#: while the policy's failed closes happen with the effector **78-84 mm** above them.  So the
#: boundary is not a taste -- 30 mm sits in an empty band between the two behaviours.
DESCENT_M = 0.030


def load_trace(path: Path) -> dict[str, np.ndarray]:
    """Just the channels this probe reads, as arrays."""

    grip, eef, cookies = [], [], []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            applied = row.get("applied_action")
            if applied is None:
                grip.append(np.nan)
            else:
                grip.append(float(applied[LEFT_GRIPPER]))
            pose = np.asarray(row["observation_before"]["observation.eef_pose"], float)
            eef.append(pose[:3])
            cookies.append(np.asarray(row["after"]["cookie_positions"], float))
    return {
        "grip": np.asarray(grip),
        "eef": np.asarray(eef),
        "cookies": np.asarray(cookies),
    }


def close_events(grip: np.ndarray) -> list[int]:
    """Indices where the jaws shut and stayed shut."""

    events = []
    was_open = False
    for step, value in enumerate(grip):
        if np.isnan(value):
            continue
        if value > OPEN_ABOVE:
            was_open = True
            continue
        if not was_open or value > CLOSED_BELOW:
            continue
        tail = grip[step : step + SETTLE]
        if len(tail) == SETTLE and np.all(tail < CLOSED_BELOW):
            events.append(step)
            was_open = False
    return events


def classify(trace: dict[str, np.ndarray], step: int) -> dict:
    """Which of the three classes this close event belongs to, with the numbers behind it."""

    hand = trace["eef"]
    cookies = trace["cookies"]
    end = min(step + LIFT_WINDOW, len(hand) - 1)
    if end <= step:
        return {"kind": "short", "held": 0, "lifted": 0, "hand_move_mm": 0.0}

    hand_delta = hand[end] - hand[step]
    hand_move = float(np.linalg.norm(hand_delta[:2]))
    before = cookies[step]
    after = cookies[end]
    delta = after - before

    near = np.linalg.norm(before[:, :2] - hand[step][:2], axis=1) <= REACH_M
    # `in_range` is reported separately from `held`: an event where Cookies were within reach
    # and none came up is a *missed* grasp, while one with nothing in range is a grasp on
    # air.  The question this probe exists to answer is which of the two the policy does.
    in_range = int(np.count_nonzero(near))
    # **Did the hand come down at all?**  The height of the effector against the height of
    # the nearest Cookie, at the moment the jaws shut.  Without this the two failure modes
    # are indistinguishable: a close 80 mm above the row and a close around the row both
    # produce "jaws shut, nothing lifted".
    height_gap_mm = None
    if in_range:
        heights = before[near, 2]
        height_gap_mm = round(float(hand[step][2] - heights.max()) * 1000.0, 1)
    descended = height_gap_mm is not None and height_gap_mm <= DESCENT_M * 1000.0

    held = lifted = 0
    for index in np.flatnonzero(near):
        move = float(np.linalg.norm(delta[index, :2]))
        rose = bool(delta[index, 2] > LIFT_RISE_M)
        if hand_move < MIN_HAND_MOVE_M:
            # The hand barely moved: co-motion cannot be measured, so a Cookie counts as
            # held only if it rode up, which is the lifted case anyway.
            if rose:
                held += 1
                lifted += 1
            continue
        same_way = float(np.dot(delta[index, :2], hand_delta[:2])) > 0.0
        if rose or (same_way and move >= TRACK_FRACTION * hand_move):
            held += 1
            if rose:
                lifted += 1

    if held == 0:
        # "Empty" splits on whether the hand was even down at the row, and the two halves
        # have nothing in common: one is a grasp that never descended, the other is a
        # descent that closed on air.
        kind = "air_close" if not descended else "missed"
    elif not descended:
        # **A close 80 mm above the row cannot be holding anything**, whatever the
        # co-motion test said.  Two events in the recorded set did read `held=1` there, and
        # the cause is that a hand travelling at that height still nudges the pile it passes
        # over -- a Cookie pushed by its neighbours tracks the motion without ever being
        # between the pads.  Height is the discriminating signal, so it overrides.
        kind = "air_close"
        held = lifted = 0
    else:
        kind = "grab_lift" if lifted else "grab_no_lift"
    return {
        "kind": kind,
        "in_range": in_range,
        "height_gap_mm": height_gap_mm,
        "held": held,
        "lifted": lifted,
        "hand_move_mm": round(hand_move * 1000.0, 1),
    }


def probe(path: Path) -> dict:
    trace = load_trace(path)
    events = close_events(trace["grip"])
    results = [{"step": step, **classify(trace, step)} for step in events]
    counts = {"grab_lift": 0, "grab_no_lift": 0, "air_close": 0, "missed": 0,
              "short": 0}
    for item in results:
        counts[item["kind"]] += 1
    return {
        "path": str(path.parent.name),
        "steps": int(len(trace["grip"])),
        "events": results,
        "counts": counts,
    }


def summarise(reports: list[dict], label: str) -> None:
    keys = ("grab_lift", "grab_no_lift", "air_close", "missed")
    total = {key: 0 for key in (*keys, "short")}
    for report in reports:
        for key in total:
            total[key] += report["counts"].get(key, 0)
    # `short` events (a close at the very last step, with no window to measure) are excluded
    # from the denominator because nothing about them can be observed.
    events = sum(total[key] for key in keys)
    print(f"\n=== {label}: {len(reports)} rollouts, {events} close events ===")
    if not events:
        print("  no close events")
        return
    for key in keys:
        print(f"  {key:<13} {total[key]:>4}  ({100.0 * total[key] / events:>5.1f}%)")
    print(f"  {'per rollout':<13} {events / len(reports):>4.1f}")
    print()
    header = f"  {'rollout':<12} " + " ".join(f"{k[:9]:>10}" for k in keys)
    print(header)
    for report in reports:
        counts = report["counts"]
        row = f"  {report['path']:<12} " + " ".join(
            f"{counts.get(k, 0):>10}" for k in keys
        )
        print(row)
    # The median height gap, which is the number that separates "closed in the air" from
    # "closed at the row": a bimodal distribution here is itself the result.
    gaps = [e["height_gap_mm"] for r in reports for e in r["events"]
            if e.get("height_gap_mm") is not None and e["kind"] in ("air_close", "missed")]
    if gaps:
        gaps.sort()
        mid = len(gaps) // 2
        median = gaps[mid] if len(gaps) % 2 else 0.5 * (gaps[mid - 1] + gaps[mid])
        print(f"\n  failed closes: n={len(gaps)} height gap median={median:.1f} mm "
              f"(a real grasp closes at ~12 mm)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["2x10", "2x5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    everything = {}
    for task in args.tasks:
        root = DIAG / task
        if not root.is_dir():
            print(f"{task}: no traces at {root}")
            continue
        seeds = args.seeds if args.seeds is not None else sorted(
            int(p.name.split("_")[1]) for p in root.glob("seed_*")
        )
        reports = []
        for seed in seeds:
            path = root / f"seed_{seed}" / "trace.jsonl"
            if not path.exists():
                continue
            reports.append(probe(path))
        everything[task] = reports
        summarise(reports, task)

    if args.json is not None:
        args.json.write_text(json.dumps(everything, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
