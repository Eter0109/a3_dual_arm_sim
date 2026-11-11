"""Where does the policy actually fail: getting the Cookie, or seating it?

The distinction decides what to do next, and it is cheap to measure from traces that already
exist.  Two failure modes look identical in the final score and need opposite remedies:

* **Reach/grasp failures** -- the Cookie is never lifted.  This is a *coverage* problem: the
  policy has not seen enough of the relevant states, so the answer is more or better data.
* **Seat/precision failures** -- the Cookie is delivered into the box but lands outside a slot,
  or a slot is occupied and later lost.  This is an *accuracy* problem, and no amount of extra
  data of the same kind fixes it.

Measured context: the expert reaches 94 % strict success on the same randomised scenes while v9
reaches 12.5 %, and the randomisation is only ±10 mm.  So the task is solvable and the scene is
not the variable -- but the traces have never been asked *which* step the policy loses.

How the two are told apart, per Cookie
--------------------------------------
`target_slot_occupancy` names, for each of the 20 target slots, which Cookie index occupies it
(``-1`` when empty).  That gives an exact per-Cookie history:

* ``ever_in_slot``   -- the Cookie occupied some slot at least once;
* ``seated_at_end``  -- it still does at the final frame;
* ``lost_after_seating`` -- it was seated and then pushed out (a precision/contact failure even
  though the Cookie arrived);
* ``never_seated``   -- it entered the box but never aligned to a slot (pure alignment failure);
* ``never_in_box``   -- it never entered the box at all (delivery failure).

Reporting by category rather than by episode matters: one episode can contain a mix, and an
episode-level label would hide that nearly all 20 Cookies arrive while 15 of them miss their
slot.

Usage
-----
    python scripts/recap/diagnose_failures.py --tags v9 succ good --seeds 24
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def load_traces(tag: str, root: Path, limit: int | None) -> dict[int, list[dict]]:
    traces: dict[int, list[dict]] = {}
    for path in sorted(glob.glob(str(root / f"eval_{tag}" / "part_*" / "traces" / "seed_*" / "trace.jsonl"))):
        seed = int(Path(path).parent.name.split("_")[1])
        traces[seed] = [json.loads(line) for line in open(path)]
    seeds = sorted(traces)
    if limit is not None:
        seeds = seeds[:limit]
    return {seed: traces[seed] for seed in seeds}


def classify(rows: list[dict]) -> dict[str, int]:
    """Per-Cookie outcome counts for one episode."""
    ever_in_box: set[int] = set()
    ever_seated: set[int] = set()
    seated_final: set[int] = set()
    for row in rows:
        after = row.get("after") or {}
        occupancy = after.get("target_slot_occupancy") or []
        # The box membership of a specific Cookie is not in the trace; `cookies_in_target` is a
        # count.  What the occupancy table does give is which Cookies were *seated*, and a seated
        # Cookie is by construction inside the box (`_target_slot_occupancy` only considers
        # Cookies the environment already counts as inside).
        for value in occupancy:
            if isinstance(value, int) and value >= 0:
                ever_seated.add(value)
                ever_in_box.add(value)
        seated_final = {v for v in occupancy if isinstance(v, int) and v >= 0}

    in_box_final = int((rows[-1].get("after") or {}).get("cookies_in_target") or 0)
    return {
        "seated_at_end": len(seated_final),
        "ever_seated": len(ever_seated),
        "lost_after_seating": len(ever_seated - seated_final),
        # Cookies that ended in the box without ever holding a slot.  `in_box_final` counts them,
        # and `ever_seated` does not, so the difference bounds the alignment failures -- it is a
        # *bound* because a Cookie can be seated, then leave the box entirely, which also widens
        # the difference.
        "in_box_final": in_box_final,
        "in_box_without_a_slot": max(0, in_box_final - len(seated_final)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/datasets"))
    parser.add_argument("--tags", nargs="+", default=["v9"])
    parser.add_argument("--seeds", type=int, default=24, help="how many of the shared seeds to use")
    args = parser.parse_args(argv)

    for tag in args.tags:
        traces = load_traces(tag, args.root, args.seeds)
        if not traces:
            print(f"[diag] {tag}: no traces")
            continue

        totals = Counter()
        episodes = 0
        perfect_seat = 0
        for seed, rows in traces.items():
            if not rows:
                continue
            episodes += 1
            counts = classify(rows)
            totals.update(counts)
            if counts["seated_at_end"] >= 20:
                perfect_seat += 1

        n = max(episodes, 1)
        print()
        print(f"=== {tag}  ({episodes} episodes) ===")
        print(f"  {'终局入槽数':22s} {totals['seated_at_end'] / n:6.2f} / 20")
        print(f"  {'曾入槽数':22s} {totals['ever_seated'] / n:6.2f} / 20")
        print(f"  {'终局盒内数':22s} {totals['in_box_final'] / n:6.2f} / 20")
        print(f"  {'盒内但无槽位（对齐失败）':22s} {totals['in_box_without_a_slot'] / n:6.2f} / 20")
        print(f"  {'入槽后又被顶出':22s} {totals['lost_after_seating'] / n:6.2f} / 20")
        print(f"  {'20 个全入槽的集数':22s} {perfect_seat}/{episodes}")

        # The split that decides the next move.  "Lost after seating" and "in box without a slot"
        # are both precision failures -- the Cookie arrived.  Delivery failures are the remainder
        # of the 20, and those are what more/better data would fix.
        arrived_but_missed = totals["in_box_without_a_slot"] + totals["lost_after_seating"]
        never_arrived = 20 * n - totals["ever_seated"] - 0
        print()
        print(f"  → 到达但对不齐 : {arrived_but_missed / n:5.2f} /20 "
              f"({100 * arrived_but_missed / max(20 * n, 1):.0f} % of all Cookie-attempts)")
        print(f"  → 从未入槽     : {max(0, never_arrived) / n:5.2f} /20 "
              f"({100 * max(0, never_arrived) / max(20 * n, 1):.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
