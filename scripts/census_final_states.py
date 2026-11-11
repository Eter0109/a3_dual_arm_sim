#!/usr/bin/env python3
"""Where every Cookie actually ended up, for the recorded v9 rollouts.

The question: of the points a rollout failed to score, how many are Cookies that reached the
target box and were refused, and where are those Cookies physically?

The judge's own verdict is a conjunction (`_containment_clauses`), and it refuses a Cookie
that is inside the box but not resting on its floor -- which is exactly how a Cookie perched
on the rim is caught.  So "refused" is not one condition, and this script separates them
using the geometry the judge uses:

    box body z        0.74898 m, and the randomisation draws only XY and yaw, so local z is
                      world z minus that constant for every episode
    floor top         local z = 0.006
    rim               local z = 0.026
    COOKIE_HALF_SIZE  [0.025, 0.00317, 0.0125] -- 50 x 6.3 x 25 mm, a slab standing on edge

Which makes the resting height of a Cookie diagnostic on its own, because the two
orientations put its centre in different places:

    standing on the floor   0.7675     standing on the rim     0.7875
    lying on the floor      0.7581     lying on the rim        0.7781

`v9` releases at z = 0.825 and the batches that fail end at 0.7776, which is 0.5 mm from
"lying on the rim" and 10 mm from "standing on the rim".  This script measures the
distribution rather than assuming it.

Usage:

    python scripts/census_final_states.py
    python scripts/census_final_states.py --task 2x5 --json /tmp/census.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIAG = Path("/root/autodl-tmp/diag/v9_run")

#: Constant across episodes: the randomiser moves the bins in XY and yaw only.
BIN_Z = 0.74898
FLOOR_LOCAL = 0.006
RIM_LOCAL = 0.026
#: COOKIE_HALF_SIZE from the environment: (x, y, z) half lengths in metres.
HALF = np.array([0.025, 0.00316667, 0.0125])

#: The four resting poses, as world-z of the Cookie's centre.
POSES = {
    "standing on floor": BIN_Z + FLOOR_LOCAL + HALF[2],
    "lying on floor": BIN_Z + FLOOR_LOCAL + HALF[1],
    "standing on rim": BIN_Z + RIM_LOCAL + HALF[2],
    "lying on rim": BIN_Z + RIM_LOCAL + HALF[1],
}


def nearest_pose(z: float) -> tuple[str, float]:
    name, target = min(POSES.items(), key=lambda item: abs(item[1] - z))
    return name, (z - target) * 1000.0


def census(task: str, seed: int) -> dict:
    root = DIAG / task / f"seed_{seed}"
    diagnostics = json.loads((root / "success_diagnostics.json").read_text())
    final = diagnostics["final_conditions"]
    positions = np.asarray(final["cookie_positions"], dtype=np.float64)
    in_target = np.asarray(final["cookie_in_target_mask"]
                          if "cookie_in_target_mask" in final
                          else final["cookies_in_target_mask"], dtype=bool)
    in_source = np.asarray(final["cookies_in_source_mask"], dtype=bool)

    required = 20 if task == "2x10" else 10
    score = int(final["cookies_in_target"])

    # A Cookie in neither mask is loose: it left the source and the judge does not count it
    # in the box, so it is either in the box and refused, or outside both bins.
    loose = np.flatnonzero(~in_target & ~in_source)
    rows = []
    for index in loose:
        z = float(positions[index, 2])
        pose, offset = nearest_pose(z)
        rows.append({
            "cookie": int(index),
            "x": round(float(positions[index, 0]), 4),
            "y": round(float(positions[index, 1]), 4),
            "z": round(z, 5),
            "local_z": round(z - BIN_Z, 5),
            "nearest_pose": pose,
            "offset_mm": round(offset, 1),
            # The target box spans y = 0.100 +/- 0.051, so anything inside that band left the
            # source and came back over the box rather than landing on the table.
            "near_target_y": bool(0.030 < positions[index, 1] < 0.170),
        })

    return {
        "task": task,
        "seed": seed,
        "required": required,
        "score": score,
        "deficit": required - score,
        "counted": int(in_target.sum()),
        "in_source": int(in_source.sum()),
        "loose": len(rows),
        "loose_rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", default=["2x5", "2x10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    everything = []
    for task in args.task:
        root = DIAG / task
        if not root.is_dir():
            continue
        seeds = args.seeds or sorted(int(p.name.split("_")[1]) for p in root.glob("seed_*"))
        print(f"\n=== {task} ===")
        header = (f"{'seed':>4} {'score':>6} {'need':>5} {'def':>4} {'counted':>8} "
                  f"{'src':>4} {'loose':>6}  loose resting poses")
        print(header)
        print("-" * len(header))
        for seed in seeds:
            report = census(task, seed)
            everything.append(report)
            from collections import Counter
            poses = Counter(row["nearest_pose"] for row in report["loose_rows"])
            summary = ", ".join(f"{k}x{v}" for k, v in poses.most_common())
            print(f"{seed:>4} {report['score']:>6} {report['required']:>5} "
                  f"{report['deficit']:>4} {report['counted']:>8} {report['in_source']:>4} "
                  f"{report['loose']:>6}  {summary}")

    # The headline number: of all the points nobody scored, how many are explained by a
    # Cookie that reached the box and was refused.
    for task in args.task:
        rows = [r for r in everything if r["task"] == task]
        if not rows:
            continue
        deficit = sum(r["deficit"] for r in rows)
        refused = sum(1 for r in rows for row in r["loose_rows"] if row["near_target_y"])
        on_rim = sum(1 for r in rows for row in r["loose_rows"]
                     if row["near_target_y"] and row["nearest_pose"].endswith("on rim"))
        print(f"\n{task}: deficit {deficit} points over {len(rows)} rollouts")
        print(f"  Cookies that reached the box and were refused: {refused}")
        print(f"  ... of those, resting ON THE RIM:              {on_rim}")

    if args.json is not None:
        args.json.write_text(json.dumps(everything, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
