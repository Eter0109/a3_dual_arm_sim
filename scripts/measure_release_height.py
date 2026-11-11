#!/usr/bin/env python3
"""What release height does the expert demonstrate, batch one versus batch two?

The policy releases ~50 mm above the box rim and the batch balances on it; the expert
demonstrations release *below* the rim.  Before treating that as "the model did not learn
it", the demonstrations themselves have to be checked: a dataset whose first batch is also
released high would teach exactly the behaviour that is being observed, and the fix would
be different (the batch is not identifiable) from a dataset that is uniformly low (the
model simply fails to copy what it saw).

Measured here, per episode:

    rim        0.7755 m -- the target box's wall top
    release    the effector's height at the moment the left jaw opens past `OPEN_AT`
    batch      which group this is, from how many Cookies were already in the target

The jaw channel is `action[:, 7]`; the effector height is `observation.eef_pose[:, 2]`.
Both are the recorded channels the policy itself sees, so this measures the demonstration
rather than a reconstruction of it.

One subtlety that decides whether the numbers mean anything: the expert *holds* its jaws at
an insertion width while descending, so an "open" threshold near the travel's range would
fire during the approach.  The expert's own release opening is ~0.49 (measured in
`batch_expert.py::_set_release_opening`), and its insertion width is 0.28-0.33, so the
threshold sits at 0.45 -- above insertion, at the release.

Usage:

    python scripts/measure_release_height.py
    python scripts/measure_release_height.py --episodes 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path("/root/autodl-tmp/datasets/a3_clean_v9/data/chunk-000/file-000.parquet")

#: The target box's wall top, from `configs/cookie_batch.yaml` (wall height 0.026 above a
#: floor at 0.7495).  A Cookie placed correctly rests at 0.7685.
RIM_Z = 0.7755
FLOOR_Z = 0.7685

JAW = 7
#: Jaw opening that counts as released: above the expert's insertion width (0.28-0.33),
#: at its release opening (0.49).
OPEN_AT = 0.45
#: The effector has to be over the target box for the release to be one.  The box spans
#: y = 0.100 +/- 0.035, and its approach corridor is a few centimetres beyond.
TARGET_Y = (0.045, 0.165)
TARGET_X = (0.010, 0.180)


def episodes(table, limit: int | None) -> list[int]:
    labels = np.asarray(table["episode_index"])
    order = np.unique(labels)
    return order[:limit] if limit else order


def main() -> int:
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    table = pq.read_table(
        DATA, columns=["observation.eef_pose", "action", "episode_index"]
    )
    eef = np.asarray(table["observation.eef_pose"].to_pylist(), dtype=np.float64)
    act = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    label = np.asarray(table["episode_index"])

    rows: list[dict] = []
    for episode in episodes(table, args.episodes):
        span = np.flatnonzero(label == episode)
        if span.size < 40:
            continue
        pose = eef[span]
        jaw = act[span, JAW]
        z, y, x = pose[:, 2], pose[:, 1], pose[:, 0]

        # A release is a rising crossing of the jaw channel while over the target box.
        # Crossings within `MERGE` frames are one release (the channel can wobble).
        releases: list[int] = []
        for step in range(2, len(jaw) - 2):
            if jaw[step - 2] < OPEN_AT <= jaw[step + 2]:
                in_box = TARGET_Y[0] < y[step] < TARGET_Y[1]
                in_x = TARGET_X[0] < x[step] < TARGET_X[1]
                if in_box and in_x:
                    if not releases or step - releases[-1] > 60:
                        releases.append(step)

        for order, step in enumerate(releases):
            rows.append(
                {
                    "episode": int(episode),
                    "batch": order + 1,
                    "frame": int(step),
                    "release_z": float(z[step]),
                    "vs_rim_mm": 1000.0 * (float(z[step]) - RIM_Z),
                }
            )

    if not rows:
        print("no releases found")
        return 1

    by_batch: dict[int, list[float]] = {}
    for row in rows:
        by_batch.setdefault(row["batch"], []).append(row["release_z"])

    print(f"{len(rows)} releases over {len({r['episode'] for r in rows})} episodes")
    print(f"rim {RIM_Z:.4f}  floor {FLOOR_Z:.4f}\n")
    header = f"{'batch':>5} {'n':>5} {'median z':>9} {'mean z':>8} {'vs rim':>8} " \
             f"{'p10':>8} {'p90':>8} {'above rim':>10}"
    print(header)
    print("-" * len(header))
    for batch in sorted(by_batch):
        values = np.asarray(by_batch[batch])
        above = int(np.count_nonzero(values > RIM_Z))
        print(
            f"{batch:>5} {values.size:>5} {np.median(values):>9.4f} "
            f"{values.mean():>8.4f} {1000.0 * (np.median(values) - RIM_Z):>+7.1f}mm "
            f"{np.percentile(values, 10):>8.4f} {np.percentile(values, 90):>8.4f} "
            f"{above:>4}/{values.size:<5}"
        )

    print()
    print("for comparison, the policy on 2x5/2x10 released at z = 0.825-0.833 "
          "(+50 to +58 mm over the rim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
