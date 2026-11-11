#!/usr/bin/env python3
"""Compare the release itself: expert demonstrations against the v9 rollouts.

Both release at z = 0.825, so the height is not the difference -- and yet the expert's
episodes are accepted and the policy's batches end up lying on the box rim.  Something
about *how* the jaws open has to differ, and this measures the candidates that the recorded
channels can settle:

    jaw      the gripper channel at the crossing -- a group of five needs enough room to
             fall clear of the pads, and too narrow a release drags the outer two
    speed    horizontal effector speed -- releasing while moving carries the group
             sideways before it has fallen past the rim
    offset   where the release happens in x, against the slot columns at 0.067 / 0.123
             (2x5) or the ten-row columns at 0.067 / 0.123 (2x10); a group dropped
             off-centre puts its outer Cookies over the rim

Both sides are measured from the channels the policy itself is conditioned on, so this is a
like-for-like comparison rather than a reconstruction.

Usage:

    python scripts/compare_release_mechanics.py
    python scripts/compare_release_mechanics.py --demo-episodes 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEMO = Path("/root/autodl-tmp/datasets/a3_clean_v9/data/chunk-000/file-000.parquet")
POLICY_2X5 = Path("/root/autodl-tmp/diag/v9_run/2x5")
POLICY_2X10 = Path("/root/autodl-tmp/diag/v9_run/2x10")

RIM_Z = 0.7755
JAW = 7
OPEN_AT = 0.45
TARGET_Y = (0.045, 0.165)
TARGET_X = (0.010, 0.180)
#: The two slot columns, in x.  Both configs put them at 0.095 +/- 0.028.
COLUMNS = (0.067, 0.123)


def releases(z: np.ndarray, x: np.ndarray, y: np.ndarray, jaw: np.ndarray,
             limit: int | None = None) -> list[dict]:
    """Rising crossings of the jaw channel while over the target box."""

    out: list[dict] = []
    last = -10 ** 9
    span = range(2, len(jaw) - 2) if limit is None else range(2, min(len(jaw) - 2, limit))
    for step in span:
        if not (jaw[step - 2] < OPEN_AT <= jaw[step + 2]):
            continue
        if not (TARGET_Y[0] < y[step] < TARGET_Y[1] and TARGET_X[0] < x[step] < TARGET_X[1]):
            continue
        if step - last <= 60:
            continue
        last = step
        # Horizontal speed over 5 steps either side, in m/s at 20 Hz.
        lo, hi = max(step - 5, 0), min(step + 5, len(z) - 1)
        speed = float(np.hypot(x[hi] - x[lo], y[hi] - y[lo])) / ((hi - lo) / 20.0)
        out.append({
            "frame": int(step),
            "z": float(z[step]),
            "jaw": float(jaw[step]),
            "speed": speed,
            "x": float(x[step]),
            "y": float(y[step]),
            "off_column_mm": 1000.0 * min(abs(float(x[step]) - c) for c in COLUMNS),
        })
    return out


def stats(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{label}: no releases")
        return
    def col(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows])

    z = col("z")
    print(f"\n{label}: n={len(rows)}")
    print(f"  release z     median {np.median(z):.4f}  ({1000*(np.median(z)-RIM_Z):+.1f} mm vs rim)")
    print(f"  jaw at release median {np.median(col('jaw')):.3f}  "
          f"p10 {np.percentile(col('jaw'),10):.3f}  p90 {np.percentile(col('jaw'),90):.3f}")
    print(f"  horiz speed   median {np.median(col('speed')):.3f} m/s  "
          f"p90 {np.percentile(col('speed'),90):.3f}")
    print(f"  off-column    median {np.median(col('off_column_mm')):.1f} mm  "
          f"p90 {np.percentile(col('off_column_mm'),90):.1f}")
    print(f"  released above rim: {int((z > RIM_Z).sum())}/{len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-episodes", type=int, default=80)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    table = pq.read_table(DEMO, columns=["observation.eef_pose", "action", "episode_index"])
    eef = np.asarray(table["observation.eef_pose"].to_pylist(), dtype=np.float64)
    act = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    labels = np.asarray(table["episode_index"])

    demo_rows: list[dict] = []
    for episode in np.unique(labels)[: args.demo_episodes]:
        span = np.flatnonzero(labels == episode)
        if span.size < 40:
            continue
        demo_rows.extend(releases(eef[span, 2], eef[span, 0], eef[span, 1], act[span, JAW]))
    stats("EXPERT demonstrations (2x5, accepted)", demo_rows)

    for task, root in (("2x5", POLICY_2X5), ("2x10", POLICY_2X10)):
        rows: list[dict] = []
        import json
        for seed_dir in sorted(root.glob("seed_*")):
            trace = seed_dir / "trace.jsonl"
            if not trace.exists():
                continue
            z, x, y, jaw = [], [], [], []
            with trace.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    pose = row["observation_before"]["observation.eef_pose"]
                    applied = row.get("applied_action")
                    z.append(pose[2]); x.append(pose[0]); y.append(pose[1])
                    jaw.append(applied[JAW] if applied else np.nan)
            rows.extend(releases(np.asarray(z), np.asarray(x), np.asarray(y),
                                 np.asarray(jaw)))
        stats(f"v9 policy ({task})", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
