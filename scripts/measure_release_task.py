#!/usr/bin/env python3
"""Release height, measured per task, with the demonstrations split correctly.

Why this script replaced two earlier ones.  `a3_clean_v9` holds **two tasks in one file** --
episodes 0-199 are the 2x10 demonstrations and 200-399 are the 2x5 ones -- and both earlier
measurements sampled the first 80 episodes without checking, so every "demonstration" number
reported was a 2x10 number.  That produced a wrong conclusion (that demonstrations release
50 mm above the rim) and a wrong story about how the policy differs from them.

The second mistake was in the detector.  A "release" was taken to be a rising crossing of the
gripper channel through 0.45, but the expert opens to ~1.0 when it starts its retreat, so that
crossing marks the *departure*, not the letting-go.  The letting-go is a much smaller rise --
0.28 (the insertion width the jaws descend at) to 0.38-0.43 -- and it happens with the
effector still inside the box.

So a release is now: the gripper channel rising by more than `RISE` over `SPAN` steps, while
the effector is over the target box **and at or below the rim**.  Anything above the rim is a
transit, not a placement.

Usage:

    python scripts/measure_release_task.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEMO = Path("/root/autodl-tmp/datasets/a3_clean_v9/data/chunk-000/file-000.parquet")
EPISODES = Path("/root/autodl-tmp/datasets/a3_clean_v9/meta/episodes")
POLICY = Path("/root/autodl-tmp/diag/v9_run")

#: Box rim at wall height 0.026 above a body at z = 0.74898.  A release above this is in the
#: air; the placements all happen below it.
RIM_Z = 0.77498
JAW = 7
RISE = 0.030
SPAN = 4
TARGET_Y = (0.045, 0.165)
TARGET_X = (0.010, 0.180)
#: 2x5 slot columns and 2x10 slot columns are the same two x values.
COLUMNS = (0.067, 0.123)


def releases(z, x, y, jaw) -> list[dict]:
    out: list[dict] = []
    last = -10 ** 9
    for step in range(SPAN, len(jaw) - SPAN):
        if not (TARGET_X[0] < x[step] < TARGET_X[1] and TARGET_Y[0] < y[step] < TARGET_Y[1]):
            continue
        if z[step] > RIM_Z + 0.006:
            continue
        if jaw[step + SPAN] - jaw[step - SPAN] <= RISE:
            continue
        if step - last <= 60:
            continue
        last = step
        out.append({
            "frame": int(step),
            "z": float(z[step]),
            "x": float(x[step]),
            "jaw": float(jaw[step]),
            "off_column_mm": 1000.0 * min(abs(float(x[step]) - c) for c in COLUMNS),
        })
    return out


def report(label: str, rows: list[dict], episodes: int) -> None:
    if not rows:
        print(f"\n{label}: no releases")
        return
    z = np.asarray([r["z"] for r in rows])
    off = np.asarray([r["off_column_mm"] for r in rows])
    print(f"\n{label}")
    print(f"  releases {len(rows)} over {episodes} episodes "
          f"({len(rows)/max(episodes,1):.2f} per episode)")
    print(f"  release z    median {np.median(z):.4f}  "
          f"({1000*(np.median(z)-RIM_Z):+.1f} mm vs rim)   "
          f"p10 {np.percentile(z,10):.4f}  p90 {np.percentile(z,90):.4f}")
    print(f"  above the rim: {int((z > RIM_Z).sum())}/{len(rows)}")
    print(f"  off-column   median {np.median(off):.1f} mm   "
          f"p90 {np.percentile(off,90):.1f} mm")


def main() -> int:
    import pyarrow.parquet as pq

    tasks = pq.read_table(EPISODES, columns=["episode_index", "tasks"])
    index = np.asarray(tasks["episode_index"])
    labels = tasks["tasks"].to_pylist()
    kind = {}
    for position, episode in enumerate(index):
        text = labels[position]
        text = text[0] if isinstance(text, (list, np.ndarray)) else text
        kind[int(episode)] = "2x5" if text.startswith("Transfer 10") else "2x10"
    print("demonstration episodes:", {k: sum(1 for v in kind.values() if v == k)
                                     for k in ("2x5", "2x10")})

    table = pq.read_table(DEMO, columns=["observation.eef_pose", "action", "episode_index"])
    eef = np.asarray(table["observation.eef_pose"].to_pylist(), dtype=np.float64)
    act = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    label = np.asarray(table["episode_index"])

    for task in ("2x5", "2x10"):
        rows: list[dict] = []
        episodes = [e for e in np.unique(label) if kind.get(int(e)) == task]
        # Twenty episodes per task is plenty for a median and keeps this to a minute.
        for episode in episodes[:20]:
            span = np.flatnonzero(label == episode)
            if span.size < 40:
                continue
            rows.extend(releases(eef[span, 2], eef[span, 0], eef[span, 1], act[span, JAW]))
        report(f"EXPERT demonstrations ({task})", rows, min(len(episodes), 20))

    for task, root in (("2x5", POLICY / "2x5"), ("2x10", POLICY / "2x10")):
        rows = []
        directories = sorted(root.glob("seed_*"))
        for directory in directories:
            truth = directory / "trace.jsonl"
            if not truth.exists():
                continue
            z, x, y, jaw = [], [], [], []
            with truth.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    pose = row["observation_before"]["observation.eef_pose"]
                    applied = row.get("applied_action")
                    z.append(pose[2]); x.append(pose[0]); y.append(pose[1])
                    jaw.append(applied[JAW] if applied else np.nan)
            rows.extend(releases(np.asarray(z), np.asarray(x), np.asarray(y),
                                 np.asarray(jaw)))
        report(f"v9 policy ({task})", rows, len(directories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
