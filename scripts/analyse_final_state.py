#!/usr/bin/env python3
"""Where did each cookie end up at the last step?  Read the trace, answer it exactly.

The score and the source-box count cannot separate the failures that matter:

    搬运成功但没入箱        入箱 0，源箱少 5      -- looks identical to
    抓在爪里超时            入箱 0，源箱少 5

`--diagnostic-dir` records `cookies_in_source_mask` and `cookies_in_target_mask` on every
step, so the classification does not have to be guessed from geometry: a cookie is in the
source bin, in the target box, or **in neither**, and the third case is the one worth naming.

An earlier version of this script binned positions by distance from the bin centres and
reported "44 on the table" for an episode in which 46 cookies had simply never moved -- the
source bin is 150 x 170 mm and a radius around its centre is not its extent.  The masks come
from the same code that computes the score, which is the whole reason to trust them.

For the "in neither" cookies the script then asks the one question the masks cannot: are they
still in the jaw?  It reports each one's distance to the effector at the last step as a
distribution rather than a threshold verdict -- a held batch is a tight cluster within a few
centimetres, a dropped one is scattered wherever it fell.

    python scripts/analyse_final_state.py <diag_root> [--effector-key observation.eef_pose]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def seed_dirs(root: Path) -> list[Path]:
    """Every directory holding a finished trace, at either of the two nesting depths.

    `--diagnostic-dir D` is given per seed and `EpisodeDiagnostic` then appends `seed_N`
    itself, so a sweep over seeds ends up one level deeper than a single run.  Accepting
    both is cheaper than making the caller remember which produced what.
    """

    direct = sorted(root.glob("seed_*"))
    if not direct:
        return []
    if any((d / "result.json").is_file() for d in direct):
        return direct
    nested = [child for d in direct for child in sorted(d.glob("seed_*"))]
    return nested or direct


def last_record(trace_path: Path) -> dict:
    """The final per-step record of a trace.

    Read line by line, not loaded whole: a 3200-step trace is tens of megabytes of per-step
    arrays and only the last line is wanted.
    """

    last: dict = {}
    with trace_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def analyse_seed(seed_dir: Path, effector_key: str) -> dict | None:
    result_path, trace_path = seed_dir / "result.json", seed_dir / "trace.jsonl"
    if not (result_path.is_file() and trace_path.is_file()):
        return None

    result = json.loads(result_path.read_text())
    episode = result["episodes"][0] if "episodes" in result else result
    record = last_record(trace_path)
    after = record.get("after") or {}

    positions = np.asarray(after.get("cookie_positions") or [], dtype=float).reshape(-1, 3)
    in_source = np.asarray(after.get("cookies_in_source_mask") or [], dtype=bool)
    in_target = np.asarray(after.get("cookies_in_target_mask") or [], dtype=bool)
    if positions.size == 0 or in_source.size != len(positions):
        return None

    neither = ~in_source & ~in_target
    effector = np.asarray(
        (record.get("observation_before") or {}).get(effector_key) or [], dtype=float
    ).reshape(-1)

    to_hand = (
        np.linalg.norm(positions[neither] - effector[:3], axis=1)
        if neither.any() and effector.size >= 3
        else np.zeros(0)
    )

    return {
        "seed": episode.get("seed"),
        "score": episode.get("score"),
        "max_score": episode.get("max_score"),
        "steps": episode.get("steps"),
        "reason": episode.get("failure_reason"),
        "source": int(in_source.sum()),
        "target": int(in_target.sum()),
        "neither": int(neither.sum()),
        "to_hand": to_hand,
        "effector": effector[:3].tolist() if effector.size >= 3 else None,
        "slots": after.get("target_slot_occupancy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("diag_root", type=Path)
    parser.add_argument("--effector-key", default="observation.eef_pose")
    args = parser.parse_args()

    rows = [r for r in (analyse_seed(d, args.effector_key) for d in seed_dirs(args.diag_root)) if r]
    if not rows:
        print(f"no finished traces under {args.diag_root}")
        return 1

    print(f"{'seed':>4} {'score':>6} {'steps':>5} | {'source':>6} {'target':>6} {'neither':>7} "
          f"| {'distance to effector (mm)':>28}")
    for row in rows:
        mm = row["to_hand"] * 1000.0
        spread = f"{mm.min():.0f}-{mm.max():.0f} (median {np.median(mm):.0f})" if mm.size else "-"
        print(f"{row['seed']:>4} {row['score']:>3}/{row['max_score']:<2} {row['steps']:>5} | "
              f"{row['source']:>6} {row['target']:>6} {row['neither']:>7} | {spread:>28}")

    print()
    for row in rows:
        if row["neither"] == 0:
            verdict = ("搬起的全部进箱，没有丢失" if row["target"] else "一个都没搬")
        else:
            mm = row["to_hand"] * 1000.0
            near = int((mm < 150).sum())
            far = row["neither"] - near
            verdict = (f"{row['neither']} 个既不在源箱也不在目标箱："
                       f"{near} 个在末端 150 mm 内（爪中或刚放开），{far} 个更远（掉落/散落）")
        print(f"seed {row['seed']} ({row['score']}/{row['max_score']}, "
              f"{row['reason'] or '任务完成'}): {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
