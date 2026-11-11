#!/usr/bin/env python3
"""Rank a recorded benchmark run by cookies placed and stage the top-N episodes.

The recording run is `bench_arm.sh --record`, which is main's own `--diagnostic-dir`.
That writes one directory per episode holding:

    front.mp4, left_wrist.mp4, right_wrist.mp4   # 256x256, control rate
    trace.jsonl                                  # per-step state / action / images
    result.json, success_diagnostics.json

The score is `cookies_in_target` at episode end.  On 2x5 the cap is 10 and on 2x10 it is
20, so the two tasks are ranked separately -- a 2x10 run that places 12 is not "better"
than a 2x5 run that places 9, it is a different task.

Nothing is re-encoded: the videos are copied byte for byte so the file a reviewer opens is
the file the policy produced.  Rendering bigger for a nicer video would change the input
that was measured, which is the one thing a demonstration clip must not do.

Usage:
    pick_best_episodes.py --record DIR [--top 3] [--stage OUT] [--task 2x5|2x10]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

CAMS = ("front", "left_wrist", "right_wrist")
CAP = {"2x5": 10, "2x10": 20}


def seed_dirs(task_dir: Path):
    """EpisodeDiagnostic nests one extra seed_N, so accept both depths.

    `--diagnostic-dir DIR` makes DIR/<seed_N>/... while a per-episode call can land at
    DIR itself.  Both appear in practice depending on how the driver was invoked.
    """
    for child in sorted(task_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "result.json").exists() or (child / "trace.jsonl").exists():
            yield child
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (
                (grandchild / "result.json").exists() or (grandchild / "trace.jsonl").exists()
            ):
                yield grandchild


def load_result(ep_dir: Path) -> dict:
    """Score comes from result.json; fall back to the last trace line's summary.

    A killed episode leaves no result.json but does leave a trace, and an episode that ran
    out of steps is still a legitimate sample -- it is how the policy actually ended.
    """
    path = ep_dir / "result.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            episodes = data.get("episodes") or [data]
            if episodes:
                return episodes[0]
        except Exception:
            pass
    trace = ep_dir / "trace.jsonl"
    if trace.exists():
        last = None
        with trace.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    last = line
        if last:
            try:
                row = json.loads(last)
                for key in ("result", "episode", "summary"):
                    if isinstance(row.get(key), dict):
                        return row[key]
            except Exception:
                pass
    return {}


def seed_of(ep_dir: Path) -> int:
    match = re.search(r"seed_(\d+)", str(ep_dir))
    return int(match.group(1)) if match else -1


def video_set(ep_dir: Path) -> dict:
    """Locate the three camera files, which may sit one level down."""
    for base in (ep_dir, *[p for p in sorted(ep_dir.iterdir()) if p.is_dir()]):
        files = {cam: base / f"{cam}.mp4" for cam in CAMS}
        if all(f.exists() and f.stat().st_size > 1024 for f in files.values()):
            return files
    return {cam: ep_dir / f"{cam}.mp4" for cam in CAMS}


def judge(record: Path, task: str, top: int, stage: Path | None) -> int:
    task_dir = record / task
    if not task_dir.is_dir():
        print(f"{task}: no directory at {task_dir}", file=sys.stderr)
        return 1

    cap = CAP[task]
    rows = []
    for ep_dir in seed_dirs(task_dir):
        result = load_result(ep_dir)
        score = result.get("cookies_in_target", result.get("score"))
        if score is None:
            continue
        rows.append(
            {
                "seed": seed_of(ep_dir),
                "score": int(score),
                # `cookies_in_source` is how many are still in the source box at the end,
                # NOT how many were carried.  Labeling it "moved" reads as "70 cookies
                # were transferred" when the clip actually placed 10 and left 70 behind.
                "left_in_source": result.get("cookies_in_source"),
                "steps": result.get("steps"),
                "dir": ep_dir,
                "files": video_set(ep_dir),
            }
        )

    if not rows:
        print(f"{task}: no scored episodes under {task_dir}", file=sys.stderr)
        return 1

    # Descending by score, then by fewer left in source so a tie goes to the episode that
    # actually emptied more of the source.  Seed breaks the remaining ties so the order is
    # stable between runs.
    rows.sort(key=lambda r: (-r["score"], r["left_in_source"] or 0, r["seed"]))

    scores = [r["score"] for r in rows]
    print(f"\n=== {task} ===  n={len(rows)}  mean {sum(scores) / len(scores):.2f}"
          f"  cap {cap}")
    for row in rows:
        mark = "*" if rows.index(row) < top else " "
        print(f" {mark} seed {row['seed']:>2}  {row['score']:>2}/{cap}"
              f"  left-in-source {row['left_in_source']}  steps {row['steps']}")

    if stage is None:
        return 0

    stage.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows[:top], start=1):
        # Name carries the score and the seed so the clip identifies itself after it is
        # copied out of the run directory.
        target = stage / f"{task}_rank{rank}_seed{row['seed']}_{row['score']}of{cap}"
        target.mkdir(parents=True, exist_ok=True)
        for cam, src in row["files"].items():
            if src.exists():
                shutil.copy2(src, target / f"{cam}.mp4")
        for extra in ("trace.jsonl", "result.json", "success_diagnostics.json"):
            src = row["dir"] / extra
            if not src.exists():
                # One nesting level down, which happens for the diagnostic subdirectories.
                nested = [p / extra for p in row["dir"].iterdir() if p.is_dir()]
                src = next((p for p in nested if p.exists()), src)
            if src.exists():
                shutil.copy2(src, target / extra)
        print(f"  staged {target}")

    manifest = [
        {
            "task": task,
            "rank": rank,
            "seed": row["seed"],
            "cookies_in_target": row["score"],
            "cookies_left_in_source": row["left_in_source"],
            "steps": row["steps"],
            "source": str(row["dir"]),
        }
        for rank, row in enumerate(rows[:top], start=1)
    ]
    (stage / f"{task}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest -> {stage / f'{task}_manifest.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True,
                        help="the directory passed to bench_arm.sh --record")
    parser.add_argument("--top", type=int, default=3, help="how many per task (default 3)")
    parser.add_argument("--stage", type=Path, default=None,
                        help="copy the winners here; default is to only print the ranking")
    parser.add_argument("--task", choices=sorted(CAP), default=None,
                        help="restrict to one task; default is both")
    args = parser.parse_args()

    if not args.record.is_dir():
        print(f"no such record directory: {args.record}", file=sys.stderr)
        return 1

    tasks = [args.task] if args.task else ["2x5", "2x10"]
    status = 0
    for task in tasks:
        status |= judge(args.record, task, args.top, args.stage)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
