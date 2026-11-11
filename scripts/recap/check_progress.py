#!/usr/bin/env python3
"""Report collection progress for both tracks without polling the shell by hand.

    python scripts/recap/check_progress.py
    python scripts/recap/check_progress.py --watch 300

Reads the per-worker logs and the label files rather than the process table, so it answers
"how much usable data exists" rather than "is something running".  Those are different
questions, and during a 7-hour collection the second one is misleading: a worker that has
been computing a single 1500-step episode for twelve minutes looks identical to a hung one.

The two tracks are reported separately because they finish at very different times and a
combined percentage would hide which one is behind.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

LOGS = Path("/root/autodl-tmp/logs")

TRACKS = {
    "rollout": {
        "root": Path("/root/autodl-tmp/datasets/a3_recap_v9roll"),
        "log_glob": "recap_collect_p*.log",
        "target": 300,
        "per_episode_seconds": 720,
    },
    "expert": {
        "root": Path("/root/autodl-tmp/datasets/a3_recap_expert"),
        "log_glob": "recap_expert_p*.log",
        "target": 100,
        "per_episode_seconds": 400,
    },
}


def load_labels(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("part_*/episode_labels.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A line written mid-flush; it will be complete on the next poll.
                    continue
    return rows


def count_workers(running_glob: str) -> int:
    """How many workers are still producing lines, from the log mtimes."""
    now = time.time()
    active = 0
    for path in LOGS.glob(running_glob):
        if now - path.stat().st_mtime < 1800:
            active += 1
    return active


def human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def report() -> None:
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    for name, spec in TRACKS.items():
        root: Path = spec["root"]
        rows = load_labels(root)
        done = len(rows)
        target = spec["target"]
        strict = sum(1 for r in rows if r.get("success"))
        frames = sum(int(r.get("steps", 0)) for r in rows)
        used = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.is_dir() else 0
        active = count_workers(spec["log_glob"])
        remaining = max(target - done, 0)
        eta_hours = remaining / max(active, 1) * spec["per_episode_seconds"] / 3600

        print(f"\n{name}: {done}/{target} episodes ({100 * done / target:.1f} %)"
              f"   workers active: {active}")
        print(f"  strict successes: {strict}"
              f" ({100 * strict / done:.1f} %)" if done else "  strict successes: -")
        print(f"  frames: {frames:,}   on disk: {human_bytes(used)}")
        if done:
            print(f"  mean score: {sum(int(r.get('cookies_in_target', 0)) for r in rows) / done:.2f}"
                  f"   mean steps: {frames // done}")
        if active:
            print(f"  ETA: {eta_hours:.1f} h")
        elif done < target:
            print("  ETA: no active worker -- collection is not running")

    free = shutil.disk_usage("/root/autodl-tmp").free
    print(f"\ndisk free: {human_bytes(free)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", type=float, default=0.0, help="reprint every N seconds")
    args = parser.parse_args()
    while True:
        report()
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
