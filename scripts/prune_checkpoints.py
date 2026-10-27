#!/usr/bin/env python3
"""Keep a run's checkpoint set inside the disk budget.

Each pi0.5 checkpoint is 21 GB -- the 3B base model plus the optimiser state -- and this
lerobot has no `save_total_limit`, so every `save_freq` writes a full copy and nothing is
ever removed.  A 15000-step run at a 1500-step interval therefore wants 210 GB out of the
130 GB left on the data disk, and dies of ENOSPC halfway through a checkpoint write, which
is how the aggregation step died before.

Deleting from the middle is deliberate.  The newest directory may still be being written and
must not be touched; the earliest is the anchor that makes the checkpoint set a curve rather
than a snapshot, and it is the one the first auxiliary-loss check runs against.  What sits
between them is on the line joining them and can be recomputed.

Two rules make this safe beside a live training process:

  * the newest `--keep-last` checkpoints are never touched;
  * the directory `checkpoints/last` points at is never touched, because the evaluation
    pipeline resolves it and a dangling symlink is worse than a full disk.

   python scripts/prune_checkpoints.py <run_dir> [--keep-first 1] [--keep-last 2]

Without `--once` it loops every `--interval` seconds until killed, which is how the training
pipeline starts it; one line per deletion says what happened.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def checkpoints(run_dir: Path) -> list[Path]:
    """Step-numbered checkpoint directories, oldest first.

    `last` is a symlink and must not be mistaken for a checkpoint.  A directory only counts
    once `pretrained_model/model.safetensors` exists, because a half-written one would sort
    last by name, be protected as "newest", and then still be written into -- the naive
    version protects garbage and deletes a good checkpoint in the same pass.
    """

    root = run_dir / "checkpoints"
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir() or not path.name.isdigit():
            continue
        if (path / "pretrained_model" / "model.safetensors").is_file():
            found.append(path)
    return sorted(found, key=lambda p: int(p.name))


def prune(run_dir: Path, keep_first: int, keep_last: int) -> list[Path]:
    found = checkpoints(run_dir)
    if len(found) <= keep_first + keep_last:
        return []

    protected = {p.resolve() for p in found[:keep_first]}
    protected |= {p.resolve() for p in found[len(found) - keep_last:]}
    last = run_dir / "checkpoints" / "last"
    if last.is_symlink():
        protected.add(last.resolve())

    removed = []
    for path in found:
        if path.resolve() in protected:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune old checkpoints of a run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--keep-first", type=int, default=1)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        for path in prune(args.run_dir, args.keep_first, args.keep_last):
            print(f"{time.strftime('%H:%M')} pruned {path.name}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
