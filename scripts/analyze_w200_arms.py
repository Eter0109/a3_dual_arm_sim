#!/usr/bin/env python3
"""Compare the per-step-window 2x10 arms against the historical per-chunk ones.

Three things this has to get right, because each one has already produced a wrong
conclusion once in this repo:

1. **An arm's total n must be printed next to any paired n.**  9.53 read an 8-seed
   snapshot as if it were the whole arm because the pairing was done on the seed
   intersection and that intersection happened to be seeds 0--7.

2. **A task file can hold more than one task.**  `a3_clean_v9` stores 2x10 in episodes
   0--199 and 2x5 in 200--399, so any per-arm figure computed without checking which
   scene each episode came from is measuring a mixture.  Here every row comes from a
   run started with `--task 2x10`.

3. **Pairing is nearly worthless on this benchmark.**  sd of the paired difference lands
   within a few percent of sqrt(var_a + var_b), which is what you get from two
   independent samples.  Both SEs are printed so the reader can see that for themselves
   rather than trusting the paired p-value.

Usage:
    analyze_w200_arms.py            # tonight's arms, then the historical comparison
    analyze_w200_arms.py --watch    # reprint as seeds land
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

LOGS = Path("/root/autodl-tmp/logs")

#: tag -> (label, per-episode log glob).  `w200` is tonight's per-step arm.  `v9_step` is
#: the 10-frame control run the same night on the same seeds.  The two `v9s`/`v9w200`
#: rows are the historical arms, both measured with the per-chunk window (9.73).
ARMS = {
    "v9_step": ("v9  10 fr  per-step", "v9_step_2x10_*.log"),
    "w200": ("w200 200 fr per-step", "w200_2x10_*.log"),
    "v9s": ("v9  10 fr  per-chunk (old)", "v9s_2x10_*.log"),
    "v9w200": ("w200 200 fr per-chunk (old)", "v9w200_2x10_*.log"),
}

#: A single-file multi-seed run writes one log with many `Seed N | Score:` lines; the
#: historical sweep wrote one log per seed.  Both shapes are parsed by the same regex.
SEED_LINE = re.compile(r"Seed\s+(\d+) \| Score:\s*(\d+)/20")


def load(pattern: str) -> dict[int, int]:
    scores: dict[int, int] = {}
    for path in sorted(LOGS.glob(pattern)):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for seed, score in SEED_LINE.findall(text):
            scores[int(seed)] = int(score)
    return scores


def describe(name: str, scores: dict[int, int]) -> None:
    v = np.array(list(scores.values()))
    if not len(v):
        print(f"  {name}: no episodes yet")
        return
    mean, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    se = sd / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    distributed = dict(sorted(Counter(v.tolist()).items()))
    print(f"  {name:24s} n={len(v):3d}  mean {mean:5.2f} +/-{se:.2f}  sd {sd:4.2f}")
    print(f"  {'':24s} dist {distributed}")


def compare(a: dict[int, int], b: dict[int, int], name_a: str, name_b: str) -> None:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        print(f"  {name_a} vs {name_b}: only {len(common)} shared seeds, nothing to say")
        return
    da = np.array([a[s] for s in common], dtype=float)
    db = np.array([b[s] for s in common], dtype=float)
    diff = db - da
    paired_se = diff.std(ddof=1) / np.sqrt(len(common))
    independent = np.sqrt(da.var(ddof=1) / len(common) + db.var(ddof=1) / len(common))
    print(
        f"  {name_a} -> {name_b}: paired n={len(common)} "
        f"(of {len(a)}/{len(b)} total)  diff {diff.mean():+.2f}"
        f"  paired SE {paired_se:.2f}  unpaired SE {independent:.2f}"
        f"  win {int((diff > 0).sum())} lose {int((diff < 0).sum())} tie {int((diff == 0).sum())}"
    )
    sd_diff = diff.std(ddof=1)
    print(
        f"  {'':24s} sd_diff {sd_diff:.2f} vs independent "
        f"{np.sqrt(da.var(ddof=1) + db.var(ddof=1)):.2f}"
        f"  -> pairing is {'informative' if sd_diff < 0.9 * np.sqrt(da.var(ddof=1) + db.var(ddof=1)) else 'not informative'}"
    )


def report() -> None:
    arms = {name: load(glob) for name, (_, glob) in ARMS.items()}
    print("\n=== arms (all 2x10, score out of 20) ===")
    for name, (label, _) in ARMS.items():
        describe(label, arms[name])

    print("\n=== tonight's pair: does the 10-frame arm move when its window is fixed? ===")
    compare(arms["v9s"], arms["v9_step"], "v9 per-chunk", "v9 per-step")

    print("\n=== tonight's pair: does the 200-frame arm move when its window is fixed? ===")
    compare(arms["v9w200"], arms["w200"], "w200 per-chunk", "w200 per-step")

    print("\n=== the question that started this: is 200 frames worse than 10? ===")
    compare(arms["v9_step"], arms["w200"], "v9 per-step", "w200 per-step")
    compare(arms["v9s"], arms["v9w200"], "v9 per-chunk", "w200 per-chunk")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=float, default=0.0, help="reprint every N seconds")
    args = parser.parse_args()
    while True:
        print(f"\n{'=' * 78}\n  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report()
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
