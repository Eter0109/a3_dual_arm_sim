#!/usr/bin/env python3
"""How many cookies does one grasp actually take?  Read from cookie geometry, not from counts.

Two earlier attempts disagreed on where the 2x10 loss comes from, and the disagreement was a
measurement artefact rather than a modelling one:

  * `80 - cookies_in_source` minus `cookies_in_target` implied 21 % of carried cookies were lost
    to whole failed batches;
  * tracking each cookie's own xy position showed only 12 cookies (of 129 moved, 8 episodes)
    ever ended outside both bins -- **9 %**, one or two per episode.

The count-based number was wrong because `cookies_in_source` is "how many are still in the
source box", which changes when a cookie is nudged across the box's internal edge as well as
when one is lifted out.  Geometry cannot be fooled that way, so everything here is computed from
`after.cookie_positions`.

The question this answers: the arm averages ~16 transfers per episode against a target of 20.
Is that because it *takes fewer cookies per grasp*, or because it *makes fewer attempts*?  Those
have opposite fixes -- grasp quality vs. grasp scheduling -- so the mean per attempt is printed
alongside the full distribution.

    python scripts/probe_grasp_batch_size.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

#: Box half-extents in xy, from `configs/cookie_2x10.yaml`.
TARGET_HALF = (0.063, 0.035)
SOURCE_HALF = (0.1186, 0.1010833333)

#: A grasp is "the source emptied between two frames"; several cookies can leave in consecutive
#: frames, so runs closer together than this are one grasp.
MERGE_FRAMES = 60


def zone(pxy: np.ndarray, tb: np.ndarray, sb: np.ndarray) -> str:
    if abs(pxy[0] - tb[0]) < TARGET_HALF[0] and abs(pxy[1] - tb[1]) < TARGET_HALF[1]:
        return "target"
    if abs(pxy[0] - sb[0]) < SOURCE_HALF[0] and abs(pxy[1] - sb[1]) < SOURCE_HALF[1]:
        return "source"
    return "table"


def analyse(ep_dir: Path) -> dict | None:
    trace = ep_dir / "trace.jsonl"
    result = ep_dir / "result.json"
    if not trace.is_file() or not result.is_file():
        return None
    meta = json.loads(result.read_text())
    tb = np.asarray(meta["target_bin_pos"], float)
    sb = np.asarray(meta["source_bin_pos"], float)

    pos = []
    with trace.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            pos.append(json.loads(line)["after"]["cookie_positions"])
    pos = np.asarray(pos, float)          # (T, 80, 3)

    first = pos[0]
    source_idx = [i for i, p in enumerate(first) if zone(p[:2], tb, sb) == "source"]
    if not source_idx:
        return None

    # For each frame, how many of the initially-source cookies are still in the source box.
    still = np.zeros(len(pos), int)
    for t in range(len(pos)):
        still[t] = sum(1 for i in source_idx if zone(pos[t, i, :2], tb, sb) == "source")

    # A grasp begins where that count drops; merge consecutive drops into one grasp.
    drops = np.flatnonzero(np.diff(still) < 0) + 1
    if not len(drops):
        return {"episode": ep_dir.parent.name, "grasps": 0, "per_grasp": [],
                "final_target": int(sum(1 for t in [len(pos) - 1]
                                        for i in source_idx
                                        if zone(pos[t, i, :2], tb, sb) == "target"))}
    groups, current = [], [int(drops[0])]
    for d in drops[1:]:
        if d - current[-1] <= MERGE_FRAMES:
            current.append(int(d))
        else:
            groups.append(current)
            current = [int(d)]
    groups.append(current)

    per_grasp = []
    for g in groups:
        start, end = g[0], g[-1] + MERGE_FRAMES
        end = min(end, len(pos) - 1)
        taken = int(still[g[0] - 1] - still[end])
        per_grasp.append(max(0, taken))

    final_t = int(sum(1 for i in source_idx
                      if zone(pos[-1, i, :2], tb, sb) == "target"))
    return {
        "episode": ep_dir.parent.name + "/" + ep_dir.name,
        "grasps": len(groups),
        "per_grasp": per_grasp,
        "moved": int(still[0] - still[-1]),
        "final_target": final_t,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diag", type=Path,
                        default=Path("/root/autodl-tmp/diag/v9_run/2x10"))
    args = parser.parse_args()

    eps = [Path(p) for p in sorted(glob.glob(str(args.diag / "seed_*")))]
    rows = [r for r in (analyse(e) for e in eps) if r]
    if not rows:
        print(f"no traces under {args.diag}")
        return 1

    print(f"{'episode':>18} {'抓取次数':>8} {'搬走':>5} {'落入':>5} {'每次抓取块数':>22}")
    print("-" * 66)
    all_sizes = []
    for r in rows:
        sizes = r["per_grasp"]
        all_sizes.extend(sizes)
        shown = " ".join(str(s) for s in sizes[:10])
        print(f"{r['episode']:>18} {r['grasps']:8d} {r['moved']:5d} "
              f"{r['final_target']:5d}   {shown}")

    sizes = np.array(all_sizes)
    print()
    print(f"=== {len(rows)} 个回合，{len(sizes)} 次抓取 ===")
    print(f"  每次抓取块数：mean {sizes.mean():.2f}  中位 {np.median(sizes):.0f}  "
          f"max {sizes.max()}")
    hist = {k: int((sizes == k).sum()) for k in range(0, 8)}
    for k, v in hist.items():
        if v:
            print(f"    {k} 块: {v:3d} 次 ({v/len(sizes)*100:5.1f}%)  {'#' * int(v/len(sizes)*50)}")
    print()
    # The two hypotheses make different predictions about this ratio.
    print(f"  每回合抓取次数：mean {np.mean([r['grasps'] for r in rows]):.1f}")
    print(f"  每回合搬走：    mean {np.mean([r['moved'] for r in rows]):.1f}  (目标 20)")
    full = int((sizes >= 5).sum())
    print(f"  抓满 5 块的比例：{full}/{len(sizes)} = {full/len(sizes)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
