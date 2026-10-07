"""Read the ablation logs and say which of the fixes the 31/40 actually depends on.

Leave-one-out: the working configuration scores 31/40, and each case reverts exactly one of
the fixes that landed on the way there.  What the table answers is *attribution* -- "the
result is the sum of a long list of small alignments" and "the result is one bug that was
hiding the others" are both stories, and they make different predictions about what happens
when a single fix is taken away.

Reports, per case: placements out of the seeds tried, and the closest the fingertip got to
the bowl's axis.  The second number matters because a run can fail at every stage and a run
can fail only at the last one, and they mean different things.
"""

from __future__ import annotations

import re
from pathlib import Path

LOGS = Path("/root/autodl-tmp/logs/ablation")

#: name -> what was reverted
CASES = [
    ("baseline", "nothing (control)"),
    ("flip_rot180", "image orientation: the 180-degree error"),
    ("no_mirror", "image mirror, both sides of it"),
    ("no_sites", "the four rendered sites"),
    ("osc_once", "500 Hz torque loop -> once per control step"),
    ("heavy_bowl", "object mass: 45.9 g bowl"),
    ("home_only", "LIBERO's per-episode start poses"),
    ("cookie_phys", "the Cookie task's contact model / multiccd"),
    ("grip_legacy", "the gripper's polarity and frozen outer jaw"),
]


def read(name: str) -> dict:
    path = LOGS / f"{name}.log"
    if not path.is_file():
        return {"missing": True}
    text = path.read_text()
    seeds = len(re.findall(r"^seed ", text, re.M))
    rows = re.findall(
        r"nearest fingertip ([\d.]+) mm from the bowl centre at step \d+; "
        r"([\d.]+) mm off its axis",
        text,
    )
    heights = [float(m) for m in re.findall(r"height ([+-][\d.]+) peak", text)]
    best = [float(x) for x in re.findall(r"\(best ([\d.]+)\)", text)]
    return {
        "missing": False,
        "seeds": seeds,
        "placed": text.count("->  PLACED"),
        "axis_min": min((float(r[1]) for r in rows), default=float("nan")),
        "lifted": sum(1 for h in heights if h > 50),
        "on_plate": sum(1 for b in best if b < 74),
        "text": text,
    }


def main() -> int:
    print(f"{'case':<14}{'reverted':<44}{'placed':>9}{'lift>50':>9}{'on plate':>10}{'axis min':>10}")
    results = {}
    for name, what in CASES:
        r = read(name)
        results[name] = r
        if r.get("missing"):
            print(f"{name:<14}{what:<44}{'--':>9}{'--':>9}{'--':>10}{'--':>10}")
            continue
        n = r["seeds"] or 1
        print(
            f"{name:<14}{what:<44}"
            f"{r['placed']:>4}/{n:<4}{r['lifted']:>4}/{n:<4}"
            f"{r['on_plate']:>5}/{n:<4}{r['axis_min']:>9.1f}mm"
        )

    base = results.get("baseline", {})
    if not base.get("missing") and base.get("placed"):
        print(f"\n相对基线（{base['placed']}/{base['seeds']}）的落差:")
        scaled = []
        for name, what in CASES:
            r = results[name]
            if r.get("missing") or name == "baseline":
                continue
            scaled.append((r["placed"] / (r["seeds"] or 1), name, what, r["placed"], r["seeds"]))
        for rate, name, what, placed, n in sorted(scaled):
            drop = base["placed"] / base["seeds"] - rate
            verdict = (
                "关键（单独移除即崩塌）" if rate <= 0.10
                else "部分贡献" if drop >= 0.20
                else "当前无影响"
            )
            print(f"  {name:<14} {placed:>2}/{n}   掉 {drop * 100:5.1f} 个点   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
