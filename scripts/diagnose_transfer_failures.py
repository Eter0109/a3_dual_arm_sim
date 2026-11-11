#!/usr/bin/env python3
"""Where do the carried cookies go?  Classify every failed transfer from the traces.

The score decomposition says the loss is not per-cookie placement precision: on 8 recorded 2x10
episodes the arm moved 138 cookies and delivered 109, and the two best episodes lost **zero**.
So the model can place cleanly; the losses are whole batches that never arrive.  That leaves two
mechanisms which look identical in the score but opposite in the fix:

  * **grasp miss** -- the gripper closed on air, the cookie stayed in the source box, and the
    source count dropped only because the effector entered the box region.  Fix: perception /
    approach accuracy.
  * **transport drop** -- the cookie was held and lost on the way, or released outside the
    target.  Fix: grip force / trajectory smoothness.

The trace separates them.  For every window in which the source count falls, this looks at the
finger state around that moment:

  * if the gripper never closes (state[7] and finger touch stay high), nothing was held
  * if it closes and *re-opens before arriving*, the cookie was dropped in transit
  * if it stays closed until the target and still nothing lands, the release missed the box

Output is a per-episode table plus the aggregate split, so the next experiment can target the
mechanism that actually dominates.

    python scripts/diagnose_transfer_failures.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

#: Finger channels, from `env.py::_observation`.  `state[7]` is the LEFT gripper opening as a
#: fraction of its travel (1.0 = open, 0.0 = closed); `state[15]` is the right one, which never
#: moves on this setup.  `force[12:14]` are the left finger's inner/outer touch sensors and
#: `force[16]` sums the left finger actuator forces.
LEFT_GRIP = 7
L_FINGER_TOUCH = slice(12, 14)
L_FINGER_ACT = 16

#: Target box half-extents used by the 2x10 arrangement (from `configs/cookie_2x10.yaml`).
TARGET_HALF = (0.063, 0.035)

CLOSED = 0.35     # below this the jaws are shut on something or on air
WINDOW_BEFORE = 30
WINDOW_AFTER = 200


def load(path: Path):
    steps, grip, touch, act, c_in, c_src, eef = [], [], [], [], [], [], []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ob = r["observation_before"]
            steps.append(r["step"])
            grip.append(ob["observation.state"][LEFT_GRIP])
            f = ob["observation.force"]
            touch.append(float(np.max(np.abs(f[L_FINGER_TOUCH]))))
            act.append(abs(float(f[L_FINGER_ACT])))
            after = r["after"]
            c_in.append(after["cookies_in_target"])
            c_src.append(after["cookies_in_source"])
            eef.append(ob["observation_eef"] if "observation_eef" in ob
                       else ob["observation.eef_pose"][:3])
    return {
        "step": np.array(steps),
        "grip": np.array(grip, float),
        "touch": np.array(touch, float),
        "act": np.array(act, float),
        "target": np.array(c_in, int),
        "source": np.array(c_src, int),
        "eef": np.array(eef, float),
    }


def classify(t: dict, i: int) -> tuple[str, dict]:
    """Label the transfer attempt that starts around frame `i`.

    `i` is a frame in which the source count fell.  The judgement is made only from the finger
    channels, because those are the only observation of "is anything being held".
    """
    end = min(i + WINDOW_AFTER, len(t["grip"]))
    grip = t["grip"][i:end]
    touch = t["touch"][i:end]
    delivered = int(t["target"][-1] - t["target"][max(i - 1, 0)]) if end == len(t["grip"]) \
        else int(t["target"][end - 1] - t["target"][max(i - 1, 0)])

    closed = grip < CLOSED
    if not closed.any():
        label = "grasp-miss"
        detail = f"jaws never closed (min grip {grip.min():.2f})"
    else:
        first = int(np.argmax(closed))
        reopened = (~closed[first:]).any()
        if reopened and delivered == 0:
            label = "transport-drop"
            detail = f"closed at +{first} then reopened, nothing delivered"
        elif delivered == 0:
            label = "release-miss"
            detail = f"stayed closed {closed.sum()} frames, nothing delivered"
        else:
            label = "ok"
            detail = f"delivered {delivered}"
    return label, {
        "delivered": delivered,
        "grip_min": float(grip.min()) if len(grip) else float("nan"),
        "grip_max": float(grip.max()) if len(grip) else float("nan"),
        "touch_max": float(touch.max()) if len(touch) else float("nan"),
        "act_max": float(t["act"][i:end].max()) if len(t["act"][i:end]) else float("nan"),
        "detail": detail,
    }


def analyse(path: Path, verbose: bool) -> dict:
    t = load(path)
    if not len(t["step"]):
        return {}
    # A source drop marks the moment the effector took something out of the box.  Consecutive
    # frames can each drop a count when a batch is grabbed, so only the first of a run counts.
    drops = []
    prev = t["source"][0]
    for i in range(1, len(t["source"])):
        if t["source"][i] < prev:
            if not drops or i - drops[-1] > WINDOW_AFTER:
                drops.append(i)
            prev = t["source"][i]
        else:
            prev = t["source"][i]

    counts = {"ok": 0, "grasp-miss": 0, "transport-drop": 0, "release-miss": 0}
    details = []
    for i in drops:
        label, info = classify(t, i)
        counts[label] += 1
        details.append((i, label, info))

    if verbose:
        seed = path.parent.name
        print(f"--- {seed}: 落入目标 {t['target'][-1]}, 搬运事件 {len(drops)} ---")
        for i, label, info in details:
            print(f"   帧{i:5d}  {label:15s} {info['detail']:42s} "
                  f"grip[{info['grip_min']:.2f},{info['grip_max']:.2f}] "
                  f"touch={info['touch_max']:.3f}")
    return {"counts": counts, "n": len(drops), "delivered": int(t["target"][-1])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diag", type=Path,
                        default=Path("/root/autodl-tmp/diag/v9_run/2x10"))
    parser.add_argument("--pattern", default="seed_*")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) / "trace.jsonl"
             for p in sorted(glob.glob(str(args.diag / args.pattern)))]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print(f"no traces under {args.diag}")
        return 1

    total = {"ok": 0, "grasp-miss": 0, "transport-drop": 0, "release-miss": 0}
    delivered = 0
    for path in paths:
        result = analyse(path, verbose=not args.quiet)
        if not result:
            continue
        for k, v in result["counts"].items():
            total[k] += v
        delivered += result["delivered"]

    n = sum(total.values())
    print()
    print(f"=== 汇总：{len(paths)} 个回合，{n} 次搬运事件，共落入 {delivered} 块 ===")
    for k in ("ok", "grasp-miss", "transport-drop", "release-miss"):
        share = total[k] / n * 100 if n else 0.0
        print(f"  {k:15s} {total[k]:3d}  {share:5.1f}%")
    failed = n - total["ok"]
    if failed:
        worst = max(("grasp-miss", "transport-drop", "release-miss"),
                    key=lambda k: total[k])
        print(f"\n  失败 {failed} 次，主要机制：{worst} "
              f"({total[worst]}/{failed} = {total[worst]/failed*100:.0f}% of failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
