#!/usr/bin/env python3
"""Render two named moments of one episode side by side, from all three cameras.

Built because the interesting comparison is not "success vs failure" in the abstract but
**two specific descents of the same episode**: one that carried its five into the target and
one that took five out of the source and delivered none.  Same seed, same scene, same
weights, so the only difference is what the arm did.

Each row is one moment, each column one camera, so the front view answers "where is the arm
and how high" while the wrist view answers "what is between the fingers".

Usage:

    python scripts/render_moment.py --task 2x5 --seed 0 \
        --steps 48 320 --labels delivered grabbed-lost --out /tmp/pair.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DIAG = Path("/root/autodl-tmp/diag/v9_run")
VIEWS = (("front", "front.mp4"), ("left_wrist", "left_wrist.mp4"),
         ("right_wrist", "right_wrist.mp4"))
LABEL_H = 26
DOT = "\u00b7"


def read_frames(video: Path, steps: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    wanted = sorted(set(steps))
    out: dict[int, np.ndarray] = {}
    index = 0
    target = 0
    while target < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index == wanted[target]:
            out[index] = frame
            target += 1
        index += 1
    capture.release()
    return out


def band(width: int, text: str, colour: tuple[int, int, int]) -> np.ndarray:
    strip = np.zeros((LABEL_H, width, 3), np.uint8)
    cv2.putText(strip, text, (4, LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                colour, 1, cv2.LINE_AA)
    return strip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="2x5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--off", nargs="+", type=int, default=None,
                        help="Per-moment step offset to also show, e.g. --off 40 0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = DIAG / args.task / f"seed_{args.seed}"
    labels = args.labels or [f"f{s}" for s in args.steps]
    camera_steps = list(args.steps)
    if args.off:
        camera_steps = [s + o for s, o in zip(args.steps, args.off)]

    frames = {name: read_frames(root / video, camera_steps) for name, video in VIEWS}

    rows = []
    for moment, (step, label) in enumerate(zip(camera_steps, labels)):
        cells = []
        for name, _video in VIEWS:
            frame = frames[name].get(step)
            if frame is None:
                frame = np.zeros((256, 256, 3), np.uint8)
            width = frame.shape[1]
            header = band(width, f"{name}  f{step}  {label}", (255, 255, 255))
            cells.append(np.vstack([header, frame]))
        rows.append(np.hstack(cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"wrote {args.out}  ({args.task} seed {args.seed}, steps {camera_steps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
