#!/usr/bin/env python3
"""Put a grasp that worked next to a close that caught nothing, frame for frame.

The question is not "does the arm look different" -- it does, obviously -- but *what the
model was looking at* when it decided to close.  So each panel keeps the same three views
the policy is fed, in the same layout, at four moments around one event:

    approach  a few steps before the jaws shut
    close     the step the gripper channel crosses its threshold
    after     once the hand has moved away, which is what separates "lifted" from "closed
              on air"

`probe_grasp_events.py` supplies the events and the classification; this only renders them.

Usage:

    python scripts/make_grasp_figure.py --seed 0 --step 426 --tag good
    python scripts/make_grasp_figure.py --seed 1 --step 288 --tag air
    python scripts/make_grasp_figure.py --tasks 2x10 --auto
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

DIAG = Path("/root/autodl-tmp/diag/v9_run")

#: The three views the checkpoint is conditioned on, and their per-episode videos.
VIEWS = (
    ("front", "front.mp4"),
    ("left_wrist", "left_wrist.mp4"),
    ("right_wrist", "right_wrist.mp4"),
)

#: Steps before and after the close to sample, so the sequence reads as an approach, the
#: decision, and its consequence.
BEFORE = 40
AFTER = 60

#: The gripper channel on the applied action that `probe_grasp_events` treats as "shut".
LEFT_GRIPPER = 7


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_grasp_events", Path(__file__).resolve().parent / "probe_grasp_events.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"{path}: no frame at {index}")
    return frame


def annotate(frame: np.ndarray, label: str, sub: str, colour) -> np.ndarray:
    out = frame.copy()
    if out.shape[2] == 3:
        out = cv2.copyMakeBorder(out, 26, 6, 6, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    else:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        out = cv2.copyMakeBorder(out, 26, 6, 6, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(out, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    cv2.putText(out, sub, (150, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                cv2.LINE_AA)
    return out


def build_panel(task: str, seed: int, step: int, kind: str) -> list[np.ndarray]:
    """Three views at three moments around one close, as rows of the figure."""

    root = DIAG / task / f"seed_{seed}"
    trace = [json.loads(line) for line in (root / "trace.jsonl").open()]

    grip = np.array([
        float(r["applied_action"][LEFT_GRIPPER]) if r.get("applied_action") else np.nan
        for r in trace
    ])
    moments = [
        ("approach", max(0, step - BEFORE)),
        ("CLOSE", step),
        ("after", min(len(trace) - 1, step + AFTER)),
    ]
    colours = {"grab_lift": (80, 220, 80), "air_close": (60, 60, 240),
               "missed": (60, 200, 240), "grab_no_lift": (240, 180, 60)}
    colour = colours.get(kind, (220, 220, 220))

    grid = []
    for label, index in moments:
        row = []
        for view, filename in VIEWS:
            frame = read_frame(root / filename, index)
            eef = np.asarray(trace[index]["observation_before"]["observation.eef_pose"][:3])
            sub = f"f{index}  g={grip[index]:.2f}  z={eef[2]:.3f}"
            row.append(annotate(frame, f"{view}  {label}", sub, colour))
        grid.append(np.hstack(row))
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/diag/grasp_figure"))
    args = parser.parse_args()

    probe = load_probe()
    args.out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        path = DIAG / args.task / f"seed_{seed}" / "trace.jsonl"
        report = probe.probe(path)
        print(f"\n=== {args.task} seed {seed}: {report['counts']} ===")

        # One image per event, named by its class so the pair can be eyeballed side by side.
        for event in report["events"]:
            step, kind = event["step"], event["kind"]
            image = np.vstack(build_panel(args.task, seed, step, kind))
            name = (f"{args.task}_seed{seed}_f{step}_{kind}"
                    f"{('_' + args.tag) if args.tag else ''}.png")
            cv2.imwrite(str(args.out / name), image)
            print(f"  {name}  gap={event.get('height_gap_mm')}mm held={event['held']} "
                  f"lifted={event['lifted']}")

    print(f"\nwrote to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
