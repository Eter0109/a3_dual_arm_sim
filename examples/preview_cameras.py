#!/usr/bin/env python3
"""Save the three policy-camera views and a side-by-side preview image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from a3_dual_arm_sim import A3CookieTransferEnv, A3DualArmEnv
from a3_dual_arm_sim.contracts import (
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
)

CAMERAS = (
    ("front", FRONT_IMAGE),
    ("left_wrist", LEFT_WRIST_IMAGE),
    ("right_wrist", RIGHT_WRIST_IMAGE),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=("sandbox", "cookie_transfer"), default="cookie_transfer"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/camera_preview"))
    parser.add_argument("--show", action="store_true", help="Open the labeled camera preview")
    args = parser.parse_args()

    env = (
        A3CookieTransferEnv(config=args.config, render_cameras=True)
        if args.scene == "cookie_transfer"
        else A3DualArmEnv(config=args.config, render_cameras=True)
    )
    try:
        options = {"randomize_cookies": False} if args.scene == "cookie_transfer" else None
        observation, _ = env.reset(seed=args.seed, options=options)
        frames = []
        files = {}
        args.output.mkdir(parents=True, exist_ok=True)
        for name, key in CAMERAS:
            frame = observation[key]
            destination = args.output / f"{name}.png"
            Image.fromarray(frame, mode="RGB").save(destination)
            frames.append(frame)
            files[name] = str(destination.resolve())

        overview_path = args.output / "all_cameras.png"
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for axis, frame, (name, _) in zip(axes, frames, CAMERAS, strict=True):
            axis.imshow(frame)
            axis.set_title(name)
            axis.axis("off")
        figure.savefig(overview_path, dpi=120)
        files["all_cameras"] = str(overview_path.resolve())

        print(
            json.dumps(
                {
                    "scene": args.scene,
                    "seed": args.seed,
                    "order": [name for name, _ in CAMERAS],
                    "shape": list(frames[0].shape),
                    "files": files,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.show:
            plt.show()
        else:
            plt.close(figure)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
