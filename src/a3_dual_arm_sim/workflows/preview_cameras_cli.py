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
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

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
    parser.add_argument("--task", choices=("2x5", "2x10"))
    parser.add_argument("--profile", choices=("basic", "medium", "advanced"))
    parser.add_argument("--randomization-config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/camera_preview"))
    parser.add_argument("--show", action="store_true", help="Open the labeled camera preview")
    args = parser.parse_args()

    if args.task or args.profile or args.randomization_config:
        if args.scene != "cookie_transfer":
            parser.error("--task, --profile and --randomization-config require cookie_transfer")
        task = args.task or "2x5"
        first_grasp = None
        if task == "2x10":
            settings, first_grasp = load_cookie_2x10_settings(args.randomization_config)
        else:
            settings = load_randomization_config(args.randomization_config)
        settings["profile"] = args.profile or settings["profile"]
        benchmark = make_cookie_benchmark(
            task,
            config_path=args.config,
            profile=settings["profile"],
            first_grasp=first_grasp,
            randomization_settings=settings,
        )
        env = benchmark.create_env(render_cameras=True)
        options = benchmark.reset_options(args.seed)
    elif args.scene == "cookie_transfer":
        env = A3CookieTransferEnv(config=args.config, render_cameras=True)
        options = {"randomize_cookies": False}
    else:
        env = A3DualArmEnv(config=args.config, render_cameras=True)
        options = None
    try:
        observation, info = env.reset(seed=args.seed, options=options)
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

        result = {
            "scene": args.scene,
            "seed": args.seed,
            "order": [name for name, _ in CAMERAS],
            "shape": list(frames[0].shape),
            "files": files,
        }
        if "scenario_metrics" in info:
            result["scene_configuration"] = env.randomization_metadata["scene"]
            result["required_cookies"] = info["required_cookies"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.show:
            plt.show()
        else:
            plt.close(figure)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
