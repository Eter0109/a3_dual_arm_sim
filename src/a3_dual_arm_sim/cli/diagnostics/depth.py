"""Save aligned RGB and metric depth diagnostics without stepping the robot.

The preview ranges are fixed across frames: front 0.5--2.5 m and wrists
0.02--0.5 m. This does not enable depth input for Qwen or LeRobot policies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    from PIL import Image

    from a3_dual_arm_sim.core.paths import project_root
    from a3_dual_arm_sim.envs.config import load_config
    from a3_dual_arm_sim.envs.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
    from a3_dual_arm_sim.envs.depth_images import depth_statistics, depth_to_grayscale

    parser = argparse.ArgumentParser(prog="a3-sim inspect depth", description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new diagnostics directory")
    parser.add_argument("--config", type=Path,
                        default=project_root() / "configs/envs/cookie_same_column.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--front-near-m", type=float, default=0.5)
    parser.add_argument("--front-far-m", type=float, default=2.5)
    parser.add_argument("--wrist-near-m", type=float, default=0.02)
    parser.add_argument("--wrist-far-m", type=float, default=0.5)
    args = parser.parse_args(argv)
    ranges = {"front": (args.front_near_m, args.front_far_m),
              "left_wrist": (args.wrist_near_m, args.wrist_far_m),
              "right_wrist": (args.wrist_near_m, args.wrist_far_m)}
    for near, far in ranges.values():
        if not np.isfinite(near) or not np.isfinite(far) or near < 0 or far <= near:
            parser.error("depth ranges must satisfy finite 0 <= near < far")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new diagnostics directory; refusing overwrite: {output}")
    env = A3CookieTransferEnv(
        load_config(args.config),
        task_config=CookieTransferTaskConfig(
            require_exact_slots=False, require_released=True, terminate_on_success=False,
        ),
        render_cameras=True,
    )
    try:
        observation, _ = env.reset(
            seed=args.seed, options={"randomize_cookies": False, "randomize_boxes": False},
        )
        initial_time = float(env.data.time)
        output.mkdir(parents=True)
        metadata = {
            "schema_version": 1, "seed": args.seed, "config": str(args.config.resolve()),
            "unit": "meter", "depth_definition": "optical-axis camera depth, not Euclidean ray distance",
            "alignment": "RGB and depth use the same camera, resolution and simulation state",
            "simulation_time": initial_time, "simulation_steps_performed": 0,
            "preview_mapping": "fixed linear distance: near white, far black; invalid black",
            "valid_depth": "finite nonnegative values; explicit valid_mask PNG",
            "policy_depth_enabled": False, "action_dataset_recorded": False,
            "cameras": {},
        }
        for camera, (near, far) in ranges.items():
            rgb = np.asarray(observation[f"observation.images.{camera}"])
            depth = env.render_depth(camera)
            if depth.shape != rgb.shape[:2]:
                raise RuntimeError(f"RGB/depth shape mismatch for camera {camera}")
            preview, valid = depth_to_grayscale(depth, near_m=near, far_m=far)
            files = {"rgb": f"{camera}_rgb.png", "raw_depth": f"{camera}_depth_m.npy",
                     "depth_preview": f"{camera}_depth_preview.png",
                     "valid_mask": f"{camera}_depth_valid_mask.png"}
            Image.fromarray(rgb).save(output / files["rgb"])
            np.save(output / files["raw_depth"], depth, allow_pickle=False)
            Image.fromarray(preview).save(output / files["depth_preview"])
            Image.fromarray(valid.astype(np.uint8) * 255).save(output / files["valid_mask"])
            metadata["cameras"][camera] = {
                "files": files, "preview_near_m": near, "preview_far_m": far,
                **depth_statistics(depth),
            }
        if float(env.data.time) != initial_time:
            raise RuntimeError("depth diagnostics unexpectedly advanced simulation time")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), **metadata}, indent=2), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
