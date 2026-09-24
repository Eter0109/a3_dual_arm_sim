#!/usr/bin/env python3
"""Compare fresh SmolVLA predictions with recorded expert actions on saved frames.

This is a teacher-forced action check, not a closed-loop success metric. The
policy is reset before each selected frame so its action queue cannot hide an
error in the current observation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.smolvla_policy import SmolVLAPolicyPlugin


CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/a3-single-box-same-column-100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 50, 100, 200, 300, 400])
    args = parser.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, download_videos=False)
    policy = SmolVLAPolicyPlugin(args.checkpoint, args.dataset_root, args.repo_id, args.device)
    errors: list[float] = []
    try:
        print("index  left_joint_mae_rad  left_gripper_abs  right_joint_mae_rad  right_gripper_abs")
        for index in args.indices:
            if index < 0 or index >= len(dataset):
                raise ValueError(f"Frame index {index} is outside dataset length {len(dataset)}")
            row = dataset[index]
            observation = {"observation.state": row["observation.state"].numpy()}
            for key in CAMERA_KEYS:
                image = row[key]
                observation[key] = np.ascontiguousarray(
                    (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                )
            task = str(row["task"])
            policy.reset(EpisodeContext(seed=0, task=task, action_mode="joint_position"))
            torch.manual_seed(args.inference_seed)
            predicted = policy.act(observation, task)
            expected = row["action"].numpy()
            error = np.abs(predicted - expected)
            errors.append(float(error[:7].mean()))
            print(
                f"{index:5d}  {error[:7].mean():18.4f}  {error[7]:16.4f}  "
                f"{error[8:15].mean():19.4f}  {error[15]:17.4f}"
            )
        print(f"mean left joint MAE over selected frames: {np.mean(errors):.4f} rad")
        print("This measures one-step imitation only; also run the camera-on closed-loop benchmark.")
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
