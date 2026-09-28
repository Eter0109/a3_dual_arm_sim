"""Teacher-forced first-action error on sampled training observations, not a validation score."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.policies.smolvla import SmolVLAPolicyPlugin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = SmolVLAPolicyPlugin(
        args.checkpoint,
        args.root,
        "local/a3-front-close-left-100",
        "cuda",
        ema_alpha=0,
        anchor_right_arm=False,
        gripper_sharpening=False,
        n_action_steps=1,
        inference_seed=123,
    )
    records = []
    try:
        for path in sorted((args.root / "data").rglob("*.parquet")):
            rows = pq.read_table(path).to_pylist()
            for ep in sorted({r["episode_index"] for r in rows}):
                episode = [r for r in rows if r["episode_index"] == ep]
                for row in [episode[i] for i in np.linspace(0, len(episode) - 1, 5, dtype=int)]:
                    obs = {
                        "observation.state": np.array(row["observation.state"], dtype=np.float32)
                    }
                    for camera in ("front", "left_wrist", "right_wrist"):
                        key = "observation.images." + camera
                        image = row[key]
                        source = (
                            io.BytesIO(image["bytes"])
                            if image["bytes"]
                            else args.root / image["path"]
                        )
                        obs[key] = np.array(Image.open(source).convert("RGB"))
                    policy.reset(
                        EpisodeContext(
                            seed=123,
                            task="transfer 10 cookies into target box",
                            action_mode="joint_position",
                        )
                    )
                    pred = policy.act(obs, "transfer 10 cookies into target box")
                    error = pred - np.asarray(row["action"])
                    records.append(
                        dict(
                            episode=ep,
                            frame=row["frame_index"],
                            left_joint_mae_rad=float(np.abs(error[:7]).mean()),
                            left_gripper_abs_error=float(abs(error[7])),
                            per_dimension_error=error.tolist(),
                        )
                    )
            print("sampled", len(records), flush=True)
    finally:
        policy.close()
    report = dict(
        checkpoint=str(args.checkpoint.resolve()),
        scope="in-sample teacher-forced first action",
        left_joint_mae_rad=float(np.mean([r["left_joint_mae_rad"] for r in records])),
        left_gripper_abs_error=float(np.mean([r["left_gripper_abs_error"] for r in records])),
        samples=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "samples"}, flush=True)


if __name__ == "__main__":
    main()
