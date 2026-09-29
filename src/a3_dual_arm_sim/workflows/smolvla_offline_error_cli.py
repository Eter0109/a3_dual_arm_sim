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
    parser.add_argument("--policy-type", choices=("smolvla", "act"), default="smolvla")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repo-id", default="Eter0109/a3-front-close-left-100")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    from a3_dual_arm_sim.policies.act import ACTPolicyPlugin
    from a3_dual_arm_sim.workflows.policy_adapters import resolve_policy_checkpoint

    plugin_class = ACTPolicyPlugin if args.policy_type == "act" else SmolVLAPolicyPlugin
    policy = plugin_class(
        resolve_policy_checkpoint(args.checkpoint),
        args.root,
        args.repo_id,
        args.device,
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
                    if len(records) >= args.max_samples:
                        break
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
                    pred = policy.predict_action(obs, "transfer 10 cookies into target box")
                    target = np.asarray(row["action"])
                    error = pred[: len(target)] - target
                    records.append(
                        {
                            "episode": ep,
                            "frame": row["frame_index"],
                            "left_joint_mae_rad": float(np.abs(error[:7]).mean()),
                            "left_gripper_abs_error": float(abs(error[7])),
                            "per_dimension_error": error.tolist(),
                        }
                    )
            print("sampled", len(records), flush=True)
            if len(records) >= args.max_samples:
                break
    finally:
        policy.close()
    if not records:
        raise ValueError("No dataset frames sampled")
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "policy_type": args.policy_type,
        "scope": "in-sample teacher-forced first action",
        "left_joint_mae_rad": float(np.mean([r["left_joint_mae_rad"] for r in records])),
        "left_gripper_abs_error": float(np.mean([r["left_gripper_abs_error"] for r in records])),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "samples"}, flush=True)


if __name__ == "__main__":
    main()
