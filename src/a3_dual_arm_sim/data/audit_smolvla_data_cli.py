"""Read-only dataset checks with an explicit JSON output report."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from safetensors.numpy import load_file

from a3_dual_arm_sim.data.audit import audit_training_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repo-id", default="Eter0109/a3-front-close-left-100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = audit_training_dataset(args.root, repo_id=args.repo_id)
    info = json.loads((args.root / "meta/info.json").read_text())
    fps = info["fps"]
    samples = {}
    frames = 0
    for path in sorted((args.root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            frames += 1
            for key in ("action", "observation.state"):
                if len(row[key]) != contract["action_dim"] or not np.isfinite(row[key]).all():
                    raise ValueError(f"Invalid {key} in episode {row['episode_index']}")
            ep = row["episode_index"]
            if ep not in samples:
                samples[ep] = {"frames": 0, "images": {}}
                for camera in ("front", "left_wrist", "right_wrist"):
                    encoded = row["observation.images." + camera]
                    image = (
                        Image.open(io.BytesIO(encoded["bytes"]))
                        if encoded["bytes"]
                        else Image.open(args.root / encoded["path"])
                    )
                    a = np.asarray(image)
                    if a.shape != (256, 256, 3) or a.dtype != np.uint8:
                        raise ValueError(f"Invalid camera {camera} in episode {ep}")
                    samples[ep]["images"][camera] = {
                        "min": int(a.min()),
                        "max": int(a.max()),
                        "std": float(a.std()),
                    }
            if row["frame_index"] != samples[ep]["frames"]:
                raise ValueError(f"Non-contiguous frames in episode {ep}")
            if abs(row["timestamp"] - row["frame_index"] / fps) >= 1e-4:
                raise ValueError(f"Timestamp mismatch in episode {ep}")
            samples[ep]["frames"] += 1
    stats = json.loads((args.root / "meta/stats.json").read_text())
    matches = {}
    for key in ("action", "observation.state"):
        for stat in ("mean", "std"):
            values = np.asarray(stats[key][stat])
            if values.shape != (contract["action_dim"],) or not np.isfinite(values).all():
                raise ValueError(f"Invalid statistics: {key}.{stat}")
    if frames != contract["frames"] or len(samples) != contract["episodes"]:
        raise ValueError("Parquet frame/episode counts differ from metadata")
    for path in args.checkpoint.glob("*processor*.safetensors") if args.checkpoint else []:
        values = load_file(path)
        matches[path.name] = all(
            np.allclose(values[f"{key}.{stat}"], stats[key][stat])
            for key in ("action", "observation.state")
            for stat in ("mean", "std")
        )
    if matches and not all(matches.values()):
        raise ValueError("Checkpoint processor statistics differ from dataset")
    report = {
        "contract": contract,
        "fps": fps,
        "frames": frames,
        "episodes": len(samples),
        "sampled_images": samples,
        "stats_match": matches,
        "image_storage": "parquet embedded images",
        "limits": "Image decode sampled once per episode; action-before alignment requires replay audit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sampled_images"}, indent=2))


if __name__ == "__main__":
    main()
