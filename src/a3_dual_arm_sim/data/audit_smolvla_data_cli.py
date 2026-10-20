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

CAMERAS = ("front", "left_wrist", "right_wrist")


def sample_video_frames(root: Path, repo_id: str, episodes: int) -> dict[int, dict]:
    """Decode one sampled frame per episode for video-typed camera features.

    LeRobot v3 stores video frames in ``videos/``; the data parquet has no camera
    columns, so the inline-image scan cannot see them. Decode the first frame of
    each episode through the dataset API instead.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=root, return_uint8=True)
    images: dict[int, dict] = {}
    for episode in range(episodes):
        start = int(dataset.meta.episodes["dataset_from_index"][episode])
        frame = dataset[start]
        entry = {}
        for camera in CAMERAS:
            image = np.asarray(frame["observation.images." + camera])
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                if np.issubdtype(image.dtype, np.floating):
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported camera dtype {image.dtype}")
            if image.shape != (256, 256, 3):
                raise ValueError(f"Invalid camera {camera} in episode {episode}")
            entry[camera] = {
                "min": int(image.min()),
                "max": int(image.max()),
                "std": float(image.std()),
            }
        images[episode] = entry
    return images


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
    video_cameras = [
        key
        for key in (f"observation.images.{camera}" for camera in CAMERAS)
        if info.get("features", {}).get(key, {}).get("dtype") == "video"
    ]
    samples = {}
    frames = 0
    for path in sorted((args.root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            frames += 1
            for key in ("action", "observation.state"):
                if len(row[key]) != contract["state_dim" if key == "observation.state" else "action_dim"] or not np.isfinite(row[key]).all():
                    raise ValueError(f"Invalid {key} in episode {row['episode_index']}")
            ep = row["episode_index"]
            if ep not in samples:
                samples[ep] = {"frames": 0, "images": {}}
                if not video_cameras:
                    for camera in CAMERAS:
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
    if video_cameras:
        for episode, images in sample_video_frames(
            args.root, args.repo_id, contract["episodes"]
        ).items():
            if episode in samples:
                samples[episode]["images"] = images
    stats = json.loads((args.root / "meta/stats.json").read_text())
    matches = {}
    for key in ("action", "observation.state"):
        for stat in ("mean", "std"):
            values = np.asarray(stats[key][stat])
            if values.shape != (contract["state_dim" if key == "observation.state" else "action_dim"],) or not np.isfinite(values).all():
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
        "image_storage": (
            "video files (frames decoded via the dataset API)"
            if video_cameras
            else "parquet embedded images"
        ),
        "limits": "Image decode sampled once per episode; action-before alignment requires replay audit.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sampled_images"}, indent=2))


if __name__ == "__main__":
    main()
