"""Read-only dataset checks with an explicit JSON output report."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = {}
    frames = 0
    for path in sorted((args.root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            frames += 1
            for key in ("action", "observation.state"):
                assert len(row[key]) == 16 and np.isfinite(row[key]).all()
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
                    assert a.shape == (256, 256, 3) and a.dtype == np.uint8
                    samples[ep]["images"][camera] = {
                        "min": int(a.min()),
                        "max": int(a.max()),
                        "std": float(a.std()),
                    }
            assert row["frame_index"] == samples[ep]["frames"]
            assert abs(row["timestamp"] - row["frame_index"] / 20) < 1e-4
            samples[ep]["frames"] += 1
    stats = json.loads((args.root / "meta/stats.json").read_text())
    matches = {}
    for path in args.checkpoint.glob("*processor*.safetensors"):
        values = load_file(path)
        matches[path.name] = all(
            np.allclose(values[f"{key}.{stat}"], stats[key][stat])
            for key in ("action", "observation.state")
            for stat in ("mean", "std")
        )
    report = {
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
