#!/usr/bin/env python3
"""Cache frozen-encoder summary features for every frame of a dataset.

Why cache at all
----------------
The value model may not touch the encoder: the whole point of reusing v9 is that its 2B-parameter
PaliGemma already represents this scene, and fine-tuning it would both cost more than the box
allows and change the thing being measured.  So the only trainable part is a small head, and the
expensive input (SigLIP over three 224x224 views) is a constant per frame.  Computing it once
turns a multi-hour end-to-end training loop into a head that trains in minutes and can be
re-shaped and re-run freely.

What is cached, and why not the transformer output
-------------------------------------------------
Two summary vectors per frame:

* **per-camera image mean** of the SigLIP patch embeddings (`[3, 1152]`),
* **language mean** of the token embeddings (`[1152]`), masked so padding is excluded.

Mean-pooling is a compromise, and the reason it is acceptable here is that it keeps the cached
feature a *pure function of the frozen encoder*: no learned query is involved, so re-training the
head does not invalidate the cache.  A learned CLS or attention pool would have to be trained
inside the extraction pass, and every head experiment would then need a fresh 5 GB pass.  If the
Spearman gate fails, the next thing to try is caching patch features and learning the pool -- a
different cache, not a different idea.

Note that this deliberately stops *before* the PaliGemma transformer.  Running the 18-layer
prefix forward per frame would multiply the cost by roughly 20 and buy a representation the head
would still have to learn to read.  The image mean already carries "how many Cookies are in the
box and where", which is most of what the target depends on.

Output
------
`features.npy` (float16, `[frames, 3*1152 + 1152]`) plus `index.json`.  Row `i` is dataset row
`i`, in the dataset's natural order, so a consumer can index by position and never has to join.

Usage:
    python scripts/recap/extract_value_features.py \
        --dataset /root/autodl-tmp/datasets/a3_recap_all \
        --checkpoint /root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model \
        --output /root/autodl-tmp/datasets/a3_recap_all/features_v9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

COLLECT = Path("/root/autodl-tmp/a3-collect")
DEFAULT_STATS = "/root/autodl-tmp/datasets/a3_multitask_v6"


def load_policy(checkpoint: Path, stats_root: str, history_frames: int):
    """Load through the deployed adapter, so the encoder is the one inference uses.

    Not `PI05Policy.from_pretrained` directly: the adapter installs the launcher patches
    (`state_proj`, the history encoder) and the dataset statistics that the checkpoint was
    trained with.  A shortcut here would silently extract features from a *different* model than
    the one the rollouts came from, which is exactly the class of bug §9.76 and §9.81 were.
    """

    sys.path.insert(0, str(COLLECT))
    import importlib

    adapter = importlib.import_module("adapters.pi05_a3")
    # `cameras` is deliberately left at its default.  It only names which *manifest* slot each
    # declared camera reads at inference time, and this script reads frames from the dataset
    # directly -- the adapter's default already pairs the three A3 views with the three openpi
    # slots the checkpoint declares.
    return adapter.load(
        {
            "ckpt": str(checkpoint),
            "stats_root": stats_root,
            "state_dims": 16,
            "history_dims": history_frames * 48,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats-root", default=DEFAULT_STATS)
    parser.add_argument("--history-frames", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=0, help="0 = all; use 2 for a smoke test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "write every Nth frame.  Default 1 so the advantage step can use consecutive "
            "frames; the cache only costs disk, so raising this is the cheap way to trade "
            "label resolution for extraction time."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="row width; required for a shard, since the file is preallocated by the launcher",
    )
    args = parser.parse_args(argv)

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_policy(args.checkpoint, args.stats_root, args.history_frames)
    policy = model["policy"].to(args.device).eval()
    preprocess = model["preprocess"]
    palm = policy.model.paligemma_with_expert

    dataset = LeRobotDataset(
        repo_id=args.dataset.name,
        root=args.dataset,
        # pyav, because torchcodec cannot load its shared libraries against this container's
        # FFmpeg -- the same reason `train_pi05_a3.sh` passes `--dataset.video_backend=pyav`.
        video_backend="pyav",
    )
    total = dataset.meta.total_frames
    if args.episodes:
        # Episode boundaries come from the episodes table so a smoke test covers whole episodes
        # rather than an arbitrary prefix of the row space.
        import pyarrow.parquet as pq

        table = pq.read_table(args.dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        total = int(table.column("dataset_to_index").to_pylist()[args.episodes - 1])
    print(f"[extract] {total} of {dataset.meta.total_frames} frames, batch {args.batch_size}", flush=True)

    # Width is probed from the encoder rather than assumed.  `embed_image` returns the
    # **post-projection** 2048-wide embeddings (SigLIP's 1152 goes through
    # `multi_modal_projector`), and hard-coding 1152 produced an 8192-wide batch against a
    # 4608-wide memmap -- a shape error at the write, an hour into the design and one line
    # from the fix.
    if args.width:
        width = args.width
        image_dim = 0
    else:
        image_dim = int(palm.embed_image(torch.zeros(1, 3, 224, 224, device=args.device)).shape[-1])
        language_dim = int(
            palm.embed_language_tokens(torch.zeros(1, 2, dtype=torch.long, device=args.device)).shape[-1]
        )
        width = 3 * image_dim + language_dim
        print(f"[extract] image {image_dim}/camera, language {language_dim}, row {width}", flush=True)

    # Several workers share one file by writing disjoint rows.  `np.memmap` on a regular file is
    # safe for that as long as no two writers touch the same page, which the contiguous row
    # bands below guarantee.  The `written` mask is what lets the trainer tell rows a strided run
    # actually filled from the zeros left behind -- without it, an incomplete extraction trains
    # on zero features and looks like a model that cannot learn.
    first = not (args.output / "features.npy").is_file()
    features = np.lib.format.open_memmap(
        args.output / "features.npy",
        mode="w+" if first else "r+",
        dtype=np.float16,
        shape=(total, width),
    )
    written = np.lib.format.open_memmap(
        args.output / "written.npy",
        mode="w+" if first else "r+",
        dtype=bool,
        shape=(total,),
    )
    if first:
        features[:] = 0
        written[:] = False
        features.flush()
        written.flush()

    # A contiguous band per shard, strided inside it, so the rows one worker owns are contiguous
    # in the file and pages never interleave between writers.
    band_start = (total * args.shard_index) // args.shard_count
    band_stop = (total * (args.shard_index + 1)) // args.shard_count
    indices = list(range(band_start, band_stop, args.stride))
    print(f"[extract] shard {args.shard_index}/{args.shard_count}: rows {band_start}..{band_stop}"
          f" stride {args.stride} -> {len(indices)} frames", flush=True)

    camera_keys = [k for k in dataset.meta.features if k.startswith("observation.images")]
    camera_keys.sort(key=lambda k: ["front", "left_wrist", "right_wrist"].index(k.rsplit(".", 1)[-1]))
    print(f"[extract] cameras: {camera_keys}", flush=True)

    def to_batch(rows: list[dict]) -> dict:
        batch = {
            "observation.state": torch.stack([r["observation.state"] for r in rows]),
            "observation.history": torch.stack([r["observation.history"] for r in rows]),
            # `task` as a list of strings is what the preprocessor tokenises.
            "task": [r["task"] for r in rows],
        }
        for key in camera_keys:
            batch[key] = torch.stack([r[key] for r in rows])
        return batch

    started = time.monotonic()
    done = 0
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start : start + args.batch_size]
            rows = [dataset[i] for i in chunk]
            ready = preprocess(to_batch(rows))
            images, _ = policy._preprocess_images(ready)

            per_camera = []
            for image in images:
                embedding = palm.embed_image(image.to(args.device))
                per_camera.append(embedding.mean(dim=1).float())
            image_mean = torch.stack(per_camera, dim=1)  # [B, cameras, D]

            tokens = ready[OBS_LANGUAGE_TOKENS].to(args.device)
            mask = ready[OBS_LANGUAGE_ATTENTION_MASK].to(args.device).float()
            lang_embedding = palm.embed_language_tokens(tokens).float()
            weights = mask.unsqueeze(-1)
            language_mean = (lang_embedding * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

            block = torch.cat([image_mean.reshape(image_mean.shape[0], -1), language_mean], dim=-1)
            features[chunk] = block.cpu().numpy().astype(np.float16)
            written[chunk] = True

            done = min(start + args.batch_size, len(indices))
            if done % (args.batch_size * 20) < args.batch_size:
                rate = done / max(time.monotonic() - started, 1e-6)
                remaining = (len(indices) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(indices)} ({100 * done / len(indices):.1f} %)"
                      f"  {rate:.0f} frames/s  eta {remaining / 60:.0f} min", flush=True)

    features.flush()
    written.flush()
    # The manifest is written by shard 0 only; a shard that overwrote it would race the others.
    if args.shard_index == 0:
        (args.output / "index.json").write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "checkpoint": str(args.checkpoint),
                    "frames": total,
                    "width": width,
                    "cameras": camera_keys,
                    "image_dim": image_dim,
                    "stride": args.stride,
                    "layout": "per-camera image mean (3 x D) then masked language mean (D)",
                    "dtype": "float16",
                },
                indent=2,
            )
            + "\n"
        )
    filled = int(written.sum())
    print(f"[extract] shard {args.shard_index} done in {(time.monotonic() - started) / 60:.1f} min;"
          f" {filled}/{total} rows written so far", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
