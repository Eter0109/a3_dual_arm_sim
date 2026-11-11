#!/usr/bin/env python3
"""Does the *live* visual feature match the cached one it was trained on?

The question this answers, and why it has to be answered before any evaluation
------------------------------------------------------------------------------
`train_dsrl.py` observes through a 256-dim projection of an 8192-dim frozen summary that
`extract_value_features.py` wrote to `features_v9/features.npy`.  At inference there is no such
file -- the summary has to be rebuilt from the frame itself, and `steering.visual_feature` rebuilds
it by slicing the `prefix_embs` tensor that `embed_prefix` already produced (three camera blocks
followed by the language block, with the language half scaled by `sqrt(hidden)`).

That reconstruction is the single most fragile step in the evaluation path, because getting it
wrong does not raise.  A mis-sliced block, a missing `sqrt` undoing, or a camera-order swap all
produce a 256-dim vector of roughly the right magnitude; the actor then sees features it was not
trained on, scores below the baseline, and the result reads as "the actor does not help" -- which
is precisely the effect this whole line of work is trying to measure.

So the comparison is made against the ground truth: rows the extraction already filled, the same
observations, the same projection.  Three levels are reported, because each one isolates a
different mistake:

* `raw`       -- the 8192-dim vector, live vs cached.  Bounded by the float16 round trip (~1e-2).
                 A camera-order or slice-length error shows up here and nowhere else.
* `projected` -- after the 256-dim projection the actor actually consumes.
* `relative`  -- projected difference over the projected spread.  This is the one to gate on: an
                 absolute diff means nothing without the scale it sits on.

Usage
-----
    python scripts/recap/verify_visual_feature.py --rows 0,5,10,15
    python scripts/recap/verify_visual_feature.py --count 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--actor", type=Path, default=None,
                        help="a train_dsrl.py checkpoint; its `projection` is used instead of a "
                             "freshly drawn one, which is the only projection that matters")
    parser.add_argument("--rows", default=None, help="explicit dataset rows, comma separated")
    parser.add_argument("--count", type=int, default=16, help="how many cached rows to sample")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import train_pi05_launcher as launcher  # noqa: F401

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    from steering import encode_prefix, load_feature_index, verify_visual_feature

    features_dir = args.dataset / "features_v9"
    index = load_feature_index(args.dataset)
    written = np.load(features_dir / "written.npy")
    cached_rows = np.flatnonzero(written)
    if len(cached_rows) == 0:
        print("[verify] no rows written; run extract_value_features.py first", flush=True)
        return 2

    # The projection has to be the *training* one.  Drawing a fresh Gaussian here would still
    # produce matching `raw` vectors while the actor's input stayed wrong, which is the failure
    # this script exists to catch -- so the actor checkpoint is preferred and a seed-derived
    # projection is only used as a fallback, with a warning.
    if args.actor is not None:
        state = torch.load(str(args.actor), map_location="cpu", weights_only=False)
        projection = np.asarray(state["projection"], dtype=np.float32)
        print(f"[verify] projection from {args.actor.name}", flush=True)
    else:
        projection_dim = 256
        projection = np.random.default_rng(args.seed).normal(
            0.0, 1.0 / np.sqrt(projection_dim), (index["width"], projection_dim)
        ).astype(np.float32)
        print(f"[verify] WARNING: no --actor given, using a seed-{args.seed} projection.  Only the"
              f" `raw` column is meaningful; the projected ones are against a projection no actor"
              f" was trained with.", flush=True)

    if args.rows:
        rows = np.array([int(x) for x in args.rows.split(",")], dtype=np.int64)
        bad = [int(r) for r in rows if not written[int(r)]]
        if bad:
            print(f"[verify] rows {bad} were never written by the extraction", flush=True)
            return 2
    else:
        rows = np.sort(np.random.default_rng(args.seed).choice(
            cached_rows, size=min(args.count, len(cached_rows)), replace=False
        ))

    print(f"[verify] dataset {args.dataset.name}, {len(rows)} rows, "
          f"{len(cached_rows)} cached of {len(written)}", flush=True)
    print(f"[verify] loading {args.checkpoint.parent.name}", flush=True)
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, _ = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    dataset = LeRobotDataset(repo_id="dsrl", root=args.dataset, video_backend="pyav")

    results = []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        items = [dataset[int(row)] for row in chunk]
        raw = {key: torch.stack([item[key] for item in items])
               for key in dataset.meta.features if key.startswith("observation.images.")}
        raw["observation.state"] = torch.stack([item["observation.state"] for item in items])
        if "observation.history" in items[0]:
            raw["observation.history"] = torch.stack(
                [item["observation.history"] for item in items]
            )
        raw["task"] = [item["task"] for item in items]
        prepared = preprocess(raw)
        with torch.no_grad():
            prefix = encode_prefix(policy, prepared, keep_embeddings=True)
            results.append(verify_visual_feature(
                policy, prepared, prefix, args.dataset, chunk, projection
            ))
        done = min(start + args.batch, len(rows))
        print(f"    {done}/{len(rows)}", flush=True)

    raw_max = max(r["raw_max_abs_diff"] for r in results)
    proj_max = max(r["projected_max_abs_diff"] for r in results)
    relative = max(r["relative"] for r in results)
    ok = all(r["ok"] for r in results)

    print("\n结果")
    print(f"  8192 维 raw 向量最大绝对误差   {raw_max:.4g}")
    print(f"  256 维投影后最大绝对误差       {proj_max:.4g}")
    print(f"  相对误差（除以投影后标准差）   {relative:.4g}")
    print()
    if ok:
        print("  判读：实时特征与缓存特征一致。评估路径可以信任。")
    else:
        print("  判读：不一致。actor 在评估时看到的输入与训练时不同，任何评估结果都无效。")
        print("        排查顺序：相机顺序 -> 语言块的 sqrt 还原 -> patch 数除法。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json  # noqa: PLC0415
        args.output.write_text(json.dumps({
            "rows": [int(r) for r in rows],
            "raw_max_abs_diff": raw_max,
            "projected_max_abs_diff": proj_max,
            "relative": relative,
            "ok": ok,
            "per_batch": results,
        }, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
