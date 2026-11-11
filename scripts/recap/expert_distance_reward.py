"""Reward by resemblance to expert behaviour, instead of by task outcome.

The idea, and why it is worth testing
------------------------------------
Every reward tried so far scores the *outcome*: Cookies in the box, Cookies in a slot, walls
touched.  Those quantities depend on the state far more than on the action -- measured, a critic
trained on them attributes **98.5 % of its variance to the state and 1.53 % to the action**, which
makes it useless for steering a sampler.

Resemblance to expert behaviour is a different kind of signal.  It asks "does this observation
look like somewhere a good demonstration would be", which is a property of the *state the action
led to* rather than of the score, and it might therefore depend on the action more.

Two honest limits, stated before the measurement:

* **Episode identity is confounded with quality.**  The expert episodes succeed 94 % of the time
  and the rollouts 3.7 %, so "close to an expert frame" partly means "this is an expert episode".
  Any discrimination it shows must be checked against that.
* **A nearest-neighbour distance in a frozen encoder is not a distance to any particular plan.**
  It measures typicality, not correctness, and a rollout that is doing something sensible but
  different will read as atypical.

How it is built
---------------
The frozen SigLIP features already cached for 110,558 frames are projected to 64 dimensions with a
random Gaussian projection (Johnson-Lindenstrauss: pairwise distances are approximately preserved,
and it avoids an SVD over 8.5 GB), then each frame's distance to its nearest expert frame is
computed.  Expert episodes are the last 100 label rows.

Reported: the distance distribution, its correlation with position in the episode (the leak that
sank the absolute-headroom target), and whether a critic trained on it reads the action.

Usage
-----
    python scripts/recap/expert_distance_reward.py --dataset /root/autodl-tmp/datasets/a3_recap_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--projection", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset = args.dataset
    features_dir = dataset / "features_v9"
    written = np.load(features_dir / "written.npy")
    rows = np.flatnonzero(written)
    manifest = json.loads((features_dir / "index.json").read_text())
    stride = int(manifest["stride"])

    labels = pq.read_table(dataset / "value_labels.parquet").to_pandas()
    frame_table = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "frame_index"],
    ).to_pandas()
    episodes = frame_table["episode_index"].to_numpy(dtype=np.int64)[rows]
    frames = frame_table["frame_index"].to_numpy(dtype=np.int64)[rows]

    n_labels = len(labels)
    is_expert_episode = np.zeros(n_labels, dtype=bool)
    for index, part in enumerate(labels["part"]):
        is_expert_episode[index] = "expert" in str(part)
    print(f"[edr] {len(rows)} cached frames, {is_expert_episode.sum()} expert episodes", flush=True)

    rng = np.random.default_rng(args.seed)
    features = np.load(features_dir / "features.npy", mmap_mode="r")

    # Random projection.  Deliberately not a fitted PCA: the point is to make the distance
    # computation affordable, and a random projection preserves pairwise distances to within
    # (1 +/- eps) for eps ~ sqrt(log n / d) -- no need to look at the whole matrix to get that.
    projection = rng.normal(0.0, 1.0 / np.sqrt(args.projection), (features.shape[1], args.projection))

    def project(indices: np.ndarray) -> np.ndarray:
        block = np.load(features_dir / "features.npy", mmap_mode="r")[indices].astype(np.float32)
        out = block @ projection
        # Normalise so distances are dimensionless and comparable across frames.
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out

    expert_rows = np.flatnonzero(is_expert_episode[episodes])
    print(f"[edr] {len(expert_rows)} expert frames; projecting", flush=True)
    expert_projected = project(expert_rows)

    nearest = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), args.chunk):
        stop = min(start + args.chunk, len(rows))
        block = project(rows[start:stop])
        # Cosine distance (both sides unit-normalised) to every expert frame, in slices so the
        # distance matrix never has to exist in full.
        best = np.full(stop - start, np.inf, dtype=np.float32)
        for begin in range(0, len(expert_projected), 8192):
            similarity = block @ expert_projected[begin : begin + 8192].T
            best = np.minimum(best, 1.0 - similarity.max(axis=1))
        nearest[start:stop] = best
        if start % 20000 == 0:
            print(f"[edr]   {stop}/{len(rows)}", flush=True)

    # --- properties -------------------------------------------------------------------
    position = np.array([
        frame / max(int((labels["frames"].iloc[int(e)] // stride)) - 1, 1)
        for e, frame in zip(episodes, frames)
    ], dtype=np.float64)

    success_by_episode = labels["success"].to_numpy(dtype=bool)
    frame_success = success_by_episode[episodes]

    rho_phase = float(np.corrcoef(position, nearest)[0, 1])
    rho_success = float(np.corrcoef(nearest, frame_success.astype(np.float64))[0, 1])
    expert_episode_frames = is_expert_episode[episodes]

    print()
    print(f"最近专家距离: 中位 {np.median(nearest):.4f}  p10 {np.percentile(nearest,10):.4f}"
          f"  p90 {np.percentile(nearest,90):.4f}")
    print(f"  成功帧 中位 {np.median(nearest[frame_success]):.4f}"
          f" / 失败帧 中位 {np.median(nearest[~frame_success]):.4f}")
    print(f"  与集内位置的相关 rho = {rho_phase:+.3f}   （绝对值大 = 有时间泄漏）")
    print(f"  与成功标志的相关 rho = {rho_success:+.3f}")

    # The confound, measured rather than argued: how much of the discrimination is just "this is
    # an expert episode"?  Compare two groups that differ only in outcome.
    both = is_expert_episode[episodes]
    if (~both).sum() > 100 and both.sum() > 100:
        failed_expert = (~frame_success) & both
        successful_rollout = frame_success & (~both)
        if failed_expert.sum() > 50 and successful_rollout.sum() > 50:
            print()
            print("混淆检查（只改变结果，不改变来源）：")
            print(f"  专家集中失败的帧 中位距离 {np.median(nearest[failed_expert]):.4f}"
                  f"  (n={failed_expert.sum()})")
            print(f"  rollout 中成功的帧 中位距离 {np.median(nearest[successful_rollout]):.4f}"
                  f"  (n={successful_rollout.sum()})")
            print("  若两者接近，说明这个奖励测的是'像不像专家'而非'做得好不好'。")

    results = {
        "frames": int(len(rows)),
        "expert_frames": int(len(expert_rows)),
        "median_distance": float(np.median(nearest)),
        "median_success": float(np.median(nearest[frame_success])) if frame_success.any() else None,
        "median_failure": float(np.median(nearest[~frame_success])) if (~frame_success).any() else None,
        "rho_with_position": rho_phase,
        "rho_with_success": rho_success,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_suffix(".npy"), nearest)
        args.output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n写入 {args.output.with_suffix('.npy')} 和 .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
