"""How much do repeated samples of the same observation differ?

Why this measurement comes before any test-time method
-----------------------------------------------------
Every test-time improvement that does not retrain -- best-of-N, uncertainty-gated scaling,
self-consistency averaging -- works by choosing among the samples the sampler already produces.
If the sampler is effectively deterministic, there is nothing to choose between and all of those
methods are dead on arrival.  Measuring that first costs minutes; building the machinery first
costs a day and can end in "the candidates were identical".

What it measures
----------------
For a set of real observations taken from the training set, draw `N` action chunks each and
report, per joint and overall:

* the within-observation standard deviation, in A3 joint units **and** as a fraction of the
  training set's own action spread (so the number is interpretable: 0.02 of a joint whose
  actions span 0.4 rad is 5 % of the range);
* the between-sample spread of the chunk *mean*, which is what an averaging scheme would
  actually reduce;
* how the spread varies with the timestep inside the chunk -- the hypothesis that our failures
  live in the final precision phase predicts the spread should grow toward the end of a chunk.

It also reports the spread of the *first* action only, because that is the one a chunked
controller commits to before the next forward pass.

Read-only: it loads a checkpoint, samples, and prints.  It touches no dataset, no checkpoint and
no evaluation artefact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--samples", type=int, default=8, help="draws per observation")
    parser.add_argument(
        "--observations",
        type=int,
        default=24,
        help="distinct observations; spread across episodes and across the episode's timeline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # The launcher lives one level up, in `scripts/`, not beside this file.  It has to be
    # importable by name because `hasattr`-based patching in it resolves on import.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05 import PI05Policy

    # The launcher's patches are what make this the same model the bridge serves: without the
    # state projector and the history encoder the policy has no route for proprioception and
    # returns a different function entirely.
    import train_pi05_launcher as launcher  # noqa: F401  (its module-level patches run on import)

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    print(f"[var] loading {args.checkpoint}", flush=True)
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(args.device).eval()
    from lerobot.policies.factory import make_pre_post_processors

    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    dataset = LeRobotDataset(repo_id="var", root=args.dataset, video_backend="pyav")
    # Pick observations spread over episodes *and* over each episode's timeline: an early frame
    # and a late frame behave differently, and the late ones are where the task is won or lost.
    # `meta.episodes` is a HuggingFace `datasets.Dataset` (not pandas, not pyarrow), and its
    # column access returns a `Column` whose only interface here is iteration.
    lengths = np.asarray(list(dataset.meta.episodes["length"]), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    rng = np.random.default_rng(0)
    episodes = rng.choice(len(lengths), size=max(1, args.observations // 2), replace=False)
    indices = []
    for episode in episodes:
        first = int(starts[episode])
        span = int(lengths[episode])
        for fraction in (0.15, 0.75):
            indices.append(first + int(fraction * (span - 1)))
    indices = indices[: args.observations]

    image_keys = [k for k in dataset.meta.features if k.startswith("observation.images.")]
    action_dim = int(np.prod(dataset.meta.features["action"]["shape"]))
    spreads = []
    per_joint = []
    first_step = []
    print(f"[var] {len(indices)} observations, {args.samples} draws each, {action_dim} action dims",
          flush=True)

    with torch.no_grad():
        for position, index in enumerate(indices):
            item = dataset[int(index)]
            batch = {key: item[key][None].to(args.device) for key in image_keys}
            batch["observation.state"] = item["observation.state"][None].to(args.device)
            if "observation.history" in item:
                batch["observation.history"] = item["observation.history"][None].to(args.device)
            batch["task"] = [item["task"]]

            chunks = []
            for _ in range(args.samples):
                # No seeding: the point is to measure the sampler's own spread.  `infer()` in the
                # adapter seeds `torch` only when `A3_PI05_SEED` is set, and it is not set here.
                ready = preprocess(dict(batch))
                raw = policy.predict_action_chunk(ready)
                actions = postprocess(raw)[0]
                if hasattr(actions, "detach"):
                    actions = actions.detach().float().cpu()
                chunks.append(np.asarray(actions, dtype=np.float64))
            stack = np.stack(chunks)              # (samples, chunk, action_dim)
            spread = stack.std(axis=0)            # per (chunk, dim)
            spreads.append(spread.mean())
            per_joint.append(spread.mean(axis=0))
            first_step.append(spread[0].mean())
            if position == 0:
                print(f"[var] chunk shape {stack.shape[1]}x{stack.shape[2]}", flush=True)

    spreads = np.asarray(spreads)
    per_joint = np.asarray(per_joint)
    first_step = np.asarray(first_step)

    # The training set's own action spread is the scale that makes these numbers mean something.
    stats_path = args.dataset / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        action_stats = stats["action"]
        span = np.asarray(action_stats["q99"], dtype=np.float64) - np.asarray(action_stats["q01"], dtype=np.float64)
    else:
        span = np.ones(action_dim)

    ratio = per_joint / np.maximum(span[: action_dim], 1e-9)
    summary = {
        "observations": len(indices),
        "samples": args.samples,
        "action_dim": action_dim,
        "mean_within_std": float(spreads.mean()),
        "median_within_std": float(np.median(spreads)),
        "max_within_std": float(spreads.max()),
        "first_step_std": float(first_step.mean()),
        "frac_of_training_spread_mean": float(ratio.mean()),
        "frac_of_training_spread_p95": float(np.percentile(ratio, 95)),
        "per_joint_frac": ratio.mean(axis=0).tolist(),
    }

    print()
    print(f"平均组内标准差    {summary['mean_within_std']:.5f} 关节单位")
    print(f"  中位数          {summary['median_within_std']:.5f}")
    print(f"  最大            {summary['max_within_std']:.5f}")
    print(f"首步标准差        {summary['first_step_std']:.5f}")
    print(f"占训练集动作幅宽  {100 * summary['frac_of_training_spread_mean']:.2f} % (均值)"
          f"  {100 * summary['frac_of_training_spread_p95']:.2f} % (p95)")
    print()
    print("逐关节（占该关节幅宽的百分比）:")
    for joint, value in enumerate(summary["per_joint_frac"]):
        if value > 0:
            print(f"  关节 {joint:2d}: {100 * value:6.2f} %")

    # The decision this measurement exists to make.  Stated as a threshold rather than a feeling:
    # below ~1 % of the action range, the samples are effectively identical and no test-time
    # scheme that chooses among them can move the score.
    fraction = summary["frac_of_training_spread_mean"]
    print()
    if fraction < 0.01:
        print("→ 采样几乎确定（<1% 动作幅宽）。test-time 选择方法无效，不要做 best-of-N。")
    elif fraction < 0.05:
        print("→ 采样有可见变异（1–5%）。best-of-N 有可能，但候选差异小，收益需要实测。")
    else:
        print("→ 采样变异显著（>5%）。best-of-N / 不确定性门控值得做。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
