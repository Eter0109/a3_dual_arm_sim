#!/usr/bin/env python3
"""How much does the steering actor actually change the action chunk?

Why measure this before spending two hours on an evaluation
-----------------------------------------------------------
An evaluation that shows "no difference" has three very different explanations, and the episode
scores cannot tell them apart:

1. the actor's noise really is no better than `N(0, I)` -- a real negative result;
2. the actor's noise is pathological (saturated at the `tanh` bound, or so small in effect that
   the sampler ignores it) -- a bug;
3. the observation the actor sees at inference differs from the one it trained on -- the bug
   `verify_visual_feature.py` exists to rule out.

This script separates (2) from (1) directly, on the same observation, with no environment and no
episode budget.  It answers, for a batch of real frames:

* how far the actor's chunk is from the default-noise chunk, relative to the action scale.  If this
  is ~0 the actor is inert; if it is larger than the episode-to-episode spread of the policy's own
  output, steering dominates the policy, which is also a problem.
* what fraction of the actor's `tanh` outputs sit within 1e-3 of +-1.  Saturation is the failure
  §9.88 recorded once already (`|mean| = 52728`): the gradient dies and the actor stops responding
  to the observation.
* how much of the chunk variance is explained by the *noise* versus by the *observation*.  Both
  chunks come from the same observation, so their difference is the noise's total effect; comparing
  that against the spread across observations says whether the actor is steering or shouting.

Usage
-----
    python scripts/recap/probe_steering_effect.py --actor <checkpoint.pt> --rows 32
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
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--batch", type=int, default=4, help="frames per VLA pass")
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

    from steering import SteeringActor, encode_prefix, sample_with_prefix

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, _ = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    num_steps = int(policy.config.num_inference_steps)
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    actor = SteeringActor(args.actor, device=args.device)
    print(f"[probe] actor {args.actor.name}: R^{actor.actor_dim} ({actor.noise_mode}), "
          f"observation {actor.observation_dim}, parts {actor.parts}", flush=True)

    dataset = LeRobotDataset(repo_id="dsrl", root=args.dataset, video_backend="pyav")
    rows = np.sort(rng.choice(len(dataset), size=args.rows, replace=False))

    def batch_of(index):
        items = [dataset[int(i)] for i in index]
        raw = {key: torch.stack([item[key] for item in items])
               for key in dataset.meta.features if key.startswith("observation.images.")}
        raw["observation.state"] = torch.stack([item["observation.state"] for item in items])
        if "observation.history" in items[0]:
            raw["observation.history"] = torch.stack(
                [item["observation.history"] for item in items]
            )
        raw["task"] = [item["task"] for item in items]
        return preprocess(raw)

    default_chunks, actor_chunks, default_chunks_2 = [], [], []
    saturation, magnitudes = [], []
    for start in range(0, len(rows), args.batch):
        index = rows[start:start + args.batch]
        prepared = batch_of(index)
        with torch.no_grad():
            observation = actor.observe(policy, prepared, encode_prefix(
                policy, prepared, keep_embeddings=True))

            prefix = encode_prefix(policy, prepared, keep_embeddings=False)
            # Two independent draws from the default prior, to measure the policy's own noise
            # sensitivity: that is the yardstick the actor's effect has to be compared against.
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            first = torch.randn(len(index), chunk_size, action_dim,
                                device=args.device, generator=generator)
            second = torch.randn(len(index), chunk_size, action_dim,
                                 device=args.device, generator=generator)
            default_chunks.append(sample_with_prefix(policy, prefix, first, num_steps, args.device))
            default_chunks_2.append(sample_with_prefix(policy, prefix, second, num_steps, args.device))

            noise = actor.noise(observation, deterministic=True)
            actor_chunks.append(sample_with_prefix(policy, prefix, noise, num_steps, args.device))

            unit = torch.tanh(actor.actor.forward(observation))
            saturation.append((unit.abs() > 0.999).float().mean().item())
            magnitudes.append(unit.abs().mean().item())
        print(f"    {min(start + args.batch, len(rows))}/{len(rows)}", flush=True)

    default = torch.cat(default_chunks).float()
    default2 = torch.cat(default_chunks_2).float()
    steered = torch.cat(actor_chunks).float()

    # Everything below is in the sampler's output units (normalised actions), which is the space
    # the critics see and the only space in which "big" is meaningful.
    scale = float(default.std())
    noise_effect = float((steered - default).abs().mean())
    noise_sensitivity = float((default2 - default).abs().mean())
    across_frame = float(default.mean(dim=1).std(dim=0).mean())
    finite = bool(torch.isfinite(steered).all() and torch.isfinite(default).all())

    print("\n结果")
    print(f"  chunk 尺度（默认噪声下的 std）        {scale:.4f}")
    print(f"  换一次默认噪声造成的平均绝对变化      {noise_sensitivity:.4f}"
          f"  ({100 * noise_sensitivity / max(scale, 1e-9):.1f} % of scale)")
    print(f"  actor 噪声 vs 默认噪声               {noise_effect:.4f}"
          f"  ({100 * noise_effect / max(scale, 1e-9):.1f} % of scale)")
    print(f"  跨观测差异（不同帧之间）              {across_frame:.4f}"
          f"  ({100 * across_frame / max(scale, 1e-9):.1f} % of scale)")
    print(f"  actor 的 tanh 饱和比例 (|u|>0.999)   {100 * float(np.mean(saturation)):.2f} %")
    print(f"  actor 的 |u| 均值                     {float(np.mean(magnitudes)):.4f}")
    print(f"  数值有限                              {finite}")
    print()
    ratio = noise_effect / max(noise_sensitivity, 1e-9)
    if not finite:
        print("  判读：出现 NaN/Inf。评估没有意义，先查采样器。")
    elif float(np.mean(saturation)) > 0.5:
        print("  判读：actor 大面积饱和在 tanh 边界，梯度已死。这是 §9.88 那个失败的同类。")
    elif ratio < 0.1:
        print("  判读：actor 的影响比换一次噪声小一个数量级——它基本是惰性的。")
        print("        若评估也无差异，那是 actor 没学出来，不是评估坏了。")
    elif ratio > 3.0:
        print("  判读：actor 的影响远超策略自身的噪声敏感性——它在压过策略，")
        print("        这通常意味着 critic 的高估被直接放大到了动作上。")
    else:
        print(f"  判读：actor 的影响是常规噪声敏感性的 {ratio:.1f} 倍，量级合理。")
        print("        评估若显示差异，可信；若无差异，是 actor 没找到更好的噪声。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json  # noqa: PLC0415
        args.output.write_text(json.dumps({
            "actor": str(args.actor),
            "rows": int(len(rows)),
            "chunk_std": scale,
            "default_noise_sensitivity": noise_sensitivity,
            "actor_effect": noise_effect,
            "effect_over_sensitivity": ratio,
            "across_frame_std": across_frame,
            "saturation_fraction": float(np.mean(saturation)),
            "unit_mean_abs": float(np.mean(magnitudes)),
            "finite": finite,
        }, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
