"""Can DSRL's training step run on one 32 GB card?

DSRL freezes the policy and trains a small actor in its *noise* space.  The gradient has to travel
from the critic, back through the frozen denoiser's ODE, into the noise vector.  Whether that fits
is a memory and time question, not a research question, and it is answerable in minutes.

What makes it affordable here, measured from the code rather than assumed
------------------------------------------------------------------------
`PI05Pytorch.sample_actions` does two clearly separated things:

* the **prefix** (PaliGemma: images + language + the A3 history token) is encoded once and its
  key/value cache is kept;
* the **denoiser** then runs `num_inference_steps = 10` (not the 25 that a diffusion reading would
  suggest), each a forward of the ~300 M action expert against that cached prefix.

So a backward pass has to retain activations for ten action-expert forwards plus one prefix
encode, and the prefix encode itself needs no gradient -- the noise enters at the denoiser, not at
the images.  That is a very different cost profile from "BPTT through 25 steps of a 2 B model".

What this script measures
-------------------------
For a range of batch sizes, one `loss.backward()` through the frozen policy with the gradient
entering at `noise`:

* peak GPU memory,
* wall time per step,
* and whether the gradient that arrives at `noise` is non-zero and finite (a zero gradient would
  mean the policy is not actually differentiable end to end, which no amount of memory would fix).

Usage
-----
    python scripts/recap/measure_dsrl_feasibility.py --batches 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import train_pi05_launcher as launcher  # noqa: F401  (module-level patches run on import)

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    device = args.device
    print(f"[dsrl] loading {args.checkpoint.name}", flush=True)
    policy = PI05Policy.from_pretrained(str(args.checkpoint)).to(device).eval()
    # Freeze: DSRL never updates the policy, and `requires_grad=False` on the parameters is what
    # keeps the optimiser state out of memory.  The *activations* still need gradients, which is
    # exactly what `torch.no_grad()` must not be applied here.
    for parameter in policy.parameters():
        parameter.requires_grad = False
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[dsrl] policy frozen: {n_trainable} trainable of {n_total}", flush=True)

    preprocess, _ = make_pre_post_processors(
        policy.config, str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    dataset = LeRobotDataset(repo_id="dsrl", root=args.dataset, video_backend="pyav")
    image_keys = [k for k in dataset.meta.features if k.startswith("observation.images.")]

    def sample_batch(size: int) -> dict:
        """One prepared batch, identical across attempts so timing is comparable."""
        items = [dataset[i] for i in range(size)]
        batch: dict = {key: torch.stack([item[key] for item in items]).to(device) for key in image_keys}
        batch["observation.state"] = torch.stack([item["observation.state"] for item in items]).to(device)
        if "observation.history" in items[0]:
            batch["observation.history"] = torch.stack(
                [item["observation.history"] for item in items]
            ).to(device)
        batch["task"] = [item["task"] for item in items]
        return batch

    model = policy.model  # PI05Pytorch: `sample_actions` lives here
    num_steps = int(policy.config.num_inference_steps)
    print(f"[dsrl] {num_steps} denoising steps, chunk {policy.config.chunk_size},"
          f" action dim {policy.config.max_action_dim}", flush=True)

    rows = []
    for size in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = sample_batch(size)
        ready = preprocess(batch)

        # `sample_actions` carries `@torch.no_grad()`, which wins over an enclosing
        # `enable_grad()` because the decorator re-enters no-grad inside the function body.  So
        # the sampling loop is re-implemented here, line for line from
        # `modeling_pi05.PI05Pytorch.sample_actions`, with the decorator removed.  This is the
        # actual cost of adding DSRL to this stack: the inference entry point cannot be reused.
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        images_prepared, img_masks = policy._preprocess_images(ready)  # noqa: SLF001
        tokens = ready["observation.language.tokens"]
        masks = ready["observation.language.attention_mask"]

        started = time.monotonic()
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images_prepared, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # `sample_actions` sets this too: with the default SDPA kernel the mask dtype must match
        # the query dtype exactly, and the stored model is bf16, giving
        # "invalid dtype for bias - should match query's dtype".
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        noise = torch.randn(
            size, model.config.chunk_size, model.config.max_action_dim,
            device=device, dtype=prefix_embs.dtype,
        ).requires_grad_(True)

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time_value = 1.0 + step * dt
            time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(size)
            v_t = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        chunk = x_t
        surrogate = (chunk.float() ** 2).mean()
        surrogate.backward()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        noise_holder = {"noise": noise}

        noise = noise_holder.get("noise")
        if noise is None or noise.grad is None:
            print(f"[dsrl] batch {size}: no gradient reached the noise -- not differentiable here")
            rows.append({"batch": size, "grad": "missing"})
            continue

        grad = noise.grad
        grad_norm = float(grad.float().norm())
        finite = bool(torch.isfinite(grad).all())
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[dsrl] batch {size:2d}: peak {peak_gb:5.2f} GB, {elapsed:6.2f} s/step,"
              f" |dL/dnoise| {grad_norm:10.4e}, finite {finite}", flush=True)
        rows.append({
            "batch": size,
            "peak_gb": peak_gb,
            "seconds": elapsed,
            "grad_norm": grad_norm,
            "finite": finite,
        })
        del batch, ready, chunk, surrogate, noise, grad
    # --- the verdict ---------------------------------------------------------------
    print()
    ok = [r for r in rows if isinstance(r.get("peak_gb"), float) and r.get("finite")]
    if not ok:
        print("→ 梯度没有到达噪声。DSRL 在这个实现里不可行（无论显存多少）。")
        verdict = "no_gradient"
    else:
        largest = max(ok, key=lambda r: r["batch"])
        headroom = 32.0 - largest["peak_gb"]
        print(f"最大可用 batch {largest['batch']}：峰值 {largest['peak_gb']:.2f} GB，"
              f"剩余 {headroom:.2f} GB")
        print(f"单步 {largest['seconds']:.2f} s，1000 步约 {largest['seconds'] * 1000 / 60:.0f} 分钟")
        if headroom > 2.0:
            print("→ 显存足够。DSRL 的 actor 更新（额外一个小网络和两个 critic）可以加进来。")
            verdict = "feasible"
        else:
            print("→ 显存接近上限，只能跑很小 batch。")
            verdict = "tight"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
