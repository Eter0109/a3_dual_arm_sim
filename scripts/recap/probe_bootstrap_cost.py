#!/usr/bin/env python3
"""Can the TD bootstrap be recomputed with the *current* actor, and at what cost?

Why this measurement decides an implementation question
------------------------------------------------------
`dsrl.py:274-278` recomputes the bootstrap action every gradient step:

    next_actions = D(s', actor_noise(s'))        # current actor, fresh denoise

`train_dsrl_online.py` stores `sampler(s', noise)` in the replay at collection time instead.  That
is only equivalent while the actor is not moving.  Measured on run4, it is not: `q` fell from 2.53
to -0.19 while the true discounted return rose from 2.28 to 4.23, a bias that grew monotonically to
**-4.3 cookies**.  `alpha` annealed 0.9 -> 0.003 over the same window, so the actor moved a lot and
the stored target became the value of a policy that no longer exists.

The obstacle
------------
Recomputing needs a `Prefix` for `s'`, and a `Prefix` holds the PaliGemma key/value cache -- far too
large to store per replay row.  So the only route is to re-run the encoder on `s'`'s images, which
means the replay must also hold images (3 x 224 x 224 x 3 = 451 KB per row, uint8).

Before designing around that, the cost has to be known, because the round already costs 11 s and the
budget is finite.  Three numbers matter, all for a batch of `n` states:

    preprocess(n)        image resize/normalise
    encode_prefix(n)     SigLIP + PaliGemma forward, producing the cache
    sample(n)            the 10-step denoise

If `encode_prefix` at n=64 is a few hundred milliseconds, a rolling bootstrap pool is affordable.
If it is seconds, the honest answer is that gap 2 stays open and must be documented as a gap rather
than papered over.

Usage
-----
    python scripts/recap/probe_bootstrap_cost.py --batches 8 32 64 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
os.environ.setdefault("A3_MAIN_ROOT", "/tmp/a3-main")

import numpy as np
import torch

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering import encode_prefix, repeat_noise, sample_with_prefix  # noqa: E402
from train_dsrl_online import (  # noqa: E402
    CAMERAS,
    HISTORY_DIM,
    HISTORY_FRAMES,
    STATE,
    Scene,
    _make_renderer,
    raw_batch,
    render_frames,
    reset_options,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=4400)
    args = parser.parse_args(argv)

    import train_pi05_launcher as launcher
    from bench_pi05_on_main import make_cookie_benchmark, training_prompt
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    vla = Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model")
    policy = PI05Policy.from_pretrained(str(vla)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, _post = make_pre_post_processors(
        policy.config, str(vla),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    num_steps = int(policy.config.num_inference_steps)

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    task = training_prompt()
    biggest = max(args.batches)

    # Build `biggest` distinct states once, then slice.  `raw_batch` wants one frame list and one
    # observation per scene, exactly as the trainer supplies them.
    frames_all, obs_all = [], []
    for index in range(biggest):
        seed = args.seed + index * 13
        env = bench.create_env(render_cameras=False)
        env.reset(seed=seed, options=reset_options(bench, seed))
        scene = Scene(index=index, env=env, renderer=_make_renderer(env.model),
                      camera_ids=dict(env._camera_ids), seed=seed,  # noqa: SLF001
                      required_cookies=int(bench.required_cookies), max_control_steps=1500)
        frames_all.append(render_frames(scene.renderer, scene))
        obs_all.append(env._observation())  # noqa: SLF001
        env.close()
    print(f"[probe] built {biggest} states")

    print()
    print(f"   {'batch':>6} {'preprocess':>11} {'encode_prefix':>14} {'denoise':>9}"
          f" {'total':>8} {'per row':>8}")
    print("   " + "-" * 62)
    results = {}
    for n in args.batches:
        # `raw_batch` builds the history window from the scene objects; a zero window is fine here
        # because only the cost is being measured, and the shape is what the encoder checks.
        raw = raw_batch([None] * n, frames_all[:n], obs_all[:n], task, args.device) \
            if False else None
        # `raw_batch` needs the scenes for their history vectors, so build it inline instead.
        stacks = {f"observation.images.{name}": [] for name in CAMERAS}
        for frames in frames_all[:n]:
            for name in CAMERAS:
                stacks[f"observation.images.{name}"].append(frames[name])
        batch = {key: torch.as_tensor(np.stack(value), device=args.device)
                 .permute(0, 3, 1, 2).float().div_(255.0)
                 for key, value in stacks.items()}
        batch[STATE] = torch.as_tensor(
            np.stack([np.asarray(o[STATE], dtype=np.float32) for o in obs_all[:n]]),
            device=args.device)
        history = np.zeros((n, HISTORY_DIM), dtype=np.float32)
        batch["observation.history"] = torch.as_tensor(history, device=args.device)
        batch["task"] = [task] * n
        del raw

        for _ in range(2):
            ready = preprocess(batch)
            prefix = encode_prefix(policy, ready, keep_embeddings=True)
            noise = torch.randn(n, chunk_size * action_dim, device=args.device)
            sample_with_prefix(policy, prefix, noise, num_steps, args.device)
        torch.cuda.synchronize()

        started = time.monotonic()
        ready = preprocess(batch)
        torch.cuda.synchronize()
        t_pre = time.monotonic() - started
        started = time.monotonic()
        prefix = encode_prefix(policy, ready, keep_embeddings=True)
        torch.cuda.synchronize()
        t_enc = time.monotonic() - started
        started = time.monotonic()
        noise = torch.randn(n, chunk_size * action_dim, device=args.device)
        sample_with_prefix(policy, prefix, noise, num_steps, args.device)
        torch.cuda.synchronize()
        t_den = time.monotonic() - started
        total = t_pre + t_enc + t_den
        results[n] = (t_pre, t_enc, t_den, total)
        print(f"   {n:6d} {t_pre * 1000:10.0f}ms {t_enc * 1000:13.0f}ms"
              f" {t_den * 1000:8.0f}ms {total * 1000:7.0f}ms {total / n * 1000:7.1f}ms")
        del batch, ready, prefix, noise
        torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("预算")
    print("=" * 70)
    round_cost = 11.0
    for n, (_p, _e, _d, total) in results.items():
        print(f"   bootstrap 池 {n:3d} 个状态 -> 每轮多 {total:5.2f} s"
              f"（一轮 {round_cost} s，+{total / round_cost * 100:4.0f}%）")
    print()
    print("   参考实现每个梯度步都重算（utd=20），所以这里只会更便宜。")
    print("   如果 encode_prefix 的 per-row 成本随 batch 下降得足够快，池子就能开大。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
