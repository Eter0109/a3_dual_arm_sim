#!/usr/bin/env python3
"""Does one round of the trainer's own updates push Q down, and by how much?

Where the investigation stands
------------------------------
`probe_td_fixed_point.py` showed the trainer's update rules converge *upward* to the correct
discounted value on synthetic data (constant state, constant action, reward +1): 0.00 -> 15.34 and
still climbing toward 100.  So the rules are not inverted.

The online loop, however, reports `q = -4.20` as early as **round 5**, before any episode has
finished, so data accumulation is not the cause.  Measured initialisation at the operating point is
between -0.45 and +0.24 depending on seed, so the negative value is produced by the first updates.

What differs between the synthetic probe and the real batch
-----------------------------------------------------------
The probe used one state and one action.  The real batch has:

* **observation** = 2,320 dims whose first 16 are un-normalised proprioception, so the magnitudes
  reaching the first Linear differ by orders of magnitude across columns;
* **chunk** = 1,600 dims of joint targets, most of them zero (the sampler pads `action_dim` 32 while
  only 16 columns reach the arm);
* **reward** = zero on 96 % of rows (a cookie lands every ~28 steps);
* **`obs_next`** = the next state, heavily correlated with `obs`.

This probe isolates them one at a time: build a batch from the real pipeline, then replace pieces
with the synthetic version and see which replacement stops Q from going negative.  The one that
matters is the answer, and it also says whether the fix belongs in the data or in the network.

Usage
-----
    python scripts/recap/probe_q_init_drift.py --rounds 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
os.environ.setdefault("A3_MAIN_ROOT", "/tmp/a3-main")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering import (  # noqa: E402
    build_observation,
    encode_prefix,
    repeat_noise,
    sample_with_prefix,
    soft_update,
    twin_critic,
    visual_feature,
)
from train_dsrl_online import (  # noqa: E402
    CAMERAS,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
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
    original_dim = int(policy.config.output_features["action"].shape[0])
    num_steps = int(policy.config.num_inference_steps)

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    task = training_prompt()
    rng = np.random.default_rng(args.seed)

    projection = torch.from_numpy(
        rng.normal(0.0, 1.0 / np.sqrt(256), (8192, 256)).astype(np.float32)).to(args.device)
    parts = {"proprio": True, "vlm": True, "visual": True}

    print(f"[probe] building {args.states} real states through the VLA pipeline")
    observations, chunks = [], []
    for index in range(args.states):
        seed = args.seed * 1000 + index * 31
        env = bench.create_env(render_cameras=False)
        env.reset(seed=seed, options=reset_options(bench, seed))
        scene = Scene(index=index, env=env, renderer=_make_renderer(env.model),
                      camera_ids=dict(env._camera_ids), seed=seed,  # noqa: SLF001
                      required_cookies=int(bench.required_cookies), max_control_steps=1500)
        frames = [render_frames(scene.renderer, scene)]
        obs = [env._observation()]  # noqa: SLF001
        scene.seed_history(obs[0])
        raw = raw_batch([scene], frames, obs, task, args.device)
        prepared = preprocess(raw)
        prefix = encode_prefix(policy, prepared, keep_embeddings=True)
        with torch.no_grad():
            visual = visual_feature(policy, prefix) @ projection
            vector = build_observation(prepared[STATE].float(), prefix.hidden_last, visual, parts)
            noise = torch.randn(1, action_dim, device=args.device)
            flat = repeat_noise(noise, chunk_size, action_dim, "single")
            made = sample_with_prefix(policy, prefix, flat, num_steps, args.device)
        padded = torch.zeros(1, chunk_size, action_dim, device=args.device)
        padded[:, :, :original_dim] = made[:, :, :original_dim]
        observations.append(vector.detach())
        chunks.append(padded.reshape(1, -1).detach())
        env.close()
    obs_real = torch.cat(observations, 0)
    chunk_real = torch.cat(chunks, 0)
    print(f"   obs {tuple(obs_real.shape)}  |obs| mean {float(obs_real.abs().mean()):.3f}"
          f"  max {float(obs_real.abs().max()):.2f}")
    print(f"   chunk {tuple(chunk_real.shape)}  |chunk| mean {float(chunk_real.abs().mean()):.3f}"
          f"  max {float(chunk_real.abs().max()):.2f}")

    # The four variants, each swapping one suspect for the synthetic version.
    variants = {
        "real everything": (obs_real, chunk_real, obs_real, chunk_real),
        "reward +1": (obs_real, chunk_real, obs_real, chunk_real),
        "chunk ~ N(0,1)": (obs_real, torch.randn_like(chunk_real), obs_real,
                            torch.randn_like(chunk_real)),
        "obs standardised": (torch.randn_like(obs_real), chunk_real,
                              torch.randn_like(obs_real), chunk_real),
    }

    print()
    header = f"   {'variant':22s} {'step1':>8s} {'step50':>8s} {'step100':>8s} {'step400':>8s}"
    print(header)
    print("   " + "-" * (len(header) - 3))
    for name, (obs, chunk, obs_next, chunk_next) in variants.items():
        torch.manual_seed(args.seed)
        critic = twin_critic(obs_real.shape[1], chunk_real.shape[1], hidden=args.hidden).to(
            args.device)
        critic_target = twin_critic(obs_real.shape[1], chunk_real.shape[1],
                                    hidden=args.hidden).to(args.device)
        critic_target.load_state_dict(critic.state_dict())
        optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
        n = obs.shape[0]
        reward = (torch.ones(n, device=args.device) if name == "reward +1"
                  else torch.zeros(n, device=args.device))
        trace = {}
        for step in range(1, args.updates + 1):
            index = torch.randint(0, n, (args.batch,), device=args.device)
            with torch.no_grad():
                next_q = torch.min(
                    torch.stack([h(obs_next[index], chunk_next[index]) for h in critic_target]),
                    dim=0).values
                target = reward[index].unsqueeze(-1) + args.gamma * next_q
            q = torch.stack([h(obs[index], chunk[index]) for h in critic])
            loss = 0.5 * sum(F.mse_loss(v, target) for v in q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            optimizer.step()
            soft_update(critic_target, critic, args.tau)
            if step == 1:
                trace[1] = float(q.mean().detach())
            if step in (50, 100, 400):
                with torch.no_grad():
                    trace[step] = float(
                        torch.stack([h(obs, chunk) for h in critic]).mean())
        cfg = (f"   {name:22s} {trace[1]:+8.3f} {trace[50]:+8.3f}"
               f" {trace[100]:+8.3f} {trace[400]:+8.3f}")
        print(cfg)

    print()
    print("=" * 78)
    print("读法")
    print("=" * 78)
    print("  只有换掉某一项之后 Q 才不再变负 -> 问题在该项的数据分布上。")
    print("  所有变体都变负 -> 问题在 critic 的输入/网络结构，而不是数据内容。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
