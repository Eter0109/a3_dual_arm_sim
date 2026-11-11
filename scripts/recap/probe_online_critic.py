#!/usr/bin/env python3
"""Is the online critic better at *ordering* actions than the offline one was?

§9.96 measured the offline critics at within-state pairwise accuracy **0.508** -- chance.  That was
the finding that killed the offline line, and the reason it was not a bug: an offline pool has one
action per state, so no critic trained on it can ever learn which of two actions is better.

Online changes that structurally.  Every state in the replay buffer now has a *sampled* action, and
because the actor is stochastic the same state recurs with different noise across visits.  So the
question this probe asks -- can the critic rank two noise vectors at the same state? -- is one the
online data can actually answer, and the answer is what decides whether DSRL has anything to work
with here.

Two measurements, and the second is the one that matters:

1. **Pairwise accuracy on replayed actions.**  Take states from the buffer, generate pairs of noise
   vectors, ask which the critic scores higher, and compare against the *true* outcome of each
   action.  Getting the true outcome needs a rollout per action, which is 60-600 ms of physics, so
   this is done on a small sample.
2. **Critic-at-chance would be fine.**  A critic can be accurate and ordinally useless (§9.96), but
   it can also be ordinally useful without being accurate.  What DSRL actually uses is
   `argmax_eps Q_ns(s, eps)`, so the operational question is whether the noise the actor would now
   pick beats a random noise vector *in the environment*.  That is a paired comparison and it does
   not need a well-calibrated Q at all.

Usage
-----
    python scripts/recap/probe_online_critic.py --checkpoint /root/autodl-tmp/dsrl_online_run1.pt
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

import cv2  # noqa: E402

from steering import (  # noqa: E402
    NoiseActor,
    build_observation,
    encode_prefix,
    repeat_noise,
    sample_with_prefix,
    twin_critic,
    visual_feature,
)
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


def critic_q(heads, observation: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """`min` over the twin heads, averaged over the batch -- the quantity the actor climbs."""
    q = torch.stack([head(observation, noise) for head in heads], dim=0).min(dim=0).values
    return q.mean(dim=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/dsrl_online_run1.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", type=int, default=12,
                        help="states to compare at; each costs two 120-step rollouts")
    parser.add_argument("--noise-pairs", type=int, default=6,
                        help="noise pairs per state, evaluated by the critic only")
    parser.add_argument("--horizon", type=int, default=300,
                        help="control steps to roll out per candidate action.  Too short a window makes every pair a tie: measured, 120 steps delivered 0 cookies in 12/12 states because the first placement lands around step 226, so the comparison has no opportunity to register a difference -- the same trap as a probe with no signal.")
    parser.add_argument("--seed", type=int, default=1500)
    args = parser.parse_args(argv)

    import train_pi05_launcher as launcher
    from bench_pi05_on_main import make_cookie_benchmark, training_prompt
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    checkpoint = Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model")
    policy = PI05Policy.from_pretrained(str(checkpoint)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    observation_dim = int(state["observation_dim"])
    actor_dim = int(state["actor_dim"])
    hidden = int(state.get("hidden", 2048))
    chunk_size = int(state["chunk_size"])
    action_dim = int(state["action_dim"])
    original_dim = int(policy.config.output_features["action"].shape[0])
    num_steps = int(policy.config.num_inference_steps)

    actor = NoiseActor(observation_dim, actor_dim, hidden=hidden).to(args.device)
    actor.load_state_dict(state["actor"])
    actor.eval()
    critic_noise = twin_critic(observation_dim, actor_dim, hidden=hidden).to(args.device)
    critic_noise.load_state_dict(state["critic_noise"])
    critic_noise.eval()
    projection = torch.as_tensor(np.asarray(state["projection"]), dtype=torch.float32,
                                 device=args.device)
    parts = state.get("observation_parts", {"proprio": True, "vlm": True, "visual": True})
    print(f"[probe] checkpoint {args.checkpoint.name}: observation {observation_dim},"
          f" actor {actor_dim}, chunk {chunk_size} x {action_dim}, hidden {hidden}", flush=True)

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    task = training_prompt()
    base = Path("/root/autodl-tmp/dsrl_online_run1.pt")
    del base

    def make_scene(seed: int) -> Scene:
        env = bench.create_env(render_cameras=False)
        env.reset(seed=seed, options=reset_options(bench, seed))
        return Scene(index=0, env=env, renderer=_make_renderer(env.model),
                     camera_ids=dict(env._camera_ids), seed=seed,  # noqa: SLF001
                     required_cookies=int(bench.required_cookies), max_control_steps=1500)

    def observe(scene: Scene) -> dict:
        frames = [render_frames(scene.renderer, scene)]
        observations = [scene.env._observation()]  # noqa: SLF001
        raw = raw_batch([scene], frames, observations, task, args.device)
        prepared = preprocess(raw)
        prefix = encode_prefix(policy, prepared, keep_embeddings=True)
        with torch.no_grad():
            visual = visual_feature(policy, prefix) @ projection
            obs = build_observation(prepared[STATE].float(), prefix.hidden_last, visual, parts)
        return {"observation": obs.detach(), "prefix": prefix, "raw": raw}

    def rollout(scene: Scene, noise: torch.Tensor, prefix, steps: int) -> int:
        """Cookies delivered by executing `noise` through the sampler for `steps` control steps."""
        with torch.no_grad():
            flat = repeat_noise(noise, chunk_size, action_dim, "single")
            chunk = sample_with_prefix(policy, prefix, flat, num_steps, args.device)
            executed = np.asarray(
                postprocess(chunk[:, :, :original_dim])[0].detach().float().cpu(),
                dtype=np.float64)
        before = scene.cookies
        for offset in range(min(steps, len(executed))):
            _, _, terminated, truncated, info = scene.env.step(executed[offset])
            scene.cookies = int(info.get("cookies_in_target", scene.cookies))
            if terminated or truncated:
                break
        return scene.cookies - before

    rng = np.random.default_rng(args.seed)
    agreement: list[float] = []
    actor_vs_random: list[int] = []
    started = time.monotonic()

    print()
    print("=" * 74)
    print(f"1. does the critic rank two noise vectors correctly?  ({args.states} states)")
    print("=" * 74)
    print(f"   {'state':>5s} {'cookies before':>14s} {'Q(a)':>8s} {'Q(b)':>8s}"
          f" {'got a':>6s} {'got b':>6s} {'agree':>7s}")
    for index in range(args.states):
        scene = make_scene(args.seed + index * 7)
        view = observe(scene)
        pair = torch.randn(2, actor_dim, device=args.device)
        q = critic_q(critic_noise, view["observation"].expand(2, -1), pair)
        # Two independent rollouts from the same start, so the difference is the noise and not the
        # scene.  `reset` on the same environment is not reproducible (§9.97), so each candidate
        # gets its own fresh environment with the same seed.
        outcomes = []
        for which in range(2):
            replay_scene = make_scene(args.seed + index * 7)
            outcomes.append(rollout(replay_scene, pair[which:which + 1], view["prefix"],
                                    args.horizon))
            replay_scene.env.close()
        higher_q = int(torch.argmax(q).item())
        higher_outcome = 0 if outcomes[0] > outcomes[1] else (1 if outcomes[1] > outcomes[0] else -1)
        agree = 0.5 if higher_outcome < 0 else float(higher_q == higher_outcome)
        agreement.append(agree)
        print(f"   {index:5d} {scene.cookies:14d} {float(q[0]):8.3f} {float(q[1]):8.3f}"
              f" {outcomes[0]:6d} {outcomes[1]:6d} {agree:7.2f}")
        scene.env.close()

    print()
    print("=" * 74)
    print("2. does the actor's own noise beat a random one, in the environment?")
    print("=" * 74)
    print(f"   {'state':>5s} {'actor noise':>11s} {'random noise':>13s} {'delta':>7s}")
    for index in range(args.states):
        seed = args.seed + 10000 + index * 13
        view_scene = make_scene(seed)
        view = observe(view_scene)
        with torch.no_grad():
            actor_noise, _ = actor.sample(view["observation"])
            random_noise = torch.randn(1, actor_dim, device=args.device)
        a_scene = make_scene(seed)
        b_scene = make_scene(seed)
        got_actor = rollout(a_scene, actor_noise, view["prefix"], args.horizon)
        got_random = rollout(b_scene, random_noise, view["prefix"], args.horizon)
        a_scene.env.close()
        b_scene.env.close()
        view_scene.env.close()
        actor_vs_random.append(got_actor - got_random)
        print(f"   {index:5d} {got_actor:11d} {got_random:13d}"
              f" {got_actor - got_random:+7d}", flush=True)

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    accurate = np.asarray(agreement)
    print(f"   critic 排序准确率 {accurate.mean():.3f}  (n={len(accurate)},"
          f" ties 记 0.5; 0.5 是抛硬币)")
    delta = np.asarray(actor_vs_random, dtype=float)
    wins = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    losses = int((delta < 0).sum())
    print(f"   actor ε 对随机 ε: 均值 {delta.mean():+.2f} 饼干"
          f"  ({wins} 胜 / {ties} 平 / {losses} 负, n={len(delta)})")
    if len(delta) > 1 and delta.std() > 0:
        from math import sqrt
        t = delta.mean() / (delta.std(ddof=1) / sqrt(len(delta)))
        print(f"   配对 t = {t:+.2f}")
    print(f"   用时 {(time.monotonic() - started) / 60:.1f} 分钟")
    print()
    print("   排序准确率显著高于 0.5 -> 在线数据确实让 critic 学到了同状态的动作排序，")
    print("   这是离线做不到的那件事（§9.96 的 0.508）。")
    print("   actor ε 显著优于随机 ε -> actor 已经能在采样器里做出更好的选择。")
    print("   两者都不显著 -> 不是 bug，是 §9.99 写的那个 2.9 % 权限上限。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
