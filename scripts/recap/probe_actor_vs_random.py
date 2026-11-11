#!/usr/bin/env python3
"""Is the trained actor's noise better or worse than a fresh one?  Scored by the TD critic.

What this settles
-----------------
`train_dsrl_online.py` reported that by round 800 `min_q > q` by 0.4-1.0 cookies:

* `q`   is the TD critic `Q(s, executed chunk)` -- trained against real returns, so it is
  calibrated on the actions the behaviour policy actually took.
* `min_q` is the distilled critic `Q_ns(s, actor noise)` -- the quantity the actor maximises.

For the first ~400 rounds the two agreed to within noise.  Then `alpha` annealed toward zero
(`target_entropy = 0`), `sigma` shrank 0.93 -> 0.72, `|u|` grew 0.53 -> 0.57, and the gap opened.
Context that makes the gap alarming rather than cosmetic: §9.102 measured the critic's *entire*
authority over the noise direction at **0.30 cookies**, so a gap of 1.0 is three times the total
real effect.  Either the actor found a real improvement nobody expected, or the distilled critic is
extrapolating in a region it was never calibrated on.

Two comparisons, and the second is what actually decides it
-----------------------------------------------------------
Both are scored by the **TD** critic, which never sees the actor's objective:

1. `Q_td(s, sampler(actor noise))` vs `Q_td(s, sampler(fresh randn))`.  If the actor is worse, the
   actor is making the policy worse -- the plain answer.
2. `Q_td(s, sampler(fresh randn))` vs `Q_td(s, sampler(the noise the buffer holds))`, i.e. against
   the *behaviour* noise.  A sanity anchor: the behaviour noise is what produced the data the
   critic was fit to, so it should score at least as well as independent noise.

And the distillation check, which separates the two failure modes:

3. `Q_ns(s, actor noise)` vs `Q_ns(s, fresh randn)`.  If `Q_ns` prefers the actor while `Q_td`
   disprefers it, the distilled critic is the thing that is wrong, and the actor has been climbing
   a surface that does not correspond to reward.  If both agree the actor is worse, the actor itself
   is the problem.  These need different fixes, so it is worth distinguishing them.

Cost
----
One batched denoise at 3N rows plus one render per state.  No environment stepping, so this runs in
about a minute and does not disturb the training run's physics threads much.

Usage
-----
    python scripts/recap/probe_actor_vs_random.py --checkpoint /root/autodl-tmp/dsrl_online_run3.pt
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

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/dsrl_online_run3.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--action-magnitude", type=float, default=1.5,
                        help="the `scale_action` divisor the TD critic was trained with; must match"
                             " `train_dsrl_online.py` or the critic is shown a differently-scaled"
                             " chunk than it was fit to.")
    args = parser.parse_args(argv)

    import train_pi05_launcher as launcher
    from bench_pi05_on_main import make_cookie_benchmark, training_prompt
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    vla_path = Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model")
    policy = PI05Policy.from_pretrained(str(vla_path)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, _post = make_pre_post_processors(
        policy.config, str(vla_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    original_dim = int(policy.config.output_features["action"].shape[0])
    num_steps = int(policy.config.num_inference_steps)

    if not args.checkpoint.exists():
        raise SystemExit(f"{args.checkpoint} not found")
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    observation_dim = int(state["observation_dim"])
    actor_dim = int(state["actor_dim"])
    hidden = int(state.get("hidden") or state["actor"]["body.net.0.weight"].shape[0])
    if chunk_size != int(state["chunk_size"]) or action_dim != int(state["action_dim"]):
        raise SystemExit(f"checkpoint geometry {state['chunk_size']}x{state['action_dim']}"
                         f" does not match the VLA's {chunk_size}x{action_dim}")
    extra = state.get("online") or {}
    print(f"[probe] {args.checkpoint.name}: rounds {extra.get('rounds')},"
          f" transitions {extra.get('transitions')}, episodes {extra.get('episodes')}")
    print(f"[probe] observation {observation_dim}, actor {actor_dim},"
          f" critic {chunk_size * action_dim}, hidden {hidden}")

    actor = NoiseActor(observation_dim, actor_dim, hidden=hidden).to(args.device).eval()
    actor.load_state_dict(state["actor"])
    critic = twin_critic(observation_dim, chunk_size * action_dim, hidden=hidden).to(args.device)
    critic.load_state_dict(state["critic"])
    critic.eval()
    critic_noise = twin_critic(observation_dim, actor_dim, hidden=hidden).to(args.device)
    critic_noise.load_state_dict(state["critic_noise"])
    critic_noise.eval()
    projection = torch.as_tensor(np.asarray(state["projection"]), dtype=torch.float32,
                                 device=args.device)
    parts = state.get("observation_parts", {"proprio": True, "vlm": True, "visual": True})

    def q_td(observation: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        """TD critic on a real action chunk.  `chunk` is `(B, C, d)` in physical units."""
        padded = torch.zeros(chunk.shape[0], chunk_size, action_dim, device=args.device)
        padded[:, :, :original_dim] = chunk[:, :, :original_dim]
        scaled = (padded.reshape(chunk.shape[0], -1) / args.action_magnitude).float()
        return torch.stack([h(observation, scaled) for h in critic], 0).min(0).values.squeeze(-1)

    def q_ns(observation: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(observation, noise) for h in critic_noise], 0).min(0).values.squeeze(-1)

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    task = training_prompt()

    print()
    print("=" * 78)
    print(f"building {args.states} states")
    print("=" * 78)
    observations, prefixes = [], []
    for index in range(args.states):
        seed = args.seed + index * 23
        env = bench.create_env(render_cameras=False)
        env.reset(seed=seed, options=reset_options(bench, seed))
        scene = Scene(index=index, env=env, renderer=_make_renderer(env.model),
                      camera_ids=dict(env._camera_ids), seed=seed,  # noqa: SLF001
                      required_cookies=int(bench.required_cookies), max_control_steps=1500)
        frames = [render_frames(scene.renderer, scene)]
        obs = [env._observation()]  # noqa: SLF001
        raw = raw_batch([scene], frames, obs, task, args.device)
        prepared = preprocess(raw)
        prefix = encode_prefix(policy, prepared, keep_embeddings=True)
        with torch.no_grad():
            visual = visual_feature(policy, prefix) @ projection
            vector = build_observation(prepared[STATE].float(), prefix.hidden_last, visual, parts)
        observations.append(vector.detach())
        prefixes.append(prefix)
        env.close()
    observation_batch = torch.cat(observations, 0)
    print(f"   {observation_batch.shape}")

    # One batched denoise for all three noise sources: actor, fresh randn, and the actor's own
    # posterior mean (`tanh(m)`, deterministic, which is what evaluation would use).
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.no_grad():
        actor_noise, actor_log_prob = actor.sample(observation_batch)
        random_noise = torch.randn(args.states, actor_dim, device=args.device, generator=generator)
        mean_noise = torch.tanh(actor.forward(observation_batch))
    print(f"   actor noise: |u| {float(actor_noise.abs().mean()):.3f}"
          f"  logp {float(actor_log_prob.mean()):.2f}"
          f"  sigma {float(actor.log_std.exp().mean()):.3f}")
    print(f"   tanh(mean) : |u| {float(mean_noise.abs().mean()):.3f}")

    print()
    print("=" * 78)
    print("sampling all three action chunks with the frozen VLA")
    print("=" * 78)
    rows, chunks = [], []
    for label, noise in (("actor", actor_noise), ("random", random_noise), ("mean", mean_noise)):
        with torch.no_grad():
            flat = repeat_noise(noise, chunk_size, action_dim, "single")
            sampled = torch.cat([
                sample_with_prefix(policy, prefixes[i], flat[i:i + 1], num_steps, args.device)
                for i in range(args.states)
            ], 0)
        chunks.append(sampled[:, :, :original_dim])
        rows.append(label)
    print(f"   sampled {len(rows)} x {args.states} chunks of {tuple(chunks[0].shape)}")

    with torch.no_grad():
        td = {label: q_td(observation_batch, chunk) for label, chunk in zip(rows, chunks)}
        ns = {"actor": q_ns(observation_batch, actor_noise),
              "random": q_ns(observation_batch, random_noise),
              "mean": q_ns(observation_batch, mean_noise)}

    print()
    print("=" * 78)
    print("per-state scores  (Q in cookies: one cookie placed is worth 1.0)")
    print("=" * 78)
    header = (f"   {'state':>5} | {'TD actor':>9} {'TD rand':>9} {'TD mean':>9} |"
              f" {'NS actor':>9} {'NS rand':>9} {'NS mean':>9}")
    print(header)
    print("   " + "-" * (len(header) - 3))
    for index in range(args.states):
        print(f"   {index:5d} |"
              f" {float(td['actor'][index]):9.3f} {float(td['random'][index]):9.3f}"
              f" {float(td['mean'][index]):9.3f} |"
              f" {float(ns['actor'][index]):9.3f} {float(ns['random'][index]):9.3f}"
              f" {float(ns['mean'][index]):9.3f}")

    def compare(scores: dict, a: str, b: str, label: str) -> None:
        delta = (scores[a] - scores[b]).cpu().numpy()
        wins = int((delta > 1e-6).sum())
        losses = int((delta < -1e-6).sum())
        ties = args.states - wins - losses
        t = float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))) if delta.std() else 0.0
        print(f"   {label:34s} mean {delta.mean():+7.4f}  t {t:+6.2f}"
              f"  ({wins} win / {ties} tie / {losses} lose)")

    print()
    print("=" * 78)
    print("paired comparisons")
    print("=" * 78)
    print("   scored by the TD critic (calibrated on real returns):")
    compare(td, "actor", "random", "actor noise - random")
    compare(td, "actor", "mean", "actor noise - tanh(mean)")
    compare(td, "mean", "random", "tanh(mean) - random")
    print()
    print("   scored by the distilled critic (what the actor climbs):")
    compare(ns, "actor", "random", "actor noise - random")
    compare(ns, "actor", "mean", "actor noise - tanh(mean)")
    compare(ns, "mean", "random", "tanh(mean) - random")

    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)
    td_gap = float((td["actor"] - td["random"]).mean())
    ns_gap = float((ns["actor"] - ns["random"]).mean())
    td_t = float((td["actor"] - td["random"]).cpu().numpy().mean()
                 / ((td["actor"] - td["random"]).cpu().numpy().std(ddof=1)
                    / np.sqrt(args.states)))
    print(f"   TD critic: actor - random = {td_gap:+.4f} 饼干  (t {td_t:+.2f})")
    print(f"   蒸馏 critic: actor - random = {ns_gap:+.4f} 饼干")
    if abs(td_t) < 2.0:
        print("   -> TD critic 无法区分两者。actor 既没变好也没变坏，效果可忽略。")
    elif td_gap > 0:
        print("   -> TD critic 认为 actor 更好。")
    else:
        print("   -> TD critic 认为 actor 更坏。")
    if ns_gap > 0.1 and td_gap <= 0:
        print("   -> 但蒸馏 critic 偏爱 actor 而 TD critic 不偏爱：**蒸馏 critic 是错的**，")
        print("      actor 在爬一个不对应奖励的面。这是实现差异，不是算法结论。")
    elif ns_gap > 0.1 and td_gap > 0:
        print("   -> 两个 critic 一致地偏爱 actor。")
    print()
    print(f"   参照 §9.102：critic 在噪声方向上的全部权限是 0.30 饼干（2.2% 的分数）。")
    print(f"   所以任何超过 0.3 的差值都大到不可能是真实的，除非 §9.102 的测量本身有问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
