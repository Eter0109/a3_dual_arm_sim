#!/usr/bin/env python3
"""Two checks that decide whether the "+4.7 cookies" result means anything.

The result being checked
------------------------
`probe_actor_vs_random.py` found the TD critic preferring the trained actor's chunk over a
random-noise chunk by **+4.73 cookies** (t +4.71, 14 win / 2 lose), and the distilled critic
agreeing (+4.54).  Taken at face value that is a 36 % improvement on the 13.18 baseline -- far larger
than the 2.2 % authority `probe_critic_noise_sensitivity.py` measured, and larger than the ~2.9 %
action authority `probe_steering_effect` measured before it.

Three of the sixteen random-noise TD scores are **negative**: -5.636, -2.220, -1.936, -1.666,
-1.492, -1.349, -0.507, -0.158.  That is impossible for a correct value function on this task:

    reward  = cookies_in_target[t+1] - cookies_in_target[t]  >= 0
    return  = sum of rewards over the episode                >= 0
    Q(s,a)  >= 0        for every state and every action

So those eight numbers are not evaluations, they are extrapolations.  A critic trained only on
behaviour chunks has never seen what `sampler(random noise)` produces, and its answer there is
whatever the MLP happens to continue to.

What this script measures
-------------------------
1. **How far outside the behaviour distribution the random-noise chunk is.**  Compares the sampled
   chunks against each other and against the actor's own posterior mean, in action units, and
   reports the commanded joint range.  If a random-noise chunk commands a joint far outside anything
   the arm does, the critic's score for it is meaningless.
2. **`std_u Q` on the *current* checkpoint.**  §9.102 measured 0.054 / range 0.295 on `run1`, which
   had trained for 200 rounds and whose `critic` was still near zero -- so `critic_noise` was
   distilling a target with no range to speak of.  If the same measurement on a checkpoint that has
   trained 5x longer gives a much larger spread, then the 2.2 % figure was an artifact of measuring
   early, and it must be retracted rather than quoted.

Both are cheap: one denoise for (1), evaluation-only for (2).

Usage
-----
    python scripts/recap/probe_score_sanity.py --states 16
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


def load(checkpoint: Path, device: str):
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    observation_dim = int(state["observation_dim"])
    actor_dim = int(state["actor_dim"])
    hidden = int(state.get("hidden") or state["actor"]["body.net.0.weight"].shape[0])
    actor = NoiseActor(observation_dim, actor_dim, hidden=hidden).to(device).eval()
    actor.load_state_dict(state["actor"])
    critic_noise = twin_critic(observation_dim, actor_dim, hidden=hidden).to(device).eval()
    critic_noise.load_state_dict(state["critic_noise"])
    critic = twin_critic(observation_dim, int(state["chunk_size"]) * int(state["action_dim"]),
                         hidden=hidden).to(device).eval()
    critic.load_state_dict(state["critic"])
    return state, actor, critic, critic_noise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/dsrl_online_run3.pt"))
    parser.add_argument("--compare", type=Path, default=Path("/root/autodl-tmp/dsrl_online_run1.pt"),
                        help="an earlier checkpoint, to test whether the noise-critic spread depends"
                             " on how long the critic has trained")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--noise-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3300)
    parser.add_argument("--action-magnitude", type=float, default=1.5)
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

    state, actor, critic, critic_noise = load(args.checkpoint, args.device)
    actor_dim = int(state["actor_dim"])
    projection = torch.as_tensor(np.asarray(state["projection"]), dtype=torch.float32,
                                 device=args.device)
    parts = state.get("observation_parts", {"proprio": True, "vlm": True, "visual": True})
    extra = state.get("online") or {}
    print(f"[probe] {args.checkpoint.name}: rounds {extra.get('rounds')},"
          f" transitions {extra.get('transitions')}, episodes {extra.get('episodes')}")

    def q_td(observation, chunk):
        padded = torch.zeros(chunk.shape[0], chunk_size, action_dim, device=args.device)
        padded[:, :, :original_dim] = chunk[:, :, :original_dim]
        scaled = (padded.reshape(chunk.shape[0], -1) / args.action_magnitude).float()
        return torch.stack([h(observation, scaled) for h in critic], 0).min(0).values.squeeze(-1)

    def q_ns(observation, noise):
        return torch.stack([h(observation, noise) for h in critic_noise], 0).min(0).values.squeeze(-1)

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    task = training_prompt()

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
            observations.append(build_observation(prepared[STATE].float(), prefix.hidden_last,
                                                  visual, parts).detach())
        prefixes.append(prefix)
        env.close()
    observation_batch = torch.cat(observations, 0)

    # =========================================================================================
    print()
    print("=" * 78)
    print("1. is the random-noise chunk inside the distribution the critic was trained on?")
    print("=" * 78)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.no_grad():
        actor_noise, _ = actor.sample(observation_batch)
        random_noise = torch.randn(args.states, actor_dim, device=args.device, generator=generator)
        mean_noise = torch.tanh(actor.forward(observation_batch))
    sampled = {}
    for label, noise in (("actor", actor_noise), ("random", random_noise), ("mean", mean_noise)):
        with torch.no_grad():
            flat = repeat_noise(noise, chunk_size, action_dim, "single")
            out = torch.cat([
                sample_with_prefix(policy, prefixes[i], flat[i:i + 1], num_steps, args.device)
                for i in range(args.states)
            ], 0)
        sampled[label] = out[:, :, :original_dim]

    print(f"   commanded joint range over the 16 commanded dims, per noise source:")
    for label, chunk in sampled.items():
        values = chunk.cpu().numpy()
        print(f"      {label:7s} min {values.min():+7.3f}  max {values.max():+7.3f}"
              f"  sd {values.std():6.3f}  mean|a| {np.abs(values).mean():6.3f}")

    print()
    print("   distance between the three chunks, per state (mean over the 16 commanded dims):")
    print(f"      {'state':>5} {'|actor-random|':>15} {'|actor-mean|':>13} {'|mean-random|':>14}")
    dist_ar, dist_am, dist_mr = [], [], []
    for index in range(args.states):
        a = sampled["actor"][index].cpu().numpy()
        r = sampled["random"][index].cpu().numpy()
        m = sampled["mean"][index].cpu().numpy()
        ar, am, mr = np.abs(a - r).mean(), np.abs(a - m).mean(), np.abs(m - r).mean()
        dist_ar.append(ar)
        dist_am.append(am)
        dist_mr.append(mr)
        print(f"      {index:5d} {ar:15.4f} {am:13.4f} {mr:14.4f}")
    print(f"      {'mean':>5} {np.mean(dist_ar):15.4f} {np.mean(dist_am):13.4f}"
          f" {np.mean(dist_mr):14.4f}")
    print()
    print("   for reference: consecutive control steps differ by 0.0021 in action and steps 16"
          " apart by 0.0250")
    print("   (measured earlier in this project), and `max_joint_step_rad` clips every command to a"
          " small per-step delta.")

    # =========================================================================================
    print()
    print("=" * 78)
    print("2. are the critic's scores possible?  (Q must be >= 0: the reward is a cookie count)")
    print("=" * 78)
    with torch.no_grad():
        td = {label: q_td(observation_batch, chunk).cpu().numpy() for label, chunk in sampled.items()}
    for label, values in td.items():
        negative = int((values < 0).sum())
        print(f"   TD Q for {label:7s}: {negative:2d}/{args.states} negative"
              f"  (impossible; a negative value is extrapolation)")
    print()
    print("   a random-noise chunk scored -5.6 means the critic was asked about a region it never"
          " saw; the number is the MLP's continuation, not a value.")

    # =========================================================================================
    print()
    print("=" * 78)
    print("3. std_u Q on this checkpoint, vs the earlier one")
    print("=" * 78)

    def spread_of(checkpoint: Path) -> None:
        if not checkpoint.exists():
            print(f"   {checkpoint.name}: not found, skipped")
            return
        st, _, crit, cn = load(checkpoint, args.device)
        if int(st["observation_dim"]) != observation_batch.shape[1]:
            print(f"   {checkpoint.name}: observation dim {st['observation_dim']} differs, skipped")
            return
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        spreads, ranges = [], []
        for index in range(args.states):
            repeat = observation_batch[index:index + 1].expand(args.noise_samples, -1)
            noise = torch.randn(args.noise_samples, actor_dim, device=args.device, generator=gen)
            with torch.no_grad():
                q = torch.stack([h(repeat, noise) for h in cn], 0).min(0).values.squeeze(-1)
            spreads.append(float(q.std()))
            ranges.append(float(q.max() - q.min()))
            del repeat, noise, q
        rounds = (st.get("online") or {}).get("rounds")
        print(f"   {checkpoint.name:32s} rounds {str(rounds):>5s}"
              f"  std_u Q {np.mean(spreads):7.4f}  max-min {np.mean(ranges):7.4f}"
              f"  ({np.mean(ranges) / 13.18 * 100:5.1f}% of the 13.18 baseline)")
        del crit, cn

    spread_of(args.checkpoint)
    spread_of(args.compare)

    print()
    print("=" * 78)
    print("结论怎么读")
    print("=" * 78)
    print("  如果 random chunk 的动作尺度远超策略的正常步长、且 TD 打出负 Q：")
    print("    -> `actor 比 random 好 +4.7` 这个数字无效，random 那一臂是外推。")
    print("  如果 std_u Q 在训练更久的 checkpoint 上明显更大：")
    print("    -> §9.102 的 2.2% 是在欠训练 checkpoint 上测的，必须撤回而不是引用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
