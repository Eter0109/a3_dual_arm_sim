#!/usr/bin/env python3
"""Can the critic guide the actor at all?  Measured without a single rollout.

Why this replaces the paired-rollout probe
-----------------------------------------
`probe_online_critic.py` compares two noise vectors by executing each for a window and counting
cookies.  Measured, that needs a window of a full episode: 120 steps gave 0 cookies in 20 of 20
comparisons, 300 steps gave 0 in 20 of 20, and the training run only reaches 5-7 cookies per 1,500
steps.  So one comparison costs ~10 minutes and ten states cost 200 minutes -- more than the whole
training budget, for ten data points.

The question that measurement was trying to answer decomposes into two, and only the second needs
rollouts:

  (a) does `Q_ns(s, u)` vary **across u at fixed s** at all?
  (b) when it varies, does the preferred u actually do better?

(a) is a property of the network and the weights, and costs one forward pass per noise vector.  If
`std_u Q_ns(s, u)` is negligible next to the value of one cookie, then (b) cannot be true no matter
what: an actor climbing a flat surface picks noise for numerical reasons, not for reward.  That
makes (a) the gate, and it is free.

The scale that matters
----------------------
The reward is cookies, the discount is 0.99, and a placement anywhere in an episode is worth
between 1 cookie (at the end) and roughly 1/(1-0.99) = 100 cookies (if it happened every step).  A
realistic placement at step t is worth about `0.99^t` at the start, so the quantity to beat is order
**0.1**.  Anything below that in `std_u Q` means the surface is flat to within the precision that
matters.

What else it reports
--------------------
* **The actor's own gradient step size.**  `Q_ns(s, tanh(m + sigma*z))` for z ~ N(0, 1): how far
  the actor's sampling moves the critic, against how far the best-to-worst noise moves it.
* **The offline critics, for reference.**  §9.96 measured v5 and v6 at 0.508 pairwise accuracy; if
  their `std_u Q` is as flat as the online one, then the flatness is the explanation and the
  ranking result is a consequence rather than a separate finding.

Usage
-----
    python scripts/recap/probe_critic_noise_sensitivity.py \
        --online /root/autodl-tmp/dsrl_online_run1.pt \
        --offline /root/autodl-tmp/dsrl_v6.pt
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


def q_over_noise(heads, observation: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """`min` over the twin heads -- what the actor climbs -- for each `(state, noise)` row."""
    q = torch.stack([head(observation, noise) for head in heads], dim=0).min(dim=0).values
    return q


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--online", type=Path,
                        default=Path("/root/autodl-tmp/dsrl_online_run1.pt"))
    parser.add_argument("--offline", type=Path, default=None,
                        help="an offline checkpoint whose critic should be measured the same way")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--noise-samples", type=int, default=256,
                        help="noise vectors per state.  256 is cheap (one matmul) and enough to"
                             " estimate a standard deviation to ~5%%.")
    parser.add_argument("--seed", type=int, default=2100)
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

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    task = training_prompt()

    def observations(count: int):
        """`count` actor observations, one per fresh scene, plus the prefixes they came from."""
        out = []
        for index in range(count):
            seed = args.seed + index * 17
            env = bench.create_env(render_cameras=False)
            env.reset(seed=seed, options=reset_options(bench, seed))
            scene = Scene(index=index, env=env, renderer=_make_renderer(env.model),
                          camera_ids=dict(env._camera_ids), seed=seed,  # noqa: SLF001
                          required_cookies=int(bench.required_cookies),
                          max_control_steps=1500)
            frames = [render_frames(scene.renderer, scene)]
            obs = [env._observation()]  # noqa: SLF001
            raw = raw_batch([scene], frames, obs, task, args.device)
            prepared = preprocess(raw)
            prefix = encode_prefix(policy, prepared, keep_embeddings=True)
            with torch.no_grad():
                visual = visual_feature(policy, prefix) @ torch.as_tensor(
                    np.asarray(torch.load(str(args.online), map_location="cpu",
                                          weights_only=False)["projection"]),
                    dtype=torch.float32, device=args.device)
                vector = build_observation(prepared[STATE].float(), prefix.hidden_last,
                                           visual, {"proprio": True, "vlm": True,
                                                    "visual": True})
            out.append(vector.detach())
            env.close()
        return torch.cat(out, dim=0)

    def measure(label: str, checkpoint: Path) -> None:
        if not checkpoint.exists():
            print(f"   {label}: {checkpoint} not found, skipped")
            return
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        observation_dim = int(state["observation_dim"])
        actor_dim = int(state["actor_dim"])
        hidden = int(state.get("hidden") or state["actor"]["body.net.0.weight"].shape[0])
        actor = NoiseActor(observation_dim, actor_dim, hidden=hidden).to(args.device).eval()
        actor.load_state_dict(state["actor"])
        critic_noise = None
        for key in ("critic_noise", "critic"):
            if key in state:
                candidate = twin_critic(
                    observation_dim,
                    actor_dim if key == "critic_noise" else int(state["chunk_size"]) * int(state["action_dim"]),
                    hidden=hidden).to(args.device).eval()
                try:
                    candidate.load_state_dict(state[key])
                    critic_noise = candidate
                    print(f"   {label}: using '{key}' as the noise critic")
                    break
                except RuntimeError:
                    continue
        if critic_noise is None:
            print(f"   {label}: no noise critic of width {actor_dim} in the checkpoint, skipped")
            return

        vectors = observations(args.states)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        spread = []
        best_shift = []
        for index in range(args.states):
            repeat = vectors[index:index + 1].expand(args.noise_samples, -1)
            noise = torch.randn(args.noise_samples, actor_dim, device=args.device,
                                generator=generator)
            with torch.no_grad():
                q = q_over_noise(critic_noise, repeat, noise)
            spread.append(float(q.std()))
            best_shift.append(float(q.max() - q.min()))
            del repeat, noise, q
        spread_array = np.asarray(spread)
        shift_array = np.asarray(best_shift)
        print(f"   {label}")
        print(f"      std_u Q(s,·)      mean {spread_array.mean():.4f}"
              f"  median {np.median(spread_array):.4f}"
              f"  max {spread_array.max():.4f}")
        print(f"      max-min over u    mean {shift_array.mean():.4f}"
              f"  max {shift_array.max():.4f}")
        # The norm to judge against is the episode score, not an invented threshold.  A cookie is
        # worth 1.0 in Q's units, and the policies measured here place 5-13 cookies per episode, so
        # the fraction below is the *most* the critic could ever gain by choosing noise: the full
        # range of Q over u, as a share of the score.
        for score, note in ((5.0, "this online run, mid-training"),
                            (13.18, "the v9 baseline")):
            share = shift_array.mean() / score * 100
            print(f"      vs a {score:5.2f}-cookie episode ({note}):"
                  f" the whole Q range is {share:4.1f}% of the score")
        print("      -> the critic DOES have structure across noise; the question is whether"
              " that structure is large enough to matter.")
        del actor, critic_noise

    print("=" * 74)
    print("does Q_ns vary across noise at a fixed state?")
    print("=" * 74)
    print(f"   {args.states} states x {args.noise_samples} noise vectors")
    measure("online", args.online)
    print()
    if args.offline is not None:
        measure("offline", args.offline)
    else:
        print("   (pass --offline to measure an offline critic the same way)")

    print()
    print("=" * 74)
    print("读法")
    print("=" * 74)
    print("  `std_u Q` 和 `max-min` 的单位是饼干：一个饼干在 Q 里值 1.0。")
    print("  所以 `max-min / 一集得分` 就是 critic 通过选噪声能拿到的**全部**收益上限。")
    print("  这个比例如果只有几个百分点，那 §9.96 的 0.508 和这里的小曲率是同一件事：")
    print("  排序需要曲率，而曲率本来就小，因为 DSRL 对这套采样器的权限本来就小（§9.95 量到 2.9%）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
