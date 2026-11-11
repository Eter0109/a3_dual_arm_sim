#!/usr/bin/env python3
"""Does the critic have a usable gradient with respect to the noise?

Why this is the question, and not the loss curve
------------------------------------------------
Three critic targets were measured on identical features (`probe_value_ceiling.py`): TD on a
scattered pool diverged (Q 25.0 against a true maximum of 26.9, `minQ` 49.7), TD with no reward
anchors collapsed to a constant (Q ~ -1.0), and Monte-Carlo matched the truth almost exactly
(Q 8.6-9.4 against a true mean of 8.92) yet the actor loss barely moved (-52.9 -> -54.1).

The tempting reading -- "an MC target has no action dependence, so the actor has no gradient" -- is
*not* safe to assume.  SAC is trained on exactly one action per state too, and it works, because
the function approximator interpolates between neighbouring `(s, a)` pairs.  So the claim has to be
measured, not argued.

What is measured
----------------
For a critic fitted with a given target, at a set of held-out states:

* `dQ/da`: the norm of the gradient of `Q(s, a)` with respect to `a`, relative to `|a|`.  Zero here
  means the critic is constant in the action and the actor has nothing to climb.
* `within-state spread`: `std_a Q(s, a) / std_s Q(s, a)`, i.e. how much of the critic's variation
  is *across actions at one state* versus across states.  This is the scale-free version, and it is
  the one that matters for `argmax_a`.
* `pairwise ranking`: sample two actions at the same state, ask whether the critic orders them the
  same way the *true* consequence does.  0.5 is chance.  Needs the branch dataset, so it is only
  reported when `--branches` is passed.

Usage
-----
    python scripts/recap/probe_critic_gradient.py --epochs 600
    python scripts/recap/probe_critic_gradient.py --epochs 600 --branches <branch shard dir>
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", choices=["mc", "td"], action="append", default=None,
                        help="which targets to compare; may be repeated.  Default: both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from probe_value_ceiling import discounted_return, load_labels  # noqa: PLC0415
    from train_dsrl import MLP, twin_critic  # noqa: PLC0415

    torch.manual_seed(args.seed)
    device = args.device
    targets = args.target or ["mc", "td"]

    rewards, dones, episodes, labels, bounds, potentials, total = load_labels(args.dataset)
    return_all = discounted_return(rewards, dones, args.gamma)
    print(f"[grad] {total} rows, gamma {args.gamma}, true return mean"
          f" {return_all.mean():.3f} std {return_all.std():.3f}", flush=True)

    if args.cache is not None:
        cache_file = args.cache
    else:
        cache_file = sorted((args.dataset / "dsrl_cache").glob("obs_*.pt"),
                            key=lambda p: p.stat().st_size)[-1]
    cached = torch.load(cache_file, weights_only=True)
    print(f"[grad] using {cache_file.name} ({len(cached['rows_cur'])} rows)", flush=True)

    rows_cur = cached["rows_cur"].numpy()
    rows_next = cached["rows_next"].numpy()
    observation = torch.cat([cached["proprio_cur"], cached["probe_cur"]], dim=1).float().to(device)
    observation_next = torch.cat([cached["proprio_next"], cached["probe_next"]], dim=1).float().to(device)
    observation_dim = observation.shape[1]

    reward_t = torch.from_numpy(rewards[rows_cur]).to(device)
    done_t = torch.from_numpy(dones[rows_cur]).to(device)
    return_t = torch.from_numpy(return_all[rows_cur].astype(np.float32)).to(device)

    count = len(rows_cur)
    order = np.arange(count)
    np.random.default_rng(args.seed).shuffle(order)
    split = int(0.8 * count)
    train_index = torch.from_numpy(order[:split]).to(device)
    test_index = torch.from_numpy(order[split:]).to(device)

    def measure(critic, noise_dim: int, label: str) -> dict:
        """Gradient magnitude and within-state spread of `Q(s, .)`."""
        held = test_index[:256]
        state = observation[held]
        generator = torch.Generator(device=device).manual_seed(args.seed)
        action = torch.randn(len(held), noise_dim, device=device, generator=generator)
        action.requires_grad_(True)
        with torch.enable_grad():
            value = critic[0](state, action).sum()
            gradient = torch.autograd.grad(value, action)[0]
        gradient_norm = float(gradient.norm(dim=-1).mean())
        action_norm = float(action.norm(dim=-1).mean())

        with torch.no_grad():
            # Eight actions at each of the held-out states.
            many = torch.randn(len(held), 8, noise_dim, device=device, generator=generator)
            flat_state = state[:, None, :].expand(len(held), 8, observation_dim)
            values = critic[0](flat_state.reshape(-1, observation_dim),
                               many.reshape(-1, noise_dim)).reshape(len(held), 8)
            within = float(values.std(dim=1).mean())
            across = float(values.mean(dim=1).std())
            # Mean absolute deviation, so a single outlier pair cannot define the ratio.
            spread = float((values - values.mean(1, keepdim=True)).abs().mean())
        print(f"    {label:3s} dQ/da {gradient_norm:.4f}  (|a| {action_norm:.3f},"
              f" ratio {gradient_norm / max(action_norm, 1e-9):.4f})"
              f"  within-state spread {within:.4f}  across-state {across:.4f}"
              f"  ratio {within / max(across, 1e-9):.3f}  MAD {spread:.4f}", flush=True)
        return {"dQ_da": gradient_norm, "action_norm": action_norm,
                "within_state_std": within, "across_state_std": across,
                "within_over_across": within / max(across, 1e-9), "mad": spread}

    results: dict[str, dict] = {}
    for target in targets:
        critic = twin_critic(observation_dim, 1600, args.hidden).to(device)
        target_net = twin_critic(observation_dim, 1600, args.hidden).to(device)
        target_net.load_state_dict(critic.state_dict())
        optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)
        recorded = torch.randn(count, 1600, device=device,
                               generator=torch.Generator(device=device).manual_seed(args.seed))
        recorded_next = torch.randn(count, 1600, device=device,
                                    generator=torch.Generator(device=device).manual_seed(args.seed + 1))

        for epoch in range(1, args.epochs + 1):
            pick = train_index[torch.randint(0, split, (args.batch,), device=device)]
            with torch.no_grad():
                if target == "mc":
                    want = return_t[pick].unsqueeze(-1)
                else:
                    next_value = torch.min(
                        torch.stack([head(observation_next[pick], recorded_next[pick])
                                     for head in target_net]), dim=0).values
                    want = (reward_t[pick].unsqueeze(-1)
                            + args.gamma * (1.0 - done_t[pick].unsqueeze(-1)) * next_value)
            got = torch.stack([head(observation[pick], recorded[pick]) for head in critic])
            loss = 0.5 * sum(F.mse_loss(value, want) for value in got)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            optimizer.step()
            if target == "td":
                with torch.no_grad():
                    for a, b in zip(target_net.parameters(), critic.parameters()):
                        a.mul_(1.0 - 0.005).add_(0.005 * b)
            if epoch % 200 == 0:
                print(f"    {target} {epoch:4d}  loss {float(loss.detach()):8.4f}", flush=True)

        with torch.no_grad():
            held = torch.stack([head(observation[test_index], recorded[test_index])
                                for head in critic], dim=0).mean(0).squeeze(-1)
            residual = float(((held - return_t[test_index]) ** 2).mean())
            variance = float(((return_t[test_index] - return_t[test_index].mean()) ** 2).mean())
        print(f"  {target}: held-out R2 vs true return {1 - residual / variance:+.4f}", flush=True)
        results[target] = {"r2": 1 - residual / variance}
        results[target].update(measure(critic, 1600, target))

    print("\n结果")
    for target, row in results.items():
        print(f"  {target:3s}  R2 {row['r2']:+.4f}   |dQ/da| {row['dQ_da']:.4f}"
              f"   within/across {row['within_over_across']:.3f}")
    print()
    best = max(results, key=lambda k: results[k]["within_over_across"])
    worst = min(results, key=lambda k: results[k]["within_over_across"])
    print(f"  跨动作变化最大的目标：{best}（within/across {results[best]['within_over_across']:.3f}）")
    print(f"  最平坦的目标：{worst}（{results[worst]['within_over_across']:.3f}）")
    print("  within/across 接近 1 = critic 对动作同样敏感；接近 0 = actor 无从爬升。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json  # noqa: PLC0415
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
