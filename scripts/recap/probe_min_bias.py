#!/usr/bin/env python3
"""Is `min` over twin critics the source of the initial negative Q, and is it self-reinforcing?

Where this stands
-----------------
`--reward-scale 30` turns `q` from -4.31 into +3.27 in 20 rounds, which shows the collapse is a
matter of the reward being too small next to the bootstrap term.  But it does not explain why `q`
was already **-4.31 by round 10, on a round where every reward in the batch was zero**.

The arithmetic says a zero-reward update cannot cross zero on its own:

    target = 0 + gamma * min_i Q_target(s', a')
    if Q > 0 everywhere, target > 0, so Q is only ever pulled toward 0, never through it.

So something has to be below zero already.  Two candidates, both structural rather than accidental:

1. **`min` over two independently initialised heads is biased low.**  For two roughly Gaussian
   heads, `E[min(a, b)] = mu - sigma / sqrt(pi) ~= mu - 0.56 sigma`.  With the measured init
   `std 0.31` that is only about -0.17, so on its own it is too small -- but it is applied at every
   bootstrap, so it compounds into whatever the fixed point is.
2. **The `min` selects the more negative head, then both heads are fitted to that same target**, so
   the two heads become correlated in the negative direction and the bias feeds itself.

This probe measures both.  It is deliberately small: the same two-head critic, a sparse reward, and
a comparison of `min`, `mean`, and a single head.  If `min` is the cause, `mean` will stay positive
where `min` goes negative, and the gap will grow rather than settle.

Usage
-----
    python scripts/recap/probe_min_bias.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering import soft_update, twin_critic  # noqa: E402


def build(n: int, obs_dim: int, action_dim: int, device: str, seed: int, nonzero: int) -> dict:
    """`n` distinct states with distinct actions, a sparse reward of +1 on `nonzero` of them."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=gen) * 0.5
    chunk = torch.randn(n, action_dim, generator=gen) * 0.34
    reward = torch.zeros(n)
    if nonzero > 0:
        reward[torch.randperm(n, generator=gen)[:nonzero]] = 1.0
    return {
        "obs": obs.to(device),
        "chunk": chunk.to(device),
        "reward": reward.to(device),
        "done": torch.zeros(n, device=device),
    }


def run(label: str, combine: str, data: dict, args, device: str) -> list[float]:
    torch.manual_seed(args.seed)
    obs_dim = data["obs"].shape[1]
    act_dim = data["chunk"].shape[1]
    critic = twin_critic(obs_dim, act_dim, hidden=args.hidden).to(device)
    target = twin_critic(obs_dim, act_dim, hidden=args.hidden).to(device)
    target.load_state_dict(critic.state_dict())
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    n = data["obs"].shape[0]

    history: list[float] = []
    head_gap: list[float] = []
    for step in range(1, args.updates + 1):
        index = torch.randint(0, n, (args.batch,), device=device)
        obs, chunk = data["obs"][index], data["chunk"][index]
        with torch.no_grad():
            stacked = torch.stack([h(data["obs"][index], data["chunk"][index]) for h in target], 0)
            if combine == "min":
                next_q = stacked.min(0).values
            elif combine == "mean":
                next_q = stacked.mean(0)
            else:  # single head, no aggregation
                next_q = stacked[0]
            tgt = (data["reward"][index].unsqueeze(-1)
                   + args.gamma * (1.0 - data["done"][index].unsqueeze(-1)) * next_q)
        stacked_q = torch.stack([h(obs, chunk) for h in critic], 0)
        loss = 0.5 * sum(F.mse_loss(v, tgt) for v in stacked_q)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optimizer.step()
        soft_update(target, critic, args.tau)

        if step in args.report_at:
            with torch.no_grad():
                full = torch.stack([h(data["obs"], data["chunk"]) for h in critic], 0)
                history.append(float(full.mean()))
                head_gap.append(float((full[0] - full[1]).mean()))
    print(f"   {label:12s} q: " + " -> ".join(f"{v:+7.3f}" for v in history))
    print(f"   {'':12s} head0-head1: " + " -> ".join(f"{v:+7.3f}" for v in head_gap))
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obs-dim", type=int, default=2320)
    parser.add_argument("--action-dim", type=int, default=1600)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--nonzero", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-at", type=int, nargs="+", default=[1, 10, 50, 100, 200, 400])
    args = parser.parse_args(argv)

    device = args.device
    print(f"   {args.states} states, {args.nonzero} with reward 1"
          f" ({args.nonzero / args.states * 100:.0f}% non-zero)")
    print(f"   expected reward per sample: {args.nonzero / args.states:.4f}")
    print()

    data = build(args.states, args.obs_dim, args.action_dim, device, args.seed, args.nonzero)
    print(f"   reported at updates {args.report_at}")
    print()
    results = {}
    for combine in ("min", "mean", "single"):
        results[combine] = run(combine, combine, data, args, device)
        print()

    print("=" * 78)
    print("读法")
    print("=" * 78)
    mn, me = results["min"], results["mean"]
    if mn[-1] < 0 <= me[-1]:
        print("  min 变负而 mean 保持非负 -> `min` 的双头下偏就是成因。")
        print("  参考配置同样是 min，但它的奖励是每步 -1（稠密），偏置被奖励量级淹没；")
        print("  这里奖励 96% 为零，于是偏置主导。这不是 bug，是参数与奖励密度不匹配。")
    elif mn[-1] < 0 and me[-1] < 0:
        print("  min 和 mean 都变负 -> 不是聚合方式的问题，继续查 replay 内容。")
    else:
        print("  min 也没变负 -> 与线上运行不一致，需检查本探针与训练器的差异。")
    print()
    print(f"  min 最终 {mn[-1]:+.3f}   mean 最终 {me[-1]:+.3f}"
          f"   差 {me[-1] - mn[-1]:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
