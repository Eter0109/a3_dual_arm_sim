#!/usr/bin/env python3
"""Why does the TD critic converge to a *negative* value on a task with positive rewards?

The observation being explained
-------------------------------
On the audit run (`--max-control-steps 320 --execute 16`, so 20 RL steps an episode), every
episode returned **+2.75 cookies** and every one of 26 logged rounds had a positive mean reward.
Yet the log reported `q = -3.5`, `target_q = -3.2`, tracking each other.

Arithmetic that rules out "noise": the critic's initialisation is
`mean -0.183, std 0.312, min -1.061` (measured), so -3.5 is ten standard deviations away.  It is
not initialisation.

Arithmetic that shows it is a *fixed point*, not a drift: the target is
`r + gamma * (1 - done) * min_i Q_target(s', a')`.  With `min Q_target ~ -3.3`, `gamma = 0.99` and
`r ~ 0.11`, the target is `0.11 - 2.86 = -2.75` -- about what was observed.  So the negative value
reproduces itself.  The question is what started it.

What this probe does
--------------------
It removes the environment, the VLA and the samplers from the picture and runs the trainer's own
update rules on a synthetic replay whose value is known exactly:

    every state identical, every action identical, reward = +1 every step, gamma = 0.99
    => the only correct answer is Q = 1 / (1 - 0.99) = 100, reached by bootstrapping

If this diverges to a negative number, the bug is in the update rules and can be read off directly.
If it converges to 100, the update rules are sound and the negative value must come from the data
the trainer feeds them -- which narrows the search to the replay contents.

Two further cases separate the two candidate causes:

* `--done-every N` puts a terminating transition every N steps, which caps the correct answer at
  `(1 - gamma^N) / (1 - gamma)`.  This checks that the `done` mask is applied as intended.
* `--negative-init` starts the critic at a large negative value, to see whether TD can climb out.
  If it cannot, the trainer is stuck in whatever basin the first updates fall into.

Usage
-----
    python scripts/recap/probe_td_fixed_point.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from steering import NoiseActor, soft_update, twin_critic  # noqa: E402


def make_data(obs_dim: int, action_dim: int, size: int, done_every: int,
              device: str) -> dict[str, torch.Tensor]:
    """A constant transition repeated `size` times: same state, same action, reward +1.

    `done` marks every `done_every`-th row, so the correct discounted value is
    `(1 - gamma ** done_every) / (1 - gamma)` when `done_every` is set, and `1 / (1 - gamma)`
    when it is 0.
    """
    obs = torch.zeros(size, obs_dim, device=device)
    chunk = torch.zeros(size, action_dim, device=device)
    done = torch.zeros(size, device=device)
    if done_every > 0:
        done[::done_every] = 1.0
    return {
        "obs": obs,
        "chunk": chunk,
        "noise": torch.zeros(size, action_dim, device=device),
        "reward": torch.ones(size, device=device),
        "done": done,
        "obs_next": obs.clone(),
        "chunk_next": chunk.clone(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--done-every", type=int, nargs="+", default=[0, 4, 20],
                        help="0 means no terminal transition, so the correct value is 100")
    parser.add_argument("--negative-init", type=float, nargs="*", default=[],
                        help="bias the critic's output layer to this value before training")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    device = args.device
    size = max(512, args.batch * 4)

    print(f"   gamma {args.gamma}  tau {args.tau}  lr {args.lr}  batch {args.batch}"
          f"  updates {args.updates}")
    print(f"   hidden {args.hidden}  obs {args.obs_dim}  action {args.action_dim}")
    print()

    for done_every in args.done_every:
        if done_every > 0:
            truth = (1.0 - args.gamma ** done_every) / (1.0 - args.gamma)
            label = f"done every {done_every:2d} steps"
        else:
            truth = 1.0 / (1.0 - args.gamma)
            label = "no terminal ever"

        inits = [None] + args.negative_init
        for bias in inits:
            critic = twin_critic(args.obs_dim, args.action_dim, hidden=args.hidden).to(device)
            critic_target = twin_critic(args.obs_dim, args.action_dim,
                                        hidden=args.hidden).to(device)
            if bias is not None:
                for head in critic:
                    last = [m for m in head.net.net if isinstance(m, torch.nn.Linear)][-1]
                    torch.nn.init.constant_(last.bias, bias)
            critic_target.load_state_dict(critic.state_dict())
            optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
            data = make_data(args.obs_dim, args.action_dim, size, done_every, device)

            trace = []
            for step in range(1, args.updates + 1):
                index = torch.randint(0, size, (args.batch,), device=device)
                obs, chunk = data["obs"][index], data["chunk"][index]
                with torch.no_grad():
                    next_q = torch.min(
                        torch.stack([h(data["obs_next"][index], data["chunk_next"][index])
                                     for h in critic_target]), dim=0).values
                    target = (data["reward"][index].unsqueeze(-1)
                              + args.gamma * (1.0 - data["done"][index].unsqueeze(-1)) * next_q)
                q = torch.stack([h(obs, chunk) for h in critic])
                loss = 0.5 * sum(F.mse_loss(v, target) for v in q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                soft_update(critic_target, critic, args.tau)
                if step in (1, 10, 100, 500, 1000, 2000, 3000, args.updates):
                    trace.append((step, float(q.mean().detach())))
            init_label = "default init" if bias is None else f"init {bias:+.1f}"
            path = " -> ".join(f"{v:+.2f}" for _, v in trace)
            final = trace[-1][1]
            verdict = "OK" if abs(final - truth) < max(2.0, 0.15 * abs(truth)) else "**WRONG**"
            print(f"   {label:22s} {init_label:12s} truth {truth:7.2f}   q: {path}")
            print(f"   {'':22s} {'':12s} final {final:+8.2f}  {verdict}")
            print()

    print("=" * 78)
    print("读法")
    print("=" * 78)
    print("  所有行都收敛到 truth -> TD 更新本身是对的，负不动点来自喂给它的数据（replay/chunk）。")
    print("  某一行收敛到负值或不动 -> 更新规则里有问题，可以从那一行直接读出来。")
    print("  --negative-init 的行如果爬不回来 -> 训练器会卡在最初几次更新落入的盆地里。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
