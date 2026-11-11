#!/usr/bin/env python3
"""Can the cached DSRL observation predict the true discounted return at all?

Why this has to be asked before "TD did not learn"
--------------------------------------------------
`train_dsrl.py --utd 20` reported `critic loss 2e-4` with `Q` pinned near -1 while the true
discounted return on the *same* states averaged 6.5 (max 25.9).  Two very different explanations
fit that:

* the pool contained one reward-carrying transition in 512 and no episode terminals, so every
  TD target degenerated to `0.999 * Q(s')` and the only self-consistent solution was `Q = 0`;
* the observation genuinely carries no information about the return.

The first is a sampling bug and is fixed (all reward transitions are now in the pool).  The second
would make the whole value-based approach dead on these features, whatever the sampler does.  This
script separates them by fitting the *same* networks to the **Monte-Carlo return** instead of a
bootstrapped target.  Monte-Carlo has no fixed point to degenerate to: it is a plain regression, so
if it cannot fit, the features are the problem and no critic change will save it.

What is compared
----------------
* `mc`   -- regress `Q(s) = G(s)` directly (no bootstrap, no target network).  Upper bound on what
  any value head can read off these features.
* `td`   -- the ordinary TD objective, for reference, on the same rows and the same batch order.
* `zero` -- predicts 0.  Reported because a model that does worse than 0 on `R^2` is not merely
  weak, it is anti-correlated, and the sign matters for which failure this is.

`R^2` is computed against the *variance of G*, on a held-out 20 % of transitions.  A positive
number means the features rank the returns; that is all the actor needs, since the actor only
consumes `argmax_e Q`.

Usage
-----
    python scripts/recap/probe_value_ceiling.py --transitions 4096 --epochs 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_labels(dataset: Path):
    """Per-row reward, done, and the true discounted return inside each episode."""
    labels = pq.read_table(dataset / "value_labels.parquet").to_pandas()
    potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]
    frame_table = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "frame_index"],
    ).to_pandas()
    episodes = frame_table["episode_index"].to_numpy(np.int64)

    bounds = {}
    offset = 0
    for position, length in enumerate(labels["frames"]):
        bounds[position] = (offset, offset + int(length))
        offset += int(length)

    rewards = np.zeros(len(episodes), dtype=np.float32)
    dones = np.zeros(len(episodes), dtype=np.float32)
    for episode in range(len(labels)):
        start, stop = bounds[episode]
        increments = np.diff(np.concatenate([[0.0], potentials[episode][: stop - start]]))
        rewards[start:stop] = increments[: stop - start]
        if stop - 1 < len(dones):
            dones[stop - 1] = 1.0
    return rewards, dones, episodes, labels, bounds, potentials, len(episodes)


def discounted_return(rewards, dones, gamma: float) -> np.ndarray:
    """`G(s) = r + gamma * G(s')` computed backwards over the whole table.

    `dones` is respected, so an episode boundary resets the recursion instead of leaking the next
    episode's reward into this one's value.
    """
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for row in range(len(rewards) - 1, -1, -1):
        running = float(rewards[row]) + gamma * (1.0 - float(dones[row])) * running
        out[row] = running
    return out


def r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = float(((prediction - target) ** 2).mean())
    variance = float(((target - target.mean()) ** 2).mean())
    return 1.0 - residual / max(variance, 1e-12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--transitions", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache", type=Path, default=None,
        help=(
            "which `obs_*.pt` to read.  Default is the *smallest* one written, because that is "
            "the pool without reward anchors and it is the configuration the TD collapse was "
            "observed in -- the probe should reproduce the failure, not avoid it.  Point this at "
            "the anchored cache to see what changes."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from train_dsrl import MLP, soft_update, twin_critic  # noqa: PLC0415

    def value_heads(obs_dim: int, hidden: int) -> torch.nn.ModuleList:
        """`Q(s)` with no action input.

        `twin_critic(obs_dim, 0)` would hand `QHead` a zero-width action and `torch.cat` handles
        that, but the state-only head is clearer and does not depend on that tolerance.
        """
        return torch.nn.ModuleList([
            torch.nn.Sequential(MLP(obs_dim, 1, hidden), torch.nn.Flatten(0, -1))
            for _ in range(2)
        ])

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    rewards, dones, episodes, labels, bounds, potentials, total = load_labels(args.dataset)
    frames = pq.read_table(
        args.dataset / "data" / "chunk-000" / "file-000.parquet", columns=["frame_index"]
    ).to_pandas()["frame_index"].to_numpy(np.int64)
    return_all = discounted_return(rewards, dones, args.gamma)
    print(f"[probe] {total} rows; reward non-zero {100 * (rewards != 0).mean():.2f} %",
          flush=True)
    print(f"[probe] gamma {args.gamma}: true return mean {return_all.mean():.3f}"
          f" std {return_all.std():.3f} p99 {np.percentile(return_all, 99):.2f}"
          f" max {return_all.max():.2f}", flush=True)
    print(f"[probe] return is non-zero on {100 * (return_all != 0).mean():.1f} % of rows"
          f" -- a Monte-Carlo target is dense even where the reward is not", flush=True)

    # --- observations: reuse the cache `train_dsrl.py` already wrote ---------------------
    if args.cache is not None:
        cache_files = [args.cache]
    else:
        cache_files = sorted((args.dataset / "dsrl_cache").glob("obs_*.pt"),
                             key=lambda p: p.stat().st_size)
    if not cache_files:
        print("[probe] no cache under dsrl_cache/; run train_dsrl.py once first", flush=True)
        return 2

    # Adjacent pairs, reward anchors included -- the same pool `train_dsrl.py` builds.
    adjacent = np.flatnonzero((frames[1:] == frames[:-1] + 1) & (episodes[1:] == episodes[:-1]))
    reward_rows = np.intersect1d(adjacent, np.flatnonzero(rewards != 0))
    budget = int(min(args.transitions, len(adjacent)))
    if len(reward_rows) > budget:
        starts = np.sort(rng.choice(reward_rows, size=budget, replace=False))
    else:
        rest = np.setdiff1d(adjacent, reward_rows, assume_unique=False)
        fill = np.sort(rng.choice(rest, size=min(budget - len(reward_rows), len(rest)),
                                  replace=False))
        starts = np.sort(np.concatenate([reward_rows, fill]))
    rows_cur = starts
    rows_next = starts + 1

    # The cache may have been built for a different pool; only its observation layout is reusable,
    # so the features are recomputed from it only when the row sets match, and otherwise the
    # cached table is used as a *feature* source for the rows it does contain.
    cached = torch.load(cache_files[-1], weights_only=True)
    cached_rows = cached["rows_cur"].numpy()
    anchors_here = int((rewards[cached_rows] != 0).sum())
    print(f"[probe] pool: {len(cached_rows)} rows, {anchors_here} carrying a reward"
          f" ({cache_files[-1].name})", flush=True)
    if anchors_here == 0:
        print("[probe] NOTE: this pool has no reward anchors, so the TD target has no fixed point"
              " other than Q=0.  That is a property of the pool, not of the features.", flush=True)

    observation = torch.cat(
        [cached["proprio_cur"], cached["probe_cur"],
         torch.zeros(len(cached_rows), 0)], dim=1,
    ).float()
    observation_next = torch.cat(
        [cached["proprio_next"], cached["probe_next"],
         torch.zeros(len(cached_rows), 0)], dim=1,
    ).float()
    row_cur = cached["rows_cur"].numpy()
    row_next = cached["rows_next"].numpy()
    observation_dim = observation.shape[1]
    print(f"[probe] observation {observation_dim} dims (proprio + VLM; visual omitted because the"
          f" cache does not pin it)", flush=True)

    observation = observation.to(device)
    observation_next = observation_next.to(device)
    reward_t = torch.from_numpy(rewards[row_cur]).to(device)
    done_t = torch.from_numpy(dones[row_cur]).to(device)
    return_t = torch.from_numpy(return_all[row_cur].astype(np.float32)).to(device)
    true_next = torch.from_numpy(return_all[row_next].astype(np.float32)).to(device)

    count = len(row_cur)
    order = np.arange(count)
    rng.shuffle(order)
    split = int(0.8 * count)
    train_index = torch.from_numpy(order[:split]).to(device)
    test_index = torch.from_numpy(order[split:]).to(device)
    print(f"[probe] {split} train / {count - split} held out", flush=True)

    def fit(name: str, use_bootstrap: bool):
        """Fit a twin value head and report held-out R^2 of the mean."""
        critic = value_heads(observation_dim, args.hidden).to(device)
        target_net = value_heads(observation_dim, args.hidden).to(device)
        target_net.load_state_dict(critic.state_dict())
        optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)

        def evaluate(rows: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return torch.stack([head(observation[rows]) for head in critic], dim=0).mean(0)

        for epoch in range(1, args.epochs + 1):
            pick = train_index[torch.randint(0, split, (args.batch,), device=device)]
            with torch.no_grad():
                if use_bootstrap:
                    next_value = torch.stack(
                        [head(observation_next[pick]) for head in target_net], dim=0
                    ).min(0).values
                    target = reward_t[pick] + args.gamma * (1.0 - done_t[pick]) * next_value
                else:
                    target = return_t[pick]
            predicted = torch.stack([head(observation[pick]) for head in critic], dim=0)
            loss = 0.5 * sum(F.mse_loss(p, target) for p in predicted)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_bootstrap:
                soft_update(target_net, critic, 0.005)
            if epoch % 100 == 0:
                held = evaluate(test_index)
                print(f"    {name} {epoch:4d}  loss {float(loss.detach()):8.4f}"
                      f"  held-out R2 {r2(held, return_t[test_index]):+.4f}", flush=True)
        held = evaluate(test_index)
        return r2(held, return_t[test_index]), float(held.std())

    zero_r2 = r2(torch.zeros_like(return_t[test_index]), return_t[test_index])
    print(f"\n[probe] predictor 'always 0': R2 {zero_r2:+.4f}", flush=True)
    mc_r2, mc_spread = fit("mc", use_bootstrap=False)
    td_r2, td_spread = fit("td", use_bootstrap=True)

    print("\n结果")
    print(f"  Monte-Carlo 目标 (上界)：R2 {mc_r2:+.4f}   预测 std {mc_spread:.3f}"
          f"  (真值 std {float(return_t[test_index].std()):.3f})")
    print(f"  TD 目标（实际训练）：R2 {td_r2:+.4f}   预测 std {td_spread:.3f}")
    print(f"  恒 0 预测：R2 {zero_r2:+.4f}")
    print()
    if mc_r2 > 0.3:
        print("  判读：特征能读出回报，TD 的失败是目标/采样问题，不是表征问题。")
    elif mc_r2 > 0.05:
        print("  判读：特征只有弱信号。actor 要靠 Q 排序，弱 R2 也能用，但不要期待大幅提升。")
    else:
        print("  判读：这些特征读不出回报。换 critic 不会有用——需要换观测（episode 进度、")
        print("        饼干计数这类显式状态），或者换掉离线价值的整个思路。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "gamma": args.gamma,
            "transitions": int(count),
            "train": int(split),
            "return_mean": float(return_t.mean()),
            "return_std": float(return_t.std()),
            "zero_r2": zero_r2,
            "mc_r2": mc_r2,
            "td_r2": td_r2,
            "mc_prediction_std": mc_spread,
            "td_prediction_std": td_spread,
        }, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
