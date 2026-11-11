"""Can an offline Q rank the actions the policy actually took?

Why this test, and why before any actor
---------------------------------------
DSRL's claim is that a small critic can steer a frozen diffusion/flow policy by choosing among
the noise it starts from.  Everything rests on the critic: if `Q(s, a)` cannot tell a good action
from a bad one *at the same state*, then no amount of actor training produces anything but a
learned way to make things worse.

So this script trains a Q and then asks the only question that matters, which is a ranking
question, not a regression one:

    for states where the policy was tried more than once and got different outcomes,
    does Q prefer the action that did better?

That is exactly the structure `eval_retryv9_a{1,2,3}` produced -- 24 scenes x 3 attempts, with
observed scores.  But those runs used `--no-record`, so they carry no observations to featurise.
The substitute used here is the training set itself: within one episode the potential rises and
falls, so "the same state at different times" provides the contrast, and the hindsight headroom
`Phi_peak - Phi_running(t)` gives every frame a per-step target instead of a single sparse
terminal reward (non-zero reward frames are 0.46 % of the data, which is far too sparse for TD).

Why a pairwise/ranking loss and not plain MSE
---------------------------------------------
MSE on headroom rewards fitting the conditional mean, and the mean of a wide distribution is a
number no action produced.  What DSRL needs is the *ordering*: given two actions at one state,
prefer the one that leads to more potential.  A margin loss on `(-headroom)` as the utility
directly encodes that, and it is invariant to the part of headroom driven by the episode's
difficulty, which MSE would try to fit.

The split is by episode, and the reported number is Spearman on held-out episodes.  A Q that
scores well on its own training states is only evidence of memorisation.

Usage
-----
    python scripts/recap/train_q_value.py --dataset /root/autodl-tmp/datasets/a3_recap_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed directly so scipy is not required."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.2, help="fraction of EPISODES held out")
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help=(
            "control steps of future the target looks at.  0 means the rest of the episode (the "
            "full headroom).  A shorter horizon concentrates credit on the action that was just "
            "taken, at the cost of signal density: `Phi` only moves on Cookie events, so within "
            "20 steps 2.2 %% of frames see any change at all."
        ),
    )
    parser.add_argument(
        "--sensitivity-states", type=int, default=0,
        help="if >0, after training measure how much of Q's variance comes from the action",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch
    from torch import nn

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset
    features_dir = dataset / "features_v9"
    labels = pq.read_table(dataset / "value_labels.parquet").to_pandas()

    written = np.load(features_dir / "written.npy")
    rows = np.flatnonzero(written)
    manifest = json.loads((features_dir / "index.json").read_text())
    stride = int(manifest["stride"])

    features = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32)
    feature_dim = features.shape[1]

    # Per-frame action and episode, from the data parquet directly: `--no-record` evaluation runs
    # carry no observations, so the Q has to be trained on the recorded training set.
    frame_table = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["action", "episode_index", "frame_index"],
    ).to_pandas()
    action_all = np.stack(frame_table["action"].values).astype(np.float32)[:, :16]
    episode_all = frame_table["episode_index"].to_numpy(dtype=np.int64)
    frame_all = frame_table["frame_index"].to_numpy(dtype=np.int64)

    actions = action_all[rows]
    episodes = episode_all[rows]
    frames = frame_all[rows]

    # The target: hindsight headroom at each cached frame.  Built from the label file's own
    # `potential` column so the Q and the value head speak about the same quantity.
    potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]

    def bucket_zscore(values: np.ndarray, buckets: int = 10) -> np.ndarray:
        """Z-score the quantity against its own episode's progress, per time bucket.

        Absolute headroom is a leak, not a signal: within an episode it correlates with frame
        position at |rho| 0.900, and half of the episodes exceed 0.9, so a model can reach 100 %
        pairwise accuracy by learning "earlier is better" and never look at the action.  Measured
        alternatives: dividing by the peak does not help (still 0.900, because the peak is a
        constant per episode and cancels); a per-bucket z-score reduces it to **0.040 with 0 %
        above 0.9**, because the model is then asked "for this phase of the task, was this frame
        better than typical", which is exactly the question a critic should answer.
        """
        n = len(values)
        if n <= 1:
            return np.zeros(n, dtype=np.float64)
        relative = np.arange(n) / (n - 1)
        index = np.minimum((relative * buckets).astype(int), buckets - 1)
        out = np.zeros(n, dtype=np.float64)
        for bucket in range(buckets):
            members = index == bucket
            if members.sum() > 1 and values[members].std() > 1e-9:
                out[members] = (values[members] - values[members].mean()) / values[members].std()
        return out

    frames_per_episode = [len(p) for p in potentials]
    # `potential` is per *control step* (1500 entries for a 1500-step episode), not per cached
    # row -- verified against `labels.frames`.  Every earlier version of this file divided the
    # frame index by `stride` and therefore read only the first fifth of every episode's potential
    # curve, which is why the target looked so weakly related to everything.
    lookahead = int(args.horizon) if args.horizon > 0 else 0
    target_by_episode = []
    for path in potentials:
        if lookahead > 0:
            future = np.array([
                path[min(i + lookahead, len(path) - 1)] for i in range(len(path))
            ], dtype=np.float64)
        else:
            future = np.full(len(path), float(path[-1]), dtype=np.float64)
        # The quantity is always "how much better did this go than the episode's own peak
        # allowed", i.e. `future - now` whichever horizon was asked for; the bucket z-score then
        # removes the part of that which is explained by being early or late.
        target_by_episode.append(bucket_zscore(-(path - future)))

    headroom = np.empty(len(rows), dtype=np.float32)
    for i, (episode, frame) in enumerate(zip(episodes, frames)):
        path = target_by_episode[int(episode)]
        step = min(int(frame), len(path) - 1)
        headroom[i] = float(path[step])

    unique_episodes = np.unique(episodes)
    rng.shuffle(unique_episodes)
    n_holdout = max(1, int(round(args.holdout * len(unique_episodes))))
    holdout_episodes = set(unique_episodes[:n_holdout].tolist())
    is_holdout = np.isin(episodes, list(holdout_episodes))
    print(f"[q] {len(rows)} frames, {len(unique_episodes)} episodes"
          f" ({n_holdout} held out, {is_holdout.sum()} held-out frames)", flush=True)
    print(f"[q] headroom: mean {headroom.mean():.2f} sd {headroom.std():.2f}"
          f" range {headroom.min():.1f}..{headroom.max():.1f}", flush=True)

    class QNet(nn.Module):
        """Q(s, a): the state feature, plus the action the policy took there.

        The action is *not* pooled away or concatenated after a projection of the state alone --
        it goes in at the input, because the whole question is "at this state, which action is
        better", and a network that sees the state only would predict the state's value no matter
        what it is asked about.
        """

        def __init__(self, feature_dim: int, action_dim: int, hidden: int = 1024):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden // 2), nn.SiLU(),
            )
            self.head = nn.Linear(hidden // 2, 1)

        def forward(self, state, action):
            return self.head(self.trunk(torch.cat([state, action], dim=-1))).squeeze(-1)

    device = args.device
    q = QNet(feature_dim, actions.shape[1]).to(device)
    optimizer = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-4)

    tensors = {
        "state": torch.from_numpy(features).to(device),
        "action": torch.from_numpy(actions).to(device),
        "target": torch.from_numpy(headroom).to(device),
    }
    train_index = torch.from_numpy(np.flatnonzero(~is_holdout)).to(device)
    holdout_index = torch.from_numpy(np.flatnonzero(is_holdout)).to(device)

    def ranking_loss(prediction, target, groups):
        """Prefer the higher-headroom action at the same state.

        `prediction` is already shaped (pairs, 2) -- column 0 is the first row of each pair and
        column 1 the second -- so the two halves are selected along dim 1, not by re-indexing the
        flat index tensor.  `groups` partitions the batch by episode; within a group the two rows
        are at different times, i.e. different actions from different states in one trajectory.

        The margin is what makes this a ranking objective rather than a regression one: it only
        cares that `prediction` orders the pair the way the target does.
        """
        left, right = groups[:, 0], groups[:, 1]
        better = (target[left] > target[right]).float() * 2.0 - 1.0   # +1 when left is better
        keep = target[left] != target[right]                          # ties carry no preference
        if int(keep.sum()) < 2:
            return torch.zeros((), device=prediction.device, requires_grad=True)
        margin = (prediction[:, 0] - prediction[:, 1]) * better
        return nn.functional.softplus(-2.0 * margin[keep]).mean()

    print()
    print(f"{'epoch':>5} {'loss':>8} {'holdout rho':>12} {'pairwise acc':>13}")
    best = {"rho": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, args.epochs + 1):
        q.train()
        permutation = train_index[torch.randperm(len(train_index), device=device)]
        total, seen = 0.0, 0
        for start in range(0, len(permutation) - args.batch_size, args.batch_size):
            batch = permutation[start : start + args.batch_size]
            # Build pairs *within* the batch: random partners, so a "better" pair is discovered
            # rather than constructed from a label ordering.
            half = len(batch) // 2
            groups = torch.stack([batch[:half], batch[half : 2 * half]], dim=1)
            prediction = q(tensors["state"][groups.reshape(-1)], tensors["action"][groups.reshape(-1)])
            prediction = prediction.view(groups.shape[0], 2)
            loss = ranking_loss(prediction, tensors["target"], groups)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * groups.shape[0]
            seen += groups.shape[0]

        if epoch % 5 == 0 or epoch == 1:
            q.eval()
            with torch.no_grad():
                held_pred = torch.cat([
                    q(tensors["state"][holdout_index[i : i + 8192]],
                      tensors["action"][holdout_index[i : i + 8192]])
                    for i in range(0, len(holdout_index), 8192)
                ]).cpu().numpy()
            held_target = tensors["target"][holdout_index].cpu().numpy()
            rho = spearman(held_pred, held_target)
            # Pairwise agreement: the metric that actually matters for steering.  Formed from
            # random pairs of held-out frames within the same episode.
            held_ep = episodes[holdout_index.cpu().numpy()]
            # Bucket by *cached row position within the episode*, which is what the target used.
            # `frames_per_episode` is indexed by episode and gives cached rows, so the ratio is
            # already in [0, 1] -- an earlier version multiplied by `stride` and divided by a raw
            # frame count, which put almost every row in bucket 9 and left only a handful of
            # valid pairs.  The pair count is printed so that mistake cannot recur silently.
            held_position = np.array([
                frame / max(frames_per_episode[e] - 1, 1)
                for e, frame in zip(held_ep, frames[holdout_index.cpu().numpy()])
            ])
            held_bucket = np.minimum((held_position * 10).astype(int), 9)

            # Group the held-out rows so pairs can be drawn within (episode, bucket).
            from collections import defaultdict
            groups = defaultdict(list)
            for index, (episode, bucket) in enumerate(zip(held_ep, held_bucket)):
                groups[(int(episode), int(bucket))].append(index)
            usable = [members for members in groups.values() if len(members) >= 4]

            acc_n, acc_d = 0, 0
            rng_local = np.random.default_rng(epoch)
            for _ in range(4000):
                if not usable:
                    break
                members = usable[int(rng_local.integers(0, len(usable)))]
                i, j = rng_local.choice(members, 2, replace=False)
                if held_target[i] == held_target[j]:
                    continue
                acc_d += 1
                predicted = held_pred[i] > held_pred[j]
                actual = held_target[i] > held_target[j]
                acc_n += int(predicted == actual)
            accuracy = acc_n / max(acc_d, 1)
            # `acc_d` is reported because a high accuracy over a handful of pairs is noise.  With
            # ~150 frames per episode per bucket the count should be in the thousands; if it is
            # not, the number means nothing and the run should not be read as evidence.
            print(f"{epoch:5d} {total / max(seen, 1):8.4f} {rho:12.4f}"
                  f" {100 * accuracy:11.1f} %  ({acc_d} pairs)", flush=True)
            if rho > best["rho"]:
                best = {"rho": rho, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in q.state_dict().items()},
                        "accuracy": accuracy}

    print()
    print(f"best epoch {best['epoch']}: Spearman {best['rho']:.4f},"
          f" pairwise accuracy {100 * best.get('accuracy', float('nan')):.1f} %")

    # The decision this experiment exists to make.
    rho = best["rho"]
    accuracy = best.get("accuracy", 0.0)
    print()
    if rho > 0.5 and accuracy > 0.75:
        print("→ Q 能区分同状态下的动作优劣。DSRL 的核心前提成立，值得实现噪声 actor。")
    elif rho > 0.25:
        print("→ Q 有弱信号。可能需要更多数据或更好的状态特征；先别做 actor。")
    else:
        print("→ Q 学不到动作优劣。DSRL 路线在当前特征/数据下不成立，止步。")

    # --- action sensitivity, measured in the same process --------------------------------
    # This is the property that decides whether a critic can steer a sampler: hold the state
    # fixed, vary only the action, and see how much of Q's variance survives.  It is measured
    # here rather than in a separate script so that every horizon is probed by the *same* code
    # on the *same* split, which is the only way the numbers are comparable across the sweep.
    sensitivity = None
    if args.sensitivity_states > 0:
        q.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
        q.eval()
        probe_pool = np.flatnonzero(is_holdout)   # held-out frames only
        probe = rng.choice(probe_pool, size=min(args.sensitivity_states, len(probe_pool)), replace=False)
        within_variances, spreads = [], []
        with torch.no_grad():
            for index in probe:
                state = tensors["state"][int(index)][None]
                # Nearby frames of the *same episode*: the state has barely moved, so the action
                # is what differs.  This is the mild perturbation a sampler-steering actor makes.
                #
                # The earlier version also required `~is_holdout` here, which excluded every
                # candidate: `is_holdout` is a per-*episode* flag, so all of a held-out episode's
                # frames carry it and nothing from that episode could be used.  Taking the
                # perturbation actions from the same held-out episode is also the right choice --
                # Q has never seen any of these states, so the probe measures generalisation
                # rather than recall.
                local = np.flatnonzero(
                    (episodes == episodes[index]) & (np.abs(frames - frames[index]) <= 10 * stride)
                )
                local = local[local != index]
                if len(local) < 3:
                    continue
                pick = rng.choice(local, size=min(8, len(local)), replace=False)
                candidate = torch.from_numpy(actions[pick]).to(device)
                values = q(state.expand(len(pick), -1), candidate).cpu().numpy()
                within_variances.append(float(np.var(values)))
                spreads.append(float(values.max() - values.min()))
        total_sample = rng.choice(len(rows), size=4096, replace=False)
        total_actions = rng.choice(len(rows), size=4096, replace=False)
        with torch.no_grad():
            total_values = q(tensors["state"][torch.from_numpy(total_sample).to(device)],
                             tensors["action"][torch.from_numpy(total_actions).to(device)]).cpu().numpy()
        total_variance = float(np.var(total_values))
        sensitivity = {
            "probes": len(within_variances),
            "within_state_variance": float(np.mean(within_variances)) if within_variances else float("nan"),
            "within_state_spread": float(np.mean(spreads)) if spreads else float("nan"),
            "total_variance": total_variance,
            "action_fraction": (float(np.mean(within_variances)) / total_variance) if within_variances and total_variance > 0 else float("nan"),
        }
        print()
        print(f"动作敏感度: 状态内方差 {sensitivity['within_state_variance']:.6f}"
              f" / 总体方差 {total_variance:.6f} = **{100 * sensitivity['action_fraction']:.2f} %**"
              f"  ({sensitivity['probes']} probes)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best["state"], args.output)
        report = {
            "holdout_episodes": n_holdout,
            "frames": int(len(rows)),
            "horizon_control_steps": args.horizon,
            "best_epoch": best["epoch"],
            "spearman": rho,
            "pairwise_accuracy": accuracy,
        }
        if sensitivity is not None:
            report["sensitivity"] = sensitivity
        (args.output.parent / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
