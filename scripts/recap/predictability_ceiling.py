"""Can the outcome be predicted at all, given the information the trace does carry?

The question this settles
------------------------
`state_recurrence.py` matched frames across episodes by joint state and found that the same
proprioceptive state still leads to wildly different remaining progress (median difference 6.00,
70 % of the total standard deviation), while neither joint distance (rho -0.021) nor visual feature
distance (rho +0.013) predicts that difference.

Two explanations remain, and they imply opposite next steps:

* **The actions matter, proprioception just cannot see the difference.**  Then counterfactual data
  is worth collecting and an action value is learnable in principle.
* **The state representation is too weak to locate the task.**  Then no critic can work, because
  neither the state nor the action is being described well enough, and the fix is features, not RL.

`target_slot_occupancy` separates the two.  It records, for each of the 20 target slots, which
Cookie occupies it -- i.e. the arrangement that the score is computed from, and therefore task
progress itself.  It is a perfect predictor by construction (the score is a function of it), so
using it as a state is not a proposal; it is an **upper bound**: it says how much of the outcome is
predictable *when the state is described adequately*.

If that upper bound is near 1.0 while the visual/joint features reach ~0, the limitation is the
representation, not the task's identifiability.  If the upper bound is also low, the outcome is not
a function of the arrangement at all and something else -- contact history, velocity, luck -- owns
it.

Three predictors, same held-out episodes, same target
-----------------------------------------------------
* **occupancy**  -- the 20-slot table, one number per slot (`-1` recoded to 0.5 so a slot can be
  "empty" without pretending it is Cookie 0);
* **joint**      -- the 16-joint proprioceptive state;
* **visual**     -- the frozen 8192-dim feature, reduced by a random projection.

Target: remaining potential `Phi_peak - Phi(t)`, which is what a critic is asked to predict.

Usage
-----
    python scripts/recap/predictability_ceiling.py --dataset /root/autodl-tmp/datasets/a3_recap_all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def spearman(a: np.ndarray, b: np.ndarray) -> float:
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--projection", type=int, default=128)
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
    written = np.load(features_dir / "written.npy")
    rows = np.flatnonzero(written)
    manifest = json.loads((features_dir / "index.json").read_text())
    stride = int(manifest["stride"])

    frame_table = pq.read_table(
        dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["observation.state", "episode_index", "frame_index"],
    ).to_pandas()
    states = np.stack(frame_table["observation.state"].values)[rows][:, :16].astype(np.float32)
    episodes = frame_table["episode_index"].to_numpy(dtype=np.int64)[rows]
    frames = frame_table["frame_index"].to_numpy(dtype=np.int64)[rows]

    labels = pq.read_table(dataset / "value_labels.parquet").to_pandas()
    potentials = [np.asarray(p, dtype=np.float64) for p in labels["potential"]]
    remaining = np.empty(len(rows), dtype=np.float32)
    for i, (episode, frame) in enumerate(zip(episodes, frames)):
        path = potentials[int(episode)]
        remaining[i] = float(path[-1]) - float(path[min(int(frame), len(path) - 1)])

    # --- the occupancy table, from the trace ------------------------------------------
    # Traces are per (part, seed); the label file's `part` column maps an episode to its root.
    traces_by_episode = np.empty(len(labels), dtype=object)
    for episode_index, (part, seed) in enumerate(zip(labels["part"], labels["seed"])):
        traces_by_episode[episode_index] = (
            Path("/root/autodl-tmp") / str(part) / "traces" / f"seed_{int(seed)}" / "trace.jsonl"
        )

    occupancy = np.full((len(rows), 20), np.nan, dtype=np.float32)
    cache: dict[int, np.ndarray] = {}
    missing = 0
    for i, (episode, frame) in enumerate(zip(episodes, frames)):
        episode = int(episode)
        if episode not in cache:
            path = traces_by_episode[episode]
            table = np.full((np.iinfo(np.int32).max, 20), np.nan, dtype=np.float32) if False else None
            entries = []
            try:
                for line in open(path):
                    if line.strip():
                        entries.append(json.loads(line)["after"].get("target_slot_occupancy") or [])
            except FileNotFoundError:
                entries = []
            if entries:
                array = np.full((len(entries), 20), np.nan, dtype=np.float32)
                for row_index, slot_values in enumerate(entries):
                    if slot_values:
                        array[row_index, : len(slot_values)] = np.asarray(slot_values, dtype=np.float32)
                cache[episode] = array
            else:
                cache[episode] = np.full((0, 20), np.nan, dtype=np.float32)
        array = cache[episode]
        step = int(frame)
        if len(array) and step < len(array):
            slot_values = array[step]
            # `-1` is "empty" and 0..19 name a Cookie; recoding empty to 0.5 keeps the ordering
            # "empty sits between nothing and the first Cookie" without pretending it is Cookie 0.
            occupancy[i] = np.where(slot_values < 0, 0.5, slot_values)
        else:
            missing += 1

    usable = ~np.isnan(occupancy[:, 0])
    print(f"[ceil] {len(rows)} frames, occupancy available for {usable.sum()}"
          f" ({100 * usable.mean():.1f}%), {missing} missing", flush=True)
    if usable.sum() < 1000:
        raise SystemExit("too few frames with an occupancy table")

    # --- visual features, projected ---------------------------------------------------
    projection = rng.normal(0.0, 1.0 / np.sqrt(args.projection), (8192, args.projection))
    visual = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32) @ projection

    predictors = {
        "occupancy (20 slots)": occupancy[usable],
        "joint (16)": states[usable],
        "visual (8192 -> 128)": visual[usable],
        "joint+visual": np.concatenate([states[usable], visual[usable]], axis=1),
    }
    target = remaining[usable]
    episodes_used = episodes[usable]

    unique = np.unique(episodes_used)
    rng.shuffle(unique)
    n_holdout = max(1, int(round(args.holdout * len(unique))))
    holdout_set = set(unique[:n_holdout].tolist())
    is_holdout = np.isin(episodes_used, list(holdout_set))
    print(f"[ceil] split: {len(unique) - n_holdout} train / {n_holdout} holdout episodes"
          f" ({is_holdout.sum()} held-out frames)", flush=True)

    device = args.device
    print()
    print(f"{'状态表示':26s} {'维度':>5s} {'holdout rho':>12s} {'R2':>8s}")
    results = {}
    for name, features in predictors.items():
        features = np.asarray(features, dtype=np.float32)
        network = nn.Sequential(
            nn.Linear(features.shape[1], 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 128), nn.SiLU(),
            nn.Linear(128, 1),
        ).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)

        x = torch.from_numpy(features).to(device)
        y = torch.from_numpy(target).to(device)
        train_index = torch.from_numpy(np.flatnonzero(~is_holdout)).to(device)
        holdout_index = torch.from_numpy(np.flatnonzero(is_holdout)).to(device)

        batch_size = 4096
        for _ in range(args.epochs):
            network.train()
            permutation = train_index[torch.randperm(len(train_index), device=device)]
            for start in range(0, len(permutation) - batch_size, batch_size):
                batch = permutation[start : start + batch_size]
                loss = nn.functional.mse_loss(network(x[batch]).squeeze(-1), y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        network.eval()
        with torch.no_grad():
            predicted = torch.cat([
                network(x[holdout_index[i : i + 8192]]).squeeze(-1)
                for i in range(0, len(holdout_index), 8192)
            ]).cpu().numpy()
        truth = y[holdout_index].cpu().numpy()
        rho = spearman(predicted, truth)
        residual = float(np.mean((predicted - truth) ** 2))
        variance = float(np.var(truth))
        r2 = 1.0 - residual / variance if variance > 0 else float("nan")
        print(f"{name:26s} {features.shape[1]:5d} {rho:12.4f} {r2:8.3f}", flush=True)
        results[name] = {"dimension": int(features.shape[1]), "spearman": rho, "r2": r2}

    # --- the decision ------------------------------------------------------------------
    ceiling = max(entry["spearman"] for entry in results.values())
    occupancy_rho = results["occupancy (20 slots)"]["spearman"]
    visual_rho = results["visual (8192 -> 128)"]["spearman"]
    print()
    print(f"最优可达 rho = {ceiling:.3f}（{'occupancy' if ceiling == occupancy_rho else '其他'}）")
    if occupancy_rho > 0.8 and max(visual_rho, results['joint (16)']['spearman']) < 0.4:
        verdict = "representation_is_the_limit"
        print("→ 用槽位表几乎能完全预测结果，而视觉/关节几乎不能。")
        print("   上限存在，缺的是状态表示 —— 不是动作价值不可识别。")
        print("   ⇒ 该修的是特征（把空间信息保留下来），不是 RL 算法。")
    elif occupancy_rho < 0.5:
        verdict = "outcome_not_a_function_of_arrangement"
        print("→ 连槽位表都预测不好结果：结果不由当前布局决定，")
        print("   接触历史或未观测变量占主导。")
    else:
        verdict = "mixed"
        print("→ 布局能部分解释结果。视觉表示的损失是其中一个原因，但不是全部。")

    results["verdict"] = verdict
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
