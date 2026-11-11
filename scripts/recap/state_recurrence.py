"""Do two frames with the same proprioceptive state go on to have the same outcome?

Why this question, and why it can be answered without simulation
---------------------------------------------------------------
The whole RL question reduces to identifiability: if the state determines the outcome, an action
value cannot be estimated from data in which each state appears once with one action, and no critic
-- however trained -- will have any action sensitivity.  A counterfactual simulator run would
answer this directly, but at 16.5 steps/s a meaningful number of branch points costs hours.

There is a cheaper natural experiment hiding in the existing data.  Every episode starts from the
same home pose and follows a periodic pick-and-place cycle, so **proprioceptive states recur across
episodes** -- measured earlier, the nearest neighbour from a different episode is 0.0000 rad away in
joint space.  Those recurrences are natural counterfactuals: same arm configuration, different
episode (hence different visual scene, and different actions taken afterwards).  Comparing their
outcomes separates what the proprioceptive state explains from what it does not.

What is measured
----------------
For each matched pair of frames from different episodes:

* distance in the 16-dim joint state;
* distance in the frozen visual features (the part of the state proprioception cannot see);
* each frame's own remaining-progress potential, `Phi_peak - Phi(t)`, and the eventual score.

Three correlations decide the question:

1. **state -> outcome.**  If joint-state distance predicts outcome difference, proprioception
   carries the outcome; if not, it does not.
2. **visual -> outcome.**  The complement.  If visual distance predicts outcome difference while
   joint distance does not, then the outcome is decided by something the policy can see but that
   is not in `observation.state` -- which is exactly the situation where an action-value learned
   from single-action data is unidentifiable.
3. **state -> visual.**  Whether a matched proprioceptive state even implies a matched scene.

The interesting case is (2) high and (1) low, and it is reported as such rather than buried.

Usage
-----
    python scripts/recap/state_recurrence.py --pairs 20000
"""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--state-threshold", type=float, default=0.02,
                        help="max joint-state distance (rad, L2 over 16 joints) for a 'matched' pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

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
    final_scores = labels["final_count"].to_numpy(dtype=np.float64)

    # Each frame's own remaining progress, and its episode's final score.
    remaining = np.empty(len(rows), dtype=np.float32)
    for i, (episode, frame) in enumerate(zip(episodes, frames)):
        path = potentials[int(episode)]
        remaining[i] = float(path[-1]) - float(path[min(int(frame), len(path) - 1)])
    episode_final = final_scores[episodes]

    # Visual features, random-projected to keep the pairwise distance computation affordable.
    rng = np.random.default_rng(args.seed)
    dimension = 64
    projection = rng.normal(0.0, 1.0 / np.sqrt(dimension), (8192, dimension))
    visual = np.load(features_dir / "features.npy", mmap_mode="r")[rows].astype(np.float32) @ projection
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-9)

    print(f"[recur] {len(rows)} frames, {len(np.unique(episodes))} episodes", flush=True)

    # Build matched pairs: nearest cross-episode neighbour by joint state, from a random sample.
    sample = rng.choice(len(rows), size=min(3000, len(rows)), replace=False)
    state_pairs, visual_pairs, outcome_pairs, remaining_pairs, episodes_pairs = [], [], [], [], []

    for index in sample:
        distances = np.linalg.norm(states - states[index], axis=1)
        distances[episodes == episodes[index]] = np.inf     # different episode only
        order = np.argpartition(distances, 5)[:5]
        for other in order:
            if not np.isfinite(distances[other]) or distances[other] > args.state_threshold:
                continue
            state_pairs.append(float(distances[other]))
            visual_pairs.append(float(1.0 - np.dot(visual[index], visual[other])))
            outcome_pairs.append(abs(float(remaining[index]) - float(remaining[other])))
            remaining_pairs.append(abs(float(episode_final[index]) - float(episode_final[other])))
            episodes_pairs.append((int(episodes[index]), int(episodes[other])))
        if len(state_pairs) >= args.pairs:
            break

    n = len(state_pairs)
    if n < 50:
        raise SystemExit(f"only {n} matched pairs below {args.state_threshold} rad; relax the threshold")

    state_distance = np.array(state_pairs)
    visual_distance = np.array(visual_pairs)
    outcome_difference = np.array(outcome_pairs)
    finale_difference = np.array(remaining_pairs)

    print(f"[recur] {n} matched pairs (different episodes, joint distance <= {args.state_threshold})")
    print(f"  关节距离   中位 {np.median(state_distance):.5f}  p90 {np.percentile(state_distance,90):.5f}")
    print(f"  视觉距离   中位 {np.median(visual_distance):.5f}  p90 {np.percentile(visual_distance,90):.5f}")
    print(f"  剩余进展差 中位 {np.median(outcome_difference):.2f}  p90 {np.percentile(outcome_difference,90):.2f}")
    print(f"  两集终分差 中位 {np.median(finale_difference):.1f}   平均 {np.mean(finale_difference):.2f}")

    rho_state = spearman(state_distance, outcome_difference)
    rho_visual = spearman(visual_distance, outcome_difference)
    rho_state_visual = spearman(state_distance, visual_distance)
    rho_visual_finale = spearman(visual_distance, finale_difference)
    rho_state_finale = spearman(state_distance, finale_difference)

    print()
    print(f"关节状态 → 剩余进展差     rho = {rho_state:+.3f}")
    print(f"视觉特征 → 剩余进展差     rho = {rho_visual:+.3f}")
    print(f"关节状态 → 视觉特征       rho = {rho_state_visual:+.3f}")
    print(f"关节状态 → 两集终分差     rho = {rho_state_finale:+.3f}")
    print(f"视觉特征 → 两集终分差     rho = {rho_visual_finale:+.3f}")

    # The decision.  The unidentifiable case is "the outcome varies but proprioception does not
    # explain it": an action value then has nothing in the data to learn from, because the same
    # joint state appears with wildly different futures and only the action differs -- and each
    # state was recorded with exactly one action.
    print()
    outcome_spread = float(np.median(outcome_difference))
    overall_spread = float(np.std(remaining))
    explained_state = abs(rho_state)
    explained_visual = abs(rho_visual)
    print(f"同一关节状态下，剩余进展仍相差 {outcome_spread:.2f}"
          f"（占总体标准差的 {100 * outcome_spread / max(overall_spread, 1e-9):.0f} %）")

    if outcome_spread < 0.15 * overall_spread:
        verdict = "state_determines_outcome"
        print("→ 同一本体状态的结果几乎相同。动作价值在这个数据里不可识别，RL 方向关闭。")
    elif explained_visual > 1.5 * max(explained_state, 1e-9):
        verdict = "visual_decides"
        print("→ 结果由视觉决定，而视觉不在 observation.state 里。")
        print("   同一关节状态配一个动作，动作价值不可识别 —— 这正是 critic 学不出动作敏感度的原因。")
        print("   要改变这一点需要反事实数据，或把视觉状态纳入配对（它每集都不同，所以配不上）。")
    else:
        verdict = "action_could_matter"
        print("→ 同一本体状态的结果差异较大，且不完全由视觉解释。反事实数据可能值得采集。")

    results = {
        "pairs": n,
        "median_state_distance": float(np.median(state_distance)),
        "median_visual_distance": float(np.median(visual_distance)),
        "median_outcome_difference": outcome_spread,
        "median_finale_difference": float(np.median(finale_difference)),
        "rho_state_outcome": rho_state,
        "rho_visual_outcome": rho_visual,
        "rho_state_visual": rho_state_visual,
        "rho_state_finale": rho_state_finale,
        "rho_visual_finale": rho_visual_finale,
        "verdict": verdict,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
