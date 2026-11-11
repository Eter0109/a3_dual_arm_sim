#!/usr/bin/env python3
"""Does the critic *rank* actions correctly?  The question DSRL's whole design rests on.

Why this is the question that matters now
-----------------------------------------
Three evaluation rounds measured what happens when the actor's noise is fed to the sampler:

    prior                  mean /20
    N(0,1)                     13.88
    actor v5 (512 wide)         9.88      critic Q ~ 2.8 against a true 9.56, i.e. badly underfit
    actor v6 (2048 wide)        3.12      critic Q ~ 7.98, i.e. accurate

Fixing the SAC implementation made the critic *accurate* and the actor *worse*.  That is only
paradoxical if you assume the critic's ordering is right: an underfit critic returns nearly the
same Q for every noise, so its argmax barely moves and the actor ends up close to the unchanged
sampler; an accurate critic moves the actor decisively, and if its ordering is wrong, decisively
in the wrong direction.

So the assumption to test is `argmax_epsilon Q(s, epsilon)` == "the better noise".  Nothing in this
project has tested it.  §9.91-9.93 verified that Q's *values* match the Monte-Carlo return on the
recorded actions; that is a statement about calibration at one action per state, and it says
nothing about ordering *across* actions.

What is measured, and in which space
------------------------------------
`branches/*.npz` holds the only counterfactual data in this project: `branch_rollouts_new.py`
restored a state, applied `recorded_action + noise` for 16 steps with one shared 16-dim vector,
then followed the recorded actions for a 120-step horizon, and recorded how many Cookies ended up
in the target.  Same state, several noises, real consequences -- exactly the contrast a value
function has to get right.

Two caveats, stated because they bound the claim:

* this is the **action-space** critic (`critic`), not the noise-space one (`critic_noise`) that the
  actor maximises.  The branch perturbations live in action space, so `critic_noise` cannot be
  evaluated on them.  But `critic_noise` is *distilled from* `critic`, so a `critic` that cannot
  rank actions cannot produce a `critic_noise` that ranks noises.  This is the necessary condition.
* the outcome is `cookies_in_target` after a 120-step horizon, not the potential-based return the
  critic was trained on.  Ranking is scale-free, so the test is still valid: it asks whether the
  critic orders two actions the way the environment does.

The statistic
-------------
Within each branch point, every pair of branches with different outcomes is a test.  Accuracy is
the fraction of pairs where the critic agrees with the environment.  0.5 is no information.
Because branches cluster within points, the error bar comes from a bootstrap over *points*, not
over pairs -- 823 pairs from 45 points is not 823 independent observations.

Usage
-----
    python scripts/recap/probe_critic_ranking.py --checkpoint <ckpt.pt> [--checkpoint ...]
"""

from __future__ import annotations

import argparse
import glob
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

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_branches(branch_dir: Path):
    """Concatenate every shard, fixing `branch_point` to a global point id.

    The shards each number their points from zero, so the ids have to be offset by the *row* count
    of the points already concatenated.  An earlier version of this loader offset by the number of
    *arrays* appended, which silently collapsed 93 points into 20 and discarded three quarters of
    the pairs.
    """
    files = sorted(glob.glob(str(branch_dir / "*.npz")))
    if not files:
        raise SystemExit(f"no branch shards under {branch_dir}")
    points, states, branch_point = [], [], []
    sigma, count_start, count_end, noise = [], [], [], []
    offset = 0
    for path in files:
        data = np.load(path, allow_pickle=True)
        rows = data["points"].shape[0]
        points.append(data["points"])
        states.append(data["point_states"])
        branch_point.append(data["branch_point"] + offset)
        offset += rows
        sigma.append(data["branch_sigma"])
        count_start.append(data["branch_count_start"])
        count_end.append(data["branch_count_end"])
        noise.append(data["branch_noise"])
    return {
        "files": files,
        "points": np.concatenate(points),
        "states": np.concatenate(states),
        "branch_point": np.concatenate(branch_point),
        "sigma": np.concatenate(sigma),
        "count_start": np.concatenate(count_start),
        "count_end": np.concatenate(count_end),
        "noise": np.concatenate(noise),
    }


def recorded_chunks(rows: np.ndarray, action_values: np.ndarray, bounds: dict,
                    episode_of: np.ndarray, chunk_size: int, action_dim: int,
                    width: int = 16) -> np.ndarray:
    """`(batch, chunk_size, action_dim)` of the recorded actions, exactly as `train_dsrl.py` builds it.

    Future rows come from the same episode and clamp at its end, so a chunk never spans episodes.
    Only the first `width` columns are filled; the rest stay zero, matching `prepare_action`'s
    padding to `max_action_dim`.
    """
    out = np.zeros((len(rows), chunk_size, action_dim), dtype=np.float32)
    for position, row in enumerate(rows):
        row = int(row)
        _, stop = bounds[int(episode_of[row])]
        for offset in range(chunk_size):
            out[position, offset, :width] = action_values[min(row + offset, stop - 1)]
    return out


def pairwise_accuracy(scores: dict[int, np.ndarray], outcomes: dict[int, np.ndarray]):
    """Fraction of within-point discordant pairs the score orders correctly.

    Returns `(accuracy, pairs, per_point)`.  `per_point` is the list of
    `(point, n_correct, n_pairs)` so the caller can bootstrap over points.
    """
    correct = 0
    total = 0
    per_point = []
    for point, value in scores.items():
        target = outcomes[point]
        hits = 0
        seen = 0
        for i in range(len(value)):
            for j in range(i + 1, len(value)):
                if target[i] == target[j]:
                    continue
                seen += 1
                # A tie in the score counts as half a hit: it carries no ordering information but
                # should not be scored as an error either.
                if value[i] == value[j]:
                    hits += 0.5
                elif (value[i] > value[j]) == (target[i] > target[j]):
                    hits += 1
        if seen:
            per_point.append((point, hits, seen))
            correct += hits
            total += seen
    return (correct / total if total else float("nan")), total, per_point


def bootstrap(per_point, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile CI for the accuracy, resampling **points** rather than pairs."""
    if not per_point:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    hits = np.array([p[1] for p in per_point], dtype=np.float64)
    sizes = np.array([p[2] for p in per_point], dtype=np.float64)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(hits), len(hits))
        draws.append(hits[pick].sum() / sizes[pick].sum())
    draws = np.array(draws)
    return float(draws.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/datasets/a3_recap_all"))
    parser.add_argument(
        "--checkpoint", type=Path, action="append", default=None,
        help="a `train_dsrl.py` checkpoint; repeat to compare several.  Its action-space `critic` "
             "is the one scored.",
    )
    parser.add_argument(
        "--pretrained", type=Path,
        default=Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import train_pi05_launcher as launcher  # noqa: F401

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    from steering import SteeringActor, build_observation, encode_prefix, visual_feature

    checkpoints = args.checkpoint or []
    if not checkpoints:
        raise SystemExit("at least one --checkpoint is required")

    data = load_branches(args.dataset / "branches")
    valid = data["points"][:, 5] > 0.5
    print(f"[rank] {len(data['files'])} shards, {len(data['points'])} points "
          f"({int(valid.sum())} valid), {len(data['branch_point'])} branches", flush=True)

    # --- the noise floor: does an unperturbed repeat already vary? ---------------------
    # If it does, some of the discordant pairs are irreducible and no critic can score 1.0.
    floor = []
    for point in np.flatnonzero(valid):
        mask = (data["branch_point"] == point) & (data["sigma"] == 0.0)
        values = data["count_end"][mask]
        if len(values) > 1:
            floor.append(float(values.max() - values.min()))
    if floor:
        print(f"[rank] control (sigma=0) repeats: {len(floor)} points, "
              f"same outcome on {sum(1 for f in floor if f == 0)} of them, "
              f"spread {np.mean(floor):.2f} on the rest", flush=True)

    # --- point -> dataset row, and the recorded action table --------------------------
    point_rows = data["points"][:, 0].astype(np.int64)
    frame_table = pq.read_table(
        args.dataset / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "frame_index", "action"],
    ).to_pandas()
    episode_of = frame_table["episode_index"].to_numpy(np.int64)
    action_values = np.stack(frame_table["action"].values)[:, :16].astype(np.float32)

    labels = pq.read_table(args.dataset / "value_labels.parquet").to_pandas()
    bounds = {}
    offset = 0
    for position, length in enumerate(labels["frames"]):
        bounds[position] = (offset, offset + int(length))
        offset += int(length)

    # --- one batch of observations for every valid point ------------------------------
    policy = PI05Policy.from_pretrained(str(args.pretrained)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, _ = make_pre_post_processors(
        policy.config, str(args.pretrained),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset = LeRobotDataset(repo_id="dsrl", root=args.dataset, video_backend="pyav")
    probe = SteeringActor(checkpoints[0], device=args.device)

    wanted = np.flatnonzero(valid)
    rows = point_rows[wanted]
    observations = []
    for start in range(0, len(rows), args.batch):
        index = rows[start:start + args.batch]
        items = [dataset[int(row)] for row in index]
        raw = {key: torch.stack([item[key] for item in items])
               for key in dataset.meta.features if key.startswith("observation.images.")}
        raw["observation.state"] = torch.stack([item["observation.state"] for item in items])
        if "observation.history" in items[0]:
            raw["observation.history"] = torch.stack(
                [item["observation.history"] for item in items]
            )
        raw["task"] = [item["task"] for item in items]
        prepared = preprocess(raw)
        with torch.no_grad():
            prefix = encode_prefix(policy, prepared, keep_embeddings=True)
            visual = visual_feature(policy, prefix)
            projection = torch.from_numpy(probe.projection).to(visual.device)
            visual = visual @ projection
            observations.append(build_observation(
                prepared["observation.state"].float(),
                prefix.hidden_last if probe.parts.get("vlm", True) else None,
                visual if probe.parts.get("visual", True) else None,
                probe.parts,
            ).detach().cpu())
        print(f"    observations {min(start + args.batch, len(rows))}/{len(rows)}", flush=True)
    observation = torch.cat(observations, dim=0)
    point_of_row = {int(row): position for position, row in enumerate(rows)}

    # --- the perturbed action chunk per branch ----------------------------------------
    # `branch_rollouts_new.py` adds one 16-dim vector to the first 16 steps of the recorded chunk.
    # Rebuilding that chunk is what lets the critic be scored on the same (s, a) the environment
    # actually ran.
    base_chunks = recorded_chunks(rows, action_values, bounds, episode_of,
                                  args.chunk, args.action_dim)
    base_by_point = {int(row): base_chunks[position] for position, row in enumerate(rows)}
    del base_chunks

    branch_rows = point_rows[data["branch_point"]]
    branch_position = np.array([point_of_row.get(int(r), -1) for r in branch_rows], dtype=np.int64)
    keep = branch_position >= 0
    print(f"[rank] {int(keep.sum())} of {len(branch_position)} branches belong to a valid point",
          flush=True)

    sigma_by_branch = data["sigma"]
    noise_by_branch = data["noise"]
    outcome_by_branch = data["count_end"]
    point_by_branch = data["branch_point"]

    results = {}
    for checkpoint in checkpoints:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        # Read the width off the critic's own first layer rather than trusting `hidden`.  That
        # field only exists in checkpoints written after §9.96, and defaulting it rebuilt a
        # 512-wide critic at 2048 and died on `load_state_dict`.
        hidden = int(state["critic"]["0.net.net.0.weight"].shape[0])
        from steering import twin_critic  # noqa: PLC0415
        critic = twin_critic(probe.observation_dim, args.chunk * args.action_dim, hidden=hidden)
        critic.load_state_dict(state["critic"])
        critic = critic.to(args.device).eval()
        print(f"\n[rank] scoring {checkpoint.parent.name}/{checkpoint.name}"
              f"  (width {hidden})", flush=True)

        # Batch the critic over branches, grouped so each batch rebuilds its chunks once.
        chunks = np.zeros((int(keep.sum()), args.chunk, args.action_dim), dtype=np.float32)
        for slot, branch in enumerate(np.flatnonzero(keep)):
            row = int(branch_rows[branch])
            chunk = base_by_point[row].copy()
            if sigma_by_branch[branch] > 0:
                chunk[:16, :16] += noise_by_branch[branch][None, :]
            chunks[slot] = chunk
        scores = torch.zeros(len(chunks), device=args.device)
        for start in range(0, len(chunks), 64):
            batch_chunks = torch.from_numpy(chunks[start:start + 64]).to(args.device)
            batch_obs = observation[branch_position[keep][start:start + 64]].to(args.device)
            flat = batch_chunks.reshape(batch_chunks.shape[0], -1)
            with torch.no_grad():
                # `min` over the twin heads, matching the actor's own objective.
                value = torch.min(torch.stack([head(batch_obs, flat) for head in critic]), dim=0).values
            scores[start:start + 64] = value.squeeze(-1)
        scores = scores.cpu().numpy()

        by_point: dict[int, list[int]] = {}
        for slot, branch in enumerate(np.flatnonzero(keep)):
            by_point.setdefault(int(point_by_branch[branch]), []).append(slot)
        score_map = {point: scores[slots] for point, slots in by_point.items()}
        outcome_map = {point: outcome_by_branch[slots] for point, slots in by_point.items()}

        accuracy, pairs, per_point = pairwise_accuracy(score_map, outcome_map)
        mean, low, high = bootstrap(per_point, n=args.bootstrap)
        scanned = int((scores > 1e4).sum())
        results[str(checkpoint)] = {
            "accuracy": accuracy, "pairs": pairs, "points": len(per_point),
            "bootstrap_mean": mean, "ci_low": low, "ci_high": high,
            "score_mean": float(scores.mean()), "score_std": float(scores.std()),
            "width": hidden, "absolute_scores_over_1e4": scanned,
        }
        print(f"  pairwise accuracy {accuracy:.3f}  over {pairs} pairs from"
              f" {len(per_point)} points", flush=True)
        print(f"  bootstrap over points: {mean:.3f}  [{low:.3f}, {high:.3f}]", flush=True)
        print(f"  Q range {scores.min():.2f}..{scores.max():.2f}  mean {scores.mean():.2f}", flush=True)

    print("\n结果")
    print(f"  {'checkpoint':28s} {'accuracy':>9s} {'CI':>16s} {'pairs':>6s} {'points':>7s}")
    for name, row in results.items():
        short = Path(name).parent.name
        print(f"  {short:28s} {row['accuracy']:9.3f}"
              f" {row['ci_low']:7.3f}..{row['ci_high']:.3f} {row['pairs']:6d} {row['points']:7d}")
    print()
    print("  判读：")
    print("    0.50 ± 0.05  -> critic 对动作无排序能力。DSRL 的 argmax 是在噪声里随机挑，")
    print("                    这解释了为什么 critic 越准、actor 越差。方法在此不可用。")
    print("    0.55~0.65    -> 有弱排序能力。actor 需要更多约束（更小的 shift、或只用排序")
    print("                    可靠的样本），而不是更多训练。")
    print("    > 0.70       -> 排序可信。此前的失败应归因于偏移幅度或参数化，不是 critic。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
