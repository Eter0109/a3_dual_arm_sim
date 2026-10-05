"""Fit the linear read-out that the closed-loop ridge rollout will execute.

The probe (`probe_observation_information.py`) showed that a ridge regression on the
fine-tuned vision tower's wrist-camera features plus the joint state predicts the expert's
next action to **0.0061 rad** cross-episode, against a 0.0075 rad expert tolerance and
0.0185 for the best VLA.  That was measured one frame at a time with the expert's own
states, so it says the observation *contains* the information -- it does not say the
information survives being acted on.

This script turns that probe into a deployable artefact: it writes the normalisation
statistics, the ridge weights and the block layout to one `.npz`, which
`scripts/adapter_ridge_cookie.py` loads and executes through the bridge.

Two fits are computed
---------------------
* **Reported**: held-out episodes only.  Standardisation stats and the ridge penalty come
  from a slice of the training episodes, and the error is reported on 60 episodes the fit
  never saw.  This is the honest number, and it should reproduce the probe's 0.0061.
* **Deployed**: everything.  Once the penalty has been chosen, refitting on all 400
  episodes is strictly more data and is what a policy should ship.  The statistics here
  are also computed over all frames, and they are the ones the adapter must apply at
  inference -- a mismatch between the two would show up as a constant offset in the
  rollout and read as a modelling failure.

Usage::

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/fit_ridge_cookie_policy.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
# The ridge maths lives in the probe, which is where it was validated; importing rather
# than re-deriving it is what keeps the deployed coefficients and the reported error from
# drifting apart.  Importing is safe: the probe only runs its CLI under `__main__`.
sys.path.insert(0, str(PROJECT / "scripts"))

from probe_observation_information import fit_ridge, score  # noqa: E402

#: Same grid the probe searched, so the reported number is comparable to it.
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)

#: The camera names in the feature cache, and which block name they carry.
VISION_BLOCKS = {"front": "base_0_rgb", "wrist": "left_wrist_0_rgb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_cookie_one_pi05"))
    parser.add_argument("--cache", type=Path,
                        default=Path("/root/autodl-tmp/logs/vision_features.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("/root/autodl-tmp/logs/ridge_cookie_policy.npz"))
    parser.add_argument("--blocks", default="state,wrist",
                        help="feature blocks in order, from state/front/wrist. "
                             "'state,wrist' is the probe's best combination (0.0061 rad)")
    parser.add_argument("--holdout-episodes", type=int, default=60)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args()


def load_labels(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Episodes, 8-wide actions and 8-wide states, in dataset frame order.

    The frame order has to match the feature cache's, and it does for the same reason the
    probe's does: both are the dataset's own order, and the cache was encoded with
    ``--stride 1``.  A stride other than one would need the same stride applied here.
    """

    import pandas as pd

    table = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
         for f in sorted(root.glob("data/**/*.parquet"))],
        ignore_index=True,
    )
    episodes = table["episode_index"].to_numpy()
    action = np.stack([np.asarray(v, dtype=np.float64) for v in table["action"]])[:, :8]
    state = np.stack([np.asarray(v, dtype=np.float64) for v in table["observation.state"]])[:, :8]
    return episodes, action, state


def standardise_with(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The stats a block is scaled by, plus the scaled values."""

    mean, scale = reference.mean(0), reference.std(0) + 1e-6
    return mean, scale, (values - mean) / scale


def assemble(blocks: list[str], raw: dict[str, np.ndarray],
             stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Concatenate the blocks in the declared order, each scaled by its own stats."""

    parts = []
    for name in blocks:
        mean, scale = stats[name]
        parts.append((raw[name] - mean) / scale)
    return np.hstack(parts)


def main() -> int:
    args = parse_args()

    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    if not blocks:
        raise SystemExit("--blocks is empty")
    unknown = [b for b in blocks if b not in ("state", *VISION_BLOCKS)]
    if unknown:
        raise SystemExit(f"unknown block(s) {unknown}; expected state and/or {sorted(VISION_BLOCKS)}")

    episodes, action, state = load_labels(args.root)
    saved = np.load(args.cache)
    raw: dict[str, np.ndarray] = {"state": state}
    for name in blocks:
        if name in VISION_BLOCKS:
            raw[name] = saved[VISION_BLOCKS[name]]
    lengths = {name: len(values) for name, values in raw.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != len(action):
        raise SystemExit(
            f"the feature cache and the labels disagree on length: {lengths} vs {len(action)} "
            "frames.  Re-encode with `probe_observation_information.py --stride 1`."
        )
    print(f"帧 {len(action)}   集 {len(np.unique(episodes))}   特征块 {blocks}")

    # -- reported: held-out episodes -------------------------------------
    rng = np.random.default_rng(args.split_seed)
    pool = np.unique(episodes)
    rng.shuffle(pool)
    test_episodes = set(pool[: args.holdout_episodes].tolist())
    te = np.array([e in test_episodes for e in episodes])
    tr = ~te

    cv_stats = {
        name: (values[tr].mean(0), values[tr].std(0) + 1e-6) for name, values in raw.items()
    }
    Xcv = assemble(blocks, raw, cv_stats)

    cut = int(tr.sum() * 0.85)
    tridx = np.flatnonzero(tr)
    a_idx, b_idx = tridx[:cut], tridx[cut:]
    best, alpha = np.inf, ALPHAS[0]
    for candidate in ALPHAS:
        value, _, _ = score(fit_ridge(Xcv[a_idx], action[a_idx], candidate),
                            Xcv[b_idx], action[b_idx])
        if value < best:
            best, alpha = value, candidate
    mean, median, _ = score(fit_ridge(Xcv[tr], action[tr], alpha), Xcv[te], action[te])
    hold = np.abs(state[te] - action[te]).max(axis=1).mean()
    print(f"留出 {int(te.sum())} 帧 / {args.holdout_episodes} 集   alpha={alpha:g}")
    print(f"  误差 mean {mean:.4f}  median {median:.4f}   （原地不动 {hold:.4f}）")

    # -- deployed: everything --------------------------------------------
    stats = {name: (values.mean(0), values.std(0) + 1e-6) for name, values in raw.items()}
    X = assemble(blocks, raw, stats)
    weights, x_mean, y_mean = fit_ridge(X, action, alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "weights": weights.astype(np.float64),
        "x_mean": x_mean.astype(np.float64),
        "y_mean": y_mean.astype(np.float64),
        "alpha": np.array(alpha, dtype=np.float64),
        "blocks": np.array(",".join(blocks)),
        "holdout_mean_error": np.array(mean, dtype=np.float64),
    }
    for name, (block_mean, block_scale) in stats.items():
        payload[f"{name}_mean"] = block_mean.astype(np.float64)
        payload[f"{name}_scale"] = block_scale.astype(np.float64)
    np.savez(args.output, **payload)

    print(f"已写出 {args.output}")
    print(f"  weights {weights.shape}  x_mean {x_mean.shape}  y_mean {y_mean.shape}")
    for name in blocks:
        block_mean, block_scale = stats[name]
        print(f"  {name:<6} mean {block_mean.shape}  scale {block_scale.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
