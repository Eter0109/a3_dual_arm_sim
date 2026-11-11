#!/usr/bin/env python3
"""How much of the progress label is in the window, and how much is in one frame?

The history encoder was designed on the premise that the last ten frames of
``[q, q_dot, a_previous]`` carry task progress and a single frame cannot.  Its R^2 of 0.853 on
that label is consistent with the premise but does not test it: the arm's own joint angles
encode *where in the source box it is reaching*, and the source box is emptied in a consistent
order, so "how far along" may be readable from one pose with no history at all.

This measures the two contributions separately, offline and on CPU:

  * **window length**: the same ridge regression over 1, 2, 5, 10, 20, 50, 100 frames.
    If R^2 is flat from 1 to 10 frames, the encoder is being handed a problem that does not
    need a recurrence.
  * **which coordinates**: the full 48 per frame against the 24 that move (left arm only) and
    the 16 that are position-only.  The right arm is frozen for every episode in this dataset,
    so its 24 columns are a constant offset -- which a linear layer can absorb exactly, making
    them free to keep but also free to drop.

Ridge rather than the encoder itself, deliberately: this is an upper bound on what is linearly
decodable, and it can be computed in seconds.  It does not prove the GRU rejects the extra
information, only that there is nothing extra there to reject.

Split is by episode, not by frame.  Frames inside one episode are a smooth trajectory, so a
frame-wise split leaks the answer: consecutive frames are nearly identical, and a model can
score well by interpolating between neighbours it has already seen.

    python scripts/probe_history_information.py --limit 60000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

#: Per-frame layout, matching `add_history_window.py`: [q(16), q_dot(16), a_previous(16)].
#: Within each block, entries 0-7 are the left arm and gripper, 8-15 the right.
LEFT = list(range(0, 8)) + list(range(16, 24)) + list(range(32, 40))
RIGHT = list(range(8, 16)) + list(range(24, 32)) + list(range(40, 48))
POSITION = list(range(0, 16)) + list(range(32, 48))   # q and a_previous, no q_dot


def load(root: Path, limit: int):
    """Frames, per-frame features, progress labels and episode ids."""
    state, vel, act, prog, epi = [], [], [], [], []
    files = sorted(glob.glob(str(root / "data" / "chunk-000" / "*.parquet")))
    for path in files:
        table = pq.read_table(
            path,
            columns=["observation.state", "observation.velocity", "action",
                     "observation.progress", "episode_index"],
        )
        state.append(np.stack(table.column("observation.state").to_pylist()))
        vel.append(np.stack(table.column("observation.velocity").to_pylist()))
        act.append(np.stack(table.column("action").to_pylist()))
        prog.append(np.stack(table.column("observation.progress").to_pylist()))
        epi.append(np.asarray(table.column("episode_index").to_pylist()))
        if sum(len(x) for x in state) >= limit:
            break
    frames = np.concatenate([np.concatenate(state, 0), np.concatenate(vel, 0),
                             np.concatenate(act, 0)], axis=1).astype(np.float32)
    return frames, np.concatenate(prog, 0).astype(np.float32), np.concatenate(epi, 0)


def build_windows(frames: np.ndarray, episodes: np.ndarray, steps: int) -> np.ndarray:
    """One window per frame, oldest first, padded by repeating frame zero.

    The padding rule is the dataset's own (`add_history_window.build_window`), so the numbers
    here describe what the encoder is actually trained on rather than an idealised version.
    Windows never cross an episode boundary.
    """
    out = np.zeros((len(frames), steps * frames.shape[1]), dtype=np.float32)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        block = frames[rows]
        padded = np.concatenate([np.repeat(block[:1], steps - 1, axis=0), block], axis=0)
        view = np.lib.stride_tricks.sliding_window_view(padded, steps, axis=0)
        # (T, width, steps) -> (T, steps, width) -> flatten
        out[rows] = view.transpose(0, 2, 1).reshape(len(rows), steps * frames.shape[1])
    return out


def ridge_r2(x_train, y_train, x_test, y_test, alpha: float) -> float:
    """R^2 on held-out episodes, with an intercept and a closed-form solution."""
    mean_x = x_train.mean(axis=0, keepdims=True)
    mean_y = float(y_train.mean())
    xt = x_train - mean_x
    xs = x_test - mean_x
    gram = xt.T @ xt
    gram[np.diag_indices_from(gram)] += alpha
    coef = np.linalg.solve(gram, xt.T @ (y_train - mean_y))
    pred = xs @ coef + mean_y
    residual = float(np.sum((y_test - pred) ** 2))
    total = float(np.sum((y_test - y_test.mean()) ** 2))
    return 1.0 - residual / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_clean_v9h"))
    parser.add_argument("--limit", type=int, default=60000,
                        help="frames to read; windows make the feature count large")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--task", choices=("all", "2x5", "2x10"), default="all")
    args = parser.parse_args()

    frames, progress, episodes = load(args.data, args.limit)

    # `progress[:, 0]` is "batches delivered", which is what the auxiliary head predicts.
    # Filtering by task keeps the label ranges comparable: 2x5 tops out at 2, 2x10 at 4.
    if args.task != "all":
        tasks = {}
        with (args.data / "a3_episode_metadata.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                tasks[row["episode_index"]] = row["task"]
        keep = np.array([("2x10" in tasks.get(int(e), "")) == (args.task == "2x10")
                         for e in episodes])
        frames, progress, episodes = frames[keep], progress[keep], episodes[keep]

    # Episode-level split.  Frames within one episode are near-duplicates, so splitting by
    # frame would report the accuracy of interpolating between neighbours.
    unique = np.unique(episodes)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(unique)
    cut = int(len(shuffled) * 0.7)
    train_eps, test_eps = set(shuffled[:cut].tolist()), set(shuffled[cut:].tolist())
    train = np.array([int(e) in train_eps for e in episodes])
    test = ~train

    y = progress[:, 0]
    print(f"frames {len(frames)}  episodes {len(unique)} "
          f"(train {len(train_eps)}, test {len(test_eps)})  label range "
          f"[{y.min():.2f}, {y.max():.2f}]")
    print()

    subsets = {"all 48": None, "left 24": LEFT, "position 32": POSITION,
               "velocity 16": list(range(16, 32))}
    header = "steps  " + "".join(f"{name:>12s}" for name in subsets)
    print(header)
    print("-" * len(header))
    for steps in args.steps:
        windows = build_windows(frames, episodes, steps)
        row = f"{steps:5d}  "
        for name, cols in subsets.items():
            if cols is None:
                block = windows
            else:
                # Column `c` of frame `k` sits at `k * 48 + c` in the flattened window.
                index = [k * 48 + c for k in range(steps) for c in cols]
                block = windows[:, index]
            r2 = ridge_r2(block[train], y[train], block[test], y[test], args.alpha)
            row += f"{r2:12.4f}"
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
