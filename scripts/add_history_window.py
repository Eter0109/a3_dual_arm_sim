#!/usr/bin/env python3
"""Add a proprioceptive history window and progress labels to a dataset.

Why a window and not the frame
-----------------------------
v4 places everything it picks up and stops after one batch of five on a task that needs
four (9.43).  Nothing in a sample says how far along it is, because pi0.5 is
`n_obs_steps = 1`.  v5 showed the information exists -- one seed moved 15 Cookies where
every earlier arm stayed at or below 5 -- but it was privileged and its attribution was
muddy (9.43).

This builds the input for a *history encoder*: the last `steps` frames of
``[q, q_dot, a_prev]`` per frame, which is the robot's own recent trajectory and nothing
else.  Deliberately no images: the encoder cannot represent "the Cookies look messy", only
"where the arm has been and what it did", so the information bottleneck is in the input
modality rather than in a loss term.  The task's progress is derivable from that -- the
hand's visits to the target box are in the window -- while the visual state of the pile is
not.

Progress labels are stored too, but as a **training target only**: they supervise an
auxiliary head that is discarded at inference, so they are legitimate supervision in the
way an auxiliary task is, and never an input.

    python scripts/add_history_window.py --source <in> --out <out> --repo-id local/x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Per-frame encoder input: 16 joint positions (the grippers are entries 7 and 15, so they
#: are already inside the state), 16 joint velocities, 16 previous action targets.
#: Repeating the gripper separately would duplicate the state for the only two entries the
#: jaw contributes.
PER_STEP = 48
#: Where the effector's y coordinate lives in the 14-wide eef pose (two 7-wide blocks, left
#: first, each xyz + quaternion).
EEF_LEFT_Y = 1
#: The target box sits at y = 0.100 and the source at y = 0.315; a visit reads y < 0.18.
#: Measured on the v4 set, this labels 570 of 570 20-cookie episodes as exactly four
#: deliveries (9.43).
TARGET_BOX_Y = 0.18
MIN_VISIT_FRAMES = 5
MIN_GAP_FRAMES = 20


def count_deliveries(eef: np.ndarray) -> np.ndarray:
    """Running count of visits to the target box, one value per frame (right-continuous).

    Same rule as `add_progress_channels.py`, and the same rule the evaluation driver
    applies online, so the label means one thing everywhere it appears.
    """

    inside = eef[:, EEF_LEFT_Y] < TARGET_BOX_Y
    counts = np.zeros(len(inside), dtype=np.float32)
    total = 0
    index = 0
    last_exit = -(10 ** 9)
    while index < len(inside):
        if inside[index]:
            end = index
            while end < len(inside) and inside[end]:
                end += 1
            if end - index >= MIN_VISIT_FRAMES and (index - last_exit) >= MIN_GAP_FRAMES:
                total += 1
            last_exit = end
            counts[index:end] = total
            index = end
        else:
            counts[index] = total
            index += 1
    return counts


def build_window(frames: np.ndarray, steps: int) -> np.ndarray:
    """``(T, PER_STEP)`` frames to ``(T, steps * PER_STEP)`` windows, oldest first.

    One window per frame, so window ``t`` is the last ``steps`` frames ending at ``t``.  The
    first ``steps - 1`` windows would reach before the episode starts and are padded by
    repeating frame zero rather than with zeros: a zero history is a configuration the
    robot is never in (every joint at 0 is a pose, and not a reachable one here), so
    padding with it would teach the encoder that "episode start" looks like an impossible
    state.  Repeating the first frame says "nothing has happened yet", which is true.
    """

    frames = np.asarray(frames, dtype=np.float32)
    if not np.all(np.isfinite(frames)):
        raise ValueError("non-finite values in the history frames")
    if len(frames) == 0:
        return np.zeros((0, steps * PER_STEP), dtype=np.float32)
    padded = np.concatenate([np.repeat(frames[:1], steps - 1, axis=0), frames], axis=0)
    windows = np.stack([padded[i: i + steps] for i in range(len(frames))])
    return windows.reshape(len(frames), steps * PER_STEP)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--steps", type=int, default=10,
                        help="Window length in frames (default 10 = 0.5 s at 20 Hz)")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--cookies-per-batch", type=int, default=5)
    args = parser.parse_args()

    import os
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    def _write_meta(target, text):
        """Replace a metadata file rather than overwriting it.

        The dataset was staged with hard links, so `write_text` on an existing path truncates
        the *shared inode* and rewrites the source dataset's metadata too -- which is exactly
        what happened: v5's 18-wide `info.json` landed in v4 and left it declaring 18 for
        16-wide data.  Unlinking first breaks the link; the write then creates a new file.
        """

        target.unlink(missing_ok=True)
        target.write_text(text, encoding="utf-8")

    files = sorted((args.source / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.source}/data")
    table = pq.read_table(files)
    columns = {name: np.stack(table[name].to_numpy()) for name in
               ("observation.state", "observation.velocity", "action",
                "observation.eef_pose", "episode_index")}
    state = columns["observation.state"].astype(np.float32)
    velocity = columns["observation.velocity"].astype(np.float32)
    action = columns["action"].astype(np.float32)
    eef = columns["observation.eef_pose"].astype(np.float32)
    episodes = np.asarray(columns["episode_index"])
    print(f"{len(state)} frames, {len(set(episodes.tolist()))} episodes, "
          f"state {state.shape[1]}, velocity {velocity.shape[1]}, action {action.shape[1]}")

    if state.shape[1] != 16 or velocity.shape[1] != 16 or action.shape[1] != 16:
        raise SystemExit("expected the canonical 16-wide A3 state/velocity/action")

    history = np.zeros((len(state), args.steps * PER_STEP), dtype=np.float32)
    progress = np.zeros((len(state), 2), dtype=np.float32)
    histogram = {}
    for episode in sorted(set(episodes.tolist())):
        rows = np.flatnonzero(episodes == episode)
        # The action at frame t is the target for t, so the previous action is t-1.  Frame
        # zero has no predecessor; it repeats its own, which pairs with the first-frame
        # padding above.
        previous = np.concatenate([action[rows[:1]], action[rows[:-1]]], axis=0)
        frames = np.concatenate([state[rows], velocity[rows], previous], axis=1)
        history[rows] = build_window(frames, args.steps)

        deliveries = count_deliveries(eef[rows])
        cookies = np.minimum(deliveries * args.cookies_per_batch,
                             args.max_batches * args.cookies_per_batch)
        # Same [-1, 1] mapping as the state: the auxiliary head predicts these directly, so
        # keeping them on the same scale as the rest of the model's numbers is one fewer
        # conversion to get wrong.
        progress[rows, 0] = (deliveries / args.max_batches) * 2.0 - 1.0
        progress[rows, 1] = (cookies / (args.max_batches * args.cookies_per_batch)) * 2.0 - 1.0
        key = int(deliveries.max())
        histogram[key] = histogram.get(key, 0) + 1

    print(f"deliveries per episode: {dict(sorted(histogram.items()))}")
    print(f"history window: {args.steps} steps x {PER_STEP} = {history.shape[1]} floats")

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.source, args.out, copy_function=os.link)

    # `FixedSizeListArray.from_arrays` rather than `pa.array(history.tolist())`: the column is
    # 1.17M rows by 480 floats, and materialising that as Python objects costs tens of GB
    # and minutes, where the flat-buffer constructor is a reshape.
    def fixed_list(column: np.ndarray) -> pa.FixedSizeListArray:
        flat = pa.array(np.ascontiguousarray(column, dtype=np.float32).reshape(-1),
                        type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, list_size=column.shape[1])

    table = table.append_column("observation.history", fixed_list(history))
    table = table.append_column("observation.progress", fixed_list(progress))

    targets = sorted((args.out / "data").glob("chunk-*/*.parquet"))
    for target in targets:
        target.unlink()
    pq.write_table(table, targets[0] if targets else
                   (args.out / "data" / "chunk-000" / "file-000.parquet"))

    info_path = args.out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.history"] = {
        "dtype": "float32", "shape": [history.shape[1]],
        "names": {"motors": [f"history_{i}" for i in range(history.shape[1])]},
    }
    info["features"]["observation.progress"] = {
        "dtype": "float32", "shape": [2],
        "names": {"motors": ["batches_fraction", "cookies_fraction"]},
    }
    info["total_frames"] = int(len(history))
    info_path.unlink(missing_ok=True)
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    stats_path = args.out / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())

        def describe(column: np.ndarray) -> dict[str, list[float]]:
            return {
                "mean": column.mean(axis=0).tolist(),
                "std": column.std(axis=0).tolist(),
                "min": column.min(axis=0).tolist(),
                "max": column.max(axis=0).tolist(),
                "q01": np.quantile(column, 0.01, axis=0).tolist(),
                "q99": np.quantile(column, 0.99, axis=0).tolist(),
            }

        stats["observation.history"] = describe(history)
        stats["observation.progress"] = describe(progress)
        stats_path.unlink(missing_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
