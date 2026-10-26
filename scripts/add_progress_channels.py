#!/usr/bin/env python3
"""Add progress channels to a dataset's state: how many batches have been delivered.

Why this exists
---------------
The v4 arm places everything it picks up (moved-out equals scored in all eight episodes)
but it stops after **one batch of five**, on a task that needs four.  The model has no
channel that says where it is in the task: pi0.5 is `n_obs_steps = 1`, so each sample sees
one frame and one instruction, and nothing distinguishes "the box is empty" from "two
batches are already in".

The state is the place to put it, and on pi0.5 that is cheap: the state is not projected by
a learned layer, it is **discretised into 256 bins and written into the prompt text**

    Task: transfer 20 cookies into target box, State: 128 45 200 ...;\nAction:

so widening it touches no weights -- the 4.05 checkpoint stays loadable, and the prompt
carries two more numbers.

What the channels are
---------------------
``d16``  batches delivered / 4, in [-1, 1] (the range the tokenizer assumes)
``d17``  cookies in the box / 20, in [-1, 1]

Batch deliveries are counted from the gripper channel, not guessed from a clock: the left
gripper's last state entry closes from ~1.0 to ~0.25 when it grips and returns to ~1.0 when
it lets go, so a falling edge that crosses the open/closed midpoint is one release.  Cookies
in the box are then ``5 x releases``, which is exact for the expert demonstrations these
datasets contain -- every release succeeded, which is what the acceptance filter guarantees.

Both are written as **new columns at the right index positions**, so a reader that knows
about 16 state entries still works and one that knows about 18 gets the progress.

Risk this does not remove
-------------------------
The demonstrations only ever contain releases at 5, 10, 15, 20, while the model's own
episodes will contain 3, 7, 12.  The batch count is the more robust channel for exactly
that reason -- it is an integer 0..4 whatever happened to the Cookies.

    python scripts/add_progress_channels.py --source <in> --out <out> --repo-id local/x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Where in `observation.eef_pose` the left effector's y coordinate lives.
#: The 14-wide A3 eef pose is two 7-wide blocks (left, right), each xyz + quaternion,
#: so [0:3] is the left arm's position.
EEF_LEFT_Y = 1
#: The target box sits at y = 0.100 with half-width 0.0508, the source box at y = 0.315
#: with half-width 0.101.  A visit to the target box therefore reads y < 0.18 with a wide
#: margin on both sides.  Measured on the v4 dataset: this classifies 570 of 570 20-cookie
#: episodes as exactly four deliveries.
TARGET_BOX_Y = 0.18
#: A delivery is a contiguous stretch inside the box, and re-entering within this many
#: frames is the same delivery rather than a new one -- the hand often dips back in while
#: releasing.
MIN_VISIT_FRAMES = 5
MIN_GAP_FRAMES = 20
#: `max_state_dim` is 32 on pi0.5 and the state is written into the prompt as integers, so
#: an extra two entries cost a handful of tokens.
CHANNELS = ("batches_delivered", "cookies_in_box")


def count_deliveries(eef: np.ndarray) -> np.ndarray:
    """Running count of visits to the target box, one value per frame.

    Position, not the gripper.  The jaw channel was tried first and is unusable: it sits at
    0.31 for most of an episode and only reaches 0.49-0.58 on a release, so any threshold
    either misses releases (0.62 gave three of four) or counts jaw chatter (0.45 gave a
    median of six).  The effector's y coordinate separates the two boxes cleanly -- they are
    215 mm apart and the hand crosses all of it on every delivery.

    The count is right-continuous: the frame that enters the box already reads the new
    total, because the chunk predicted from that frame is the one that should know the
    delivery is under way.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--max-batches", type=int, default=4,
        help="Divisor for the batch channel: 4 for the 20-cookie task, 2 for 10.",
    )
    parser.add_argument(
        "--cookies-per-batch", type=int, default=5,
        help="Multiplier turning releases into a cookie count.",
    )
    parser.add_argument("--report-only", action="store_true",
                        help="Print the counts per episode and write nothing.")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((args.source / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.source}/data")

    table = pq.read_table(files)
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    eef = np.stack(table["observation.eef_pose"].to_numpy()).astype(np.float32)
    episodes = np.asarray(table["episode_index"].to_numpy())
    print(f"{len(state)} frames, {len(set(episodes.tolist()))} episodes, "
          f"state width {state.shape[1]}, eef width {eef.shape[1]}")

    new_state = np.empty((len(state), state.shape[1] + len(CHANNELS)), dtype=np.float32)
    new_state[:, : state.shape[1]] = state
    summary = []
    for episode in sorted(set(episodes.tolist())):
        rows = np.flatnonzero(episodes == episode)
        counts = count_deliveries(eef[rows])
        cookies = np.minimum(counts * args.cookies_per_batch,
                             args.max_batches * args.cookies_per_batch)
        # Both channels are mapped into [-1, 1] because the tokenizer discretises over that
        # range; a raw 0..4 would clip into the top bin and lose all resolution.
        new_state[rows, state.shape[1]] = (counts / args.max_batches) * 2.0 - 1.0
        new_state[rows, state.shape[1] + 1] = (
            cookies / (args.max_batches * args.cookies_per_batch)
        ) * 2.0 - 1.0
        summary.append((episode, int(counts.max()), len(rows)))

    print(f"\n{'episode':>8} {'deliveries':>11} {'frames':>7}")
    for episode, deliveries, frames in summary[:12]:
        print(f"{episode:>8} {deliveries:>11} {frames:>7}")
    if len(summary) > 12:
        print(f"  ... {len(summary) - 12} more")
    histogram = {}
    for _episode, deliveries, _frames in summary:
        histogram[deliveries] = histogram.get(deliveries, 0) + 1
    print(f"\ndeliveries per episode: {dict(sorted(histogram.items()))}")

    if args.report_only:
        return 0

    # The dataset has to stay a valid LeRobot dataset, so the camera streams and the
    # episode metadata are hard-linked from the source (free on this filesystem, and the
    # rewrite only touches the parquet data and `info.json`), and only the state column
    # and the declared feature shape change.  Writing a fresh dataset from frames would
    # mean decoding and re-encoding every video to change two numbers.
    if args.out.exists():
        import shutil
        shutil.rmtree(args.out)
    import os
    import shutil as _shutil
    _shutil.copytree(args.source, args.out, copy_function=os.link)

    index = table.schema.get_field_index("observation.state")
    table = table.set_column(index, pa.field("observation.state", pa.list_(pa.float32())),
                             pa.array(new_state.tolist(), type=pa.list_(pa.float32())))
    for target in sorted((args.out / "data").glob("chunk-*/*.parquet")):
        target.unlink()
    pq.write_table(table, sorted((args.out / "data").glob("chunk-*/*.parquet"))[0]
                   if list((args.out / "data").glob("chunk-*/*.parquet")) else
                   (args.out / "data" / "chunk-000" / "file-000.parquet"))

    info_path = args.out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["shape"] = [new_state.shape[1]]
    info["total_frames"] = int(len(new_state))
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    # The normalisation statistics are per-dimension, so the two new channels need entries
    # or the normaliser drops them (or fails).  They are computed from the values just
    # written, and quantiles are what pi0.5's normalisation mapping asks for.
    stats_path = args.out / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        column = new_state
        entry = {
            "mean": column.mean(axis=0).tolist(),
            "std": column.std(axis=0).tolist(),
            "min": column.min(axis=0).tolist(),
            "max": column.max(axis=0).tolist(),
            "q01": np.quantile(column, 0.01, axis=0).tolist(),
            "q99": np.quantile(column, 0.99, axis=0).tolist(),
        }
        stats["observation.state"] = entry
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    (args.out / "progress_report.json").write_text(json.dumps(
        {"episodes": len(summary), "max_batches": args.max_batches,
         "cookies_per_batch": args.cookies_per_batch,
         "deliveries_per_episode": dict(sorted(histogram.items())),
         "rows": summary}, indent=2) + "\n")
    print(f"\nwrote {args.out} with {new_state.shape[1]}-wide state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
