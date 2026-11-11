"""Build a training dataset from a subset of an existing one, at episode granularity.

Why episode granularity, and not a frame-level cut
--------------------------------------------------
The obvious refinement would be to truncate a bad episode where its potential stops
increasing, keeping the good approach and dropping the fumbling tail.  That is not possible
here without rewriting video: LeRobot v3 concatenates many episodes into each mp4 and addresses
them by **absolute** `from_timestamp` / `to_timestamp` in that file, so shortening one episode
means either re-encoding the shard or leaving the extra frames in place and hoping nothing reads
them.

Dropping *whole* episodes needs none of that.  Video timestamps are absolute positions, so an
excluded episode simply leaves a gap nobody seeks to, and the remaining episodes keep their
original file_index and timestamps unchanged.  Only the parquet row numbers move, which is
local, checkable arithmetic -- and this project has already paid twice for dataset surgery done
without a verification step (merge_parts' per-camera video files, and the label misalignment
that silently trained a value head on 98 shuffled rows).

What has to be rewritten
------------------------
* `data/*.parquet`: keep the selected rows, renumber `episode_index` densely from 0, rebuild
  `index` as the global row counter.  `frame_index` is per-episode and stays as it is.
* `meta/episodes/*.parquet`: keep the selected rows, renumber `episode_index`, rebuild
  `dataset_from_index` / `dataset_to_index` to the new cumulative row offsets.  The video
  `file_index`, `from_timestamp` and `to_timestamp` columns are copied **verbatim** -- that is
  what makes the sharded videos still correct after the cut.
* `meta/tasks.parquet`: copied unchanged.  Unused task rows are harmless and keeping the table
  identical means `task_index` values stay valid, which removes an entire class of rewrite bug.
* `videos/`, `images/`, `traces/`: hard-linked, not copied.

The output is a full standalone dataset: it loads, it trains, and it can be evaluated.

Usage
-----
    python scripts/recap/subset_dataset.py \
        --dataset /root/autodl-tmp/datasets/a3_recap_all \
        --labels  /root/autodl-tmp/datasets/a3_recap_all/value_labels.parquet \
        --output  /root/autodl-tmp/datasets/a3_recap_succ \
        --select  success
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def link_tree(source: Path, target: Path) -> int:
    """Hard-link every file under `source` into `target`, preserving layout."""
    linked = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, destination)
        linked += 1
    return linked


def select_episodes(labels, select: str, min_potential: float) -> np.ndarray:
    """Boolean mask over label rows, which are in dataset episode order.

    `labels` is `value_labels.parquet`, whose row order is the dataset's episode order -- the
    same positional mapping `train_value.py` relies on and `build_value_labels.py --dataset`
    now verifies.  The mask is checked against the episodes table below rather than trusted.
    """
    success = labels["success"].to_numpy(dtype=bool)
    potential = labels["value_target"].to_numpy(dtype=np.float64)
    if select == "success":
        return success
    if select == "potential":
        return potential >= min_potential
    raise SystemExit(f"unknown --select {select!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--select",
        choices=("success", "potential"),
        default="success",
        help=(
            "`success` keeps episodes that met the strict contract; `potential` keeps episodes "
            "whose layered potential reached --min-potential, which admits the near misses that "
            "put every Cookie in the box but left one or two unseated."
        ),
    )
    parser.add_argument("--min-potential", type=float, default=28.0)
    args = parser.parse_args(argv)

    dataset, output = args.dataset, args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    labels = pq.read_table(args.labels).to_pandas()
    episodes_path = dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pq.read_table(episodes_path).to_pandas()
    if len(labels) != len(episodes):
        raise SystemExit(
            f"{len(labels)} label rows against {len(episodes)} episodes; they describe different sets"
        )

    # Positional pairing between labels and episodes is the assumption everything else rests on.
    # An episode's frame count is a fingerprint of it, so a mismatch means the two files are not
    # in the same order and any subset built from them would be wrong in a way that still trains.
    agree = (labels["frames"].to_numpy() == episodes["length"].to_numpy()).sum()
    if agree != len(labels):
        raise SystemExit(
            f"only {agree}/{len(labels)} label rows match their episode's frame count; "
            "the label file and the dataset are not in the same order"
        )

    mask = select_episodes(labels, args.select, args.min_potential)
    keep = np.flatnonzero(mask)
    if len(keep) == 0:
        raise SystemExit("selection is empty")

    # Old episode_index -> new.  Episodes are numbered densely in the output so LeRobot's
    # `episode_index` stays a contiguous range.
    remap = np.full(len(episodes), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep), dtype=np.int64)

    old_lengths = episodes["length"].to_numpy(dtype=np.int64)
    new_lengths = old_lengths[keep]
    total_frames = int(new_lengths.sum())
    old_total = int(old_lengths.sum())
    print(f"[subset] {len(keep)}/{len(episodes)} episodes, {total_frames}/{old_total} frames"
          f" ({100 * total_frames / old_total:.1f} %)")

    output.mkdir(parents=True)
    for name in ("videos", "images", "traces"):
        if (dataset / name).is_dir():
            print(f"[subset] linked {link_tree(dataset / name, output / name)} file(s) from {name}/")

    # --- data -------------------------------------------------------------------------
    for source in sorted((dataset / "data").glob("chunk-*/*.parquet")):
        relative = source.relative_to(dataset / "data")
        table = pq.read_table(source)
        episode_column = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        rows = np.flatnonzero(remap[episode_column] >= 0)
        columns = {name: table.column(name).take(pa.array(rows)) for name in table.column_names}
        columns["episode_index"] = pa.array(remap[episode_column[rows]])
        # `index` is the global row counter -- verified to be 0..N-1 on the source -- and every
        # episode's `frame_index` restarts at 0, so both are rebuilt/kept rather than shifted.
        columns["index"] = pa.array(np.arange(len(rows), dtype=np.int64))
        target = output / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), target)
        print(f"[subset] {relative}: {len(rows)} rows kept")

    # --- meta/episodes ----------------------------------------------------------------
    episodes_out = episodes.iloc[keep].copy().reset_index(drop=True)
    episodes_out["episode_index"] = np.arange(len(keep), dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(new_lengths)])
    episodes_out["dataset_from_index"] = cumulative[:-1].astype(np.int64)
    episodes_out["dataset_to_index"] = cumulative[1:].astype(np.int64)
    target = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(episodes_out, preserve_index=False), target)

    # --- meta, verbatim ---------------------------------------------------------------
    for name in ("tasks.parquet", "stats.json", "info.json"):
        source = dataset / "meta" / name
        if source.is_file():
            (output / "meta" / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output / "meta" / name)

    # `info.json` carries the totals, so leaving it copied verbatim would tell LeRobot the wrong
    # number of episodes and frames -- and the loader allocates and iterates from these.
    info_path = output / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        info["total_episodes"] = int(len(keep))
        info["total_frames"] = total_frames
        # The source holds one chunk and one file per stream; the subset inherits that layout.
        info_path.write_text(json.dumps(info, indent=4) + "\n")
        print(f"[subset] info.json: total_episodes={info['total_episodes']}"
              f" total_frames={info['total_frames']}")

    # --- verify -----------------------------------------------------------------------
    check = pq.read_table(output / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    problems = []
    if len(check) != len(keep):
        problems.append(f"episodes table has {len(check)} rows, expected {len(keep)}")
    if int(check["length"].sum()) != total_frames:
        problems.append(f"episodes table sums to {int(check['length'].sum())} frames, expected {total_frames}")
    if not (check["episode_index"].to_numpy() == np.arange(len(keep))).all():
        problems.append("episode_index is not densely renumbered from 0")
    if not (check["dataset_from_index"].to_numpy()[0] == 0):
        problems.append("first dataset_from_index is not 0")
    data_rows = sum(pq.read_metadata(p).num_rows for p in sorted((output / "data").glob("chunk-*/*.parquet")))
    if data_rows != total_frames:
        problems.append(f"data has {data_rows} rows, episodes table says {total_frames}")
    # The video columns must be untouched: they are absolute positions in the shared shards.
    for column in ("videos/observation.images.front/from_timestamp",
                   "videos/observation.images.front/to_timestamp"):
        if column in episodes.columns and not np.allclose(
            check[column].to_numpy(), episodes[column].to_numpy()[keep], rtol=0, atol=0
        ):
            problems.append(f"{column} was modified; subsets must not move video timestamps")

    if problems:
        for problem in problems:
            print(f"[subset] PROBLEM: {problem}")
        return 1
    print("[subset] verified: episodes, frame counts, dense renumbering and video timestamps all agree")
    print(f"[subset] wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
