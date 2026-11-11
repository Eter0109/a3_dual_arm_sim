#!/usr/bin/env python3
"""Merge the collector's `part_*` datasets into one root a trainer can open.

Why this exists
---------------
`LeRobotDataset.create` takes an exclusive lock on its root, so the collector runs N
processes with N independent roots (`part_00`, `part_01`, ...).  `lerobot-train` accepts a
single `--dataset.root`, so the parts have to be joined before training.

What makes it more than a `cat`
-------------------------------
Four things are per-part and silently wrong if copied:

1. **`task_index` is a per-part index into a per-part string table.**  Measured on the first
   merge: `part_00`'s index 0 is "Grasp 4 cookies first" and `part_01`'s index 0 is "Grasp 7
   cookies first".  Concatenating the data parquets unchanged would attach the wrong
   instruction to every part after the first, and nothing errors -- the training loss would
   just be fitting text that does not match the scene.  Tasks are therefore remapped through
   a global string table, so equal sentences share an index and unequal ones never do.

2. **`episode_index` and `index` are global.**  `index` is the row number across the whole
   dataset and `dataset_from_index`/`dataset_to_index` are its episode boundaries, so both
   are recomputed as running offsets rather than carried over.

3. **Video file references.**  Each part has one mp4 per camera.  They are copied as separate
   `file-{k}.mp4` entries, each part getting its own `file_index`, so `from_timestamp` /
   `to_timestamp` (which are relative to their own file) stay valid without re-encoding.
   The template is `videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4`.

4. **Statistics.**  Every non-video feature is recomputed from the merged rows.  Image
   statistics are merged by frame-weighted moments instead of being decoded, and that is an
   approximation -- see `_merge_image_stats` for why it is acceptable here.

Usage
-----
    python scripts/recap/merge_parts.py \
        --parts /root/autodl-tmp/datasets/a3_recap/part_* \
        --output /root/autodl-tmp/datasets/a3_recap_merged
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

#: Statistics LeRobot records for every feature.  Recomputed, not averaged: averaging
#: quantiles across parts gives a number that is not a quantile of the union.
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILE_LEVELS = (1, 10, 50, 90, 99)

#: Keys whose statistics describe pixels.  Recomputed exactly would mean decoding every
#: video; merged approximately instead, which is safe because `normalization_mapping` maps
#: VISUAL to IDENTITY, so these numbers are never used to scale anything.
IMAGE_PREFIX = "observation.images"


def load_tasks(part: Path) -> dict[int, str]:
    """`task_index -> sentence` for one part, read from its own table.

    The table is stored with the sentence as the *index* and a single `task_index` column,
    which is why this does not read a column called "task".
    """

    table = pq.read_table(part / "meta" / "tasks.parquet").to_pandas()
    tasks: dict[int, str] = {}
    for sentence, row in table.iterrows():
        tasks[int(row["task_index"])] = str(sentence)
    return tasks


def build_global_tasks(parts: list[Path]) -> tuple[dict[str, int], dict[Path, dict[int, int]]]:
    """One sentence table for the union, and a per-part local->global index map.

    Ordering is by first appearance across the sorted parts, so a merge is reproducible:
    the same inputs always produce the same global indices.
    """

    global_index: dict[str, int] = {}
    remaps: dict[Path, dict[int, int]] = {}
    for part in parts:
        local = load_tasks(part)
        remap: dict[int, int] = {}
        for local_index in sorted(local):
            sentence = local[local_index]
            if sentence not in global_index:
                global_index[sentence] = len(global_index)
            remap[local_index] = global_index[sentence]
        remaps[part] = remap
    return global_index, remaps


def part_data_files(part: Path) -> list[Path]:
    return sorted((part / "data").glob("chunk-*/*.parquet"))


def part_episodes_file(part: Path) -> Path:
    files = sorted((part / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if len(files) != 1:
        raise SystemExit(f"{part} has {len(files)} episodes parquet files; expected exactly one")
    return files[0]


def part_video_files(part: Path, key: str) -> list[Path]:
    return sorted((part / "videos" / key).glob("chunk-*/*.mp4"))


def _merge_image_stats(stats: list[dict[str, Any]], counts: list[int]) -> dict[str, Any]:
    """Frame-weighted merge of per-part image statistics.

    `mean`/`std` combine exactly under the pooled-variance identity, and `min`/`max` are the
    extremes, so those four are right.  The quantiles are not: a pooled quantile is not a
    weighted quantile of part quantiles unless the parts are identically distributed.

    That inexactness is deliberate.  `normalization_mapping` maps VISUAL to IDENTITY for this
    policy, so image statistics are written but never used to scale a model input; decoding
    every frame to get them exactly would cost more than the whole collection run.  If a
    future policy normalises images, this function has to be replaced with a real pass.
    """

    total = float(sum(counts))
    weights = [c / total for c in counts]
    mean = np.sum([w * np.asarray(s["mean"], dtype=np.float64) for w, s in zip(weights, stats)], axis=0)
    # Pooled variance: E[x^2] - E[x]^2 with the per-part second moments rebuilt from mean/std.
    second = np.sum(
        [
            w * (np.asarray(s["std"], dtype=np.float64) ** 2 + np.asarray(s["mean"], dtype=np.float64) ** 2)
            for w, s in zip(weights, stats)
        ],
        axis=0,
    )
    var = np.clip(second - mean**2, 0.0, None)
    merged: dict[str, Any] = {
        "min": np.minimum.reduce([np.asarray(s["min"], dtype=np.float64) for s in stats]).tolist(),
        "max": np.maximum.reduce([np.asarray(s["max"], dtype=np.float64) for s in stats]).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "count": [int(total)],
    }
    for level, name in zip(QUANTILE_LEVELS, ("q01", "q10", "q50", "q90", "q99")):
        merged[name] = np.sum(
            [w * np.asarray(s[name], dtype=np.float64) for w, s in zip(weights, stats)], axis=0
        ).tolist()
    return merged


def compute_stats(rows: dict[str, np.ndarray], image_stats: dict[str, dict[str, Any]],
                  image_counts: dict[str, list[int]]) -> dict[str, Any]:
    """Statistics for the merged dataset, per feature."""

    stats: dict[str, Any] = {}
    for key, values in rows.items():
        if key.startswith(IMAGE_PREFIX):
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        entry: dict[str, Any] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "count": [int(values.shape[0])],
        }
        # `axis=0` is not optional: `np.percentile` flattens by default, which would turn a
        # per-dimension quantile into one number pooled across every dimension.
        for level, name in zip(QUANTILE_LEVELS, ("q01", "q10", "q50", "q90", "q99")):
            entry[name] = np.percentile(values, level, axis=0).tolist()
        stats[key] = entry
    for key, per_part in image_stats.items():
        stats[key] = _merge_image_stats(per_part, image_counts[key])
    return stats


def _shift_stat_column(values: list, delta: float) -> list:
    """Add `delta` to a per-episode statistics column that may be scalar or list-valued.

    The `stats/{feature}/{stat}` columns in the episodes table hold one array per episode
    (length 1 for a scalar feature, N for a vector one), so `to_pylist()` gives a list of
    lists for most of them and a flat list for a few.  Both shapes appear in the same table.
    """

    if values and isinstance(values[0], (list, tuple)):
        return [[float(v) + delta for v in row] for row in values]
    return [float(v) + delta for v in values]


def merge(parts: list[Path], output: Path, *, repo_id: str, keep_traces: bool = True) -> dict[str, Any]:
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    global_tasks, remaps = build_global_tasks(parts)
    print(f"[merge] {len(parts)} parts, {len(global_tasks)} distinct instruction(s)")

    output.mkdir(parents=True)
    (output / "data" / "chunk-000").mkdir(parents=True)
    (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    info = json.loads((parts[0] / "meta" / "info.json").read_text())
    feature_keys = list(info["features"].keys())
    video_keys = [k for k in feature_keys if k.startswith(IMAGE_PREFIX)]

    data_tables: list[pa.Table] = []
    episode_tables: list[pa.Table] = []
    image_stats: dict[str, dict[str, Any]] = {k: [] for k in video_keys}
    image_counts: dict[str, list[int]] = {k: [] for k in video_keys}
    # Running `file_index` per camera across parts.  Each part contributes as many files as
    # its own sharding produced, which differs by camera (see the video block below).
    video_offset: dict[str, int] = {k: 0 for k in video_keys}

    row_offset = 0
    episode_offset = 0
    for part_index, part in enumerate(parts):
        remap = remaps[part]
        print(f"  {part.name}: remapping {len(remap)} task(s)", end="")

        part_row_start = row_offset

        # --- data rows ---------------------------------------------------------------
        for path in part_data_files(part):
            table = pq.read_table(path)
            task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
            remapped = np.array([remap[int(t)] for t in task_index], dtype=np.int64)
            episode = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
            names = table.column_names
            table = table.set_column(names.index("task_index"), "task_index", pa.array(remapped))
            table = table.set_column(
                names.index("episode_index"), "episode_index", pa.array(episode + episode_offset)
            )
            table = table.set_column(
                names.index("index"),
                "index",
                pa.array(np.arange(row_offset, row_offset + table.num_rows, dtype=np.int64)),
            )
            row_offset += table.num_rows
            data_tables.append(table)

        # --- episodes ----------------------------------------------------------------
        episodes = pq.read_table(part_episodes_file(part))
        episode_rows = episodes.num_rows
        columns = {name: episodes.column(name) for name in episodes.column_names}

        def replace(name: str, values) -> None:
            columns[name] = pa.array(values)

        indexes = np.asarray(columns["episode_index"].to_pylist(), dtype=np.int64)
        replace("episode_index", indexes + episode_offset)
        # The per-episode statistics for `episode_index` describe that same global number.
        for stat in ("min", "max", "mean", "median", "q01", "q10", "q50", "q90", "q99"):
            key = f"stats/episode_index/{stat}"
            if key in columns:
                replace(key, _shift_stat_column(columns[key].to_pylist(), episode_offset))
        for stat in ("min", "max", "mean", "median", "q01", "q10", "q50", "q90", "q99"):
            key = f"stats/index/{stat}"
            if key in columns:
                replace(key, _shift_stat_column(columns[key].to_pylist(), part_row_start))

        lengths = np.asarray(columns["length"].to_pylist(), dtype=np.int64)
        # Boundaries in the merged table: this part starts at `part_row_start` and its
        # episodes are laid end to end from there.
        cumulative = np.cumsum(lengths)
        replace("dataset_from_index", part_row_start + cumulative - lengths)
        replace("dataset_to_index", part_row_start + cumulative)
        # `task_index` statistics are over the remapped values.  The episodes table has no
        # `task_index` column -- only `tasks`, a one-element list of sentences -- so the
        # global index is looked up through the task table rather than remapped.
        task_stats_present = [s for s in STAT_NAMES if f"stats/task_index/{s}" in columns]
        if task_stats_present:
            episode_tasks = np.array(
                [global_tasks[row[0]] for row in columns["tasks"].to_pylist()], dtype=np.float64
            )
            for stat in task_stats_present:
                if stat == "min":
                    value = episode_tasks.min()
                elif stat == "max":
                    value = episode_tasks.max()
                elif stat == "mean":
                    value = episode_tasks.mean()
                elif stat == "std":
                    value = episode_tasks.std()
                elif stat == "count":
                    value = float(episode_tasks.size)
                else:
                    level = int(stat[1:])
                    value = float(np.percentile(episode_tasks, level))
                replace(f"stats/task_index/{stat}", [[float(value)] for _ in range(episode_rows)])

        # --- videos: flatten each part's shards into a global file sequence --------
        #
        # A part does **not** necessarily have one video per camera.  LeRobot encodes each
        # camera independently and rolls to `file-001.mp4` when the current one reaches its
        # size cap, so the shard points differ per camera: measured on a 38-episode part,
        # `left_wrist` had 2 files (28 + 10 episodes) while `front` and `right_wrist` had 1.
        # A merge that assumed "one file per part per camera" therefore fails on real data,
        # and one that copies only `file-000` would silently truncate the last 10 episodes'
        # video to the first file's frames.
        #
        # So: keep every shard, give the merged dataset one contiguous `file_index` space per
        # camera, and rewrite each episode's reference through the part-local index it
        # already had.  `from/to_timestamp` are relative to their own file and need no change.
        for key in video_keys:
            files = part_video_files(part, key)
            if not files:
                raise SystemExit(f"{part} has no {key} video; cannot keep timestamps valid")
            target_dir = output / "videos" / key / "chunk-000"
            target_dir.mkdir(parents=True, exist_ok=True)
            # `file-000.mp4, file-001.mp4, ...` sort correctly by name, which is why the
            # local index in the table can be used as a position into this list.
            local_to_global = {}
            for local_index, source in enumerate(files):
                global_index = video_offset[key] + local_index
                shutil.copy2(source, target_dir / f"file-{global_index:03d}.mp4")
                local_to_global[local_index] = global_index
            video_offset[key] += len(files)

            local_file_index = np.asarray(
                columns[f"videos/{key}/file_index"].to_pylist(), dtype=np.int64
            )
            missing = sorted({int(v) for v in local_file_index} - set(local_to_global))
            if missing:
                raise SystemExit(
                    f"{part} {key}: episodes reference file_index {missing} but only "
                    f"{len(files)} file(s) exist"
                )
            replace(f"videos/{key}/chunk_index", np.zeros(episode_rows, dtype=np.int64))
            replace(
                f"videos/{key}/file_index",
                np.array([local_to_global[int(v)] for v in local_file_index], dtype=np.int64),
            )
            # from/to_timestamp are already relative to this file's start, so they carry over.
            part_stats = json.loads((part / "meta" / "stats.json").read_text())
            image_stats[key].append(part_stats[key])
            image_counts[key].append(int(part_stats[key]["count"][0]))

        episode_tables.append(pa.table(columns))
        episode_offset += episode_rows
        print(f", {episode_rows} episode(s)")

    # --- write ------------------------------------------------------------------
    merged_data = pa.concat_tables(data_tables)
    pq.write_table(merged_data, output / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.concat_tables(episode_tables), output / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    print(f"[merge] {merged_data.num_rows} frames, {episode_offset} episodes")

    # --- statistics -------------------------------------------------------------
    rows: dict[str, np.ndarray] = {}
    for key in feature_keys:
        if key.startswith(IMAGE_PREFIX):
            continue
        column = merged_data.column(key)
        values = np.stack(column.to_pylist()).astype(np.float64)
        rows[key] = values
    # These are per-row scalars in the parquet, not part of `features`, but LeRobot records
    # statistics for them and the trainer's normalizer reads the ones it knows.
    for key in ("index", "episode_index", "frame_index", "timestamp"):
        if key in merged_data.column_names and key not in rows:
            rows[key] = np.asarray(merged_data.column(key).to_pylist(), dtype=np.float64)
    stats = compute_stats(rows, image_stats, image_counts)
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # --- tasks ------------------------------------------------------------------
    import pandas as pd

    task_frame = pd.DataFrame(
        {"task_index": list(global_tasks.values())}, index=pd.Index(list(global_tasks.keys()), name="task")
    )
    task_frame.to_parquet(output / "meta" / "tasks.parquet")

    # --- info.json --------------------------------------------------------------
    info["total_episodes"] = episode_offset
    info["total_frames"] = merged_data.num_rows
    info["total_tasks"] = len(global_tasks)
    info["splits"] = {"train": f"0:{episode_offset}"}
    info["repo_id"] = repo_id
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    # --- sidecars ---------------------------------------------------------------
    if keep_traces:
        trace_root = output / "traces"
        label_lines: list[str] = []
        # Trace directories are named `seed_N`, and two runs can legitimately use the same
        # seed range -- the rollout track and the expert track were both started at 1000,
        # because a seed selects the *scene* and what differs between them is the policy.
        # Copying verbatim therefore collides on every seed.  Each source part gets its own
        # parent directory, so `part0_rollout/seed_1000` and `part1_expert/seed_1000`
        # coexist; the merged table's `episode_index` is what disambiguates them.
        for index, part in enumerate(parts):
            source = part / "traces"
            if source.is_dir():
                bucket = trace_root / f"part{index:02d}_{part.name}"
                for seed_dir in sorted(source.iterdir()):
                    if seed_dir.is_dir():
                        target = bucket / seed_dir.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            raise SystemExit(f"duplicate trace directory {target}")
                        shutil.copytree(seed_dir, target)
            labels = part / "episode_labels.jsonl"
            if labels.is_file():
                label_lines.extend(l for l in labels.read_text().splitlines() if l.strip())
        if label_lines:
            (output / "episode_labels.jsonl").write_text("\n".join(label_lines) + "\n")

    return {
        "parts": len(parts),
        "episodes": episode_offset,
        "frames": merged_data.num_rows,
        "tasks": len(global_tasks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", required=True, help="part_* directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="a3_recap_merged")
    parser.add_argument("--no-traces", action="store_true", help="skip copying traces/ and labels")
    args = parser.parse_args(argv)

    parts = [Path(p) for p in args.parts]
    missing = [p for p in parts if not (p / "meta" / "info.json").is_file()]
    if missing:
        raise SystemExit("not a LeRobot dataset root: " + ", ".join(str(p) for p in missing))

    summary = merge(parts, Path(args.output), repo_id=args.repo_id, keep_traces=not args.no_traces)
    print(f"[merge] wrote {args.output}")
    print(f"[merge] {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
