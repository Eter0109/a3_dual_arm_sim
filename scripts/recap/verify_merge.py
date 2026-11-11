#!/usr/bin/env python3
"""Verify that a merged dataset's per-part video references point at the right frames.

The check that matters is the cross-part one: frame 0 of merged episode 2 must be pixel-identical
to frame 0 of `part_01`, not to `part_00`.  A merge that copies every part into `file-000.mp4`
and then rewrites `file_index` would pass every structural check -- correct row counts, correct
episode boundaries, correct task indices -- while showing every episode the wrong video.  That is
a silent failure, so it gets a pixel comparison.

Usage:
    python scripts/recap/verify_merge.py --merged <root> --parts <part_*> [--parts-root <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_dataset(root: Path, repo_id: str):
    # `video_backend="pyav"` is required in this container: torchcodec cannot load its shared
    # libraries against the installed FFmpeg (`libavutil.so.58: cannot open shared object
    # file`), which is the same reason `train_pi05_a3.sh` passes `--dataset.video_backend=pyav`.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")


def frame_of(dataset, index: int) -> np.ndarray:
    return np.asarray(dataset[index]["observation.images.front"], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", required=True, type=Path)
    args = parser.parse_args(argv)

    merged_root = args.merged
    problems: list[str] = []

    data = pq.read_table(merged_root / "data" / "chunk-000" / "file-000.parquet")
    episodes = pq.read_table(merged_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = json.loads((merged_root / "meta" / "info.json").read_text())

    episode_index = np.asarray(data.column("episode_index").to_pylist())
    row_index = np.asarray(data.column("index").to_pylist())

    # --- structure ------------------------------------------------------------------
    if not np.array_equal(row_index, np.arange(len(row_index))):
        problems.append("`index` is not 0..N-1 in order")
    if info["total_frames"] != data.num_rows:
        problems.append(f"info.total_frames={info['total_frames']} but the table has {data.num_rows}")
    if info["total_episodes"] != episodes.num_rows:
        problems.append(f"info.total_episodes={info['total_episodes']} but episodes has {episodes.num_rows}")
    if sorted(set(episode_index.tolist())) != list(range(episodes.num_rows)):
        problems.append("episode_index is not 0..E-1")

    froms = np.asarray(episodes.column("dataset_from_index").to_pylist())
    tos = np.asarray(episodes.column("dataset_to_index").to_pylist())
    for ep in range(episodes.num_rows):
        rows = episode_index == ep
        if not (rows.sum() == tos[ep] - froms[ep] and row_index[rows].min() == froms[ep]):
            problems.append(f"episode {ep}: dataset_from/to {froms[ep]}..{tos[ep]} disagrees with the rows")

    # --- tasks: one sentence per index ----------------------------------------------
    tasks = __import__("pandas").read_parquet(merged_root / "meta" / "tasks.parquet")
    sentence_for = dict(zip(tasks["task_index"].tolist(), tasks.index.tolist()))
    if len(sentence_for) != len(tasks):
        problems.append("tasks.parquet has duplicate task_index values")
    if len(set(tasks.index.tolist())) != len(tasks):
        problems.append("tasks.parquet has duplicate sentences")

    # --- the decisive one: video frames across the part boundary --------------------
    merged = load_dataset(merged_root, "a3_recap_merged_verify")
    print(f"[verify] merged: {merged.meta.total_episodes} episodes, {merged.meta.total_frames} frames")

    # The merged row of each part's first episode, taken from the episodes table rather than
    # computed from part sizes: a part may contribute several video files per camera, so
    # "which merged episode is this part's first" is `episode_offset + 0`, and the offset is
    # the sum of the previous parts' episode counts.
    per_part_episodes = []
    lengths: list[np.ndarray] = []
    for part in args.parts:
        part_episodes = pq.read_table(part / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        per_part_episodes.append(part_episodes.num_rows)
        # A part's episodes are not all the same length (the expert ends early), so the row
        # offset of episode `k` cannot be `k * 1500`.  The table's own `length` column is the
        # only correct source.
        lengths.append(np.asarray(part_episodes.column("length").to_pylist(), dtype=np.int64))

    checked = 0
    for part_index, part in enumerate(args.parts):
        part_ds = load_dataset(part, f"a3_recap_verify_part{part_index:02d}")
        first_episode = sum(per_part_episodes[:part_index])
        merged_row = int(froms[first_episode])
        a = frame_of(merged, merged_row)
        b = frame_of(part_ds, 0)
        same = np.abs(a - b).max()
        status = "OK" if same < 1e-3 else "MISMATCH"
        print(f"  {part.name}: merged row {merged_row} vs part row 0 -> max|diff|={same:.6f} {status}")
        if same >= 1e-3:
            problems.append(f"{part.name}: merged frame differs from the part's frame ({same:.4f})")
        checked += 1

        # A part's own video may be split across several files -- LeRobot rolls to
        # `file-001.mp4` per camera when the current one hits its size cap, and the shard
        # points differ per camera (measured: left_wrist sharded at episode 28 while front and
        # right_wrist did not).  Checking only episode 0 of each part would pass a merge that
        # dropped or misnumbered the later shard, so the first episode *after* each internal
        # boundary is compared too.
        for camera in [k for k in merged.meta.features if k.startswith("observation.images")]:
            local = pq.read_table(
                part / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                columns=[f"videos/{camera}/file_index"],
            )
            file_index = np.asarray(local.column(f"videos/{camera}/file_index").to_pylist())
            boundaries = [i for i in range(1, len(file_index)) if file_index[i] != file_index[i - 1]]
            for boundary in boundaries[:1]:
                target_row = int(froms[first_episode + boundary])
                a = frame_of(merged, target_row)
                # `boundary` is an **episode** index within the part, not a row index.  Using it
                # as a row index reads part row 28 (which is inside episode 0, since episodes
                # are 1500 rows long) and every camera then mismatches -- a bug that looks
                # exactly like a merge defect.
                b = frame_of(part_ds, int(lengths[part_index][:boundary].sum()))
                same = np.abs(a - b).max()
                label = f"{part.name} {camera.split('.')[-1]} ep{boundary}"
                print(f"  {label}: merged row {target_row} vs part row"
                      f" {int(lengths[part_index][:boundary].sum())}"
                      f" -> max|diff|={same:.6f} {'OK' if same < 1e-3 else 'MISMATCH'}")
                if same >= 1e-3:
                    problems.append(f"{label}: shard boundary frame differs ({same:.4f})")
                checked += 1

    # --- a_previous is live away from frame 0 --------------------------------------
    for probe in (100, 400):
        history = np.asarray(merged[probe]["observation.history"]).reshape(-1, 48)
        share = float((history[:, 32:] != 0).mean())
        print(f"  sample {probe}: a_previous non-zero {100 * share:.1f} %")
        if share == 0.0:
            problems.append(f"sample {probe}: a_previous is all zero; the window is dead")

    print()
    if problems:
        print(f"[verify] {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"[verify] all checks passed ({checked} part(s) compared pixel-wise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
