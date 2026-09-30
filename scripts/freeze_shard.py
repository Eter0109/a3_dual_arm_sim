"""Freeze a still-running collection into a self-consistent, mergeable dataset.

A LeRobot collection cannot be merged while it runs, for a reason that has nothing to
do with the data being wrong:

* **The parquet has no footer until it is closed.**  ``ParquetWriter`` appends row
  groups and writes the schema and row-group offsets on ``close()``, which LeRobot
  calls when a file reaches ``data_files_size_in_mb`` or when the process exits.  A
  file without a footer is unreadable, and its absence is how "still being written" is
  told apart from "corrupt".
* **The episode table is flushed every ten episodes.**  ``metadata_buffer_size``
  defaults to 10, so ``meta/episodes`` trails ``info.json``, which is updated per
  episode.  A reader that trusts the first and indexes the second runs off the end.

``merge_cookie_shards.py`` checks that count and frame total agree exactly -- correct
for a finished run, impossible for a running one.  So this writes a *frozen view*
instead: a directory built mostly from hard links that contains only what is complete
and describes it consistently.  The source collection keeps running, untouched.

Hard links rather than copies, because sealed parquet files are never appended to
again (the writer moves to a new file) and video files are replaced by the ffmpeg
concatenation rather than modified in place.  So a 50 GB collection costs a few
megabytes of directory entries.

What is deliberately left alone: **the video files keep any trailing frames** from
episodes that were dropped.  Trimming them would mean re-encoding, and it is not
needed -- ``aggregate_datasets`` offsets each source by the duration of the video file
it read, so a longer tail simply shifts the next shard further along a timeline nothing
references.  The kept episodes' own timestamps still point at their own frames, because
LeRobot v3 writes them on a global clock.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def is_sealed(path: Path) -> bool:
    """Whether the writer has closed this parquet file.

    A parquet file begins and ends with ``PAR1``; the trailing copy is written by
    ``close()``.  Checking it separates the normal mid-write state from corruption,
    which a bare try/except would conflate.
    """

    if path.stat().st_size < 8:
        return False
    with open(path, "rb") as handle:
        handle.seek(-4, os.SEEK_END)
        return handle.read(4) == b"PAR1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default=None, help="Defaults to the shard name")
    args = parser.parse_args()

    source, destination = args.shard.resolve(), args.output.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise SystemExit(f"{source} does not look like a LeRobot dataset")
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {destination}")

    info = json.loads((source / "meta" / "info.json").read_text())
    sealed = [
        path for path in sorted((source / "data").glob("chunk-*/*.parquet")) if is_sealed(path)
    ]
    if not sealed:
        raise SystemExit(f"{source.name}: no sealed parquet yet, nothing to freeze")

    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in sealed)

    episode_files = sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise SystemExit(f"{source.name}: no episode metadata")
    episodes = pa.concat_tables([pq.read_table(path) for path in episode_files]).to_pylist()
    episodes.sort(key=lambda row: row["episode_index"])

    # An episode is usable only if the data files already contain all of its rows.  The
    # episode table can describe an episode whose frames are still in the writer's
    # buffer, so this is a test on the data, not on the metadata.
    keep = [row for row in episodes if int(row["dataset_to_index"]) <= rows]
    if not keep:
        raise SystemExit(f"{source.name}: no complete episode yet ({rows} rows sealed)")
    total_frames = sum(int(row["length"]) for row in keep)
    dropped = len(episodes) - len(keep)

    print(f"{source.name}:")
    print(f"  sealed data    : {len(sealed)} files, {rows} rows")
    print(f"  episodes       : {len(keep)} complete"
          + (f", {dropped} not yet ({rows - total_frames} rows beyond them)" if dropped else ""))

    # --- data: one file, exactly the kept rows ------------------------------------
    # Trimming rather than linking the whole file, so the parquet's row count equals the
    # frames the episode table refers to.  Leaving the tail in would make `len(dataset)`
    # disagree with `total_frames`, and a sampler indexing by the latter would be fine
    # while one indexing by the former would not -- the kind of ambiguity worth
    # removing rather than relying on.
    columns = pq.ParquetFile(sealed[0]).schema_arrow.names
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in sealed])
    table = table.slice(0, total_frames)
    data_dir = destination / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_dir / "file-000.parquet", compression="snappy")

    # Every episode now lives in that one file.
    for row in keep:
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        # Row positions are already contiguous from zero, because episodes are written
        # in order and `keep` is a prefix of them.
    if dropped:
        kept_indices = [int(row["episode_index"]) for row in keep]
        if kept_indices != list(range(len(keep))):
            raise SystemExit(
                f"{source.name}: kept episodes are not a contiguous run from 0 "
                f"({kept_indices[:5]}...); the frozen view would misnumber them"
            )

    # --- video: linked whole, with their original chunk/file numbering ------------
    for camera in sorted((source / "videos").iterdir()):
        for path in sorted(camera.glob("chunk-*/*.mp4")):
            target = destination / "videos" / camera.name / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)

    # --- metadata -----------------------------------------------------------------
    (destination / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(keep),
        destination / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    for name in ("stats.json", "tasks.parquet"):
        source_path = source / "meta" / name
        if source_path.is_file():
            # Copied rather than linked: `stats.json` is rewritten by whichever tool
            # touches the frozen view next, and a hard link would carry that back into
            # the running collection's directory.
            (destination / "meta" / name).write_bytes(source_path.read_bytes())

    info_out = dict(info)
    info_out["total_episodes"] = len(keep)
    info_out["total_frames"] = total_frames
    # A single-file dataset, so the templates name constants.
    info_out["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    if "splits" in info_out:
        info_out["splits"] = {"train": f"0:{len(keep)}"}
    (destination / "meta" / "info.json").write_text(json.dumps(info_out, indent=4) + "\n")

    # The merge reads this for the `repo_id` it passes to `aggregate_datasets`, and
    # rejects shards that share one.  The source run has no summary yet, so it is
    # synthesized here with a name unique to the frozen view.
    summary = {
        "schema_version": 2,
        "repo_id": args.repo_id or f"{source.name}-frozen",
        "frozen_from": str(source),
        "requested_episodes": len(keep),
        "accepted_episodes": len(keep),
        "attempts": len(keep) + dropped,
        "task": info.get("robot_type", "A3_dual_arm"),
        "success_contract": "frozen_mid_run",
        "prompt": "",
        "position_noise_m": 0.0,
        "yaw_noise_rad": 0.0,
        "note": (
            "Frozen view of a running collection: only sealed data, only complete "
            "episodes, info.json rewritten to match. The source run is unaffected."
        ),
    }
    (destination / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "a3_episode_metadata.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_index": index,
                    "frames": int(row["length"]),
                    "success": True,
                    "frozen": True,
                },
                ensure_ascii=False,
            )
            for index, row in enumerate(keep)
        )
        + "\n"
    )

    print(f"  frozen         : {len(keep)} episodes, {total_frames} frames -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
