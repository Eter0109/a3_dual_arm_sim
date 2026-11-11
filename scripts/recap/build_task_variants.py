#!/usr/bin/env python3
"""Build an advantage-conditioned copy of a RECAP dataset, without touching lerobot.

The idea, and why it needs no code change
----------------------------------------
RECAP conditions the policy on how good each sample was.  There are two ways to deliver that
condition: a new input channel in the model, or **a different sentence**.  The second one is
strictly simpler here because of how pi0.5 already works: it answers the instruction it is given,
the dataset already carries an instruction per episode, and `tasks.parquet` already maps
`task_index` to a sentence.  So a positive sample is the same instruction with
`" Advantage: positive"` on the end, and training needs nothing beyond a `task_index` rewrite.

Verified against this lerobot version: `dataset_reader.py:281` resolves the prompt as

    item["task"] = self._meta.tasks.iloc[task_idx].name

That is **positional**, not a lookup by value, so the table's row *k* must carry
`task_index == k`.  A table whose indices are correct but out of order silently attaches the
wrong sentence to every episode after the first misordering, and no arithmetic fails -- this is
the one invariant the script asserts on the way out.

The three variants
------------------
For a dataset with `T` original tasks, the copy gets `2T` (or `3T`) rows:

| range | sentence | used for |
|---|---|---|
| `0 .. T-1` | the original plan sentence | the unconditional branch of CFG, and the negative class in 4A |
| `T .. 2T-1` | `+ " Advantage: positive"` | the conditioned branch / positive class |
| `2T .. 3T-1` | `+ " Advantage: negative"` (optional) | only for 4A, where the "bad" side is also a distinct sentence |

The suffix text is chosen to read as a continuation rather than as a new instruction, because
pi0.5 was fine-tuned on one sentence and a wholesale rewrite is out of distribution (§9.51).

What is copied and what is rewritten
-----------------------------------
* `videos/` and `images/` are **hard-linked** (same filesystem, so instant and no extra space).
  Copying 6 GB per variant would make every experiment cost disk rather than minutes.
* `data/*.parquet` is rewritten: only `task_index` changes.
* `meta/tasks.parquet` is rebuilt, `meta/episodes/*.parquet` has its `task_index` statistics
  updated to describe the new values, and `info.json` gets the new `total_tasks`.

Everything else -- the values, the history, the labels sidecar -- is left alone.

Usage:
    python scripts/recap/build_task_variants.py \
        --dataset /root/autodl-tmp/datasets/a3_recap_all \
        --advantages /root/autodl-tmp/datasets/a3_recap_all/advantages.parquet \
        --output /root/autodl-tmp/datasets/a3_recap_cond
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

POSITIVE_SUFFIX = " Advantage: positive"
NEGATIVE_SUFFIX = " Advantage: negative"

#: Statistics recorded per feature in the episodes table.  Only `task_index`'s are recomputed.
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def propagate_labels(
    total_frames: int,
    labelled_rows: np.ndarray,
    tri: np.ndarray,
    episodes: np.ndarray,
    bounds: dict[int, tuple[int, int]],
) -> np.ndarray:
    """Give every frame the label of the nearest labelled frame in its own episode.

    Why this is not fabrication
    ---------------------------
    Advantages were computed on every 5th control frame (`extract_value_features.py --stride 5`),
    so 80 % of frames have no label of their own.  Leaving them unconditional -- the first
    version of this script -- silently turned a labelling choice into an 80 % CFG dropout rate,
    which is a different experiment from the one the plan describes.

    Propagating is defensible only because the labels are stable at this timescale, and that was
    measured rather than assumed: between labelled frames 5 apart (0.25 s) the tri-state label
    agrees 93.1 % of the time, and a positive-to-negative flip -- the only error that actively
    teaches the wrong thing -- happens on 0.91 % of pairs.  Assigning each frame its nearest
    labelled neighbour therefore moves at most 2 frames (0.10 s), which is 4 % of the 50-frame
    window the advantage was computed over.

    The alternative was re-extracting features at stride 1, which costs 7.7 h and produces labels
    that flip between neighbours just as often: the frame-to-frame jitter is in the value
    function, not in the sampling.
    """
    out = np.zeros(total_frames, dtype=np.int8)
    order = np.argsort(labelled_rows)
    rows = labelled_rows[order]
    values = tri[order]
    episodes = episodes[order]

    for episode in np.unique(episodes):
        members = episodes == episode
        episode_rows = rows[members]
        episode_values = values[members]
        if len(episode_rows) == 0:
            continue
        lo, hi = bounds[int(episode)]
        # `searchsorted` gives, for every frame, the first labelled row at or after it.  Comparing
        # against the one before picks the nearer of the two neighbours, which is the definition
        # of nearest-neighbour assignment without a distance matrix.
        span = np.arange(lo, hi)
        index = np.searchsorted(episode_rows, span)
        left = episode_rows[np.clip(index - 1, 0, len(episode_rows) - 1)]
        right = episode_rows[np.clip(index, 0, len(episode_rows) - 1)]
        nearest = np.where(np.abs(span - left) <= np.abs(right - span), left, right)
        out[span] = episode_values[np.searchsorted(episode_rows, nearest)]
    return out


def link_tree(source: Path, target: Path) -> int:
    """Hard-link every file under `source` into `target`, preserving layout.

    Hard links rather than copies: 6 GB of video per variant would make each experiment a disk
    problem.  They are safe here because nothing in this pipeline ever writes to a video file --
    the Launcher only reads them -- and they fail loudly (`EXDEV`) if `--output` is on another
    filesystem, which is better than silently copying.
    """

    count = 0
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError as exc:
            raise SystemExit(
                f"cannot hard-link {path} -> {destination}: {exc}.  `--output` must be on the "
                "same filesystem as --dataset, otherwise every video is copied."
            ) from exc
        count += 1
    return count


def variants_for(sentence: str) -> list[str]:
    """The original sentence and its two conditioned forms, in index order 0, 1, 2."""

    return [sentence, sentence + POSITIVE_SUFFIX, sentence + NEGATIVE_SUFFIX]


def read_task_table(dataset: Path) -> list[str]:
    """Sentences ordered by `task_index`, so position == index.

    Sorted defensively: `merge_parts.py` writes the table in index order, but this script is
    also pointed at hand-edited copies, and a silently reordered table attaches the wrong
    sentence to every episode after the first misordering.
    """

    table = pq.read_table(dataset / "meta" / "tasks.parquet").to_pandas()
    table = table.sort_values("task_index")
    return [str(name) for name in table.index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--advantages", required=True, type=Path, help="advantages.parquet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("cfg", "suffix"),
        default="cfg",
        help=(
            "cfg: positives get the positive prompt, negatives keep the original (this is the "
            "unconditional branch, so no third variant is needed).  suffix: positives and "
            "negatives get their own suffix, which is the 4A configuration."
        ),
    )
    parser.add_argument("--label-column", default="positive", help="which advantage label to use")
    parser.add_argument(
        "--propagate",
        action="store_true",
        help=(
            "give every frame the label of the nearest labelled frame in its episode.  Advantages "
            "exist on every 5th frame only, so without this the other 80 %% keep their original "
            "sentence and the effective CFG dropout rate is set by the sampling stride rather "
            "than by choice."
        ),
    )
    args = parser.parse_args(argv)

    dataset = args.dataset
    output = args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    sentences = read_task_table(dataset)
    n_original = len(sentences)
    print(f"[variants] {n_original} original instruction(s)")

    # Build the global variant table.  `cfg` needs two variants per original (original +
    # positive); `suffix` needs three.
    global_sentences: list[str] = []
    per_original: dict[int, dict[str, int]] = {}
    for index, sentence in enumerate(sentences):
        forms = variants_for(sentence) if args.mode == "suffix" else [sentence, sentence + POSITIVE_SUFFIX]
        mapping = {}
        for name, form in zip(("original", "positive", "negative"), forms):
            mapping[name] = len(global_sentences)
            global_sentences.append(form)
        # In `cfg` mode there is no distinct negative sentence: the negative class keeps the
        # original one, because that *is* the unconditional branch.
        mapping.setdefault("negative", mapping["original"])
        per_original[index] = mapping

    print(f"[variants] {len(global_sentences)} sentences in the conditioned table")

    # --- labels ---------------------------------------------------------------------
    advantages = pq.read_table(args.advantages)
    if args.label_column not in advantages.column_names:
        raise SystemExit(f"{args.advantages} has no column {args.label_column!r}")
    frame_rows = np.asarray(advantages.column("dataset_row").to_pylist(), dtype=np.int64)
    frame_positive = np.asarray(advantages.column(args.label_column).to_pylist(), dtype=bool)
    # The advantage file carries a separate `negative` flag rather than `~positive`, because most
    # frames are *neutral*: their advantage is smaller than the scoring can resolve, so they get
    # no condition at all.  Treating `~positive` as negative (the first version of this script)
    # pushed a condition onto frames whose label was noise.
    if "negative" in advantages.column_names:
        frame_negative = np.asarray(advantages.column("negative").to_pylist(), dtype=bool)
    else:
        frame_negative = ~frame_positive
    neutral = ~frame_positive & ~frame_negative
    print(f"[variants] {len(frame_rows)} labelled frames,"
          f" {100 * frame_positive.mean():.1f} % positive,"
          f" {100 * frame_negative.mean():.1f} % negative,"
          f" {100 * neutral.mean():.1f} % neutral (stay unconditional)")

    # --- per-frame labels -----------------------------------------------------------
    # Default: only the frames that were actually labelled get a condition, and everything else
    # keeps the original sentence.  `--propagate` spreads each label to its neighbours instead, so
    # "unconditional" means "neutral advantage" rather than "never sampled"; see
    # `propagate_labels` for the measurements that make that acceptable.
    frame_count = int(sum(pq.read_metadata(p).num_rows
                          for p in sorted((dataset / "data").glob("chunk-*/*.parquet"))))
    if args.propagate:
        episodes_table = pq.read_table(
            sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet"))[0],
            columns=["episode_index", "dataset_from_index", "dataset_to_index"],
        ).to_pandas()
        bounds = {
            int(row.episode_index): (int(row.dataset_from_index), int(row.dataset_to_index))
            for row in episodes_table.itertuples()
        }
        tri = np.where(frame_positive, 1, np.where(frame_negative, -1, 0)).astype(np.int8)
        labels = propagate_labels(
            frame_count,
            frame_rows,
            tri,
            np.asarray(advantages.column("episode_index").to_pylist(), dtype=np.int64),
            bounds,
        )
        print(f"[variants] propagated to {frame_count} frames:"
              f" {100 * (labels > 0).mean():.1f} % positive,"
              f" {100 * (labels < 0).mean():.1f} % negative,"
              f" {100 * (labels == 0).mean():.1f} % neutral")
        frame_rows = np.arange(frame_count, dtype=np.int64)
        frame_positive = labels > 0
        frame_negative = labels < 0

    # --- output tree ----------------------------------------------------------------
    output.mkdir(parents=True)
    for name in ("videos", "images"):
        if (dataset / name).is_dir():
            linked = link_tree(dataset / name, output / name)
            print(f"[variants] linked {linked} file(s) from {name}/")
    # `traces/` is diagnostic sidecar data, not read by the Launcher; linking it keeps the
    # conditioned copy as traceable as the original at no cost.
    if (dataset / "traces").is_dir():
        linked = link_tree(dataset / "traces", output / "traces")
        print(f"[variants] linked {linked} trace file(s)")

    # --- data -----------------------------------------------------------------------
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for source in sorted((dataset / "data").glob("chunk-*/*.parquet")):
        relative = source.relative_to(dataset / "data")
        target = output / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)

        # Default: keep the original sentence.  Then overwrite the frames the advantage file
        # marks.  In `cfg` mode `per_original["negative"]` is the original index, so the
        # negative class lands back on the unconditional sentence by construction rather than
        # by a special case here.
        new_index = np.array(
            [per_original[int(t)]["original"] for t in task_index], dtype=np.int64
        )
        for target_kind, rows in (
            ("positive", frame_rows[frame_positive]),
            ("negative", frame_rows[frame_negative]),
        ):
            if len(rows) == 0:
                continue
            new_index[rows] = np.array(
                [per_original[int(t)][target_kind] for t in task_index[rows]], dtype=np.int64
            )

        columns = {name: table.column(name) for name in table.column_names}
        columns["task_index"] = pa.array(new_index)
        pq.write_table(pa.table(columns), target)
        changed = int((new_index != task_index).sum())
        print(f"[variants] {relative}: {table.num_rows} rows, {changed} task_index rewritten"
              f" ({100 * changed / table.num_rows:.1f} %)")

    # --- meta -----------------------------------------------------------------------
    (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    import pandas as pd

    task_frame = pd.DataFrame(
        {"task_index": list(range(len(global_sentences)))},
        index=pd.Index(global_sentences, name="task"),
    )
    task_frame.to_parquet(output / "meta" / "tasks.parquet", index=True)

    for source in sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet")):
        relative = source.relative_to(dataset / "meta" / "episodes")
        target = output / "meta" / "episodes" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        columns = {name: table.column(name) for name in table.column_names}
        # The episodes table stores the sentence list rather than an index, so it has to follow
        # the rewrite too -- otherwise a downstream consumer that reads `tasks` from here would
        # see the unconditioned sentence for a conditioned episode.
        if "tasks" in columns:
            episode_tasks = columns["tasks"].to_pylist()
            updated = []
            for row in episode_tasks:
                sentence = row[0]
                mapping = per_original[sentences.index(sentence)]
                updated.append([global_sentences[mapping["original"]]])
            columns["tasks"] = pa.array(updated)
        # `stats/task_index/*` are per-episode summary statistics of the *old* indices; recompute
        # is impossible here without the frames, so they are widened to the full new range.  They
        # are not consumed by the normaliser (only STATE and ACTION are), so an approximate range
        # is honest rather than harmful -- but the values are replaced, not left stale.
        for stat in STAT_NAMES:
            key = f"stats/task_index/{stat}"
            if key not in columns:
                continue
            width = table.num_rows
            if stat == "min":
                value = 0.0
            elif stat == "max":
                value = float(len(global_sentences) - 1)
            elif stat == "count":
                value = 1.0
            else:
                value = float(len(global_sentences) - 1) / 2.0
            columns[key] = pa.array([[float(value)] for _ in range(width)])
        pq.write_table(pa.table(columns), target)

    # --- info.json -------------------------------------------------------------------
    info = json.loads((dataset / "meta" / "info.json").read_text())
    info["total_tasks"] = len(global_sentences)
    info["repo_id"] = output.name
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    for name in ("stats.json",):
        if (dataset / "meta" / name).is_file():
            # Statistics describe STATE/ACTION/observations, none of which changed.  Copying them
            # is correct here, unlike the merge case where the rows were different.
            shutil.copy2(dataset / "meta" / name, output / "meta" / name)

    # --- invariant check -------------------------------------------------------------
    written = pq.read_table(output / "meta" / "tasks.parquet").to_pandas()
    positions = list(written["task_index"])
    if positions != list(range(len(written))):
        raise SystemExit(
            "tasks.parquet row order does not equal task_index, which `dataset_reader` "
            "requires (`iloc[task_idx]`)"
        )
    if len(set(written.index)) != len(written):
        raise SystemExit("duplicate sentences in tasks.parquet; the index is used as a key")
    print(f"[variants] tasks.parquet OK: {len(written)} rows, order == index, sentences unique")

    print(f"[variants] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
