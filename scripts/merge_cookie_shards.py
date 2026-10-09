"""Merge per-shard LeRobot datasets into one training dataset.

LeRobot's writer needs a directory per process, so parallel collection produces
one dataset per shard. `aggregate_datasets` concatenates them in the order given,
which this script relies on to renumber the per-shard `a3_episode_metadata.jsonl`
records and re-emit a single `collection_summary.json` describing the whole run.

Assumes every shard shares the same repo_id prefix and task, which the collection
scripts guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

SUMMARY_NAME = "collection_summary.json"
METADATA_NAME = "a3_episode_metadata.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="Shard dataset roots, in the order they should be concatenated",
    )
    parser.add_argument("--repo-id", required=True, help="repo_id for the merged dataset")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset root")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


#: The fields a `collection_summary.json` carries **once per shard** rather than once per
#: collection.  Everything else in a shard's summary is a statement about the collection -- what
#: scene it was, which cameras, and above all how much the scene was allowed to vary -- and is
#: carried through unchanged by `combine_summaries`.
_PER_SHARD_FIELDS = frozenset({"repo_id", "shard_index", "shard_count", "results"})


def combine_summaries(
    summaries: list[dict], *, repo_id: str, shard_names: list[str], accepted_episodes: int
) -> dict:
    """One collection's summary from its shards' summaries.

    Written as "take everything, then fix up what is per shard" rather than "take these fields",
    because the enumerated version had to be edited every time the summary schema grew and nobody
    edited it: the merged summary of a fully randomized collection listed neither
    `scene_randomization_m` nor `placements_randomized` nor `cookies_jittered`, so the one field
    that says the boxes were drawn from a range -- the thing that makes this dataset different
    from a fixed-layout one -- was missing from the artifact people train on, while every shard's
    own summary had it.

    The caller supplies ``shard_names`` rather than reading them off the summaries because the
    merge writes them in the order it concatenated, which is the order the episode indices follow.
    """

    if not summaries:
        raise ValueError("no shard summaries to combine")
    if len(shard_names) != len(summaries):
        raise ValueError(
            f"{len(shard_names)} shard names for {len(summaries)} shard summaries"
        )

    combined = {key: value for key, value in summaries[0].items() if key not in _PER_SHARD_FIELDS}
    attempts = sum(int(summary.get("attempts", 0)) for summary in summaries)
    combined.update(
        repo_id=repo_id,
        shards=list(shard_names),
        shard_count=len(summaries),
        requested_episodes=sum(int(summary.get("requested_episodes", 0)) for summary in summaries),
        # The merged dataset's own episode count, not the shards' claims about themselves: this
        # is the number of records the merge actually produced, and a summary that disagreed with
        # its own dataset is the failure this field exists to make visible.
        accepted_episodes=accepted_episodes,
        attempts=attempts,
        success_rate=accepted_episodes / attempts if attempts else 0.0,
    )
    return combined


def main() -> int:
    args = parse_args()
    shards: list[Path] = [path.expanduser().resolve() for path in args.shards]
    output: Path = args.output.expanduser().resolve()

    for shard in shards:
        if not (shard / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Shard has no LeRobot metadata: {shard}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    source_repo_ids = []
    for shard in shards:
        summary = json.loads((shard / SUMMARY_NAME).read_text(encoding="utf-8"))
        source_repo_ids.append(summary["repo_id"])
    if len(set(source_repo_ids)) != len(source_repo_ids):
        raise ValueError("Shards must have distinct repo_ids; the collection scripts ensure this")

    print(f"merging {len(shards)} shards -> {output}")
    aggregate_datasets(
        repo_ids=source_repo_ids,
        aggr_repo_id=args.repo_id,
        roots=shards,
        aggr_root=output,
    )

    # Renumber the per-shard episode records. aggregate_datasets concatenates in
    # the order of repo_ids, so episode indices shift by the running total.
    merged: list[dict] = []
    for shard in shards:
        for record in read_jsonl(shard / METADATA_NAME):
            merged.append({**record, "episode_index": len(merged), "source_shard": str(shard.name)})
    (output / METADATA_NAME).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in merged) + "\n",
        encoding="utf-8",
    )

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    expected_frames = sum(record["frames"] for record in merged)
    if info["total_episodes"] != len(merged):
        raise RuntimeError(
            f"Episode count mismatch after merge: info={info['total_episodes']} metadata={len(merged)}"
        )
    if info["total_frames"] != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch after merge: info={info['total_frames']} metadata={expected_frames}"
        )

    shard_summaries = [
        json.loads((shard / SUMMARY_NAME).read_text(encoding="utf-8")) for shard in shards
    ]
    combined = combine_summaries(
        shard_summaries,
        repo_id=args.repo_id,
        shard_names=[shard.name for shard in shards],
        accepted_episodes=len(merged),
    )
    (output / SUMMARY_NAME).write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"  episodes    : {info['total_episodes']}")
    print(f"  frames      : {info['total_frames']}")
    print(f"  attempts    : {combined['attempts']}")
    print(f"  success rate: {combined['success_rate']:.1%}")
    print(f"  output      : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
