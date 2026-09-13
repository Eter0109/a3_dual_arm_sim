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

    first = json.loads((shards[0] / SUMMARY_NAME).read_text(encoding="utf-8"))
    attempts = 0
    for shard in shards:
        attempts += json.loads((shard / SUMMARY_NAME).read_text(encoding="utf-8")).get("attempts", 0)
    combined = {
        "schema_version": 2,
        "task": first["task"],
        "success_contract": first["success_contract"],
        "prompt": first["prompt"],
        "repo_id": args.repo_id,
        "shards": [shard.name for shard in shards],
        "shard_count": len(shards),
        "position_noise_m": first["position_noise_m"],
        "yaw_noise_rad": first["yaw_noise_rad"],
        "requested_episodes": sum(
            json.loads((shard / SUMMARY_NAME).read_text(encoding="utf-8"))["requested_episodes"]
            for shard in shards
        ),
        "accepted_episodes": len(merged),
        "attempts": attempts,
        "success_rate": len(merged) / attempts if attempts else 0.0,
    }
    (output / SUMMARY_NAME).write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"  episodes    : {info['total_episodes']}")
    print(f"  frames      : {info['total_frames']}")
    print(f"  attempts    : {attempts}")
    print(f"  success rate: {combined['success_rate']:.1%}")
    print(f"  output      : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
