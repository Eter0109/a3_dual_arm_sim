"""Merge per-shard collection datasets into one training dataset.

LeRobot's writer needs a private directory per process, so running the
collection in parallel necessarily produces one dataset per shard.  This
concatenates them into the single dataset that training expects.

Two checks are hard failures rather than warnings, because both would produce a
dataset that looks fine and trains badly:

*Duplicate seeds.*  A randomised episode is a pure function of its seed, so the
same seed in two shards means two identical episodes.  `collect_sharded.sh`
makes this impossible by construction, but "impossible by construction" is
exactly the kind of claim that breaks quietly, so it is verified here against
the saved sidecars.

*Mismatched shard settings.*  `aggregate_datasets` will happily concatenate
shards collected under different randomization profiles or different storage
formats, and the result is a dataset whose metadata describes only some of its
episodes.  Mixing `use_videos` is especially bad: the aggregated feature schema
records `video` or `image` once for the whole dataset.

    python scripts/merge_codex_shards.py \
        --out-root datasets/codex_advanced_1000 \
        --repo-id local/a3-codex-advanced-1000 \
        --repo-prefix local/a3-codex-advanced-1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARY_NAME = "collection_summary.json"
METADATA_NAME = "a3_episode_metadata.jsonl"
ATTEMPTS_NAME = "attempts.jsonl"

#: Settings that must agree across every shard for the merged dataset to be
#: described honestly by a single metadata record.
UNIFORM_KEYS = (
    "task",
    "policy",
    "stored_action_mode",
    "config",
    "randomization_profile",
    "randomization_parameters",
    "source_column",
    "randomization_version",
    "max_steps",
    "use_videos",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True, help="Root holding the shard_* datasets")
    parser.add_argument("--repo-id", required=True, help="repo_id for the merged dataset")
    parser.add_argument(
        "--repo-prefix",
        default=None,
        help="Shard repo_id prefix used at collection time; defaults to --repo-id",
    )
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit shard roots, in merge order; defaults to every shard_* under --out-root",
    )
    parser.add_argument("--output", type=Path, default=None, help="Destination; defaults to <out-root>/merged")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base: Path = args.out_root.expanduser().resolve()
    output: Path = (args.output or base / "merged").expanduser().resolve()
    prefix = args.repo_prefix or args.repo_id

    shards = (
        [p.expanduser().resolve() for p in args.shards]
        if args.shards
        else sorted(p for p in base.glob("shard_*") if p.is_dir())
    )
    if not shards:
        raise SystemExit(f"no shard_* directories under {base}")
    for shard in shards:
        if not (shard / "meta" / "info.json").is_file():
            raise SystemExit(f"shard has no LeRobot metadata: {shard}")
        if not (shard / SUMMARY_NAME).is_file():
            raise SystemExit(f"shard has no {SUMMARY_NAME}: {shard}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    summaries = {shard: json.loads((shard / SUMMARY_NAME).read_text(encoding="utf-8")) for shard in shards}
    reference = summaries[shards[0]]
    for shard, summary in summaries.items():
        for key in UNIFORM_KEYS:
            if key in reference or key in summary:
                if summary.get(key) != reference.get(key):
                    raise SystemExit(
                        f"shard {shard.name} differs from {shards[0].name} on {key!r}: "
                        f"{summary.get(key)!r} vs {reference.get(key)!r}"
                    )

    # Seed hygiene, checked on saved episodes rather than attempts: a duplicate
    # attempt is harmless, a duplicate saved episode is wasted dataset.
    seed_owner: dict[int, str] = {}
    duplicates: list[tuple[int, str, str]] = []
    saved_by_shard: dict[Path, list[dict]] = {}
    for shard in shards:
        records = read_jsonl(shard / METADATA_NAME)
        saved_by_shard[shard] = records
        for record in records:
            seed = record.get("seed")
            if seed in seed_owner and seed is not None:
                duplicates.append((seed, seed_owner[seed], shard.name))
            else:
                seed_owner[seed] = shard.name
    if duplicates:
        for seed, first, second in duplicates[:20]:
            print(f"  duplicate seed {seed}: saved by {first} and {second}")
        raise SystemExit(
            f"{len(duplicates)} duplicated seeds across shards; merged dataset would contain "
            "identical episodes. Check the shard seed blocks before merging."
        )

    repo_ids = []
    for shard in shards:
        suffix = shard.name.removeprefix("shard_")
        repo_ids.append(f"{prefix}-shard-{suffix}")

    total_saved = sum(len(records) for records in saved_by_shard.values())
    print(f"merging {len(shards)} shards -> {output}")
    print(f"  episodes to merge: {total_saved}")
    if total_saved == 0:
        raise SystemExit("nothing to merge: every shard has zero saved episodes")

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.repo_id,
        roots=shards,
        aggr_root=output,
    )

    # aggregate_datasets concatenates in the order of repo_ids, so episode
    # indices shift by the running total per shard.
    merged: list[dict] = []
    for shard in shards:
        for record in saved_by_shard[shard]:
            merged.append({**record, "episode_index": len(merged), "source_shard": shard.name})
    (output / METADATA_NAME).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in merged) + "\n",
        encoding="utf-8",
    )

    attempts: list[dict] = []
    for shard in shards:
        for record in read_jsonl(shard / ATTEMPTS_NAME):
            attempts.append({**record, "source_shard": shard.name})
    (output / ATTEMPTS_NAME).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in attempts) + "\n",
        encoding="utf-8",
    )

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    expected_frames = sum(record["frames"] for record in merged)
    if info["total_episodes"] != len(merged):
        raise RuntimeError(
            f"episode count mismatch after merge: info={info['total_episodes']} metadata={len(merged)}"
        )
    if info["total_frames"] != expected_frames:
        raise RuntimeError(
            f"frame count mismatch after merge: info={info['total_frames']} metadata={expected_frames}"
        )

    combined = {
        **reference,
        "repo_id": args.repo_id,
        "merged": True,
        "shards": [
            {
                "name": shard.name,
                "root": str(shard),
                "episodes": len(saved_by_shard[shard]),
                "attempts": len(read_jsonl(shard / ATTEMPTS_NAME)),
                "seed_min": min((r["seed"] for r in saved_by_shard[shard]), default=None),
                "seed_max": max((r["seed"] for r in saved_by_shard[shard]), default=None),
            }
            for shard in shards
        ],
        "shard_count": len(shards),
        "accepted_episodes": len(merged),
        "attempts": len(attempts),
        "success_rate": len(merged) / len(attempts) if attempts else 0.0,
        "seed_min": min((r["seed"] for r in merged), default=None),
        "seed_max": max((r["seed"] for r in merged), default=None),
    }
    (output / SUMMARY_NAME).write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"  frames       : {info['total_frames']}")
    print(f"  attempts     : {len(attempts)}")
    print(f"  success rate : {combined['success_rate']:.1%}")
    print(f"  seed range   : {combined['seed_min']}..{combined['seed_max']}")
    print(f"  output       : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
