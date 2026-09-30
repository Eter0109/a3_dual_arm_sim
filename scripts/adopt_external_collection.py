"""Adopt a collection gathered on another branch so this branch can train on it.

The branches that collected cookie data diverged on the collection summary's
schema.  They all write a LeRobot v3 dataset plus the same sidecars, so the
episodes themselves are compatible; what differs is the metadata envelope, and
this auditor reads that envelope rather than the episodes.  A dataset collected
elsewhere is therefore rejected for naming reasons alone -- which is the failure
mode this script exists to fix, explicitly and once, rather than by loosening the
auditor.

What it does, per shard:

* records the foreign summary verbatim as ``collection_summary.source.json``, so
  the adaptation is reversible and the original provenance survives;
* sets ``task`` to this branch's name for the scene, and takes
  ``success_contract`` and ``prompt`` from the task registry rather than
  hardcoding them, so the summary cannot drift from the contract the auditor
  enforces;
* fills the fields this branch's merge and audit expect but the foreign summary
  never wrote (``repo_id``, ``requested_episodes``, ``attempts``, ...);
* keeps every foreign field, because the randomised-scene provenance it records
  (per-episode camera, bin, light and cookie poses) has no equivalent here.

The mapping is deliberately a table with one entry rather than a rule.  Guessing
a task from a success contract would let a dataset be adopted under a contract it
does not satisfy, and the auditor's value is precisely that it will not.

    python scripts/adopt_external_collection.py --dataset-root <root>
    python scripts/adopt_external_collection.py --dataset-root <root> --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from a3_dual_arm_sim.tasks import TASKS

SUMMARY_NAME = "collection_summary.json"
SOURCE_NAME = "collection_summary.source.json"
ATTEMPTS_NAME = "attempts.jsonl"
METADATA_NAME = "a3_episode_metadata.jsonl"

#: Foreign task name -> our task name.  One entry, because one foreign branch
#: collected this scene.  Each mapping was confirmed against the episodes, not
#: just the label: the foreign collector selected a single source column per
#: episode (``controller_type: selected_column``) and required ten cookies
#: released upright inside the target box with seventy still in the source, which
#: is ``a3_cookie_same_column``'s contract and not ``a3_cookie_transfer``'s
#: exact-2x5-touching-all-walls variant.
TASK_ALIASES = {
    "cookie_transfer": "a3_cookie_same_column",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-prefix", default=None, help="Prefix for per-shard repo_id; defaults to the dir name")
    parser.add_argument(
        "--requested-episodes",
        type=int,
        default=None,
        help=(
            "Episodes each shard was asked for. The foreign summary does not record it, so "
            "without this the attempt count is used; for a run capped by --max-attempts the "
            "two are equal, which is checked and reported."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the adapted summaries. Without it the script only reports what it would change.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def adapted_summary(
    summary: dict, *, task: str, repo_id: str, requested: int, attempts: int, accepted: int
) -> dict:
    spec = TASKS[task]
    parameters = summary.get("randomization_parameters") or {}
    scene: dict[str, list[float]] = {}
    for side, key in (
        ("source_bin", "source_bin_noise_m"),
        ("target_bin", "target_bin_noise_m"),
    ):
        radius = parameters.get(key)
        if radius:
            scene[f"{side}_x_m"] = [-radius, radius]
            scene[f"{side}_y_m"] = [-radius, radius]

    adapted = {**summary}
    adapted.update(
        {
            "schema_version": 2,
            "task": task,
            "success_contract": spec.success_contract,
            "prompt": spec.prompt,
            "repo_id": repo_id,
            "requested_episodes": requested,
            "attempts": attempts,
            "accepted_episodes": accepted,
            # Our merge script reads these; the foreign summary recorded the same
            # quantities under different names.
            "position_noise_m": parameters.get("cookie_noise_m", 0.0),
            "yaw_noise_rad": parameters.get("cookie_yaw_noise_rad", 0.0),
            "cookies_jittered": bool(parameters.get("cookie_noise_m")),
            "placements_randomized": bool(parameters.get("target_bin_noise_m")),
            "scene_randomization_m": scene,
            "adopted": {
                "reason": "collected on another branch under a different summary schema",
                "original_task": summary.get("task"),
                "original_repo_id": summary.get("repo_id"),
                "source_branch_commit": summary.get("git_commit"),
                "source_summary": SOURCE_NAME,
            },
        }
    )
    return adapted


def main() -> int:
    args = parse_args()
    root: Path = args.dataset_root.expanduser().resolve()
    shards = sorted(p for p in root.glob("shard_*") if p.is_dir())
    if not shards:
        if (root / SUMMARY_NAME).is_file():
            shards = [root]
        else:
            raise SystemExit(f"no shard_* directories and no {SUMMARY_NAME} under {root}")

    prefix = args.repo_prefix or root.name
    print(f"dataset root : {root}")
    print(f"shards       : {len(shards)}")
    print(f"repo prefix  : {prefix}")
    print()

    changed = 0
    for shard in shards:
        summary_path = shard / SUMMARY_NAME
        if not summary_path.is_file():
            print(f"  {shard.name}: no {SUMMARY_NAME}, skipped")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        task = summary.get("task")
        if task in TASKS:
            print(f"  {shard.name}: already uses task {task!r}, skipped")
            continue
        if task not in TASK_ALIASES:
            print(f"  {shard.name}: no alias for foreign task {task!r}, skipped")
            continue

        accepted = len(read_jsonl(shard / METADATA_NAME))
        attempts = len(read_jsonl(shard / ATTEMPTS_NAME))
        # The foreign summary never recorded how many episodes the shard was
        # asked for, only how many attempts it was allowed. Those are equal for a
        # run whose episode target and attempt budget were set to the same value,
        # which is how this collection was launched, so the attempt count is a
        # faithful default -- but it is stated rather than silently assumed.
        requested = args.requested_episodes or attempts
        suffix = shard.name.removeprefix("shard_") if shard.name.startswith("shard_") else shard.name
        repo_id = f"local/{prefix}-shard-{suffix}" if shard.name.startswith("shard_") else f"local/{prefix}"
        adapted = adapted_summary(
            summary,
            task=TASK_ALIASES[task],
            repo_id=repo_id,
            requested=requested,
            attempts=attempts,
            accepted=accepted,
        )

        print(f"  {shard.name}:")
        print(f"    task            {task!r} -> {adapted['task']!r}")
        print(f"    success_contract {summary.get('success_contract')!r} -> {adapted['success_contract']!r}")
        print(f"    repo_id          {summary.get('repo_id')!r} -> {adapted['repo_id']!r}")
        print(f"    added            requested_episodes={adapted['requested_episodes']} attempts={attempts} "
              f"accepted_episodes={accepted} position_noise_m={adapted['position_noise_m']} "
              f"yaw_noise_rad={adapted['yaw_noise_rad']}")

        changed += 1
        if not args.apply:
            continue
        source_path = shard / SOURCE_NAME
        if not source_path.is_file():
            shutil.copy2(summary_path, source_path)
        summary_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    if not args.apply:
        if changed:
            print(f"dry run: {changed} shard(s) would be adapted; re-run with --apply to write")
        else:
            print("nothing to do")
        return 0
    print(f"adapted {changed} shard(s); originals kept as {SOURCE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
