"""Merge the separately collected 2x5 and 2x10 cookie datasets into one multi-task dataset.

Each benchmark records its own target layout, so a single-policy run needs both
components combined into one LeRobot dataset that keeps both task strings; the
policy then switches tasks by prompt and scene. The merged directory receives its
own ``collection_summary.json`` with ``target_layout: "multi"`` and a components
list, which the training audit and the SmolVLA trainer use to treat the two
episode families with their own rules (2x10 plan/grasp-report checks vs the 2x5
instruction check) and to size the tokenizer for the long plan prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.paths import project_root


def _from_project(path: Path) -> Path:
    return path if path.is_absolute() else project_root() / path


def _read_summary(root: Path) -> dict:
    path = root / "collection_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing collection summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_cookie_datasets(
    roots: list[Path], repo_ids: list[str], output: Path, output_repo_id: str
) -> dict:
    if len(roots) != 2 or len(repo_ids) != 2:
        raise ValueError("exactly two datasets (2x5 and 2x10) are required")
    roots = [Path(root).expanduser().resolve() for root in roots]
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Merge output already exists: {output}")
    summaries = [_read_summary(root) for root in roots]
    five = [index for index, item in enumerate(summaries) if item.get("target_layout") != "2x10"]
    ten = [index for index, item in enumerate(summaries) if item.get("target_layout") == "2x10"]
    if len(five) != 1 or len(ten) != 1:
        raise ValueError("expected exactly one 2x5 and one 2x10 component")
    index5, index10 = five[0], ten[0]
    for index in (index5, index10):
        if summaries[index].get("stored_action_mode") != "joint_position":
            raise ValueError(f"component {repo_ids[index]!r} does not store joint_position actions")

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    datasets = [LeRobotDataset(repo_id=repo_ids[index], root=roots[index]) for index in (index5, index10)]
    merge_datasets(datasets, output_repo_id, output_dir=output)
    merged_meta = LeRobotDatasetMetadata(output_repo_id, root=output)

    rows = []
    per_component_episodes = {}
    for index in (index5, index10):
        metadata_path = roots[index] / "a3_episode_metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing A3 episode metadata: {metadata_path}")
        component_rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in component_rows:
            row["source_dataset"] = repo_ids[index]
        rows.extend(component_rows)
        per_component_episodes[index] = len(component_rows)
    if len(rows) != int(merged_meta.total_episodes):
        raise RuntimeError(
            f"merged dataset has {merged_meta.total_episodes} episodes but the A3 "
            f"metadata rows total {len(rows)}; refusing an inconsistent merge"
        )
    (output / "a3_episode_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary5, summary10 = summaries[index5], summaries[index10]
    summary = {
        "schema_version": max(int(summary5.get("schema_version", 2)), int(summary10.get("schema_version", 2))),
        "task": "cookie_transfer",
        "target_layout": "multi",
        "policy": "multi",
        "prompt_style": summary5.get("prompt_style", "training"),
        "task_instruction": summary5.get("task_instruction", "transfer 10 cookies into target box"),
        "source_action_mode": "cartesian_delta",
        "stored_action_mode": "joint_position",
        "use_videos": bool(summary5.get("use_videos", True) and summary10.get("use_videos", True)),
        "first_grasp_mode": summary10.get("first_grasp_mode", "random"),
        "seed_start": min(int(summary5.get("seed_start", 0)), int(summary10.get("seed_start", 0))),
        "components": [
            {
                "repo_id": repo_ids[index5],
                "root": str(roots[index5]),
                "target_layout": "2x5",
                "episodes": per_component_episodes[index5],
                "seed_start": summary5.get("seed_start"),
                "randomization_profile": summary5.get("randomization_profile"),
                "source_column": summary5.get("source_column"),
            },
            {
                "repo_id": repo_ids[index10],
                "root": str(roots[index10]),
                "target_layout": "2x10",
                "episodes": per_component_episodes[index10],
                "seed_start": summary10.get("seed_start"),
                "randomization_profile": summary10.get("randomization_profile"),
                "source_column": summary10.get("source_column"),
                "first_grasp_mode": summary10.get("first_grasp_mode"),
            },
        ],
    }
    for key in (
        "placement_order",
        "complete_demonstrations",
        "training_arm_mode",
        "training_joint_order",
        "environment_action_dim",
        "training_state_contract",
    ):
        val5, val10 = summary5.get(key), summary10.get(key)
        if val5 is not None and val5 == val10:
            summary[key] = val5
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs=2, required=True, help="2x5 and 2x10 datasets")
    parser.add_argument("--repo-ids", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New merged dataset root")
    parser.add_argument("--output-repo-id", required=True)
    args = parser.parse_args()
    summary = merge_cookie_datasets(
        [_from_project(root) for root in args.roots],
        args.repo_ids,
        _from_project(args.output),
        args.output_repo_id,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())