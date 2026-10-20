from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from a3_dual_arm_sim.contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
)
from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.data.merge_cookie_datasets_cli import merge_cookie_datasets
from a3_dual_arm_sim.data.recording import ACTION, lerobot_features
from a3_dual_arm_sim.learning.training import tokenizer_max_length_for
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

TASK_2X5 = "transfer 10 cookies into target box"


def _mini_component(root: Path, repo_id: str, *, task: str, seed_start: int, episodes: int = 2) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # LeRobotDataset.create refuses an existing root and creates it itself.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        root=root,
        robot_type="A3_dual_arm",
        features=lerobot_features(64, 64, use_videos=False),
        use_videos=False,
        image_writer_threads=0,
    )
    rows = []
    for episode in range(episodes):
        for _ in range(3):
            dataset.add_frame(
                {
                    FRONT_IMAGE: np.zeros((64, 64, 3), np.uint8),
                    LEFT_WRIST_IMAGE: np.zeros((64, 64, 3), np.uint8),
                    RIGHT_WRIST_IMAGE: np.zeros((64, 64, 3), np.uint8),
                    STATE: np.zeros(16, np.float32),
                    VELOCITY: np.zeros(16, np.float32),
                    EEF_POSE: np.zeros(14, np.float32),
                    FORCE: np.zeros(18, np.float32),
                    ACTION: np.zeros(16, np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()
        rows.append({"episode_index": episode, "success": True, "seed": seed_start + episode, "task": task})
    (root / "a3_episode_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_component_summaries(tmp_path: Path) -> tuple[Path, Path]:
    root5 = tmp_path / "cookie_2x5"
    root10 = tmp_path / "cookie_2x10"
    _mini_component(root5, "local/mini-2x5", task=TASK_2X5, seed_start=20001)
    plan = Cookie2x10Plan.from_seed("random", 21001)
    _mini_component(root10, "local/mini-2x10", task=plan.prompt(1), seed_start=21001)
    (root5 / "collection_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "cookie_transfer",
                "task_instruction": TASK_2X5,
                "prompt_style": "training",
                "stored_action_mode": "joint_position",
                "source_column": "1",
                "seed_start": 20001,
                "use_videos": False,
            }
        ),
        encoding="utf-8",
    )
    (root10 / "collection_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "cookie_transfer",
                "target_layout": "2x10",
                "first_grasp_mode": "random",
                "stored_action_mode": "joint_position",
                "source_column": "1",
                "seed_start": 21001,
                "use_videos": False,
            }
        ),
        encoding="utf-8",
    )
    return root5, root10


def test_merge_round_trip_preserves_episodes_tasks_and_provenance(tmp_path: Path) -> None:
    root5, root10 = _write_component_summaries(tmp_path)
    output = tmp_path / "cookie_multi"
    summary = merge_cookie_datasets(
        [root5, root10], ["local/mini-2x5", "local/mini-2x10"], output, "local/mini-multi"
    )
    assert summary["target_layout"] == "multi"
    assert summary["first_grasp_mode"] == "random"
    assert summary["task_instruction"] == TASK_2X5
    assert [component["episodes"] for component in summary["components"]] == [2, 2]

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/mini-multi", root=output)
    assert int(meta.total_episodes) == 4
    rows = [
        json.loads(line)
        for line in (output / "a3_episode_metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4
    assert [row["source_dataset"] for row in rows] == ["local/mini-2x5"] * 2 + ["local/mini-2x10"] * 2
    episode_tasks = {task for tasks in list(meta.episodes["tasks"]) for task in tasks}
    assert TASK_2X5 in episode_tasks
    assert any(task != TASK_2X5 for task in episode_tasks)


def test_merge_refuses_existing_output_and_wrong_components(tmp_path: Path) -> None:
    root5, root10 = _write_component_summaries(tmp_path)
    with pytest.raises(ValueError, match="exactly one 2x5 and one 2x10"):
        merge_cookie_datasets(
            [root5, root5], ["local/mini-2x5", "local/mini-2x5"], tmp_path / "bad", "local/bad"
        )
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        merge_cookie_datasets(
            [root5, root10], ["local/mini-2x5", "local/mini-2x10"], output, "local/mini-multi"
        )


def _audit_fixture(tmp_path: Path) -> Path:
    """Handcrafted merged-style dataset for the audit's multi dispatch."""
    root = tmp_path / "multi"
    (root / "meta").mkdir(parents=True)
    features = {}
    for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE):
        features[key] = {"dtype": "image", "shape": [256, 256, 3]}
    for key, shape in ((STATE, [16]), (ACTION, [16])):
        features[key] = {"dtype": "float32", "shape": shape}
    plan = Cookie2x10Plan.from_seed("random", 21001)
    cookies = list(range(20))
    split, start = [], 0
    for size in plan.batch_sizes:
        split.append(cookies[start : start + size])
        start += size
    rows = [
        {"episode_index": 0, "success": True, "seed": 20001, "task": TASK_2X5,
         "frames": 12, "source_dataset": "local/mini-2x5"},
        {
            "episode_index": 1,
            "success": True,
            "seed": 21001,
            "task": plan.prompt(1),
            "frames": 12,
            "source_dataset": "local/mini-2x10",
            "randomization": {
                "grasp_counts": list(plan.batch_sizes),
                "first_grasp_count": plan.first_grasp,
                "cookies_in_target": 20,
                "cookies_in_source": 60,
                "target_column_counts": [10, 10],
                "source_column": 1,
                "grasp_reports": [
                    {
                        "planned_count": size,
                        "lifted": size,
                        "cookies": members,
                        "released": True,
                    }
                    for size, members in zip(plan.batch_sizes, split, strict=True)
                ],
            },
        },
    ]
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
                "total_episodes": 2,
                "total_frames": 24,
                "total_tasks": 2,
            }
        ),
        encoding="utf-8",
    )
    (root / "collection_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "cookie_transfer",
                "stored_action_mode": "joint_position",
                "target_layout": "multi",
                "components": [
                    {"repo_id": "local/mini-2x5", "target_layout": "2x5"},
                    {"repo_id": "local/mini-2x10", "target_layout": "2x10"},
                ],
                "first_grasp_mode": "random",
                "task_instruction": TASK_2X5,
                "use_videos": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "a3_episode_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0, 1], "length": [12, 12]}), episodes_path)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0] * 12 + [1] * 12}), data_path)
    return root


def test_audit_multi_dispatch_checks_each_component(tmp_path: Path) -> None:
    root = _audit_fixture(tmp_path)
    result = audit_training_dataset(root, repo_id="local/multi")
    assert result["episodes"] == 2

    # 2x5 instruction mismatch is caught by the 2x5 rule.
    rows = [json.loads(line) for line in (root / "a3_episode_metadata.jsonl").read_text().splitlines()]
    rows[0]["task"] = "Transfer 10 cookies from source column 1 into the target box in two batches of five."
    (root / "a3_episode_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="prompt-style mismatch"):
        audit_training_dataset(root, repo_id="local/multi")

    # Restore the instruction and corrupt the 2x10 plan instead.
    rows[0]["task"] = TASK_2X5
    rows[1]["randomization"]["cookies_in_source"] = 61
    (root / "a3_episode_metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="2x10 episode counts"):
        audit_training_dataset(root, repo_id="local/multi")


def test_tokenizer_length_for_multi_and_single_task_layouts() -> None:
    assert tokenizer_max_length_for({"target_layout": "2x10"}) == 128
    assert tokenizer_max_length_for({"target_layout": "multi"}) == 128
    assert tokenizer_max_length_for({"target_layout": "2x5"}) is None
    assert tokenizer_max_length_for({}) is None
