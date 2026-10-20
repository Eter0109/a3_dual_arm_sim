from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _check_cookie_2x5_instruction(episode: dict, collection_summary: dict) -> None:
    """Require the recorded instruction to equal the summary's declared one.

    A model trained on per-column prompts evaluated with the short default (or
    vice versa) sees a language distribution it never learned.
    """
    expected_instruction = collection_summary.get("task_instruction")
    if episode.get("task") != expected_instruction:
        raise ValueError(
            "2x5 episode task instruction disagrees with the collection summary "
            f"({episode.get('task')!r} != {expected_instruction!r}); "
            "prompt-style mismatch between collection and evaluation"
        )


def _check_cookie_2x10_episode(episode: dict, collection_summary: dict) -> None:
    from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

    metadata = episode.get("randomization", {})
    plan = Cookie2x10Plan.from_seed(collection_summary["first_grasp_mode"], episode["seed"])
    if (
        metadata.get("grasp_counts") != list(plan.batch_sizes)
        or metadata.get("first_grasp_count") != plan.first_grasp
        or metadata.get("cookies_in_target") != 20
        or metadata.get("cookies_in_source") != 60
        or metadata.get("target_column_counts") != [10, 10]
        or episode.get("task") != plan.prompt(metadata.get("source_column"))
    ):
        raise ValueError("2x10 episode counts or task instruction disagree with its plan")
    reports = metadata.get("grasp_reports", [])
    if len(reports) != 4 or any(
        report.get("planned_count") != planned
        or report.get("lifted") != planned
        or len(report.get("cookies", [])) != planned
        or not report.get("released")
        for report, planned in zip(reports, plan.batch_sizes, strict=True)
    ):
        raise ValueError("2x10 dataset requires four confirmed grasp reports, including zero skips")
    transferred = [cookie for report in reports for cookie in report["cookies"]]
    if len(set(transferred)) != 20:
        raise ValueError("2x10 grasp reports must describe twenty distinct cookies")


def audit_training_dataset(root: Path, *, repo_id: str) -> dict[str, Any]:
    """Fail fast on the parts of the LeRobot v3 contract used by A3 learned policies."""
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    metadata_path = root / "a3_episode_metadata.jsonl"
    collection_summary_path = root / "collection_summary.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing A3 episode metadata: {metadata_path}")
    if not collection_summary_path.is_file():
        raise FileNotFoundError(f"Missing grasp collection summary: {collection_summary_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    collection_summary = json.loads(collection_summary_path.read_text(encoding="utf-8"))
    if collection_summary.get("schema_version", 0) < 2:
        raise ValueError(
            "Dataset uses the obsolete transient-lift success contract; recollect it with the "
            "current stable-grasp expert"
        )
    if collection_summary.get("task") not in {"a3_grasp", "cookie_transfer"}:
        raise ValueError("Unsupported A3 training task")
    if (
        collection_summary.get("task") == "cookie_transfer"
        and collection_summary.get("stored_action_mode") != "joint_position"
    ):
        raise ValueError("Cookie training datasets require joint_position actions")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("A3 policy training requires a LeRobot v3.0 dataset")
    features = info.get("features", {})
    arm_mode = collection_summary.get("training_arm_mode", "dual")
    if arm_mode not in {"dual", "left"}:
        raise ValueError("Unsupported training_arm_mode")
    if arm_mode == "left" and collection_summary.get("task") != "cookie_transfer":
        raise ValueError("Left-only training is restricted to cookie_transfer")
    action_dim = 8 if arm_mode == "left" else 16
    state_contract = collection_summary.get("training_state_contract")
    if state_contract not in (None, "left_joint7_gripper1_eef_world_xyz_m3"):
        raise ValueError(f"Unsupported training_state_contract: {state_contract}")
    if state_contract and arm_mode != "left":
        raise ValueError("Left EEF position requires left-arm training mode")
    state_dim = 11 if state_contract else action_dim
    expected_shapes = {
        **{key: [256, 256, 3] for key in CAMERA_KEYS},
        "observation.state": [state_dim],
        "action": [action_dim],
    }
    for key, shape in expected_shapes.items():
        actual = features.get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"Feature {key!r} has shape {actual}, expected {shape}")

    # The summary records the storage format the collection *intended*.  LeRobot
    # decides the format from the declared feature dtype, and the two can drift
    # apart -- a recorder that passes `use_videos` without threading it into
    # `lerobot_features` writes PNG frames inline while the summary claims video.
    # The cost is about ten times the disk, which is worth catching here rather
    # than when a disk fills at episode 900.
    if "use_videos" in collection_summary:
        declared = {features.get(key, {}).get("dtype") for key in CAMERA_KEYS}
        expected_dtype = "video" if collection_summary["use_videos"] else "image"
        if declared != {expected_dtype}:
            raise ValueError(
                f"Collection summary says use_videos={collection_summary['use_videos']} "
                f"but the dataset declares camera dtypes {sorted(declared)}; "
                f"expected {{{expected_dtype!r}}}"
            )

    episodes = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes or not all(item.get("success") is True for item in episodes):
        raise ValueError("Training dataset must contain at least one successful expert episode")
    if info.get("total_episodes") != len(episodes):
        raise ValueError("LeRobot and A3 episode metadata counts do not match")

    # Sidecar rows can survive a writer interruption even when their buffered
    # Parquet episodes do not. Check the persisted training data, not just the
    # recorder's success ledger.
    import pyarrow.parquet as pq

    episode_lengths = {}
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            index = row["episode_index"]
            if index in episode_lengths:
                raise ValueError(f"Duplicate persisted episode {index}")
            episode_lengths[index] = row["length"]
    expected_episodes = set(range(len(episodes)))
    if set(episode_lengths) != expected_episodes:
        raise ValueError(
            f"Persisted episode metadata differs from A3 ledger: "
            f"missing={sorted(expected_episodes - set(episode_lengths))}"
        )
    data_lengths = {index: 0 for index in expected_episodes}
    frame_count = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        for index in pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist():
            if index not in data_lengths:
                raise ValueError(f"Unexpected persisted episode {index}")
            data_lengths[index] += 1
            frame_count += 1
    if frame_count != info.get("total_frames") or any(
        data_lengths[index] != episode_lengths[index] or data_lengths[index] != episodes[index]["frames"]
        for index in expected_episodes
    ):
        raise ValueError("Persisted Parquet frame counts differ from episode metadata")

    target_layout = collection_summary.get("target_layout")
    if target_layout == "2x10":
        for episode in episodes:
            _check_cookie_2x10_episode(episode, collection_summary)
    elif target_layout == "multi":
        components = {
            component["repo_id"]: component["target_layout"]
            for component in collection_summary.get("components", [])
        }
        for episode in episodes:
            source = episode.get("source_dataset")
            layout = components.get(source)
            if layout == "2x10":
                _check_cookie_2x10_episode(episode, collection_summary)
            elif layout == "2x5":
                _check_cookie_2x5_instruction(episode, collection_summary)
            else:
                raise ValueError(f"Unknown merged episode source {source!r}")
    else:
        for episode in episodes:
            _check_cookie_2x5_instruction(episode, collection_summary)

    return {
        "root": str(root),
        "repo_id": repo_id,
        "episodes": len(episodes),
        "frames": int(info["total_frames"]),
        "tasks": int(info["total_tasks"]),
        "camera_keys": list(CAMERA_KEYS),
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
