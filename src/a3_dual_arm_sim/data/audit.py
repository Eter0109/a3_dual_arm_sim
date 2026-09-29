"""Validate canonical A3 LeRobot action datasets before collection or training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def audit_training_dataset(root: Path, *, repo_id: str) -> dict[str, Any]:
    """Fail fast on the parts of the LeRobot v3 contract used by SmolVLA."""
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
    if collection_summary.get("task") not in {"a3_grasp", "cookie_transfer", "cookie_skills"}:
        raise ValueError("Unsupported A3 training task")
    if (
        collection_summary.get("task") in {"cookie_transfer", "cookie_skills"}
        and collection_summary.get("stored_action_mode") != "joint_position"
    ):
        raise ValueError("Cookie SmolVLA datasets require joint_position actions")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("SmolVLA training requires a LeRobot v3.0 dataset")
    features = info.get("features", {})
    expected_shapes = {
        **{key: [256, 256, 3] for key in CAMERA_KEYS},
        "observation.state": [16],
        "action": [16],
    }
    for key, shape in expected_shapes.items():
        actual = features.get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"Feature {key!r} has shape {actual}, expected {shape}")

    episodes = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes or not all(item.get("success") is True for item in episodes):
        raise ValueError("Training dataset must contain at least one successful expert episode")
    if info.get("total_episodes") != len(episodes):
        raise ValueError("LeRobot and A3 episode metadata counts do not match")

    if collection_summary.get("task") == "cookie_skills":
        if collection_summary.get("schema_version") != 3:
            raise ValueError("Cookie skill datasets require schema_version 3")
        segments_path = root / "a3_skill_segments.jsonl"
        if not segments_path.is_file():
            raise FileNotFoundError(f"Missing skill metadata: {segments_path}")
        segments = [
            json.loads(line)
            for line in segments_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(segments) != len(episodes) or len(segments) % 4:
            raise ValueError("Skill segment count must match episodes in groups of four")
        expected = [(0, "PICK_FIVE"), (0, "PLACE_FIVE"),
                    (1, "PICK_FIVE"), (1, "PLACE_FIVE")]
        for start in range(0, len(segments), 4):
            group = segments[start:start + 4]
            if [(item["batch_index"], item["skill"]) for item in group] != expected:
                raise ValueError("Skill episodes must be PICK/PLACE/PICK/PLACE")
            if len({item["parent_seed"] for item in group}) != 1:
                raise ValueError("Skill episodes in a group must share a parent seed")
        for episode, segment in zip(episodes, segments, strict=True):
            if episode["episode_index"] != segment["episode_index"]:
                raise ValueError("LeRobot and skill episode indices differ")
            if episode["task"] != segment["instruction"]:
                raise ValueError("LeRobot and skill instructions differ")
            if episode["frames"] != segment["frames"]:
                raise ValueError("LeRobot and skill frame counts differ")

    return {
        "root": str(root),
        "repo_id": repo_id,
        "episodes": len(episodes),
        "frames": int(info["total_frames"]),
        "tasks": int(info["total_tasks"]),
        "camera_keys": list(CAMERA_KEYS),
        "state_dim": 16,
        "action_dim": 16,
    }
