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
    expected_shapes = {
        **{key: [256, 256, 3] for key in CAMERA_KEYS},
        "observation.state": [action_dim],
        "action": [action_dim],
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

    return {
        "root": str(root),
        "repo_id": repo_id,
        "episodes": len(episodes),
        "frames": int(info["total_frames"]),
        "tasks": int(info["total_tasks"]),
        "camera_keys": list(CAMERA_KEYS),
        "state_dim": action_dim,
        "action_dim": action_dim,
    }
