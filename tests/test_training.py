from __future__ import annotations

import json
from pathlib import Path

import pytest

from a3_dual_arm_sim.training import (
    CAMERA_KEYS,
    audit_training_dataset,
    build_train_command,
    prepare_a3_smolvla_source,
)


def _dataset(root: Path, *, success: bool = True) -> Path:
    features = {
        **{key: {"dtype": "image", "shape": [256, 256, 3]} for key in CAMERA_KEYS},
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action": {"dtype": "float32", "shape": [16]},
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
                "total_episodes": 1,
                "total_frames": 12,
                "total_tasks": 1,
            }
        )
    )
    (root / "a3_episode_metadata.jsonl").write_text(
        json.dumps({"episode_index": 0, "success": success}) + "\n"
    )
    (root / "collection_summary.json").write_text(
        json.dumps({"schema_version": 2, "task": "a3_grasp"}) + "\n"
    )
    return root


def test_training_dataset_audit_rejects_old_success_contract(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "old")
    (root / "collection_summary.json").write_text(
        json.dumps({"schema_version": 1, "task": "a3_grasp"}) + "\n"
    )
    with pytest.raises(ValueError, match="obsolete transient-lift"):
        audit_training_dataset(root, repo_id="local/test")


def test_training_dataset_audit_requires_success_and_contract(tmp_path: Path) -> None:
    summary = audit_training_dataset(_dataset(tmp_path / "good"), repo_id="local/test")
    assert summary["frames"] == 12
    assert summary["action_dim"] == 16

    with pytest.raises(ValueError, match="successful"):
        audit_training_dataset(_dataset(tmp_path / "failed", success=False), repo_id="local/test")


def test_training_dataset_audit_accepts_cookie_joint_actions(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "cookie")
    (root / "collection_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "cookie_transfer",
                "stored_action_mode": "joint_position",
            }
        )
        + "\n"
    )
    assert audit_training_dataset(root, repo_id="local/test")["episodes"] == 1
    summary = json.loads((root / "collection_summary.json").read_text())
    summary["stored_action_mode"] = "cartesian_delta"
    (root / "collection_summary.json").write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="joint_position"):
        audit_training_dataset(root, repo_id="local/test")


def test_adapted_checkpoint_uses_three_cameras_and_16d_io(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"type": "smolvla", "max_state_dim": 32, "max_action_dim": 32})
    )
    (base / "model.safetensors").write_bytes(b"weights")
    (base / "policy_preprocessor.json").write_text("{}")

    adapted = prepare_a3_smolvla_source(base, tmp_path / "adapted", device="cpu")
    config = json.loads((adapted / "config.json").read_text())
    assert set(config["input_features"]) == {*CAMERA_KEYS, "observation.state"}
    assert config["input_features"]["observation.state"]["shape"] == [16]
    assert config["output_features"]["action"]["shape"] == [16]
    assert (adapted / "model.safetensors").is_symlink()


def test_train_command_is_local_reproducible_smoke(tmp_path: Path) -> None:
    command = build_train_command(
        dataset_root=tmp_path / "data",
        repo_id="local/a3",
        policy_source=tmp_path / "policy",
        output_dir=tmp_path / "out",
        steps=1,
        batch_size=1,
        seed=7,
    )
    joined = " ".join(command)
    assert "--steps=1" in joined
    assert "--dataset.video_backend=pyav" in joined
    assert "--eval_freq=0" in joined
    assert "--wandb.enable=false" in joined
