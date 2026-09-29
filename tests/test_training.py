from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from a3_dual_arm_sim.learning.training import (
    CAMERA_KEYS,
    audit_training_dataset,
    build_train_act_command,
    build_train_command,
    dataset_camera_config,
    prepare_a3_smolvla_source,
    train_act,
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
    assert (adapted / "model.safetensors").exists()
    if sys.platform != "win32":
        assert (adapted / "model.safetensors").is_symlink()
    left = prepare_a3_smolvla_source(base, tmp_path / 'left', device='cpu', action_dim=8)
    left_config = json.loads((left/'config.json').read_text())
    assert left_config['input_features']['observation.state']['shape'] == [8]
    assert left_config['output_features']['action']['shape'] == [8]


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
    assert "--env_eval_freq=0" in joined
    assert "--wandb.enable=false" in joined
    assert "--save_freq=2000" in command
    assert "--ema.enable=true" in command
    assert "--ema.decay=0.99" in command


def test_training_schedule_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"type": "smolvla", "max_state_dim": 32, "max_action_dim": 32})
    )
    (base / "model.safetensors").write_bytes(b"weights")
    path = prepare_a3_smolvla_source(
        base,
        tmp_path / "adapted",
        device="cpu",
        lr=3e-5,
        decay_steps=20000,
        warmup_steps=100,
        num_steps=25,
    )
    config = json.loads((path / "config.json").read_text())
    assert config["optimizer_lr"] == 3e-5
    assert config["scheduler_decay_steps"] == 20000
    assert config["scheduler_warmup_steps"] == 100
    assert config["num_steps"] == 25


def test_train_act_command_and_dry_run(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "act_dataset")
    (root / "collection_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task": "cookie_transfer",
                "stored_action_mode": "joint_position",
                "training_arm_mode": "left",
                "config": {
                    "cameras": {
                        "front_position_m": [0.5, -0.18, 1.25],
                        "front_fovy_deg": 48.0,
                    }
                },
            }
        )
        + "\n"
    )
    # Fix dataset state/action shape for left-only arm (8D)
    meta = json.loads((root / "meta" / "info.json").read_text())
    meta["features"]["observation.state"]["shape"] = [8]
    meta["features"]["action"]["shape"] = [8]
    (root / "meta" / "info.json").write_text(json.dumps(meta))

    cameras = dataset_camera_config(root)
    assert cameras is not None
    assert cameras.front_fovy_deg == 48.0
    assert cameras.front_position_m == (0.5, -0.18, 1.25)

    command = build_train_act_command(
        dataset_root=root,
        repo_id="local/test-act",
        output_dir=tmp_path / "out_act",
        steps=50,
        batch_size=2,
        seed=42,
        device="cuda",
        chunk_size=100,
        n_action_steps=100,
    )
    joined = " ".join(command)
    assert "--policy.type=act" in joined
    assert "--policy.chunk_size=100" in joined
    assert "--policy.n_action_steps=100" in joined
    assert "--steps=50" in joined
    assert "--batch_size=2" in joined

    res = train_act(
        dataset_root=root,
        repo_id="local/test-act",
        output_dir=tmp_path / "out_act",
        steps=50,
        batch_size=2,
        seed=42,
        device="cuda",
        dry_run=True,
    )
    assert res["episodes"] == 1
    assert res["action_dim"] == 8
    assert res["device"] == "cuda"
    assert res["command"] == command
