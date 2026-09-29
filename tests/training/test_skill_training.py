from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from a3_dual_arm_sim.training import skills as training


def segments_for_parents(count):
    return [
        {"parent_seed": parent, "episode_index": parent * 4 + position,
         "batch_index": position // 2,
         "skill": "PICK_FIVE" if position % 2 == 0 else "PLACE_FIVE",
         "source_column_number": parent % 4 + 1}
        for parent in range(count) for position in range(4)
    ]


def test_parent_groups_never_leak_between_training_and_validation():
    segments = segments_for_parents(100)
    split = training.grouped_skill_split(segments)
    assert len(split["train_parent_seeds"]) == 80
    assert len(split["validation_parent_seeds"]) == 20
    assert len(split["train_episode_indices"]) == 320
    assert len(split["validation_episode_indices"]) == 80
    assert set(split["train_parent_seeds"]).isdisjoint(split["validation_parent_seeds"])
    for parent in range(100):
        indices = {parent * 4 + offset for offset in range(4)}
        assert indices.issubset(split["train_episode_indices"]) or indices.issubset(
            split["validation_episode_indices"]
        )
    assert training.grouped_skill_split(list(reversed(segments))) == split


def test_split_seed_deterministic_and_configurable():
    segments = segments_for_parents(20)
    split = training.grouped_skill_split(segments, seed=8)
    assert training.grouped_skill_split(segments, seed=8) == split
    assert training.grouped_skill_split(segments, seed=9) != split


def test_requires_two_complete_rollouts_and_valid_fraction():
    with pytest.raises(ValueError, match="two successful"):
        training.grouped_skill_split(segments_for_parents(1))
    for fraction in (0, 1, -0.2):
        with pytest.raises(ValueError, match="between zero"):
            training.grouped_skill_split(segments_for_parents(2), validation_fraction=fraction)
    incomplete = segments_for_parents(2)[:-1]
    with pytest.raises(ValueError, match="four PICK"):
        training.grouped_skill_split(incomplete)


def test_reject_duplicate_episode_and_inconsistent_source():
    segments = segments_for_parents(2)
    with pytest.raises(ValueError, match="Duplicate skill episode"):
        training.grouped_skill_split(segments + [segments[0]])
    segments[1]["source_column_number"] = 4
    with pytest.raises(ValueError, match="inconsistent source"):
        training.grouped_skill_split(segments)


def command_kwargs(tmp_path):
    return {
        "dataset_root": tmp_path / "dataset", "repo_id": "local/a3-skills",
        "base_model": tmp_path / "base", "policy_source": tmp_path / "prepared",
        "output_dir": tmp_path / "output", "train_episodes": [0, 1, 2, 3],
        "policy_features": {"observation.state": {"type": "STATE", "shape": [16]}},
        "base_config": {"max_state_dim": 32, "max_action_dim": 32, "chunk_size": 50},
    }


def test_smol_command_passes_grouped_episode_list(tmp_path):
    command = training.build_skill_train_command(policy_type="smolvla", **command_kwargs(tmp_path))
    assert "--dataset.episodes=[0,1,2,3]" in command
    assert "--batch_size=4" in command
    assert f"--policy.path={tmp_path / 'prepared'}" in command
    assert "--eval_freq=0" in command


def test_pi05_training_intentionally_adapts_32d_base_to_16d_dataset(tmp_path):
    command = training.build_skill_train_command(policy_type="pi05", **command_kwargs(tmp_path))
    assert "--policy.type=pi05" in command
    assert f"--policy.pretrained_path={tmp_path / 'base'}" in command
    assert "--policy.max_action_dim=32" in command
    assert '--policy.output_features={"action":{"type":"ACTION","shape":[16]}}' in command
    normalization = next(arg for arg in command if arg.startswith("--policy.normalization_mapping="))
    assert json.loads(normalization.split("=", 1)[1])["ACTION"] == "MEAN_STD"
    assert "--policy.use_relative_actions=false" in command
    assert "--batch_size=1" in command


def local_args(tmp_path, monkeypatch, *, policy_type="smolvla"):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "collection_summary.json").write_text(json.dumps({"task": "cookie_skills"}))
    (root / "a3_skill_segments.jsonl").write_text(
        "\n".join(json.dumps(item) for item in segments_for_parents(5))
    )
    (root / "meta" / "info.json").write_text(json.dumps({"features": {
        key: {"shape": [256, 256, 3]} for key in training.CAMERAS
    }}))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({
        "type": policy_type, "max_state_dim": 32, "max_action_dim": 32,
    }))
    (base / "model.safetensors").touch()
    monkeypatch.setattr(training, "audit_training_dataset", lambda *args, **kwargs: {"episodes": 20})
    return argparse.Namespace(
        root=root, repo_id="local/a3-skills", base_model=base, output=tmp_path / "output",
        policy_type=policy_type, validation_fraction=0.2, seed=0, steps=20000,
        batch_size=None, device="cuda", num_workers=0, manifest=None, run=False,
    )


@pytest.mark.parametrize("policy_type", ["smolvla", "pi05"])
def test_dry_run_does_not_write_files_or_start_training(tmp_path, monkeypatch, policy_type):
    args = local_args(tmp_path, monkeypatch, policy_type=policy_type)
    before = set(tmp_path.rglob("*"))
    monkeypatch.setattr(training.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched training"))
    monkeypatch.setattr(training, "prepare_a3_smolvla_source",
                        lambda *args, **kwargs: pytest.fail("modified checkpoint"))
    result = training.prepare_training(args)
    assert result["dry_run"]
    assert set(tmp_path.rglob("*")) == before
    assert len(result["manifest"]["validation_episode_indices"]) == 4


def test_run_writes_manifest_outside_training_output_then_launches(tmp_path, monkeypatch):
    args = local_args(tmp_path, monkeypatch, policy_type="pi05")
    args.run = True
    calls = []
    monkeypatch.setattr(training.subprocess, "run", lambda command, **kwargs: calls.append(command))
    result = training.prepare_training(args)
    assert len(calls) == 1
    assert not args.output.exists()
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["validation_episode_indices"]
    assert "no online evaluation" in manifest["evaluation_note"]


def test_run_refuses_overwrite_existing_manifest(tmp_path, monkeypatch):
    args = local_args(tmp_path, monkeypatch, policy_type="pi05")
    args.run = True
    args.manifest = tmp_path / "manifest.json"
    args.manifest.write_text("original")
    with pytest.raises(FileExistsError, match="overwrite validation"):
        training.prepare_training(args)
    assert args.manifest.read_text() == "original"


def test_old_whole_task_dataset_rejected(tmp_path, monkeypatch):
    args = local_args(tmp_path, monkeypatch)
    (args.root / "collection_summary.json").write_text(json.dumps({"task": "cookie_batch"}))
    with pytest.raises(ValueError, match="not the old whole-task"):
        training.prepare_training(args)


def test_checkpoint_mismatch_and_existing_output_rejected(tmp_path, monkeypatch):
    args = local_args(tmp_path, monkeypatch)
    args.policy_type = "pi05"
    with pytest.raises(ValueError, match="does not match"):
        training.prepare_training(args)
    args.output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        training.prepare_training(args)
