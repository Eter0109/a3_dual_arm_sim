import json
from pathlib import Path

import pytest

from a3_dual_arm_sim.learning.checkpoint_bundle import (
    load_processors,
    local_repo_id,
    processor_files,
)


def test_local_id_and_override(tmp_path):
    assert local_repo_id(tmp_path / "cookie trial") == "local/cookie-trial"
    assert local_repo_id(tmp_path, "owner/data") == "owner/data"


def test_partial_processor_is_not_reconstructed(tmp_path):
    (tmp_path / "policy_preprocessor.json").write_text('{"steps": []}')
    with pytest.raises(FileNotFoundError, match="policy_postprocessor"):
        load_processors(None, tmp_path, tmp_path)


def test_missing_processor_state_is_rejected(tmp_path):
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": "missing.safetensors"}]})
    )
    with pytest.raises(FileNotFoundError, match="processor state"):
        processor_files(tmp_path)


def test_cli_config_override_and_default_output(tmp_path, monkeypatch):
    import sys

    from a3_dual_arm_sim import cli
    from a3_dual_arm_sim.workflows import cli_handlers

    config = tmp_path / "defaults.json"
    config.write_text('{"steps": 23, "batch_size": 7}')
    captured = {}

    def train(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli_handlers, "train_smolvla", train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "a3-sim",
            "train-smolvla",
            "--root",
            "datasets/trial",
            "--train-config=" + str(config),
            "--batch-size",
            "4",
        ],
    )
    assert cli.main() == 0
    assert captured["steps"] == 23
    assert captured["batch_size"] == 4
    assert captured["output_dir"] == Path("outputs/trial_smolvla")
    assert captured["repo_id"] is None
