from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from a3_dual_arm_sim.learning.download import download_training_inputs
from a3_dual_arm_sim.learning.training import build_train_command, resolve_base_model


def test_local_base_does_not_contact_hub(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Local model must not contact Hub")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=forbidden)
    )
    assert resolve_base_model(tmp_path) == tmp_path.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_base_model(tmp_path / "missing")


def test_hub_base_preserves_revision(monkeypatch, tmp_path):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    assert resolve_base_model("lerobot/smolvla_base", revision="pinned") == tmp_path
    assert calls == [{"repo_id": "lerobot/smolvla_base", "revision": "pinned"}]


def test_public_training_rejects_unavailable_ema(tmp_path):
    with pytest.raises(ValueError, match="does not support EMA"):
        build_train_command(
            dataset_root=tmp_path,
            repo_id="test/a3",
            policy_source=tmp_path,
            output_dir=tmp_path / "out",
            steps=10,
            batch_size=1,
            seed=1,
            use_ema=True,
        )


def test_download_pins_all_inputs_and_resumes_only_same_revision(tmp_path, monkeypatch):
    calls = []

    class API:
        def dataset_info(self, repo, revision):
            return SimpleNamespace(sha="dataset-sha")

        def model_info(self, repo, revision=None):
            return SimpleNamespace(sha="base-sha" if repo == "lerobot/smolvla_base" else "vlm-sha")

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        root = Path(kwargs.get("local_dir", tmp_path / "cache"))
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps({"vlm_model_name": "test/vlm"}))
        return str(root)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=API, snapshot_download=snapshot)
    )
    args = (tmp_path / "data", tmp_path / "model", tmp_path / "manifest.json")
    result = download_training_inputs(*args)
    assert result["dataset"]["revision"] == "dataset-sha"
    assert result["model"]["revision"] == "base-sha"
    assert result["vlm"]["revision"] == "vlm-sha"
    assert [call[1]["revision"] for call in calls] == ["dataset-sha", "base-sha", "vlm-sha"]
    download_training_inputs(*args)
    marker = args[0] / ".a3_download_revision.json"
    marker.write_text(json.dumps({"repo_id": "wrong", "revision": "old"}))
    with pytest.raises(FileExistsError, match="unrelated"):
        download_training_inputs(*args)
