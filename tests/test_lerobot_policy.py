"""Unit tests for the LeRobot policy adapter's pure logic.

Building the adapter loads a model, so the parts that decide *whether* it can be
built are covered here instead: key translation, key validation, and the
environment-variable configuration surface.
"""

from __future__ import annotations

import pytest

from a3_dual_arm_sim.contracts import (
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    ContractError,
)
from a3_dual_arm_sim.lerobot_policy import (
    ENVIRONMENT_KEYS,
    config_from_environment,
    translated_keys,
    validate_declared_keys,
)

A3_CAMERAS = (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)


def test_environment_keys_cover_the_policy_visible_observation() -> None:
    # The three cameras and the state vector are what a learned policy consumes;
    # the rest is available for checkpoints that declare proprioception extras.
    assert set(A3_CAMERAS) | {STATE} <= ENVIRONMENT_KEYS


def test_translated_keys_passes_names_through_without_a_map() -> None:
    declared = (*A3_CAMERAS, STATE)
    assert translated_keys(declared, {}) == declared


def test_translated_keys_maps_foreign_names_back_to_a3_keys() -> None:
    rename_map = {
        FRONT_IMAGE: "observation.images.camera1",
        LEFT_WRIST_IMAGE: "observation.images.camera2",
        RIGHT_WRIST_IMAGE: "observation.images.camera3",
    }
    declared = (*rename_map.values(), STATE)
    assert translated_keys(declared, rename_map) == (*A3_CAMERAS, STATE)


def test_declared_keys_matching_a3_are_accepted() -> None:
    validate_declared_keys((*A3_CAMERAS, STATE), {})


def test_foreign_camera_names_are_rejected_without_a_map() -> None:
    declared = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
        STATE,
    )
    with pytest.raises(ContractError) as error:
        validate_declared_keys(declared, {})
    message = str(error.value)
    assert "camera1" in message
    # The failure must be actionable, so it names what is available too.
    assert FRONT_IMAGE in message
    assert "rename_map" in message


def test_a_rename_map_makes_foreign_camera_names_acceptable() -> None:
    rename_map = {
        FRONT_IMAGE: "observation.images.camera1",
        LEFT_WRIST_IMAGE: "observation.images.camera2",
        RIGHT_WRIST_IMAGE: "observation.images.camera3",
    }
    validate_declared_keys((*rename_map.values(), STATE), rename_map)


def test_a_partially_mapped_checkpoint_still_reports_the_gap() -> None:
    rename_map = {FRONT_IMAGE: "observation.images.camera1"}
    with pytest.raises(ContractError, match="camera2"):
        validate_declared_keys(
            ("observation.images.camera1", "observation.images.camera2"), rename_map
        )


def test_config_from_environment_reads_every_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A3_POLICY_CHECKPOINT", "/tmp/ckpt")
    monkeypatch.setenv("A3_POLICY_DATASET_ROOT", "/tmp/data")
    monkeypatch.setenv("A3_POLICY_REPO_ID", "local/example")
    monkeypatch.setenv("A3_POLICY_TYPE", "act")
    monkeypatch.setenv("A3_POLICY_DEVICE", "cpu")
    monkeypatch.setenv("A3_POLICY_DTYPE", "float32")
    monkeypatch.setenv("A3_POLICY_N_ACTION_STEPS", "8")
    monkeypatch.setenv("A3_POLICY_LOAD_VLM_WEIGHTS", "false")
    monkeypatch.setenv(
        "A3_POLICY_RENAME_MAP", '{"observation.images.front": "observation.images.image"}'
    )

    config = config_from_environment()

    assert str(config.checkpoint) == "/tmp/ckpt"
    assert str(config.dataset_root) == "/tmp/data"
    assert config.repo_id == "local/example"
    assert config.policy_type == "act"
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.n_action_steps == 8
    assert config.load_vlm_weights is False
    assert config.rename_map == {FRONT_IMAGE: "observation.images.image"}


def test_config_from_environment_defaults_to_auto_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A3_POLICY_CHECKPOINT", "/tmp/ckpt")
    monkeypatch.setenv("A3_POLICY_DATASET_ROOT", "/tmp/data")
    for name in (
        "A3_POLICY_REPO_ID",
        "A3_POLICY_TYPE",
        "A3_POLICY_DEVICE",
        "A3_POLICY_DTYPE",
        "A3_POLICY_N_ACTION_STEPS",
        "A3_POLICY_LOAD_VLM_WEIGHTS",
        "A3_POLICY_RENAME_MAP",
    ):
        monkeypatch.delenv(name, raising=False)

    config = config_from_environment()

    assert config.device == "auto"
    assert config.repo_id == "local/a3-dual-arm"
    assert config.rename_map == {}
    # None means "keep whatever the checkpoint says", which matters because
    # load_vlm_weights=True on a self-contained checkpoint triggers a 2 GB fetch.
    assert config.load_vlm_weights is None
    assert config.n_action_steps is None


def test_config_from_environment_requires_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("A3_POLICY_CHECKPOINT", raising=False)
    monkeypatch.delenv("A3_POLICY_DATASET_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="A3_POLICY_CHECKPOINT"):
        config_from_environment()


def test_rename_map_must_be_a_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A3_POLICY_CHECKPOINT", "/tmp/ckpt")
    monkeypatch.setenv("A3_POLICY_DATASET_ROOT", "/tmp/data")
    monkeypatch.setenv("A3_POLICY_RENAME_MAP", '["not", "an", "object"]')
    with pytest.raises(TypeError, match="JSON object"):
        config_from_environment()


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ({"model.vlm_with_expert.vlm.layer.weight": None}, True),
        ({"model.vlm_with_expert.lm_expert.layer.weight": None}, False),
    ],
)
def test_embedded_vlm_detection_reads_the_weight_header(
    tmp_path, weights: dict[str, None], expected: bool
) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from a3_dual_arm_sim.checkpoint import checkpoint_embeds_vlm

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensors = {
        name: torch.zeros(1) if value is None else value for name, value in weights.items()
    }
    safetensors_torch.save_file(tensors, checkpoint / "model.safetensors")

    assert checkpoint_embeds_vlm(checkpoint) is expected


def test_embedded_vlm_detection_handles_a_missing_weight_file(tmp_path) -> None:
    from a3_dual_arm_sim.checkpoint import checkpoint_embeds_vlm

    assert checkpoint_embeds_vlm(tmp_path / "absent") is False
