from __future__ import annotations

import copy
import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from a3_dual_arm_sim.core.contracts import (
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    ContractError,
    EpisodeContext,
)
from a3_dual_arm_sim.policies.lerobot import (
    LeRobotPolicyPlugin,
    _load_local_contract,
    validate_checkpoint_contract,
)


def checkpoint_config(policy_type="smolvla"):
    return {
        "type": policy_type,
        "input_features": {
            STATE: {"type": "STATE", "shape": [16]},
            **{key: {"type": "VISUAL", "shape": [3, 256, 256]}
               for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [16]}},
    }


@pytest.mark.parametrize("policy_type", ["smolvla", "pi05"])
def test_checkpoint_requires_a3_contract(policy_type):
    config = checkpoint_config(policy_type)
    assert validate_checkpoint_contract(config) == {key: key for key in config["input_features"]}


@pytest.mark.parametrize("dimension", [7, 14, 32])
def test_reject_generic_or_wrong_action_head(dimension):
    config = checkpoint_config("pi05")
    config["output_features"]["action"]["shape"] = [dimension]
    with pytest.raises(ValueError, match="action shape"):
        validate_checkpoint_contract(config)


@pytest.mark.parametrize("key", ["use_relative_actions", "adapt_to_pi_aloha",
                                 "use_delta_joint_actions_aloha"])
def test_reject_relative_action_semantics(key):
    config = checkpoint_config()
    config[key] = True
    with pytest.raises(ValueError, match="absolute joint"):
        validate_checkpoint_contract(config)


def test_explicit_camera_mapping_and_missing_mapping():
    config = checkpoint_config("pi05")
    camera = config["input_features"].pop(FRONT_IMAGE)
    config["input_features"]["observation.images.camera1"] = camera
    with pytest.raises(ValueError, match="explicit mapping"):
        validate_checkpoint_contract(config)
    result = validate_checkpoint_contract(
        config, feature_map={"observation.images.camera1": FRONT_IMAGE},
    )
    assert result["observation.images.camera1"] == FRONT_IMAGE


def test_reject_camera_free_or_privileged_inputs():
    config = checkpoint_config()
    config["input_features"] = {STATE: config["input_features"][STATE]}
    with pytest.raises(ValueError, match="at least one"):
        validate_checkpoint_contract(config)
    config["input_features"]["cookie_positions"] = {"type": "ENV", "shape": [30]}
    with pytest.raises(ValueError, match="Unsupported checkpoint observation"):
        validate_checkpoint_contract(config)


def test_reject_wrong_state_and_duplicate_mapping():
    config = checkpoint_config()
    config["input_features"][STATE]["shape"] = [7]
    with pytest.raises(ValueError, match="state shape"):
        validate_checkpoint_contract(config)
    with pytest.raises(ValueError, match="multiple checkpoint"):
        validate_checkpoint_contract(checkpoint_config(), feature_map={LEFT_WRIST_IMAGE: FRONT_IMAGE})


def test_reject_policy_mismatch_and_unknown_mapping():
    with pytest.raises(ValueError, match="Expected smolvla"):
        validate_checkpoint_contract(checkpoint_config("pi05"), expected_policy_type="smolvla")
    with pytest.raises(ValueError, match="unknown checkpoint"):
        validate_checkpoint_contract(checkpoint_config(), feature_map={"unknown": FRONT_IMAGE})


def make_local_files(tmp_path, config=None):
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "dataset"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config or checkpoint_config()))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (checkpoint / name).write_text("{}")
    meta = dataset / "meta"
    episodes = meta / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    (episodes / "file-000.parquet").touch()
    (meta / "stats.json").write_text("{}")
    (meta / "tasks.parquet").touch()
    (meta / "info.json").write_text(json.dumps({
        "robot_type": "A3_dual_arm",
        "features": {
            STATE: {"shape": [16]}, "action": {"shape": [16]},
            **{key: {"dtype": "image", "shape": [256, 256, 3]}
               for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)},
        },
    }))
    return checkpoint, dataset


def test_saved_processors_and_all_local_metadata_required(tmp_path):
    checkpoint, dataset = make_local_files(tmp_path)
    _load_local_contract(checkpoint, dataset, feature_map=None, expected_policy_type=None)
    (checkpoint / "policy_postprocessor.json").unlink()
    with pytest.raises(FileNotFoundError, match="saved policy_postprocessor"):
        _load_local_contract(checkpoint, dataset, feature_map=None, expected_policy_type=None)


def test_reject_dataset_without_mapped_rgb_camera(tmp_path):
    checkpoint, dataset = make_local_files(tmp_path)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"][RIGHT_WRIST_IMAGE]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Dataset lacks RGB camera"):
        _load_local_contract(checkpoint, dataset, feature_map=None, expected_policy_type=None)


def test_constructor_loads_only_metadata_and_saved_processors(tmp_path, monkeypatch):
    checkpoint, dataset = make_local_files(tmp_path, checkpoint_config("pi05"))
    calls = []
    config = SimpleNamespace(n_action_steps=50, chunk_size=50)

    def metadata(repo_id, *, root):
        calls.append(("metadata", repo_id, root))
        return SimpleNamespace(features={}, stats={})

    def load_config(path, **kwargs):
        calls.append(("config", path, kwargs))
        return config

    model = SimpleNamespace(eval=lambda: calls.append(("eval",)))

    def policy(cfg, **kwargs):
        calls.append(("policy", cfg, kwargs))
        return model

    def processors(**kwargs):
        calls.append(("processors", kwargs))
        return (lambda value: value), (lambda value: value)

    exports = {
        "torch": {"device": lambda name: SimpleNamespace(type=name.split(":")[0])},
        "lerobot.configs.policies": {
            "PreTrainedConfig": SimpleNamespace(from_pretrained=load_config),
        },
        "lerobot.datasets.dataset_metadata": {"LeRobotDatasetMetadata": metadata},
        "lerobot.policies.factory": {
            "make_policy": policy, "make_pre_post_processors": processors,
        },
        "lerobot.policies.utils": {"prepare_observation_for_inference": lambda *args: args},
    }
    for name, attributes in exports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    plugin = LeRobotPolicyPlugin(checkpoint, dataset, "local/a3", "cuda:0", policy_type="pi05")
    assert plugin.policy_type == "pi05"
    assert config.use_amp
    assert [call[0] for call in calls] == ["config", "metadata", "policy", "processors", "eval"]
    kwargs = calls[-2][1]
    assert kwargs["pretrained_path"] == str(checkpoint)
    assert "dataset_stats" not in kwargs  # Preserve the checkpoint's normalization.


def make_fake_plugin(output):
    calls = []
    plugin = object.__new__(LeRobotPolicyPlugin)
    plugin._feature_map = {STATE: STATE, "observation.images.camera1": FRONT_IMAGE}
    plugin._device = "cpu"
    plugin._torch = SimpleNamespace(inference_mode=nullcontext)
    plugin._prepare_observation = lambda obs, device, task, robot: calls.append(
        ("prepare", obs, device, task, robot)
    ) or obs
    plugin._preprocessor = lambda value: value
    plugin._postprocessor = lambda value: value

    class Tensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return output

    plugin._policy = SimpleNamespace(select_action=lambda batch: Tensor())
    return plugin, calls


def test_inference_maps_cameras_passes_language_and_returns_16_absolute_joints():
    output = np.arange(16, dtype=np.float32)[None]
    plugin, calls = make_fake_plugin(output)
    observation = {STATE: np.zeros(16), FRONT_IMAGE: np.zeros((8, 8, 3), dtype=np.uint8)}
    np.testing.assert_equal(plugin.act(observation, "Pick five cookies"), output[0])
    assert calls[0][3:] == ("Pick five cookies", "A3_dual_arm")
    assert calls[0][1]["observation.images.camera1"] is observation[FRONT_IMAGE]


@pytest.mark.parametrize("output", [np.zeros(7), np.zeros((2, 16)), np.full(16, np.nan)])
def test_invalid_runtime_action_is_not_silently_reshaped(output):
    plugin, _ = make_fake_plugin(output)
    observation = {STATE: np.zeros(16), FRONT_IMAGE: np.zeros((8, 8, 3), dtype=np.uint8)}
    with pytest.raises(ContractError):
        plugin.act(observation, "pick")


def test_missing_observation_and_bad_camera_rejected_before_model():
    plugin, calls = make_fake_plugin(np.zeros(16))
    with pytest.raises(ValueError, match="Missing observation"):
        plugin.act({STATE: np.zeros(16)}, "pick")
    with pytest.raises(ValueError, match="uint8"):
        plugin.act({STATE: np.zeros(16), FRONT_IMAGE: np.zeros((8, 8, 3))}, "pick")
    assert not calls


def test_skill_reset_clears_model_queue_and_both_processor_histories():
    calls = []
    plugin = object.__new__(LeRobotPolicyPlugin)
    plugin._torch = SimpleNamespace(manual_seed=lambda seed: calls.append(("seed", seed)))
    plugin._policy = SimpleNamespace(reset=lambda: calls.append(("model",)))
    plugin._preprocessor = SimpleNamespace(reset=lambda: calls.append(("pre",)))
    plugin._postprocessor = SimpleNamespace(reset=lambda: calls.append(("post",)))
    plugin.reset(EpisodeContext(seed=9, task="place", action_mode="joint_position"))
    assert calls == [("seed", 9), ("model",), ("pre",), ("post",)]


def test_generic_checkpoint_rejected_before_importing_heavy_dependencies(tmp_path, monkeypatch):
    config = copy.deepcopy(checkpoint_config("pi05"))
    config["output_features"]["action"]["shape"] = [7]
    checkpoint, dataset = make_local_files(tmp_path, config)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="action shape"):
        LeRobotPolicyPlugin(checkpoint, dataset, "local/a3", "cuda")
