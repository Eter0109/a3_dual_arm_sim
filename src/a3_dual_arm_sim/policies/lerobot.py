"""LeRobot executor accepting only A3-finetuned absolute 16D joint policies."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.core.contracts import (
    FRONT_IMAGE,
    JOINT_ACTION_DIM,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    ActionMode,
    EpisodeContext,
    validate_action,
)
from a3_dual_arm_sim.core.paths import configure_hf_cache

A3_CAMERAS = (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)
SUPPORTED_POLICY_TYPES = frozenset(("smolvla", "pi05"))


def validate_checkpoint_contract(
    config: Mapping[str, Any],
    *,
    feature_map: Mapping[str, str] | None = None,
    expected_policy_type: str | None = None,
) -> dict[str, str]:
    """Validate before weights load; map checkpoint input keys to A3 input keys.

    Example: {"observation.images.camera1": "observation.images.front"}.
    This mapping never changes joint ordering, units, or action semantics.
    """
    policy_type = config.get("type")
    if policy_type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(f"Unsupported LeRobot policy type {policy_type!r}; use smolvla or pi05")
    if expected_policy_type is not None and policy_type != expected_policy_type:
        raise ValueError(f"Expected {expected_policy_type} checkpoint, found {policy_type!r}")
    relative_keys = ("use_relative_actions", "adapt_to_pi_aloha", "use_delta_joint_actions_aloha")
    if any(config.get(key, False) for key in relative_keys):
        raise ValueError("A3 executor v1 requires absolute joint actions, not relative/ALOHA actions")
    if config.get("empty_cameras", 0):
        raise ValueError("A3 executor requires real camera features, not empty cameras")
    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    action = outputs.get("action", {})
    if tuple(action.get("shape", ())) != (JOINT_ACTION_DIM,) or action.get("type") != "ACTION":
        raise ValueError("Checkpoint must declare A3 action shape (16,), not a generic/7D head")
    if set(outputs) != {"action"}:
        raise ValueError("A3 executor supports only the joint-position action output")
    mapping = dict(feature_map or {})
    unknown = set(mapping) - set(inputs)
    if unknown:
        raise ValueError(f"Feature mapping contains unknown checkpoint keys: {unknown}")
    resolved: dict[str, str] = {}
    states = cameras = 0
    for key, feature in inputs.items():
        source = mapping.get(key, key)
        kind = feature.get("type")
        shape = tuple(feature.get("shape", ()))
        if kind == "STATE":
            if source != STATE or shape != (JOINT_ACTION_DIM,):
                raise ValueError(f"State {key!r} must map to A3 observation.state shape (16,)")
            states += 1
        elif kind == "VISUAL":
            if source not in A3_CAMERAS:
                raise ValueError(f"Camera {key!r} needs an explicit mapping to an A3 camera")
            if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) < 1:
                raise ValueError(f"Camera {key!r} must declare a valid (3, H, W) RGB shape")
            cameras += 1
        else:
            raise ValueError(f"Unsupported checkpoint observation feature {key!r} ({kind!r})")
        resolved[key] = source
    if states != 1:
        raise ValueError("Checkpoint must include exactly one 16D A3 state feature")
    if not cameras:
        raise ValueError("VLA checkpoint must include at least one real A3 camera feature")
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Do not map multiple checkpoint inputs to the same A3 observation")
    return resolved


def _load_local_contract(
    checkpoint: Path,
    dataset_root: Path,
    *,
    feature_map: Mapping[str, str] | None,
    expected_policy_type: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot checkpoint config.json: {checkpoint}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    mapping = validate_checkpoint_contract(
        raw_config, feature_map=feature_map, expected_policy_type=expected_policy_type,
    )
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(f"Missing saved {filename}; export checkpoint with its processors")
    for filename in ("info.json", "stats.json", "tasks.parquet"):
        if not (dataset_root / "meta" / filename).is_file():
            raise FileNotFoundError(f"Missing local LeRobot v3 metadata: meta/{filename}")
    if not any((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        raise FileNotFoundError("Missing local LeRobot v3 episode metadata")
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("robot_type") != "A3_dual_arm":
        raise ValueError("Dataset metadata must identify robot_type=A3_dual_arm")
    features = info.get("features", {})
    for key in (STATE, "action"):
        if tuple(features.get(key, {}).get("shape", ())) != (JOINT_ACTION_DIM,):
            raise ValueError(f"A3 dataset {key!r} must have shape (16,)")
    for policy_key, source_key in mapping.items():
        feature = features.get(source_key, {})
        if source_key in A3_CAMERAS:
            shape = tuple(feature.get("shape", ()))
            if feature.get("dtype") not in ("image", "video") or len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"Dataset lacks RGB camera mapped to {policy_key!r}: {source_key}")
    return raw_config, mapping


class LeRobotPolicyPlugin:
    """Use saved LeRobot processors and metadata, never open recorded videos.

    Foundation checkpoints for other robots are not compatible merely because
    their padded maximum action dimension is large enough for A3.
    """

    action_mode: ActionMode = "joint_position"
    requires_camera_rendering = True

    def __init__(
        self,
        checkpoint: Path,
        dataset_root: Path,
        repo_id: str,
        device: str,
        *,
        feature_map: Mapping[str, str] | None = None,
        policy_type: str | None = None,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        configure_hf_cache()
        checkpoint = Path(checkpoint).expanduser().resolve()
        dataset_root = Path(dataset_root).expanduser().resolve()
        raw_config, mapping = _load_local_contract(
            checkpoint, dataset_root, feature_map=feature_map, expected_policy_type=policy_type,
        )
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            try:
                from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
            except ImportError:  # LeRobot 0.4 export location.
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:
            extra = "train-pi" if raw_config["type"] == "pi05" else "train"
            raise RuntimeError(f"Install LeRobot support: pip install -e '.[{extra}]'") from exc
        self.policy_type = str(raw_config["type"])
        self._feature_map = mapping
        self._input_keys = tuple(mapping)
        self._torch = torch
        self._prepare_observation = prepare_observation_for_inference
        self._device = torch.device(device)
        # Importing factory above registers concrete config classes.
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        overrides = dict(config_overrides or {})
        allowed = {"vlm_model_name", "n_action_steps", "num_steps", "num_inference_steps"}
        if set(overrides) - allowed:
            raise ValueError(f"Unsafe/unsupported config overrides: {set(overrides) - allowed}")
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"{self.policy_type} has no config option {key!r}")
            setattr(config, key, value)
        if "n_action_steps" in overrides and not 1 <= config.n_action_steps <= config.chunk_size:
            raise ValueError("n_action_steps must be positive and no larger than checkpoint chunk_size")
        config.pretrained_path = checkpoint
        config.device = device
        config.use_amp = self._device.type == "cuda"
        metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
        # Factory overwrites output_features from metadata: validate saved head
        # BEFORE calling it, or a 7D checkpoint could appear to have 16D output.
        rename_map = {source: key for key, source in mapping.items() if key != source}
        self._policy = make_policy(config, ds_meta=metadata, rename_map=rename_map or None)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._policy.eval()

    def reset(self, context: EpisodeContext) -> None:
        if context.action_mode != self.action_mode:
            raise ValueError("LeRobot executor requires joint_position action mode")
        self._torch.manual_seed(context.seed)
        self._policy.reset()
        # Processors can retain history independently of the model action queue.
        for name in ("_preprocessor", "_postprocessor"):
            reset = getattr(getattr(self, name, None), "reset", None)
            if callable(reset):
                reset()

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("LeRobot policy has been closed")
        policy_observation = {}
        for key, source in self._feature_map.items():
            if source not in observation:
                raise ValueError(f"Missing observation {source!r} (checkpoint input {key!r})")
            value = np.asarray(observation[source])
            if source in A3_CAMERAS:
                if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise ValueError(f"{source} must be uint8 HxWx3 RGB")
            elif value.shape != (JOINT_ACTION_DIM,) or not np.all(np.isfinite(value)):
                raise ValueError("A3 policy state must be a finite shape (16,) vector")
            policy_observation[key] = observation[source]
        batch = self._prepare_observation(policy_observation, self._device, task, "A3_dual_arm")
        batch = self._preprocessor(batch)
        with self._torch.inference_mode():
            action = self._postprocessor(self._policy.select_action(batch))
        value = action.detach().float().cpu().numpy()
        if value.shape == (1, JOINT_ACTION_DIM):
            value = value[0]
        return validate_action(value, self.action_mode)

    def close(self) -> None:
        self._policy = None


def make_policy() -> LeRobotPolicyPlugin:
    """Factory configured through A3_LEROBOT_* environment variables."""
    checkpoint = os.environ.get("A3_LEROBOT_CHECKPOINT")
    dataset_root = os.environ.get("A3_LEROBOT_DATASET_ROOT")
    if not checkpoint or not dataset_root:
        raise RuntimeError("Set A3_LEROBOT_CHECKPOINT and A3_LEROBOT_DATASET_ROOT")
    return LeRobotPolicyPlugin(
        Path(checkpoint), Path(dataset_root),
        os.environ.get("A3_LEROBOT_REPO_ID", "local/a3-single-box-skills"),
        os.environ.get("A3_LEROBOT_DEVICE", "cuda"),
        feature_map=json.loads(os.environ.get("A3_LEROBOT_FEATURE_MAP", "{}")),
        policy_type=os.environ.get("A3_LEROBOT_POLICY_TYPE"),
    )
