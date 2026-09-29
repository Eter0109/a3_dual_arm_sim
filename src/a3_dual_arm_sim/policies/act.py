from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.config import CookieSceneConfig
from a3_dual_arm_sim.contracts import JOINT_ACTION_DIM, ActionMode, EpisodeContext
from a3_dual_arm_sim.paths import project_root

_RUNTIME = project_root() / ".runtime"
os.environ.setdefault("HF_HOME", str(_RUNTIME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_RUNTIME / "datasets"))

LEFT_ARM_ACTION_DIM = 8


def left_arm_hold_action() -> np.ndarray:
    """Right-arm command paired with a left-arm-only policy: the deployment hold pose.

    Left-arm-only datasets never move the right arm, and their recorded right-arm
    columns are exactly this constant, so replaying it reproduces the training
    distribution without the policy having to predict it.
    """
    return np.asarray(
        CookieSceneConfig().deployment_home[LEFT_ARM_ACTION_DIM:], dtype=np.float64
    )


def expand_left_arm_action(action: np.ndarray, hold: np.ndarray | None = None) -> np.ndarray:
    """Widen an 8D left-arm action to the environment's 16D joint action."""
    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.size == JOINT_ACTION_DIM:
        return value
    if value.size != LEFT_ARM_ACTION_DIM:
        raise ValueError(
            f"Expected an {LEFT_ARM_ACTION_DIM}D left-arm or {JOINT_ACTION_DIM}D bimanual action, "
            f"got {value.size}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("action contains NaN or infinity")
    tail = (
        left_arm_hold_action()
        if hold is None
        else np.asarray(hold, dtype=np.float64).reshape(-1)
    )
    if tail.size != JOINT_ACTION_DIM - LEFT_ARM_ACTION_DIM:
        raise ValueError(
            f"Right-arm hold action must have {JOINT_ACTION_DIM - LEFT_ARM_ACTION_DIM} values"
        )
    return np.concatenate([value, tail])


def _policy_feature_dim(features: Any, key: str) -> int | None:
    """Leading dimension of a LeRobot ``PolicyFeature`` entry, if it is a vector."""
    feature = features.get(key) if hasattr(features, "get") else None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not shape:
        return None
    return int(shape[0])


def project_observation_value(value: Any, expected_dim: int | None) -> Any:
    """Trim a bimanual observation to the leading joint dims a policy expects.

    The environment always reports 16D proprioception; a left-arm-only checkpoint
    consumes only the first 8 entries (``L_q1..L_q7, L_gripper``).
    """
    if expected_dim is None:
        return value
    array = np.asarray(value)
    if array.ndim == 1 and array.shape[0] > expected_dim:
        return array[:expected_dim]
    return value


class ACTPolicyPlugin:
    """LeRobot ACT checkpoint adapter for the common A3 policy protocol."""

    action_mode: ActionMode = "joint_position"

    def __init__(
        self,
        checkpoint: Path,
        dataset_root: Path,
        repo_id: str,
        device: str = "cuda",
        temporal_ensemble_coeff: float | None = 0.01,
        hold_action: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:
            raise RuntimeError("Install LeRobot support with: pip install -e '.[train]'") from exc

        checkpoint = checkpoint.expanduser().resolve()
        dataset_root = dataset_root.expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing ACT config.json: {checkpoint}")

        self._torch = torch
        self._prepare_observation = prepare_observation_for_inference
        self._device = torch.device(device)

        try:
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                download_videos=False,
                return_uint8=True,
            )
        except TypeError:
            dataset = LeRobotDataset(
                repo_id,
                root=dataset_root,
                download_videos=False,
            )

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.device = device
        config.use_amp = device == "cuda"
        if temporal_ensemble_coeff is not None and hasattr(config, "temporal_ensemble_coeff"):
            config.temporal_ensemble_coeff = temporal_ensemble_coeff

        self._input_keys = tuple(config.input_features)
        self._input_dims = {
            key: _policy_feature_dim(config.input_features, key) for key in self._input_keys
        }
        self._action_dim = _policy_feature_dim(config.output_features, "action") or JOINT_ACTION_DIM
        self._hold_action = (
            None
            if self._action_dim == JOINT_ACTION_DIM
            else (
                left_arm_hold_action()
                if hold_action is None
                else np.asarray(hold_action, dtype=np.float64).reshape(-1)
            )
        )
        self._policy = make_policy(config, ds_meta=dataset.meta)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._policy.eval()

    def reset(self, context: EpisodeContext | None = None) -> None:
        self._policy.reset()

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        policy_observation = {
            key: project_observation_value(observation[key], self._input_dims.get(key))
            for key in self._input_keys
        }
        batch = self._prepare_observation(
            policy_observation, self._device, task, "A3_dual_arm"
        )
        batch = self._preprocessor(batch)
        with self._torch.inference_mode():
            action = self._policy.select_action(batch)
            action = self._postprocessor(action)
        action = action.detach().float().cpu().numpy().reshape(-1)
        return expand_left_arm_action(action, self._hold_action)

    def close(self) -> None:
        self._policy = None


def make_policy() -> ACTPolicyPlugin:
    """Factory configured through environment variables for ``a3-sim run``."""
    checkpoint = os.environ.get("A3_ACT_CHECKPOINT")
    dataset_root = os.environ.get("A3_ACT_DATASET_ROOT")
    if not checkpoint or not dataset_root:
        raise RuntimeError(
            "Set A3_ACT_CHECKPOINT and A3_ACT_DATASET_ROOT before loading this policy"
        )
    return ACTPolicyPlugin(
        Path(checkpoint),
        Path(dataset_root),
        os.environ.get("A3_ACT_REPO_ID", "local/a3-front-close-left-100"),
        os.environ.get("A3_ACT_DEVICE", "cuda"),
    )
