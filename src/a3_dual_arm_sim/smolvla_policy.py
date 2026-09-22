from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ActionMode, EpisodeContext
from .paths import project_root

_RUNTIME = project_root() / ".runtime"
os.environ.setdefault("HF_HOME", str(_RUNTIME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_RUNTIME / "datasets"))


class SmolVLAPolicyPlugin:
    """LeRobot SmolVLA checkpoint adapter for the common A3 policy protocol."""

    action_mode: ActionMode = "joint_position"

    def __init__(self, checkpoint: Path, dataset_root: Path, repo_id: str, device: str) -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:
            raise RuntimeError("Install SmolVLA support with: pip install -e '.[train]'") from exc

        checkpoint = checkpoint.expanduser().resolve()
        dataset_root = dataset_root.expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Missing SmolVLA config.json: {checkpoint}")
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
        self._input_keys = tuple(config.input_features)
        self._policy = make_policy(config, ds_meta=dataset.meta)
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._policy.eval()

    def reset(self, context: EpisodeContext) -> None:
        self._policy.reset()

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        policy_observation = {key: observation[key] for key in self._input_keys}
        batch = self._prepare_observation(
            policy_observation, self._device, task, "A3_dual_arm"
        )
        batch = self._preprocessor(batch)
        with self._torch.inference_mode():
            action = self._policy.select_action(batch)
            action = self._postprocessor(action)
        return action.detach().float().cpu().numpy().reshape(16)

    def close(self) -> None:
        self._policy = None


def make_policy() -> SmolVLAPolicyPlugin:
    """Factory configured through environment variables for ``a3-sim run``."""
    checkpoint = os.environ.get("A3_SMOLVLA_CHECKPOINT")
    dataset_root = os.environ.get("A3_SMOLVLA_DATASET_ROOT")
    if not checkpoint or not dataset_root:
        raise RuntimeError(
            "Set A3_SMOLVLA_CHECKPOINT and A3_SMOLVLA_DATASET_ROOT before loading this policy"
        )
    return SmolVLAPolicyPlugin(
        Path(checkpoint),
        Path(dataset_root),
        os.environ.get("A3_SMOLVLA_REPO_ID", "local/a3-grasp"),
        os.environ.get("A3_SMOLVLA_DEVICE", "cuda"),
    )
