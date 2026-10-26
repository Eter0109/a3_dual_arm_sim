"""ACT inference adapter: checkpoint processors, queued actions, and right-arm hold."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.contracts import ActionMode, EpisodeContext
from a3_dual_arm_sim.learning.act_training import validate_act_horizon


class ACTPolicyPlugin:
    action_mode: ActionMode = "joint_position"

    def __init__(
        self,
        checkpoint: Path,
        dataset_root: Path,
        repo_id: str,
        device: str,
        *,
        n_action_steps: int | None = None,
        inference_seed: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.utils import prepare_observation_for_inference

        from a3_dual_arm_sim.learning.checkpoint_bundle import load_processors, processor_files
        from a3_dual_arm_sim.workflows.policy_adapters import resolve_policy_checkpoint

        checkpoint = resolve_policy_checkpoint(checkpoint)
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.type != "act":
            raise ValueError(f"Expected ACT checkpoint, got {config.type!r}")
        self._action_dim = config.output_features["action"].shape[0]
        self._state_dim = config.input_features["observation.state"].shape[0]
        if (self._action_dim, self._state_dim) not in ((8, 8), (8, 11), (16, 16)):
            raise ValueError("Expected 8/8 or 11/8 left-arm, or 16/16 dual-arm state/action")
        if n_action_steps is not None:
            config.n_action_steps = n_action_steps
        if temporal_ensemble_coeff is not None:
            config.temporal_ensemble_coeff = temporal_ensemble_coeff
        validate_act_horizon(
            config.chunk_size, config.n_action_steps, config.temporal_ensemble_coeff
        )
        config.device = device
        # All backbone weights are in the trained checkpoint; no ImageNet download at deployment.
        config.pretrained_backbone_weights = None
        self._torch = torch
        self._device = torch.device(device)
        self._prepare_observation = prepare_observation_for_inference
        self._input_keys = tuple(config.input_features)
        self.inference_seed = inference_seed
        try:
            processor_files(checkpoint)
        except FileNotFoundError:
            # Legacy checkpoint without saved processors: fall back to the training
            # dataset statistics, with the action-dimension cross-check.
            metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root.expanduser().resolve())
            if list(metadata.features["action"]["shape"]) != [self._action_dim]:
                raise ValueError("Dataset action dimension differs from ACT checkpoint")
        self._policy = (
            ACTPolicy.from_pretrained(checkpoint, config=config, strict=True).to(device).eval()
        )
        self._preprocessor, self._postprocessor = load_processors(
            config, checkpoint, dataset_root, device
        )
        self._right_hold = None
        self.last_raw_action = None

    def reset(self, context: EpisodeContext) -> None:
        if self.inference_seed is not None:
            self._torch.manual_seed(self.inference_seed)
        self._policy.reset()
        self._right_hold = None
        self.last_raw_action = None

    def predict_action(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        obs = {key: observation[key] for key in self._input_keys}
        state = np.asarray(obs["observation.state"])
        if self._state_dim == 11:
            if state.shape == (16,):
                pose = np.asarray(observation.get("observation.eef_pose"))
                if pose.shape != (14,):
                    raise ValueError("11-D inference requires the environment's 14-D EEF pose")
                state = np.concatenate((state[:8], pose[:3]))
            if state.shape != (11,) or not np.isfinite(state).all():
                raise ValueError("11-D inference requires left joints, gripper and world EEF xyz")
        elif self._action_dim == 8:
            if state.shape not in ((8,), (16,)):
                raise ValueError("Left-only inference requires 8-D or 16-D state")
            state = state[:8]
        elif state.shape != (16,):
            raise ValueError("Dual-arm inference requires 16-D state")
        obs["observation.state"] = state.copy()
        batch = self._preprocessor(
            self._prepare_observation(obs, self._device, task, "A3_dual_arm")
        )
        with self._torch.inference_mode():
            action = self._postprocessor(self._policy.select_action(batch))
        result = action.detach().float().cpu().numpy().reshape(self._action_dim)
        if not np.isfinite(result).all():
            raise ValueError("ACT produced non-finite actions")
        return result

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        if self._action_dim == 8:
            state = np.asarray(observation["observation.state"])
            if state.shape != (16,):
                raise ValueError("Left-only deployment requires the environment's 16-D state")
            if self._right_hold is None:
                self._right_hold = state[8:].copy()
        action = self.predict_action(observation, task)
        if self._action_dim == 8:
            action = np.concatenate((action, self._right_hold))
        self.last_raw_action = action.copy()
        return action

    def close(self) -> None:
        self._policy = None


def make_policy() -> ACTPolicyPlugin:
    """Factory for a3-sim run, configured via A3_ACT_* environment variables."""
    checkpoint = os.environ.get("A3_ACT_CHECKPOINT")
    dataset = os.environ.get("A3_ACT_DATASET_ROOT")
    if not checkpoint or not dataset:
        raise RuntimeError("Set A3_ACT_CHECKPOINT and A3_ACT_DATASET_ROOT")
    return ACTPolicyPlugin(
        Path(checkpoint),
        Path(dataset),
        os.environ.get("A3_ACT_REPO_ID", "Eter0109/a3-front-close-left-100"),
        os.environ.get("A3_ACT_DEVICE", "cpu"),
    )
