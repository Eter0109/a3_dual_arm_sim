"""Inference adapter for the A3 Cookie π₀.₅ checkpoint."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.contracts import EpisodeContext


class PI05PolicyPlugin:
    action_mode = "joint_position"

    def __init__(self, checkpoint: Path, device: str = "cuda", inference_seed: int = 0,
                 n_action_steps: int | None = None,
                 num_inference_steps: int | None = None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.utils import prepare_observation_for_inference
        from safetensors.torch import load_file

        from a3_dual_arm_sim.learning.checkpoint_bundle import load_processors

        checkpoint = checkpoint.expanduser().resolve()
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.type != "pi05":
            raise ValueError(f"Expected π₀.₅ checkpoint, got {config.type!r}")
        if config.input_features["observation.state"].shape != (32,):
            raise ValueError("Expected a checkpoint with 32-D state capacity")
        if config.output_features["action"].shape != (16,):
            raise ValueError("Expected 16-D dual-arm action")
        n_action_steps = 16 if n_action_steps is None else n_action_steps
        if not 1 <= n_action_steps <= config.chunk_size:
            raise ValueError(f"n_action_steps must be in [1, {config.chunk_size}]")
        config.n_action_steps = n_action_steps
        if num_inference_steps is not None:
            if num_inference_steps < 1:
                raise ValueError("num_inference_steps must be positive")
            config.num_inference_steps = num_inference_steps
        # Construct/load on CPU to avoid a second full GPU weight allocation.
        config.device = "cpu"
        config.gradient_checkpointing = False
        self._torch = torch
        self._device = torch.device(device)
        self._prepare = prepare_observation_for_inference
        self.inference_seed = inference_seed
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self._actions = deque()
        # The installed π₀.₅ loader can return randomly initialized weights on a
        # loading exception. Load strictly so an evaluation cannot proceed silently.
        self._policy = PI05Policy(config)
        weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        converted = self._policy._convert_openpi_state_dict(weights)
        # Older LeRobot checkpoints saved a state projection that π₀.₅ never
        # reads. The current converter skips unprefixed state_proj keys but
        # misses the same two tensors when they carry the model. prefix.
        obsolete = {"model.state_proj.weight", "model.state_proj.bias"}
        extra = set(converted) - set(self._policy.state_dict())
        if extra - obsolete:
            raise RuntimeError(f"Unexpected π₀.₅ checkpoint keys: {sorted(extra)}")
        for key in extra:
            del converted[key]
        self._policy.load_state_dict(converted, strict=True)
        del weights, converted
        # PI05 calls the language backbones directly, never either vocabulary
        # output head. Keep those unused heads on CPU so FP32 action inference
        # fits a 16-GB GPU. Loading remains strict, including their weights.
        backbone = self._policy.model.paligemma_with_expert
        parked_heads = []
        if self._device.type == "cuda":
            for owner in (backbone.paligemma, backbone.gemma_expert):
                parked_heads.append((owner, owner.lm_head))
                owner.lm_head = None
        try:
            self._policy.to(device=device, dtype=torch.float32).eval()
        finally:
            self._unused_heads = []
            for owner, head in parked_heads:
                # Do not register CPU parameters back onto the policy: LeRobot
                # derives policy.device from its first registered parameter.
                self._unused_heads.append(head.float().eval())
                owner.lm_head = torch.nn.Identity()
                owner.lm_head.register_forward_pre_hook(self._reject_unused_head)
        config.device = device
        config.dtype = torch.float32
        # Newer LeRobot also gates vision autocast on this field; .float() alone
        # would still run the vision matmuls in BF16.
        self._policy.model.paligemma_with_expert.precision = torch.float32
        self.inference_protocol = {
            "chunk_size": config.chunk_size,
            "execute": config.n_action_steps,
            "num_inference_steps": config.num_inference_steps,
            "seed": inference_seed,
            "seed_scope": "every_chunk",
            "dtype": "float32",
            "image_resize": "cv2.INTER_AREA uint8 224x224 before /255",
            "deterministic_algorithms": "warn_only",
            "unused_lm_heads_on_cpu": bool(parked_heads),
        }
        # This checkpoint contains a disabled legacy step removed from newer LeRobot.
        # Keep the downloaded files intact and omit only that no-op in a temporary view.
        self._processor_view = tempfile.TemporaryDirectory(prefix="a3_pi05_processors_")
        processor_dir = Path(self._processor_view.name)
        for name in (
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ):
            (processor_dir / name).symlink_to(checkpoint / name)
        pipeline = json.loads((checkpoint / "policy_preprocessor.json").read_text())
        pipeline["steps"] = [
            step for step in pipeline["steps"]
            if not (step["registry_name"] == "delta_actions_processor"
                    and step.get("config", {}).get("enabled") is False)
        ]
        # Prefer an already cached tokenizer so processor loading does not need
        # a Hub request (which can fail with an invalid user proxy setting).
        from huggingface_hub import snapshot_download

        for step in pipeline["steps"]:
            if step["registry_name"] == "tokenizer_processor":
                tokenizer_name = step["config"]["tokenizer_name"]
                try:
                    cached = snapshot_download(tokenizer_name, local_files_only=True)
                except Exception:
                    break  # Let the normal Hub loader fetch an uncached tokenizer.
                if (Path(cached) / "tokenizer_config.json").is_file():
                    step["config"]["tokenizer_name"] = cached
                break
        (processor_dir / "policy_preprocessor.json").unlink()
        (processor_dir / "policy_preprocessor.json").write_text(json.dumps(pipeline))
        self._preprocessor, self._postprocessor = load_processors(
            config, processor_dir, device=device
        )
        self.last_raw_action = None

    @staticmethod
    def _reject_unused_head(module, inputs):
        raise RuntimeError("PI05 action inference unexpectedly called a parked vocabulary head")

    def reset(self, context: EpisodeContext) -> None:
        self._torch.manual_seed(self.inference_seed)
        self._policy.reset()
        self._actions.clear()
        self.last_raw_action = None

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        if self._actions:
            result = self._actions.popleft().copy()
            self.last_raw_action = result.copy()
            return result
        import cv2

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("π₀.₅ A3 deployment requires finite 16-D environment state")
        raw = {
            key: cv2.resize(observation[key], (224, 224), interpolation=cv2.INTER_AREA)
            for key in (
                "observation.images.front",
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            )
        }
        # Saved quantile statistics are 16-D; the policy pads internally to its
        # configured 32-D maximum after normalization and state tokenization.
        raw["observation.state"] = state.copy()
        # The provider builds/scales tensors and normalizes state on CPU, then
        # its saved device processor transfers the completed batch to CUDA.
        batch = self._preprocessor(self._prepare(raw, self._torch.device("cpu"), task, "A3_dual_arm"))
        with self._torch.inference_mode():
            self._torch.manual_seed(self.inference_seed)
            chunk = self._postprocessor(self._policy.predict_action_chunk(batch))[0]
        actions = chunk.detach().float().cpu().numpy()
        if actions.shape != (self._policy.config.chunk_size, 16):
            raise ValueError(f"Unexpected π₀.₅ action chunk shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("π₀.₅ produced non-finite actions")
        self._actions.extend(actions[: self._policy.config.n_action_steps].copy())
        result = self._actions.popleft().copy()
        self.last_raw_action = result.copy()
        return result

    def close(self) -> None:
        self._actions.clear()
        self._unused_heads = []
        self._policy = None
        self._processor_view.cleanup()
