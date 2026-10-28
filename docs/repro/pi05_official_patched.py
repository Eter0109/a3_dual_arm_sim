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



def _skip_prefixed_state_proj() -> None:
    """Teach lerobot's pi05 key converter to skip `model.state_proj.*` as well.

    The converter already skips the unprefixed form with a warning; ours arrived prefixed
    because A3's training launcher wraps the projector under the model, so the check misses
    it and a strict load fails on two keys the stock architecture cannot hold.  Idempotent.
    """

    from lerobot.policies.pi05 import modeling_pi05 as module

    original = module.PI05Policy._fix_pytorch_state_dict_keys
    if getattr(original, "_a3_skips_state_proj", False):
        return

    def patched(self, state_dict, model_config):
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not (key.startswith("state_proj.") or key.startswith("model.state_proj."))
        }
        return original(self, filtered, model_config)

    patched._a3_skips_state_proj = True
    module.PI05Policy._fix_pytorch_state_dict_keys = patched


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
        # A3 ADAPTATION -- the only change to main's loader, and it is main's own logic
        # re-placed rather than re-invented.
        #
        # main drops `model.state_proj.{weight,bias}` as obsolete before a strict load.
        # This machine's lerobot 0.5.1 wheel has no `_convert_openpi_state_dict` (that is
        # a different build of the same version number), and its own converter skips
        # `state_proj.*` **without the `model.` prefix** -- our launcher saved the tensors
        # prefixed, so they survive, `strict=True` raises, and `from_pretrained` catches the
        # exception and **returns the model with randomly initialised weights**.  The run
        # then scores 0/10 and never reports an error.  main's comment in this file already
        # names that failure mode; this is the concrete way to reach it.
        #
        # Teaching the converter to skip the prefixed form makes `strict=True` succeed, so
        # the load is verified by lerobot itself ("All keys loaded successfully!") instead
        # of being asserted by hand.
        _skip_prefixed_state_proj()
        self._policy = PI05Policy.from_pretrained(checkpoint, config=config)
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
        prepared = self._prepare(raw, self._torch.device("cpu"), task, "A3_dual_arm")
        if os.environ.get("A3_PI05_DEBUG"):
            # What actually reaches the model, as opposed to what the harness logged.
            import numpy as _np
            for _k, _v in sorted(prepared.items()):
                if hasattr(_v, "shape"):
                    _a = _v.detach().float().cpu().numpy()
                    print(f"[dbg] in  {_k:52s} {tuple(_a.shape)} "
                          f"min {_a.min():.4f} max {_a.max():.4f} mean {_a.mean():.4f}", flush=True)
            _sm = _np.asarray(raw["observation.state"], dtype=_np.float32)
            print(f"[dbg] task={task!r}", flush=True)
            print(f"[dbg] raw state min {_sm.min():.3f} max {_sm.max():.3f}", flush=True)
        batch = self._preprocessor(prepared)
        if os.environ.get("A3_PI05_DEBUG") and not getattr(self, "_dbg_keys", False):
            import numpy as _np
            print(f"[dbg] AFTER preprocessor, keys:", flush=True)
            for _k, _v in sorted(batch.items()):
                if hasattr(_v, "shape"):
                    _a = _v.detach().float().cpu().numpy()
                    _extra = ""
                    if _a.size <= 40:
                        _extra = f" values {_np.round(_a.reshape(-1)[:40], 3).tolist()}"
                    print(f"[dbg]   {_k:52s} {tuple(_a.shape)} min {_a.min():.4f} "
                          f"max {_a.max():.4f}{_extra}", flush=True)
                else:
                    print(f"[dbg]   {_k:52s} = {_v!r}"[:300], flush=True)
            self._dbg_keys = True
        with self._torch.inference_mode():
            self._torch.manual_seed(self.inference_seed)
            chunk = self._postprocessor(self._policy.predict_action_chunk(batch))[0]
        actions = chunk.detach().float().cpu().numpy()
        if os.environ.get("A3_PI05_DEBUG") and not getattr(self, "_dbg_done", False):
            import numpy as _np
            _q = _np.asarray(observation["observation.state"], dtype=_np.float32)
            print(f"[dbg] chunk {actions.shape} min {actions.min():.3f} max {actions.max():.3f} "
                  f"mean {actions.mean():.3f} | first row vs state max|diff| "
                  f"{_np.abs(actions[0, :16] - _q).max():.3f}", flush=True)
            self._dbg_done = True
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
