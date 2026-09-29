"""Lazy visual-language backends shared by planners and verifiers.

Transport and model loading know nothing about skills, simulation or datasets.
Credentials are read from the environment and never included in exceptions.
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib import error, parse, request

import numpy as np
from PIL import Image

from a3_dual_arm_sim.core.paths import project_root


def _project_checkpoint_path(identifier: str) -> str:
    """Anchor bundled checkpoint conventions, not arbitrary Hugging Face IDs."""
    path = Path(identifier)
    if not path.is_absolute() and path.parts and path.parts[0] in ("models", "outputs"):
        return str(project_root() / path)
    return identifier


class AgenticModelError(RuntimeError):
    """A model or transport failed; callers must stop rather than invent output."""


class ModelOutputError(AgenticModelError):
    """The response did not satisfy the restricted semantic interface."""


@dataclass(frozen=True)
class ChatBackendConfig:
    kind: Literal["http", "local_qwen"]
    model: str
    base_url: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 60.0
    max_tokens: int = 512
    image_max_edge: int = 512
    jpeg_quality: int = 85
    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = True
    fine_tuned: bool = False
    model_family: Literal["qwen2_5_vl", "qwen3_5"] = "qwen2_5_vl"
    adapter_path: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("http", "local_qwen"):
            raise ValueError("kind must be http or local_qwen")
        if not self.model.strip():
            raise ValueError("a model identifier or local checkpoint is required")
        if not 0 < self.timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be in (0, 600]")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be in 1..4096")
        if not 32 <= self.image_max_edge <= 2048:
            raise ValueError("image_max_edge must be in 32..2048")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in 1..95")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("unsupported local model dtype")
        if self.model_family not in ("qwen2_5_vl", "qwen3_5"):
            raise ValueError("model_family must be qwen2_5_vl or qwen3_5")
        if self.adapter_path and self.kind != "local_qwen":
            raise ValueError("adapter_path requires the local Qwen backend")
        if self.kind == "http":
            parsed = parse.urlsplit(self.base_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("base_url must be an HTTP(S) API base URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("base_url must not contain credentials, query, or fragment")


class ChatBackend(Protocol):
    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> str: ...

    def close(self) -> None: ...


def _rgb_image(array: np.ndarray, max_edge: int) -> Image.Image:
    image = np.asarray(array)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera images must be uint8 RGB arrays shaped (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("camera images must not be empty")
    result = Image.fromarray(image)
    result.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return result


class OpenAICompatibleChatBackend:
    """Small multimodal chat client; no SDK, implicit retries, or key logging."""

    def __init__(self, config: ChatBackendConfig) -> None:
        if config.kind != "http":
            raise ValueError("HTTP backend requires kind=http")
        self.config = config
        # Preserve any provider-specific prefix rather than replacing its path.
        self.endpoint = config.base_url.rstrip("/") + "/chat/completions"

    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> str:
        api_key = os.environ.get(self.config.api_key_env) if self.config.api_key_env else None
        if self.config.api_key_env and not api_key:
            raise AgenticModelError("required API key environment variable is not set")
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            buffer = io.BytesIO()
            _rgb_image(image, self.config.image_max_edge).save(
                buffer, format="JPEG", quality=self.config.jpeg_quality
            )
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + encoded},
            })
        payload = json.dumps({
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        outbound = request.Request(self.endpoint, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(outbound, timeout=self.config.timeout_seconds) as response:
                raw = response.read(1024 * 1024 + 1)
        except error.HTTPError as exc:
            raise AgenticModelError(f"chat service returned HTTP {exc.code}") from None
        except (error.URLError, TimeoutError, OSError):
            raise AgenticModelError("chat service connection failed or timed out") from None
        if len(raw) > 1024 * 1024:
            raise AgenticModelError("chat service response exceeded the size limit")
        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            output = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise ModelOutputError("chat response was truncated by its token limit")
            if not isinstance(output, str) or not output.strip():
                raise ModelOutputError("chat service returned no text response")
            return output.strip()
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelOutputError("chat service returned an invalid response envelope") from None

    def close(self) -> None:
        pass


class LocalQwenVLBackend:
    """Lazy Qwen visual inference with an explicit, checkpoint-matched family."""

    def __init__(self, config: ChatBackendConfig) -> None:
        if config.kind != "local_qwen":
            raise ValueError("local Qwen backend requires kind=local_qwen")
        self.config = config
        self._model = None
        self._processor = None
        self.adapter_loaded = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor

            if self.config.model_family == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration

                model_class = Qwen3_5ForConditionalGeneration
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration

                model_class = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise AgenticModelError(
                "local Qwen requires torch, accelerate, and transformers supporting "
                + self.config.model_family
            ) from None
        dtype = "auto" if self.config.dtype == "auto" else getattr(torch, self.config.dtype)
        try:
            model_checkpoint = _project_checkpoint_path(self.config.model)
            model = model_class.from_pretrained(
                model_checkpoint,
                torch_dtype=dtype,
                device_map=self.config.device,
                local_files_only=self.config.local_files_only,
            )
            processor = AutoProcessor.from_pretrained(
                model_checkpoint, local_files_only=self.config.local_files_only
            )
            if self.config.adapter_path:
                from peft import PeftModel

                adapter = Path(_project_checkpoint_path(self.config.adapter_path)).resolve(strict=True)
                if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
                    raise ValueError("adapter_path must contain a saved PEFT adapter")
                manifest_path = adapter / "verifier_training_manifest.json"
                if not manifest_path.is_file():
                    raise ValueError("verifier adapter requires its training manifest")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("model_family") != self.config.model_family:
                    raise ValueError("verifier adapter model family does not match the base")
                base_model = manifest.get("base_model")
                if not isinstance(base_model, str) or (
                    Path(_project_checkpoint_path(base_model)).resolve()
                    != Path(model_checkpoint).resolve()
                ):
                    raise ValueError("verifier adapter base checkpoint does not match")
                model = PeftModel.from_pretrained(
                    model, str(adapter), is_trainable=False, local_files_only=True,
                )
                self.adapter_loaded = True
            model.eval()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            raise AgenticModelError(
                "could not load the local Qwen checkpoint; check files and model dependencies"
            ) from None
        self._model, self._processor = model, processor

    def complete(self, prompt: str, images: Sequence[np.ndarray]) -> str:
        pil_images = [_rgb_image(image, self.config.image_max_edge) for image in images]
        self._load()
        import torch

        content = [{"type": "image", "image": image} for image in pil_images]
        content.append({"type": "text", "text": prompt})
        template_options = {"tokenize": False, "add_generation_prompt": True}
        if self.config.model_family == "qwen3_5":
            # Qwen3.5 thinks by default. Its official template's hard switch
            # prevents reasoning text from contaminating JSON/Yes/No contracts.
            template_options["enable_thinking"] = False
        try:
            text = self._processor.apply_chat_template(
                [{"role": "user", "content": content}],
                **template_options,
            )
            inputs = self._processor(
                text=[text], images=pil_images, padding=True, return_tensors="pt"
            ).to(self._model.device)
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, max_new_tokens=self.config.max_tokens, do_sample=False,
                )
            input_length = inputs["input_ids"].shape[1]
            output = self._processor.batch_decode(
                generated[:, input_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            raise AgenticModelError("local Qwen inference failed") from None
        if not output:
            raise ModelOutputError("local Qwen returned an empty response")
        return output

    def close(self) -> None:
        self._model = None
        self._processor = None
        self.adapter_loaded = False


def make_chat_backend(config: ChatBackendConfig) -> ChatBackend:
    if config.kind == "http":
        return OpenAICompatibleChatBackend(config)
    return LocalQwenVLBackend(config)


def camera_view(images: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    # Full observation keys and compact frame-buffer keys are both supported.
    for key in (name, "observation.images." + name):
        if key in images:
            result = np.asarray(images[key])
            _rgb_image(result, 512)  # Check the contract before any model request.
            return result
    raise ValueError(f"missing {name} camera image")
