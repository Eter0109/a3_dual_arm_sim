"""Inspection helpers shared by policy loading and training.

Kept separate because both the inference adapter and the training entry point
have to answer the same question about a checkpoint before they can use it.
"""

from __future__ import annotations

from pathlib import Path

#: Prefix that identifies the vision-language tower inside a SmolVLA checkpoint.
_VLM_PREFIX = "model.vlm_with_expert.vlm"


def checkpoint_embeds_vlm(checkpoint: Path) -> bool:
    """Whether a checkpoint's weights already contain the vision-language model.

    ``lerobot/smolvla_base`` ships the full 450M parameters yet still sets
    ``load_vlm_weights=True`` in its config. Left alone, LeRobot fetches
    SmolVLM2's separate 2 GB of weights and would overwrite them anyway, or fail
    outright on a host without Hub access. Reading the safetensors header settles
    it without materialising any tensor data.
    """

    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        return False
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - safetensors ships with lerobot
        return False
    with safe_open(weights, framework="pt") as handle:
        # The handle exposes keys() but is not iterable, so materialise them.
        keys = list(handle.keys())
    return any(key.startswith(_VLM_PREFIX) for key in keys)
