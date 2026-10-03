"""Launch lerobot's trainer on this machine, with the shims the π0.5 checkpoint needs.

Why a launcher rather than `python -m lerobot.scripts.lerobot_train`
-------------------------------------------------------------------
``lerobot/pi05_base`` was saved by an older lerobot than the one installed here, and the
difference is not cosmetic: `policy_preprocessor.json` lists a step called
``relative_actions_processor``, while this version registers the same class under other
names.  ``PolicyProcessorPipeline.from_pretrained`` resolves step names by a plain dict
lookup and raises ``ImportError: Failed to load processor step from registry`` when it
misses -- an error that reads like a missing dependency, not a version skew.

Three ways to fix it, and why this is the one
--------------------------------------------
* Edit ``policy_preprocessor.json`` -- rejected: that is modifying a downloaded checkpoint,
  and the next `snapshot_download` would silently undo it.
* Patch the installed lerobot -- rejected: a venv-local edit that leaves no trace in this
  repository, so the next person to build an environment would hit the same wall with
  nothing to read.
* Register the alias here, before handing over -- chosen.  It is three lines, it lives
  next to the reason for it, and it is idempotent.

``setdefault`` rather than assignment, so if a future lerobot registers the old name
itself this becomes a no-op instead of an error.

The alias is safe because the step is ``enabled: false`` in this checkpoint: the class is
instantiated and then does nothing, so the *name* is the only thing that had to match.

Usage
-----
The same arguments `lerobot-train` takes::

    python scripts/train_pi05_launcher.py --policy.path=lerobot/pi05_base ...
"""

from __future__ import annotations

import os
import sys


def _register_legacy_step_names() -> None:
    """Teach this lerobot the step name the checkpoint was saved with."""

    from lerobot.processor import RelativeActionsProcessorStep
    from lerobot.processor.pipeline import ProcessorStepRegistry

    ProcessorStepRegistry._registry.setdefault(
        "relative_actions_processor", RelativeActionsProcessorStep
    )


#: Which parts of PaliGemma to un-freeze again, overridable from the environment.
#:
#: Why a partial un-freeze rather than the documented full fine-tune
#: -----------------------------------------------------------------
#: `train_expert_only=true` freezes all 3.45 B of PaliGemma and trains the 0.70 B action
#: expert.  The published LIBERO recipe trains everything -- but on 8xH100.  On this 32 GB
#: card the arithmetic is unforgiving, and it is worth writing down because "it did not fit"
#: is otherwise indistinguishable from "it was not tried":
#:
#:   resident bf16 weights, all 4.14 B ................  8.29 GB   (fixed, cannot shrink)
#:   every *trainable* parameter costs 6 bytes more ...  2 B gradient + 4 B AdamW moments
#:     (`torch.optim.AdamW` allocates its moments with `zeros_like`, so they inherit the
#:      parameter's bf16 dtype -- that is why the moments are 4 B and not 16 B)
#:
#:   full un-freeze     4.14 B trainable -> 8.29 + 24.9 = 33.1 GB   does not fit
#:   vision + 18 layers 3.09 B trainable -> 8.29 + 18.5 = 26.8 GB   fits
#:
#: So the 1.05 B that stays frozen is deliberately the pair that is cheapest to do without:
#: `embed_tokens` and `lm_head`.  The latter is the strongest candidate to leave alone --
#: π0.5's subtask prediction was not released, so its text head is not used at inference at
#: all.
#:
#: This is a *scope* choice about what to test, not a claim that 75% is enough; the run it
#: exists for is judged by whether the teacher-forced per-step error moves, on the same
#: ruler as the frozen checkpoints.
_UNFREEZE_ENV_LAYERS = "A3_UNFREEZE_LLM_LAYERS"
_UNFREEZE_ENV_VISION = "A3_UNFREEZE_VISION"


def _unfreeze_request() -> tuple[int, bool]:
    """The requested scope; 0 layers and vision off reproduce the plain frozen run."""

    raw = os.environ.get(_UNFREEZE_ENV_LAYERS, "18")
    vision = os.environ.get(_UNFREEZE_ENV_VISION, "1").strip().lower() not in ("0", "false", "")

    try:
        layers = int(raw)
    except ValueError:
        # Loud rather than defaulted: a typo'd scope would silently produce a frozen run that
        # looks like a negative result about un-freezing.
        raise SystemExit(f"{_UNFREEZE_ENV_LAYERS}={raw!r} is not an integer") from None

    return max(0, layers), vision


def _apply_unfreeze(model, mode: bool) -> None:
    """Re-enable gradients on the requested PaliGemma submodules, and match their mode.

    The mode argument is not decoration.  `PaliGemmaWithExpertModel.train` ends with
    `self.paligemma.eval()` whenever `train_expert_only` is set, and `nn.Module.eval()` on the
    outer policy routes through that same `train(False)`.  Restoring `requires_grad` without
    also restoring `train`/`eval` would leave the un-frozen blocks in eval mode during
    training -- dropout off, and any mode-dependent module frozen -- which trains silently
    differently from what the flag list says.
    """

    layers, vision = _unfreeze_request()
    paligemma = model.paligemma

    if vision:
        tower = paligemma.model.vision_tower
        tower.train(mode)
        for param in tower.parameters():
            param.requires_grad = True

    if layers:
        stack = paligemma.model.language_model.layers
        depth = len(stack)
        for block in stack[depth - min(layers, depth) :]:
            block.train(mode)
            for param in block.parameters():
                param.requires_grad = True


def _patch_partial_unfreeze() -> None:
    """Apply the un-freeze after every place lerobot re-applies the freeze.

    Two call sites, and patching only the first would look like it worked:
      * `PaliGemmaWithExpertModel.__init__` ends with `_set_requires_grad()`
      * `PaliGemmaWithExpertModel.train()` re-freezes on every mode change, and the training
        loop calls `policy.train()` before the first step.
    """

    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel

    original_init = PaliGemmaWithExpertModel.__init__
    original_train = PaliGemmaWithExpertModel.train

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _apply_unfreeze(self, True)

    def train(self, mode: bool = True):
        original_train(self, mode)
        _apply_unfreeze(self, mode)
        # lerobot's override drops `nn.Module.train`'s `self` return; restoring it is what the
        # base class promises, and callers chain off it.
        return self

    PaliGemmaWithExpertModel.__init__ = __init__
    PaliGemmaWithExpertModel.train = train


#: Set to 0 to keep `torch.optim.AdamW`'s default (multi-tensor) implementation.
_ENV_FUSED_ADAMW = "A3_FUSED_ADAMW"


def _patch_fused_adamw() -> None:
    """Force AdamW onto its fused kernel, which is the only variant that fits.

    `torch.optim.AdamW` defaults to the `foreach` (multi-tensor) path, and that path
    materialises a same-sized temporary.  `_multi_tensor_adam` computes

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

    before the update, and with 3.09 B trainable bf16 parameters that temporary alone is
    5.75 GiB.  At that point the run holds 7.72 GiB of weights, 5.75 GiB of gradients and
    11.50 GiB of moments -- 30.43 GiB of the card's 31.36 GiB for the *un-frozen* scope,
    against 12.5 GiB for the frozen one.  So the OOM arrives inside a square root, and the
    traceback names `_foreach_sqrt` rather than anything a reader would connect to memory.

    This is why the batch size is not the knob it looks like: dropping batch 4 to batch 2
    moved the allocation by 0.6 GiB, because with `gradient_checkpointing` the activations
    were never the large term.

    The fused kernel performs the same update in one pass with no such temporary.  It is
    env-gated so a run can go back to the default without editing this file.
    """

    if os.environ.get(_ENV_FUSED_ADAMW, "1").strip().lower() in ("0", "false", ""):
        return

    import torch

    original_init = torch.optim.AdamW.__init__

    def __init__(self, params, *args, **kwargs):
        kwargs.setdefault("fused", True)
        original_init(self, params, *args, **kwargs)

    torch.optim.AdamW.__init__ = __init__


def main() -> int:
    _register_legacy_step_names()
    _patch_partial_unfreeze()
    _patch_fused_adamw()

    # Imported after the shim so nothing that resolves processors can run first.
    from lerobot.scripts.lerobot_train import main as lerobot_main

    # Not `raise SystemExit(lerobot_main())` -- lerobot's main returns None and signals
    # failure by exception, so the exit code has to come from the absence of one.
    lerobot_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
