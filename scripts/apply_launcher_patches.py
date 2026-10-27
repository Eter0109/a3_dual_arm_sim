#!/usr/bin/env python3
"""Re-create the launcher's extra modules in an inference process.

A checkpoint trained by `train_pi05_launcher.py` contains parameters that pi0.5 does not
have: `model.state_proj.*` (one learned prefix token for proprioception) and, for the v6
line, `model.history_encoder.*` plus `model.history_proj.*` (a latent of the robot's own
recent trajectory).  `PI05Policy.from_pretrained` cannot build either, and it reacts to the
first one in a way that is easy to miss:

    if key.startswith("state_proj."):
        logging.warning(f"Skipping state_proj key in pi05 mode: {key}")

That warning scrolls past in a 4-minute model load.  What it means for an evaluation is that
the checkpoint runs as a **plain pi0.5**: the projector is absent, the token is never
appended, and the state reaches the model only through the prompt bins.  For v5 that is the
difference between "18 numbers in the prompt plus a learned projection" and "18 numbers in
the prompt", and for v6 it is the difference between having a history channel and not having
one at all -- the forward pass simply ignores `observation.history`, so a benchmark would
report a number for an arm that was never run.

So the modules have to exist before `from_pretrained` is called, which is what this does.
It reads the checkpoint's own key list rather than trusting the caller: a projector is
created only when `model.state_proj.*` is present, and the history encoder's geometry (width,
per-step width) is taken from the stored GRU so a mismatch fails loudly instead of loading
into the wrong shape.

    from apply_launcher_patches import apply
    summary = apply(Path(ckpt), history_dims=480)

Geometry has one part that cannot be recovered from the weights: the number of steps in the
window.  `to_gates.weight` is `(3*width, per_step)` and `from_hidden.weight` is
`(3*width, width)`, so `width` and `per_step` are both readable, but the recurrence is
length-agnostic -- ten frames and twenty frames are the same parameter shapes.  The window
width therefore has to come from the dataset (`history_dims`), and `steps` is
`history_dims // per_step`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parent


def checkpoint_keys(checkpoint: Path) -> set[str]:
    """Every parameter name in a checkpoint, sharded or single-file."""

    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        import json

        return set(json.loads(index.read_text()).get("weight_map", {}))
    single = checkpoint / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            return set(handle.keys())
    return set()


def apply(checkpoint: Path, history_dims: int = 0) -> dict:
    """Enable and install whichever launcher modules this checkpoint needs.

    Returns a summary for the caller's log; raises on a geometry mismatch rather than
    quietly loading a checkpoint into the wrong module.
    """

    keys = checkpoint_keys(checkpoint)
    needs_state = any(key.startswith("model.state_proj.") for key in keys)
    needs_history = any(key.startswith("model.history_encoder.") for key in keys)
    summary: dict[str, object] = {"state_projector": needs_state, "history_encoder": needs_history}
    if not (needs_state or needs_history):
        return summary

    if LAUNCHER_DIR.as_posix() not in sys.path:
        sys.path.insert(0, LAUNCHER_DIR.as_posix())
    import train_pi05_launcher as launcher

    if needs_state:
        os.environ.setdefault("A3_STATE_PROJECTOR", "1")
        launcher._patch_state_projector()

    if needs_history:
        from safetensors import safe_open

        single = checkpoint / "model.safetensors"
        gate_shape = ()
        hidden_shape = ()
        if single.is_file():
            with safe_open(str(single), framework="pt") as handle:
                gate_shape = tuple(handle.get_slice("model.history_encoder.gru.to_gates.weight").get_shape())
                hidden_shape = tuple(
                    handle.get_slice("model.history_encoder.gru.from_hidden.weight").get_shape()
                )
        if len(gate_shape) != 2 or len(hidden_shape) != 2:
            raise RuntimeError(
                f"{checkpoint} declares model.history_encoder.* but its shapes could not be "
                f"read; the checkpoint is sharded, and this only handles model.safetensors."
            )
        per_step = int(gate_shape[1])
        width = int(hidden_shape[1])
        if history_dims <= 0:
            raise RuntimeError(
                "this checkpoint has a history encoder, so `history_dims` must be given: the "
                "number of steps in the window is the one part of the geometry the weights do "
                "not record."
            )
        steps, remainder = divmod(int(history_dims), per_step)
        if remainder or steps < 1:
            raise RuntimeError(
                f"history_dims={history_dims} is not a multiple of the checkpoint's per-step "
                f"width {per_step}, so the window cannot be reconstructed."
            )
        os.environ["A3_HISTORY_ENCODER"] = "1"
        os.environ["A3_HISTORY_STEPS"] = str(steps)
        os.environ["A3_HISTORY_PER_STEP"] = str(per_step)
        os.environ["A3_HISTORY_WIDTH"] = str(width)
        # Inference wants the mean, not a sample: the bridge is already stochastic through
        # the flow-matching noise, and a second source would make one seed unrepeatable.
        os.environ["A3_HISTORY_SAMPLE"] = "0"
        launcher._patch_history_encoder()
        summary.update(steps=steps, per_step=per_step, width=width)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install the launcher's inference patches.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--history-dims", type=int, default=0)
    arguments = parser.parse_args()
    print(apply(arguments.checkpoint, arguments.history_dims))
