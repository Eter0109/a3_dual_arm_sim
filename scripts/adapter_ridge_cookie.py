"""Bridge adapter: closed-loop rollout of the ridge read-out on the vision tower.

What this is for
----------------
Every measurement so far has been *one frame at a time with the expert's own state*.  The
probe says a linear map from (joint state, wrist-camera features) reproduces the expert's
next action to 0.0061 rad, and that the best VLA manages 0.0185.  But 0.0061 is an
open-loop error on states the expert visited.  A policy is judged on the states **it**
visits, and only a rollout answers whether that error is small enough to place a Cookie.

This adapter is the cheapest possible thing that can answer it.  It is not a
demonstration that a linear model is a good policy -- it is an *upper bound* on what the
observation supports, with the action head removed from the equation entirely.  Two
outcomes, both informative:

* It places Cookies.  Then the observation is sufficient **and** the π0.5 action head is
  the thing that is broken, which is exactly the interface hypothesis.
* It does not.  Then no amount of training on this interface helps, and the 0.0075 rad
  tolerance -- measured by injecting iid noise into expert actions -- is the wrong ruler
  for a systematic 0.006 rad error.

Loading the tower, and why the whole policy comes with it
--------------------------------------------------------
The features are the fine-tuned checkpoint's own SigLIP tower, because that is what the
probe measured and therefore the only features whose 0.0061 transfers.  Loading it means
constructing a `PI05Policy` and then discarding everything except the tower: extracting
215 vision tensors out of a 9.3 GB `model.safetensors` by hand would have to reproduce
the module's own key layout, and getting that subtly wrong would produce plausible
features that need not match the cache.  The cost is one ~9 GB load at startup, which the
bridge pays once per comparison.

Config (all passed as ``--config key=value``)
---------------------------------------------
``coeffs``  path to the `.npz` written by `scripts/fit_ridge_cookie_policy.py`
``tower``   checkpoint whose vision tower the features come from
``device``  ``cuda`` (default) or ``cpu``
``images``  comma-separated request image names, **in the block order the coefficients
            were fitted with** -- e.g. ``image,image2`` for a ``front,wrist`` fit.  Only
            the images the fitted blocks actually use are read, so a ``state,wrist`` fit
            needs one name and the manifest can drop the other camera.

The adapter is deliberately stateless: the ridge has no recurrent state, so ``reset`` is
absent and the bridge's per-episode reset does nothing.
"""

from __future__ import annotations

from typing import Any

#: π0.5 resizes with aspect-preserving padding to 224 and maps [0,1] -> [-1,1].  Copied
#: rather than called, so that the pixels the tower is handed here are provably the same
#: preprocessing the cache was encoded with.
TOWER_SIZE = (224, 224)

#: Which block each request image slot feeds, matched against the coefficient file's
#: ``blocks`` entry.  The pairing is positional, so the manifest's camera order is the
#: only thing that decides which physical camera is "front".
BLOCK_ORDER = ("front", "wrist")


def _tower_features(frames: list[Any], torch, F, tower, device: str):
    """Mean-pooled last-hidden-state for each frame, one batch.

    Same arithmetic as `probe_observation_information._tower_step`: resize and normalise on
    the GPU for the whole batch, because per-frame host-side interpolation with
    ``antialias=True`` cost more than the forward pass (36 frames/s against 184).
    """

    import numpy as np

    batch = torch.stack(
        [torch.from_numpy(np.array(frame, dtype=np.uint8)).permute(2, 0, 1) for frame in frames]
    ).to(device).float().div_(255.0)
    if tuple(batch.shape[-2:]) != TOWER_SIZE:
        batch = F.interpolate(batch, size=TOWER_SIZE, mode="bilinear",
                              align_corners=False, antialias=True)
    batch = batch.mul_(2.0).sub_(1.0)
    hidden = tower(pixel_values=batch).last_hidden_state
    return hidden.mean(dim=1)


def load(config: dict[str, Any]) -> dict[str, Any]:
    """Build the tower and load the fitted coefficients.  Called once."""

    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_grad_enabled(False)

    coeffs_path = config.get("coeffs")
    if not coeffs_path:
        raise ValueError("adapter_ridge_cookie needs coeffs=<path to ridge_coeffs.npz>")
    tower_ckpt = config.get("tower")
    if not tower_ckpt:
        raise ValueError(
            "adapter_ridge_cookie needs tower=<checkpoint>; the features must come from "
            "the same tower the coefficients were fitted on"
        )
    device = str(config.get("device", "cuda"))

    saved = np.load(coeffs_path)
    blocks = [b for b in str(saved["blocks"]).split(",") if b]
    vision_blocks = [b for b in blocks if b in BLOCK_ORDER]
    names = [n.strip() for n in str(config.get("images", "")).split(",") if n.strip()]
    if len(names) != len(vision_blocks):
        raise ValueError(
            f"the coefficients use {len(vision_blocks)} camera block(s) {vision_blocks} but "
            f"images= names {len(names)} ({names}).  The pairing is positional and a "
            "mismatch would feed the wrist features to the front block without complaining."
        )

    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(str(tower_ckpt)).to(device).eval()
    tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    # 3B of language weights are dead weight from here on; the tower is all that is read.
    del policy
    torch.cuda.empty_cache()

    statistics = {}
    for block in blocks:
        statistics[block] = (
            np.asarray(saved[f"{block}_mean"], dtype=np.float64),
            np.asarray(saved[f"{block}_scale"], dtype=np.float64),
        )

    return {
        "torch": torch,
        "F": F,
        "np": np,
        "tower": tower,
        "device": device,
        "blocks": blocks,
        #: request image name -> block name, positionally.
        "images": dict(zip(names, vision_blocks, strict=True)),
        "statistics": statistics,
        "weights": np.asarray(saved["weights"], dtype=np.float64),
        "x_mean": np.asarray(saved["x_mean"], dtype=np.float64),
        "y_mean": np.asarray(saved["y_mean"], dtype=np.float64),
        "holdout_mean_error": float(saved["holdout_mean_error"]),
        "tower_ckpt": str(tower_ckpt),
        "calls": 0,
    }


def infer(model: dict[str, Any], request: Any) -> dict[str, Any]:
    """One inference: A3 observation in, the ridge's next target joint position out."""

    import time

    np = model["np"]
    torch = model["torch"]

    features: dict[str, Any] = {}
    if any(block in model["images"].values() for block in ("front", "wrist")):
        frames, slots = [], []
        for image_name, block in model["images"].items():
            frames.append(request.image(image_name))
            slots.append(block)
        started = time.monotonic()
        pooled = _tower_features(frames, torch, model["F"], model["tower"], model["device"])
        elapsed_ms = (time.monotonic() - started) * 1000.0
        for index, block in enumerate(slots):
            features[block] = pooled[index].float().cpu().numpy().astype(np.float64)
    else:
        elapsed_ms = 0.0

    if "state" in model["blocks"]:
        features["state"] = np.asarray(request.vector("observation.state"), dtype=np.float64)

    parts = []
    for block in model["blocks"]:
        mean, scale = model["statistics"][block]
        parts.append((features[block] - mean) / scale)
    x = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

    action = (x - model["x_mean"]) @ model["weights"] + model["y_mean"]
    model["calls"] += 1

    # One row, because a ridge has no chunk: it maps the observation to a single target,
    # and the manifest's `execute: 1` makes the bridge re-plan from the live state every
    # step.  That is the closed-loop condition the probe was standing in for.
    return {
        "actions": action.reshape(1, -1),
        "space": "joint_position",
        # `raw` means "already physical, apply nothing" -- the ridge was fitted on A3's own
        # joint units, so there is no transform left to undo.
        "units": "raw",
        "info": {
            "action_units": "radians",
            "ridge_blocks": list(model["blocks"]),
            "ridge_holdout_mean_error": model["holdout_mean_error"],
            "ridge_calls": model["calls"],
            "ridge_tower": model["tower_ckpt"],
            "tower_ms": round(elapsed_ms, 2),
        },
    }
