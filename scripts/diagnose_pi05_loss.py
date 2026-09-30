"""Why is the π0.5 fine-tune loss ~1e12 at the first step?

A loss that large on step one is not divergence -- there is nothing to diverge from yet.
It means the loss function is being handed numbers that are not the ones it expects, and
the useful question is which tensor is wrong.

This builds the same dataset and the same processor pipeline that `lerobot_train.py`
builds, pulls one batch, and prints what actually reaches the model: the normalized
action the loss is scored against, the state that becomes the prompt, and the image
range.  It also computes the loss twice -- once normally, once with the action replaced by
zeros -- because if the loss stays enormous with zero targets then the target is not the
culprit.

    /root/autodl-tmp/a3-collect/.venv/bin/python scripts/diagnose_pi05_loss.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

DATASET = Path("/root/autodl-tmp/gr00t-runs/b1/merged")
REPO = "lerobot/pi05_base"
TOKENIZER = "/root/autodl-tmp/models/paligemma-tokenizer"
RENAME = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
}


def describe(name: str, tensor: torch.Tensor) -> None:
    array = tensor.detach().float().cpu().numpy()
    print(
        f"  {name:<44} {str(tuple(array.shape)):<18} "
        f"min {array.min():+.4f}  max {array.max():+12.4f}  "
        f"mean {array.mean():+.4f}  absmax {np.abs(array).max():.4e}"
    )


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    print("dataset metadata ...")
    meta = LeRobotDatasetMetadata("local/a3", root=DATASET)
    print(f"  episodes {meta.total_episodes}  frames {meta.total_frames}")

    print("\ndataset statistics for the columns the loss uses:")
    for key in ("action", "observation.state"):
        if key not in meta.stats:
            print(f"  {key}: ABSENT from meta.stats")
            continue
        stats = meta.stats[key]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        denominator = q99 - q01
        print(f"  {key}: width {q01.shape[0]}")
        print(f"    q01        {np.round(q01, 4)}")
        print(f"    q99        {np.round(q99, 4)}")
        print(f"    q99-q01    {np.round(denominator, 6)}")
        # QUANTILES divides by this and guards only against exact zero.  A joint that is
        # constant across the dataset -- A3's right gripper is -- normalizes to exactly
        # -1 in training, and anything slightly off becomes a huge number.
        if np.any(np.abs(denominator) < 1e-3):
            print(f"    ^^ degenerate: {np.where(np.abs(denominator) < 1e-3)[0].tolist()}")

    print("\nbuilding policy ...")
    policy = make_policy(
        cfg=_config(),
        ds_meta=meta,
        rename_map=RENAME,
    )

    print("\nconfig after make_policy:")
    print(f"  output_features: { {k: v.shape for k, v in policy.config.output_features.items()} }")
    print(f"  input_features : { {k: v.shape for k, v in policy.config.input_features.items()} }")
    print(f"  normalization  : { {k: str(v) for k, v in policy.config.normalization_mapping.items()} }")
    print(f"  max_action_dim : {policy.config.max_action_dim}")

    dataset = LeRobotDataset("local/a3", root=DATASET, video_backend="pyav")
    preprocess, _ = make_pre_post_processors(
        policy.config,
        REPO,
        dataset_stats=meta.stats,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": TOKENIZER},
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )

    frame = dict(dataset[0])
    print(f"\nraw dataset frame keys: {sorted(frame)}")
    describe("raw frame['action']", torch.as_tensor(frame["action"]))
    describe("raw frame['observation.state']", torch.as_tensor(frame["observation.state"]))

    batch = preprocess(frame)
    print("\nafter preprocess:")
    for key in sorted(batch):
        value = batch[key]
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            describe(key, value)

    loss, info = policy.forward(batch)
    print(f"\nloss with the real action : {loss.item():.6e}")
    print(f"  per-dimension (first 16): {np.round(info['loss_per_dim'][:16], 6)}")

    # Control: a zero target.  If the loss is still enormous, the target is not the
    # problem -- the prediction, the mask, or the reduction is.
    zeroed = dict(batch)
    zeroed["action"] = torch.zeros_like(batch["action"])
    loss_zero, info_zero = policy.forward(zeroed)
    print(f"loss with a zero action   : {loss_zero.item():.6e}")
    print(f"  per-dimension (first 16): {np.round(info_zero['loss_per_dim'][:16], 6)}")

    print("\nwhat the model predicts for this batch:")
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    describe("predicted chunk", chunk)
    return 0


def _config():
    """The policy config, loaded the way `TrainPipelineConfig.validate` loads it.

    CPU and float32 rather than the training settings: this is a diagnostic about tensor
    values, and doing it in bf16 would put its own rounding on top of whatever is wrong.
    """

    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(REPO)
    config.pretrained_path = Path(REPO)
    config.device = "cpu"
    config.dtype = "float32"
    return config


if __name__ == "__main__":
    sys.exit(main())
