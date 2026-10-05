"""Is the policy predicting the task, or the conditional mean?

A fine-tuned VLA that has actually learned the task should beat the *trivial* predictors
on the frames it was trained on.  Two of them cost nothing to compute and are therefore
the honest yardstick:

* ``hold still``  -- predict the current state.  On this data the expert moves 0.0135 rad
  per step, so this baseline's error is 0.0135 by construction.  A model that is worse than
  this has learned nothing about *where to go*.
* ``global mean action`` -- predict the dataset's average action (or, for a relative-action
  model, the average delta, which is approximately zero).  This is what a network collapses
  to when it cannot fit the mapping: minimising MSE against a fixed target set puts the
  output at the conditional mean, and for a *delta* target that mean is "do not move",
  which is a stable fixed point in closed loop rather than a near miss.

The third number is the one that separates "wrong magnitude" from "wrong direction": the
cosine similarity between the delta the model asks for and the delta the expert asks for,
both measured against the current state.  A collapsed model has a near-zero mean cosine
and an error close to the hold-still error; a model that tracks has a cosine near one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_TASK = "pick up the one yellow block from the source bin and put it in the target box"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="inferences per frame, averaged; pi0.5's flow head is stochastic and a single "
             "draw carries ~0.006 rad of noise, which is half the effect sizes measured here",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="override the policy's chunk_size at inference (0 = keep the trained value). "
             "The action expert's *suffix* is `chunk_size` tokens that attend to each other "
             "bidirectionally, so the predicted first action is a function of the whole "
             "50-row block; shrinking the block to 1 row measures how much of the error is "
             "the sibling rows talking to each other versus the prefix alone.",
    )
    return parser.parse_args()


def _checkpoint_has_state_projector(ckpt: str) -> bool:
    """Whether this checkpoint carries the extra proprioceptive token.

    Needed because the token is a runtime patch, not part of the vendored model: a probe
    that builds the stock `PI05Policy` for a checkpoint trained *with* the patch leaves
    `model.state_proj.*` unexpected, `load_state_dict` discards them, and the evaluation
    then runs a 968-token prefix against a model that was trained on 969.  That is not a
    small mismatch -- it is a different model, and it read as "the state channel made things
    slightly worse and killed the vision reliance" until the loader's own message gave it
    away.
    """

    import glob as _glob
    import os

    from safetensors import safe_open

    path = Path(ckpt)
    if path.is_dir():
        files = [path / "model.safetensors"]
    else:
        try:
            from huggingface_hub import snapshot_download

            cached = snapshot_download(ckpt, local_files_only=True)
            files = [Path(p) for p in _glob.glob(os.path.join(cached, "*.safetensors"))]
        except Exception:
            return False
    for candidate in files:
        if candidate.is_file():
            with safe_open(str(candidate), framework="pt") as handle:
                if any("state_proj" in key for key in handle.keys()):
                    return True
    return False


def build(ckpt: str, dataset):
    """Policy plus processors, with the dataset's statistics installed.

    Identical to ``probe_pi05_single_arm.build``; see there for why the statistics have to
    go through the step's own override rather than through ``dataset_stats=``.
    """

    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.processor import RelativeActionsProcessorStep
    from lerobot.processor.pipeline import ProcessorStepRegistry

    # `pi05_base`'s shipped `policy_preprocessor.json` lists `relative_actions_processor`,
    # the name this step had when that checkpoint was written; this lerobot registers the
    # same class as `delta_actions_processor`.  Deserialization is a plain dict lookup, so
    # the miss surfaces as
    #     ImportError: Failed to load processor step from registry ...
    # which reads like a missing dependency rather than version skew between a checkpoint
    # and the code loading it.  Harmless to alias: the step is `enabled: false` there, so
    # the class is constructed and does nothing.
    ProcessorStepRegistry._registry.setdefault(
        "relative_actions_processor", RelativeActionsProcessorStep
    )

    # Before `from_pretrained`, because the projection has to exist for the checkpoint's
    # own `state_proj` weights to have somewhere to land.
    if _checkpoint_has_state_projector(ckpt):
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_pi05_launcher import _patch_state_projector

        _patch_state_projector()

    policy = PI05Policy.from_pretrained(ckpt).to("cuda").eval()
    action_width = int(np.asarray(dataset.meta.stats["action"]["q01"]).reshape(-1).shape[0])
    policy.config.output_features["action"] = PolicyFeature(
        type=FeatureType.ACTION, shape=(action_width,)
    )

    overrides = {
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    post_overrides = {
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }
    pre, post = make_pre_post_processors(
        policy.config, ckpt, preprocessor_overrides=overrides, postprocessor_overrides=post_overrides
    )
    return torch, policy, pre, post


def main() -> int:
    args = parse_args()

    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=str(args.root), video_backend="pyav")
    torch, policy, preprocess, postprocess = build(args.ckpt, dataset)

    if args.chunk and args.chunk != policy.config.chunk_size:
        # Every use of `chunk_size` inside the model -- the suffix length, the attention
        # masks, the noise shape and the output slice -- reads `self.config.chunk_size` at
        # call time, so a single assignment changes them consistently.  `n_action_steps`
        # only governs the queue in `select_action`, which this probe does not use, but it
        # is validated against `chunk_size` at construction, so keep it inside the bound.
        trained_chunk = policy.config.chunk_size
        policy.config.chunk_size = args.chunk
        policy.config.n_action_steps = min(policy.config.n_action_steps, args.chunk)
        print(f"chunk_size overridden: {trained_chunk} -> {args.chunk}")

    table = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "frame_index"]) for f in
         sorted(args.root.glob("data/**/*.parquet"))],
        ignore_index=True,
    )
    episode_indices = table["episode_index"].to_numpy()
    frame_indices = table["frame_index"].to_numpy()

    # The mean action, in the dataset's own action column.  For a relative-action
    # checkpoint the column still holds absolute actions (only the *statistics* describe
    # deltas), so this is expressed as "the average pose", which is what the model would
    # produce if it collapsed.
    all_actions = []
    for path in sorted(args.root.glob("data/**/*.parquet")):
        column = pd.read_parquet(path, columns=["action"])["action"].to_numpy()
        all_actions.append(np.stack([np.asarray(v, dtype=np.float64) for v in column]))
    mean_action = np.concatenate(all_actions).mean(axis=0)

    print(f"checkpoint : {args.ckpt}")
    print(f"dataset    : {args.repo_id}  ({dataset.num_episodes} episodes)")
    print(f"mean action: {np.round(mean_action, 3)}")
    print()

    indices: list[int] = []
    for offset in range(args.episodes):
        mask = episode_indices == (args.episode + offset)
        if not mask.any():
            break
        first = int(frame_indices[mask].min())
        length = int(mask.sum())
        indices.extend(list(range(first, first + length, args.stride))[: args.limit])

    rows = []
    for index in indices:
        frame = dict(dataset[index])
        expert = np.asarray(frame["action"], dtype=np.float64)
        state = np.asarray(frame["observation.state"], dtype=np.float64)[: expert.shape[0]]
        stripped = {k: v for k, v in frame.items()
                    if k not in ("task", "timestamp", "frame_index", "episode_index", "index",
                                 "task_index")}
        stripped.pop("action", None)
        stripped.pop("observation.state", None)
        # The state is put back explicitly: the dataset's state may be wider than the
        # action, and the pipeline wants exactly the width its statistics describe.
        batch = preprocess({**stripped, "observation.state": frame["observation.state"],
                            "task": DEFAULT_TASK})
        with torch.no_grad():
            draws = [postprocess(policy.predict_action_chunk(batch)).detach().float().cpu().numpy()
                     for _ in range(max(1, args.samples))]
        # The processor preserves the batch dimension, so each draw is (1, chunk, dim);
        # leaving it in place makes `[0]` the whole batch row rather than the first action.
        stacked = np.stack([(d[0] if d.ndim == 3 else d)[0] for d in draws])
        predicted = stacked.mean(0)

        wanted = expert - state
        asked = predicted - state
        rows.append(
            {
                "model": float(np.abs(predicted - expert).max()),
                "hold": float(np.abs(state - expert).max()),
                "mean": float(np.abs(mean_action - expert).max()),
                "wanted_norm": float(np.linalg.norm(wanted)),
                "asked_norm": float(np.linalg.norm(asked)),
                "cosine": float(np.dot(wanted, asked) / (np.linalg.norm(wanted) * np.linalg.norm(asked) + 1e-12)),
                "held": float(np.abs(predicted - state).max()),
            }
        )

    def line(key: str) -> str:
        values = np.array([r[key] for r in rows])
        return f"  mean {values.mean():.5f}   median {np.median(values):.5f}   max {values.max():.5f}"

    print(f"scored on {len(rows)} frames the model was TRAINED on\n")
    print("mean |prediction - expert| per joint")
    print("  model        " + line("model"))
    print("  hold still   " + line("hold") + "   <- the state it was handed")
    print("  global mean  " + line("mean") + "   <- what a collapsed network outputs")
    print()
    print("how far the model moves from the state it was handed")
    print("  |predicted - state|  " + line("held"))
    print("  |expert    - state|  " + line("wanted_norm") + "   <- the motion it should ask for")
    print("  |predicted - state|  " + line("asked_norm"))
    print()
    cos = np.array([r["cosine"] for r in rows])
    print(f"cosine(model's delta, expert's delta)  mean {cos.mean():+.3f}   median {np.median(cos):+.3f}")
    print("  1.0 = right direction, 0.0 = unrelated, negative = opposing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
