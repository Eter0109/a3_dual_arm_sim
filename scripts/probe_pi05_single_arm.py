"""Teacher forcing on the single-arm A3 cookie task.

Why this does not go through the A3 adapter
-------------------------------------------
``adapters/pi05_a3.py`` is built for the dual-arm 16-dimensional interface, and two of its
assumptions are wrong here.  Both fail in the way this project keeps meeting: not loudly, and
not in a place that names the cause.

* ``_carries_statistics()`` returns True for any checkpoint whose normaliser step has a
  non-empty ``features`` field.  ``lerobot/pi05_libero_finetuned`` is such a checkpoint, and
  its statistics describe *LIBERO*: an 8-wide state of end-effector pose plus gripper, and a
  7-wide end-effector delta action.  A3's single-arm state is 7 joints plus a gripper -- also
  eight numbers -- so the widths agree, nothing raises, and the model is quietly normalised
  into the wrong coordinate system.  That is the failure `docs/pi05-integration.md` section 5
  describes, arriving somewhere new.
* ``_load_statistics()`` requires the dataset's state statistics to be exactly ``A3_DIM``
  (16) wide, so the single-arm dataset raises there instead.

What this does instead is what ``lerobot_train.py`` does: build the policy and its processor
pipeline from the checkpoint, then install the *dataset's* statistics into the normaliser and
unnormaliser steps through those steps' own overrides.  That is the wiring a fine-tune uses,
so the number below describes the checkpoint rather than the plumbing around it -- which is
the only kind of number worth acting on when the question is "would this make a better
starting point".

What to compare against
-----------------------
The headline is the row-0 error in radians, so it lines up with the 0.0359 the dual-arm probe
reports for ``--which adv``: both are the first action of the predicted chunk, which is the
one the arm acts on first.  Beside it this prints the demonstration's own per-step motion over
the same frames, because an error below that is not something the arm could even act on.

    python scripts/probe_pi05_single_arm.py --ckpt lerobot/pi05_libero_finetuned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

DEFAULT_ROOT = Path("/root/autodl-tmp/datasets/a3_cookie_single_arm")
DEFAULT_TASK = (
    "grasp five yellow blocks together from the source bin, insert them upright into the mark"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default="local/a3_single_arm")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--limit", type=int, default=45, help="frames per episode; 0 for all")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="inferences per frame; above one, separates sampling spread from bias",
    )
    return parser.parse_args()


def build(ckpt: str, dataset):
    """Policy plus processors, with the dataset's statistics installed.

    Mirrors ``lerobot_train.py:278``.  The override has to go on the step that consumes it
    rather than through `make_pre_post_processors(..., dataset_stats=...)`: that argument is
    ignored whenever the pipeline already contains a normaliser step, which every fine-tuned
    checkpoint's pipeline does, and the symptom of getting it wrong is an absence rather than
    an error.
    """

    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(ckpt).to("cuda").eval()

    # `predict_action_chunk` slices its output to `output_features[ACTION].shape[0]`
    # (modeling_pi05.py:1243).  `lerobot/pi05_libero_finetuned` declares 7 -- LIBERO's
    # end-effector delta -- while this task's action is 8 joint values.  Left alone the chunk
    # comes back seven wide and the unnormaliser then broadcasts it against eight-wide
    # statistics, which fails as
    #     RuntimeError: The size of tensor a (7) must match the size of tensor b (8)
    # naming tensor shapes and nothing about a checkpoint written for a different robot.
    # The action dim is not a weight shape -- `max_action_dim` stays 32 and the projections are
    # untouched -- so overriding it changes what is kept, not what is loaded.
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

    print(f"checkpoint : {args.ckpt}")
    print(f"dataset    : {args.repo_id}  ({dataset.num_episodes} episodes, {dataset.num_frames} frames)")

    table = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "frame_index"])
         for f in sorted(args.root.glob("data/**/*.parquet"))],
        ignore_index=True,
    )
    episode_indices = table["episode_index"].to_numpy()
    frame_indices = table["frame_index"].to_numpy()

    indices: list[int] = []
    per_episode: list[tuple[int, int]] = []
    for offset in range(args.episodes):
        episode = args.episode + offset
        mask = episode_indices == episode
        if not mask.any():
            break
        first = int(frame_indices[mask].min())
        length = int(mask.sum())
        chosen = list(range(first, first + length, args.stride))
        if args.limit:
            chosen = chosen[: args.limit]
        indices.extend(chosen)
        per_episode.append((episode, length))
    if not indices:
        raise SystemExit(f"episode {args.episode} is not in the dataset")

    print("episodes   : " + ", ".join(f"{e} ({n} frames)" for e, n in per_episode))
    print(f"scoring    : {len(indices)} frames at stride {args.stride}")
    print()

    first_errors, motions, chunk_profiles = [], [], []
    signed: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    spreads: list[float] = []

    for index in indices:
        frame = dict(dataset[index])
        expert = np.asarray(frame["action"], dtype=np.float64)
        for noisy in ("task", "timestamp", "frame_index", "episode_index", "index", "task_index"):
            frame.pop(noisy, None)

        batch = preprocess({**frame, "task": DEFAULT_TASK})
        with torch.no_grad():
            draws = []
            for _ in range(max(1, args.samples)):
                out = postprocess(policy.predict_action_chunk(batch)).detach().float().cpu().numpy()
                # The processor preserves the batch dimension, so this arrives as
                # (1, chunk, dim).  Leaving it in place makes `chunk[0]` the whole bach row
                # rather than the first action, and every number that follows broadcasts
                # silently instead of raising.
                draws.append(out[0] if out.ndim == 3 else out)
        draws = np.stack(draws)
        chunk = draws[0]
        predicted = chunk[0]

        first_errors.append(float(np.abs(predicted - expert).max()))
        signed.append(predicted - expert)
        absolute.append(np.abs(predicted - expert))

        # The expert's own *single-step* motion, so the error has a same-unit yardstick.
        # Taking it from the sampled frames instead would measure `stride` steps of travel and
        # inflate the reference by that factor, flattering the model by exactly that much.
        # Frames at an episode boundary are skipped: the previous index belongs to the episode
        # before, and the jump between two episodes is not motion the policy was asked to make.
        previous = episode_indices[index - 1] if index > 0 else -1
        if index > 0 and previous == episode_indices[index]:
            neighbour = np.asarray(dict(dataset[index - 1])["action"], dtype=np.float64)
            motions.append(float(np.abs(expert - neighbour).max()))

        if len(draws) > 1:
            spreads.append(float(np.abs(draws - chunk).max()))

        # The expert's own chunk: this frame and the ones after it inside the same episode.
        horizon = min(len(chunk), 50)
        following = [
            i for i in range(int(index), int(index) + horizon)
            if i < len(dataset) and episode_indices[i] == episode_indices[index]
        ]
        if following:
            expert_chunk = np.stack(
                [np.asarray(dict(dataset[i])["action"], dtype=np.float64) for i in following]
            )
            chunk_profiles.append(np.abs(chunk[: len(following)] - expert_chunk).max(axis=1))

    errors = np.array(first_errors)
    motions_arr = np.array(motions)
    signed_arr = np.stack(signed)
    absolute_arr = np.stack(absolute)

    print("  |predicted - expert| at chunk row 0, max over joints (rad)")
    print(f"    mean {errors.mean():.5f}   median {np.median(errors):.5f}   "
          f"p90 {np.percentile(errors, 90):.5f}   max {errors.max():.5f}")
    print()
    print("  the demonstration's own per-step motion over the same frames (rad)")
    print(f"    mean {motions_arr.mean():.5f}   median {np.median(motions_arr):.5f}   "
          f"max {motions_arr.max():.5f}")
    print()
    ratio = errors.mean() / motions_arr.mean() if len(motions_arr) and motions_arr.mean() else float("nan")
    print(f"  error / demonstration motion = {ratio:.2f}")
    print(f"  worst frame: index {indices[int(errors.argmax())]}, error {errors.max():.5f} rad")

    if len(spreads) > 1:
        print()
        print(f"  sampling spread per frame (rad): mean {np.mean(spreads):.5f}")
        print(f"  error that survives averaging  : {errors.mean() - np.mean(spreads):.5f}")

    if chunk_profiles:
        # The per-frame chunks have different lengths: a frame near the end of an episode has
        # fewer following frames inside the same episode, so the arrays cannot be stacked
        # directly.  Padding with NaN and taking a NaN-aware mean keeps every frame's
        # contribution at the positions it does have.
        width = max(len(profile) for profile in chunk_profiles)
        padded = np.full((len(chunk_profiles), width), np.nan)
        for row, profile in enumerate(chunk_profiles):
            padded[row, : len(profile)] = profile
        mean_profile = np.nanmean(padded, axis=0)
        print()
        print("  error by position inside the predicted chunk (rad)")
        shown = mean_profile[: min(len(mean_profile), 16)]
        scale = shown.max() or 1.0
        for row, value in enumerate(shown):
            print(f"    row {row:>2}  {value:.5f}  " + "#" * int(round(24 * value / scale)))
        print(f"    row 0 {shown[0]:.5f}  ->  row {len(shown)-1} {shown[-1]:.5f}   "
              f"growth x{shown[-1] / shown[0]:.2f}")

    names = [f"j{i}" for i in range(absolute_arr.shape[1])]
    print()
    print("  per joint: mean signed error (bias) and mean absolute error")
    print("    joint    bias      |err|     |err|/motion")
    for column, name in enumerate(names):
        bias = signed_arr[:, column].mean()
        magnitude = absolute_arr[:, column].mean()
        print(f"    {name:<7s} {bias:+.5f}  {magnitude:.5f}     {magnitude / motions_arr.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
