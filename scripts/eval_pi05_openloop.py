"""Open-loop error of a fine-tuned pi0.5 on held-in data.

Feeds the model observations from the training set and compares its predicted action
chunk against what the expert actually did.  No simulator, no rollout, a few minutes --
and it is the cheapest way to find out whether a checkpoint is worth the time a
closed-loop evaluation costs.

Why the baselines matter more than the error
--------------------------------------------
A dataset of one task in one scene has a narrow action distribution, so a model that
ignores its input entirely and emits the dataset's mean action can score a deceptively
low error.  An absolute error figure therefore says nothing on its own.  What is reported
below is the model against two trivial predictors on the same samples:

``hold``
    Repeat the current joint positions for every step of the chunk.
``mean``
    Emit the dataset's per-dimension mean action at every step, regardless of the
    observation.  This is the one that catches a collapsed policy: a fine-tune that has
    stopped looking at its input converges towards it, and its advantage over ``mean``
    shrinks towards zero.

A model is only useful if it beats both by a margin.  ``mean`` is fitted on a disjoint
slice of episodes, so the baseline is not scored on the data it was fitted to.

What this does and does not say
-------------------------------
These observations are training data.  A good score is **necessary and not sufficient**:
it rules out a broken adapter or a checkpoint that failed to train, and it says nothing
about whether the policy can act on its own predictions for 400 steps.  The GR00T
fine-tune is the cautionary example -- it scored 48x better than ``mean`` here and still
transferred zero cookies, because what it learned was a hover.

    Usage::

        /root/autodl-tmp/a3-collect/.venv/bin/python scripts/eval_pi05_openloop.py \\
            [--checkpoint PATH] [--episodes 24] [--frames 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

#: Where the adapter and the dataset live.
ADAPTERS = Path("/root/autodl-tmp/a3-collect/adapters")
DATASET = Path("/root/autodl-tmp/gr00t-runs/b1/merged")
TOKENIZER = "/root/autodl-tmp/models/paligemma-tokenizer"
#: The base checkpoint, for evaluating before a fine-tune exists.
BASE_REPO = "lerobot/pi05_base"

#: Dataset key -> the name the bridge would deliver it under.  Written out rather than
#: inferred so a rename on either side fails here instead of silently feeding the wrong
#: tensor.
IMAGE_KEYS = {
    "observation.images.front": "front_view",
    "observation.images.left_wrist": "left_wrist_view",
    "observation.images.right_wrist": "right_wrist_view",
}
#: The four action blocks, and where each sits in A3's 16-value action column.
#:
#: Note that the dataset stores **one** 16-dimension ``action`` column, not four
#: separate keys.  The GR00T open-loop script could ask for ``action.left_arm_joint_pos``
#: directly because GR00T's own reader splits the column using the modality config it was
#: trained with; LeRobot's ``LeRobotDataset`` does not, so the split happens here.  A
#: script written for the other reader fails on the first frame with a KeyError naming a
#: key that looks like it should exist.
BLOCKS = (
    ("left_arm_joint_pos", 0, 7),
    ("left_gripper", 7, 8),
    ("right_arm_joint_pos", 8, 15),
    ("right_gripper", 15, 16),
)
TRAINING_PROMPT = (
    "grasp five yellow blocks together from the source bin, insert them upright "
    "into the marked slots of the 2 by 5 box, then return to the source bin, grasp "
    "five more, and fill all ten slots"
)
#: A3's joint width, checked against the dataset at load.
A3_DIM = 16
EPISODE_LENGTH = 400


class _Request:
    """The subset of the bridge's Request the adapter reads.

    Constructed here rather than imported because the bridge's own dataclass lives on the
    A3 side and pulling it in would make this script depend on that checkout for no
    reason -- the adapter only ever calls ``.task``, ``.chunk``, ``.image()`` and
    ``.vector()``.
    """

    def __init__(self, arrays: dict[str, np.ndarray], task: str, chunk: int) -> None:
        self._arrays = arrays
        self.task = task
        self.chunk = chunk
        self.episode_id = "openloop"
        self.step = 0

    def image(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def vector(self, name: str) -> np.ndarray:
        return self._arrays[name]


def load_adapter():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pi05_a3", ADAPTERS / "pi05_a3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # `pyav` for the same reason the training uses it: A3 records AV1, and this platform's
    # torchcodec cannot load its shared libraries.
    return LeRobotDataset("local/a3", root=DATASET, video_backend="pyav")


def to_observation(frame: dict) -> dict[str, np.ndarray]:
    """One bridge-shaped observation from a dataset frame.

    The dataset hands back float tensors in [0, 1] channels-first, because that is what a
    training batch wants; the bridge delivers uint8 ``(H, W, C)``.  The adapter asserts on
    the dtype, so this has to convert back -- and converting here rather than loosening
    that assertion is deliberate, since the assertion is what would catch a real bridge
    sending the wrong encoding.
    """

    arrays: dict[str, np.ndarray] = {}
    for key, name in IMAGE_KEYS.items():
        image = frame[key]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        # Channels-first float in [0, 1] -> channels-last uint8.
        image = np.transpose(image, (1, 2, 0))
        arrays[name] = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    state = frame["observation.state"]
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    arrays["observation.state"] = np.asarray(state, dtype=np.float64).reshape(-1)
    return arrays


def expert_chunk(dataset, episode: int, frame_index: int, chunk: int) -> np.ndarray:
    """The next ``chunk`` actions the expert took, as an ``(chunk, 16)`` array.

    Assembled from consecutive frames, because that is what the model predicts and what
    the comparison has to be against.  A subtlety worth stating: LeRobot's dataset returns
    the action **at** a frame, shape ``(16,)``, while GR00T's reader returned a *chunk* of
    16 future actions per sample, shape ``(16, 7)`` per block, because its dataset class
    applies the action horizon itself.  A script carried over from that one would compare
    a 16-step prediction against a single step and report an error that is really a shape
    mismatch.

    The caller keeps ``frame_index`` at least ``chunk`` frames from the end of the
    episode, so the window never crosses into the next one -- a chunk that spanned two
    episodes would score the model against a discontinuity no policy could predict.
    """

    rows = []
    for offset in range(chunk):
        index = episode * EPISODE_LENGTH + frame_index + offset
        action = dataset[index]["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape[0] != A3_DIM:
            raise ValueError(
                f"the dataset's action column is {value.shape[0]} wide; A3's joint action "
                f"is {A3_DIM}.  The block slices below index into it by position, so a "
                "different width means they would read the wrong joints."
            )
        rows.append(value)
    return np.stack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        default=BASE_REPO,
        help="a Hub id or a local checkpoint directory; defaults to the base",
    )
    parser.add_argument("--episodes", type=int, default=24, help="episodes to score")
    parser.add_argument("--frames", type=int, default=8, help="frames per episode")
    parser.add_argument("--chunk", type=int, default=16, help="steps to compare per call")
    parser.add_argument(
        "--baseline-episodes",
        default="160-200",
        help="episode range the `mean` baseline is fitted on, disjoint from the scored set",
    )
    args = parser.parse_args()

    adapter = load_adapter()
    dataset = build_dataset()
    print(f"dataset: {DATASET.name}  frames={len(dataset)}")

    low, high = (int(part) for part in args.baseline_episodes.split("-"))
    frames = np.linspace(
        args.chunk + 2, EPISODE_LENGTH - args.chunk - 2, args.frames
    ).astype(int)
    plan = [(ep, int(fr)) for ep in range(args.episodes) for fr in frames]
    print(
        f"scoring {len(plan)} observations from episodes 0..{args.episodes - 1} "
        f"(the model trained on these)"
    )

    # `mean` is fitted on episodes the score does not cover, so the comparison is not
    # circular.
    baseline_chunks = []
    for episode in range(low, high):
        for frame_index in frames[:4]:
            baseline_chunks.append(
                expert_chunk(dataset, episode, int(frame_index), args.chunk)
            )
    mean_action = np.concatenate(baseline_chunks, axis=0).mean(axis=0)
    print(f"`mean` baseline fitted on episodes {low}..{high - 1} "
          f"({len(baseline_chunks)} chunks)")

    config: dict = {"chunk": args.chunk}
    if args.checkpoint != BASE_REPO:
        config["ckpt"] = args.checkpoint
    model = adapter.load(config)
    print(f"checkpoint  : {model['repo']}")
    print(f"statistics  : {model['stats_root']}")
    print(f"self-contained: {model['self_contained']}\n")

    sums = {"model": 0.0, "hold": 0.0, "mean": 0.0}
    block_errors = {name: 0.0 for name, _, _ in BLOCKS}
    block_absolute = {name: 0.0 for name, _, _ in BLOCKS}
    per_sample: list[tuple[int, int, float, float]] = []
    count = 0
    prompts_seen: set[str] = set()

    for index, (episode, frame_index) in enumerate(plan, start=1):
        frame = dict(dataset[episode * EPISODE_LENGTH + frame_index])
        arrays = to_observation(frame)
        expert = expert_chunk(dataset, episode, frame_index, args.chunk)
        # The dataset carries the prompt each frame was recorded under, so read it rather
        # than using the constant -- and collect them, so a dataset whose frames disagree
        # about their task is visible instead of averaged into a single assumption.
        task = str(frame.get("task", TRAINING_PROMPT))
        prompts_seen.add(task)
        request = _Request(arrays, task, args.chunk)
        predicted = adapter.infer(model, request)["actions"]

        current = arrays["observation.state"]
        hold = np.broadcast_to(current, expert.shape)

        errors = {
            "model": float(np.mean((predicted - expert) ** 2)),
            "hold": float(np.mean((hold - expert) ** 2)),
            "mean": float(np.mean((mean_action[: expert.shape[0]] - expert) ** 2)),
        }
        for key, value in errors.items():
            sums[key] += value
        per_sample.append((episode, frame_index, errors["model"], errors["hold"]))

        for name, start, end in BLOCKS:
            span = slice(start, end)
            block_errors[name] += float(np.mean((predicted[:, span] - expert[:, span]) ** 2))
            block_absolute[name] += float(np.abs(predicted[:, span] - expert[:, span]).mean())
        count += 1

        if index % 24 == 0 or index == len(plan):
            print(f"  {index:>4}/{len(plan)}")

    print(f"\nmean squared error over {count} chunks of {args.chunk} steps x 16 joints")
    print("(radians^2 for joints, opening^2 for grippers)\n")
    print(f"  {'predictor':<10} {'mse':>10}   relative to `mean`")
    for name, total in sorted(sums.items(), key=lambda item: item[1]):
        mse = total / count
        print(f"  {name:<10} {mse:>10.5f}   {mse / (sums['mean'] / count):>6.2f}x")

    print("\nper-block for the model:")
    for name, _, _ in BLOCKS:
        print(
            f"  {name:<24} mse {block_errors[name] / count:>9.5f}   "
            f"mae {block_absolute[name] / count:>7.4f}"
        )
    if len(prompts_seen) > 1:
        print(f"\nNOTE: the scored frames carry {len(prompts_seen)} different task strings; "
              "the model was evaluated on all of them")

    worst = sorted(per_sample, key=lambda item: item[2], reverse=True)[:3]
    print("\nworst chunks (episode, frame, model mse, hold mse):")
    for episode, frame_index, model_error, hold_error in worst:
        print(f"  ep {episode:>3} frame {frame_index:>3}   {model_error:>8.5f}   {hold_error:>8.5f}")

    model_mse = sums["model"] / count
    hold_mse = sums["hold"] / count
    mean_mse = sums["mean"] / count
    print("\n" + "=" * 68)
    print(f"  model {model_mse:.5f} | hold {hold_mse:.5f} | mean {mean_mse:.5f}")
    if model_mse < hold_mse and model_mse < mean_mse:
        print(f"  the model beats both ({mean_mse / model_mse:.2f}x over `mean`, "
              f"{hold_mse / model_mse:.2f}x over `hold`)")
        print("  -> the checkpoint is predicting, not collapsing.")
    elif model_mse >= mean_mse:
        print("  the model does NOT beat `mean` -- it is close to a constant predictor")
        print("  -> expect the closed-loop run to move to the dataset's average pose and stay")
    else:
        print("  the model beats `mean` but not `hold`")
        print("  -> it uses its input, but not as well as not moving would be.")
    print("\n  These observations are training data, so a good score is necessary and not")
    print("  sufficient.  The GR00T fine-tune scored 48x better than `mean` here and still")
    print("  transferred no cookies: what it learned was a hover, which this cannot see.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
