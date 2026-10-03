"""Teacher forcing: can the model reproduce the expert at all, given true observations?

Every measurement so far says the policy fails in closed loop, and that leaves two very
different explanations calling for opposite work:

1. **The input path is wrong.**  The model is fed something it was not trained on -- a
   camera, an encoding, a convention -- so its predictions are unrelated to the expert's
   and no amount of data fixes it.
2. **Compounding error.**  The prediction tracks the expert's next action well from the
   expert's own state, but once the model takes a step of its own the state leaves the
   training distribution and the errors accumulate.  That is the ordinary failure of
   behaviour cloning, and its remedies are more data, more diversity, or DAgger.

Holding the observations to ground truth separates them: step through one recorded
episode, feed the model the dataset's own images and state at every frame, and compare
the action it would take against the one the expert took.

Inference goes through the adapter's own ``infer`` rather than a reimplementation, so the
answer describes the path the evaluation actually uses -- same projection, same prompt,
same discretized state.  A hand-built batch here could differ from production in exactly
the way this probe exists to detect.

Two reference points are printed beside the error, because a raw number in radians means
nothing alone: the per-step motion the demonstration itself commands over the same frames
(an error below that is not something the arm could act on), and how often the prediction
is the *previous* action repeated -- the signature of a policy that has stopped attending
to the observation and is replaying a default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, "/root/autodl-tmp/a3-collect")

from a3_dual_arm_sim.bridge.manifest import load_manifest  # noqa: E402

MANIFESTS = {
    "b1": Path("/root/a3_dual_arm_sim/configs/policies/pi05_a3_finetuned.yaml"),
    "adv": Path("/root/a3_dual_arm_sim/configs/policies/pi05_a3_adv40k.yaml"),
    # The un-frozen continuation of `adv`, on the same data.  Present so the comparison is
    # one flag apart: `--which adv` against `--which unfreeze3k` differs in the weights and
    # in nothing else the probe can see.
    "unfreeze3k": Path("/root/a3_dual_arm_sim/configs/policies/pi05_a3_adv40k_unfreeze3k.yaml"),
}
DATASETS = {
    "b1": ("local/a3", Path("/root/autodl-tmp/gr00t-runs/b1/merged")),
    "adv": (
        "local/a3_cookie_adv_1000",
        Path("/root/autodl-tmp/datasets/a3_cookie_adv_1000_merged"),
    ),
    "unfreeze3k": (
        "local/a3_cookie_adv_1000",
        Path("/root/autodl-tmp/datasets/a3_cookie_adv_1000_merged"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--which", choices=sorted(MANIFESTS), default="adv")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many consecutive episodes to score, starting at --episode",
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--limit", type=int, default=45, help="frames per episode; 0 for all")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "Inferences per frame.  Above one, the probe also reports how much the "
            "model's own samples disagree with each other -- which is what separates "
            "imprecision from bias, and they call for different work."
        ),
    )
    parser.add_argument(
        "--ablate-vision",
        action="store_true",
        help=(
            "Replace every camera frame with a constant before inference.  A model that "
            "reads its images cannot track the expert without them, so the change in "
            "error measures how much of the prediction the pictures are carrying -- and "
            "if blanking them costs almost nothing, the bottleneck is not the images."
        ),
    )
    return parser.parse_args()


class _Request:
    """The two accessors the adapter reads; see ``adapters/pi05_a3.py``."""

    def __init__(self, arrays: dict[str, np.ndarray], task: str) -> None:
        self._arrays, self.task, self.chunk = arrays, task, 50

    def image(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def vector(self, name: str) -> np.ndarray:
        return self._arrays[name]


def main() -> int:
    args = parse_args()
    import pandas as pd

    from adapters import pi05_a3  # type: ignore[import-not-found]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = load_manifest(MANIFESTS[args.which])
    repo_id, root = DATASETS[args.which]
    checkpoint = ""
    for token in manifest.transport.command:
        if str(token).startswith("ckpt="):
            checkpoint = str(token).split("=", 1)[1]
    if not checkpoint:
        raise SystemExit("the manifest names no checkpoint")

    print(f"model      : {manifest.name}")
    print(f"checkpoint : {checkpoint}")
    print(f"dataset    : {repo_id}")
    print()

    model = pi05_a3.load({"ckpt": checkpoint, "device": "cuda"})
    dataset = LeRobotDataset(repo_id, root=str(root), video_backend="pyav")

    table = pd.concat(
        [
            pd.read_parquet(f, columns=["episode_index", "frame_index"])
            for f in sorted(root.glob("data/**/*.parquet"))
        ],
        ignore_index=True,
    )
    episode_indices = table["episode_index"].to_numpy()
    frame_indices = table["frame_index"].to_numpy()

    # Across several episodes, not one.  An episode's opening is the easy part -- the arm
    # starts from the same pose every time and the first motion is a memorised approach --
    # so a single-episode sample flatters any model, and the point of this probe is the
    # opposite of flattering it.
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

    # The scene's own prompt rather than the dataset's ``task`` column: the column holds
    # the expert's own label ("Transfer 10 cookies from source column 3..."), which is not
    # the string either model was conditioned on.  The evaluation sends the scene prompt,
    # so this must too or the probe measures a different question than the closed loop.
    from a3_dual_arm_sim.cli import _default_task

    task = _default_task("a3_cookie_same_column")
    print(
        f"episodes: "
        + ", ".join(f"{ep} ({length} frames)" for ep, length in per_episode)
        + f"  --  scoring {len(indices)} frames at stride {args.stride}"
    )
    print(f"task string: {task[:88]}")
    print()

    errors: list[float] = []
    motions: list[float] = []
    worst: tuple[float, int] | None = None
    signed_errors: list[np.ndarray] = []
    absolute_errors: list[np.ndarray] = []
    # Filled only when several samples are drawn per frame.  The comparison they exist
    # for: an error that is mostly the model's own sampling spread is imprecision, and
    # averaging or sharpening addresses it; an error that survives averaging is bias,
    # and it would mean the model is aiming at the wrong action rather than a nearly
    # right one.
    single_sample_errors: list[float] = []
    averaged_errors: list[float] = []
    spreads: list[float] = []
    chunk_position_errors: list[np.ndarray] = []

    for index in indices:
        frame = dict(dataset[index])
        # Keyed by the manifest's own names (``front_view``), not by the dataset's
        # (``observation.images.front``): the adapter looks observations up by the name
        # the manifest gave them, which is what the gateway feeds it.
        arrays: dict[str, np.ndarray] = {}
        for name, spec in manifest.observation.images.items():
            image = frame[spec.source].numpy()
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            arrays[name] = (image * 255.0).clip(0, 255).astype(np.uint8)
        for name, spec in manifest.observation.vectors.items():
            arrays[name] = np.asarray(frame[spec.source], dtype=np.float64)
        if args.ablate_vision:
            arrays.update({name: np.zeros_like(value) for name, value in arrays.items()
                           if name in manifest.observation.images})

        request = _Request(arrays, task)
        # The whole chunk, not just its first row.  Where the error sits inside the
        # chunk is what decides the execution length: an error that is flat in the
        # index means the model is equally good at every horizon and the remedy is
        # precision; an error that grows means the far end of the chunk is guesswork,
        # and the remedy is to execute fewer of the rows the model predicts.  Reading
        # only row 0 cannot tell those apart, and row 0 is the one the arm acts on
        # first, so it is also the one that flatters the model most.
        draws = np.stack(
            [
                np.asarray(pi05_a3.infer(model, request)["actions"], dtype=np.float64)
                for _ in range(max(1, args.samples))
            ]
        )
        predicted = draws[0][0]
        expert = np.asarray(frame["action"], dtype=np.float64)

        # The chunk the expert would have produced from this frame: the recorded actions
        # at this step and the ones after it, so index ``k`` compares like with like.
        # Frames past the end of this episode are dropped rather than pulled from the
        # next one, whose arm starts somewhere else.
        horizon = min(draws.shape[2], int(manifest.action.chunk))
        following = [
            i
            for i in range(int(index), int(index) + horizon)
            if i < len(dataset) and episode_indices[i] == episode_indices[index]
        ]
        expert_chunk = np.stack(
            [np.asarray(dict(dataset[i])["action"], dtype=np.float64) for i in following]
        )
        # One vector per frame, one entry per chunk position, so the average across
        # frames is an average at each horizon.  Recording a scalar here instead -- the
        # mean over the chunk -- produced a plot whose x axis was the frame index while
        # claiming to be the chunk position, and whose every-``limit`` spike was the
        # first frame of each episode rather than anything about the horizon.
        chunk_position_errors.append(
            np.abs(draws[0][: len(following)] - expert_chunk).max(axis=1)
        )

        if len(draws) > 1:
            # All three statistics are the same shape: mean over frames of the per-frame
            # max over joints.  Mixing a mean with a max-of-max made averaging look
            # harmful, which it cannot be -- the mean of several samples is by
            # construction no further from their own centre than a single sample is.
            window = draws[:, : len(following)]
            averaged = window.mean(axis=0)
            single_sample_errors.append(float(np.abs(window - expert_chunk).max(axis=2).mean()))
            averaged_errors.append(float(np.abs(averaged - expert_chunk).max(axis=1).mean()))
            spreads.append(float(np.abs(window - averaged).max(axis=2).mean()))

        delta = np.abs(predicted - expert)
        errors.append(float(delta.max()))
        signed_errors.append(predicted - expert)
        absolute_errors.append(delta)
        if worst is None or errors[-1] > worst[0]:
            worst = (errors[-1], index)

        # The expert's own *single-step* motion, so the error has a same-unit yardstick.
        # Reading it from the sampled frames instead would measure ``stride`` steps of
        # travel and inflate the reference by that factor -- the comparison this exists to
        # make would then flatter the model by exactly that much.  Frames at an episode
        # boundary are skipped: the previous index belongs to the episode before, and the
        # jump between two episodes is not motion the policy was ever asked to make.
        previous_episode = episode_indices[index - 1] if index > 0 else -1
        if index > 0 and previous_episode == episode_indices[index]:
            neighbour = np.asarray(dict(dataset[index - 1])["action"], dtype=np.float64)
            motions.append(float(np.abs(expert - neighbour).max()))

    errors_arr = np.array(errors)
    motions_arr = np.array(motions)
    signed_errors = np.stack(signed_errors)
    absolute_errors = np.stack(absolute_errors)

    print(f"  |predicted - expert| over {len(errors_arr)} frames, max over 16 joints (rad)")
    print(
        f"    mean {errors_arr.mean():.5f}   median {np.median(errors_arr):.5f}   "
        f"p90 {np.percentile(errors_arr, 90):.5f}   max {errors_arr.max():.5f}"
    )
    print()
    print("  the demonstration's own per-step motion over the same frames (rad)")
    print(
        f"    mean {motions_arr.mean():.5f}   median {np.median(motions_arr):.5f}   "
        f"p90 {np.percentile(motions_arr, 90):.5f}   max {motions_arr.max():.5f}"
    )
    print()
    print(f"  error / demonstration motion = {errors_arr.mean() / motions_arr.mean():.2f}")
    if worst:
        print(f"  worst frame: index {worst[1]}, error {worst[0]:.5f} rad")
    print()

    if chunk_position_errors:
        width = min(len(v) for v in chunk_position_errors)
        chunk = np.mean([v[:width] for v in chunk_position_errors], axis=0)
        peak = float(chunk.max())
        print(f"  error by position inside the predicted chunk (rad, {width} rows)")
        for k in range(width):
            bar = "#" * int(40 * chunk[k] / max(peak, 1e-9))
            print(f"    row {k:3d}   {chunk[k]:.5f}  {bar}")
        print(
            f"    row 0 {chunk[0]:.5f}  ->  row {width - 1} {chunk[-1]:.5f}   "
            f"growth x{chunk[-1] / max(chunk[0], 1e-9):.2f}"
        )
        print()

    if spreads:
        spread = np.array(spreads)
        single = np.array(single_sample_errors)
        averaged = np.array(averaged_errors)
        print(f"  {args.samples} samples per frame, over {len(spread)} frames (rad)")
        print(f"    one sample vs expert      : mean {single.mean():.5f}")
        print(f"    mean of samples vs expert : mean {averaged.mean():.5f}")
        print(f"    spread across samples     : mean {spread.mean():.5f}")
        print(
            f"    -> averaging removes {100 * (1 - averaged.mean() / single.mean()):.0f} % of the "
            f"single-sample error; the rest ({averaged.mean():.5f}) does not move with the "
            "noise and is bias"
        )
        print()

    # Per-joint breakdown.  The distinction that matters: an error that is *signed and
    # stable* across frames is a bias, and a bias on one joint is a convention or a
    # wiring mismatch -- fixable, and not a statement about the model's ability.  An
    # error whose magnitude is stable but whose sign wanders is imprecision, which is
    # what more training or more data would address.  Averaging the two together hides
    # which one this is.
    names = [f"L{i}" for i in range(7)] + ["L-grip"] + [f"R{i}" for i in range(7)] + ["R-grip"]
    print("  per joint: mean signed error (bias) and mean absolute error")
    print("    joint    bias      |err|     |err|/motion")
    for joint, name in enumerate(names):
        signed = float(np.mean(signed_errors[:, joint]))
        absolute = float(np.mean(absolute_errors[:, joint]))
        motion = float(np.mean(motions_arr)) or 1.0
        flag = "  <-- large" if absolute > 0.25 * motions_arr.mean() else ""
        print(f"    {name:8s} {signed:+.5f}  {absolute:.5f}     {absolute / motion:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
