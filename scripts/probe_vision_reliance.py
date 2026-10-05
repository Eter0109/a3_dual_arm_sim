"""How much does this checkpoint depend on each input, in radians?

The information probe showed the wrist camera carries what the task needs (a linear decoder
on state + wrist reaches 0.0061 rad, under the 0.0075 the expert fails at) while the
policy sits at 0.0185 -- the same number a state-only lookup achieves.  That is consistent
with "the policy ignores the camera", but consistent is not the same as measured.

This measures it.  Each observation is rebuilt with one input at a time replaced by an
unrelated frame's, and the error against the expert's action is reported per variant:

    true        error the checkpoint normally gets
    wrist off   wrist image borrowed from another episode
    front off   front image borrowed
    both off    both borrowed
    state off   joint angles borrowed -- the control, and the scale

A checkpoint that reads the wrist camera must get *worse* in the `wrist off` row by an
amount comparable to how much better it is than the state-only floor.  A flat column is a
policy that never learned to look, and running the same table at several checkpoints turns
"undersized run" into a curve rather than a guess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_action_baselines import DEFAULT_TASK, build  # noqa: E402

VARIANTS = ("true", "wrist off", "front off", "both off", "state off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-episode", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=str(args.root), video_backend="pyav")
    torch, policy, preprocess, postprocess = build(args.ckpt, dataset)

    table = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "frame_index"])
         for f in sorted(args.root.glob("data/**/*.parquet"))],
        ignore_index=True,
    )
    episodes = table["episode_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    images = [k for k in dict(dataset[0]) if k.startswith("observation.images")]

    # Sampled frames, spread over episodes so the scale is not one arm pose.
    chosen: list[int] = []
    for offset in range(args.episodes):
        episode = args.first_episode + offset
        mask = episodes == episode
        if not mask.any():
            break
        first = int(frames[mask].min())
        length = int(mask.sum())
        step = max(1, length // (args.frames + 1))
        chosen.extend(first + step * (i + 1) for i in range(args.frames))
    chosen = [i for i in chosen if i < len(episodes)]

    # The donor frame is half the dataset away, which lands in a different episode and so
    # in an unrelated arm pose.  `+ len//2` rather than a random draw keeps every compare
    # reproducible without carrying a seed through the loop.
    donor = [(i + len(episodes) // 2) % len(episodes) for i in chosen]

    def predict(frame: dict) -> np.ndarray:
        keep = {k: v for k, v in frame.items()
                if k not in ("task", "timestamp", "frame_index", "episode_index",
                             "index", "task_index", "action")}
        batch = preprocess({**keep, "task": DEFAULT_TASK})
        with torch.no_grad():
            draws = [postprocess(policy.predict_action_chunk(batch)).float().cpu().numpy()
                     for _ in range(max(1, args.samples))]
        return np.stack([(d[0] if d.ndim == 3 else d)[0] for d in draws]).mean(0)

    errors: dict[str, list[float]] = {name: [] for name in VARIANTS}
    for index, source in zip(chosen, donor):
        frame = dict(dataset[index])
        other = dict(dataset[source])
        expert = np.asarray(frame["action"], dtype=np.float64)
        width = expert.shape[0]

        def score(variant: str, obs: dict) -> None:
            predicted = predict(obs)[:width]
            errors[variant].append(float(np.abs(predicted - expert).max()))

        score("true", frame)
        score("wrist off", {**frame, **{k: other[k] for k in images if "wrist" in k}})
        score("front off", {**frame, **{k: other[k] for k in images if "wrist" not in k}})
        score("both off", {**frame, **{k: other[k] for k in images}})
        score("state off", {**frame, "observation.state": other["observation.state"]})

    label = args.label or Path(args.ckpt).parent.parent.name
    print(f"\n{label}    ({len(chosen)} 帧, 每帧 {args.samples} 次采样)")
    base = float(np.mean(errors["true"]))
    for name in VARIANTS:
        values = np.array(errors[name])
        delta = values.mean() - base
        print(f"    {name:<10} 误差 {values.mean():.4f} rad   median {np.median(values):.4f}"
              f"   Δ {delta:+.4f}")
    print(f"    参照: 状态+腕部线性探针 0.0061 · 只有状态 0.0189 · 专家失败门槛 0.0075")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
