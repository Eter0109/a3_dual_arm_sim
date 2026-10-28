#!/usr/bin/env python3
"""Build the clean v9 base dataset: expert-only, balanced, uniformly prompted.

Why a new dataset rather than another sampler filter
----------------------------------------------------
Every arm so far inherited `a3_b2_main_aligned_v2_merged` whole, and that set carries a
defect the sampler had to work around: 500 of its 1200 2x5 episodes are 302 frames long
and deliver **one** batch under an instruction that asks for two.  That is fixable with
`A3_MIN_EPISODE_FRAMES`, but a filter cannot fix the other two problems:

* the 2x10 plan prompts are wildly unbalanced -- 190 of 440 episodes are the `n=5` plan
  and the rarest has 20.  Evaluation draws the plan uniformly, so most of what is tested
  was almost never trained;
* `a3_graft_v1_merged` is half policy rollouts.  55% of its frames are the model imitating
  its own stopping behaviour, which is the behaviour this line is trying to remove.

So this writes a set with none of those properties, to serve as a clean base rather than
as another arm.

What it selects
---------------
===========================================  =========  ============
source                                        episodes   instrument
===========================================  =========  ============
`a3_b2_main_aligned_v2_merged`, frames >= 400   200/500   main's 2x5 sentence
`a3_2x10_n5` + `a3_2x10_v1`, per plan            20 x 10   that episode's own plan
===========================================  =========  ============

200 : 200 keeps the two tasks at 50 % of episodes by construction, and 20 per plan keeps
the ten plan prompts uniform -- which is what the benchmark samples.  Episodes are taken
at even spacing across the candidate list rather than as a prefix, so the selection spans
the whole seed range instead of one shard.

Frames are not balanced (2x10 episodes are ~1067 long against ~549), so the gradient sees
roughly 32 : 68.  That is deliberate and is worth stating: 50 % of *demonstrations* and
50 % of *frames* cannot both hold when one task's episodes are twice as long, and the
brief asked for the former.

Instructions
------------
Every string is one main's benchmark produces, so training and evaluation agree by
construction and no relabelling step is needed:

* 2x5 -- `Transfer 10 cookies from source column 1 into the target box in two batches of
  five.` (`--prompt-style task` for task 2x5)
* 2x10 -- the per-episode plan sentence, read from the episode's own metadata
  (`--prompt-style task` for task 2x10, `first_grasp: random`)

    python scripts/build_clean_v9.py --out .../a3_clean_v9 --repo-id local/a3-clean-v9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_dataset_by_task import (  # noqa: E402
    CAMERAS,
    FPS,
    IMAGE_KEY,
    SCALAR_COLUMNS,
    SIZE,
    decode_episode,
    load_scalars,
)

#: main's 2x5 instruction, verbatim (`EpisodeExecution._episode_task`, column 1).
TWO_BY_FIVE_PROMPT = (
    "Transfer 10 cookies from source column 1 into the target box in two batches of five."
)


def plan_of(instruction: str) -> str:
    """Which of the ten plan prompts an episode was collected under.

    The plan is `[n, 10-n, n, 10-n]` and `n` is drawn per episode, so it is a property of
    the episode and not of the task.  The skip variant has no `n` at all.
    """

    if "Skip grasp" in instruction:
        return "skip"
    match = re.search(r"Grasp (\d+) cookies first", instruction)
    return f"n={match.group(1)}" if match else "unknown"


def episodes_of(root: Path) -> list[dict]:
    path = root / "a3_episode_metadata.jsonl"
    if not path.is_file():
        raise SystemExit(f"{root} has no a3_episode_metadata.jsonl")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evenly(items: list, count: int) -> list:
    """`count` items at even spacing across `items`, endpoints included.

    A prefix would take one shard's worth of seeds; the plan prompts draw their `n` from
    the seed, so the spacing has to span the whole range or the selection is skewed.
    """

    if count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(indices.tolist()))]


def select_two_by_five(root: Path, episodes: int, min_frames: int) -> list[tuple[Path, dict, str]]:
    records = episodes_of(root)
    long_enough = [r for r in records if int(r["frames"]) >= min_frames]
    print(
        f"  2x5 {root.name}: {len(records)} episodes, {len(long_enough)} have >= {min_frames} "
        f"frames (the rest end after one batch under an instruction that asks for two)",
        flush=True,
    )
    chosen = evenly(sorted(long_enough, key=lambda r: int(r["episode_index"])), episodes)
    return [(root, record, TWO_BY_FIVE_PROMPT) for record in chosen]


def select_two_by_ten(
    roots: list[Path], per_plan: int
) -> list[tuple[Path, dict, str]]:
    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for root in roots:
        for record in episodes_of(root):
            instruction = (record.get("randomization") or {}).get("task_instruction")
            if not instruction:
                raise SystemExit(
                    f"{root.name}/ep {record['episode_index']} has no recorded plan prompt"
                )
            grouped[plan_of(instruction)].append((root, record))

    print(f"  2x10: {sum(len(v) for v in grouped.values())} episodes over {len(grouped)} plans",
          flush=True)
    out: list[tuple[Path, dict, str]] = []
    for plan in sorted(grouped):
        candidates = grouped[plan]
        chosen = evenly(candidates, per_plan)
        print(f"       {plan:5s} {len(candidates):4d} available -> {len(chosen):3d} kept",
              flush=True)
        for root, record in chosen:
            out.append((root, record, record["randomization"]["task_instruction"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--two-by-five", type=Path,
        default=Path("/root/autodl-tmp/datasets/a3_b2_main_aligned_v2_merged"),
    )
    parser.add_argument("--two-by-five-episodes", type=int, default=200)
    parser.add_argument("--min-frames", type=int, default=400)
    parser.add_argument(
        "--two-by-ten", type=Path, nargs="+",
        default=[
            Path("/root/autodl-tmp/datasets/a3_2x10_n5"),
            Path("/root/autodl-tmp/datasets/a3_2x10_v1"),
        ],
        help="Directories whose shard_*/subdirectories hold the 2x10 collections.",
    )
    parser.add_argument("--per-plan", type=int, default=20)
    args = parser.parse_args()

    from a3_dual_arm_sim.collection import LeRobotV3Recorder
    from a3_dual_arm_sim.contracts import EpisodeContext
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ten_roots = sorted(
        child for parent in args.two_by_ten for child in parent.glob("shard_*") if child.is_dir()
    )
    print(f"sources: 2x5 {args.two_by_five.name}, 2x10 {len(ten_roots)} shards", flush=True)

    selection = select_two_by_five(args.two_by_five, args.two_by_five_episodes, args.min_frames)
    selection += select_two_by_ten(ten_roots, args.per_plan)

    by_root: dict[Path, list[tuple[dict, str]]] = defaultdict(list)
    for root, record, instruction in selection:
        by_root[root].append((record, instruction))

    total = len(selection)
    expected = args.two_by_five_episodes + args.per_plan * 10
    if total < expected:
        raise SystemExit(f"selected {total} episodes, expected {expected}")

    recorder = LeRobotV3Recorder(
        args.out, repo_id=args.repo_id, fps=FPS, image_height=SIZE, image_width=SIZE,
        action_mode="joint_position", use_videos=True, rollover_mb=500.0,
    )

    written = frames_written = 0
    for root in sorted(by_root):
        picks = by_root[root]
        dataset = LeRobotDataset(args.repo_id, root=str(root), video_backend="pyav")
        by_index = {int(e["episode_index"]): e for e in dataset.meta.episodes}
        wanted = {int(record["episode_index"]) for record, _ in picks}
        scalars, scalar_episodes = load_scalars(root, wanted)

        # Rows arrive in dataset order, so a per-episode cursor reconstructs the
        # within-episode frame index without trusting the `frame_index` column.
        counts = {index: 0 for index in wanted}
        lookup: dict[tuple[int, int], int] = {}
        for row, episode in enumerate(scalar_episodes):
            key = int(episode)
            lookup[(key, counts[key])] = row
            counts[key] += 1

        for record, instruction in picks:
            index = int(record["episode_index"])
            episode = by_index[index]
            count = int(record["frames"])
            cameras = decode_episode(root, episode, 0, count)
            context = EpisodeContext(seed=index, task=instruction, action_mode="joint_position")
            recorder.start_episode(
                context, record.get("controller_type", "expert"), "joint_position"
            )
            for offset in range(count):
                observation = {
                    IMAGE_KEY.format(camera): cameras[IMAGE_KEY.format(camera)][offset]
                    for camera in CAMERAS
                }
                for name in SCALAR_COLUMNS:
                    if name != "action":
                        observation[name] = scalars[name][lookup[(index, offset)]]
                recorder.add_frame(observation, scalars["action"][lookup[(index, offset)]])
            recorder.finish_episode(
                success=record.get("success"), metrics=record.get("metrics")
            )
            written += 1
            frames_written += count
            if written % 25 == 0 or written == total:
                print(f"  [{written}/{total}] {frames_written} frames", flush=True)

    recorder.close()
    print(f"\nwrote {written} episodes / {frames_written} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
