#!/usr/bin/env python3
"""Copy whole episodes out of a dataset, selected by a substring of their instruction.

Needed because the 20-cookie collection draws `first_grasp` per episode, so the
plan prompt varies: "Grasp 5 cookies first, then 5 ..." is one of eleven strings.  Of the
280 episodes collected, 30 happen to be the 5-5-5-5 plan -- the one whose action pattern
is *one* constant sequence, and therefore the only one that can be taught under a single
constant instruction without the model having to infer a plan from the picture.

`aggregate_datasets` concatenates whole datasets, so an episode-level subset needs a new
dataset rather than a filter over an existing one; this writes that subset.  Camera
streams are decoded with one ffmpeg call per camera per episode (random per-frame access
through the LeRobot reader costs ~150 ms/frame, this costs ~2 ms).

    python scripts/filter_dataset_by_task.py \\
        --source /root/autodl-tmp/datasets/a3_2x10_v1/shard_01 \\
        --out /root/autodl-tmp/datasets/a3_2x10_n5/shard_01 \\
        --repo-id local/a3-2x10-n5-01 --contains "Grasp 5 cookies first"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_dual_arm_sim.contracts import EpisodeContext  # noqa: E402
from a3_dual_arm_sim.recording import LeRobotV3Recorder  # noqa: E402

CAMERAS = ("front", "left_wrist", "right_wrist")
IMAGE_KEY = "observation.images.{}"
SCALAR_COLUMNS = (
    "observation.state",
    "observation.velocity",
    "observation.eef_pose",
    "observation.force",
    "action",
)
FPS = 20
SIZE = 256


def load_scalars(root: Path, wanted: set[int]) -> dict[str, np.ndarray]:
    """Every non-camera column, restricted to the episodes being kept."""

    import pyarrow.parquet as pq

    tables = []
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        data = pq.read_table(path, columns=list(SCALAR_COLUMNS) + ["episode_index"])
        episodes = np.asarray(data["episode_index"].to_numpy())
        mask = np.isin(episodes, list(wanted))
        tables.append(data.filter(mask))
    combined = __import__("pyarrow").concat_tables(tables)
    return {
        name: np.stack(combined[name].to_numpy()).astype(np.float32)
        for name in SCALAR_COLUMNS
    }, np.asarray(combined["episode_index"].to_numpy())


def decode_episode(root: Path, episode: dict, start: int, end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        key = f"videos/{IMAGE_KEY.format(camera)}"
        path = (
            root / "videos" / IMAGE_KEY.format(camera)
            / f"chunk-{episode[f'{key}/chunk_index']:03d}"
            / f"file-{episode[f'{key}/file_index']:03d}.mp4"
        )
        command = [
            "ffmpeg", "-v", "error",
            "-ss", f"{episode[f'{key}/from_timestamp'] + start / FPS:.6f}",
            "-to", f"{episode[f'{key}/from_timestamp'] + end / FPS:.6f}",
            "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        raw = subprocess.run(command, capture_output=True, check=True).stdout
        count, frame_bytes = end - start, SIZE * SIZE * 3
        if len(raw) < count * frame_bytes:
            raise SystemExit(
                f"episode {episode['episode_index']} {camera}: decoded "
                f"{len(raw) // frame_bytes}/{count} frames"
            )
        out[IMAGE_KEY.format(camera)] = np.frombuffer(
            raw[: count * frame_bytes], dtype=np.uint8
        ).reshape(count, SIZE, SIZE, 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--contains", required=True, help="Substring of the instruction")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.source / "a3_episode_metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    keep = [r for r in records if args.contains in r["task"]]
    if not keep:
        print(f"nothing in {args.source.name} matches {args.contains!r}")
        return 0
    print(f"{args.source.name}: {len(keep)} of {len(records)} episodes match", flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=str(args.source), video_backend="pyav")
    by_index = {int(e["episode_index"]): e for e in dataset.meta.episodes}
    wanted = {int(r["episode_index"]) for r in keep}
    scalars, scalar_episodes = load_scalars(args.source, wanted)

    # Row lookup: (episode, frame-within-episode) -> index into the scalar arrays.  Rows
    # arrive in dataset order, so a per-episode cursor reconstructs the within-episode
    # frame index without trusting the `frame_index` column.
    counts = {int(e): 0 for e in wanted}
    lookup: dict[tuple[int, int], int] = {}
    for row, episode in enumerate(scalar_episodes):
        key = int(episode)
        lookup[(key, counts[key])] = row
        counts[key] += 1

    recorder = LeRobotV3Recorder(
        args.out, repo_id=args.repo_id, fps=FPS, image_height=SIZE, image_width=SIZE,
        action_mode="joint_position", use_videos=True, rollover_mb=500.0,
    )
    written = frames_written = 0
    for record in keep:
        index = int(record["episode_index"])
        episode = by_index[index]
        count = int(record["frames"])
        cameras = decode_episode(args.source, episode, 0, count)
        context = EpisodeContext(
            seed=index, task=record["task"], action_mode="joint_position"
        )
        recorder.start_episode(context, record.get("controller_type", "unknown"), "joint_position")
        for offset in range(count):
            row = lookup[(index, offset)]
            observation = {
                IMAGE_KEY.format(camera): cameras[IMAGE_KEY.format(camera)][offset]
                for camera in CAMERAS
            }
            for name in SCALAR_COLUMNS:
                if name != "action":
                    observation[name] = scalars[name][row]
            recorder.add_frame(observation, scalars["action"][row])
        recorder.finish_episode(success=record.get("success"), metrics=record.get("metrics"))
        written += 1
        frames_written += count
        print(f"  [{written}/{len(keep)}] src ep {index}: {count} frames", flush=True)

    recorder.close()
    print(f"\nwrote {written} episodes / {frames_written} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
