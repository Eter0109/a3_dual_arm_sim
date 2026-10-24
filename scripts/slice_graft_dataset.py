#!/usr/bin/env python3
"""Rebuild the grafted collection as a dataset with a chosen whole/handover mix.

Why this exists
---------------
The 200-episode graft dataset is half policy, half expert.  Training more steps on it
made the model a *better imitator of its own first batch* -- 15k steps scored 4.50 on
main's randomised benchmark against 5.10 for 2000, and the 15k arm stopped at exactly
five Cookies in seven of ten episodes.  That is the composition leaking into the
gradient: every episode's first half is the 4.05 policy doing what it does, which is
mostly "stop at five".

Two facts make the fix a data edit rather than a training-loop change:

* pi0.5 is not history-conditioned -- `n_obs_steps = 1`, one observation per sample --
  so masking the loss on pre-handover frames and dropping those frames are the same
  gradient, and dropping is the cheaper of the two.  There is no state being carried
  from the first phase into the second.
* the handover is recorded per episode (`graft_policy_steps` in
  `a3_episode_metadata.jsonl`), so the cut point is already known.

What it writes
--------------
Episodes of two kinds, interleaved in the order given:

* ``full``   -- an entire episode, unchanged.  The anchor: without any first-batch
                frames there is nothing holding the model's existing first-batch
                competence, and the 15k arm is what that looks like.
* ``expert`` -- the episode from the handover (plus a margin) to the end.  Its first
                frame is the observation the policy itself produced, so the sample is
                in the deployment distribution by construction.

The fraction of frames that are post-handover is the knob: with `--full-episodes 50`
over 167 tier-A episodes it goes from 42% (the collection as recorded) to ~79%.

Reading is the slow part, so both halves are done the cheap way: the scalar columns
come straight out of the parquet files with pyarrow (no video), and the three camera
streams are decoded by ffmpeg, one call per camera per episode, seeking to the
episode's own timestamp range.  Per-frame random access through the LeRobot dataset
costs ~150 ms/frame here (each access re-seeks the container); this costs ~2 ms.

    python scripts/slice_graft_dataset.py --out <root> --repo-id local/x \\
        --full-episodes 50 --smoke 5          # writes 5 episodes and stops
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_dual_arm_sim.contracts import EpisodeContext  # noqa: E402
from a3_dual_arm_sim.recording import LeRobotV3Recorder  # noqa: E402
from a3_dual_arm_sim.tasks import TASKS  # noqa: E402

CAMERAS = ("front", "left_wrist", "right_wrist")
IMAGE_KEY = "observation.images.{}"
SCALAR_COLUMNS = (
    "observation.state",
    "observation.velocity",
    "observation.eef_pose",
    "observation.force",
    "action",
)


def load_scalars(root: Path) -> dict[str, np.ndarray]:
    """Every non-camera column of the dataset, in dataset order."""

    import pyarrow.parquet as pq

    files = sorted((root / "data" / "chunk-000").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    table = pq.read_table(files, columns=list(SCALAR_COLUMNS))
    return {
        name: np.stack(table[name].to_numpy()).astype(np.float32)
        for name in SCALAR_COLUMNS
    }


def decode_episode(root: Path, episode: dict, start: int, end: int) -> dict[str, np.ndarray]:
    """Decode frames ``[start, end)`` of one episode's three camera streams.

    ``start``/``end`` are *within-episode* frame indices.  ffmpeg is given the episode's
    own timestamp window and asked for exactly that many frames, so an off-by-one shows
    up as a short read here rather than as a silently shifted trajectory.
    """

    fps = 20.0
    out: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        key = f"videos/{IMAGE_KEY.format(camera)}"
        path = (
            root
            / "videos"
            / IMAGE_KEY.format(camera)
            / f"chunk-{episode[f'{key}/chunk_index']:03d}"
            / f"file-{episode[f'{key}/file_index']:03d}.mp4"
        )
        from_ts = episode[f"{key}/from_timestamp"] + start / fps
        to_ts = episode[f"{key}/from_timestamp"] + end / fps
        command = [
            "ffmpeg", "-v", "error",
            "-ss", f"{from_ts:.6f}", "-to", f"{to_ts:.6f}",
            "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        raw = subprocess.run(command, capture_output=True, check=True).stdout
        count = end - start
        frame_bytes = 256 * 256 * 3
        if len(raw) < count * frame_bytes:
            raise SystemExit(
                f"episode {episode['episode_index']} camera {camera}: decoded "
                f"{len(raw) // frame_bytes} frames, expected {count}"
            )
        out[IMAGE_KEY.format(camera)] = (
            np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8)
            .reshape(count, 256, 256, 3)
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=Path("/root/autodl-tmp/datasets/a3_graft_v1_merged")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--full-episodes",
        type=int,
        default=50,
        help=(
            "How many episodes contribute an anchor segment.  With --anchor full that is "
            "the whole episode; with --anchor prefix it is the first batch only."
        ),
    )
    parser.add_argument(
        "--anchor",
        choices=("full", "prefix"),
        default="full",
        help=(
            "What the anchor segments contain.  `full` keeps whole episodes, which also keeps "
            "the policy's own failed attempt at the second batch; `prefix` stops at the "
            "handover (minus the margin), so the dataset contains no example of releasing "
            "the second batch from above.  Measured: the policy releases at 0.8468 m and the "
            "expert at 0.7596 m, an 87 mm difference, and a mix that kept whole episodes "
            "still taught the higher one."
        ),
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=50,
        help=(
            "Frames to drop after the handover before an `expert` episode starts.  A "
            "sample's target is the next 50 actions, so a sample taken exactly at the "
            "handover has half its target in the previous phase."
        ),
    )
    parser.add_argument("--tier", default="A", help="Only slice this tier ('A', 'any')")
    parser.add_argument("--smoke", type=int, default=0, help="Write this many episodes and stop")
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args()

    meta_dir = args.source / "a3_episode_metadata.jsonl"
    records = [json.loads(line) for line in meta_dir.read_text().splitlines()]
    summary = json.loads((args.source / "collection_summary.json").read_text())

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=str(args.source), video_backend="pyav")
    by_index = {int(e["episode_index"]): e for e in dataset.meta.episodes}

    wanted = []
    for record in records:
        metrics = record.get("metrics", {})
        if args.tier != "any" and metrics.get("graft_tier") != args.tier:
            continue
        index = int(record["episode_index"])
        policy_steps = metrics.get("graft_policy_steps")
        if not isinstance(policy_steps, int):
            continue
        wanted.append((index, policy_steps, record["frames"], metrics))

    if not wanted:
        raise SystemExit("nothing matched")
    print(f"{len(wanted)} tier-{args.tier} episodes, {sum(w[2] for w in wanted)} frames total")

    # The anchor is taken from the front of the list, the cut copies from all of it, so
    # the two halves share episodes and the only difference between them is the window.
    full = wanted[: args.full_episodes]
    plan: list[tuple[str, int, int, int, dict]] = []
    for index, policy_steps, frames, metrics in full:
        if args.anchor == "prefix":
            # Stop before the policy's own attempt at the second batch: its release height
            # is the behaviour being corrected, and leaving it in the data teaches it.
            plan.append(("prefix", index, 0, max(1, policy_steps - args.margin), metrics))
        else:
            plan.append(("full", index, 0, frames, metrics))
    for index, policy_steps, frames, metrics in wanted:
        start = min(policy_steps + args.margin, frames - 1)
        plan.append(("expert", index, start, frames, metrics))
    if args.smoke:
        # Half of each kind, because the two paths fail differently: `full` decodes an
        # entire episode, `expert` seeks into the middle of one and has to land on the
        # right frame.  A smoke test that only ran the first kind would say nothing
        # about the path the whole exercise is for.
        keep_full = args.smoke // 2
        plan = [p for p in plan if p[0] in ("full", "prefix")][:keep_full] + [
            p for p in plan if p[0] == "expert"
        ][: args.smoke - keep_full]

    scalars = load_scalars(args.source)
    task = TASKS["a3_cookie_batch_second_batch_aligned"]
    recorder = LeRobotV3Recorder(
        args.out,
        repo_id=args.repo_id,
        fps=20,
        image_height=256,
        image_width=256,
        action_mode="joint_position",
        use_videos=True,
        rollover_mb=500.0,
    )

    written = 0
    frames_written = 0
    for kind, index, start, end, metrics in plan:
        episode = by_index[index]
        base = int(episode["dataset_from_index"])
        cameras = decode_episode(args.source, episode, start, end)
        count = end - start
        ctx = EpisodeContext(seed=args.seed_base + written, task=task.prompt,
                             action_mode="joint_position")
        recorder.start_episode(ctx, f"graft_slice_{kind}", "joint_position")
        for offset in range(count):
            row = base + start + offset
            observation = {
                "observation.images.front": cameras["observation.images.front"][offset],
                "observation.images.left_wrist": cameras["observation.images.left_wrist"][offset],
                "observation.images.right_wrist": cameras["observation.images.right_wrist"][offset],
                **{name: scalars[name][row] for name in SCALAR_COLUMNS if name != "action"},
            }
            recorder.add_frame(observation, scalars["action"][row])
        recorder.finish_episode(
            success=True,
            metrics={
                "slice_kind": kind,
                "source_episode": index,
                "slice_from_frame": start,
                "graft_tier": metrics.get("graft_tier"),
                "graft_policy_steps": metrics.get("graft_policy_steps"),
                "graft_expert_steps": metrics.get("graft_expert_steps"),
            },
        )
        written += 1
        frames_written += count
        print(
            f"  [{written}/{len(plan)}] {kind:6s} src={index:3d} frames {start}..{end} "
            f"({count}) total={frames_written}",
            flush=True,
        )

    recorder.close()

    detail = dict(summary)
    detail["slice"] = {
        "source": str(args.source),
        "tier": args.tier,
        "full_episodes": args.full_episodes,
        "margin": args.margin,
        "episodes": written,
        "frames": frames_written,
    }
    detail["accepted_episodes"] = written
    (args.out / "collection_summary.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {written} episodes / {frames_written} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
