"""Evaluate trained ACT policy on A3 Cookie Transfer Task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.data.audit import dataset_camera_config
from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.policies.act import ACTPolicyPlugin
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, EpisodeScore


def parse_seeds(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def find_latest_checkpoint(base_dir: Path) -> Path | None:
    """Find the latest checkpoint directory under base_dir."""
    candidates = sorted(base_dir.glob("checkpoints/*/pretrained_model/config.json"))
    if candidates:
        return candidates[-1].parent
    if (base_dir / "pretrained_model" / "config.json").is_file():
        return base_dir / "pretrained_model"
    if (base_dir / "config.json").is_file():
        return base_dir
    return None


def run_evaluated_episode_with_video(
    benchmark: CookieBatchBenchmark,
    policy: ACTPolicyPlugin,
    seed: int,
    episode_idx: int,
    video_path: Path | None = None,
) -> EpisodeScore:
    """Run an episode, optionally recording a combined 3-camera video."""
    env = benchmark.create_env(render_cameras=True)
    frames: list[np.ndarray] = []

    try:
        observation, info = env.reset(seed=seed, options={"randomize_cookies": benchmark.randomize_cookies})
        policy.reset(None)
        if benchmark.render:
            env.render()
        steps = 0
        t_start = time.monotonic()
        terminated = False
        truncated = False

        while steps < benchmark.max_steps:
            steps += 1
            if video_path is not None:
                img_front = observation.get("observation.images.front")
                img_left = observation.get("observation.images.left_wrist")
                img_right = observation.get("observation.images.right_wrist")
                if img_front is not None and img_left is not None and img_right is not None:
                    combined = np.hstack([img_left, img_front, img_right])
                    frames.append(combined)

            action = policy.act(observation, "transfer 10 cookies into target box")
            observation, _, terminated, truncated, info = env.step(action)
            if benchmark.render:
                env.render()

            if steps % 20 == 0 or terminated or truncated:
                now = time.monotonic()
                elapsed = max(1e-3, now - t_start)
                speed = steps / elapsed
                cur_in_target = int(info.get("cookies_in_target", 0))
                print(
                    f"\r  -> Step [{steps:4d}/{benchmark.max_steps}] ({steps * 100 // benchmark.max_steps:2d}%) | "
                    f"{speed:.1f} steps/s | Cookies in target: {cur_in_target}/10 | Elapsed: {elapsed:.0f}s",
                    end="",
                    flush=True,
                )

            if terminated or truncated:
                break

        print()  # newline after step progress
        wall_time = time.monotonic() - t_start
        cookies_in_target = int(info.get("cookies_in_target", 0))
        cookies_in_source = int(info.get("cookies_in_source", 0))
        env_success = bool(info.get("success", False))
        success = env_success and cookies_in_target == 10

        target_pos = (
            env.data.xpos[env._target_bin_body].round(4).tolist()
            if hasattr(env, "_target_bin_body") and env._target_bin_body is not None
            else [0.0, 0.0, 0.0]
        )
        source_pos = (
            env.data.xpos[env._source_bin_body][:2].round(4).tolist()
            if hasattr(env, "_source_bin_body") and env._source_bin_body is not None
            else [0.0, 0.0]
        )

        score_res = EpisodeScore(
            episode=episode_idx,
            seed=seed,
            score=cookies_in_target,
            max_score=10,
            success=success,
            steps=steps,
            wall_seconds=round(wall_time, 2),
            cookies_in_target=cookies_in_target,
            cookies_in_source=cookies_in_source,
            target_bin_pos=target_pos,
            source_bin_pos=source_pos,
            phase=str(info.get("phase", "UNKNOWN")),
            failure_reason=info.get("failure_reason"),
        )

        if video_path is not None and frames:
            try:
                import cv2
                video_path.parent.mkdir(parents=True, exist_ok=True)
                h, w, c = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = benchmark.config.control_hz
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
            except ImportError:
                pass

        return score_res

    finally:
        env.close()


def evaluate_act(
    *,
    checkpoint: Path,
    dataset_root: Path,
    repo_id: str,
    seeds: list[int],
    max_steps: int = 2500,
    device: str = "cuda",
    temporal_ensemble_coeff: float = 0.01,
    render: bool = False,
    save_video: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()

    dataset_cameras = dataset_camera_config(dataset_root)

    print("=" * 68)
    print("  A3 Dual-Arm ACT Policy Evaluation")
    print(f"  Checkpoint        : {checkpoint}")
    print(f"  Dataset Root      : {dataset_root}")
    print(f"  Seeds ({len(seeds):2d})         : {seeds}")
    print(f"  Max Steps         : {max_steps}")
    print(f"  Device            : {device}")
    print(f"  Temporal Ensemble : coeff={temporal_ensemble_coeff}")
    print(f"  Save Video        : {'Enabled' if save_video else 'Disabled'}")
    if dataset_cameras is not None:
        print(
            f"  Front Camera      : pos={tuple(round(v, 3) for v in dataset_cameras.front_position_m)} "
            f"fovy={dataset_cameras.front_fovy_deg:g}deg (from dataset)"
        )
    print("=" * 68, flush=True)

    print("\n[*] Loading ACT neural policy...", flush=True)
    policy = ACTPolicyPlugin(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        repo_id=repo_id,
        device=device,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )

    benchmark = CookieBatchBenchmark(
        max_steps=max_steps,
        render=render,
        cameras=dataset_cameras,
    )

    results = []
    total_score = 0
    total_success = 0
    start_time = time.time()
    videos_dir = project_root() / "artifacts" / "videos"

    print("\n--- Running Evaluation Episodes ---")
    for idx, seed in enumerate(seeds, 1):
        print(f"\n[Episode {idx}/{len(seeds)}] Running Seed {seed}...", flush=True)
        video_file = videos_dir / f"act_seed_{seed}.mp4" if save_video else None

        ep_res = run_evaluated_episode_with_video(
            benchmark=benchmark,
            policy=policy,
            seed=seed,
            episode_idx=idx,
            video_path=video_file,
        )
        score = ep_res.score
        success = ep_res.success
        total_score += score
        total_success += int(success)
        results.append(ep_res.to_dict())

        status_str = "SUCCESS (10/10)" if success else f"PARTIAL ({score}/10 cookies)"
        video_note = f" | Video: {video_file.name}" if video_file and video_file.is_file() else ""
        print(
            f"  Result: {status_str} | Steps: {ep_res.steps} | Time: {ep_res.wall_seconds:.1f}s{video_note}",
            flush=True,
        )

    duration = time.time() - start_time
    mean_score = total_score / len(seeds) if seeds else 0.0
    success_rate = total_success / len(seeds) if seeds else 0.0

    summary = {
        "policy_type": "act",
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "total_episodes": len(seeds),
        "total_successes": total_success,
        "success_rate": success_rate,
        "mean_score": mean_score,
        "max_possible_score": 10.0,
        "evaluation_duration_s": round(duration, 2),
        "episodes": results,
    }

    if output_path is not None:
        out_file = output_path.expanduser().resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  Report Saved To : {out_file}")

    print("\n" + "=" * 68)
    print("  ACT EVALUATION SUMMARY")
    print(f"  Episodes Tested : {len(seeds)}")
    print(f"  Success Rate    : {success_rate * 100:.1f}% ({total_success}/{len(seeds)})")
    print(f"  Mean Score      : {mean_score:.2f} / 10.00 cookies")
    print("=" * 68 + "\n", flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained ACT model directory or its parent",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/a3-front-close-left-100"),
        help="Path to dataset root for stats normalization and camera geometry",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/a3-front-close-left-100",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Evaluation seeds: range (e.g. 0-9) or comma-separated list",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2500,
        help="Maximum simulator control steps per episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cuda", "cpu"),
        default="cuda",
        help="Execution device",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="Temporal ensembling smoothing weight (default: 0.01)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open 3D MuJoCo graphical viewer during rollout",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record stitched camera MP4 videos into artifacts/videos/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cookie_act_evaluation.json"),
        help="Output JSON evaluation report path",
    )
    args = parser.parse_args()

    root = project_root()
    checkpoint = args.checkpoint
    if checkpoint is None:
        default_base = root / "outputs" / "train" / "a3_act_25k"
        checkpoint = find_latest_checkpoint(default_base)
        if checkpoint is None:
            all_checkpoints = sorted(root.glob("outputs/train/**/a3*act*/checkpoints/*/pretrained_model/config.json"))
            if all_checkpoints:
                checkpoint = all_checkpoints[-1].parent

    if checkpoint is None or not (checkpoint / "config.json").is_file():
        print(f"[!] Error: No valid ACT checkpoint found. Please provide --checkpoint path.", file=sys.stderr)
        return 1

    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    seeds = parse_seeds(args.seeds)
    out_file = args.output if args.output.is_absolute() else root / args.output

    evaluate_act(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        seeds=seeds,
        max_steps=args.max_steps,
        device=args.device,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        render=args.render,
        save_video=args.save_video,
        output_path=out_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
