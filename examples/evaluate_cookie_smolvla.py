#!/usr/bin/env python3
"""
Evaluate trained SmolVLA policy on A3 Cookie Transfer Task.
-----------------------------------------------------------
Supports:
- Comprehensive metrics evaluation (success rate, cookies transferred, steps, duration)
- Seen (training distribution) vs Unseen (generalization) seed evaluation
- Real-time interactive 3D GUI visualization (--render)
- Video recording of rollout cameras (--save-video) to MP4
- Structured JSON and markdown report export to artifacts/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.policies.smolvla import SmolVLAPolicyPlugin
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
    policy: SmolVLAPolicyPlugin,
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
        steps = 0
        t_start = time.monotonic()
        terminated = False
        truncated = False

        while steps < benchmark.max_steps:
            steps += 1
            if video_path is not None:
                # Combine Front (256x256), Left Wrist (256x256), Right Wrist (256x256)
                img_front = observation.get("observation.images.front")
                img_left = observation.get("observation.images.left_wrist")
                img_right = observation.get("observation.images.right_wrist")
                if img_front is not None and img_left is not None and img_right is not None:
                    # Stitch horizontally: [Left Wrist | Front (main) | Right Wrist]
                    combined = np.hstack([img_left, img_front, img_right])
                    frames.append(combined)

            action = policy.act(observation, "transfer 10 cookies into target box")
            observation, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        wall_time = time.monotonic() - t_start
        cookies_in_target = int(info.get("cookies_in_target", 0))
        cookies_in_source = int(info.get("cookies_in_source", 0))
        env_success = bool(info.get("success", False))
        success = env_success and cookies_in_target == 10

        target_pos = (
            env.data.xpos[env._target_bin_body].round(4).tolist()
            if hasattr(env, "_target_bin_body")
            else []
        )
        source_pos = (
            env.data.xpos[env._source_bin_body][:2].round(4).tolist()
            if hasattr(env, "_source_bin_body")
            else []
        )

        failure_reason = None
        if not success:
            if terminated:
                failure_reason = "environment_safety_terminated"
            elif truncated or steps >= benchmark.max_steps:
                failure_reason = "max_steps_exceeded"
            else:
                failure_reason = f"only {cookies_in_target}/10 cookies in target box"

        # Save video if frames were captured
        if video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            h, w, c = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (w, h))
            for f in frames:
                # RGB to BGR for OpenCV
                out_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out_writer.release()

        return EpisodeScore(
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
            phase="smolvla_policy",
            failure_reason=failure_reason,
        )
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy on Cookie Transfer")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained SmolVLA checkpoint (defaults to latest in outputs/train/a3_smolvla_25k)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/a3_single_box_same_column_200"),
        help="Dataset root used to retrieve feature statistics for normalization",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/a3-single-box-same-column-200",
        help="Dataset repo-id",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Seed list or range to evaluate (e.g. '0-9', '0,1,2', '100-109')",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum simulation steps per episode (default: 1200)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open MuJoCo 3D viewer window to watch real-time robot policy execution",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record 3-camera MP4 videos of the evaluation episodes to artifacts/videos/",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cookie_smolvla_evaluation.json"),
        help="Output JSON evaluation report path",
    )
    args = parser.parse_args()

    root = project_root()
    checkpoint = args.checkpoint
    if checkpoint is None:
        default_base = root / "outputs" / "train" / "a3_smolvla_25k"
        checkpoint = find_latest_checkpoint(default_base)
        if checkpoint is None:
            all_checkpoints = sorted(root.glob("outputs/train/**/pretrained_model/config.json"))
            if all_checkpoints:
                checkpoint = all_checkpoints[-1].parent

    if checkpoint is None or not (checkpoint / "config.json").is_file():
        print(f"[!] Error: No valid SmolVLA checkpoint found. Please provide --checkpoint path.", file=sys.stderr)
        return 1

    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    if not dataset_root.exists():
        print(f"[!] Warning: Dataset root {dataset_root} not found. Normalizer stats might be missing.", file=sys.stderr)

    seeds = parse_seeds(args.seeds)
    print("=" * 68)
    print("  A3 Dual-Arm SmolVLA Policy Evaluation")
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Dataset Root  : {dataset_root}")
    print(f"  Seeds ({len(seeds):2d})     : {seeds}")
    print(f"  Max Steps     : {args.max_steps}")
    print(f"  Device        : {args.device}")
    print(f"  Viewer GUI    : {'Enabled (MuJoCo Window)' if args.render else 'Headless'}")
    print(f"  Save Video    : {'Enabled (artifacts/videos/)' if args.save_video else 'Disabled'}")
    print("=" * 68, flush=True)

    print("\n[*] Loading SmolVLA neural policy...", flush=True)
    policy = SmolVLAPolicyPlugin(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )

    benchmark = CookieBatchBenchmark(
        max_steps=args.max_steps,
        render=args.render,
    )

    results = []
    total_score = 0
    total_success = 0
    start_time = time.time()
    videos_dir = root / "artifacts" / "videos"

    print("\n--- Running Evaluation Episodes ---")
    for idx, seed in enumerate(seeds, 1):
        print(f"\n[Episode {idx}/{len(seeds)}] Running Seed {seed}...", flush=True)
        video_file = videos_dir / f"smolvla_seed_{seed}.mp4" if args.save_video else None
        
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
        "checkpoint": str(checkpoint),
        "total_episodes": len(seeds),
        "total_successes": total_success,
        "success_rate": success_rate,
        "mean_score": mean_score,
        "max_possible_score": 10.0,
        "evaluation_duration_s": round(duration, 2),
        "episodes": results,
    }

    out_file = args.output if args.output.is_absolute() else root / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 68)
    print("  EVALUATION SUMMARY")
    print(f"  Episodes Tested : {len(seeds)}")
    print(f"  Success Rate    : {success_rate * 100:.1f}% ({total_success}/{len(seeds)})")
    print(f"  Mean Score      : {mean_score:.2f} / 10.00 cookies")
    print(f"  Report Saved To : {out_file.resolve()}")
    if args.save_video:
        print(f"  Videos Saved To : {videos_dir.resolve()}")
    print("=" * 68 + "\n", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
