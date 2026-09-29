#!/usr/bin/env python3
"""
Evaluate trained Diffusion Policy on A3 Cookie Transfer Task.
--------------------------------------------------------------
Features:
- Headless / GUI evaluation across seen and unseen seeds
- Success rate, cookie placement count, step count, and latency metrics
- Optional video rendering (--save-video) to MP4
- Structured JSON evaluation report generation
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

# Ensure EGL headless on Linux
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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
                img_front = observation.get("observation.images.front")
                img_left = observation.get("observation.images.left_wrist")
                img_right = observation.get("observation.images.right_wrist")
                if img_front is not None and img_left is not None and img_right is not None:
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

        if video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            h, w, c = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (w, h))
            for f in frames:
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
            phase="diffusion_policy",
            failure_reason=failure_reason,
        )
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy on Cookie Transfer")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained Diffusion Policy checkpoint",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/a3_autodl_dataset"),
        help="Dataset root used to retrieve feature statistics for normalization",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/a3-autodl-cookie",
        help="Dataset repo-id",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-9",
        help="Seed list or range to evaluate (e.g. '0-9', '0-19')",
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
        help="Open MuJoCo 3D viewer window",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record MP4 videos of evaluation episodes",
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
        default=Path("artifacts/cookie_diffusion_evaluation.json"),
        help="Output JSON evaluation report path",
    )
    args = parser.parse_args()

    root = project_root()
    checkpoint = args.checkpoint
    if checkpoint is None:
        # Search under outputs/train
        candidates = sorted(root.glob("outputs/train/**/pretrained_model/config.json"))
        if candidates:
            checkpoint = candidates[-1].parent

    if checkpoint is None or not (checkpoint / "config.json").is_file():
        print(f"[!] Error: No valid checkpoint found. Please provide --checkpoint path.", file=sys.stderr)
        return 1

    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    seeds = parse_seeds(args.seeds)

    print("=" * 68)
    print("  A3 Dual-Arm Diffusion Policy Evaluation")
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Dataset Root  : {dataset_root}")
    print(f"  Seeds ({len(seeds):2d})     : {seeds}")
    print(f"  Max Steps     : {args.max_steps}")
    print(f"  Device        : {args.device}")
    print(f"  Output Report : {args.output}")
    print("=" * 68, flush=True)

    print("\n[*] Loading Diffusion neural policy...", flush=True)
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

    video_dir = root / "artifacts" / "videos" if args.save_video else None

    for i, seed in enumerate(seeds):
        video_path = video_dir / f"eval_diffusion_ep{i:02d}_seed{seed}.mp4" if video_dir else None
        print(f"[*] Running Eval Episode {i+1}/{len(seeds)} (Seed: {seed})...", end=" ", flush=True)

        t0 = time.time()
        res = run_evaluated_episode_with_video(
            benchmark=benchmark,
            policy=policy,
            seed=seed,
            episode_idx=i,
            video_path=video_path,
        )
        elapsed = time.time() - t0

        status_str = "SUCCESS" if res.success else "FAILED"
        print(f"{status_str} | Cookies: {res.score}/10 | Steps: {res.steps} ({elapsed:.1f}s)")
        if not res.success and res.failure_reason:
            print(f"    Reason: {res.failure_reason}")

        total_score += res.score
        if res.success:
            total_success += 1
        results.append(res)

    success_rate = (total_success / len(seeds)) * 100.0
    avg_score = total_score / len(seeds)

    summary = {
        "policy_type": "diffusion",
        "checkpoint": str(checkpoint),
        "total_episodes": len(seeds),
        "success_rate_percent": round(success_rate, 2),
        "average_cookies_placed": round(avg_score, 2),
        "max_cookies_per_episode": 10,
        "results": [
            {
                "seed": r.seed,
                "success": r.success,
                "score": r.score,
                "steps": r.steps,
                "wall_seconds": r.wall_seconds,
                "failure_reason": r.failure_reason,
            }
            for r in results
        ],
    }

    out_path = args.output if args.output.is_absolute() else root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 68)
    print("  EVALUATION SUMMARY")
    print(f"  Success Rate : {success_rate:.1f}% ({total_success}/{len(seeds)})")
    print(f"  Average Score: {avg_score:.2f} / 10 cookies")
    print(f"  Report Saved : {out_path}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
