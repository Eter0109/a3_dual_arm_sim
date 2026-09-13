"""Score a trained checkpoint on the cookie-transfer task over several episodes.

The environment runs on the CPU-only login node because MuJoCo aborts on the GPU
worker (its CPU lacks AVX), and the policy runs there too -- CPU inference is
affordable here because LeRobot's action queue means one forward pass covers
`n_action_steps` control steps.

Reports the task's own success criterion (exact 2x5 fill, all four walls, held
for a second) plus how many cookies actually landed in the bin, so a partial
policy is distinguishable from a broken one.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

TASK = "transfer exactly ten upright square cookie blocks into the 2x5 box"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT / "outputs/datasets/a3_cookie_overnight",
        help="Supplies the action contract and normalization statistics",
    )
    parser.add_argument("--repo-id", default="local/a3-cookie-overnight")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # These must be set before lerobot is imported anywhere.
    os.environ.setdefault("HF_HOME", str(PROJECT / ".runtime/hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from a3_dual_arm_sim.contracts import STATE, EpisodeContext
    from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv
    from a3_dual_arm_sim.lerobot_policy import (
        LeRobotPolicyAdapter,
        PolicyRuntimeConfig,
    )

    policy = LeRobotPolicyAdapter(
        PolicyRuntimeConfig(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            device=args.device,
        )
    )
    env = A3CookieTransferEnv(render_cameras=True)
    env.use_fast_render()

    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {policy.device}")
    print(f"env keys   : {policy.declared_keys}")
    print()

    results: list[dict] = []
    try:
        for index in range(args.episodes):
            seed = args.first_seed + index
            observation, _ = env.reset(seed=seed, options={"randomize_cookies": False})
            policy.reset(EpisodeContext(seed=seed, task=TASK, action_mode="joint_position"))
            home = np.asarray(observation[STATE], dtype=np.float64).copy()

            started = time.perf_counter()
            info: dict = {}
            terminated = truncated = False
            steps = 0
            action_delta = 0.0
            previous = None
            while not terminated and not truncated and steps < args.max_steps:
                action = policy.act(observation, TASK)
                if previous is not None:
                    action_delta = max(action_delta, float(np.abs(action - previous).max()))
                previous = action.copy()
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1

            state = np.asarray(observation[STATE], dtype=np.float64)
            elapsed = time.perf_counter() - started
            record = {
                "seed": seed,
                "success": bool(info.get("success", False)),
                "cookies_in_target": int(info.get("cookies_in_target", 0)),
                "cookies_in_source": int(info.get("cookies_in_source", 0)),
                "exact_fill": bool(info.get("exact_2x5_fill", False)),
                "touches_all_walls": bool(info.get("target_touches_all_walls", False)),
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "safety_reason": info.get("safety_reason"),
                "max_joint_travel_rad": float(np.abs(state - home).max()),
                "max_action_delta_rad": action_delta,
                "wall_time_s": elapsed,
            }
            results.append(record)

            verdict = "SUCCESS" if record["success"] else "failed "
            print(
                f"  ep{index} seed={seed:4d} [{verdict}] "
                f"in_target={record['cookies_in_target']:2d}/10 "
                f"in_source={record['cookies_in_source']:2d}/20 "
                f"steps={steps:4d} joint_travel={record['max_joint_travel_rad']:.2f}rad "
                f"{elapsed:5.0f}s",
                flush=True,
            )
    finally:
        policy.close()
        env.close()

    successes = sum(record["success"] for record in results)
    in_target = [record["cookies_in_target"] for record in results]
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "cookies_in_target": {
            "mean": float(np.mean(in_target)) if in_target else 0.0,
            "min": int(min(in_target)) if in_target else 0,
            "max": int(max(in_target)) if in_target else 0,
        },
        "safety_stops": sum(1 for record in results if record["safety_reason"]),
        "mean_wall_time_s": float(np.mean([record["wall_time_s"] for record in results]))
        if results
        else 0.0,
        "results": results,
    }

    print()
    print("=== summary ===")
    print(f"  success rate     : {successes}/{len(results)} ({summary['success_rate']:.0%})")
    print(f"  cookies in target: mean={summary['cookies_in_target']['mean']:.1f} "
          f"min={summary['cookies_in_target']['min']} max={summary['cookies_in_target']['max']}")
    print(f"  safety stops     : {summary['safety_stops']}")
    print(f"  mean episode time: {summary['mean_wall_time_s']:.0f}s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"  written to       : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
