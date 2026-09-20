#!/usr/bin/env python3
"""Run benchmark evaluation on single-box batch transfer policies and experts.

Scores policies based on the number of cookies successfully placed in the target box
across a series of episodes (default: 20) with randomized box and cookie placement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.benchmark import CookieBatchBenchmark, run_cookie_batch_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark single-box cookie batch transfer policies"
    )
    parser.add_argument(
        "--policy",
        default="same_column",
        help="Policy to benchmark: 'same_column', 'cross_column', or 'module:factory' (default: same_column)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of benchmark episodes (default: 20)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Initial seed for episode sequence (default: 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps allowed per episode (default: 1000)",
    )
    parser.add_argument(
        "--no-randomize-boxes",
        action="store_true",
        help="Disable small randomized displacement of source/target boxes",
    )
    parser.add_argument(
        "--no-randomize-cookies",
        action="store_true",
        help="Disable small randomized jitter of cookies",
    )
    parser.add_argument(
        "--target-box-noise",
        type=float,
        default=0.002,
        help="Target box random position noise in meters (default: 0.002 = 2mm)",
    )
    parser.add_argument(
        "--source-box-noise",
        type=float,
        default=0.001,
        help="Source box random position noise in meters (default: 0.001 = 1mm)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes to speed up evaluation (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON benchmark report (e.g. artifacts/benchmark_results.json)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare both 'same_column' and 'cross_column' expert models on the same seeds",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open viewer to visually watch simulation execution",
    )
    args = parser.parse_args()

    benchmark = CookieBatchBenchmark(
        max_steps=args.max_steps,
        randomize_boxes=not args.no_randomize_boxes,
        target_bin_noise_m=args.target_box_noise,
        source_bin_noise_m=args.source_box_noise,
        randomize_cookies=not args.no_randomize_cookies,
        render=args.render,
    )

    workers = 1 if args.render else args.workers

    if args.compare:
        print("\n=======================================================")
        print(f" BENCHMARK COMPARISON: same_column vs cross_column")
        print(f" Total Episodes per policy: {args.episodes}")
        print(f" Seeds: {args.seed_start} .. {args.seed_start + args.episodes - 1}")
        print(f" Parallel Workers: {workers}")
        print("=======================================================\n")

        res_same = benchmark.evaluate(
            policy="same_column",
            num_episodes=args.episodes,
            seed_start=args.seed_start,
            policy_name="same_column",
            workers=workers,
        )

        res_cross = benchmark.evaluate(
            policy="cross_column",
            num_episodes=args.episodes,
            seed_start=args.seed_start,
            policy_name="cross_column",
            workers=workers,
        )

        print("\n" + "=" * 68)
        print("                 BENCHMARK COMPARISON RESULTS")
        print("=" * 68)
        header = f"{'Metric':<28} | {'same_column':<16} | {'cross_column':<16}"
        print(header)
        print("-" * 68)
        print(f"{'Episodes':<28} | {res_same.total_episodes:<16} | {res_cross.total_episodes:<16}")
        print(f"{'Total Score':<28} | {f'{res_same.total_score}/{res_same.max_possible_score}':<16} | {f'{res_cross.total_score}/{res_cross.max_possible_score}':<16}")
        print(f"{'Mean Score / Episode':<28} | {f'{res_same.mean_score:.2f}/10':<16} | {f'{res_cross.mean_score:.2f}/10':<16}")
        print(f"{'Success Rate (10/10)':<28} | {f'{res_same.success_rate*100:.1f}%':<16} | {f'{res_cross.success_rate*100:.1f}%':<16}")
        print(f"{'Mean Steps / Episode':<28} | {f'{res_same.mean_steps:.1f}':<16} | {f'{res_cross.mean_steps:.1f}':<16}")
        print(f"{'Mean Wall Time / Episode':<28} | {f'{res_same.mean_wall_seconds:.2f}s':<16} | {f'{res_cross.mean_wall_seconds:.2f}s':<16}")
        print("=" * 68 + "\n")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "same_column": res_same.to_dict(),
                "cross_column": res_cross.to_dict(),
            }
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Saved comparison report to {args.output.resolve()}")

        return 0

    # Single policy benchmark
    result = benchmark.evaluate(
        policy=args.policy,
        num_episodes=args.episodes,
        seed_start=args.seed_start,
        workers=workers,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved benchmark report to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
