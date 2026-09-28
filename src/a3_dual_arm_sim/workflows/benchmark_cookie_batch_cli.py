"""Run benchmark evaluation on single-box batch transfer policies and experts.

Scores policies based on the number of cookies successfully placed in the target box
across a series of episodes (default: 20) with randomized box and cookie placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic


class DiagnosticBenchmark(CookieBatchBenchmark):
    """Same execution path with optional observer-only traces."""

    diagnostic_dir = None

    def run_episode(self, policy, seed, episode_idx=1, **kwargs):
        if self.diagnostic_dir is None:
            return super().run_episode(policy, seed, episode_idx, **kwargs)
        trace = EpisodeDiagnostic(self.diagnostic_dir / f"seed_{seed}", self.config.control_hz)
        try:
            result = super().run_episode(policy, seed, episode_idx, diagnostic=trace, **kwargs)
            (trace.root / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
            return result
        finally:
            trace.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark single-box cookie batch transfer policies"
    )
    parser.add_argument(
        "--policy",
        default="same_column",
        help="Policy: same_column, cross_column, smolvla[:checkpoint_or_run], or module:factory",
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
        default=0.010,
        help="Target box random position noise in meters (default: 0.010 = 1cm)",
    )
    parser.add_argument(
        "--source-box-noise",
        type=float,
        default=0.010,
        help="Source box random position noise in meters (default: 0.010 = 1cm)",
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
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--diagnostic-dir", type=Path, default=None)
    args = parser.parse_args()
    overrides = any(
        v is not None for v in (args.inference_seed, args.n_action_steps, args.dataset_root)
    )
    if args.compare and (overrides or args.diagnostic_dir):
        parser.error(
            "Use separate expert runs for diagnostics; --compare does not accept overrides"
        )
    if overrides and not args.policy.startswith("smolvla:"):
        parser.error("Policy overrides require smolvla:<explicit checkpoint>")

    benchmark = DiagnosticBenchmark(
        max_steps=args.max_steps,
        randomize_boxes=not args.no_randomize_boxes,
        target_bin_noise_m=args.target_box_noise,
        source_bin_noise_m=args.source_box_noise,
        randomize_cookies=not args.no_randomize_cookies,
        render=args.render,
    )

    workers = 1 if args.render else args.workers
    if args.diagnostic_dir is not None:
        benchmark.diagnostic_dir = (project_root() / args.diagnostic_dir).resolve()
        benchmark.diagnostic_dir.mkdir(parents=True, exist_ok=False)
        metadata = vars(args).copy()
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True
        ).strip()
        metadata["git_diff"] = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=project_root(), text=True
        )
        if args.policy.startswith("smolvla:"):
            from a3_dual_arm_sim.workflows.benchmark import resolve_smolvla_checkpoint

            checkpoint = resolve_smolvla_checkpoint(args.policy.split(":", 1)[1]).resolve()
            metadata["checkpoint"] = str(checkpoint)
            with (checkpoint / "model.safetensors").open("rb") as stream:
                metadata["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata["cookie_id_convention"] = "cookie_positions array index is the cookie ID"
        (benchmark.diagnostic_dir / "manifest.json").write_text(
            json.dumps(metadata, default=str, indent=2)
        )
        workers = 1
    policy = args.policy
    if overrides:
        policy = SmolVLAPolicyAdapter(
            args.policy.split(":", 1)[1],
            dataset_root=(
                project_root() / (args.dataset_root or Path("datasets/a3_front_close_left_100"))
            ).resolve(),
            inference_seed=args.inference_seed,
            n_action_steps=args.n_action_steps,
        )
        workers = 1

    if args.compare:
        print("\n=======================================================")
        print(" BENCHMARK COMPARISON: same_column vs cross_column")
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
        print(
            f"{'Total Score':<28} | {f'{res_same.total_score}/{res_same.max_possible_score}':<16} | {f'{res_cross.total_score}/{res_cross.max_possible_score}':<16}"
        )
        print(
            f"{'Mean Score / Episode':<28} | {f'{res_same.mean_score:.2f}/10':<16} | {f'{res_cross.mean_score:.2f}/10':<16}"
        )
        print(
            f"{'Success Rate (10/10)':<28} | {f'{res_same.success_rate * 100:.1f}%':<16} | {f'{res_cross.success_rate * 100:.1f}%':<16}"
        )
        print(
            f"{'Mean Steps / Episode':<28} | {f'{res_same.mean_steps:.1f}':<16} | {f'{res_cross.mean_steps:.1f}':<16}"
        )
        print(
            f"{'Mean Wall Time / Episode':<28} | {f'{res_same.mean_wall_seconds:.2f}s':<16} | {f'{res_cross.mean_wall_seconds:.2f}s':<16}"
        )
        print("=" * 68 + "\n")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "same_column": res_same.to_dict(),
                "cross_column": res_cross.to_dict(),
            }
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"Saved comparison report to {args.output.resolve()}")

        return 0

    # Single policy benchmark
    try:
        result = benchmark.evaluate(
            policy=policy,
            num_episodes=args.episodes,
            seed_start=args.seed_start,
            workers=workers,
        )
    finally:
        if not isinstance(policy, str):
            policy.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Saved benchmark report to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
