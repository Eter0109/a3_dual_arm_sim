"""Run benchmark evaluation on single-box batch transfer policies and experts.

Scores policies based on the number of cookies successfully placed in the target box
across a series of episodes (default: 20) with randomized box and cookie placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.benchmark import (
    ACTPolicyAdapter,
    CookieBatchBenchmark,
    SmolVLAPolicyAdapter,
)
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


def run_metadata(args, benchmark):
    metadata = vars(args).copy()
    metadata["config"] = asdict(benchmark.config)
    metadata["success_conditions"] = {
        "required_cookies_in_target": 10,
        "required_cookies_in_source": 70,
        "require_exact_slots": True,
        "success_hold_steps": benchmark.success_hold_steps,
        "slot_tolerance_m": benchmark.config.cookie_transfer.target_slot_tolerance_m,
        "wall_contact_tolerance_m": benchmark.wall_contact_tolerance_m,
    }
    metadata["randomization"] = {
        "boxes": benchmark.randomize_boxes,
        "cookies": benchmark.randomize_cookies,
        "target_bin_noise_m": benchmark.target_bin_noise_m,
        "source_bin_noise_m": benchmark.source_bin_noise_m,
        "cookie_noise_m": benchmark.cookie_noise_m,
        "cookie_yaw_noise_rad": benchmark.cookie_yaw_noise_rad,
        "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
    }
    for key, command in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_diff", ["git", "diff", "HEAD"]),
    ):
        try:
            result = subprocess.run(
                command, cwd=project_root(), capture_output=True, text=True, check=False
            )
            metadata[key] = result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            metadata[key] = None
    if args.policy.startswith(("smolvla:", "act:")):
        from a3_dual_arm_sim.workflows.benchmark import resolve_smolvla_checkpoint

        checkpoint = resolve_smolvla_checkpoint(args.policy.split(":", 1)[1])
        metadata["checkpoint"] = str(checkpoint)
        metadata["weights_kind"] = (
            "ema" if checkpoint.name == "pretrained_model_ema" else "ordinary"
        )
        with (checkpoint / "model.safetensors").open("rb") as stream:
            metadata["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata["checkpoint_config"] = json.loads((checkpoint / "config.json").read_text())
        dataset_root = (
            project_root() / (args.dataset_root or Path("datasets/a3_front_close_left_100"))
        ).resolve()
        metadata["dataset_root"] = str(dataset_root)
        marker = dataset_root / ".a3_download_revision.json"
        metadata["dataset_revision"] = json.loads(marker.read_text()) if marker.is_file() else None
    metadata["cookie_id_convention"] = "cookie_positions array index is the cookie ID"
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark single-box cookie batch transfer policies"
    )
    parser.add_argument(
        "--policy",
        default="same_column",
        help="Policy: same_column, cross_column, smolvla[:checkpoint_or_run], act:checkpoint_or_run, or module:factory",
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
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument(
        "--slot-tolerance-x", type=float, default=0.015, help="Slot x tolerance in meters"
    )
    parser.add_argument(
        "--slot-tolerance-y", type=float, default=0.005, help="Slot y tolerance in meters"
    )
    parser.add_argument(
        "--wall-contact-tolerance",
        type=float,
        default=0.030,
        help="Group boundary coverage tolerance in meters",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--repo-id", default="Eter0109/a3-front-close-left-100")
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="ACT only; requires n-action-steps=1",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--diagnostic-dir", type=Path, default=None)
    args = parser.parse_args()
    overrides = any(
        v is not None
        for v in (
            args.inference_seed,
            args.n_action_steps,
            args.dataset_root,
            args.device,
            args.temporal_ensemble_coeff,
        )
    )
    if args.compare and (overrides or args.diagnostic_dir):
        parser.error(
            "Use separate expert runs for diagnostics; --compare does not accept overrides"
        )
    if overrides and not args.policy.startswith(("smolvla:", "act:")):
        parser.error("Policy overrides require smolvla:<checkpoint> or act:<checkpoint>")

    if args.temporal_ensemble_coeff is not None and not args.policy.startswith("act:"):
        parser.error("Temporal ensembling requires act:<checkpoint>")

    benchmark = DiagnosticBenchmark(
        max_steps=args.max_steps,
        randomize_boxes=not args.no_randomize_boxes,
        target_bin_noise_m=args.target_box_noise,
        source_bin_noise_m=args.source_box_noise,
        randomize_cookies=not args.no_randomize_cookies,
        render=args.render,
        success_hold_steps=args.success_hold_steps,
        slot_tolerance_m=(args.slot_tolerance_x, args.slot_tolerance_y),
        wall_contact_tolerance_m=args.wall_contact_tolerance,
    )

    workers = 1 if args.render else args.workers
    if args.diagnostic_dir is not None:
        benchmark.diagnostic_dir = (project_root() / args.diagnostic_dir).resolve()
        benchmark.diagnostic_dir.mkdir(parents=True, exist_ok=False)
        metadata = run_metadata(args, benchmark)
        (benchmark.diagnostic_dir / "manifest.json").write_text(
            json.dumps(metadata, default=str, indent=2)
        )
        workers = 1
    policy = args.policy
    if overrides or args.policy.startswith("act:"):
        adapter = ACTPolicyAdapter if args.policy.startswith("act:") else SmolVLAPolicyAdapter
        extra = {}
        if args.policy.startswith("act:"):
            extra["temporal_ensemble_coeff"] = args.temporal_ensemble_coeff
        policy = adapter(
            args.policy.split(":", 1)[1],
            dataset_root=(
                project_root() / (args.dataset_root or Path("datasets/a3_front_close_left_100"))
            ).resolve(),
            repo_id=args.repo_id,
            device=args.device,
            inference_seed=args.inference_seed,
            n_action_steps=args.n_action_steps,
            **extra,
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
            f"{'Complete Task Success':<28} | {f'{res_same.success_rate * 100:.1f}%':<16} | {f'{res_cross.success_rate * 100:.1f}%':<16}"
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
                "run": run_metadata(args, benchmark),
                "same_column": res_same.to_dict(),
                "cross_column": res_cross.to_dict(),
            }
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"Saved comparison report to {args.output.resolve()}")

        return 0

    # Capture configuration before close() releases the loaded model.
    loaded_policy = getattr(getattr(policy, "plugin", None), "_policy", None)
    effective_config = asdict(loaded_policy.config) if loaded_policy is not None else None
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
        report = result.to_dict()
        report["run"] = run_metadata(args, benchmark)
        if effective_config is not None:
            report["run"]["effective_policy_config"] = effective_config
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        print(f"Saved benchmark report to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
