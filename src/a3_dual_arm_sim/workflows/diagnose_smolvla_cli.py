"""Resumable single-GPU development evaluation; acceptance seeds are excluded."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/a3_front_close_left_100")
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/smolvla_front_close_left_20k/model/checkpoints/020000"),
    )
    parser.add_argument(
        "--stage",
        choices=["baseline", "matrix", "horizon", "rules", "verify", "acceptance"],
        default="baseline",
    )
    parser.add_argument("--candidate", type=Path, help="JSON policy settings for horizon/verify")
    parser.add_argument(
        "--baseline-checkpoint-root",
        type=Path,
        default=Path("outputs/smolvla_front_close_left_20k/model/checkpoints/020000"),
    )
    args = parser.parse_args()
    root = project_root()
    args.dataset_root = (root / args.dataset_root).resolve()
    args.root = args.root if args.root.is_absolute() else root / args.root
    args.checkpoint_root = (
        args.checkpoint_root if args.checkpoint_root.is_absolute() else root / args.checkpoint_root
    )
    if args.stage in ("horizon", "rules", "verify", "acceptance") and args.candidate is None:
        parser.error("--candidate required")
    defaults = {
        "ema_alpha": 0.75,
        "anchor_right_arm": True,
        "gripper_sharpening": True,
        "n_action_steps": 8,
        "num_steps": 25,
        "align_vertical": False,
        "clamp_z": False,
    }
    raw = dict(defaults, ema_alpha=0, anchor_right_arm=False, gripper_sharpening=False)
    if args.stage == "baseline":
        variants = {"expert": None, "ordinary_default": dict(defaults, weights="pretrained_model")}
        seeds = range(1000, 1003)
    elif args.stage == "matrix":
        variants = {
            f"{weight}_{mode}": dict(settings, weights=weight)
            for weight in ("pretrained_model", "pretrained_model_ema")
            for mode, settings in (("default", defaults), ("raw", raw))
        }
        seeds = range(1000, 1005)
    else:
        candidate = json.loads(args.candidate.read_text())
        variants = (
            {f"horizon_{n}": dict(candidate, n_action_steps=n) for n in (1, 4, 8)}
            if args.stage == "horizon"
            else {"candidate": candidate}
        )
        if args.stage == "rules":
            variants = {
                "raw": candidate,
                "anchor": dict(candidate, anchor_right_arm=True),
                "gripper": dict(candidate, gripper_sharpening=True),
                "smooth": dict(candidate, ema_alpha=0.75),
            }
        seeds = range(1000, 1020 if args.stage == "verify" else 1005)
        if args.stage == "acceptance":
            baseline_root = (root / args.baseline_checkpoint_root).resolve()
            variants = {
                "baseline": dict(defaults, weights=str(baseline_root / "pretrained_model")),
                "candidate": candidate,
            }
            seeds = range(3000, 3020)
    bench = CookieBatchBenchmark(max_steps=1000)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": args.stage,
        "variants": variants,
        "seeds": list(seeds),
        "config": asdict(bench.config),
        "max_steps": 1000,
        "checkpoint_root": str(args.checkpoint_root),
        "dataset_root": str(args.dataset_root),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True),
        "hashes": {
            name: digest(args.checkpoint_root / name / "model.safetensors")
            for name in ("pretrained_model", "pretrained_model_ema")
        },
        "variant_hashes": {
            name: digest(args.checkpoint_root / setting["weights"] / "model.safetensors")
            for name, setting in variants.items()
            if setting is not None
        },
        "sources": {
            str(path.relative_to(root)): path.read_text()
            for path in [
                Path(__file__).resolve(),
                root / "src/a3_dual_arm_sim/workflows/diagnostics.py",
                root / "src/a3_dual_arm_sim/policies/smolvla.py",
                root / "src/a3_dual_arm_sim/workflows/benchmark.py",
            ]
        },
    }
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Experiment manifest changed; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = {}
    for name, settings in variants.items():
        directory = args.root / name
        directory.mkdir(exist_ok=True)
        policy = None
        results = []
        try:
            for seed in seeds:
                result_path = directory / f"seed_{seed}.json"
                if result_path.exists():
                    results.append(json.loads(result_path.read_text()))
                    continue
                if settings is not None and policy is None:
                    params = {k: v for k, v in settings.items() if k != "weights"}
                    policy = SmolVLAPolicyAdapter(
                        args.checkpoint_root / settings["weights"],
                        dataset_root=args.dataset_root,
                        device="cuda",
                        **params,
                    )
                if policy is not None:
                    policy.plugin.inference_seed = seed
                trace = None
                if args.stage == "baseline":
                    # Incomplete traces are preserved; retries get a fresh directory.
                    attempt = 0
                    trace_path = directory / f"seed_{seed}_trace_{attempt}"
                    while trace_path.exists():
                        attempt += 1
                        trace_path = directory / f"seed_{seed}_trace_{attempt}"
                    trace = EpisodeDiagnostic(trace_path, bench.config.control_hz)
                try:
                    score = bench.run_episode(policy or "same_column", seed, diagnostic=trace)
                finally:
                    if trace:
                        trace.close()
                result = score.to_dict()
                result.update(inference_seed=seed, settings=settings)
                temporary = result_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2))
                temporary.replace(result_path)
                results.append(result)
                print(name, json.dumps(result), flush=True)
        finally:
            if policy is not None:
                policy.close()
        summary[name] = {
            "successes": sum(r["success"] for r in results),
            "episodes": len(results),
            "mean_target": sum(r["cookies_in_target"] for r in results) / len(results),
        }
        (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
