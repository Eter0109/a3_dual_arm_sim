"""One-off retraining probe: small seed x config matrix with optional traces.

Used to isolate whether the retrained checkpoint regressed or whether the FK/IK
geometric post-processing (align_vertical / clamp_z) is destroying the policy's
actions. Results are per-seed JSON files, resumable like the diagnostic stages.
"""

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic

FULL = dict(
    ema_alpha=0.75, anchor_right_arm=True, gripper_sharpening=True, n_action_steps=8, num_steps=25
)
RAW = dict(
    ema_alpha=0, anchor_right_arm=False, gripper_sharpening=False, n_action_steps=8, num_steps=25
)
CONFIGS = {
    "full_fkik": dict(FULL, align_vertical=True, clamp_z=True),
    "full_nofkik": dict(FULL, align_vertical=False, clamp_z=False),
    "raw_fkik": dict(RAW, align_vertical=True, clamp_z=True),
    "raw_nofkik": dict(RAW, align_vertical=False, clamp_z=False),
    "raw_n4": dict(RAW, align_vertical=False, clamp_z=False, n_action_steps=4),
    "raw_n16": dict(RAW, align_vertical=False, clamp_z=False, n_action_steps=16),
    "raw_clamponly": dict(RAW, align_vertical=False, clamp_z=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", default="full_fkik,full_nofkik,raw_fkik")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    root = project_root()
    args.root = args.root if args.root.is_absolute() else root / args.root
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    args.root.mkdir(parents=True, exist_ok=True)
    bench = CookieBatchBenchmark(max_steps=1000)
    summary = {}
    for name in args.variants.split(","):
        settings = CONFIGS[name]
        directory = args.root / name
        directory.mkdir(exist_ok=True)
        policy = None
        rows = []
        try:
            for seed in args.seeds:
                result_path = directory / f"seed_{seed}.json"
                if result_path.exists():
                    rows.append(json.loads(result_path.read_text()))
                    continue
                if policy is None:
                    policy = SmolVLAPolicyAdapter(
                        checkpoint,
                        dataset_root=root / "datasets/a3_front_close_left_100",
                        device="cuda",
                        **settings,
                    )
                policy.plugin.inference_seed = seed
                trace = None
                if args.trace:
                    trace_path = directory / f"seed_{seed}_trace_0"
                    attempt = 0
                    while trace_path.exists():
                        attempt += 1
                        trace_path = directory / f"seed_{seed}_trace_{attempt}"
                    trace = EpisodeDiagnostic(trace_path, bench.config.control_hz)
                try:
                    score = bench.run_episode(policy, seed, diagnostic=trace)
                finally:
                    if trace:
                        trace.close()
                result = score.to_dict()
                result.update(inference_seed=seed, settings=settings)
                result_path.write_text(json.dumps(result, indent=2))
                rows.append(result)
                print(
                    name,
                    f"seed={seed}",
                    f"target={result['cookies_in_target']}",
                    f"source={result['cookies_in_source']}",
                    f"steps={result['steps']}",
                    result["failure_reason"],
                    flush=True,
                )
        finally:
            if policy is not None:
                policy.close()
        summary[name] = dict(
            successes=sum(r["success"] for r in rows),
            episodes=len(rows),
            mean_target=sum(r["cookies_in_target"] for r in rows) / len(rows),
        )
        (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
