"""Evaluate final cookie counts on both tasks and all three randomization profiles.

Run from the repository root:
    MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.evaluate_smolvla_grid_cli \
        --max-steps 500 --episodes 20 --output outputs/smolvla_grid_500

Each round evaluates all six cells on the same seed. The 2x5 task uses the
training instruction; 2x10 uses its native instruction and is transfer testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
from a3_dual_arm_sim.workflows.policy_adapters import SmolVLAPolicyAdapter

PROFILES = ("basic", "medium", "advanced")
TASKS = ("2x5", "2x10")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(rows, planned):
    result = []
    for task in TASKS:
        for profile in PROFILES:
            counts = [
                row["cookies_in_target"]
                for row in rows
                if row["benchmark"] == task and row["profile"] == profile
            ]
            capacity = 10 if task == "2x5" else 20
            result.append(
                {
                    "benchmark": task,
                    "profile": profile,
                    "completed": len(counts),
                    "planned": planned,
                    "mean_final_count": sum(counts) / len(counts) if counts else None,
                    "total_final_count": sum(counts),
                    "total_capacity": capacity * len(counts),
                    "min_final_count": min(counts) if counts else None,
                    "max_final_count": max(counts) if counts else None,
                    "count_distribution": dict(sorted(Counter(counts).items())),
                    "counts": counts,
                }
            )
    return result


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "outputs/smolvla_front_close_left_40k/model/checkpoints/040000/pretrained_model",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=root / "datasets/a3_front_close_left_100"
    )
    parser.add_argument("--repo-id", default="local/a3-front-close-left-100")
    parser.add_argument("--output", type=Path, default=root / "outputs/smolvla_grid_500")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per cell")
    parser.add_argument("--seed-start", type=int, default=13000)
    parser.add_argument("--max-steps", type=int, default=500, help="Same budget for both tasks")
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=25, help="Flow inference steps")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print experiment settings without loading a policy"
    )
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.n_action_steps, args.num_steps) < 1:
        parser.error("episode and step counts must be positive")
    checkpoint, dataset, output = (
        args.checkpoint.resolve(),
        args.dataset_root.resolve(),
        args.output.resolve(),
    )
    settings5 = load_randomization_config(root / "configs/randomization.yaml")
    settings10, first_grasp = load_cookie_2x10_settings(root / "configs/randomization_2x10.yaml")
    for settings in (settings5, settings10):
        settings["source_column"] = "1"
    protocol = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset),
        "repo_id": args.repo_id,
        "episodes_per_cell": args.episodes,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "source_column": "1",
        "first_grasp_2x10": first_grasp,
        "n_action_steps": args.n_action_steps,
        "num_steps": args.num_steps,
        "device": args.device,
        "metric": "final cookies_in_target, mean and total; not peak count or success rate",
        "prompt_2x5": "transfer 10 cookies into target box",
        "prompt_2x10": "native Cookie2x10Plan prompt; transfer evaluation",
        "postprocessing": "no extra rules; checkpoint tokenizer unchanged",
        "randomization_2x5": settings5,
        "randomization_2x10": settings10,
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "preprocessor": json.loads((checkpoint / "policy_preprocessor.json").read_text()),
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2, ensure_ascii=False))
        return 0
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU-enabled execution environment")
    protocol.update(
        {
            "weights_sha256": digest(checkpoint / "model.safetensors"),
            "dataset_stats_sha256": digest(dataset / "meta/stats.json"),
            "script_sha256": digest(Path(__file__)),
            "shared_source_sha256": {
                name: digest(root / "src/a3_dual_arm_sim" / name)
                for name in (
                    "workflows/benchmark.py",
                    "workflows/episode_execution.py",
                    "workflows/cookie_2x10_execution.py",
                    "policies/smolvla.py",
                )
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        }
    )
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise RuntimeError("Protocol changed; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "protocol.json", protocol)
    rows_path = output / "episodes.jsonl"
    rows = (
        [json.loads(line) for line in rows_path.read_text().splitlines()]
        if rows_path.exists()
        else []
    )
    done = {(r["benchmark"], r["profile"], r["seed"]) for r in rows}
    if len(done) != len(rows):
        raise RuntimeError("Duplicate episode records")
    (output / "pid").write_text(str(os.getpid()))

    def save_summary(status):
        results = summarize(rows, args.episodes)
        atomic_json(
            output / "summary.json",
            {"status": status, "max_steps": args.max_steps, "results": results},
        )
        lines = [
            f"Status: {status}; max_steps={args.max_steps}",
            "",
            "| Task | Profile | N | Mean final count | Total |",
            "|---|---|---:|---:|---:|",
        ]
        for item in results:
            mean = item["mean_final_count"]
            mean_text = f"{mean:.2f}" if mean is not None else "pending"
            lines.append(
                f"| {item['benchmark']} | {item['profile']} | {item['completed']}/{args.episodes} | {mean_text} | {item['total_final_count']}/{item['total_capacity']} |"
            )
        (output / "summary.md").write_text("\n".join(lines) + "\n")

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save_summary("running")
    try:
        policy = SmolVLAPolicyAdapter(
            checkpoint,
            dataset_root=dataset,
            repo_id=args.repo_id,
            device=args.device,
            n_action_steps=args.n_action_steps,
            num_steps=args.num_steps,
        )
        runners = {}
        for task in TASKS:
            for profile in PROFILES:
                settings = dict(settings5 if task == "2x5" else settings10, profile=profile)
                extra = {"first_grasp": first_grasp} if task == "2x10" else {}
                runners[task, profile] = make_cookie_benchmark(
                    task=task,
                    prompt_style="training",
                    profile=profile,
                    source_column="1",
                    randomization_settings=settings,
                    max_steps=args.max_steps,
                    **extra,
                )
        for episode, seed in enumerate(range(args.seed_start, args.seed_start + args.episodes), 1):
            for (task, profile), benchmark in runners.items():
                if (task, profile, seed) in done:
                    continue
                atomic_json(
                    output / "current.json", {"benchmark": task, "profile": profile, "seed": seed}
                )
                print(f"START {task} {profile} seed={seed}", flush=True)
                policy.plugin.inference_seed = seed
                evaluated = benchmark.evaluate(
                    policy, num_episodes=1, seed_start=seed, workers=1, verbose=False
                )
                score = evaluated.episodes[0]
                score.episode = episode
                row = {**asdict(score), "benchmark": task, "profile": profile}
                with rows_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
                done.add((task, profile, seed))
                save_summary("running")
                print(
                    f"DONE {task} {profile} seed={seed} final={score.cookies_in_target}/{score.max_score} steps={score.steps} wall={score.wall_seconds}s",
                    flush=True,
                )
        save_summary("completed")
        print("ALL SIX TESTS COMPLETE", flush=True)
    except KeyboardInterrupt:
        save_summary("stopped")
        print("Stopped; incomplete episode excluded. Use --resume to continue.", flush=True)
        return 130
    except Exception as exc:
        atomic_json(output / "error.json", {"error": repr(exc)})
        save_summary("error")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
