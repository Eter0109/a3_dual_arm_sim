"""Evaluate final cookie counts on both tasks and all three randomization profiles.

Run from the repository root:
    MUJOCO_GL=egl python -m a3_dual_arm_sim.workflows.evaluate_smolvla_grid_cli \
        --max-steps 500 --episodes 20 --output outputs/smolvla_grid_500

Each round evaluates the selected cells on the same seed. Prompt styles are
explicit per task; 2x10 evaluation with a 2x5 checkpoint is transfer testing.
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

from a3_dual_arm_sim.paths import project_root, resource_root
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


def source_revision():
    """Only report the source checkout revision, never a wheel user's unrelated repo."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def summarize(rows, planned, profiles=PROFILES):
    result = []
    for task in TASKS:
        for profile in profiles:
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
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output", type=Path, default=root / "outputs/smolvla_grid_500")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per cell")
    parser.add_argument("--seed-start", type=int, default=13000)
    parser.add_argument("--max-steps", type=int, default=500, help="Same budget for both tasks")
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=25, help="Flow inference steps")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument(
        "--prompt-style-2x5", choices=("training", "task", "short"), default="training"
    )
    parser.add_argument(
        "--prompt-style-2x10", choices=("training", "task", "short"), default="task"
    )
    parser.add_argument("--workers", type=int, default=1, help="CPU environments sharing one model")
    parser.add_argument(
        "--inactive-right-arm",
        choices=("auto", "off"),
        default="auto",
        help="auto: hold only a demonstrated constant right arm; off: use raw right state/actions",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print experiment settings without loading a policy"
    )
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.n_action_steps, args.num_steps, args.workers) < 1:
        parser.error("episode and step counts must be positive")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must not contain duplicates")
    prompt_styles = {"2x5": args.prompt_style_2x5, "2x10": args.prompt_style_2x10}
    checkpoint, dataset, output = (
        args.checkpoint.resolve(),
        args.dataset_root.resolve() if args.dataset_root else None,
        args.output.resolve(),
    )
    settings5 = load_randomization_config(resource_root() / "configs/randomization.yaml")
    settings10, first_grasp = load_cookie_2x10_settings(
        resource_root() / "configs/randomization_2x10.yaml"
    )
    for settings in (settings5, settings10):
        settings["source_column"] = "1"
    protocol = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset) if dataset else None,
        "repo_id": args.repo_id,
        "episodes_per_cell": args.episodes,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "source_column": "1",
        "first_grasp_2x10": first_grasp,
        "n_action_steps": args.n_action_steps,
        "num_steps": args.num_steps,
        "device": args.device,
        "workers": args.workers,
        "gpu_model_instances": 1 if args.device == "cuda" else 0,
        "inactive_right_arm_mode": args.inactive_right_arm,
        "profiles": args.profiles,
        "prompt_styles": prompt_styles,
        "metric": "final cookies_in_target, mean and total; not peak count or success rate",
        "prompt_2x5": (
            "transfer 10 cookies into target box"
            if prompt_styles["2x5"] in ("training", "short")
            else "column-specific task prompt; actual text saved per episode"
        ),
        "prompt_2x10": (
            "transfer 20 cookies into target box"
            if prompt_styles["2x10"] == "short"
            else "native Cookie2x10Plan prompt; actual text saved per episode"
        ),
        "postprocessing": "optional inactive right arm auto-hold; no geometric or gripper rules; tokenizer unchanged",
        "randomization_2x5": settings5,
        "randomization_2x10": settings10,
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "preprocessor": (
            json.loads((checkpoint / "policy_preprocessor.json").read_text())
            if (checkpoint / "policy_preprocessor.json").is_file()
            else None
        ),
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2, ensure_ascii=False))
        return 0
    import torch

    from a3_dual_arm_sim.learning.checkpoint_bundle import inference_fingerprints

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU-enabled execution environment")
    protocol.update(
        {
            "weights_sha256": digest(checkpoint / "model.safetensors"),
            "dataset_stats_sha256": digest(dataset / "meta/stats.json") if dataset else None,
            "processor_sha256": inference_fingerprints(checkpoint, dataset),
            "script_sha256": digest(Path(__file__)),
            "shared_source_sha256": {
                name: digest(Path(__file__).resolve().parents[1] / name)
                for name in (
                    "workflows/benchmark.py",
                    "workflows/episode_execution.py",
                    "workflows/cookie_2x10_execution.py",
                    "policies/smolvla.py",
                    "workflows/shared_smolvla_grid.py",
                )
            },
            "git_commit": source_revision(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
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
        results = summarize(rows, args.episodes, args.profiles)
        atomic_json(
            output / "summary.json",
            {
                "status": status,
                "max_steps": args.max_steps,
                "results": results,
                "completed": len(rows),
                "planned": 6 * args.episodes,
                "workers": args.workers,
            },
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
    policy = None

    def record_row(row):
        key = (row["benchmark"], row["profile"], row["seed"])
        if key in done:
            raise RuntimeError("Duplicate episode completion")
        with rows_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        done.add(key)
        save_summary("running")
        print(
            f"DONE {key} final={row['cookies_in_target']}/{row['max_score']} steps={row['steps']}",
            flush=True,
        )

    try:
        if args.workers > 1:
            from a3_dual_arm_sim.workflows.shared_smolvla_grid import run_parallel_grid

            jobs = []
            for episode, seed in enumerate(
                range(args.seed_start, args.seed_start + args.episodes), 1
            ):
                for task in TASKS:
                    for profile in args.profiles:
                        if (task, profile, seed) in done:
                            continue
                        settings = dict(settings5 if task == "2x5" else settings10, profile=profile)
                        options = {
                            "prompt_style": prompt_styles[task],
                            "profile": profile,
                            "source_column": "1",
                            "randomization_settings": settings,
                            "max_steps": args.max_steps,
                        }
                        if task == "2x10":
                            options["first_grasp"] = first_grasp
                        jobs.append(
                            {
                                "benchmark": task,
                                "profile": profile,
                                "seed": seed,
                                "episode": episode,
                                "options": options,
                            }
                        )
            run_parallel_grid(
                {
                    "checkpoint": str(checkpoint),
                    "dataset_root": str(dataset) if dataset is not None else None,
                    "repo_id": args.repo_id,
                    "device": args.device,
                    "n_action_steps": args.n_action_steps,
                    "num_steps": args.num_steps,
                    "inactive_right_arm_mode": args.inactive_right_arm,
                },
                jobs,
                args.workers,
                output,
                record_row,
            )
            save_summary("completed")
            return 0
        policy = SmolVLAPolicyAdapter(
            checkpoint,
            dataset_root=dataset,
            repo_id=args.repo_id,
            device=args.device,
            n_action_steps=args.n_action_steps,
            num_steps=args.num_steps,
            inactive_right_arm_mode=args.inactive_right_arm,
        )
        runners = {}
        for task in TASKS:
            for profile in args.profiles:
                settings = dict(settings5 if task == "2x5" else settings10, profile=profile)
                extra = {"first_grasp": first_grasp} if task == "2x10" else {}
                runners[task, profile] = make_cookie_benchmark(
                    task=task,
                    prompt_style=prompt_styles[task],
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
                row = {
                    **asdict(score),
                    "benchmark": task,
                    "profile": profile,
                    "prompt_style": prompt_styles[task],
                    "task_instruction": benchmark._episode_task(1, seed),
                }
                record_row(row)
        save_summary("completed")
        print("ALL SELECTED TESTS COMPLETE", flush=True)
    except KeyboardInterrupt:
        save_summary("stopped")
        print("Stopped; incomplete episode excluded. Use --resume to continue.", flush=True)
        return 130
    except Exception as exc:
        atomic_json(output / "error.json", {"error": repr(exc)})
        save_summary("error")
        raise
    finally:
        if policy is not None:
            policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
