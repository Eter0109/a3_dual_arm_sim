"""Fresh processes release GPU memory between training and fixed simulation evaluation."""

import argparse
import hashlib
import os
from pathlib import Path

from .config import read_json, write_json
from .study import file_hash


def checkpoint_hashes(checkpoint):
    return {
        path.relative_to(checkpoint).as_posix(): file_hash(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }


def train(request, output):
    from a3_dual_arm_sim.learning.training import train_smolvla

    protocol, config = request["protocol"], request["config"]
    result = train_smolvla(
        dataset_root=Path(protocol["dataset_root"]),
        repo_id=protocol["repo_id"],
        base_model=Path(protocol["base_model"]),
        output_dir=Path(request["model_output"]),
        steps=protocol["steps"],
        batch_size=config["batch_size"],
        seed=protocol["train_seed"],
        device=protocol["device"],
        lr=config["lr"],
        decay_lr=config["decay_lr"],
        warmup_steps=config["warmup_steps"],
        save_freq=protocol["save_freq"],
        num_workers=protocol["num_workers"],
        offline=protocol["offline"],
    )
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.parent.name.isdigit() or int(checkpoint.parent.name) != protocol["steps"]:
        raise RuntimeError("Training did not complete the fixed update budget")
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError("Training produced no checkpoint weights")
    write_json(output, result)


def evaluate(request, output):
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark, SmolVLAPolicyAdapter
    from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic

    protocol, config = request["protocol"], request["config"]
    spec = protocol[request["split"]]
    checkpoint = Path(request["checkpoint"])
    settings = read_json(protocol["randomization_config"])
    report = {"synthetic": False, "episodes": [], "checkpoint": str(checkpoint)}
    report["checkpoint_hashes"] = checkpoint_hashes(checkpoint)
    if request.get("expected_checkpoint_hashes") not in (None, report["checkpoint_hashes"]):
        raise RuntimeError("Checkpoint files changed after development evaluation")
    with (checkpoint / "model.safetensors").open("rb") as stream:
        report["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    report["checkpoint_config"] = read_json(checkpoint / "config.json")

    class TraceBenchmark(CookieBatchBenchmark):
        def run_episode(self, policy, seed, episode_idx=1, **kwargs):
            trace = None
            trace_root = Path(output).parent / "traces" / case_name / f"seed_{seed}"
            if episode_idx <= protocol["trace_episodes"]:
                trace = EpisodeDiagnostic(trace_root, self.config.control_hz)
            try:
                result = super().run_episode(policy, seed, episode_idx, diagnostic=trace, **kwargs)
                if trace:
                    write_json(trace_root / "result.json", result.to_dict())
                return result
            finally:
                if trace:
                    trace.close()

    policy = SmolVLAPolicyAdapter(
        checkpoint,
        dataset_root=protocol["dataset_root"],
        repo_id=protocol["repo_id"],
        device=protocol["device"],
        inference_seed=protocol["train_seed"],
        n_action_steps=config["n_action_steps"],
    )
    try:
        for case in protocol["cases"]:
            case_name = f"{case['profile']}_column_{case['source_column']}"
            benchmark = TraceBenchmark(
                config_path=protocol["scene_config"],
                max_steps=protocol["max_steps"],
                profile=case["profile"],
                source_column=str(case["source_column"]),
                randomization_settings=settings,
            )

            def save_episode(episode, case=case, case_name=case_name):
                row = {
                    **episode.to_dict(),
                    "profile": case["profile"],
                    "column_mode": str(case["source_column"]),
                }
                if episode.episode <= protocol["trace_episodes"]:
                    row["trace"] = str(
                        Path(output).parent / "traces" / case_name / f"seed_{episode.seed}"
                    )
                report["episodes"].append(row)
                write_json(Path(output).with_suffix(".partial.json"), report)

            benchmark.evaluate(
                policy,
                num_episodes=spec["episodes"],
                seed_start=spec["seed_start"],
                workers=1,
                on_episode_end=save_episode,
            )
    finally:
        policy.close()
    if checkpoint_hashes(checkpoint) != report["checkpoint_hashes"]:
        raise RuntimeError("Checkpoint files changed during evaluation")
    write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    if request["protocol"]["offline"]:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    {"train": train, "evaluate": evaluate}[args.stage](request, args.output)


if __name__ == "__main__":
    main()
