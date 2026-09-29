"""Prepare an isolated merged dataset, fixed holdout, and training command after the development gate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from a3_dual_arm_sim.learning.training import (
    build_train_command,
    default_base_model,
    prepare_a3_smolvla_source,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    if not gate["stage_three_required"]:
        raise RuntimeError("Development gate does not require retraining")
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from a3_dual_arm_sim.data.audit import audit_training_dataset

    for dataset in (args.original, args.additional):
        audit = audit_training_dataset(dataset, repo_id="local/a3")
        if audit["episodes"] != 100:
            raise ValueError("Expected exactly 100 successful episodes per source")
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = [
        LeRobotDataset("local/a3", root=p, video_backend="pyav")
        for p in (args.original, args.additional)
    ]
    for d in datasets:
        for k in d.meta.stats:
            if "images" in k and "std" in d.meta.stats[k]:
                d.meta.stats[k]["std"] = np.ones((3, 1, 1), dtype=np.float32)
    merged = merge_datasets(datasets, "local/a3-improvement-200", args.output / "dataset")
    train = list(range(80)) + list(range(100, 200))
    validation = list(range(80, 100))
    split = {
        "train_episodes": train,
        "validation_episodes": validation,
        "original_train": list(range(80)),
        "original_validation": validation,
    }
    (args.output / "split.json").write_text(json.dumps(split, indent=2))
    per_episode = []
    for p in (merged.root / "meta/episodes").rglob("*.parquet"):
        for row in pq.read_table(p).to_pylist():
            if row["episode_index"] not in train:
                continue
            stats = {}
            for key, value in row.items():
                if key.startswith("stats/"):
                    _, feature, stat = key.split("/", 2)
                    stats.setdefault(feature, {})[stat] = np.asarray(value)
            if not stats:
                raise RuntimeError("Missing episode statistics")
            per_episode.append(stats)
    if len(per_episode) != 180:
        raise RuntimeError("Training statistics episode count mismatch")
    stats = aggregate_stats(per_episode)
    for key in ("action", "observation.state"):
        stats[key]["std"][8:] = np.maximum(stats[key]["std"][8:], 0.05)
    (merged.root / "meta/stats.json").write_text(
        json.dumps(stats, default=lambda x: x.tolist(), indent=2)
    )
    source = prepare_a3_smolvla_source(
        default_base_model(), args.output / "policy_source", device="cuda", decay_steps=20000
    )
    command = build_train_command(
        dataset_root=merged.root,
        repo_id="local/a3-improvement-200",
        policy_source=source,
        output_dir=args.output / "model",
        steps=20000,
        batch_size=32,
        seed=1000,
        save_freq=2500,
    )
    command = [x for x in command if not x.startswith("--eval_steps=")]
    command += [
        "--eval_steps=2000",
        "--dataset.eval_split=0.1",
        "--dataset.episodes=" + json.dumps(train + validation),
        "--accelerator.gradient_accumulation.steps=2",
        "--dataset.image_transforms.enable=true",
    ]
    (args.output / "train_command.json").write_text(json.dumps(command, indent=2))
    print(json.dumps({"split": split, "command": command}, indent=2))


if __name__ == "__main__":
    main()
