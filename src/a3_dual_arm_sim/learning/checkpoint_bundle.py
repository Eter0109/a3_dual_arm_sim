"""Checkpoint-owned inference processors and local dataset provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def local_repo_id(root: Path, repo_id: str | None = None) -> str:
    return repo_id or "local/" + (
        re.sub(r"[^a-zA-Z0-9._-]+", "-", root.resolve().name).strip("-") or "dataset"
    )


def processor_files(checkpoint: Path) -> list[Path]:
    files = []
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = checkpoint / name
        files.append(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; supply --dataset-root or run checkpoint_bundle to complete this legacy checkpoint"
            )
        for step in json.loads(path.read_text()).get("steps", []):
            if step.get("state_file"):
                state = checkpoint / step["state_file"]
                if not state.is_file():
                    raise FileNotFoundError(f"Missing processor state: {state}")
                files.append(state)
    return files


def inference_fingerprints(checkpoint: Path, dataset_root: Path | None = None):
    try:
        files = processor_files(checkpoint)
    except FileNotFoundError:
        if dataset_root is None or any(
            (checkpoint / n).exists()
            for n in ("policy_preprocessor.json", "policy_postprocessor.json")
        ):
            raise
        files = [checkpoint / "config.json", dataset_root / "meta/stats.json"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def load_processors(config, checkpoint, dataset_root=None, device="cpu"):
    from lerobot.policies.factory import make_pre_post_processors

    try:
        processor_files(checkpoint)
    except FileNotFoundError:
        # Do not replace a partial/corrupt saved pipeline with guessed statistics.
        if any(
            (checkpoint / n).exists()
            for n in ("policy_preprocessor.json", "policy_postprocessor.json")
        ):
            raise
        if dataset_root is None:
            raise
        import torch

        stats = json.loads((Path(dataset_root) / "meta/stats.json").read_text())
        stats = {k: {n: torch.tensor(v) for n, v in values.items()} for k, values in stats.items()}
        for key, feature in {**config.input_features, **config.output_features}.items():
            if key in ("observation.state", "action") and (
                key not in stats
                or stats[key].get("mean", torch.empty(0)).numel() != feature.shape[0]
            ):
                raise ValueError(f"Dataset statistics do not match checkpoint feature {key}")
        return make_pre_post_processors(config, dataset_stats=stats)
    return make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )


def complete_checkpoint(checkpoint: Path, dataset_root: Path, repo_id=None, training_config=None):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    checkpoint, dataset_root = checkpoint.resolve(), dataset_root.resolve()
    if (
        not (checkpoint / "policy_preprocessor.json").exists()
        and not (checkpoint / "policy_postprocessor.json").exists()
    ):
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = "cpu"
        pre, post = load_processors(config, checkpoint, dataset_root)
        pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
        post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    files = processor_files(checkpoint)
    stats = (dataset_root / "meta/stats.json").read_bytes()
    record = {
        "repo_id": local_repo_id(dataset_root, repo_id),
        "dataset_root": str(dataset_root),
        "dataset_stats_sha256": hashlib.sha256(stats).hexdigest(),
        "processor_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "training_config": training_config,
    }
    (checkpoint / "a3_inference_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Complete legacy SmolVLA processors and record dataset provenance in place"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    print(
        json.dumps(complete_checkpoint(args.checkpoint, args.dataset_root, args.repo_id), indent=2)
    )


if __name__ == "__main__":
    main()
