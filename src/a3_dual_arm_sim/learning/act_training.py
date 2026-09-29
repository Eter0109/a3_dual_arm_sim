"""Train fresh ACT policies with the public LeRobot API."""

from __future__ import annotations

import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

from a3_dual_arm_sim.data.audit import CAMERA_KEYS, audit_training_dataset


def validate_act_horizon(
    chunk_size: int, n_action_steps: int, temporal_ensemble_coeff: float | None = None
) -> None:
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (chunk_size, n_action_steps)):
        raise ValueError("chunk_size and n_action_steps must be integers")
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError("Require 1 <= n_action_steps <= chunk_size")
    if temporal_ensemble_coeff is not None:
        if not math.isfinite(temporal_ensemble_coeff):
            raise ValueError("temporal_ensemble_coeff must be finite")
        if n_action_steps != 1:
            raise ValueError("ACT temporal ensembling requires n_action_steps=1")


def train_act(
    *,
    dataset_root: Path,
    output_dir: Path,
    repo_id: str = "Eter0109/a3-front-close-left-100",
    steps: int = 20000,
    batch_size: int = 8,
    seed: int = 1000,
    device: str = "cuda",
    lr: float = 1e-5,
    chunk_size: int = 50,
    n_action_steps: int = 8,
    temporal_ensemble_coeff: float | None = None,
    save_freq: int = 2000,
    num_workers: int = 2,
    pretrained_backbone: bool = True,
    dry_run: bool = False,
) -> dict:
    if (
        min(steps, batch_size, save_freq) <= 0
        or num_workers < 0
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError(
            "Positive steps, batch_size, save_freq, lr and non-negative workers required"
        )
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    validate_act_horizon(chunk_size, n_action_steps, temporal_ensemble_coeff)
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists: {output_dir}")
    audit = audit_training_dataset(dataset_root, repo_id=repo_id)
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    dim = audit["action_dim"]
    policy = ACTConfig(
        device=device,
        push_to_hub=False,
        use_amp=False,
        input_features={
            **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in CAMERA_KEYS},
            "observation.state": PolicyFeature(FeatureType.STATE, (dim,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (dim,))},
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        optimizer_lr=lr,
        optimizer_lr_backbone=lr,
        pretrained_backbone_weights=(
            "ResNet18_Weights.IMAGENET1K_V1" if pretrained_backbone else None
        ),
    )
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, root=str(dataset_root), video_backend="pyav"),
        policy=policy,
        output_dir=output_dir,
        job_name="a3_act",
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_freq=0,
        log_freq=min(20, steps),
        save_freq=save_freq,
        save_checkpoint=True,
    )
    config_dir = output_dir.parent / f".{output_dir.name}_act_config"
    config.save_pretrained(config_dir)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={config_dir / 'train_config.json'}",
    ]
    marker = dataset_root / ".a3_download_revision.json"
    report = {
        "command": command,
        "config": config.to_dict(),
        "audit": audit,
        "dataset_revision": json.loads(marker.read_text()) if marker.exists() else None,
        "runtime": {
            name: importlib.metadata.version(name) for name in ("lerobot", "torch", "torchvision")
        },
        "dry_run": dry_run,
    }
    launch = output_dir.parent / f"{output_dir.name}_launch.json"
    launch.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if not dry_run:
        subprocess.run(command, check=True)
    return report


def main(args) -> int:
    result = train_act(
        dataset_root=args.root,
        output_dir=args.output,
        repo_id=args.repo_id,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        pretrained_backbone=args.pretrained_backbone,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0
