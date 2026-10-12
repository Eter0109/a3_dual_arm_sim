# Explicit same-name imports preserve the legacy public API.
# ruff: noqa: PLC0414
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from a3_dual_arm_sim.data.audit import CAMERA_KEYS as CAMERA_KEYS
from a3_dual_arm_sim.data.audit import audit_training_dataset as audit_training_dataset


def default_base_model() -> str:
    return "lerobot/smolvla_base"


def resolve_base_model(
    model: str | Path, *, revision: str | None = None, offline: bool = False
) -> Path:
    """Accept a local checkpoint or download an explicitly selected Hub revision."""
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()
    if isinstance(model, Path) or str(model).startswith(("/", ".", "~")):
        raise FileNotFoundError(f"Missing local base model: {model}")
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": str(model), "revision": revision}
    if offline:
        kwargs["local_files_only"] = True
    return Path(snapshot_download(**kwargs)).resolve()


def _local_hub_snapshot(repo_id: str) -> Path | None:
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        )
    )
    repository = cache_root / f"models--{repo_id.replace('/', '--')}"
    main_ref = repository / "refs" / "main"
    if not main_ref.is_file():
        return None
    snapshot = repository / "snapshots" / main_ref.read_text(encoding="utf-8").strip()
    return snapshot if (snapshot / "config.json").is_file() else None


def prepare_a3_smolvla_source(
    base_model: Path,
    runtime_dir: Path,
    *,
    device: str,
    lr: float = 5e-5,
    decay_lr: float = 5e-6,
    warmup_steps: int = 500,
    decay_steps: int = 20000,
    num_steps: int = 25,
    action_dim: int = 16,
    tokenizer_max_length: int | None = None,
) -> Path:
    """Create a lightweight adapted checkpoint view without copying the 1.2 GB weights."""
    base_model = base_model.expanduser().resolve()
    runtime_dir = runtime_dir.expanduser().resolve()
    config_path = base_model / "config.json"
    weights_path = base_model / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot pretrained policy directory: {base_model}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, got {config.get('type')!r}")
    if action_dim not in (8, 16):
        raise ValueError("action_dim must be 8 or 16")
    if tokenizer_max_length is not None and tokenizer_max_length <= 0:
        raise ValueError("tokenizer_max_length must be positive")
    if (
        int(config.get("max_state_dim", 0)) < action_dim
        or int(config.get("max_action_dim", 0)) < action_dim
    ):
        raise ValueError("The base checkpoint cannot represent the A3 16D state/action contract")

    visual = {"type": "VISUAL", "shape": [3, 256, 256]}
    config["input_features"] = {
        **{key: visual.copy() for key in CAMERA_KEYS},
        "observation.state": {"type": "STATE", "shape": [action_dim]},
    }
    config["output_features"] = {"action": {"type": "ACTION", "shape": [action_dim]}}
    config["device"] = device
    config["use_amp"] = device == "cuda"
    # The policy checkpoint restores the complete VLM and expert state.
    # Construct the VLM from config instead of downloading/loading its weights twice.
    config["load_vlm_weights"] = False
    config["push_to_hub"] = False
    config["repo_id"] = None

    # Training schedule for A3 expert demonstrations
    config["optimizer_lr"] = lr
    config["scheduler_decay_lr"] = decay_lr
    config["scheduler_warmup_steps"] = warmup_steps
    config["scheduler_decay_steps"] = decay_steps
    config["num_steps"] = num_steps
    if tokenizer_max_length is not None:
        config["tokenizer_max_length"] = tokenizer_max_length

    vlm_model_name = config.get("vlm_model_name")
    dependencies = base_model / "a3_dependencies.json"
    cached_vlm = _local_hub_snapshot(vlm_model_name) if vlm_model_name else None
    if dependencies.is_file():
        dependency = json.loads(dependencies.read_text())
        pinned = Path(dependency["root"])
        if dependency["repo_id"] != vlm_model_name or not (pinned / "config.json").is_file():
            raise ValueError("Downloaded VLM dependency is unavailable; rerun download-smolvla")
        cached_vlm = pinned
    if cached_vlm is not None:
        config["vlm_model_name"] = str(cached_vlm)

    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for source in base_model.iterdir():
        if source.name in {"config.json", "model.safetensors", "train_config.json"}:
            continue
        if source.is_file():
            shutil.copy2(source, runtime_dir / source.name)
    preprocessor_path = runtime_dir / "policy_preprocessor.json"
    if preprocessor_path.is_file():
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                if cached_vlm is not None:
                    step["config"]["tokenizer_name"] = str(cached_vlm)
                if tokenizer_max_length is not None:
                    step["config"]["max_length"] = tokenizer_max_length
        preprocessor_path.write_text(json.dumps(preprocessor, indent=2) + "\n", encoding="utf-8")
    runtime_weights = runtime_dir / "model.safetensors"
    if runtime_weights.exists():
        try:
            if not runtime_weights.samefile(weights_path):
                runtime_weights.unlink()
        except OSError:
            runtime_weights.unlink()
    if not runtime_weights.exists():
        try:
            runtime_weights.symlink_to(weights_path)
        except OSError:
            try:
                runtime_weights.hardlink_to(weights_path)
            except OSError:
                shutil.copy2(weights_path, runtime_weights)
    return runtime_dir


def build_train_command(
    *,
    dataset_root: Path,
    repo_id: str,
    policy_source: Path,
    output_dir: Path,
    steps: int,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
    use_ema: bool = False,
    ema_decay: float = 0.99,
    save_freq: int = 2000,
) -> list[str]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root.expanduser().resolve()}",
        "--dataset.video_backend=pyav",
        f"--policy.path={policy_source.expanduser().resolve()}",
        f"--output_dir={output_dir.expanduser().resolve()}",
        "--job_name=a3_grasp_smolvla",
        f"--seed={seed}",
        f"--num_workers={num_workers}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--eval_freq=0",
        f"--log_freq={min(20, steps)}",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        "--wandb.enable=false",
    ]
    if use_ema:
        raise ValueError(
            "Public LeRobot 0.5.1 does not support EMA training; use_ema must be False"
        )
    return cmd


def train_smolvla(
    *,
    dataset_root: Path,
    repo_id: str,
    base_model: str | Path,
    output_dir: Path,
    steps: int = 10000,
    batch_size: int = 16,
    seed: int = 1000,
    device: str = "cuda",
    lr: float = 5e-5,
    decay_lr: float = 5e-6,
    warmup_steps: int = 500,
    save_freq: int = 2000,
    num_workers: int = 2,
    use_ema: bool = False,
    ema_decay: float = 0.99,
    dry_run: bool = False,
    model_revision: str | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    if steps <= 0 or batch_size <= 0 or save_freq <= 0 or num_workers < 0:
        raise ValueError(
            "steps, batch_size, save_freq must be positive and num_workers non-negative"
        )
    audit = audit_training_dataset(dataset_root, repo_id=repo_id)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists: {output_dir}")
    base_model = resolve_base_model(base_model, revision=model_revision, offline=offline)
    collection = json.loads((dataset_root / "collection_summary.json").read_text())
    policy_source = prepare_a3_smolvla_source(
        base_model,
        output_dir.parent / f".{output_dir.name}_a3_smolvla_source",
        device=device,
        lr=lr,
        decay_lr=decay_lr,
        warmup_steps=min(warmup_steps, max(1, steps - 1)),
        decay_steps=steps,
        action_dim=audit["action_dim"],
        # The four-grasp prompt exceeds the base checkpoint's 48-token limit.
        tokenizer_max_length=128 if collection.get("target_layout") == "2x10" else None,
    )
    command = build_train_command(
        dataset_root=dataset_root,
        repo_id=repo_id,
        policy_source=policy_source,
        output_dir=output_dir,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        use_ema=use_ema,
        ema_decay=ema_decay,
        save_freq=save_freq,
    )
    result = {**audit, "device": device, "output_dir": str(output_dir), "command": command}
    if dry_run:
        return result

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / f"{output_dir.name}_launch.json").write_text(
        json.dumps({**result, "base_model": str(base_model), "weights": "ordinary"}, indent=2)
        + "\n"
    )
    subprocess.run(command, check=True, env=env)
    checkpoints = sorted((output_dir / "checkpoints").glob("*/pretrained_model/config.json"))
    if not checkpoints:
        raise RuntimeError("LeRobot exited without writing a final SmolVLA checkpoint")
    numbered = [p for p in checkpoints if p.parent.parent.name.isdigit()]
    latest = max(numbered, key=lambda p: int(p.parent.parent.name)) if numbered else checkpoints[-1]
    result["checkpoint"] = str(latest.parent)
    return result
