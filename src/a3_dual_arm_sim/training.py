from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def audit_training_dataset(root: Path, *, repo_id: str) -> dict[str, Any]:
    """Fail fast on the parts of the LeRobot v3 contract used by SmolVLA."""
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    metadata_path = root / "a3_episode_metadata.jsonl"
    collection_summary_path = root / "collection_summary.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing A3 episode metadata: {metadata_path}")
    if not collection_summary_path.is_file():
        raise FileNotFoundError(f"Missing grasp collection summary: {collection_summary_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    collection_summary = json.loads(collection_summary_path.read_text(encoding="utf-8"))
    if collection_summary.get("schema_version", 0) < 2:
        raise ValueError(
            "Dataset uses the obsolete transient-lift success contract; recollect it with the "
            "current stable-grasp expert"
        )
    if collection_summary.get("task") not in {"a3_grasp", "cookie_transfer"}:
        raise ValueError("Unsupported A3 training task")
    if (
        collection_summary.get("task") == "cookie_transfer"
        and collection_summary.get("stored_action_mode") != "joint_position"
    ):
        raise ValueError("Cookie SmolVLA datasets require joint_position actions")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("SmolVLA training requires a LeRobot v3.0 dataset")
    features = info.get("features", {})
    expected_shapes = {
        **{key: [256, 256, 3] for key in CAMERA_KEYS},
        "observation.state": [16],
        "action": [16],
    }
    for key, shape in expected_shapes.items():
        actual = features.get(key, {}).get("shape")
        if actual != shape:
            raise ValueError(f"Feature {key!r} has shape {actual}, expected {shape}")

    episodes = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not episodes or not all(item.get("success") is True for item in episodes):
        raise ValueError("Training dataset must contain at least one successful expert episode")
    if info.get("total_episodes") != len(episodes):
        raise ValueError("LeRobot and A3 episode metadata counts do not match")

    return {
        "root": str(root),
        "repo_id": repo_id,
        "episodes": len(episodes),
        "frames": int(info["total_frames"]),
        "tasks": int(info["total_tasks"]),
        "camera_keys": list(CAMERA_KEYS),
        "state_dim": 16,
        "action_dim": 16,
    }


def default_base_model() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "vla_ur5e_sim"
        / "assets"
        / "policy"
        / "base"
        / "pretrained_model"
    )


def _local_hub_snapshot(repo_id: str) -> Path | None:
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
    )
    repository = cache_root / f"models--{repo_id.replace('/', '--')}"
    main_ref = repository / "refs" / "main"
    if not main_ref.is_file():
        return None
    snapshot = repository / "snapshots" / main_ref.read_text(encoding="utf-8").strip()
    return snapshot if (snapshot / "config.json").is_file() else None


def prepare_a3_smolvla_source(base_model: Path, runtime_dir: Path, *, device: str) -> Path:
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
    if int(config.get("max_state_dim", 0)) < 16 or int(config.get("max_action_dim", 0)) < 16:
        raise ValueError("The base checkpoint cannot represent the A3 16D state/action contract")

    visual = {"type": "VISUAL", "shape": [3, 256, 256]}
    config["input_features"] = {
        **{key: visual.copy() for key in CAMERA_KEYS},
        "observation.state": {"type": "STATE", "shape": [16]},
    }
    config["output_features"] = {"action": {"type": "ACTION", "shape": [16]}}
    config["device"] = device
    config["use_amp"] = device == "cuda"
    config["push_to_hub"] = False
    config["repo_id"] = None
    vlm_model_name = config.get("vlm_model_name")
    cached_vlm = _local_hub_snapshot(vlm_model_name) if vlm_model_name else None
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
    if cached_vlm is not None and preprocessor_path.is_file():
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                step["config"]["tokenizer_name"] = str(cached_vlm)
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
) -> list[str]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    return [
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
        "--num_workers=0",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--eval_freq=0",
        "--log_freq=1",
        "--save_checkpoint=true",
        f"--save_freq={steps}",
        "--wandb.enable=false",
    ]


def train_smolvla(
    *,
    dataset_root: Path,
    repo_id: str,
    base_model: Path,
    output_dir: Path,
    steps: int,
    batch_size: int,
    seed: int,
    device: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    audit = audit_training_dataset(dataset_root, repo_id=repo_id)
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists: {output_dir}")
    policy_source = prepare_a3_smolvla_source(
        base_model,
        output_dir.parent / ".a3_smolvla_source",
        device=device,
    )
    command = build_train_command(
        dataset_root=dataset_root,
        repo_id=repo_id,
        policy_source=policy_source,
        output_dir=output_dir,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )
    result = {**audit, "device": device, "output_dir": str(output_dir), "command": command}
    if dry_run:
        return result

    env = os.environ.copy()
    env.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    )
    subprocess.run(command, check=True, env=env)
    checkpoints = sorted((output_dir / "checkpoints").glob("*/pretrained_model/config.json"))
    if not checkpoints:
        raise RuntimeError("LeRobot exited without writing a final SmolVLA checkpoint")
    result["checkpoint"] = str(checkpoints[-1].parent)
    return result
