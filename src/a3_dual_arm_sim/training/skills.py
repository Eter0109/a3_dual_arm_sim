"""Prepare skill-policy training; only --run starts a training process.

The four skill episodes from one parent rollout always stay in the same split.
The validation manifest reserves data; this launcher does NOT perform online
MuJoCo evaluation or claim a validation success rate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from a3_dual_arm_sim.core.contracts import FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE, STATE
from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.training.policies import prepare_a3_smolvla_source

CAMERAS = (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)


def grouped_skill_split(
    segments: list[dict[str, Any]], *, validation_fraction: float = 0.2, seed: int = 0,
) -> dict[str, Any]:
    """Deterministic split by parent seed, never independently by skill episode."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[int, list[dict[str, Any]]] = {}
    seen_episodes: set[int] = set()
    for segment in segments:
        episode = int(segment["episode_index"])
        if episode in seen_episodes:
            raise ValueError(f"Duplicate skill episode index {episode}")
        seen_episodes.add(episode)
        groups.setdefault(int(segment["parent_seed"]), []).append(segment)
    if len(groups) < 2:
        raise ValueError("Need at least two successful parent rollouts for a grouped holdout")
    expected = [(0, "PICK_FIVE"), (0, "PLACE_FIVE"), (1, "PICK_FIVE"), (1, "PLACE_FIVE")]
    for parent, items in groups.items():
        items.sort(key=lambda item: int(item["episode_index"]))
        if [(int(item["batch_index"]), item["skill"]) for item in items] != expected:
            raise ValueError(f"Parent seed {parent} must contain four PICK/PLACE/PICK/PLACE episodes")
        if len({item["source_column_number"] for item in items}) != 1:
            raise ValueError(f"Parent seed {parent} has inconsistent source-column labels")
    parents = sorted(groups)
    random.Random(seed).shuffle(parents)
    validation_count = min(len(parents) - 1, max(1, round(len(parents) * validation_fraction)))
    validation_parents = sorted(parents[:validation_count])
    train_parents = sorted(parents[validation_count:])

    def episodes(selected):
        return sorted(int(item["episode_index"]) for parent in selected for item in groups[parent])

    def source_counts(selected):
        return dict(sorted(Counter(str(groups[parent][0]["source_column_number"])
                                   for parent in selected).items()))

    return {
        "split_method": "deterministic_parent_seed_groups",
        "split_seed": seed,
        "requested_validation_fraction": validation_fraction,
        "train_parent_seeds": train_parents,
        "validation_parent_seeds": validation_parents,
        "train_episode_indices": episodes(train_parents),
        "validation_episode_indices": episodes(validation_parents),
        "train_source_column_counts": source_counts(train_parents),
        "validation_source_column_counts": source_counts(validation_parents),
    }


def build_skill_train_command(
    *,
    dataset_root: Path,
    repo_id: str,
    policy_type: str,
    base_model: Path,
    policy_source: Path,
    output_dir: Path,
    train_episodes: list[int],
    policy_features: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
    steps: int = 20000,
    save_freq: int = 5000,
    batch_size: int | None = None,
    seed: int = 0,
    device: str = "cuda",
    num_workers: int = 0,
) -> list[str]:
    if policy_type not in {"smolvla", "pi05"}:
        raise ValueError("policy_type must be smolvla or pi05")
    batch_size = batch_size if batch_size is not None else (4 if policy_type == "smolvla" else 1)
    if steps < 1 or batch_size < 1 or save_freq < 1 or num_workers < 0:
        raise ValueError("steps/batch_size/save_freq must be positive and num_workers non-negative")
    if not train_episodes or len(set(train_episodes)) != len(train_episodes):
        raise ValueError("train_episode_indices must be nonempty and unique")
    command = [
        sys.executable, "-m", "a3_dual_arm_sim.training.lerobot_fast_dataset",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={dataset_root}",
        f"--dataset.episodes={json.dumps(train_episodes, separators=(',', ':'))}",
        "--dataset.video_backend=pyav", f"--output_dir={output_dir}",
        f"--job_name=a3_cookie_skills_{policy_type}", f"--seed={seed}",
        f"--num_workers={num_workers}", f"--batch_size={batch_size}", f"--steps={steps}",
        "--eval_freq=0", "--log_freq=100", "--save_checkpoint=true",
        f"--save_freq={min(save_freq, steps)}", "--wandb.enable=false",
        f"--policy.device={device}", "--policy.push_to_hub=false",
    ]
    if policy_type == "smolvla":
        command.append(f"--policy.path={policy_source}")
    else:
        # A3 actions stored by the recorder are absolute joints. MEAN_STD is an
        # explicit training choice for these datasets, not a quantile fallback.
        normalization = {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
        command.extend([
            "--policy.type=pi05", f"--policy.pretrained_path={base_model}",
            f"--policy.input_features={json.dumps(policy_features, separators=(',', ':'))}",
            '--policy.output_features={"action":{"type":"ACTION","shape":[16]}}',
            f"--policy.normalization_mapping={json.dumps(normalization, separators=(',', ':'))}",
            "--policy.use_relative_actions=false", "--policy.dtype=bfloat16",
            "--policy.gradient_checkpointing=true", "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true", "--policy.n_action_steps=10",
        ])
        for key in ("paligemma_variant", "action_expert_variant", "max_state_dim",
                    "max_action_dim", "chunk_size"):
            if key in base_config:
                command.append(f"--policy.{key}={base_config[key]}")
    return command


def prepare_training(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    audit = audit_training_dataset(root, repo_id=args.repo_id)
    summary = json.loads((root / "collection_summary.json").read_text(encoding="utf-8"))
    if summary.get("task") != "cookie_skills":
        raise ValueError("Use the labeled cookie skill dataset, not the old whole-task baseline")
    segments = [json.loads(line) for line in
                (root / "a3_skill_segments.jsonl").read_text(encoding="utf-8").splitlines() if line]
    split = grouped_skill_split(segments, validation_fraction=args.validation_fraction, seed=args.seed)
    config_path = base_model / "config.json"
    if not config_path.is_file() or not (base_model / "model.safetensors").is_file():
        raise FileNotFoundError("--base-model must be a local LeRobot checkpoint with its weights")
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    if base_config.get("type") != args.policy_type:
        raise ValueError("--policy-type does not match --base-model config")
    if min(int(base_config.get("max_state_dim", 0)),
           int(base_config.get("max_action_dim", 0))) < 16:
        raise ValueError("Base checkpoint cannot represent the 16D A3 state/action contract")
    if base_config.get("use_relative_actions", False):
        raise ValueError("Use an absolute-action base checkpoint for A3 skill training")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    policy_features = {STATE: {"type": "STATE", "shape": [16]}}
    for key in CAMERAS:
        height, width, channels = info["features"][key]["shape"]
        if channels != 3:
            raise ValueError(f"Invalid RGB camera shape: {key}")
        policy_features[key] = {"type": "VISUAL", "shape": [3, height, width]}
    source = output.parent / f".{output.name}_smolvla_source"
    manifest_path = (args.manifest or output.parent / f"{output.name}_validation_manifest.json").resolve()
    command = build_skill_train_command(
        dataset_root=root, repo_id=args.repo_id, policy_type=args.policy_type,
        base_model=base_model, policy_source=source, output_dir=output,
        train_episodes=split["train_episode_indices"], policy_features=policy_features,
        base_config=base_config, steps=args.steps, batch_size=args.batch_size,
        save_freq=getattr(args, "save_freq", 5000), seed=args.seed, device=args.device,
        num_workers=args.num_workers,
    )
    manifest = {
        "schema_version": 1, "dataset_root": str(root), "repo_id": args.repo_id,
        "dataset_git_commit": summary.get("git_commit"), "policy_type": args.policy_type,
        "base_model": str(base_model), "training_output": str(output),
        "normalization": "checkpoint" if args.policy_type == "smolvla" else "MEAN_STD",
        "normalization_stats_scope": "existing full dataset metadata; train-only stats not recomputed",
        "evaluation_note": "Reserved heldout episodes only; no online evaluation performed",
        **split,
    }
    result = {"dry_run": not args.run, "audit": audit, "manifest_path": str(manifest_path),
              "manifest": manifest, "command": command}
    if args.run:
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite validation manifest: {manifest_path}")
        if manifest_path == output or output in manifest_path.parents:
            raise ValueError("Keep manifest outside training output so LeRobot can create that directory")
        if args.policy_type == "smolvla":
            if source.exists():
                raise FileExistsError(f"Prepared policy source already exists: {source}")
            prepare_a3_smolvla_source(base_model, source, device=args.device)
            prepared_config_path = source / "config.json"
            prepared_config = json.loads(prepared_config_path.read_text(encoding="utf-8"))
            prepared_config["input_features"] = policy_features
            prepared_config_path.write_text(json.dumps(prepared_config, indent=2) + "\n", encoding="utf-8")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        environment = os.environ.copy()
        environment["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(command, check=True, env=environment)
    return result
