"""Answer-only, language-LoRA training for the A3 temporal visual verifier.

ML dependencies are deliberately lazy.  The simulator is never instantiated and
labels, fault metadata, and simulator state never become model inputs.  Each
JSONL row supplies the same image order and prompt as the runtime verifier.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image

from a3_dual_arm_sim.agents.prompts import verifier_completion_prompt, verifier_diagnosis_prompt
from a3_dual_arm_sim.core.skills import CookieSkill

DEFAULT_LORA_SUFFIXES = (
    "q_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
)
IGNORE_INDEX = -100
ALLOWED_ANSWERS = {
    "completion": frozenset(("Yes", "No")),
    "diagnosis": frozenset(("Stuck", "StillTrying")),
}


@dataclass(frozen=True)
class VerifierTrainingConfig:
    base_model: str = "models/Qwen3.5-4B"
    model_family: str = "qwen3_5"
    steps: int = 300
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    seed: int = 7
    image_max_edge: int = 256
    max_sequence_length: int = 4096
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    save_every: int = 100
    log_every: int = 10
    evaluation_limit: int | None = None
    generation_tokens: int = 8
    init_adapter: str | None = None

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("base_model must be a local checkpoint path")
        if self.init_adapter is not None and (
            not isinstance(self.init_adapter, str) or not self.init_adapter.strip()
        ):
            raise ValueError("init_adapter must be a local adapter path or None")
        if self.model_family not in ("qwen3_5", "qwen2_5_vl"):
            raise ValueError("unsupported verifier model family")
        for name in (
            "steps", "gradient_accumulation", "rank", "alpha", "max_sequence_length",
            "save_every", "log_every", "generation_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.image_max_edge) is not int or not 32 <= self.image_max_edge <= 2048:
            raise ValueError("image_max_edge must be in 32..2048")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.evaluation_limit is not None and (
            type(self.evaluation_limit) is not int or self.evaluation_limit < 1
        ):
            raise ValueError("evaluation_limit must be positive or None")


def select_language_lora_targets(
    linear_module_names: Sequence[str], *, suffixes: Sequence[str] = DEFAULT_LORA_SUFFIXES,
) -> list[str]:
    """Match *actual* language Linear modules; never adapt the vision tower.

    Qwen3.5 uses gated delta/linear attention as well as regular attention, so
    q_proj/v_proj alone omit most layers.  Exact full module names avoid matching
    identically named vision projections.  Missing whole attention families fail.
    """
    if not suffixes or any(not isinstance(item, str) or not item for item in suffixes):
        raise ValueError("LoRA suffixes must be nonempty text")
    result = sorted({
        name for name in linear_module_names
        if isinstance(name, str)
        and ".language_model." in "." + name
        and not any(piece in name.split(".") for piece in ("visual", "vision", "vision_tower"))
        and name.rsplit(".", 1)[-1] in suffixes
    })
    if not result:
        raise ValueError("no requested LoRA targets match instantiated language Linear modules")
    return result


def _integer_seed_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        type(seed) is not int or seed < 0 for seed in value
    ) or len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique nonnegative integer parent seeds")
    return sorted(value)


def validate_init_adapter(
    config: VerifierTrainingConfig, *, base_model: Path, data_report: dict,
    requested_targets: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Check local warm-start weights and ancestral training-seed isolation.

    This is not optimizer/scheduler/RNG resumption. Both saved PEFT configuration
    and our training audit must match. Validation runs before allocating the base
    model, and exact instantiated target names are checked again after loading.
    """
    if config.init_adapter is None:
        return None
    root = Path(config.init_adapter).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("init_adapter must be an existing local adapter directory")
    required = ("adapter_config.json", "verifier_training_manifest.json",
                "adapter_model.safetensors")
    for name in required:
        path = (root / name).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"init adapter requires a local, contained {name}")
    try:
        adapter = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (root / "verifier_training_manifest.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        raise ValueError("init adapter contains invalid configuration/manifest JSON") from None
    if not isinstance(adapter, dict) or not isinstance(manifest, dict):
        raise ValueError("init adapter configuration and training manifest must be objects")
    if manifest.get("schema_version") != 1 or manifest.get("model_family") != config.model_family:
        raise ValueError("init adapter training manifest has incompatible model_family/schema")
    base_model = base_model.resolve(strict=True)
    for source, field in ((adapter, "base_model_name_or_path"), (manifest, "base_model")):
        value = source.get(field)
        if not isinstance(value, str) or not value or Path(value).resolve() != base_model:
            raise ValueError(f"init adapter {field} does not match the requested base_model")
    resolved = manifest.get("base_model_resolved")
    if resolved is not None and (
        not isinstance(resolved, str) or Path(resolved).resolve() != base_model
    ):
        raise ValueError("init adapter resolved base_model does not match the requested checkpoint")
    if adapter.get("peft_type") != "LORA" or adapter.get("task_type") != "CAUSAL_LM":
        raise ValueError("init adapter must contain a causal language LoRA")
    training = manifest.get("training_config")
    if not isinstance(training, dict):
        raise ValueError("init adapter lacks audited training hyperparameters")
    for adapter_key, training_key, expected in (
        ("r", "rank", config.rank), ("lora_alpha", "alpha", config.alpha),
        ("lora_dropout", "dropout", config.dropout),
    ):
        for source, key in ((adapter, adapter_key), (training, training_key)):
            actual = source.get(key)
            if (
                type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError(f"init adapter {key} does not match requested LoRA settings")
    if (
        adapter.get("bias", "none") != "none" or adapter.get("modules_to_save")
        or adapter.get("use_dora", False) or adapter.get("use_rslora", False)
        or adapter.get("rank_pattern") or adapter.get("alpha_pattern")
        or adapter.get("lora_bias", False) or adapter.get("fan_in_fan_out", False)
        or adapter.get("target_parameters") or adapter.get("layer_replication")
        or adapter.get("trainable_token_indices") or adapter.get("layers_to_transform")
        or adapter.get("exclude_modules") or adapter.get("use_qalora", False)
    ):
        raise ValueError("init adapter includes unsupported extra trainable/scaling settings")
    targets = adapter.get("target_modules")
    audited_targets = manifest.get("lora_target_modules")
    for names in (targets, audited_targets):
        if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not name for name in names
        ) or len(names) != len(set(names)):
            raise ValueError("init adapter targets must be explicit unique full module names")
    targets = sorted(targets)
    if targets != sorted(audited_targets) or targets != select_language_lora_targets(targets):
        raise ValueError("init adapter targets do not match its audited language-only modules")
    if requested_targets is not None and targets != sorted(requested_targets):
        raise ValueError("init adapter target_modules differ from instantiated requested targets")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("group_split") != "parent_seed":
        raise ValueError("init adapter lacks leakage-safe parent-seed provenance")
    source_seeds = _integer_seed_list(dataset.get("train_parent_seeds"), name="source training seeds")
    lineage = manifest.get("training_parent_seeds_all")
    if lineage is None:
        if manifest.get("init_adapter_path") or training.get("init_adapter"):
            raise ValueError("warm-start source lacks its complete ancestral training-seed audit")
        lineage = source_seeds
    else:
        lineage = _integer_seed_list(lineage, name="ancestral training seeds")
        if not set(source_seeds) <= set(lineage):
            raise ValueError("ancestral seed audit does not include source training seeds")
    holdout_seeds = _integer_seed_list(
        data_report.get("holdout_parent_seeds"), name="new holdout seeds",
    )
    leaked = sorted(set(lineage) & set(holdout_seeds))
    if leaked:
        raise ValueError(f"init adapter training parent seeds leak into new holdout: {leaked}")
    if type(manifest.get("completed_steps")) is not int or manifest["completed_steps"] < 1:
        raise ValueError("init adapter has no completed training-step audit")
    return {
        "path": str(root), "source_completed_steps": manifest["completed_steps"],
        "source_training_parent_seeds": source_seeds,
        "ancestral_training_parent_seeds": lineage,
        "new_holdout_parent_seeds": holdout_seeds,
        "holdout_seed_leakage_checked": True, "source_target_modules": targets,
        "reset_optimizer": True, "exact_training_resume": False,
    }


def load_trainable_init_adapter(model: Any, adapter_path: str, peft_model_class: Any) -> Any:
    """Fail closed on PEFT's incomplete/mismatched adapter-weight warnings only.

    PEFT can warn and return a partly freshly initialized adapter when keys are
    missing. Normal dependency/performance warnings remain warnings. Explicitly
    disabling ignored size mismatches also lets tensor-shape errors propagate.
    The class argument allows lightweight tests without importing ML libraries.
    """
    integrity_patterns = (
        r".*missing adapter keys.*",
        r".*(?:weights|checkpoint|tensor).*ignore_mismatched_sizes.*",
        r".*(?:adapter|lora).*(?:mismatch|invalid shape|incorrect shape).*",
        r".*(?:mismatch|invalid shape|incorrect shape).*(?:adapter|lora).*",
    )
    try:
        with warnings.catch_warnings():
            for pattern in integrity_patterns:
                warnings.filterwarnings("error", message=pattern, category=Warning)
            return peft_model_class.from_pretrained(
                model, adapter_path, is_trainable=True, local_files_only=True,
                ignore_mismatched_sizes=False,
            )
    except Warning as exc:
        if not any(re.search(pattern, str(exc), re.IGNORECASE) for pattern in integrity_patterns):
            raise
        raise RuntimeError(
            "init adapter checkpoint is incomplete or mismatched: " + str(exc)[:1000]
        ) from exc


def build_answer_labels(
    prompt_ids: Sequence[int], answer_ids: Sequence[int], *, stop_id: int,
    attention_mask: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Only answer and assistant terminator are supervised; all prompt/pad masked."""
    if not prompt_ids or not answer_ids:
        raise ValueError("prompt and answer tokens must be nonempty")
    if any(type(token) is not int or token < 0 for token in (*prompt_ids, *answer_ids, stop_id)):
        raise ValueError("token ids must be nonnegative integers")
    ids = list(prompt_ids) + list(answer_ids) + [stop_id]
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids) + [stop_id]
    if attention_mask is not None:
        if len(attention_mask) != len(ids) or any(value not in (0, 1) for value in attention_mask):
            raise ValueError("attention mask must match the complete token sequence")
        labels = [label if attended else IGNORE_INDEX for label, attended in
                  zip(labels, attention_mask, strict=True)]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("all answer tokens were masked")
    return ids, labels


def answer_prediction_positions(labels: Sequence[int]) -> tuple[list[int], list[int]]:
    """Align causal prediction positions to supervised tokens without full-vocab logits."""
    positions, targets = [], []
    for index, label in enumerate(labels):
        if label != IGNORE_INDEX:
            if index == 0:
                raise ValueError("first token cannot be an autoregressive answer target")
            positions.append(index - 1)
            targets.append(label)
    if not positions:
        raise ValueError("no supervised answer tokens")
    return positions, targets


def _read_training_rows(path: Path, root: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row["split"] != split:
                raise ValueError("row has invalid split")
            if type(row["parent_seed"]) is not int or row["parent_seed"] < 0:
                raise ValueError("parent_seed must be a nonnegative integer")
            stage = row["stage"]
            if stage not in ALLOWED_ANSWERS or row["answer"] not in ALLOWED_ANSWERS[stage]:
                raise ValueError("answer does not match the two-stage verifier contract")
            frame_steps = row["frame_steps"]
            if (
                not isinstance(frame_steps, list) or not 2 <= len(frame_steps) <= 64
                or any(type(step) is not int or step < 0 for step in frame_steps)
                or any(a >= b for a, b in pairwise(frame_steps))
            ):
                raise ValueError("frame_steps must be 2..64 increasing nonnegative integers")
            if not isinstance(row["images"], list) or len(row["images"]) != 2 * len(frame_steps):
                raise ValueError("images must be chronological front/left_wrist pairs")
            for item in row["images"]:
                if not isinstance(item, str):
                    raise TypeError("image paths must be text")
                image = Path(item).resolve(strict=True)
                if not image.is_relative_to(root) or not image.is_file():
                    raise ValueError("image path escapes the prepared dataset root")
            if not isinstance(row["subgoal"], str) or not row["subgoal"].strip():
                raise ValueError("subgoal must be nonempty text")
            skill = CookieSkill(row["skill"])
            instruction_pattern = (
                r"Pick up five cookies from source column [1-4], batch [12]\."
                if skill is CookieSkill.PICK_FIVE else
                r"Place the five held cookies into target box column [12]\."
            )
            if re.fullmatch(instruction_pattern, row["subgoal"]) is None:
                raise ValueError("subgoal must be a canonical A3 skill instruction")
            expected_prompt = (
                verifier_completion_prompt(skill, row["subgoal"], len(frame_steps))
                if stage == "completion" else
                verifier_diagnosis_prompt(row["subgoal"], len(frame_steps))
            )
            if row["prompt"] != expected_prompt:
                raise ValueError("saved prompt does not match the runtime verifier prompt")
            if not isinstance(row["id"], str) or not row["id"]:
                raise ValueError("example id must be nonempty text")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ValueError(f"invalid {split} row {number}: {exc}") from None
        rows.append(row)
    if not rows:
        raise ValueError(f"{split} dataset has no examples")
    return rows


def load_training_dataset(data_root: Path) -> tuple[list[dict], list[dict], dict]:
    root = data_root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("prepared dataset manifest must use schema_version 1")
    train = _read_training_rows(root / "train.jsonl", root, "train")
    holdout = _read_training_rows(root / "holdout.jsonl", root, "holdout")
    train_seeds = {row["parent_seed"] for row in train}
    holdout_seeds = {row["parent_seed"] for row in holdout}
    if train_seeds & holdout_seeds:
        raise ValueError("parent seed leakage between train and holdout")
    ids = [row["id"] for row in train + holdout]
    if len(ids) != len(set(ids)):
        raise ValueError("example ids must be unique across train and holdout")
    image_sets = [{path for row in rows for path in row["images"]} for rows in (train, holdout)]
    if image_sets[0] & image_sets[1]:
        raise ValueError("camera image leakage between train and holdout")
    coverage = {
        split: dict(Counter(row["answer"] for row in rows))
        for split, rows in (("train", train), ("holdout", holdout))
    }
    for split, counts in coverage.items():
        if set(counts) != {"Yes", "No", "Stuck", "StillTrying"}:
            raise ValueError(f"{split} needs all four completion/diagnosis answer classes")
    contract = manifest.get("temporal_contract")
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("temporal contract must be an object")
        count, stride = contract.get("window_frames"), contract.get("frame_stride")
        if type(count) is not int or count < 2 or type(stride) is not int or stride < 1:
            raise ValueError("invalid temporal frame count or stride")
        if contract.get("horizon_steps") != (count - 1) * stride:
            raise ValueError("temporal horizon does not match frame count and stride")
        for row in train + holdout:
            frame_steps = row["frame_steps"]
            if len(frame_steps) != count or any(
                b - a != stride for a, b in pairwise(frame_steps)
            ):
                raise ValueError("row does not match the prepared temporal contract")
    return train, holdout, {
        "train_examples": len(train), "holdout_examples": len(holdout),
        "train_parent_seeds": sorted(train_seeds), "holdout_parent_seeds": sorted(holdout_seeds),
        "answer_class_counts": coverage, "group_split": "parent_seed",
        "temporal_contract": contract,
        "simulator_truth_used_as_model_input": False,
    }


def _load_row_images(row: dict, image_max_edge: int) -> list[Image.Image]:
    result = []
    for path in row["images"]:
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        image.thumbnail((image_max_edge, image_max_edge), Image.Resampling.LANCZOS)
        result.append(image)
    return result


def encode_verifier_row(
    processor: Any, row: dict, config: VerifierTrainingConfig, *, include_answer: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The exact inference prefix, plus independently tokenized answer/terminator.

    Appending token IDs avoids fragile prefix matching across BPE boundaries and
    never labels an expanded visual placeholder. Batch size is intentionally one.
    """
    import torch

    images = _load_row_images(row, config.image_max_edge)
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": row["prompt"]})
    options = {"tokenize": False, "add_generation_prompt": True}
    if config.model_family == "qwen3_5":
        options["enable_thinking"] = False
    text = processor.apply_chat_template([{"role": "user", "content": content}], **options)
    encoded = dict(processor(text=[text], images=images, padding=True, return_tensors="pt"))
    prompt_ids = encoded["input_ids"][0].tolist()
    if not encoded["attention_mask"].all().item():
        raise ValueError("batch-one prompt must not contain padding")
    stats = {"prompt_tokens": len(prompt_ids), "images": len(images),
             "answer_tokens": 0, "supervised_tokens": 0}
    if include_answer:
        answer_ids = processor.tokenizer.encode(row["answer"], add_special_tokens=False)
        stop_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if type(stop_id) is not int or stop_id < 0 or stop_id == processor.tokenizer.unk_token_id:
            raise ValueError("processor lacks the Qwen assistant terminator token")
        full_ids, labels = build_answer_labels(prompt_ids, answer_ids, stop_id=stop_id)
        extra = len(full_ids) - len(prompt_ids)
        for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
            if key in encoded:
                old = encoded[key]
                fill = 1 if key == "attention_mask" else 0
                encoded[key] = torch.cat((old, old.new_full((1, extra), fill)), dim=1)
        encoded["input_ids"] = encoded["input_ids"].new_tensor([full_ids])
        encoded["labels"] = encoded["input_ids"].new_tensor([labels])
        stats.update(answer_tokens=len(answer_ids), supervised_tokens=len(answer_ids) + 1,
                     answer_token_ids=answer_ids, terminator_token_id=stop_id)
    if encoded["input_ids"].shape[1] > config.max_sequence_length:
        raise ValueError("visual sequence exceeds configured limit; no silent truncation allowed")
    return encoded, stats


def _answer_loss(model: Any, encoded: dict[str, Any], device: Any) -> Any:
    import torch

    inputs = {key: value.to(device) for key, value in encoded.items() if key != "labels"}
    positions, targets = answer_prediction_positions(encoded["labels"][0].tolist())
    kept = torch.tensor(positions, dtype=torch.long, device=device)
    # Compute vocab logits only at the answer-prediction positions. This is
    # equivalent to causal CE with labels=-100 everywhere else, but avoids a
    # sequence_length * 248k-vocabulary allocation for the multi-image prompt.
    outputs = model(**inputs, use_cache=False, logits_to_keep=kept)
    logits = outputs.logits
    if logits.shape[:2] != (1, len(targets)):
        raise RuntimeError("model did not honor answer-only logits_to_keep")
    return torch.nn.functional.cross_entropy(
        logits[0].float(), torch.tensor(targets, dtype=torch.long, device=device),
    )


def _write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _balanced_epoch_order(rows: list[dict], rng: random.Random) -> list[int]:
    """Equalize completion/diagnosis answer classes without touching holdout."""
    groups = {answer: [] for answer in ("Yes", "No", "Stuck", "StillTrying")}
    for index, row in enumerate(rows):
        groups[row["answer"]].append(index)
    if any(not values for values in groups.values()):
        raise ValueError("all four answer classes are required for balanced training")
    size = max(len(values) for values in groups.values())
    order = []
    for values in groups.values():
        rng.shuffle(values)
        order.extend(values[index % len(values)] for index in range(size))
    rng.shuffle(order)
    return order


def evaluate_trained_verifier(
    model: Any, processor: Any, rows: list[dict], config: VerifierTrainingConfig, device: Any,
) -> dict:
    import torch

    selected = rows if config.evaluation_limit is None else rows[:config.evaluation_limit]
    records = []
    losses = []
    model.eval()
    with torch.inference_mode():
        for row in selected:
            encoded, _ = encode_verifier_row(processor, row, config, include_answer=True)
            loss = float(_answer_loss(model, encoded, device).item())
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite holdout loss")
            losses.append(loss)
            prefix, _ = encode_verifier_row(processor, row, config, include_answer=False)
            prompt_length = prefix["input_ids"].shape[1]
            generated = model.generate(
                **{key: value.to(device) for key, value in prefix.items()},
                do_sample=False, max_new_tokens=config.generation_tokens, use_cache=True,
            )
            predicted = processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            records.append({
                "id": row["id"], "parent_seed": row["parent_seed"], "stage": row["stage"],
                "skill": row["skill"], "expected": row["answer"], "predicted": predicted,
                "correct": predicted == row["answer"], "answer_loss": loss,
                "valid_contract": predicted in ALLOWED_ANSWERS[row["stage"]],
            })
    stage_metrics = {}
    for stage in ALLOWED_ANSWERS:
        subset = [item for item in records if item["stage"] == stage]
        stage_metrics[stage] = {
            "examples": len(subset),
            "accuracy": sum(item["correct"] for item in subset) / len(subset) if subset else None,
            "invalid_answers": sum(not item["valid_contract"] for item in subset),
        }
    completion = [item for item in records if item["stage"] == "completion"]
    confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for item in completion:
        if item["valid_contract"]:
            confusion[{("Yes", "Yes"): "TP", ("No", "Yes"): "FP",
                       ("Yes", "No"): "FN", ("No", "No"): "TN"}[
                item["expected"], item["predicted"]]] += 1
    # Invalid outputs on a positive example are misses, not removed from recall.
    positives = sum(item["expected"] == "Yes" for item in completion)
    per_answer = {}
    for answer in ("Yes", "No", "Stuck", "StillTrying"):
        subset = [item for item in records if item["expected"] == answer]
        per_answer[answer] = {
            "examples": len(subset), "correct": sum(item["correct"] for item in subset),
            "recall": sum(item["correct"] for item in subset) / len(subset) if subset else None,
        }
    per_skill = {}
    for skill in ("PICK_FIVE", "PLACE_FIVE"):
        subset = [item for item in records if item["skill"] == skill]
        per_skill[skill] = {
            "examples": len(subset),
            "accuracy": sum(item["correct"] for item in subset) / len(subset) if subset else None,
        }
    return {
        "examples": len(records), "available_holdout_examples": len(rows),
        "mean_answer_loss": sum(losses) / len(losses),
        "exact_answer_accuracy": sum(item["correct"] for item in records) / len(records),
        "stage_metrics": stage_metrics, "completion_confusion": confusion,
        "per_answer": per_answer,
        "per_skill": per_skill,
        "completion_recall": confusion["TP"] / positives if positives else None,
        "protocol": "independent parent-seed holdout; teacher-forced loss and greedy answers",
        "simulator_truth_used_as_model_input": False, "samples": records,
    }


def train_verifier_lora(
    data_root: Path, output_root: Path, config: VerifierTrainingConfig,
) -> dict:
    """Train a local base checkpoint on one GPU, then save a standalone PEFT adapter."""
    train, holdout, data_report = load_training_dataset(data_root)
    base = Path(config.base_model).resolve(strict=True)
    if not base.is_dir():
        raise ValueError("base_model must be an existing local checkpoint directory")
    init_audit = validate_init_adapter(config, base_model=base, data_report=data_report)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("training output must be a new directory; resume is not implicit")
    try:
        import numpy as np
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoProcessor

        if config.model_family == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_class = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        raise RuntimeError("verifier training requires torch, transformers 5.3, and PEFT") from None
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this training command requires a BF16-capable CUDA GPU")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda:0")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1, "model_family": config.model_family,
        "base_model": config.base_model, "base_model_resolved": str(base),
        "training_config": asdict(config), "dataset": data_report,
        "data_root": str(data_root.resolve()), "batch_size": 1,
        "effective_batch_size": config.gradient_accumulation,
        "dtype": "bfloat16", "quantized": False,
        "vision_frozen": True, "supervision": "answer tokens plus assistant terminator only",
        "enable_thinking": False if config.model_family == "qwen3_5" else None,
        "sampler": "answer-class-balanced training epochs; holdout never oversampled",
        "sampling_bins": ["completion/Yes", "completion/No",
                          "diagnosis/Stuck", "diagnosis/StillTrying"],
        "simulator_truth_used_as_model_input": False,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "temporal_contract": data_report["temporal_contract"],
        "init_adapter_path": init_audit["path"] if init_audit else None,
        "initialization": "adapter weight warm-start" if init_audit else "fresh LoRA",
        "reset_optimizer": True, "exact_training_resume": False,
        "init_adapter_audit": init_audit,
        "training_parent_seeds_all": sorted(set(data_report["train_parent_seeds"]) | (
            set(init_audit["ancestral_training_parent_seeds"]) if init_audit else set()
        )),
    }
    _write_json_new(output_root / "training_config.json", manifest)
    completed_steps = 0
    try:
        model = model_class.from_pretrained(
            str(base), torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(str(base), local_files_only=True)
        model.requires_grad_(False)
        actual_linear_names = [name for name, module in model.named_modules()
                               if isinstance(module, torch.nn.Linear)]
        targets = select_language_lora_targets(actual_linear_names)
        if config.model_family == "qwen3_5" and not any(
            name.endswith(".in_proj_qkv") for name in targets
        ):
            raise RuntimeError("Qwen3.5 linear-attention LoRA projections were not found")
        if init_audit:
            validate_init_adapter(
                config, base_model=base, data_report=data_report, requested_targets=targets,
            )
            model = load_trainable_init_adapter(model, init_audit["path"], PeftModel)
        else:
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=config.rank, lora_alpha=config.alpha,
                lora_dropout=config.dropout, bias="none", target_modules=targets,
            ))
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model.config.use_cache = False
        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise RuntimeError("only LoRA parameters are allowed to be trainable")
        if any(".visual." in name or ".vision." in name for name, _ in trainable):
            raise RuntimeError("vision parameters must remain frozen")
        manifest.update(
            lora_target_modules=targets, lora_target_count=len(targets),
            trainable_parameters=sum(parameter.numel() for _, parameter in trainable),
            total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        )
        raw_counts = data_report["answer_class_counts"]["train"]
        manifest["balanced_epoch_answer_counts"] = {
            answer: max(raw_counts.values()) for answer in raw_counts
        }
        _, token_stats = encode_verifier_row(processor, train[0], config, include_answer=True)
        manifest["tokenization_probe"] = token_stats
        _write_json_new(output_root / "training_audit.json", manifest)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable], lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer.zero_grad(set_to_none=True)
        rng = random.Random(config.seed)
        order: list[int] = []
        offset = 0
        seen_examples = 0
        seen_answer_tokens = 0
        loss_history = []
        model.train()
        for name, module in model.named_modules():
            if name.endswith(".visual"):
                module.eval()
        with (output_root / "progress.jsonl").open("x", encoding="utf-8") as progress:
            for step in range(1, config.steps + 1):
                step_losses = []
                for _ in range(config.gradient_accumulation):
                    if offset >= len(order):
                        order = _balanced_epoch_order(train, rng)
                        offset = 0
                    row = train[order[offset]]
                    offset += 1
                    encoded, stats = encode_verifier_row(
                        processor, row, config, include_answer=True,
                    )
                    loss = _answer_loss(model, encoded, device)
                    value = float(loss.detach().item())
                    if not math.isfinite(value):
                        raise RuntimeError("nonfinite training loss")
                    (loss / config.gradient_accumulation).backward()
                    step_losses.append(value)
                    seen_examples += 1
                    seen_answer_tokens += stats["answer_tokens"]
                    del encoded, loss
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in trainable], config.max_grad_norm,
                ).item())
                if not math.isfinite(gradient_norm):
                    raise RuntimeError("nonfinite LoRA gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps = step
                record = {
                    "step": step, "answer_loss": sum(step_losses) / len(step_losses),
                    "gradient_norm": gradient_norm, "examples_seen": seen_examples,
                    "answer_tokens_seen": seen_answer_tokens,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                loss_history.append(record["answer_loss"])
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                if step == 1 or step % config.log_every == 0 or step == config.steps:
                    print(json.dumps(record), flush=True)
                if step % config.save_every == 0 and step != config.steps:
                    checkpoint = output_root / "checkpoints" / f"step_{step:06d}"
                    checkpoint.mkdir(parents=True, exist_ok=False)
                    model.save_pretrained(checkpoint, safe_serialization=True)
                    processor.save_pretrained(checkpoint)
                    _write_json_new(checkpoint / "verifier_training_manifest.json", {
                        **manifest, "completed_steps": step, "checkpoint_only": True,
                    })
        adapter = output_root / "adapter"
        adapter.mkdir(exist_ok=False)
        model.save_pretrained(adapter, safe_serialization=True)
        processor.save_pretrained(adapter)
        final_manifest = {
            **manifest, "completed_steps": completed_steps, "examples_seen": seen_examples,
            "answer_tokens_seen": seen_answer_tokens, "checkpoint_only": False,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        _write_json_new(adapter / "verifier_training_manifest.json", final_manifest)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_disable()
        evaluation = evaluate_trained_verifier(model, processor, holdout, config, device)
        _write_json_new(output_root / "holdout_evaluation.json", evaluation)
        summary = {
            "status": "completed", "adapter_path": str(adapter.resolve()),
            "completed_steps": completed_steps,
            "first_answer_loss": loss_history[0], "last_answer_loss": loss_history[-1],
            "holdout": {key: value for key, value in evaluation.items() if key != "samples"},
            "dataset": data_report,
            "init_adapter_path": manifest["init_adapter_path"],
            "reset_optimizer": True, "exact_training_resume": False,
            "training_parent_seeds_all": manifest["training_parent_seeds_all"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "note": "Small-domain LoRA experiment; closed-loop skill validation remains separate.",
        }
        _write_json_new(output_root / "training_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except Exception as exc:
        _write_json_new(output_root / "training_failure.json", {
            "status": "failed", "completed_steps": completed_steps,
            "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise
