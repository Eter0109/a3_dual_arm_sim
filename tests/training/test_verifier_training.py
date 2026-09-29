from __future__ import annotations

import json
import random
import warnings
from dataclasses import replace

import pytest
from PIL import Image

from a3_dual_arm_sim.agents.prompts import verifier_completion_prompt, verifier_diagnosis_prompt
from a3_dual_arm_sim.core.skills import CookieSkill
from a3_dual_arm_sim.training.verifier import (
    VerifierTrainingConfig,
    _balanced_epoch_order,
    answer_prediction_positions,
    build_answer_labels,
    load_trainable_init_adapter,
    load_training_dataset,
    select_language_lora_targets,
    validate_init_adapter,
)


def test_only_answer_and_terminator_have_labels():
    ids, labels = build_answer_labels([7, 90, 90, 8], [12, 13], stop_id=3)
    assert ids == [7, 90, 90, 8, 12, 13, 3]
    assert labels == [-100, -100, -100, -100, 12, 13, 3]
    assert answer_prediction_positions(labels) == ([3, 4, 5], [12, 13, 3])


def test_padding_is_never_supervised():
    _, labels = build_answer_labels([7], [12, 13], stop_id=3, attention_mask=[1, 1, 0, 0])
    assert labels == [-100, 12, -100, -100]
    assert answer_prediction_positions(labels) == ([0], [12])


@pytest.mark.parametrize("prompt,answer,stop", [([], [1], 2), ([1], [], 2), ([True], [1], 2),
                                               ([1], [-1], 2), ([1], [2], -1)])
def test_invalid_token_contract_rejected(prompt, answer, stop):
    with pytest.raises(ValueError):
        build_answer_labels(prompt, answer, stop_id=stop)


def test_invalid_label_alignment_rejected():
    with pytest.raises(ValueError, match="first token"):
        answer_prediction_positions([3, -100])
    with pytest.raises(ValueError, match="no supervised"):
        answer_prediction_positions([-100, -100])


def test_targets_match_actual_language_modules_including_delta_attention():
    names = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.v_proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.visual.blocks.0.attn.out_proj", "lm_head",
    ]
    targets = select_language_lora_targets(names)
    assert len(targets) == 5
    assert all(".language_model." in name for name in targets)
    assert any(name.endswith("in_proj_qkv") for name in targets)


def test_no_language_targets_fails_instead_of_adapting_vision():
    with pytest.raises(ValueError, match="no requested"):
        select_language_lora_targets(["model.visual.blocks.0.attn.q_proj"])


@pytest.mark.parametrize("changes", [{"steps": 0}, {"gradient_accumulation": -1}, {"rank": True},
                                      {"learning_rate": float("nan")}, {"weight_decay": -1},
                                      {"dropout": 1}, {"evaluation_limit": 0},
                                      {"model_family": "unknown"}, {"image_max_edge": 2},
                                      {"init_adapter": ""}, {"init_adapter": 1}])
def test_training_config_rejects_invalid_settings(changes):
    with pytest.raises(ValueError):
        replace(VerifierTrainingConfig(), **changes)


def test_balanced_sampler_is_reproducible_and_uses_all_classes():
    rows = [{"answer": answer} for answer in ("Yes", "Yes", "No", "Stuck", "StillTrying")]
    first = _balanced_epoch_order(rows, random.Random(7))
    assert first == _balanced_epoch_order(rows, random.Random(7))
    answers = [rows[index]["answer"] for index in first]
    assert len(first) == 8
    assert all(answers.count(answer) == 2 for answer in ("Yes", "No", "Stuck", "StillTrying"))


def _dataset(tmp_path):
    rows = {"train": [], "holdout": []}
    for split, seed in (("train", 0), ("holdout", 100)):
        images = []
        for index in range(4):
            path = tmp_path / f"image_{seed}_{index}.png"
            Image.new("RGB", (8, 8)).save(path)
            images.append(str(path))
        for stage, answer in (("completion", "Yes"), ("completion", "No"),
                              ("diagnosis", "Stuck"), ("diagnosis", "StillTrying")):
            subgoal = "Pick up five cookies from source column 1, batch 1."
            prompt = (verifier_completion_prompt(CookieSkill.PICK_FIVE, subgoal, 2)
                      if stage == "completion" else verifier_diagnosis_prompt(subgoal, 2))
            rows[split].append({
                "id": f"{seed}_{answer}", "split": split, "parent_seed": seed,
                "stage": stage, "answer": answer, "skill": "PICK_FIVE", "subgoal": subgoal,
                "prompt": prompt, "images": images, "frame_steps": [0, 10],
            })
    _write_rows(tmp_path, rows)
    return rows


def _write_rows(tmp_path, rows):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    for split, values in rows.items():
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8",
        )


def test_prepared_data_has_grouped_holdout_and_all_answers(tmp_path):
    _dataset(tmp_path)
    train, holdout, report = load_training_dataset(tmp_path)
    assert len(train) == len(holdout) == 4
    assert report["train_parent_seeds"] == [0]
    assert report["holdout_parent_seeds"] == [100]
    assert report["simulator_truth_used_as_model_input"] is False


def test_parent_seed_leakage_rejected(tmp_path):
    rows = _dataset(tmp_path)
    for row in rows["holdout"]:
        row["parent_seed"] = 0
    _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="parent seed leakage"):
        load_training_dataset(tmp_path)


def test_modified_prompt_rejected_to_preserve_inference_template(tmp_path):
    rows = _dataset(tmp_path)
    rows["train"][0]["prompt"] += " Truth: completed"
    _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="runtime verifier prompt"):
        load_training_dataset(tmp_path)


def test_image_leakage_rejected(tmp_path):
    rows = _dataset(tmp_path)
    for row in rows["holdout"]:
        row["images"] = rows["train"][0]["images"]
    _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="camera image leakage"):
        load_training_dataset(tmp_path)


def test_image_escape_rejected(tmp_path):
    rows = _dataset(tmp_path)
    rows["train"][0]["images"][0] = str(tmp_path.parent)
    _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="image path escapes"):
        load_training_dataset(tmp_path)


def test_subgoal_cannot_embed_privileged_labels(tmp_path):
    rows = _dataset(tmp_path)
    row = rows["train"][0]
    row["subgoal"] += " Truth: completed"
    row["prompt"] = verifier_completion_prompt(CookieSkill.PICK_FIVE, row["subgoal"], 2)
    _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="canonical A3"):
        load_training_dataset(tmp_path)


def test_missing_answer_class_fails_explicitly(tmp_path):
    rows = _dataset(tmp_path)
    rows["holdout"] = rows["holdout"][:-1]
    _write_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="all four"):
        load_training_dataset(tmp_path)


def test_temporal_contract_is_reported_and_checked(tmp_path):
    _dataset(tmp_path)
    manifest = {"schema_version": 1,
                "temporal_contract": {"window_frames": 2, "frame_stride": 10, "horizon_steps": 10}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _, _, report = load_training_dataset(tmp_path)
    assert report["temporal_contract"] == manifest["temporal_contract"]
    manifest["temporal_contract"]["frame_stride"] = 5
    manifest["temporal_contract"]["horizon_steps"] = 5
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row does not match"):
        load_training_dataset(tmp_path)


def _init_adapter(tmp_path):
    base = tmp_path / "base_model"
    base.mkdir()
    adapter_path = tmp_path / "source_adapter"
    adapter_path.mkdir()
    targets = ["model.language_model.layers.0.linear_attn.in_proj_qkv",
               "model.language_model.layers.0.linear_attn.out_proj",
               "model.language_model.layers.1.self_attn.q_proj"]
    adapter = {
        "base_model_name_or_path": str(base), "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
        "target_modules": targets, "modules_to_save": None,
    }
    manifest = {
        "schema_version": 1, "model_family": "qwen3_5", "base_model": str(base),
        "base_model_resolved": str(base), "completed_steps": 160,
        "training_config": {"rank": 8, "alpha": 16, "dropout": 0.05},
        "lora_target_modules": targets,
        "dataset": {"group_split": "parent_seed", "train_parent_seeds": [0, 1, 2]},
    }
    _write_adapter(adapter_path, adapter, manifest)
    (adapter_path / "adapter_model.safetensors").write_bytes(b"fixture metadata validation only")
    config = VerifierTrainingConfig(base_model=str(base), init_adapter=str(adapter_path))
    report = {"train_parent_seeds": [0, 1, 2, 3], "holdout_parent_seeds": [100, 101]}
    return base, adapter_path, adapter, manifest, config, report


def _write_adapter(path, adapter, manifest):
    (path / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
    (path / "verifier_training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_default_initialization_remains_fresh_without_source_access(tmp_path):
    assert validate_init_adapter(VerifierTrainingConfig(), base_model=tmp_path / "absent",
                                 data_report={}) is None


def test_init_adapter_warmstart_validates_configuration_and_seed_isolation(tmp_path):
    base, _, adapter, _, config, report = _init_adapter(tmp_path)
    audit = validate_init_adapter(config, base_model=base, data_report=report,
                                  requested_targets=adapter["target_modules"])
    assert audit["source_completed_steps"] == 160
    assert audit["ancestral_training_parent_seeds"] == [0, 1, 2]
    assert audit["holdout_seed_leakage_checked"] is True
    assert audit["reset_optimizer"] is True
    assert audit["exact_training_resume"] is False


@pytest.mark.parametrize("field,value,match", [
    ("r", 4, "LoRA settings"), ("lora_alpha", 32, "LoRA settings"),
    ("lora_dropout", 0.1, "LoRA settings"), ("peft_type", "IA3", "causal language LoRA"),
    ("base_model_name_or_path", "other_checkpoint", "base_model"),
    ("modules_to_save", ["lm_head"], "unsupported extra"),
])
def test_init_adapter_rejects_incompatible_peft_configuration(tmp_path, field, value, match):
    base, path, adapter, manifest, config, report = _init_adapter(tmp_path)
    adapter[field] = value
    _write_adapter(path, adapter, manifest)
    with pytest.raises(ValueError, match=match):
        validate_init_adapter(config, base_model=base, data_report=report)


@pytest.mark.parametrize("field,value,match", [
    ("model_family", "qwen2_5_vl", "model_family"), ("base_model", "other_model", "base_model"),
    ("completed_steps", 0, "completed training-step"), ("training_config", {}, "LoRA settings"),
    ("dataset", {"train_parent_seeds": [0]}, "parent-seed provenance"),
])
def test_init_adapter_rejects_incompatible_training_manifest(tmp_path, field, value, match):
    base, path, adapter, manifest, config, report = _init_adapter(tmp_path)
    manifest[field] = value
    _write_adapter(path, adapter, manifest)
    with pytest.raises(ValueError, match=match):
        validate_init_adapter(config, base_model=base, data_report=report)


def test_init_adapter_rejects_vision_targets_even_if_audit_agrees(tmp_path):
    base, path, adapter, manifest, config, report = _init_adapter(tmp_path)
    adapter["target_modules"] = ["model.visual.blocks.0.attn.q_proj"]
    manifest["lora_target_modules"] = adapter["target_modules"]
    _write_adapter(path, adapter, manifest)
    with pytest.raises(ValueError, match="no requested|audited language-only"):
        validate_init_adapter(config, base_model=base, data_report=report)


def test_init_adapter_requires_exact_instantiated_requested_targets(tmp_path):
    base, _, _, _, config, report = _init_adapter(tmp_path)
    with pytest.raises(ValueError, match="instantiated requested targets"):
        validate_init_adapter(config, base_model=base, data_report=report,
                              requested_targets=["model.language_model.layers.1.self_attn.v_proj"])


def test_init_adapter_rejects_new_holdout_previously_seen_in_training(tmp_path):
    base, _, _, _, config, report = _init_adapter(tmp_path)
    report["holdout_parent_seeds"] = [2, 100]
    with pytest.raises(ValueError, match="leak into new holdout"):
        validate_init_adapter(config, base_model=base, data_report=report)


def test_init_adapter_rejects_leakage_from_any_ancestor_not_only_last_training(tmp_path):
    base, path, adapter, manifest, config, report = _init_adapter(tmp_path)
    manifest["init_adapter_path"] = "older_adapter"
    manifest["training_parent_seeds_all"] = [0, 1, 2, 100]
    _write_adapter(path, adapter, manifest)
    with pytest.raises(ValueError, match="leak into new holdout"):
        validate_init_adapter(config, base_model=base, data_report=report)


def test_init_adapter_rejects_incomplete_ancestral_seed_audit(tmp_path):
    base, path, adapter, manifest, config, report = _init_adapter(tmp_path)
    manifest["init_adapter_path"] = "older_adapter"
    _write_adapter(path, adapter, manifest)
    with pytest.raises(ValueError, match="complete ancestral"):
        validate_init_adapter(config, base_model=base, data_report=report)
    manifest["training_parent_seeds_all"] = [0, 1]
    _write_adapter(path, adapter, manifest)
    with pytest.raises(ValueError, match="does not include source"):
        validate_init_adapter(config, base_model=base, data_report=report)


def test_init_adapter_rejects_pickle_only_weights(tmp_path):
    base, path, _, _, config, report = _init_adapter(tmp_path)
    (path / "adapter_model.safetensors").unlink()
    (path / "adapter_model.bin").write_bytes(b"unsafe pickle placeholder")
    with pytest.raises(FileNotFoundError):
        validate_init_adapter(config, base_model=base, data_report=report)


@pytest.mark.parametrize("message", [
    "Found missing adapter keys while loading the checkpoint: ['lora_A.default.weight'].",
    "Some weights are being ignored because you passed `ignore_mismatched_sizes=True`.",
    "Adapter 'default': 2 LoRA tensor(s) have invalid shape and are incomplete.",
    "Mismatched adapter weights in checkpoint.",
    "LoRA weight shape mismatch while loading.",
])
def test_adapter_integrity_warnings_are_hard_errors(message):
    class Loader:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            warnings.warn(message, UserWarning, stacklevel=2)
            return "partly initialized adapter"

    with pytest.raises(RuntimeError, match="incomplete or mismatched"):
        load_trainable_init_adapter(object(), "local_adapter", Loader)


@pytest.mark.parametrize("message,category", [
    ("The optional optimized kernel is not installed; using torch implementation.", UserWarning),
    ("Adapter loading on CUDA uses the existing model device.", UserWarning),
    ("The torch_dtype argument is deprecated.", FutureWarning),
    ("The ignore_mismatched_sizes keyword is deprecated.", FutureWarning),
])
def test_normal_dependency_and_performance_warnings_are_not_blocked(message, category):
    seen = {}
    result = object()

    class Loader:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            seen.update(kwargs)
            warnings.warn(message, category, stacklevel=2)
            return result

    with pytest.warns(category, match="optional|CUDA|deprecated"):
        assert load_trainable_init_adapter(object(), "local_adapter", Loader) is result
    assert seen == {"is_trainable": True, "local_files_only": True,
                    "ignore_mismatched_sizes": False}


def test_actual_adapter_size_mismatch_errors_propagate():
    class Loader:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            raise RuntimeError("size mismatch for lora_A: checkpoint [4, 8], model [8, 8]")

    with pytest.raises(RuntimeError, match="size mismatch for lora_A"):
        load_trainable_init_adapter(object(), "local_adapter", Loader)


def test_existing_strict_warning_policy_is_not_reclassified_as_checkpoint_corruption():
    class Loader:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            warnings.warn("ordinary dependency deprecation", FutureWarning, stacklevel=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(FutureWarning, match="ordinary dependency"):
            load_trainable_init_adapter(object(), "local_adapter", Loader)
