"""Calibrate a visual verifier on saved, independently labeled camera windows.

Completion and conditional Stuck/StillTrying diagnosis are scored separately.
Labels are used only after inference. truth_audit.jsonl is never opened.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from a3_dual_arm_sim.data.verifier import (
    OUTCOME_CLASSES,
    read_verifier_samples,
    request_from_verifier_sample,
    resolve_visual_frame_paths,
    verifier_sample_outcome,
)


def select_balanced_samples(
    samples: list[dict[str, Any]], limit: int | None = 8
) -> list[dict[str, Any]]:
    """Deterministic interleaving, with a positive first whenever one exists.

    Malformed labels remain in the selection and produce evaluation errors;
    they cannot silently disappear from the dataset accounting.
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("sample limit must be a positive integer or None")
    queues: dict[str, deque] = {"Yes": deque(), "No": deque(), "Invalid": deque()}
    for sample in samples:
        label = sample.get("label", {})
        value = label.get("completion") if isinstance(label, dict) else None
        queues[value if value in ("Yes", "No") else "Invalid"].append(sample)
    result = []
    while any(queues.values()) and (limit is None or len(result) < limit):
        for value in ("Yes", "No", "Invalid"):
            if queues[value] and (limit is None or len(result) < limit):
                result.append(queues[value].popleft())
    return result


def select_class_balanced_samples(
    samples: list[dict[str, Any]], per_class_limit: int, *, selection_seed: int = 7,
) -> list[dict[str, Any]]:
    """Bounded three-outcome stratification, not natural-prevalence sampling.

    Invalid labels are retained in a separate bounded queue, so malformed
    examples cannot silently vanish from evaluation error accounting.
    """
    if type(per_class_limit) is not int or per_class_limit < 1:
        raise ValueError("per_class_limit must be a positive integer")
    if type(selection_seed) is not int:
        raise ValueError("selection_seed must be an integer")
    classes = (*OUTCOME_CLASSES, "invalid")
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in classes}
    for sample in samples:
        try:
            outcome = verifier_sample_outcome(sample)
        except (KeyError, TypeError, ValueError):
            outcome = "invalid"
        groups[outcome].append(sample)
    queues = {}
    for index, name in enumerate(classes):
        random.Random(selection_seed + index).shuffle(groups[name])
        queues[name] = deque(groups[name][:per_class_limit])
    result = []
    while any(queues.values()):
        for name in classes:
            if queues[name]:
                result.append(queues[name].popleft())
    return result


def load_visual_frames(data_root: Path, sample: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    steps, paths = resolve_visual_frame_paths(data_root, sample)
    result = []
    for frame_index in range(len(steps)):
        pair = {}
        for camera_index, camera in enumerate(("front", "left_wrist")):
            path = paths[frame_index * 2 + camera_index]
            with Image.open(path) as image:
                pair[camera] = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        result.append(pair)
    return result


def evaluate_samples(
    data_root: Path, verifier: Any, *, limit: int | None = 8,
    parent_seeds: list[int] | None = None,
    per_class_limit: int | None = None, selection_seed: int = 7,
) -> dict[str, Any]:
    all_samples = read_verifier_samples(data_root)
    if parent_seeds is not None:
        if not parent_seeds or any(type(seed) is not int for seed in parent_seeds):
            raise ValueError("parent seeds must be nonempty integers")
        eligible = [sample for sample in all_samples if sample.get("parent_seed") in parent_seeds]
        if not eligible:
            raise ValueError("no samples match the requested parent seeds")
    else:
        eligible = all_samples
    selected = (
        select_class_balanced_samples(eligible, per_class_limit, selection_seed=selection_seed)
        if per_class_limit is not None else select_balanced_samples(eligible, limit)
    )
    confusion = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    diagnoses = {
        expected: {predicted: 0 for predicted in ("StillTrying", "Stuck")}
        for expected in ("StillTrying", "Stuck")
    }
    joint = {expected: {predicted: 0 for predicted in OUTCOME_CLASSES} for expected in OUTCOME_CLASSES}
    label_coverage = {name: 0 for name in OUTCOME_CLASSES}
    records = []
    errors = 0
    for sample in selected:
        record = {"sample_index": sample.get("sample_index"), "error": None}
        try:
            request = request_from_verifier_sample(sample)
            frames = load_visual_frames(data_root, sample)
            # The model receives only semantic skills and real camera images.
            # In particular, labels/audit metadata are never passed to check().
            decision = verifier.check(request, frames)
            if decision.status not in ("completed", "continuing", "stuck"):
                raise ValueError("verifier returned an unsupported decision")
            expected = sample["label"]["completion"]
            if expected not in ("Yes", "No"):
                raise ValueError("completion label must be exactly Yes or No")
            expected_outcome = verifier_sample_outcome(sample)
            predicted = "Yes" if decision.status == "completed" else "No"
            category = {
                ("Yes", "Yes"): "TP", ("No", "Yes"): "FP",
                ("Yes", "No"): "FN", ("No", "No"): "TN",
            }[expected, predicted]
            confusion[category] += 1
            joint[expected_outcome][decision.status] += 1
            record.update(
                expected=expected, predicted=predicted, status=decision.status,
                reason=decision.reason, category=category,
            )
            if expected == "No":
                expected_diagnosis = sample["label"]["diagnosis"]
                predicted_diagnosis = (
                    "Stuck" if decision.status == "stuck" else
                    "StillTrying" if decision.status == "continuing" else None
                )
                record.update(
                    expected_diagnosis=expected_diagnosis,
                    predicted_diagnosis=predicted_diagnosis,
                    diagnosis_scored=predicted_diagnosis is not None,
                )
                if predicted_diagnosis is not None:
                    diagnoses[expected_diagnosis][predicted_diagnosis] += 1
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as exc:
            errors += 1
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    # Count selected label coverage independently of model failures, after all
    # inference. A failed call must not make an outcome disappear from coverage.
    invalid_labels = 0
    for sample in selected:
        try:
            label_coverage[verifier_sample_outcome(sample)] += 1
        except (KeyError, TypeError, ValueError):
            invalid_labels += 1
    valid = sum(confusion.values())
    diagnosis_valid = sum(sum(row.values()) for row in diagnoses.values())
    incomplete = label_coverage["continuing"] + label_coverage["stuck"]
    recall_denominator = confusion["TP"] + confusion["FN"]
    precision_denominator = confusion["TP"] + confusion["FP"]
    specificity_denominator = confusion["TN"] + confusion["FP"]
    recall = confusion["TP"] / recall_denominator if recall_denominator and not errors else None
    specificity = (
        confusion["TN"] / specificity_denominator if specificity_denominator and not errors else None
    )
    return {
        "dataset_root": str(data_root),
        "available_samples": len(all_samples),
        "eligible_samples": len(eligible),
        "parent_seeds": parent_seeds,
        "per_class_limit": per_class_limit,
        "selection_seed": selection_seed if per_class_limit is not None else None,
        "selected_samples": len(selected),
        "valid_predictions": valid,
        "errors": errors,
        "completion_confusion": confusion,
        "completion_accuracy": (
            (confusion["TP"] + confusion["TN"]) / valid if valid and not errors else None
        ),
        "completion_recall": recall,
        "completion_precision": (
            confusion["TP"] / precision_denominator if precision_denominator and not errors else None
        ),
        "completion_specificity": specificity,
        "completion_balanced_accuracy": (
            (recall + specificity) / 2 if recall is not None and specificity is not None else None
        ),
        "outcome_class_coverage": label_coverage,
        "invalid_label_samples": invalid_labels,
        "outcome_confusion": joint,
        "outcome_accuracy": (
            sum(joint[name][name] for name in OUTCOME_CLASSES) / valid
            if valid and not errors else None
        ),
        "conditional_diagnosis_confusion": diagnoses,
        "conditional_diagnosis_valid_predictions": diagnosis_valid,
        "conditional_diagnosis_accuracy": (
            (diagnoses["StillTrying"]["StillTrying"] + diagnoses["Stuck"]["Stuck"])
            / diagnosis_valid if diagnosis_valid and not errors else None
        ),
        "conditional_diagnosis_coverage": diagnosis_valid / incomplete if incomplete else None,
        "diagnosis_note": (
            "Diagnosis is scored only for true No samples for which the model also predicted No. "
            "False Yes predictions have no diagnosis and reduce diagnosis coverage; "
            "the three-outcome confusion includes these gating failures."
        ),
        "selection": (
            "deterministically shuffled and bounded per completed/continuing/stuck class; "
            "artificially stratified subset, not natural outcome prevalence"
            if per_class_limit is not None else
            "deterministic positive/negative interleaving; positives first"
        ),
        "accuracy_note": (
            "No accuracy claimed because one or more selected samples failed."
            if errors else
            "Artificially stratified small subset; not natural-prevalence task accuracy."
            if per_class_limit is not None else
            "Small calibration subset only; not a benchmark performance claim."
        ),
        "simulator_truth_used_as_model_input": False,
        "diagnosis_accuracy_evaluated": True,
        "samples": records,
    }
