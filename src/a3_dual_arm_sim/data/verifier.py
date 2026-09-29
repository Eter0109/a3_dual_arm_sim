"""Prepare temporally supervised visual-verifier data without truth leakage.

All attempts and dataset roots sharing a simulator seed stay in the same split.
Truth/audit files are never opened. Labels become assistant targets only; the
user message consists of the same prompt and RGB image order used at runtime.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image

from a3_dual_arm_sim.agents.prompts import verifier_completion_prompt, verifier_diagnosis_prompt
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest

OUTCOME_CLASSES = ("completed", "continuing", "stuck")
CAMERAS = ("front", "left_wrist")


def read_verifier_samples(data_root: Path) -> list[dict[str, Any]]:
    """Read public sample records only, never privileged truth/fault audits."""
    samples = []
    for line_number, line in enumerate(
        (data_root / "samples.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid verifier sample JSON at line {line_number}") from None
        if not isinstance(sample, dict):
            raise TypeError(f"verifier sample at line {line_number} is not an object")
        samples.append(sample)
    if not samples:
        raise ValueError("verifier dataset has no samples")
    return samples


def request_from_verifier_sample(sample: dict[str, Any]) -> SkillRequest:
    """Reconstruct exactly the saved subgoal, including legacy prompt-only data."""
    subgoal = sample["input"]["subgoal"]
    if not isinstance(subgoal, str):
        raise TypeError("sample subgoal must be text")
    raw = sample.get("request")
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("sample request must be an object")
        skill = CookieSkill(raw["skill"])
        batch, source, target = (
            raw["batch_index"], raw["source_column_number"], raw["target_slot_number"]
        )
        if any(type(number) is not int for number in (batch, source, target)):
            raise ValueError("sample request indices must be integers")
    else:
        skill = CookieSkill(sample["skill"])
        batch = sample["batch_index"]
        if type(batch) is not int:
            raise ValueError("sample batch index must be an integer")
        if skill is CookieSkill.PICK_FIVE:
            match = re.fullmatch(
                r"Pick up five cookies from source column ([1-4]), batch ([12])\.", subgoal
            )
            if match is None or int(match[2]) != batch + 1:
                raise ValueError("old PICK sample has an unsupported or inconsistent subgoal")
            source, target = int(match[1]), batch + 1
        else:
            match = re.fullmatch(
                r"Place the five held cookies into target box column ([12])\.", subgoal
            )
            if match is None:
                raise ValueError("old PLACE sample has an unsupported subgoal")
            source, target = 1, int(match[1])
    if batch not in (0, 1) or not 1 <= source <= 4 or target != batch + 1:
        raise ValueError("sample request is outside the single-box skill contract")
    if "skill" in sample and sample["skill"] != skill.value:
        raise ValueError("sample skill and request metadata disagree")
    if "batch_index" in sample and sample["batch_index"] != batch:
        raise ValueError("sample batch and request metadata disagree")
    result = SkillRequest(skill, batch, source, target)
    if result.instruction != subgoal:
        raise ValueError("sample request does not reproduce its recorded training subgoal")
    return result


def resolve_visual_frame_paths(
    data_root: Path, sample: dict[str, Any]
) -> tuple[list[int], list[Path]]:
    """Validate chronology and safe RGB paths in front/wrist interleaved order."""
    root = data_root.resolve(strict=True)
    frames = sample["input"]["frames"]
    if not isinstance(frames, list) or not 2 <= len(frames) <= 64:
        raise ValueError("sample must contain 2..64 temporal frame pairs")
    steps: list[int] = []
    paths: list[Path] = []
    for entry in frames:
        step = entry["step"]
        if type(step) is not int or step < 0 or (steps and step <= steps[-1]):
            raise ValueError("sample frame steps must be chronological increasing integers")
        steps.append(step)
        for camera in CAMERAS:
            relative = entry["images"][camera]
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or PureWindowsPath(relative).drive
            ):
                raise ValueError("saved camera paths must be relative to the verifier dataset")
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("saved camera path escapes the verifier dataset")
            paths.append(path)
    return steps, paths


def verifier_sample_outcome(sample: dict[str, Any]) -> str:
    """Validate both targets without deriving them from an injected fault name."""
    label = sample["label"]
    if not isinstance(label, dict):
        raise TypeError("verifier label must be an object")
    completion, diagnosis = label.get("completion"), label.get("diagnosis")
    if completion == "Yes" and diagnosis is None:
        return "completed"
    if completion == "No" and diagnosis in ("StillTrying", "Stuck"):
        return "continuing" if diagnosis == "StillTrying" else "stuck"
    raise ValueError("labels must be Yes/null or No/StillTrying or No/Stuck")


def _coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["outcome"] for record in records)
    return {
        "source_windows": len(records),
        "outcome_classes": {name: counts[name] for name in OUTCOME_CLASSES},
        "completion_targets": {
            "Yes": counts["completed"], "No": counts["continuing"] + counts["stuck"]
        },
        "diagnosis_targets": {"StillTrying": counts["continuing"], "Stuck": counts["stuck"]},
        "chat_examples": counts["completed"] + 2 * (counts["continuing"] + counts["stuck"]),
        "skill_outcome_classes": {
            skill.value: {
                outcome: sum(
                    record["request"].skill is skill and record["outcome"] == outcome
                    for record in records
                ) for outcome in OUTCOME_CLASSES
            } for skill in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE)
        },
    }


def prepare_verifier_dataset(
    data_roots: Sequence[Path],
    output_root: Path,
    *,
    holdout_seeds: Sequence[int] | None = None,
    holdout_fraction: float = 0.25,
    split_seed: int = 7,
    max_samples_per_class: int | None = None,
    require_class_coverage: bool = True,
    window_frames: int | None = None,
    frame_stride: int | None = None,
) -> dict[str, Any]:
    """Create new chat JSONL splits and copied, neutrally named visual windows.

    ``max_samples_per_class`` caps source windows independently in each split;
    it never manufactures, relabels, or moves samples between seed groups.
    Explicit ``holdout_seeds`` is preferable for predefined evaluation sets.
    """
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("use a new verifier training output root")
    if not data_roots or type(split_seed) is not int:
        raise ValueError("provide dataset roots and an integer split seed")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if max_samples_per_class is not None and (
        type(max_samples_per_class) is not int or max_samples_per_class < 1
    ):
        raise ValueError("max_samples_per_class must be a positive integer")
    if window_frames is not None and (
        type(window_frames) is not int or not 2 <= window_frames <= 64
    ):
        raise ValueError("window_frames must be an integer in 2..64")
    if frame_stride is not None and (type(frame_stride) is not int or frame_stride < 1):
        raise ValueError("frame_stride must be a positive integer")
    roots = [Path(root).resolve(strict=True) for root in data_roots]
    if len(set(roots)) != len(roots):
        raise ValueError("dataset roots must not be repeated")
    records = []
    skipped_initial_short_windows = 0
    input_windows = 0
    for root_index, root in enumerate(roots):
        seen_indices: set[int] = set()
        for sample in read_verifier_samples(root):
            input_windows += 1
            sample_index, seed = sample.get("sample_index"), sample.get("parent_seed")
            if type(sample_index) is not int or sample_index < 0 or sample_index in seen_indices:
                raise ValueError("sample_index must be a unique nonnegative integer within its root")
            if type(seed) is not int:
                raise ValueError("each sample needs an integer parent_seed for leakage-safe splitting")
            seen_indices.add(sample_index)
            request = request_from_verifier_sample(sample)
            steps, paths = resolve_visual_frame_paths(root, sample)
            outcome = verifier_sample_outcome(sample)
            if window_frames is not None:
                if len(steps) < window_frames:
                    skipped_initial_short_windows += 1
                    continue
                if len(steps) > window_frames:
                    raise ValueError("sample exceeds the expected temporal window_frames")
            if frame_stride is not None and any(
                second - first != frame_stride for first, second in pairwise(steps)
            ):
                raise ValueError("sample frame steps do not match the expected frame_stride")
            for path in paths:
                with Image.open(path) as image:
                    image.verify()
            records.append({
                "root_index": root_index, "sample_index": sample_index, "parent_seed": seed,
                "request": request, "frame_steps": steps, "paths": paths,
                "outcome": outcome,
            })
    seeds = sorted({record["parent_seed"] for record in records})
    if len(seeds) < 2:
        raise ValueError("at least two independent parent seeds are needed for train/holdout")
    if holdout_seeds is None:
        shuffled = seeds.copy()
        random.Random(split_seed).shuffle(shuffled)
        count = min(len(seeds) - 1, max(1, round(len(seeds) * holdout_fraction)))
        held_seeds = set(shuffled[:count])
    else:
        if not holdout_seeds or any(type(seed) is not int for seed in holdout_seeds):
            raise ValueError("holdout_seeds must contain integers")
        held_seeds = set(holdout_seeds)
        if not held_seeds < set(seeds):
            raise ValueError("holdout seeds must be present and leave at least one training seed")
    splits = {"train": [], "holdout": []}
    for record in records:
        split = "holdout" if record["parent_seed"] in held_seeds else "train"
        splits[split].append(record)
    available_coverage = {split: _coverage(items) for split, items in splits.items()}
    if max_samples_per_class is not None:
        for split, items in splits.items():
            selected = set()
            for class_index, outcome in enumerate(OUTCOME_CLASSES):
                indices = [index for index, item in enumerate(items) if item["outcome"] == outcome]
                random.Random(split_seed + class_index).shuffle(indices)
                selected.update(indices[:max_samples_per_class])
            splits[split] = [item for index, item in enumerate(items) if index in selected]
    coverage = {split: _coverage(items) for split, items in splits.items()}
    if require_class_coverage:
        for split, counts in coverage.items():
            missing = [name for name, count in counts["outcome_classes"].items() if count == 0]
            if missing:
                raise ValueError(f"{split} lacks verifier outcome classes: {', '.join(missing)}")
    output_root.mkdir(parents=True, exist_ok=False)
    output_root = output_root.resolve(strict=True)
    index = 0
    provenance = []
    evaluation_samples = []
    for split, items in splits.items():
        with (output_root / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for record in items:
                window_id = f"window_{index:08d}"
                image_directory = output_root / "images" / window_id
                image_directory.mkdir(parents=True, exist_ok=False)
                copied = []
                for image_index, source in enumerate(record["paths"]):
                    camera = CAMERAS[image_index % 2]
                    target = image_directory / f"{image_index // 2:02d}_{camera}{source.suffix.lower()}"
                    shutil.copyfile(source, target)
                    copied.append(str(target))
                request = record["request"]
                common = {
                    "parent_seed": record["parent_seed"], "split": split,
                    "skill": request.skill.value, "subgoal": request.instruction,
                    "frame_steps": record["frame_steps"], "images": copied,
                }
                completion = "Yes" if record["outcome"] == "completed" else "No"
                rows = [{
                    **common, "id": window_id + "_completion", "stage": "completion",
                    "prompt": verifier_completion_prompt(
                        request.skill, request.instruction, len(record["frame_steps"])
                    ), "answer": completion,
                }]
                if completion == "No":
                    rows.append({
                        **common, "id": window_id + "_diagnosis", "stage": "diagnosis",
                        "prompt": verifier_diagnosis_prompt(
                            request.instruction, len(record["frame_steps"])
                        ),
                        "answer": "StillTrying" if record["outcome"] == "continuing" else "Stuck",
                    })
                evaluation_samples.append({
                    "sample_index": index, "parent_seed": record["parent_seed"],
                    "skill": request.skill.value, "batch_index": request.batch_index,
                    "request": asdict(request),
                    "input": {
                        "subgoal": request.instruction,
                        "frames": [{
                            "step": step,
                            "images": {
                                camera: Path(copied[frame_index * 2 + camera_index])
                                .relative_to(output_root).as_posix()
                                for camera_index, camera in enumerate(CAMERAS)
                            },
                        } for frame_index, step in enumerate(record["frame_steps"])],
                    },
                    "label": {
                        "completion": completion,
                        "diagnosis": rows[1]["answer"] if completion == "No" else None,
                    },
                })
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                provenance.append({
                    "window_id": window_id, "root_index": record["root_index"],
                    "sample_index": record["sample_index"], "parent_seed": record["parent_seed"],
                    "split": split, "request": asdict(request),
                })
                index += 1
    with (output_root / "samples.jsonl").open("x", encoding="utf-8") as stream:
        for sample in evaluation_samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1, "output_root": str(output_root),
        "source_roots": [str(root) for root in roots],
        "train_seeds": sorted(set(seeds) - held_seeds), "holdout_seeds": sorted(held_seeds),
        "grouping": "parent_seed across all roots, attempts, skills, and adjacent windows",
        "split_seed": split_seed, "max_samples_per_class": max_samples_per_class,
        "coverage": coverage, "available_coverage": available_coverage,
        "camera_order": list(CAMERAS), "temporal_order": "oldest frame first",
        "input_windows": input_windows,
        "skipped_initial_short_windows": skipped_initial_short_windows,
        "temporal_contract": {
            "window_frames": window_frames, "frame_stride": frame_stride,
            "horizon_steps": (
                (window_frames - 1) * frame_stride
                if window_frames is not None and frame_stride is not None else None
            ),
        },
        "model_input_contains_labels": False, "simulator_truth_used_as_model_input": False,
        "audit_files_opened": False, "source_windows": provenance,
        "evaluation_samples_file": "samples.jsonl",
    }
    with (output_root / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest
