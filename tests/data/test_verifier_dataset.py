from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from a3_dual_arm_sim.agents.prompts import (
    verifier_completion_prompt,
    verifier_diagnosis_prompt,
)
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest
from a3_dual_arm_sim.data.verifier import prepare_verifier_dataset, read_verifier_samples
from a3_dual_arm_sim.evaluation import verifier as evaluation


def source_dataset(root: Path, *, seeds=(0, 1), outcomes=("completed", "continuing", "stuck")):
    root.mkdir(parents=True)
    samples = []
    for seed in seeds:
        for outcome in outcomes:
            index = len(samples)
            request = SkillRequest(CookieSkill.PICK_FIVE, 0, 1, 1)
            frames = []
            for frame in range(3):
                images = {}
                for camera_index, camera in enumerate(("front", "left_wrist", "right_wrist")):
                    relative = Path("images") / f"{index}_{frame}_{camera}.png"
                    (root / relative).parent.mkdir(exist_ok=True)
                    value = (index * 20 + frame * 3 + camera_index) % 256
                    Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(root / relative)
                    images[camera] = relative.as_posix()
                frames.append({"step": frame * 10, "images": images})
            samples.append({
                "sample_index": index, "parent_seed": seed, "attempt": index,
                "request": asdict(request), "skill": request.skill.value, "batch_index": 0,
                "input": {"subgoal": request.instruction, "frames": frames},
                "label": {
                    "completion": "Yes" if outcome == "completed" else "No",
                    "diagnosis": None if outcome == "completed" else
                    "StillTrying" if outcome == "continuing" else "Stuck",
                    "private_marker": "DO_NOT_PASS_LABEL_METADATA",
                },
            })
    save_source(root, samples)
    (root / "truth_audit.jsonl").write_text("DO_NOT_OPEN_PRIVILEGED_TRUTH", encoding="utf-8")
    (root / "fault_audit.jsonl").write_text("DO_NOT_OPEN_FAULT_MARKER", encoding="utf-8")
    return samples


def save_source(root, samples):
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_three_class_conversion_uses_runtime_prompts_and_no_privileged_metadata(tmp_path):
    root, output = tmp_path / "injected_drop_dataset", tmp_path / "prepared"
    source_dataset(root)
    manifest = prepare_verifier_dataset([root], output, holdout_seeds=[1])
    for split, seed in (("train", 0), ("holdout", 1)):
        rows = read_rows(output / f"{split}.jsonl")
        assert len(rows) == 5
        assert {row["parent_seed"] for row in rows} == {seed}
        assert [row["answer"] for row in rows] == ["Yes", "No", "StillTrying", "No", "Stuck"]
        for row in rows:
            assert row["frame_steps"] == [0, 10, 20]
            assert len(row["images"]) == 6
            assert "injected_drop" not in str(row)
            assert "DO_NOT_" not in str(row)
            for image_index, path in enumerate(row["images"]):
                assert Path(path).is_file() and Path(path).is_relative_to(output)
                assert ("front" if image_index % 2 == 0 else "left_wrist") in Path(path).name
            request = SkillRequest(CookieSkill.PICK_FIVE, 0, 1, 1)
            expected_prompt = (
                verifier_completion_prompt(request.skill, request.instruction, 3)
                if row["stage"] == "completion" else
                verifier_diagnosis_prompt(request.instruction, 3)
            )
            assert row["prompt"] == expected_prompt
    assert manifest["coverage"]["train"]["outcome_classes"] == {
        "completed": 1, "continuing": 1, "stuck": 1,
    }
    assert manifest["model_input_contains_labels"] is False
    assert manifest["audit_files_opened"] is False
    evaluation_samples = read_verifier_samples(output)
    assert len(evaluation_samples) == 6
    assert len(evaluation.load_visual_frames(output, evaluation_samples[0])) == 3
    assert {sample["parent_seed"] for sample in evaluation_samples} == {0, 1}


def test_all_roots_and_attempts_with_same_seed_stay_in_one_split(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    source_dataset(first, seeds=(0, 1, 2))
    source_dataset(second, seeds=(0, 1, 2))
    output = tmp_path / "output"
    manifest = prepare_verifier_dataset([first, second], output, holdout_seeds=[2])
    train, holdout = read_rows(output / "train.jsonl"), read_rows(output / "holdout.jsonl")
    assert {row["parent_seed"] for row in train} == {0, 1}
    assert {row["parent_seed"] for row in holdout} == {2}
    assert manifest["coverage"]["holdout"]["source_windows"] == 6
    assert not {row["images"][0] for row in train} & {row["images"][0] for row in holdout}


def test_same_split_seed_produces_same_groups_and_cap_does_not_move_seeds(tmp_path):
    root = tmp_path / "root"
    source_dataset(root, seeds=tuple(range(6)))
    first = prepare_verifier_dataset([root], tmp_path / "out1", split_seed=13,
                                     max_samples_per_class=2)
    second = prepare_verifier_dataset([root], tmp_path / "out2", split_seed=13,
                                      max_samples_per_class=2)
    assert first["train_seeds"] == second["train_seeds"]
    assert first["holdout_seeds"] == second["holdout_seeds"]
    assert first["coverage"] == second["coverage"]
    assert all(count <= 2 for split in first["coverage"].values()
               for count in split["outcome_classes"].values())


@pytest.mark.parametrize("mutation", [
    {"parent_seed": True}, {"parent_seed": None}, {"sample_index": True},
    {"sample_index": -1},
    {"label": {"completion": "Yes", "diagnosis": "Stuck"}},
    {"label": {"completion": "No", "diagnosis": None}},
    {"label": {"completion": "No", "diagnosis": "Failed"}},
])
def test_bad_identity_or_labels_fail_without_creating_output(tmp_path, mutation):
    root, output = tmp_path / "root", tmp_path / "prepared"
    samples = source_dataset(root)
    samples[0].update(mutation)
    save_source(root, samples)
    with pytest.raises(ValueError):
        prepare_verifier_dataset([root], output)
    assert not output.exists()


def test_duplicate_sample_ids_are_rejected_within_root(tmp_path):
    root = tmp_path / "root"
    samples = source_dataset(root)
    samples[1]["sample_index"] = samples[0]["sample_index"]
    save_source(root, samples)
    with pytest.raises(ValueError, match="unique"):
        prepare_verifier_dataset([root], tmp_path / "out")


@pytest.mark.parametrize("bad_path", ["../outside.png", "C:\\private\\secret.png"])
def test_unsafe_image_paths_are_rejected(tmp_path, bad_path):
    root = tmp_path / "root"
    samples = source_dataset(root)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "outside.png")
    samples[0]["input"]["frames"][0]["images"]["front"] = bad_path
    save_source(root, samples)
    with pytest.raises(ValueError):
        prepare_verifier_dataset([root], tmp_path / "out")


@pytest.mark.parametrize("step", [-1, 0, True])
def test_invalid_chronology_is_rejected(tmp_path, step):
    root = tmp_path / "root"
    samples = source_dataset(root)
    samples[0]["input"]["frames"][1]["step"] = step
    save_source(root, samples)
    with pytest.raises(ValueError, match="chronological"):
        prepare_verifier_dataset([root], tmp_path / "out")


def test_one_seed_cannot_fake_independent_holdout(tmp_path):
    root = tmp_path / "root"
    source_dataset(root, seeds=(7,))
    with pytest.raises(ValueError, match="independent"):
        prepare_verifier_dataset([root], tmp_path / "out")


@pytest.mark.parametrize("held", [[0, 1], [9], [], [True]])
def test_invalid_holdout_seed_list_rejected(tmp_path, held):
    root = tmp_path / "root"
    source_dataset(root)
    with pytest.raises(ValueError, match="holdout"):
        prepare_verifier_dataset([root], tmp_path / "out", holdout_seeds=held)


def test_missing_class_is_reported_not_invented(tmp_path):
    root = tmp_path / "root"
    source_dataset(root, outcomes=("completed", "continuing"))
    with pytest.raises(ValueError, match="lacks.*stuck"):
        prepare_verifier_dataset([root], tmp_path / "rejected")
    manifest = prepare_verifier_dataset([root], tmp_path / "diagnostic",
                                        require_class_coverage=False)
    assert manifest["coverage"]["train"]["outcome_classes"]["stuck"] == 0


def test_existing_output_and_duplicate_roots_rejected(tmp_path):
    root = tmp_path / "root"
    source_dataset(root)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_verifier_dataset([root], output)
    assert marker.read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ValueError, match="repeated"):
        prepare_verifier_dataset([root, root], tmp_path / "duplicate")


def test_expected_temporal_contract_filters_short_initial_windows_and_reports_it(tmp_path):
    root = tmp_path / "root"
    samples = source_dataset(root)
    short = dict(samples[1])
    short["sample_index"] = len(samples)
    short["input"] = {
        **short["input"], "frames": short["input"]["frames"][:2],
    }
    samples.append(short)
    save_source(root, samples)
    manifest = prepare_verifier_dataset(
        [root], tmp_path / "out", holdout_seeds=[1], window_frames=3, frame_stride=10,
    )
    assert manifest["input_windows"] == 7
    assert manifest["skipped_initial_short_windows"] == 1
    assert manifest["coverage"]["train"]["source_windows"] == 3
    assert manifest["temporal_contract"] == {
        "window_frames": 3, "frame_stride": 10, "horizon_steps": 20,
    }


def test_full_windows_with_unexpected_stride_are_not_silently_mixed(tmp_path):
    root = tmp_path / "root"
    samples = source_dataset(root)
    samples[0]["input"]["frames"][1]["step"] = 5
    save_source(root, samples)
    with pytest.raises(ValueError, match="frame_stride"):
        prepare_verifier_dataset([root], tmp_path / "out", window_frames=3, frame_stride=10)


def test_class_coverage_is_checked_after_short_window_filtering(tmp_path):
    root = tmp_path / "root"
    samples = source_dataset(root)
    samples[2]["input"]["frames"] = samples[2]["input"]["frames"][:2]
    save_source(root, samples)
    with pytest.raises(ValueError, match="train lacks.*stuck"):
        prepare_verifier_dataset([root], tmp_path / "out", holdout_seeds=[1], window_frames=3)
