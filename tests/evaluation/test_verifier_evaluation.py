from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from a3_dual_arm_sim.agents.backends import AgenticModelError
from a3_dual_arm_sim.agents.verification import (
    TemporalSkillVerifier,
    VerificationDecision,
)
from a3_dual_arm_sim.cli.evaluation import verifier as verifier_cli
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest
from a3_dual_arm_sim.data.verifier import request_from_verifier_sample
from a3_dual_arm_sim.evaluation import verifier as evaluation


def saved_sample(root: Path, index: int, label: str, *, request_metadata=False):
    request = SkillRequest(CookieSkill.PICK_FIVE, 0, 3, 1)
    frames = []
    for frame_index in range(2):
        paths = {}
        for camera_index, camera in enumerate(("front", "left_wrist", "right_wrist")):
            relative = Path("images") / f"{index}_{frame_index}_{camera}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            value = index * 20 + frame_index * 3 + camera_index
            Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)
            paths[camera] = relative.as_posix()
        frames.append({"step": frame_index * 5, "images": paths})
    result = {
        "sample_index": index, "skill": request.skill.value, "batch_index": request.batch_index,
        "input": {"subgoal": request.instruction, "frames": frames},
        "label": {"completion": label, "diagnosis": None if label == "Yes" else "StillTrying"},
    }
    if request_metadata:
        result["request"] = {
            "skill": request.skill.value, "batch_index": request.batch_index,
            "source_column_number": request.source_column_number,
            "target_slot_number": request.target_slot_number,
        }
    return result


def save_dataset(root, samples):
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    (root / "truth_audit.jsonl").write_text("PRIVATE_TRUTH_MARKER", encoding="utf-8")


def test_balanced_selection_includes_positive_and_is_deterministic():
    samples = [
        {"sample_index": index, "label": {"completion": label}}
        for index, label in enumerate(("No", "No", "Yes", "No", "Yes", "Yes"))
    ]
    selected = evaluation.select_balanced_samples(samples, limit=4)
    assert [sample["sample_index"] for sample in selected] == [2, 0, 4, 1]
    assert evaluation.select_balanced_samples(samples, limit=1)[0]["sample_index"] == 2
    assert len(evaluation.select_balanced_samples(samples, limit=None)) == 6


def test_selection_keeps_invalid_labels_in_error_accounting():
    samples = [{"label": {"completion": "Unknown"}}, {"label": {"completion": "Yes"}}]
    assert len(evaluation.select_balanced_samples(samples, limit=None)) == 2
    with pytest.raises(ValueError):
        evaluation.select_balanced_samples(samples, limit=0)


def test_per_class_selection_includes_stuck_deterministically_and_keeps_bad_labels():
    samples = []
    for index, outcome in enumerate(["continuing"] * 10 + ["stuck"] * 4 + ["completed"] * 3):
        samples.append({"sample_index": index, "label": {
            "completion": "Yes" if outcome == "completed" else "No",
            "diagnosis": None if outcome == "completed" else
            "StillTrying" if outcome == "continuing" else "Stuck",
        }})
    samples.append({"sample_index": 17, "label": {"completion": "Invalid"}})
    first = evaluation.select_class_balanced_samples(samples, 2, selection_seed=13)
    second = evaluation.select_class_balanced_samples(samples, 2, selection_seed=13)
    assert first == second and len(first) == 7
    assert sum(sample["label"].get("diagnosis") == "Stuck" for sample in first) == 2
    assert any(sample["sample_index"] == 17 for sample in first)
    with pytest.raises(ValueError):
        evaluation.select_class_balanced_samples(samples, 0)


@pytest.mark.parametrize("metadata", [False, True])
def test_request_reconstruction_matches_saved_skill_prompt(tmp_path, metadata):
    sample = saved_sample(tmp_path, 0, "Yes", request_metadata=metadata)
    request = request_from_verifier_sample(sample)
    assert request == SkillRequest(CookieSkill.PICK_FIVE, 0, 3, 1)
    assert request.instruction == sample["input"]["subgoal"]


def test_legacy_place_uses_only_its_actual_instruction():
    sample = {
        "skill": "PLACE_FIVE", "batch_index": 1,
        "input": {"subgoal": "Place the five held cookies into target box column 2."},
    }
    request = request_from_verifier_sample(sample)
    assert request.source_column_number == 1
    assert request.target_slot_number == 2
    assert request.instruction == sample["input"]["subgoal"]


@pytest.mark.parametrize("mutation", [
    {"batch_index": True}, {"batch_index": 1}, {"skill": "PUSH_BOX"},
    {"request": {"skill": "PICK_FIVE", "batch_index": 0,
                 "source_column_number": 3, "target_slot_number": 2}},
])
def test_inconsistent_requests_are_rejected(tmp_path, mutation):
    sample = saved_sample(tmp_path, 0, "Yes")
    sample.update(mutation)
    with pytest.raises((ValueError, KeyError)):
        request_from_verifier_sample(sample)


def test_unknown_prompt_is_not_replaced_with_a_default_skill():
    with pytest.raises(ValueError):
        request_from_verifier_sample({
            "skill": "PICK_FIVE", "batch_index": 0,
            "input": {"subgoal": "Pick some cookies wherever they are."},
        })


def test_camera_loading_ignores_right_wrist_and_preserves_frame_order(tmp_path):
    sample = saved_sample(tmp_path, 0, "Yes")
    sample["input"]["frames"][0]["images"]["right_wrist"] = "does-not-exist.jpg"
    frames = evaluation.load_visual_frames(tmp_path, sample)
    assert len(frames) == 2
    assert set(frames[0]) == {"front", "left_wrist"}
    assert frames[0]["front"].dtype == np.uint8
    assert int(frames[0]["front"][0, 0, 0]) < int(frames[1]["front"][0, 0, 0])


def test_saved_image_paths_cannot_escape_dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    sample = saved_sample(root, 0, "Yes")
    external = tmp_path / "private.jpg"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(external)
    sample["input"]["frames"][0]["images"]["front"] = "../private.jpg"
    with pytest.raises(ValueError, match="escapes"):
        evaluation.load_visual_frames(root, sample)


def test_nonchronological_frames_are_rejected(tmp_path):
    sample = saved_sample(tmp_path, 0, "Yes")
    sample["input"]["frames"][1]["step"] = 0
    with pytest.raises(ValueError, match="chronological"):
        evaluation.load_visual_frames(tmp_path, sample)


def test_confusion_matrix_scores_all_four_cases_without_truth_model_input(tmp_path):
    samples = [saved_sample(tmp_path, index, label) for index, label in enumerate(
        ("Yes", "Yes", "No", "No")
    )]
    save_dataset(tmp_path, samples)

    class Verifier:
        def __init__(self):
            self.statuses = iter(("completed", "completed", "continuing", "stuck"))
            self.calls = []

        def check(self, request, frames):
            self.calls.append((request, frames))
            return VerificationDecision(next(self.statuses), "mock visual decision")

    verifier = Verifier()
    result = evaluation.evaluate_samples(tmp_path, verifier, limit=None)
    # Balanced order is positive0, negative2, positive1, negative3.
    assert result["completion_confusion"] == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
    assert result["completion_accuracy"] == 0.5
    assert result["errors"] == 0 and result["valid_predictions"] == 4
    assert len(verifier.calls) == 4
    assert all(set(frames[0]) == {"front", "left_wrist"} for _, frames in verifier.calls)
    assert result["simulator_truth_used_as_model_input"] is False


def test_model_errors_are_counted_and_suppress_accuracy_claims(tmp_path):
    samples = [saved_sample(tmp_path, 0, "Yes"), saved_sample(tmp_path, 1, "No")]
    save_dataset(tmp_path, samples)

    class Verifier:
        calls = 0

        def check(self, request, frames):
            self.calls += 1
            if self.calls == 1:
                raise AgenticModelError("service connection failed")
            return VerificationDecision("continuing", "observed motion")

    result = evaluation.evaluate_samples(tmp_path, Verifier())
    assert result["errors"] == 1 and result["valid_predictions"] == 1
    assert result["completion_accuracy"] is None
    assert result["completion_confusion"]["TN"] == 1
    assert "connection failed" in result["samples"][0]["error"]
    assert result["outcome_class_coverage"] == {"completed": 1, "continuing": 1, "stuck": 0}
    assert result["conditional_diagnosis_accuracy"] is None


def test_conditional_diagnosis_and_gating_errors_are_reported_separately(tmp_path):
    samples = [saved_sample(tmp_path, index, "No") for index in range(4)]
    for sample in samples[2:]:
        sample["label"]["diagnosis"] = "Stuck"
    save_dataset(tmp_path, samples)

    class Verifier:
        statuses = iter(("continuing", "stuck", "stuck", "completed"))

        def check(self, request, frames):
            return VerificationDecision(next(self.statuses), "test")

    result = evaluation.evaluate_samples(tmp_path, Verifier(), limit=None)
    assert result["completion_confusion"] == {"TP": 0, "FP": 1, "FN": 0, "TN": 3}
    assert result["completion_recall"] is None
    assert result["completion_specificity"] == 0.75
    assert result["conditional_diagnosis_valid_predictions"] == 3
    assert result["conditional_diagnosis_accuracy"] == pytest.approx(2 / 3)
    assert result["conditional_diagnosis_coverage"] == 0.75
    assert result["conditional_diagnosis_confusion"] == {
        "StillTrying": {"StillTrying": 1, "Stuck": 1},
        "Stuck": {"StillTrying": 0, "Stuck": 1},
    }
    assert result["outcome_confusion"]["stuck"]["completed"] == 1
    assert result["outcome_accuracy"] == 0.5
    assert result["samples"][-1]["diagnosis_scored"] is False


def test_holdout_seed_filter_never_scores_training_seeds(tmp_path):
    samples = [saved_sample(tmp_path, index, "Yes") for index in range(3)]
    for index, sample in enumerate(samples):
        sample["parent_seed"] = index
    save_dataset(tmp_path, samples)

    class Verifier:
        calls = 0

        def check(self, request, frames):
            self.calls += 1
            return VerificationDecision("completed", "test")

    verifier = Verifier()
    result = evaluation.evaluate_samples(tmp_path, verifier, limit=None, parent_seeds=[2])
    assert verifier.calls == 1
    assert result["eligible_samples"] == 1
    assert result["available_samples"] == 3
    assert result["samples"][0]["sample_index"] == 2
    with pytest.raises(ValueError, match="no samples"):
        evaluation.evaluate_samples(tmp_path, verifier, parent_seeds=[9])


def test_diagnosis_label_not_invented_and_invalid_label_counts_as_error(tmp_path):
    sample = saved_sample(tmp_path, 0, "No")
    sample["label"]["diagnosis"] = None
    save_dataset(tmp_path, [sample])

    class Verifier:
        def check(self, request, frames):
            return VerificationDecision("continuing", "test")

    result = evaluation.evaluate_samples(tmp_path, Verifier(), limit=None)
    assert result["errors"] == 1
    assert result["invalid_label_samples"] == 1
    assert result["conditional_diagnosis_valid_predictions"] == 0


def test_temporal_verifier_backend_receives_only_images_and_subgoal_not_labels(tmp_path):
    sample = saved_sample(tmp_path, 0, "Yes")
    sample["label"]["private_annotation"] = "DO_NOT_PASS_LABEL_METADATA"
    save_dataset(tmp_path, [sample])

    class Backend:
        def complete(self, prompt, images):
            assert "PRIVATE_TRUTH_MARKER" not in prompt
            assert "DO_NOT_PASS_LABEL_METADATA" not in prompt
            assert all(isinstance(image, np.ndarray) for image in images)
            assert len(images) == 4
            return "Yes"

    result = evaluation.evaluate_samples(tmp_path, TemporalSkillVerifier(Backend()))
    assert result["completion_confusion"]["TP"] == 1


def test_cli_reports_backend_metadata_and_writes_summary_without_downloads(tmp_path, monkeypatch):
    sample = saved_sample(tmp_path, 0, "Yes")
    save_dataset(tmp_path, [sample])
    config = tmp_path / "agent.yaml"
    config.write_text(
        "verifier:\n  kind: local_qwen\n  model: local/mock-model\n  fine_tuned: false\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    class Backend:
        closed = False

        def complete(self, prompt, images):
            return "Yes"

        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr(verifier_cli, "make_chat_backend", lambda config: backend)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_cookie_verifier.py", "--data-root", str(tmp_path),
        "--agent-config", str(config), "--all", "--output", str(output),
    ])
    assert verifier_cli.main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["model"] == "local/mock-model"
    assert summary["visual_verifier_fine_tuned"] is False
    assert summary["selected_samples"] == 1 and backend.closed


def test_cli_reports_actually_loaded_adapter_before_close_not_config_marker(tmp_path, monkeypatch):
    sample = saved_sample(tmp_path, 0, "Yes")
    save_dataset(tmp_path, [sample])
    config = tmp_path / "agent.yaml"
    config.write_text(
        "verifier:\n  kind: local_qwen\n  model: local/mock-model\n"
        "  fine_tuned: true\n  adapter_path: local/mock-adapter\n", encoding="utf-8",
    )
    output = tmp_path / "result.json"

    class Backend:
        adapter_loaded = False

        def complete(self, prompt, images):
            self.adapter_loaded = True
            return "Yes"

        def close(self):
            self.adapter_loaded = False

    backend = Backend()
    monkeypatch.setattr(verifier_cli, "make_chat_backend", lambda config: backend)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_cookie_verifier.py", "--data-root", str(tmp_path),
        "--agent-config", str(config), "--per-class-limit", "1", "--output", str(output),
    ])
    assert verifier_cli.main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["adapter_loaded"] is True
    assert summary["visual_verifier_fine_tuned"] is True
    assert summary["configured_fine_tuned"] is True
    assert summary["adapter_path"] == "local/mock-adapter"
    assert "stratified" in summary["selection"]
    assert backend.adapter_loaded is False
