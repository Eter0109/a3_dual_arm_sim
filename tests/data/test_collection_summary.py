import json

from a3_dual_arm_sim.data.verifier_collection import write_collection_summary


def test_verifier_collection_summary_counts_actual_labels_not_case_names(tmp_path):
    labels = [
        {"completion": "Yes", "diagnosis": None},
        {"completion": "No", "diagnosis": "StillTrying"},
        {"completion": "No", "diagnosis": "Stuck"},
    ]
    (tmp_path / "samples.jsonl").write_text(
        "\n".join(json.dumps({"label": label}) for label in labels) + "\n",
        encoding="utf-8",
    )
    results = [{"case": "natural", "whole_box_success": False}]
    summary = write_collection_summary(tmp_path, results)
    assert summary["sample_class_counts"] == {"completed": 1, "continuing": 1, "stuck": 1}
    assert summary["training_started"] is False
    assert summary["collection_status"] == "complete"
    assert summary["results"][0]["whole_box_success"] is False
    assert json.loads((tmp_path / "collection_summary.json").read_text()) == summary


def test_empty_collection_is_reported_without_inventing_labels(tmp_path):
    summary = write_collection_summary(tmp_path, [])
    assert summary["attempts"] == 0
    assert not any(summary["sample_class_counts"].values())
