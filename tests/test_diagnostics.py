import json

import cv2
import numpy as np

from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic


def test_trace_keeps_before_observation_and_distinct_actions(tmp_path):
    diagnostic = EpisodeDiagnostic(tmp_path / "trace")
    observation = {
        "observation.state": np.arange(16),
        "observation.images.front": np.full((32, 32, 3), 127, dtype=np.uint8),
    }
    diagnostic.add_frame(
        observation,
        np.ones(16),
        {"applied_action": np.full(16, 2), "cookies_in_target": 5},
        raw_action=np.zeros(16),
        phase="RELEASE",
    )
    diagnostic.close()
    row = json.loads((tmp_path / "trace/trace.jsonl").read_text())
    assert row["observation_before"]["observation.state"] == list(range(16))
    assert row["raw_action"] == [0] * 16
    assert row["processed_action"] == [1] * 16
    assert row["applied_action"] == [2] * 16
    reader = cv2.VideoCapture(str(tmp_path / "trace/front.mp4"))
    ok, frame = reader.read()
    reader.release()
    assert ok and frame.shape == (32, 32, 3)


def test_trace_identifies_missing_slots_and_boundary(tmp_path):
    diagnostic = EpisodeDiagnostic(tmp_path / "trace")
    diagnostic.add_frame(
        {"observation.state": np.zeros(16)},
        np.zeros(16),
        {
            "applied_action": np.zeros(16),
            "cookies_in_target": 10,
            "cookies_in_source": 70,
            "count_fill": True,
            "target_slot_occupancy": [0, 1, 2, 3, 4, 5, 6, 7, 8, -1],
            "target_touches_all_walls": False,
            "exact_2x5_fill": False,
            "success_criterion": "exact_slots",
            "success_hold_count": 0,
            "required_success_hold_steps": 5,
            "success": False,
        },
    )
    diagnostic.close()
    row = json.loads((tmp_path / "trace/trace.jsonl").read_text())
    assert row["after"]["unmet_success_conditions"] == [
        "slot_occupancy",
        "boundary_coverage",
        "hold_steps",
    ]
    summary = json.loads((tmp_path / "trace/success_diagnostics.json").read_text())
    assert summary["first_count_fill_step"] == 0
    assert summary["first_exact_fill_step"] is None
    assert summary["max_success_hold_count"] == 0
