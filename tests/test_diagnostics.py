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
