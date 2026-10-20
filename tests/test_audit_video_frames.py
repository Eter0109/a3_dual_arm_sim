from __future__ import annotations

from pathlib import Path

import numpy as np

from a3_dual_arm_sim.contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
)
from a3_dual_arm_sim.data.audit_smolvla_data_cli import sample_video_frames
from a3_dual_arm_sim.data.recording import ACTION, lerobot_features


def test_video_camera_frames_decode_through_the_dataset_api(tmp_path: Path) -> None:
    # Video-typed datasets keep frames in videos/ with no camera columns in the
    # data parquet; the audit must decode them via the dataset API instead of
    # indexing the parquet row (which raised KeyError before the fix).
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "video_dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/audit-video",
        fps=20,
        root=root,
        robot_type="A3_dual_arm",
        features=lerobot_features(256, 256, use_videos=True),
        use_videos=True,
        image_writer_threads=0,
    )
    for episode in range(2):
        for _ in range(3):
            dataset.add_frame(
                {
                    FRONT_IMAGE: np.full((256, 256, 3), episode * 80 + 10, np.uint8),
                    LEFT_WRIST_IMAGE: np.zeros((256, 256, 3), np.uint8),
                    RIGHT_WRIST_IMAGE: np.zeros((256, 256, 3), np.uint8),
                    STATE: np.zeros(16, np.float32),
                    VELOCITY: np.zeros(16, np.float32),
                    EEF_POSE: np.zeros(14, np.float32),
                    FORCE: np.zeros(18, np.float32),
                    ACTION: np.zeros(16, np.float32),
                    "task": "transfer 10 cookies into target box",
                }
            )
        dataset.save_episode()
    # Episodes metadata is buffered; flush it so the dataset is readable.
    dataset.finalize()

    samples = sample_video_frames(root, "local/audit-video", 2)
    assert sorted(samples) == [0, 1]
    for camera in ("front", "left_wrist", "right_wrist"):
        entry = samples[0][camera]
        assert 0 <= entry["min"] <= entry["max"] <= 255
        assert entry["std"] >= 0.0
    # The two episodes carry different front frames, so their statistics differ.
    assert samples[0]["front"] != samples[1]["front"]