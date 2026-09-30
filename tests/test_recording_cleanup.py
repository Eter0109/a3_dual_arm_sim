"""A discarded episode must not leave its staged PNG frames on disk.

Frames are staged as PNGs and only removed once an episode is encoded, so
discarding has to clean them up.  LeRobot's ``clear_episode_buffer`` finds them
through ``meta.image_keys``, which is empty when the cameras are declared as
``video`` -- the mode the collection scripts now use by default.  The leak is
bounded to the last discarded episode of a run, because the staging directory is
named after the episode index, which does not advance on discard; a later
attempt overwrites it and a successful save deletes it.  That bounded leak is
still worth a test: it only shows up at the end of a long collection, which is
exactly when nobody is watching.

Driven through the real recorder rather than by calling the cleanup directly, so
the test fails if the cleanup stops being wired into ``discard_episode``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from a3_dual_arm_sim.contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    EpisodeContext,
)
from a3_dual_arm_sim.data.recording import LeRobotV3Recorder

CAMERAS = (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE)


def staged_dirs(root: Path) -> list[Path]:
    """Directories holding staged frames, for any camera."""
    images = root / "images"
    if not images.is_dir():
        return []
    return [path for path in images.glob("*/*") if path.is_dir()]


def fake_frame(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One observation and action.  Content is irrelevant; only the staging path is under test."""
    observation = {
        **{camera: rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for camera in CAMERAS},
        STATE: rng.normal(size=16).astype(np.float32),
        VELOCITY: rng.normal(size=16).astype(np.float32),
        EEF_POSE: rng.normal(size=14).astype(np.float32),
        FORCE: rng.normal(size=18).astype(np.float32),
    }
    return observation, rng.normal(size=16).astype(np.float32)


@pytest.mark.parametrize("use_videos", [True, False])
def test_discarded_episode_leaves_no_staged_frames(tmp_path: Path, use_videos: bool) -> None:
    pytest.importorskip("lerobot")
    rng = np.random.default_rng(0)
    root = tmp_path / "dataset"
    recorder = LeRobotV3Recorder(
        root,
        repo_id="local/a3-discard-test",
        fps=20,
        image_height=256,
        image_width=256,
        use_videos=use_videos,
    )
    try:
        recorder.start_episode(EpisodeContext(seed=1, task="discard test", action_mode="joint_position"), "test")
        for _ in range(4):
            observation, action = fake_frame(rng)
            recorder.add_frame(observation, action)
        assert staged_dirs(root), "frames should be staged on disk before the episode is discarded"
        recorder.discard_episode()
    finally:
        recorder.close()

    leftovers = staged_dirs(root)
    assert not leftovers, f"staged frames survived discard: {[p.name for p in leftovers]}"


def test_saved_episode_leaves_no_staged_frames(tmp_path: Path) -> None:
    """The counterpart guarantee: a saved episode cleans up too, in video mode."""
    pytest.importorskip("lerobot")
    rng = np.random.default_rng(1)
    root = tmp_path / "dataset"
    recorder = LeRobotV3Recorder(
        root,
        repo_id="local/a3-save-test",
        fps=20,
        image_height=256,
        image_width=256,
        use_videos=True,
    )
    try:
        recorder.start_episode(EpisodeContext(seed=1, task="save test", action_mode="joint_position"), "test")
        for _ in range(4):
            observation, action = fake_frame(rng)
            recorder.add_frame(observation, action)
        recorder.finish_episode(success=True)
    finally:
        recorder.close()

    leftovers = staged_dirs(root)
    assert not leftovers, f"staged frames survived save: {[p.name for p in leftovers]}"
    assert (root / "videos").is_dir(), "video mode should have produced encoded video"
