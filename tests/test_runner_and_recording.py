from __future__ import annotations

import json
from pathlib import Path

import pytest

from a3_dual_arm_sim.env import A3DualArmEnv
from a3_dual_arm_sim.policy import HoldPolicy, load_policy
from a3_dual_arm_sim.recording import LeRobotV3Recorder, MemoryRecorder
from a3_dual_arm_sim.runner import EpisodeRunner
from a3_dual_arm_sim.teleop import KeyboardTeleopPolicy


def test_policy_loader_and_source_neutral_runner() -> None:
    policy = load_policy("a3_dual_arm_sim.policy:make_hold_policy")
    recorder = MemoryRecorder()
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, policy, task="test task", recorder=recorder)
    try:
        result = runner.run(seed=3, max_steps=3)
        assert result.steps == 3
        assert len(recorder.episodes) == 1
        assert len(recorder.episodes[0]) == 3
        assert recorder.episodes[0][0]["action"].shape == (16,)
    finally:
        runner.close()


def test_keyboard_events_select_arm_pause_stop_and_discard() -> None:
    policy = KeyboardTeleopPolicy()
    policy.handle_key(ord("2"))
    policy.handle_key(ord("w"))
    action = policy.act({}, "")
    assert action[6] == 1.0
    assert action[0] == 0.0
    assert action[7] == 1.0
    assert action[13] == 1.0
    policy.handle_key(ord("p"))
    assert not policy.recording
    policy.handle_key(ord("x"))
    assert policy.stop_requested and policy.discard_requested
    policy.handle_key(ord(" "))
    assert policy.emergency_requested


def test_lerobot_v3_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    root = tmp_path / "dataset"
    env = A3DualArmEnv(render_cameras=False)
    recorder = LeRobotV3Recorder(
        root,
        repo_id="local/a3-test",
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
    )
    runner = EpisodeRunner(env, HoldPolicy(), task="dataset test", recorder=recorder)
    try:
        result = runner.run(seed=9, max_steps=2)
        assert result.steps == 2
    finally:
        runner.close()
    metadata = [json.loads(line) for line in (root / "a3_episode_metadata.jsonl").read_text().splitlines()]
    assert metadata == [
        {
            "episode_index": 0,
            "seed": 9,
            "task": "dataset test",
            "controller_type": "HoldPolicy",
            "source_action_mode": "joint_position",
            "stored_action_mode": "joint_position",
            "frames": 2,
            "success": None,
        }
    ]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "local/a3-test", root=root, download_videos=False, return_uint8=True
    )
    assert len(dataset) == 2
    assert dataset.num_episodes == 1
    assert tuple(dataset[0]["action"].shape) == (16,)
    assert tuple(dataset[0]["observation.force"].shape) == (18,)
    assert dataset[0]["task"] == "dataset test"
