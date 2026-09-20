from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from a3_dual_arm_sim.cli import _validate_camera_recording, parser
from a3_dual_arm_sim.env import A3DualArmEnv
from a3_dual_arm_sim.policy import HoldPolicy, load_policy
from a3_dual_arm_sim.recording import LeRobotV3Recorder, MemoryRecorder
from a3_dual_arm_sim.runner import EpisodeRunner
from a3_dual_arm_sim.teleop import KeyboardTeleopPolicy


def test_run_and_teleop_expose_explicit_policy_camera_switch() -> None:
    run_args = parser().parse_args(
        ["run", "--policy", "package.module:factory", "--no-camera-render"]
    )
    teleop_args = parser().parse_args(["teleop", "--no-camera-render"])
    assert not run_args.camera_render
    assert not teleop_args.camera_render


def test_recording_rejects_disabled_policy_cameras(tmp_path: Path) -> None:
    args = parser().parse_args(
        ["teleop", "--no-camera-render", "--record", str(tmp_path / "dataset")]
    )
    with pytest.raises(ValueError, match="black policy-camera frames"):
        _validate_camera_recording(args)


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


def test_policy_can_declare_success_the_environment_cannot_see():
    """A policy may know the task was achieved when ``info`` says otherwise.

    The two-box relay is the case: its criterion is ten Cookies in *each* box with
    sixty left in the source, which the single-bin task config cannot express, so the
    environment reports ``success=False`` for an episode that fully succeeded.  The
    override has to be applied inside the runner rather than by the caller, because
    it also feeds the discard rule -- deciding it afterwards would throw away the
    episodes it exists to keep.
    """

    class _DeclaringPolicy(HoldPolicy):
        #: Claims success even though the environment never reports it.
        success_override = True

    policy = _DeclaringPolicy()
    recorder = MemoryRecorder()
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(
        env, policy, task="test task", recorder=recorder, save_failed_episodes=False
    )
    try:
        result = runner.run(seed=5, max_steps=3)
        assert not result.final_info.get("success", False), (
            "this test is only meaningful while the environment disagrees"
        )
        assert result.success, "the policy's declaration must win"
        assert not result.discarded, "an episode the policy calls a success must be kept"
        assert len(recorder.episodes) == 1
    finally:
        runner.close()


def test_episode_is_discarded_when_neither_side_reports_success():
    """The override must not turn every episode into a success."""

    recorder = MemoryRecorder()
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(
        env,
        HoldPolicy(),
        task="test task",
        recorder=recorder,
        save_failed_episodes=False,
    )
    try:
        result = runner.run(seed=5, max_steps=3)
        assert not result.success
        assert result.discarded
        assert recorder.episodes == []
    finally:
        runner.close()


def test_keyboard_events_select_arm_pause_stop_and_discard() -> None:
    policy = KeyboardTeleopPolicy()
    policy.handle_key(ord("2"))
    policy.handle_key(ord("w"))
    action = policy.act({}, "")
    assert action[6] == 1.0
    assert action[0] == 0.0
    assert action[7] == 0.5
    assert action[13] == 1.0
    policy.handle_key(ord("p"))
    assert not policy.recording
    policy.handle_key(ord("x"))
    assert policy.stop_requested and policy.discard_requested
    policy.handle_key(ord(" "))
    assert policy.emergency_requested


@pytest.mark.parametrize(
    ("positive_key", "negative_key", "axis"),
    [
        ("w", "s", 0),
        ("a", "d", 1),
        ("r", "f", 2),
        ("i", "k", 3),
        ("j", "l", 4),
        ("u", "o", 5),
    ],
)
def test_keyboard_motion_pairs_have_opposite_signs_for_both_arms(
    positive_key: str, negative_key: str, axis: int
) -> None:
    policy = KeyboardTeleopPolicy()
    policy.handle_key(ord("3"))
    policy.handle_key(ord(positive_key))
    positive = policy.act({}, "").reshape(2, 7)
    policy.handle_key(ord(negative_key))
    negative = policy.act({}, "").reshape(2, 7)
    np.testing.assert_array_equal(positive[:, axis], (0.5, 0.5))
    np.testing.assert_array_equal(negative[:, axis], (-0.5, -0.5))


def test_held_key_continues_until_release_and_respects_speed() -> None:
    policy = KeyboardTeleopPolicy(speed=0.25)
    policy.set_key_state("w", True)
    first = policy.act({}, "")
    second = policy.act({}, "")
    assert first[0] == pytest.approx(0.25)
    assert second[0] == pytest.approx(0.25)
    policy.set_key_state("w", False)
    released = policy.act({}, "")
    assert released[0] == pytest.approx(0.0)


def test_debug_policy_cannot_enable_recording() -> None:
    policy = KeyboardTeleopPolicy(recording=False, recording_available=False)
    policy.handle_key(ord("p"))
    assert not policy.recording
    policy.toggle_recording()
    assert not policy.recording


def test_keyboard_gripper_keys_close_and_open_selected_gripper() -> None:
    policy = KeyboardTeleopPolicy()
    policy.handle_key(ord("1"))
    policy.handle_key(ord("["))
    closing = policy.act({}, "")
    assert closing[6] == pytest.approx(0.9)
    assert closing[13] == pytest.approx(1.0)
    policy.handle_key(ord("]"))
    opening = policy.act({}, "")
    assert opening[6] == pytest.approx(1.0)
    assert opening[13] == pytest.approx(1.0)


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

    try:
        dataset = LeRobotDataset(
            "local/a3-test", root=root, download_videos=False, return_uint8=True
        )
    except TypeError:
        dataset = LeRobotDataset("local/a3-test", root=root, download_videos=False)
    assert len(dataset) == 2
    assert dataset.num_episodes == 1
    assert tuple(dataset[0]["action"].shape) == (16,)
    assert tuple(dataset[0]["observation.force"].shape) == (18,)
    assert dataset[0]["task"] == "dataset test"
