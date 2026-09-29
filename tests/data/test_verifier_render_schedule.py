from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from a3_dual_arm_sim.cli.data import verifier as command
from a3_dual_arm_sim.data.verifier_collection import run_failure_attempt
from a3_dual_arm_sim.data.verifier_recording import TemporalVerificationRecorder
from a3_dual_arm_sim.evaluation.skill_truth import SkillTruth


class RenderAwareEnv:
    """Tiny deterministic physics stand-in; no MuJoCo or GPU is instantiated."""

    def __init__(self, *, fail_step=None):
        self.render_cameras = True
        self.fail_step = fail_step
        self.steps = 0
        self.position = 0.0
        self.height = 1.0
        self.last_applied_action = np.zeros(16)
        self.last_applied_action[7] = 0.3
        self.last_applied_action[15] = 0.8
        self.render_history = []
        self.actions = []
        self.states = []

    def observation(self):
        value = 40 + self.steps * 3 if self.render_cameras else 0
        return {
            "physical_state": np.asarray([self.position, self.height]),
            **{
                f"observation.images.{camera}": np.full((4, 4, 3), value, dtype=np.uint8)
                for camera in TemporalVerificationRecorder.CAMERAS
            },
        }

    def privileged_cookie_position(self, index):
        return np.asarray([self.position + index * 0.01, 0, self.height])

    def step(self, action):
        self.steps += 1
        self.render_history.append(self.render_cameras)
        if self.steps == self.fail_step:
            raise RuntimeError("fake_physics_error")
        self.actions.append(np.asarray(action).copy())
        self.position += float(action[0])
        if action[6] == 1:
            self.height -= 0.02
        self.last_applied_action[7] = (float(action[6]) + 1) / 2
        self.last_applied_action[15] = (float(action[13]) + 1) / 2
        self.states.append((self.position, self.height))
        return self.observation(), 0, False, False, {"cookies_in_target": 0, "success": False}


class StateOnlyExpert:
    def __init__(self, *, phase="DESCEND"):
        self.expert = SimpleNamespace(phase=SimpleNamespace(name=phase))
        self.starts = []

    def start(self, request):
        self.starts.append(request)

    def act(self, observation):
        action = np.zeros(14)
        action[0] = 0.1 + float(observation["physical_state"][0]) / 100
        action[6], action[13] = -0.4, 0.6
        return action

    def ready_to_switch(self):
        return True


class StateTruthEvaluator:
    def __init__(self, env):
        self.env = env
        self.latest = None
        self.cookie_indices = tuple(range(5))
        self.truths = []

    def reset(self, request):
        self.start_position = self.env.position
        self.start_step = self.env.steps

    def update(self):
        moved = self.env.position - self.start_position
        complete = moved >= 0.35
        diagnosis = (
            None
            if complete
            else (
                "Stuck" if moved == 0 and self.env.steps - self.start_step >= 5 else "StillTrying"
            )
        )
        self.latest = SkillTruth(
            complete,
            "actual_complete" if complete else "actual_incomplete",
            self.cookie_indices,
            diagnosis=diagnosis,
        )
        self.truths.append(asdict(self.latest))
        return self.latest


def collect(root, *, optimized=None, case="natural", skill_hold_steps=0, terminal_hold_steps=0):
    env = RenderAwareEnv()
    executor = StateOnlyExpert(phase="CLOSE" if case == "miss_grasp" else "DESCEND")
    evaluator = StateTruthEvaluator(env)
    recorder = TemporalVerificationRecorder(root, max_frames=3, frame_stride=3)
    kwargs = {} if optimized is None else {"render_recorded_frames_only": optimized}
    try:
        result = run_failure_attempt(
            env,
            executor,
            evaluator,
            recorder,
            observation=env.observation(),
            case=case,
            seed=7,
            max_steps=70,
            record_every_steps=3,
            post_fault_steps=12,
            skill_hold_steps=skill_hold_steps,
            terminal_hold_steps=terminal_hold_steps,
            **kwargs,
        )
    finally:
        recorder.close()
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    return env, executor, evaluator, result, samples


@pytest.mark.parametrize("case", ["natural", "miss_grasp", "drop", "stuck"])
def test_strided_rendering_preserves_physics_actions_truth_and_saved_images(tmp_path, case):
    full = collect(tmp_path / "full", optimized=False, case=case)
    sparse = collect(tmp_path / "sparse", optimized=True, case=case)
    full_env, full_expert, full_truth, full_result, full_samples = full
    sparse_env, sparse_expert, sparse_truth, sparse_result, sparse_samples = sparse
    np.testing.assert_array_equal(full_env.actions, sparse_env.actions)
    np.testing.assert_array_equal(full_env.states, sparse_env.states)
    assert full_truth.truths == sparse_truth.truths
    assert full_expert.starts == sparse_expert.starts
    assert full_samples == sparse_samples
    ignored = {"wall_seconds", "render_recorded_frames_only"}
    assert {k: v for k, v in full_result.items() if k not in ignored} == {
        k: v for k, v in sparse_result.items() if k not in ignored
    }
    assert all(full_env.render_history)
    assert any(not enabled for enabled in sparse_env.render_history)
    assert sparse_env.render_cameras is True
    for sample in sparse_samples:
        for frame in sample["input"]["frames"]:
            assert frame["step"] % 3 == 0
            for path in frame["images"].values():
                with Image.open(tmp_path / "sparse" / path) as image:
                    assert np.asarray(image).min() > 0
                assert (tmp_path / "full" / path).read_bytes() == (
                    tmp_path / "sparse" / path
                ).read_bytes()


def test_skill_start_and_terminal_hold_windows_still_contain_real_rgb(tmp_path):
    env, executor, _truth, result, samples = collect(
        tmp_path / "held",
        optimized=True,
        skill_hold_steps=4,
        terminal_hold_steps=4,
    )
    assert len(executor.starts) == result["completed_skills"] == 4
    assert result["skill_hold_completed_count"] == 3
    assert result["terminal_hold_completed"]
    assert result["actual_terminal_hold_steps"] >= 4
    assert samples[-1]["input"]["frames"][-1]["step"] == result["steps"]
    assert {sample["batch_index"] for sample in samples} == {0, 1}
    for sample in samples:
        assert sample["input"]["frames"][-1]["step"] % 3 == 0
        for frame in sample["input"]["frames"]:
            for path in frame["images"].values():
                with Image.open(tmp_path / "held" / path) as image:
                    assert np.asarray(image).min() > 0
    assert env.render_cameras is True


def test_default_behavior_renders_every_control_step(tmp_path):
    env, _executor, _truth, result, _samples = collect(tmp_path / "default")
    assert result["render_recorded_frames_only"] is False
    assert env.render_history == [True] * result["steps"]


def test_render_flag_restored_after_normal_return_when_previously_disabled(tmp_path):
    env = RenderAwareEnv()
    observation = env.observation()
    env.render_cameras = False
    recorder = TemporalVerificationRecorder(tmp_path / "restore", max_frames=2, frame_stride=3)
    try:
        result = run_failure_attempt(
            env,
            StateOnlyExpert(),
            StateTruthEvaluator(env),
            recorder,
            observation=observation,
            case="natural",
            seed=0,
            max_steps=3,
            render_recorded_frames_only=True,
        )
    finally:
        recorder.close()
    assert result["steps"] == 3
    assert env.render_history == [False, False, True]
    assert env.render_cameras is False


@pytest.mark.parametrize("original_flag", [False, True])
def test_render_flag_restored_after_physics_exception(tmp_path, original_flag):
    env = RenderAwareEnv(fail_step=2)
    observation = env.observation()  # Real reset RGB before changing the original flag.
    env.render_cameras = original_flag
    recorder = TemporalVerificationRecorder(tmp_path / "error", max_frames=2, frame_stride=3)
    try:
        with pytest.raises(RuntimeError, match="fake_physics_error"):
            run_failure_attempt(
                env,
                StateOnlyExpert(),
                StateTruthEvaluator(env),
                recorder,
                observation=observation,
                case="natural",
                seed=0,
                max_steps=5,
                render_recorded_frames_only=True,
            )
    finally:
        recorder.close()
    assert env.render_cameras is original_flag
    assert not (tmp_path / "error" / "samples.jsonl").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_render_schedule_is_opt_in(monkeypatch, tmp_path, enabled):
    received = []
    monkeypatch.setattr(
        command, "collect_verifier_dataset", lambda args: received.append(args) or 0
    )
    argv = ["--root", str(tmp_path / "dataset")]
    if enabled:
        argv.append("--render-recorded-frames-only")
    assert command.main(argv) == 0
    assert received[0].render_recorded_frames_only is enabled
