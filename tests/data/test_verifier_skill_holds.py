"""Pure command/label boundary checks; no renderer or physics rollout is needed."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from a3_dual_arm_sim.core.skills import CookieSkill
from a3_dual_arm_sim.data import verifier_collection as collection
from a3_dual_arm_sim.evaluation.skill_truth import SkillTruth


class HoldEnv:
    def __init__(self, *, stop_at=None, truncated=False):
        self.last_applied_action = np.zeros(16)
        self.last_applied_action[[7, 15]] = [0.3, 0.8]
        self.steps = 0
        self.stop_at = stop_at
        self.truncated = truncated
        self.actions = []

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        stopped = self.stop_at is not None and self.steps >= self.stop_at
        return (
            {},
            0,
            stopped and not self.truncated,
            stopped and self.truncated,
            {
                "cookies_in_target": 10,
                "success": True,
                "safety_reason": "test_safety_stop" if stopped else None,
            },
        )


class HoldExecutor:
    def __init__(self, *, ready=True):
        self.expert = SimpleNamespace(phase=SimpleNamespace(name="MOVE_TO_SLOT"))
        self.ready = ready
        self.starts = []
        self.act_calls = 0

    def start(self, request):
        self.starts.append(request)

    def act(self, _observation):
        self.act_calls += 1
        return np.full(14, 0.05)

    def ready_to_switch(self):
        return self.ready


class HoldEvaluator:
    def __init__(self, env, *, lose_at=None):
        self.env = env
        self.lose_at = lose_at
        self.latest = None
        self.cookie_indices = tuple(range(5))

    def reset(self, request):
        self.request = request

    def update(self):
        complete = self.lose_at is None or self.env.steps < self.lose_at
        self.latest = SkillTruth(
            complete,
            "five_held" if complete else "grasp_lost",
            self.cookie_indices,
            diagnosis=None if complete else "Stuck",
        )
        return self.latest


class HoldRecorder:
    def __init__(self, *, frame_stride=1):
        self.frame_stride = frame_stride
        self.starts = []
        self.rows = []

    def start(self, request, *, seed, attempt):
        self.request = request
        self.starts.append(request)

    def observe(self, _observation, *, step):
        self.step = step

    def record(self, truth):
        self.rows.append((self.request, self.step, truth))
        return len(self.rows) - 1


def test_skill_hold_extends_three_nonfinal_skills_without_recounting_terminal():
    env, executor, recorder, audit = HoldEnv(), HoldExecutor(), HoldRecorder(), []
    result = collection.run_failure_attempt(
        env,
        executor,
        HoldEvaluator(env),
        recorder,
        observation={},
        case="natural",
        seed=1,
        max_steps=30,
        record_every_steps=20,
        skill_hold_steps=3,
        terminal_hold_steps=2,
        audit_sink=audit.append,
    )
    assert result["completed_skills"] == 4 and len(executor.starts) == 4
    assert result["steps"] == 15 and executor.act_calls == 4
    assert result["requested_skill_hold_steps"] == 3
    assert result["actual_skill_hold_steps"] == 9
    assert result["skill_hold_completed_count"] == 3
    assert [item["request"]["skill"] for item in result["skill_hold_reports"]] == [
        CookieSkill.PICK_FIVE,
        CookieSkill.PLACE_FIVE,
        CookieSkill.PICK_FIVE,
    ]
    assert result["skill_hold_label_counts"] == {"Yes": 9}
    assert result["actual_terminal_hold_steps"] == 2
    assert result["terminal_hold_completed"] is True
    assert sum(event["event"] == "skill_hold_started" for event in audit) == 3
    assert sum(event["event"] == "terminal_hold_started" for event in audit) == 1
    for action in env.actions[1:4]:
        np.testing.assert_array_equal(
            action, collection.hold_cartesian_action(env.last_applied_action)
        )


def test_skill_hold_lost_completion_stops_without_switching_or_positive_override():
    env, executor, recorder, audit = HoldEnv(), HoldExecutor(), HoldRecorder(), []
    result = collection.run_failure_attempt(
        env,
        executor,
        HoldEvaluator(env, lose_at=3),
        recorder,
        observation={},
        case="natural",
        seed=0,
        max_steps=20,
        record_every_steps=20,
        skill_hold_steps=5,
        audit_sink=audit.append,
    )
    assert result["stop_reason"] == "skill_hold_completion_lost"
    assert result["steps"] == 3 and result["completed_skills"] == 0
    assert len(executor.starts) == 1 and executor.act_calls == 1
    assert result["skill_hold_completed_count"] == 0
    assert result["skill_hold_label_counts"] == {"Yes": 1, "No": 1, "Stuck": 1}
    assert result["skill_hold_reports"][0]["final_truth_complete"] is False
    assert not recorder.rows[-1][2].complete
    assert any(event["event"] == "skill_hold_completion_lost" for event in audit)


def test_skill_hold_waits_for_executor_readiness_before_starting():
    env, executor, audit = HoldEnv(), HoldExecutor(ready=False), []
    result = collection.run_failure_attempt(
        env,
        executor,
        HoldEvaluator(env),
        HoldRecorder(),
        observation={},
        case="natural",
        seed=0,
        max_steps=6,
        record_every_steps=1,
        skill_hold_steps=3,
        audit_sink=audit.append,
    )
    assert result["completed_skills"] == 0 and executor.act_calls == 6
    assert result["actual_skill_hold_steps"] == 0
    assert not any(event["event"] == "skill_hold_started" for event in audit)


def test_skill_hold_zero_preserves_original_switch_boundary():
    env = HoldEnv()
    result = collection.run_failure_attempt(
        env,
        HoldExecutor(),
        HoldEvaluator(env),
        HoldRecorder(),
        observation={},
        case="natural",
        seed=0,
        max_steps=10,
        record_every_steps=1,
    )
    assert result["steps"] == 4 and result["completed_skills"] == 4
    assert result["requested_skill_hold_steps"] == 0
    assert result["actual_skill_hold_steps"] == 0 and not result["skill_hold_reports"]


def test_skill_hold_switches_only_on_camera_stride_boundaries():
    env, recorder = HoldEnv(), HoldRecorder(frame_stride=3)
    result = collection.run_failure_attempt(
        env,
        HoldExecutor(),
        HoldEvaluator(env),
        recorder,
        observation={},
        case="natural",
        seed=0,
        max_steps=40,
        record_every_steps=3,
        skill_hold_steps=4,
        terminal_hold_steps=4,
    )
    assert result["steps"] == 36 and result["completed_skills"] == 4
    assert result["actual_skill_hold_steps"] == 18
    assert result["actual_terminal_hold_steps"] == 6
    assert all(item["actual_hold_steps"] == 6 for item in result["skill_hold_reports"])
    assert all(step % 3 == 0 for _, step, _ in recorder.rows)


def test_skill_hold_is_bounded_by_total_step_limit():
    env = HoldEnv()
    result = collection.run_failure_attempt(
        env,
        HoldExecutor(),
        HoldEvaluator(env),
        HoldRecorder(),
        observation={},
        case="natural",
        seed=0,
        max_steps=3,
        record_every_steps=1,
        skill_hold_steps=5,
    )
    assert result["stop_reason"] == "step_limit" and result["completed_skills"] == 0
    assert result["actual_skill_hold_steps"] == 2
    assert result["skill_hold_reports"][0]["completed"] is False


def test_skill_hold_loss_audit_does_not_attach_truth_to_an_older_camera_frame():
    env, recorder, audit = HoldEnv(), HoldRecorder(frame_stride=3), []
    result = collection.run_failure_attempt(
        env,
        HoldExecutor(),
        HoldEvaluator(env, lose_at=5),
        recorder,
        observation={},
        case="natural",
        seed=0,
        max_steps=20,
        record_every_steps=3,
        skill_hold_steps=6,
        audit_sink=audit.append,
    )
    assert result["steps"] == 5 and result["completed_skills"] == 0
    assert result["stop_reason"] == "skill_hold_completion_lost"
    assert all(step % 3 == 0 for _, step, _ in recorder.rows)
    assert recorder.rows[-1][1] == 3 and recorder.rows[-1][2].complete
    loss = next(event for event in audit if event["event"] == "skill_hold_completion_lost")
    assert loss["step"] == 5 and loss["truth"]["complete"] is False


@pytest.mark.parametrize("truncated", [False, True])
def test_skill_hold_obeys_environment_termination(truncated):
    env = HoldEnv(stop_at=3, truncated=truncated)
    result = collection.run_failure_attempt(
        env,
        HoldExecutor(),
        HoldEvaluator(env),
        HoldRecorder(),
        observation={},
        case="natural",
        seed=0,
        max_steps=20,
        record_every_steps=1,
        skill_hold_steps=5,
    )
    assert result["stop_reason"] == "test_safety_stop" and result["completed_skills"] == 0
    assert result["skill_hold_reports"][0]["completed"] is False


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_skill_hold_rejects_invalid_limits(steps):
    env = HoldEnv()
    with pytest.raises(ValueError, match="skill_hold_steps"):
        collection.run_failure_attempt(
            env,
            HoldExecutor(),
            HoldEvaluator(env),
            HoldRecorder(),
            observation={},
            case="natural",
            seed=0,
            skill_hold_steps=steps,
        )


def test_skill_hold_cli_defaults_to_zero_and_forwards_explicit_value(monkeypatch, tmp_path):
    from a3_dual_arm_sim.cli.data import verifier as cli

    calls = []
    monkeypatch.setattr(cli, "collect_verifier_dataset", lambda args: calls.append(args) or 0)
    assert cli.main(["--root", str(tmp_path / "default")]) == 0
    assert calls[-1].skill_hold_steps == 0
    assert cli.main(["--root", str(tmp_path / "held"), "--skill-hold-steps", "80"]) == 0
    assert calls[-1].skill_hold_steps == 80
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--root", str(tmp_path / "negative"), "--skill-hold-steps", "-1"])
    assert stopped.value.code == 2


def test_collection_initialization_failure_records_no_labels_and_continues(monkeypatch, tmp_path):
    from a3_dual_arm_sim.agents import executors
    from a3_dual_arm_sim.data import verifier_recording
    from a3_dual_arm_sim.envs import cookie_transfer
    from a3_dual_arm_sim.evaluation import benchmark

    @dataclass
    class Config:
        control_hz: int = 20

    resets, constructors, rollouts, closed = [], [], [], []

    class Env:
        def __init__(self, *_args, **_kwargs):
            pass

        def reset(self, *, seed, options):
            resets.append(seed)
            return {}, {}

        def close(self):
            closed.append("env")

    class Recorder:
        def __init__(self, _root, **_kwargs):
            pass

        def close(self):
            closed.append("recorder")

    class Executor:
        def __init__(self, _env, *, seed, source_column_number):
            constructors.append((seed, source_column_number))
            if len(constructors) == 1:
                raise RuntimeError("target column 1 unreachable with vertical grasp: preflight")

        def close(self):
            closed.append("executor")

    def rollout(_env, _executor, _evaluator, _recorder, **kwargs):
        rollouts.append((kwargs["seed"], kwargs["case"], kwargs["skill_hold_steps"]))
        return {"seed": kwargs["seed"], "case": kwargs["case"], "sample_counts": {}}

    monkeypatch.setattr(
        benchmark,
        "CookieBatchBenchmark",
        lambda *_args, **_kwargs: SimpleNamespace(config=Config()),
    )
    monkeypatch.setattr(cookie_transfer, "A3CookieTransferEnv", Env)
    monkeypatch.setattr(verifier_recording, "TemporalVerificationRecorder", Recorder)
    monkeypatch.setattr(executors, "ExpertSkillExecutor", Executor)
    monkeypatch.setattr(collection, "run_failure_attempt", rollout)
    root = tmp_path / "new"
    args = argparse.Namespace(
        root=root,
        config=tmp_path / "config.yaml",
        max_steps=100,
        post_fault_steps=80,
        record_every=20,
        frame_stride=10,
        window_frames=5,
        terminal_hold_steps=80,
        skill_hold_steps=40,
        seeds=[10, 11],
        seed=0,
        cases=["natural", "stuck"],
        source_column=2,
        profile="diverse",
    )
    assert collection.collect_verifier_dataset(args) == 0
    assert resets == [10, 10, 11, 11]
    assert rollouts == [(10, "stuck", 40), (11, "natural", 40), (11, "stuck", 40)]
    attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 4 and attempts[0]["initialization_failed"] is True
    assert attempts[0]["sample_counts"] == {} and attempts[0]["steps"] == 0
    assert attempts[0]["visual_labels_recorded"] is False
    assert not (root / "samples.jsonl").exists()
    audit = [json.loads(line) for line in (root / "fault_audit.jsonl").read_text().splitlines()]
    assert [event["event"] for event in audit] == ["expert_initialization_failed"]
    manifest = json.loads((root / "collection_manifest.json").read_text())
    assert manifest["skill_hold_steps"] == 40
    assert closed.count("executor") == 3 and "env" in closed and "recorder" in closed
