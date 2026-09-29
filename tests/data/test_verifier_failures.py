from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest
from a3_dual_arm_sim.data.verifier_collection import (
    SyntheticFailureInjection,
    hold_cartesian_action,
    run_failure_attempt,
)
from a3_dual_arm_sim.evaluation.skill_truth import SkillTruth
from a3_dual_arm_sim.experts.single_cookie import CookiePhase


def request(skill=CookieSkill.PICK_FIVE):
    return SkillRequest(skill, 0, 1, 1)


def joint_action():
    value = np.zeros(16)
    value[7], value[15] = 0.3, 0.8
    return value


def test_hold_preserves_grippers_and_changes_only_explicit_opening():
    action = hold_cartesian_action(joint_action())
    assert np.all(action[:6] == 0)
    assert np.all(action[7:13] == 0)
    assert action[6] == pytest.approx(-0.4)
    assert action[13] == pytest.approx(0.6)
    assert hold_cartesian_action(joint_action(), left_opening=1)[6] == 1


def test_miss_grasp_only_injects_after_close_and_preserves_motion():
    injector = SyntheticFailureInjection("miss_grasp")
    source = np.full(14, 0.2)
    unchanged, event = injector.apply(source, request=request(), expert_phase="APPROACH",
                                      joint_action=joint_action(), step=1)
    np.testing.assert_equal(unchanged, source)
    assert event is None and not injector.active
    action, event = injector.apply(source, request=request(), expert_phase="CLOSE",
                                   joint_action=joint_action(), step=5)
    assert action[6] == 1
    np.testing.assert_equal(action[:6], source[:6])
    assert source[6] == 0.2
    assert event["synthetic"] and injector.trigger_step == 5
    assert not injector.bypass_expert


def test_stuck_freezes_command_after_descent_without_forcing_labels():
    injector = SyntheticFailureInjection("stuck")
    action, event = injector.apply(np.ones(14), request=request(), expert_phase="DESCEND",
                                   joint_action=joint_action(), step=7)
    np.testing.assert_equal(action, hold_cartesian_action(joint_action()))
    assert injector.bypass_expert and event["case"] == "stuck"
    assert "label" not in event and "diagnosis" not in event


def test_drop_does_not_trigger_until_place_skill():
    injector = SyntheticFailureInjection("drop")
    _, event = injector.apply(np.zeros(14), request=request(), expert_phase="LIFT",
                              joint_action=joint_action(), step=1)
    assert event is None and not injector.active
    action, event = injector.apply(np.ones(14), request=request(CookieSkill.PLACE_FIVE),
                                   expert_phase="MOVE_TO_SLOT", joint_action=joint_action(), step=2)
    assert action[6] == 1 and np.all(action[:6] == 0)
    assert event["step"] == 2 and injector.bypass_expert


def test_natural_has_no_injection_even_at_failure_phases():
    injector = SyntheticFailureInjection("natural")
    source = np.zeros(14)
    action, event = injector.apply(source, request=request(), expert_phase="CLOSE",
                                   joint_action=joint_action(), step=1)
    assert not injector.synthetic and not injector.active and event is None
    np.testing.assert_equal(action, source)


class FakeEnv:
    def __init__(self):
        self.last_applied_action = joint_action()
        self.steps = 0
        self.height = 1.0

    def privileged_cookie_position(self, index):
        return np.asarray([index * 0.01, 0, self.height])

    def step(self, action):
        self.steps += 1
        if action[6] == 1:
            self.height -= 0.02
        return {}, 0, False, False, {"cookies_in_target": 0, "success": False}


class FakeExecutor:
    def __init__(self, *, phase="CLOSE", ready=False, fail=False):
        self.expert = SimpleNamespace(phase=CookiePhase[phase])
        self.ready = ready
        self.fail = fail
        self.starts = []

    def start(self, request):
        self.starts.append(request)

    def act(self, observation):
        if self.fail:
            raise RuntimeError("natural_contact_failure")
        return np.zeros(14)

    def ready_to_switch(self):
        return self.ready


class FakeEvaluator:
    def __init__(self, *, pick_complete=False, diagnosis="StillTrying"):
        self.pick_complete = pick_complete
        self.diagnosis = diagnosis
        self.cookie_indices = tuple(range(5))
        self.latest = None

    def reset(self, request):
        self.request = request

    def update(self):
        complete = self.pick_complete and self.request.skill is CookieSkill.PICK_FIVE
        self.latest = SkillTruth(complete, "actual_true" if complete else "actual_incomplete",
                                 self.cookie_indices, diagnosis=None if complete else self.diagnosis)
        return self.latest


class FakeRecorder:
    def __init__(self):
        self.truths = []
        self.starts = []
        self.frames = []

    def start(self, request, *, seed, attempt):
        self.starts.append(request)

    def observe(self, observation, *, step):
        self.frames.append((observation, step))

    def record(self, truth):
        self.truths.append(truth)
        return len(self.truths) - 1


def test_truth_success_not_overridden_by_failure_case_name():
    env, executor, evaluator, recorder = FakeEnv(), FakeExecutor(), FakeEvaluator(pick_complete=True), FakeRecorder()
    result = run_failure_attempt(env, executor, evaluator, recorder, observation={}, case="miss_grasp",
                                seed=0, max_steps=4, record_every_steps=1, post_fault_steps=3)
    assert result["fault_triggered"]
    assert result["sample_counts"]["Yes"] == 4
    assert "No" not in result["sample_counts"]
    assert all(truth.complete for truth in recorder.truths)
    assert result["actual_truth_reasons"] == {"actual_true": 4}


def test_stuck_case_cannot_force_truth_stuck_diagnosis():
    env, executor = FakeEnv(), FakeExecutor(phase="DESCEND")
    evaluator, recorder = FakeEvaluator(diagnosis="StillTrying"), FakeRecorder()
    result = run_failure_attempt(env, executor, evaluator, recorder, observation={}, case="stuck",
                                seed=0, max_steps=4, record_every_steps=1, post_fault_steps=3)
    assert result["sample_counts"] == {"No": 4, "StillTrying": 4}
    assert "Stuck" not in result["sample_counts"]


def test_drop_switch_requires_actual_pick_completion_and_audits_physical_height():
    env, executor = FakeEnv(), FakeExecutor(phase="MOVE_TO_SLOT", ready=True)
    evaluator, recorder = FakeEvaluator(pick_complete=True), FakeRecorder()
    audit = []
    result = run_failure_attempt(env, executor, evaluator, recorder, observation={}, case="drop",
                                seed=4, max_steps=4, record_every_steps=1, post_fault_steps=2,
                                audit_sink=audit.append)
    assert result["completed_skills"] == 1
    assert result["fault_triggered"] and result["fault_trigger_step"] == 2
    assert result["observed_drop_height_loss_over_30mm"]
    assert recorder.starts[1].skill is CookieSkill.PLACE_FIVE
    assert any("cookie_positions_at_injection_m" in event for event in audit)
    assert all("case" not in observation for observation, _ in recorder.frames)


def test_drop_not_claimed_when_pick_never_completes():
    result = run_failure_attempt(FakeEnv(), FakeExecutor(phase="MOVE_TO_SLOT", ready=True),
                                FakeEvaluator(), FakeRecorder(), observation={}, case="drop",
                                seed=0, max_steps=3, record_every_steps=1)
    assert not result["fault_triggered"]
    assert not result["observed_drop_height_loss_over_30mm"]
    assert result["stop_reason"] == "step_limit"


def test_natural_expert_failure_is_not_marked_synthetic():
    result = run_failure_attempt(FakeEnv(), FakeExecutor(fail=True), FakeEvaluator(), FakeRecorder(),
                                observation={}, case="natural", seed=0, max_steps=3,
                                record_every_steps=1, post_fault_steps=2)
    assert not result["synthetic"] and not result["fault_triggered"]
    assert result["expert_error"] == "natural_contact_failure"
    assert result["stop_reason"] == "post_expert_error_window_finished"


@pytest.mark.parametrize("case, phase", [
    ("miss_grasp", CookiePhase.CLOSE), ("stuck", CookiePhase.DESCEND),
])
def test_attempt_triggers_with_real_integer_valued_cookie_phase(case, phase):
    executor = FakeExecutor(phase=phase.name)
    assert isinstance(executor.expert.phase.value, int)
    audit = []
    result = run_failure_attempt(
        FakeEnv(), executor, FakeEvaluator(), FakeRecorder(), observation={}, case=case,
        seed=0, max_steps=2, record_every_steps=1, post_fault_steps=1,
        audit_sink=audit.append,
    )
    assert result["fault_triggered"] and result["fault_trigger_step"] == 1
    injected = [event for event in audit if event["event"] == "synthetic_control_injection"]
    assert injected[0]["phase"] == phase.name


class CompletingEvaluator(FakeEvaluator):
    def __init__(self, env, *, fail_after_step=None):
        super().__init__()
        self.env = env
        self.fail_after_step = fail_after_step

    def update(self):
        complete = self.fail_after_step is None or self.env.steps <= self.fail_after_step
        self.latest = SkillTruth(
            complete, "actual_true" if complete else "cookies_moved_after_success",
            self.cookie_indices, diagnosis=None if complete else "Stuck",
        )
        return self.latest


class TailEnv(FakeEnv):
    def __init__(self, *, stop_at=None, truncated=False):
        super().__init__()
        self.stop_at = stop_at
        self.truncated = truncated
        self.actions = []

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        stopped = self.stop_at is not None and self.steps >= self.stop_at
        return {}, 0, stopped and not self.truncated, stopped and self.truncated, {
            "cookies_in_target": 10, "success": True,
            "safety_reason": "test_safety_stop" if stopped else None,
        }


def test_terminal_hold_records_full_tail_without_recounting_or_advancing_expert():
    env = TailEnv()
    executor = FakeExecutor(ready=True)
    evaluator, recorder, audit = CompletingEvaluator(env), FakeRecorder(), []
    result = run_failure_attempt(
        env, executor, evaluator, recorder, observation={}, case="natural", seed=3,
        max_steps=20, record_every_steps=1, terminal_hold_steps=5, audit_sink=audit.append,
    )
    assert result["completed_skills"] == 4 and len(executor.starts) == 4
    assert result["terminal_hold_started_step"] == 4
    assert result["steps"] == 9 and result["actual_terminal_hold_steps"] == 5
    assert result["terminal_hold_completed"] is True
    assert result["terminal_hold_final_truth_complete"] is True
    assert result["stop_reason"] == "terminal_hold_window_finished"
    assert result["terminal_hold_label_counts"] == {"Yes": 5}
    assert recorder.starts[-1] == SkillRequest(CookieSkill.PLACE_FIVE, 1, 1, 2)
    assert len([event for event in audit if event["event"] == "terminal_hold_started"]) == 1
    for action in env.actions[4:]:
        np.testing.assert_equal(action, hold_cartesian_action(env.last_applied_action))


def test_terminal_hold_zero_retains_original_finish_boundary():
    env = TailEnv()
    result = run_failure_attempt(
        env, FakeExecutor(ready=True), CompletingEvaluator(env), FakeRecorder(),
        observation={}, case="natural", seed=0, max_steps=10, record_every_steps=1,
    )
    assert result["steps"] == 4 and result["completed_skills"] == 4
    assert result["stop_reason"] == "all_skills_completed"
    assert result["actual_terminal_hold_steps"] == 0
    assert result["terminal_hold_started_step"] is None
    assert result["terminal_hold_completed"] is False


def test_terminal_hold_remains_bounded_by_total_step_budget():
    env = TailEnv()
    result = run_failure_attempt(
        env, FakeExecutor(ready=True), CompletingEvaluator(env), FakeRecorder(),
        observation={}, case="natural", seed=0, max_steps=6, record_every_steps=1,
        terminal_hold_steps=5,
    )
    assert result["completed_skills"] == 4 and result["steps"] == 6
    assert result["actual_terminal_hold_steps"] == 2
    assert result["terminal_hold_completed"] is False
    assert result["stop_reason"] == "step_limit"


@pytest.mark.parametrize("truncated", [False, True])
def test_terminal_hold_obeys_environment_termination_before_requested_tail(truncated):
    env = TailEnv(stop_at=6, truncated=truncated)
    result = run_failure_attempt(
        env, FakeExecutor(ready=True), CompletingEvaluator(env), FakeRecorder(),
        observation={}, case="natural", seed=0, max_steps=20, record_every_steps=1,
        terminal_hold_steps=5,
    )
    assert result["completed_skills"] == 4 and result["steps"] == 6
    assert result["actual_terminal_hold_steps"] == 2
    assert result["terminal_hold_completed"] is False
    assert result["stop_reason"] == "test_safety_stop"


def test_terminal_hold_success_can_become_failure_without_forced_positive_labels():
    env = TailEnv()
    recorder = FakeRecorder()
    result = run_failure_attempt(
        env, FakeExecutor(ready=True), CompletingEvaluator(env, fail_after_step=5), recorder,
        observation={}, case="natural", seed=0, max_steps=20, record_every_steps=20,
        terminal_hold_steps=5,
    )
    assert result["completed_skills"] == 4
    assert result["actual_terminal_hold_steps"] == 5
    assert result["terminal_hold_completed"] is True
    assert result["terminal_hold_final_truth_complete"] is False
    assert result["terminal_hold_label_counts"] == {"Yes": 1, "No": 4, "Stuck": 4}
    assert all(not truth.complete for truth in recorder.truths[-4:])
    assert recorder.truths[-1].reason == "cookies_moved_after_success"


def test_terminal_hold_ends_on_a_real_camera_stride_boundary():
    env = TailEnv()
    recorder = FakeRecorder()
    recorder.frame_stride = 3
    result = run_failure_attempt(
        env, FakeExecutor(ready=True), CompletingEvaluator(env), recorder,
        observation={}, case="natural", seed=0, max_steps=30, record_every_steps=3,
        terminal_hold_steps=4,
    )
    # Each skill completes on the next sampled frame: 3, 6, 9, 12.
    assert result["terminal_hold_started_step"] == 12
    assert result["actual_terminal_hold_steps"] == 6
    assert result["steps"] == 18
    assert result["terminal_hold_completed"] is True


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_terminal_hold_rejects_negative_or_noninteger_limits(steps):
    with pytest.raises(ValueError, match="terminal_hold_steps"):
        run_failure_attempt(
            FakeEnv(), FakeExecutor(), FakeEvaluator(), FakeRecorder(),
            observation={}, case="natural", seed=0, terminal_hold_steps=steps,
        )
