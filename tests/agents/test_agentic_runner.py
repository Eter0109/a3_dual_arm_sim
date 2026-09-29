from types import SimpleNamespace

import numpy as np
import pytest

from a3_dual_arm_sim.agents.controller import (
    AgenticController,
    AgentLoopConfig,
    TemporalFrameBuffer,
)
from a3_dual_arm_sim.agents.executors import ExpertSkillExecutor, recovery_action
from a3_dual_arm_sim.agents.planning import FixedSkillPlanner
from a3_dual_arm_sim.agents.verification import VerificationDecision
from a3_dual_arm_sim.core.contracts import FRONT_IMAGE, LEFT_WRIST_IMAGE
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest
from a3_dual_arm_sim.experts.single_cookie import CookiePhase


def observation():
    return {FRONT_IMAGE: np.zeros((8, 8, 3), dtype=np.uint8),
            LEFT_WRIST_IMAGE: np.ones((8, 8, 3), dtype=np.uint8)}


class Env:
    last_applied_action = np.r_[np.zeros(7), .25, np.zeros(7), .8]

    def __init__(self, success=True, terminate=False):
        self.success = success
        self.terminate = terminate
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return observation(), 0, self.terminate, False, {
            "success": self.success, "cookies_in_target": 10 if self.success else 0,
            "cookies_in_source": 70,
        }


class Executor:
    def __init__(self):
        self.starts = []

    def start(self, request):
        self.starts.append(request)

    def act(self, obs):
        return Env.last_applied_action.copy()


class Verifier:
    def __init__(self, status):
        self.status = status
        self.seen = []

    def check(self, request, frames):
        self.seen.append(frames)
        assert all(set(frame) == {FRONT_IMAGE, LEFT_WRIST_IMAGE} for frame in frames)
        return VerificationDecision(self.status, "mock")


def run(verifier, env=None, **kwargs):
    executor = Executor()
    result = AgenticController(FixedSkillPlanner(), verifier, config=AgentLoopConfig(
        verify_every_steps=2, frame_interval=1, skill_timeout_steps=4,
        max_steps=30, recovery_steps=2, max_retries=1,
        min_stuck_steps=2, stuck_confirmations=2,
    )).run(env or Env(), executor, task="transfer ten", observation=observation(), seed=0,
           **kwargs)
    return result, executor


def test_four_skills_completed_only_on_verifier_decisions():
    verifier = Verifier("completed")
    result, executor = run(verifier)
    assert result["success"] and result["completed_skills"] == 4
    assert len(executor.starts) == 4
    assert result["steps"] == 8


def test_oracle_truth_does_not_override_visual_no():
    truth = SimpleNamespace(complete=True, reason="actual_complete")
    evaluator = SimpleNamespace(reset=lambda request: None, update=lambda: truth)
    result, executor = run(Verifier("continuing"), truth_evaluator=evaluator)
    assert not result["success"] and result["completed_skills"] == 0
    assert len(executor.starts) == 2
    assert "retry limit" in result["failure_reason"]


def test_false_visual_completion_fails_whole_box_score():
    result, _ = run(Verifier("completed"), Env(success=False))
    assert not result["success"] and result["completed_skills"] == 4
    assert "whole-box" in result["failure_reason"]


def test_safety_termination_never_asks_verifier():
    verifier = Verifier("completed")
    result, _ = run(verifier, Env(terminate=True))
    assert not result["success"] and not verifier.seen


def test_stuck_triggers_bounded_recovery_and_same_skill_retry():
    env = Env()
    result, executor = run(Verifier("stuck"), env)
    assert not result["success"]
    assert executor.starts[0] == executor.starts[1]
    assert sum(len(action) == 14 for action in env.actions) == 2


def test_recovery_preserves_gripper_commands():
    action = recovery_action(Env.last_applied_action)
    assert action[2] > 0 and action[6] == -.5 and action[13] == pytest.approx(.6)
    assert np.all(action[7:13] == 0)


def test_frame_buffer_owned_copies_and_clears():
    obs = observation()
    buffer = TemporalFrameBuffer(interval=5)
    buffer.observe(obs, 0)
    obs[FRONT_IMAGE][:] = 255
    assert np.all(buffer.frames[0][FRONT_IMAGE] == 0)
    buffer.observe(obs, 1)
    assert len(buffer.frames) == 1
    buffer.clear()
    assert not buffer.frames


def test_config_rejects_invalid_limits():
    with pytest.raises(ValueError):
        AgentLoopConfig(max_retries=-1)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_config_rejects_invalid_minimum_stuck_age(value):
    with pytest.raises(ValueError, match="min_stuck_steps"):
        AgentLoopConfig(min_stuck_steps=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_config_rejects_invalid_stuck_confirmations(value):
    with pytest.raises(ValueError, match="stuck_confirmations"):
        AgentLoopConfig(stuck_confirmations=value)


def run_sequence(statuses, *, env=None, recorder=None, evaluator=None, **options):
    class SequenceVerifier(Verifier):
        def __init__(self):
            super().__init__("continuing")
            self.statuses = iter(statuses)

        def check(self, request, frames):
            self.status = next(self.statuses, "continuing")
            return super().check(request, frames)

    settings = {
        "verify_every_steps": 1, "frame_interval": 1, "window_frames": 2,
        "skill_timeout_steps": 8, "max_steps": 50, "recovery_steps": 1,
        "max_retries": 0, "min_stuck_steps": 0, "stuck_confirmations": 2,
    }
    settings.update(options)
    verifier = SequenceVerifier()
    executor = Executor()
    actual_env = env or Env()
    result = AgenticController(FixedSkillPlanner(), verifier, config=AgentLoopConfig(
        **settings,
    )).run(actual_env, executor, task="transfer ten", observation=observation(), seed=0,
           verification_recorder=recorder, truth_evaluator=evaluator)
    return result, executor, verifier, actual_env


def test_stuck_waits_for_minimum_age_then_two_consecutive_confirmations():
    result, _, _, _ = run_sequence(["stuck"] * 5, min_stuck_steps=4)
    assert result["steps"] == 5
    assert not result["success"]
    deferred = [event for event in result["events"] if event["event"] == "stuck_deferred"]
    assert [event["skill_step"] for event in deferred] == [1, 2, 3, 4]
    assert [event["stuck_streak"] for event in deferred] == [0, 0, 0, 1]
    # Audits retain raw model predictions, not scheduler-renamed continuing labels.
    assert all(event["status"] == "stuck" for event in result["events"]
               if event["event"] == "verify")


def test_completed_is_not_delayed_by_stuck_age_or_confirmation_count():
    result, _, _, _ = run_sequence(
        ["stuck", "completed", "completed", "completed", "completed"],
        min_stuck_steps=40, stuck_confirmations=3,
    )
    assert result["success"] and result["completed_skills"] == 4
    assert result["steps"] == 5
    assert not any(event["event"] == "recover" for event in result["events"])


def test_progress_resets_consecutive_stuck_streak():
    result, _, _, _ = run_sequence(["stuck", "continuing", "stuck", "stuck"])
    assert result["steps"] == 4
    deferred = [event for event in result["events"] if event["event"] == "stuck_deferred"]
    assert [event["step"] for event in deferred] == [1, 3]
    assert [event["stuck_streak"] for event in deferred] == [1, 1]


def test_retry_resets_stuck_age_and_confirmation_streak():
    result, executor, _, _ = run_sequence(
        ["stuck"] * 5 + ["completed"] * 4,
        min_stuck_steps=3, max_retries=1,
    )
    assert result["success"] and result["completed_skills"] == 4
    assert result["steps"] == 10
    assert executor.starts[0] == executor.starts[1]
    assert [event["step"] for event in result["events"] if event["event"] == "recover"] == [4]
    retried = [event for event in result["events"]
               if event["event"] == "stuck_deferred" and event["step"] == 6]
    assert len(retried) == 1 and retried[0]["skill_step"] == 1
    assert retried[0]["stuck_streak"] == 0


def test_legacy_immediate_stuck_behavior_is_explicitly_configurable():
    result, _, _, _ = run_sequence(["stuck"], stuck_confirmations=1)
    assert result["steps"] == 1
    assert not any(event["event"] == "stuck_deferred" for event in result["events"])


def test_old_config_without_new_fields_remains_loadable():
    old_options = {
        "verify_every_steps": 20, "frame_interval": 5, "window_frames": 2,
        "skill_timeout_steps": 600, "max_steps": 2400,
        "max_retries": 2, "recovery_steps": 15,
    }
    cfg = AgentLoopConfig(**old_options)
    assert cfg.window_frames == 2 and cfg.frame_interval == 5
    assert cfg.min_stuck_steps == 40 and cfg.stuck_confirmations == 2


def test_five_frames_span_two_simulated_seconds_at_twenty_hz():
    class StepImageEnv(Env):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            obs[FRONT_IMAGE][:] = len(self.actions)
            return obs, reward, terminated, truncated, info

    result, _, verifier, _ = run_sequence(
        ["completed"] * 4, env=StepImageEnv(), verify_every_steps=20,
        frame_interval=10, window_frames=5, min_stuck_steps=40,
        skill_timeout_steps=100, max_steps=200,
    )
    assert result["success"] and result["steps"] == 160
    image_steps = [int(frame[FRONT_IMAGE][0, 0, 0]) for frame in verifier.seen[0]]
    assert image_steps == [0, 10, 20, 30, 40]
    assert (image_steps[-1] - image_steps[0]) / 20 == 2.0


def test_verification_waits_for_current_frame_when_cadences_do_not_align():
    result, _, _, _ = run_sequence(
        ["completed"] * 4, verify_every_steps=3, frame_interval=2,
        skill_timeout_steps=8,
    )
    assert result["success"] and result["steps"] == 24
    verification_steps = [event["step"] for event in result["events"]
                          if event["event"] == "verify"]
    assert verification_steps == [6, 12, 18, 24]


def test_mismatched_recorder_window_is_rejected_before_motion():
    recorder = SimpleNamespace(max_frames=5, frame_stride=10)
    result, executor, _, env = run_sequence(["completed"], recorder=recorder)
    assert not result["success"]
    assert "recorder" in result["failure_reason"]
    assert not executor.starts and not env.actions


def test_recorder_keeps_raw_deferred_stuck_prediction_and_matching_window():
    class Recorder:
        max_frames = 2
        frame_stride = 1

        def __init__(self):
            self.predictions = []

        def start(self, request, **kwargs):
            pass

        def observe(self, obs, **kwargs):
            pass

        def record(self, truth, *, prediction):
            self.predictions.append(prediction)

    recorder = Recorder()
    # True completion cannot bypass the visual stuck/continuing decisions.
    truth = SimpleNamespace(complete=True, reason="truth_complete")
    evaluator = SimpleNamespace(reset=lambda request: None, update=lambda: truth)
    result, _, _, _ = run_sequence(
        ["stuck", "continuing"] * 4, recorder=recorder, evaluator=evaluator,
    )
    assert not result["success"] and result["completed_skills"] == 0
    assert recorder.predictions == ["stuck", "continuing"] * 4


def test_expert_verification_diagnostics_are_read_only_phase_counters():
    executor = ExpertSkillExecutor.__new__(ExpertSkillExecutor)
    executor.current = SkillRequest(CookieSkill.PLACE_FIVE, 1, 1, 2)
    executor.expert = SimpleNamespace(
        phase=CookiePhase.VERIFY_RELEASE, batch_index=1, phase_steps=7,
        _stable=7, completed_cookie_indices=[0, 1, 2, 3, 4],
        failed=False, failure_reason=None,
    )
    before = vars(executor.expert).copy()
    assert executor.verification_diagnostics() == {
        "executor_ready_to_switch": False,
        "expert_phase": "VERIFY_RELEASE", "expert_batch_index": 1,
        "expert_phase_steps": 7, "expert_stable_steps": 7,
        "expert_completed_cookie_count": 5,
        "expert_failed": False, "expert_failure_reason": None,
    }
    assert vars(executor.expert) == before
    executor.expert.phase = CookiePhase.DONE
    executor.expert.batch_index = 2
    assert executor.verification_diagnostics()["executor_ready_to_switch"] is True


def test_verify_audit_distinguishes_early_visual_yes_from_executor_readiness():
    env = Env()

    class GatedExecutor(Executor):
        def ready_to_switch(self):
            return len(env.actions) >= 3

        def verification_diagnostics(self):
            return {
                "executor_ready_to_switch": self.ready_to_switch(),
                "expert_phase": "RETRACT" if len(env.actions) < 3 else "DONE",
                "expert_batch_index": 1 if len(env.actions) < 3 else 2,
                "expert_stable_steps": len(env.actions),
                # Audit extensions cannot overwrite the actual model decision.
                "status": "stuck", "truth_complete": False,
            }

    verifier = Verifier("completed")
    executor = GatedExecutor()
    truth = SimpleNamespace(complete=True, reason="placed_before_retreat")
    evaluator = SimpleNamespace(reset=lambda request: None, update=lambda: truth)
    result = AgenticController(FixedSkillPlanner(), verifier, config=AgentLoopConfig(
        verify_every_steps=1, frame_interval=1, window_frames=2,
        skill_timeout_steps=6, max_steps=20, max_retries=0,
    )).run(env, executor, task="transfer ten", observation=observation(), seed=0,
           truth_evaluator=evaluator)
    assert result["success"] and result["completed_skills"] == 4
    assert result["steps"] == 6
    verify_events = [event for event in result["events"] if event["event"] == "verify"]
    assert [event["executor_ready_to_switch"] for event in verify_events[:3]] == [False, False, True]
    assert [event["expert_phase"] for event in verify_events[:3]] == ["RETRACT", "RETRACT", "DONE"]
    assert all(event["status"] == "completed" and event["truth_complete"]
               for event in verify_events)
    assert [event["step"] for event in result["events"] if event["event"] == "complete"][0] == 3
    # Verifier's input still consists exclusively of the two visual camera keys.
    assert all(set(frame) == {FRONT_IMAGE, LEFT_WRIST_IMAGE}
               for frames in verifier.seen for frame in frames)


def test_executor_audit_and_truth_cannot_promote_visual_no_to_success():
    env = Env()

    class ReadyExecutor(Executor):
        def ready_to_switch(self):
            return True

        def verification_diagnostics(self):
            return {"executor_ready_to_switch": True, "expert_phase": "DONE",
                    "expert_batch_index": 2, "expert_stable_steps": 20}

    truth = SimpleNamespace(complete=True, reason="real_box_complete")
    evaluator = SimpleNamespace(reset=lambda request: None, update=lambda: truth)
    result = AgenticController(FixedSkillPlanner(), Verifier("continuing"), config=AgentLoopConfig(
        verify_every_steps=1, frame_interval=1, window_frames=2,
        skill_timeout_steps=3, max_steps=10, max_retries=0,
    )).run(env, ReadyExecutor(), task="transfer ten", observation=observation(), seed=0,
           truth_evaluator=evaluator)
    assert not result["success"] and result["completed_skills"] == 0
    assert result["cookies_in_target"] == 10
    verify_events = [event for event in result["events"] if event["event"] == "verify"]
    assert len(verify_events) == 3
    assert all(event["status"] == "continuing" and event["truth_complete"]
               and event["executor_ready_to_switch"] for event in verify_events)


def test_planner_failure_has_structured_result_and_no_actions():
    class BrokenPlanner:
        def plan(self, task, images):
            raise RuntimeError("model unavailable")

    env = Env()
    result = AgenticController(BrokenPlanner(), Verifier("completed")).run(
        env, Executor(), task="transfer ten", observation=observation(), seed=0,
    )
    assert not result["success"] and result["completed_skills"] == 0
    assert not env.actions and "model unavailable" in result["failure_reason"]


def test_expert_held_pick_boundary_refuses_recovery_and_reselection():
    executor = ExpertSkillExecutor.__new__(ExpertSkillExecutor)
    executor.current = SkillRequest(CookieSkill.PICK_FIVE, 0, 1, 1)
    executor.source_column_number = 1
    executor.expert = SimpleNamespace(phase=CookiePhase.MOVE_TO_SLOT, batch_index=0)
    reason = executor.recovery_refusal_reason()
    assert "disagreement" in reason
    assert "without retreat or reselection" in reason
    with pytest.raises(RuntimeError, match="disagreement"):
        executor.start(executor.current)
    executor.expert.phase = CookiePhase.CLOSE
    assert executor.recovery_refusal_reason() is None


@pytest.mark.parametrize("phase", [CookiePhase.MOVE_TO_SLOT, CookiePhase.OPEN, CookiePhase.DONE])
def test_expert_place_refuses_recovery_before_any_replay(phase):
    executor = ExpertSkillExecutor.__new__(ExpertSkillExecutor)
    executor.current = SkillRequest(CookieSkill.PLACE_FIVE, 0, 1, 1)
    executor.expert = SimpleNamespace(phase=phase, batch_index=0)
    assert "PLACE recovery unsupported" in executor.recovery_refusal_reason()


@pytest.mark.parametrize("status", ["continuing", "stuck"])
def test_recovery_refusal_never_moves_or_restarts_executor(status):
    class SafeExecutor(Executor):
        def recovery_refusal_reason(self):
            return "verifier/executor disagreement; recovery unsupported"

    env = Env()
    executor = SafeExecutor()
    result = AgenticController(FixedSkillPlanner(), Verifier(status), config=AgentLoopConfig(
        verify_every_steps=2, frame_interval=1, skill_timeout_steps=4,
        max_steps=30, recovery_steps=2, max_retries=1,
    )).run(env, executor, task="transfer ten", observation=observation(), seed=0)
    assert not result["success"] and result["completed_skills"] == 0
    assert "disagreement" in result["failure_reason"]
    assert len(executor.starts) == 1
    assert all(len(action) == 16 for action in env.actions)
    assert not any(event["event"] == "recover" for event in result["events"])
    assert any(event["event"] == "recovery_refused" for event in result["events"])


def test_refusal_does_not_block_successful_verification():
    class SafeExecutor(Executor):
        def recovery_refusal_reason(self):
            return "recovery must not be used"

    executor = SafeExecutor()
    result = AgenticController(FixedSkillPlanner(), Verifier("completed"), config=AgentLoopConfig(
        verify_every_steps=2, frame_interval=1, skill_timeout_steps=4,
        max_steps=30, recovery_steps=2, max_retries=1,
    )).run(Env(), executor, task="transfer ten", observation=observation(), seed=0)
    assert result["success"] and result["completed_skills"] == 4
    assert len(executor.starts) == 4
