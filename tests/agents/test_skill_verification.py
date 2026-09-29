from __future__ import annotations

import json
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest
from a3_dual_arm_sim.data.verifier_recording import TemporalVerificationRecorder
from a3_dual_arm_sim.evaluation.skill_truth import (
    GroundTruthSkillEvaluator,
    SkillTruth,
    SkillTruthConfig,
    request_cookie_indices,
)


class FakeEnv:
    def __init__(self):
        self.SOURCE_POSITIONS = np.array([
            [x, y * 0.01] for x in (0.1, 0.2, 0.3, 0.4) for y in range(20)
        ])
        self.positions = np.column_stack([self.SOURCE_POSITIONS, np.full(80, 0.77)])
        self.local_positions = self.positions.copy()
        self.source_mask = np.ones(80, dtype=bool)
        self.target_mask = np.zeros(80, dtype=bool)
        self.finger_contacts = [(False, False) for _ in range(80)]
        self.TARGET_SLOTS_LOCAL = np.array([
            [x, row * 0.01] for x in (-0.025, 0.025) for row in range(5)
        ])
        self.TARGET_SLOT_TOLERANCE = np.array([0.01, 0.0025])
        self._eef_sites = (0, 1)
        self._left_finger_geoms = (1000, 1001)
        self._cookie_collision_geoms = tuple(frozenset([i + 100]) for i in range(80))
        self.model = object()
        self.current_joint_action = np.zeros(16)
        self.data = SimpleNamespace(
            time=0.0,
            site_xpos=np.array([[0.1, 0.02, 0.9], [0.4, 0.4, 0.9]]),
            site_xmat=np.array([np.eye(3).ravel(), np.eye(3).ravel()]),
            contact=[],
            ncon=0,
        )

    def privileged_cookie_position(self, index):
        return self.positions[index].copy()

    def privileged_cookie_target_position(self, index):
        return self.local_positions[index].copy()

    def privileged_cookie_in_source(self, index):
        return bool(self.source_mask[index])

    def privileged_cookie_in_target(self, index):
        return bool(self.target_mask[index])

    def privileged_left_finger_contacts(self, index):
        return self.finger_contacts[index]

    def tick(self, evaluator, times=1):
        result = None
        for _ in range(times):
            self.data.time += 0.05
            result = evaluator.update()
        return result

    def chain(self, indices):
        nodes = [1000, *(i + 100 for i in indices), 1001]
        self.data.contact = [
            SimpleNamespace(geom1=first, geom2=second)
            for first, second in pairwise(nodes)
        ]
        self.data.ncon = len(self.data.contact)


@pytest.fixture
def env(monkeypatch):
    def normal_force(_model, _data, _index, force):
        force[0] = 0.5

    monkeypatch.setattr("mujoco.mj_contactForce", normal_force)
    return FakeEnv()


def request(skill=CookieSkill.PICK_FIVE, *, batch=0, source=1, target=1):
    return SkillRequest(skill, batch, source, target)


def test_original_source_ids_are_selected_not_current_moved_positions(env):
    expected = (45, 46, 47, 48, 49)
    env.positions[:] = 0
    assert request_cookie_indices(env, request(batch=1, source=3)) == expected
    with pytest.raises(ValueError, match="batch index"):
        request_cookie_indices(env, request(batch=2))


def test_pick_requires_all_five_and_force_bearing_chain_not_direct_double_contacts(env):
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request())
    indices = evaluator.cookie_indices
    env.positions[list(indices), 2] += 0.08
    env.source_mask[list(indices)] = False
    env.chain(indices)
    # End pads touch only exterior cookies; inner cookies have no pad contacts.
    env.finger_contacts[indices[0]] = (True, False)
    env.finger_contacts[indices[-1]] = (False, True)
    assert not env.tick(evaluator).complete  # fast initial rise is not settled
    assert not env.tick(evaluator, times=4).complete
    result = env.tick(evaluator)
    assert result.complete
    assert result.cookie_indices == (0, 1, 2, 3, 4)
    assert result.stable_steps == 5
    assert evaluator.check(request()).success
    assert evaluator.latest.stable_steps == 5  # no same-frame hold accumulation


def test_pick_does_not_accept_lifting_only_four_or_losing_contact_chain(env):
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request())
    env.positions[:4, 2] += 0.08
    env.source_mask[:5] = False
    env.chain(range(5))
    result = env.tick(evaluator, times=10)
    assert not result.complete
    assert result.reason == "not_all_five_lifted"
    env.positions[4, 2] += 0.08
    env.data.contact.pop(2)
    env.data.ncon -= 1
    result = env.tick(evaluator, times=10)
    assert not result.complete
    assert result.reason == "broken_grasp_contact_chain"


def test_pick_relative_tool_slip_resets_temporal_hold(env):
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request())
    env.positions[:5, 2] += 0.08
    env.source_mask[:5] = False
    env.chain(range(5))
    env.tick(evaluator, times=4)
    env.positions[2, 0] += 0.008
    result = env.tick(evaluator)
    assert not result.complete
    assert result.stable_steps == 0
    assert result.reason == "cookies_slipping_relative_to_tool"


def test_pick_retry_preserves_initial_source_lift_reference(env):
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request())
    env.positions[:5, 2] += 0.08
    env.source_mask[:5] = False
    env.chain(range(5))
    assert env.tick(evaluator, times=6).complete
    evaluator.reset(request())
    assert env.tick(evaluator, times=5).complete


def test_place_never_counts_previous_five_and_requires_correct_target_column(env):
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request(CookieSkill.PLACE_FIVE, batch=1, target=2))
    env.target_mask[:5] = True
    env.local_positions[:5, 0] = 0.025
    assert not env.tick(evaluator, times=10).complete
    env.target_mask[5:10] = True
    env.local_positions[5:10, 0] = -0.025
    result = env.tick(evaluator, times=10)
    assert not result.complete
    assert result.reason == "cookies_in_wrong_target_column"
    env.local_positions[5:10, 0] = 0.025
    assert not env.tick(evaluator, times=5).complete
    assert env.tick(evaluator).complete


def test_place_requires_release_even_when_environment_allows_contact(env):
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request(CookieSkill.PLACE_FIVE))
    env.local_positions[:5, 0] = -0.025
    env.target_mask[:5] = True
    env.finger_contacts[2] = (False, True)
    result = env.tick(evaluator, times=10)
    assert not result.complete
    assert result.reason == "cookies_still_contacting_gripper"
    env.finger_contacts[2] = (False, False)
    assert env.tick(evaluator, times=6).complete
    env.target_mask[3] = False
    assert not env.tick(evaluator).complete


def test_final_place_respects_whole_task_stable_hold(env):
    env.task_config = SimpleNamespace(success_hold_steps=20)
    evaluator = GroundTruthSkillEvaluator(env)
    evaluator.reset(request(CookieSkill.PLACE_FIVE, batch=1, target=2))
    env.local_positions[5:10, 0] = 0.025
    env.target_mask[5:10] = True
    assert not env.tick(evaluator, times=19).complete
    assert env.tick(evaluator).complete


def test_stuck_requires_observed_stationary_no_progress_window(env):
    evaluator = GroundTruthSkillEvaluator(env, config=SkillTruthConfig(stalled_steps=4))
    evaluator.reset(request())
    assert env.tick(evaluator, times=3).diagnosis == "StillTrying"
    assert env.tick(evaluator).diagnosis == "Stuck"
    env.data.site_xpos[0, 0] += 0.01
    assert env.tick(evaluator).diagnosis == "StillTrying"
    with pytest.raises(ValueError, match="does not match"):
        evaluator.check(request(batch=1))


def test_closing_gripper_is_not_stationary_stuck(env):
    evaluator = GroundTruthSkillEvaluator(env, config=SkillTruthConfig(stalled_steps=4))
    evaluator.reset(request())
    env.tick(evaluator, times=3)
    env.current_joint_action[7] += 0.1
    assert env.tick(evaluator).diagnosis == "StillTrying"


def observation():
    return {
        **{
            f"observation.images.{camera}": np.full((8, 8, 3), 128, dtype=np.uint8)
            for camera in TemporalVerificationRecorder.CAMERAS
        },
        "privileged.cookie_positions": np.ones((80, 3)),
        "observation.state": np.arange(16),
    }


def test_temporal_export_has_image_only_inputs_and_separate_truth_audit(tmp_path):
    recorder = TemporalVerificationRecorder(tmp_path, max_frames=3, frame_stride=2)
    recorder.start(request(), seed=11)
    obs = observation()
    recorder.observe(obs, step=0)
    assert recorder.record(SkillTruth(False, "incomplete", (0, 1, 2, 3, 4))) is None
    recorder.observe(obs, step=1)
    recorder.observe(obs, step=2)
    result = SkillTruth(False, "cookies_still_in_source", (0, 1, 2, 3, 4), diagnosis="StillTrying")
    assert recorder.record(result) == 0
    sample = json.loads((tmp_path / "samples.jsonl").read_text())
    audit = json.loads((tmp_path / "truth_audit.jsonl").read_text())
    assert sample["label"] == {"completion": "No", "diagnosis": "StillTrying"}
    assert set(sample["input"]) == {"subgoal", "frames"}
    assert "cookie_indices" not in sample
    assert "reason" not in sample
    assert "observation.state" not in str(sample)
    assert [frame["step"] for frame in sample["input"]["frames"]] == [0, 2]
    for frame in sample["input"]["frames"]:
        for path in frame["images"].values():
            assert (tmp_path / path).is_file()
    assert audit["cookie_indices"] == [0, 1, 2, 3, 4]


def test_temporal_export_can_label_stuck_or_success_and_keeps_attempts(tmp_path):
    recorder = TemporalVerificationRecorder(tmp_path, frame_stride=1)
    recorder.start(request(), seed=3, attempt=1)
    recorder.observe(observation(), step=0)
    recorder.observe(observation(), step=1)
    recorder.record(SkillTruth(False, "no_progress", (0, 1, 2, 3, 4)), diagnosis="Stuck")
    recorder.start(request(CookieSkill.PLACE_FIVE), seed=3)
    recorder.observe(observation(), step=10)
    recorder.observe(observation(), step=11)
    success = SkillTruth(True, "released", (0, 1, 2, 3, 4), stable_steps=6)
    with pytest.raises(ValueError, match="successful verification"):
        recorder.record(success, diagnosis="Stuck")
    recorder.record(success)
    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert samples[0]["attempt"] == 1
    assert samples[0]["label"]["diagnosis"] == "Stuck"
    assert samples[1]["label"] == {"completion": "Yes", "diagnosis": None}
    with pytest.raises(FileExistsError):
        TemporalVerificationRecorder(tmp_path)
    recorder.close()
