from __future__ import annotations

import json

import numpy as np
import pytest

from a3_dual_arm_sim.agents.executors import PolicySkillExecutor
from a3_dual_arm_sim.agents.planning import SingleBoxPlanner
from a3_dual_arm_sim.core.contracts import EpisodeContext
from a3_dual_arm_sim.core.skills import CookieSkill, SkillOutcome
from a3_dual_arm_sim.data.skills import (
    CookieSkillEpisodeRecorder,
    CookieSkillLabel,
    label_expert_step,
    skill_for_phase,
)
from a3_dual_arm_sim.experts.single_cookie import CookiePhase


@pytest.mark.parametrize(
    ("phase", "skill"),
    [
        (CookiePhase.SELECT_COOKIE, CookieSkill.PICK_FIVE),
        (CookiePhase.ALIGN, CookieSkill.PICK_FIVE),
        (CookiePhase.LIFT, CookieSkill.PICK_FIVE),
        (CookiePhase.MOVE_TO_SLOT, CookieSkill.PLACE_FIVE),
        (CookiePhase.VERIFY_RELEASE, CookieSkill.PLACE_FIVE),
    ],
)
def test_phase_to_skill(phase, skill):
    assert skill_for_phase(phase) is skill


def test_label_uses_pre_action_phase_and_selected_source_column():
    positions = np.array([[x, y, 0.0] for x in (0.1, 0.2, 0.3, 0.4) for y in range(10)])
    label = label_expert_step(CookiePhase.SELECT_COOKIE, 1, [20, 21, 22, 23, 24], positions)
    assert label.skill is CookieSkill.PICK_FIVE
    assert label.batch_index == 1
    assert label.source_column_number == 3
    assert label.cookie_indices == (20, 21, 22, 23, 24)


def test_skill_recorder_saves_four_fixed_task_episodes_only_after_success(monkeypatch, tmp_path):
    class Writer:
        def __init__(self, root, **_kwargs):
            self.root = tmp_path
            self._episode_index = 0
            self.episodes = []

        def start_episode(self, context, controller_type):
            self.context = context
            self.frames = []

        def add_frame(self, observation, action):
            self.frames.append((observation, action))

        def finish_episode(self, success):
            self.episodes.append((self.context.task, self.frames, success))
            self._episode_index += 1

        def close(self):
            pass

    monkeypatch.setattr("a3_dual_arm_sim.data.skills.LeRobotV3Recorder", Writer)
    recorder = CookieSkillEpisodeRecorder(
        tmp_path, repo_id="local/test-skills", fps=20, image_height=256, image_width=256
    )
    observation = {
        "observation.images.front": np.zeros((1, 1, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.zeros((1, 1, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.zeros((1, 1, 3), dtype=np.uint8),
        "observation.state": np.zeros(16),
        "observation.velocity": np.zeros(16),
        "observation.eef_pose": np.zeros(14),
        "observation.force": np.zeros(18),
    }
    labels = [
        CookieSkillLabel(skill, batch, 1, batch + 1, tuple(range(batch * 5, batch * 5 + 5)))
        for batch in range(2)
        for skill in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE)
    ]
    context = EpisodeContext(7, "transfer 10 cookies", "cartesian_delta")
    recorder.start_episode(context, "same_column")
    for label in labels:
        recorder.add_expert_frame(observation, np.zeros(16), label)
    assert recorder.writer.episodes == []
    recorder.discard_episode()
    assert recorder.saved_rollouts == 0

    recorder.start_episode(context, "same_column")
    for label in labels:
        recorder.add_expert_frame(observation, np.zeros(16), label)
    recorder.finish_episode(True)
    assert recorder.saved_rollouts == 1
    assert [item[0] for item in recorder.writer.episodes] == [
        label.instruction for label in labels
    ]
    assert all(item[2] is True and len(item[1]) == 1 for item in recorder.writer.episodes)
    segments = [
        json.loads(line)
        for line in (tmp_path / "a3_skill_segments.jsonl").read_text().splitlines()
    ]
    assert [(item["skill"], item["batch_index"]) for item in segments] == [
        (label.skill.value, label.batch_index) for label in labels
    ]
    assert all(item["parent_seed"] == 7 for item in segments)


def test_agent_planner_and_policy_adapter_clear_action_chunk_per_skill():
    class FakePolicy:
        action_mode = "joint_position"

        def __init__(self):
            self.resets = []
            self.tasks = []

        def reset(self, context):
            self.resets.append(context)

        def act(self, observation, task):
            self.tasks.append(task)
            return np.zeros(16)

        def close(self):
            pass

    planner = SingleBoxPlanner(source_column_number=2, max_retries=1)
    policy = FakePolicy()
    executor = PolicySkillExecutor(policy, seed=11)
    first = planner.next_skill()
    assert first.skill is CookieSkill.PICK_FIVE
    executor.start(first)
    executor.act({})
    assert policy.tasks == [first.instruction]
    planner.update(first, SkillOutcome(False, "missed"))
    assert planner.next_skill() == first
    executor.start(first)
    planner.update(first, SkillOutcome(True))
    second = planner.next_skill()
    assert second.skill is CookieSkill.PLACE_FIVE
    executor.start(second)
    assert [item.task for item in policy.resets] == [
        first.instruction, first.instruction, second.instruction
    ]
    planner.update(second, SkillOutcome(True))
    for _ in range(2):
        request = planner.next_skill()
        planner.update(request, SkillOutcome(True))
    assert planner.done and planner.next_skill() is None
