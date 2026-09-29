"""Single-box, five-cookie skill labels and transactional episode recording.

The physical simulation remains one continuous rollout.  A successful rollout
is committed as four independent LeRobot episodes so action chunks cannot span
the boundary between picking and placing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.core.contracts import EpisodeContext
from a3_dual_arm_sim.core.skills import CookieSkill, skill_instruction
from a3_dual_arm_sim.data.lerobot import LeRobotV3Recorder
from a3_dual_arm_sim.experts.single_cookie import CookiePhase

_PICK_PHASES = {
    CookiePhase.SELECT_COOKIE,
    CookiePhase.ALIGN,
    CookiePhase.APPROACH,
    CookiePhase.PRE_CLOSE,
    CookiePhase.DESCEND,
    CookiePhase.CLOSE,
    CookiePhase.VERIFY_GRASP,
    CookiePhase.LIFT,
    CookiePhase.VERIFY_LIFT,
}
_PLACE_PHASES = {
    CookiePhase.MOVE_TO_SLOT,
    CookiePhase.DESCEND_TO_PLACE,
    CookiePhase.OPEN,
    CookiePhase.RETRACT,
    CookiePhase.VERIFY_RELEASE,
}


def skill_for_phase(phase: CookiePhase) -> CookieSkill:
    """Classify the phase *before* the expert produces the recorded action."""
    if phase in _PICK_PHASES:
        return CookieSkill.PICK_FIVE
    if phase in _PLACE_PHASES:
        return CookieSkill.PLACE_FIVE
    raise ValueError(f"phase {phase.name} is not a single-box training skill")


@dataclass(frozen=True)
class CookieSkillLabel:
    skill: CookieSkill
    batch_index: int
    source_column_number: int
    target_slot_number: int
    cookie_indices: tuple[int, ...]

    @property
    def instruction(self) -> str:
        return skill_instruction(
            self.skill,
            batch_index=self.batch_index,
            source_column_number=self.source_column_number,
            target_slot_number=self.target_slot_number,
        )

    @property
    def segment_key(self) -> tuple[int, CookieSkill]:
        return self.batch_index, self.skill


def label_expert_step(
    phase_before: CookiePhase,
    batch_index_before: int,
    cookie_indices_after: list[int],
    source_positions: np.ndarray,
) -> CookieSkillLabel:
    """Resolve selected cookies after act(), but attribute its action to the old phase."""
    if not cookie_indices_after:
        raise ValueError("expert selected no cookies for the current skill")
    positions = np.asarray(source_positions)
    column_x = float(positions[cookie_indices_after[0], 0])
    columns = sorted({float(x) for x in positions[:, 0]})
    source_column_number = int(np.argmin(np.abs(np.asarray(columns) - column_x))) + 1
    return CookieSkillLabel(
        skill=skill_for_phase(phase_before),
        batch_index=batch_index_before,
        source_column_number=source_column_number,
        target_slot_number=batch_index_before + 1,
        cookie_indices=tuple(int(i) for i in cookie_indices_after),
    )


_FRAME_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.state",
    "observation.velocity",
    "observation.eef_pose",
    "observation.force",
)


class CookieSkillEpisodeRecorder:
    """Commit four skill episodes only after the full expert rollout succeeds.

    Frames for one rollout are held in memory; no partial skill data is emitted
    if the expert fails.  This keeps the existing successful-episode contract.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str,
        fps: int,
        image_height: int,
        image_width: int,
        resume: bool = False,
    ) -> None:
        self.writer = LeRobotV3Recorder(
            root,
            repo_id=repo_id,
            fps=fps,
            image_height=image_height,
            image_width=image_width,
            resume=resume,
        )
        self.root = self.writer.root
        self._segments_path = self.root / "a3_skill_segments.jsonl"
        self._context: EpisodeContext | None = None
        self._controller_type = "unknown"
        self._frames: list[tuple[dict[str, Any], np.ndarray, CookieSkillLabel]] = []
        if resume:
            if not self._segments_path.is_file():
                raise FileNotFoundError("resume requires a3_skill_segments.jsonl")
            segment_lines = [
                line for line in self._segments_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(segment_lines) != self.writer._episode_index:
                raise RuntimeError("LeRobot and skill segment counts differ")
            if self.writer._episode_index % 4:
                raise RuntimeError("skill dataset ended in a partial four-segment rollout")

    @property
    def saved_rollouts(self) -> int:
        return self.writer._episode_index // 4

    def start_episode(self, context: EpisodeContext, controller_type: str) -> None:
        if self._context is not None:
            raise RuntimeError("a rollout is already active")
        self._context = context
        self._controller_type = controller_type
        self._frames = []

    def add_expert_frame(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        label: CookieSkillLabel,
    ) -> None:
        if self._context is None:
            raise RuntimeError("start_episode must be called before add_expert_frame")
        frame = {key: np.asarray(observation[key]).copy() for key in _FRAME_KEYS}
        self._frames.append((frame, np.asarray(action, dtype=np.float32).copy(), label))

    def add_frame(self, observation: dict[str, Any], action: np.ndarray) -> None:
        raise RuntimeError("skill recording requires expert phase annotations")

    def finish_episode(self, success: bool | None = None) -> None:
        if self._context is None:
            raise RuntimeError("no rollout is active")
        if success is not True:
            self.discard_episode()
            return
        segments: list[tuple[CookieSkillLabel, int, int]] = []
        for index, (_, _, label) in enumerate(self._frames):
            if not segments or label.segment_key != segments[-1][0].segment_key:
                segments.append((label, index, index + 1))
            else:
                prior_label, start, _ = segments[-1]
                if label != prior_label:
                    raise RuntimeError("skill label changed within a segment")
                segments[-1] = (prior_label, start, index + 1)
        expected = [
            (0, CookieSkill.PICK_FIVE),
            (0, CookieSkill.PLACE_FIVE),
            (1, CookieSkill.PICK_FIVE),
            (1, CookieSkill.PLACE_FIVE),
        ]
        if [label.segment_key for label, _, _ in segments] != expected:
            raise RuntimeError("successful rollout did not produce PICK/PLACE/PICK/PLACE")

        for label, start, end in segments:
            episode_index = self.writer._episode_index
            self.writer.start_episode(
                EpisodeContext(
                    seed=self._context.seed,
                    task=label.instruction,
                    action_mode=self._context.action_mode,
                ),
                self._controller_type,
            )
            for frame, action, _ in self._frames[start:end]:
                self.writer.add_frame(frame, action)
            self.writer.finish_episode(success=True)
            metadata = {
                "episode_index": episode_index,
                "parent_seed": self._context.seed,
                "parent_task": self._context.task,
                "skill": label.skill.value,
                "instruction": label.instruction,
                "batch_index": label.batch_index,
                "source_column_number": label.source_column_number,
                "target_slot_number": label.target_slot_number,
                "cookie_indices": list(label.cookie_indices),
                "parent_start_step": start,
                "parent_end_step_exclusive": end,
                "frames": end - start,
                "success": True,
            }
            with self._segments_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        self._context = None
        self._frames = []

    def discard_episode(self) -> None:
        self._context = None
        self._frames = []

    def close(self) -> None:
        self.discard_episode()
        self.writer.close()
