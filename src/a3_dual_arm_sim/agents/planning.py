"""Single-box semantic planning, independent of physical control and training."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np

from a3_dual_arm_sim.agents.backends import ChatBackend, ModelOutputError, camera_view
from a3_dual_arm_sim.core.skills import CookieSkill, SkillOutcome, SkillRequest


class SingleBoxPlanner:
    """Pick/place twice; retry failed skills without changing the task order."""

    def __init__(self, *, source_column_number: int = 1, max_retries: int = 2) -> None:
        if not 1 <= source_column_number <= 4:
            raise ValueError("source_column_number must be in 1..4")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.source_column_number = source_column_number
        self.max_retries = max_retries
        self.reset()

    def reset(self) -> None:
        self.batch_index = 0
        self.holding = False
        self.retries = 0
        self.failed = False

    @property
    def done(self) -> bool:
        return self.batch_index == 2 and not self.failed

    def next_skill(self) -> SkillRequest | None:
        if self.done or self.failed:
            return None
        return SkillRequest(
            skill=CookieSkill.PLACE_FIVE if self.holding else CookieSkill.PICK_FIVE,
            batch_index=self.batch_index,
            source_column_number=self.source_column_number,
            target_slot_number=self.batch_index + 1,
        )

    def update(self, request: SkillRequest, outcome: SkillOutcome) -> None:
        if request != self.next_skill():
            raise ValueError("outcome does not match the pending skill")
        if not outcome.success:
            self.retries += 1
            self.failed = self.retries > self.max_retries
            return
        self.retries = 0
        if request.skill is CookieSkill.PICK_FIVE:
            self.holding = True
        else:
            self.holding = False
            self.batch_index += 1


class FixedSkillPlanner:
    """Explicit control baseline, not a language-model planner."""

    def __init__(self, source_column_number: int = 1) -> None:
        if not 1 <= source_column_number <= 4:
            raise ValueError("source column must be 1..4")
        self.source_column_number = source_column_number

    def plan(self, task: str, initial_images: Mapping[str, np.ndarray]) -> list[SkillRequest]:
        return [
            SkillRequest(skill, batch, self.source_column_number, batch + 1)
            for batch in range(2)
            for skill in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE)
        ]


def validate_single_box_plan(
    requests: Sequence[SkillRequest], *, allowed_source_columns: Sequence[int] = (1, 2, 3, 4)
) -> None:
    expected = [CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE] * 2
    if len(requests) != 4:
        raise ModelOutputError("single-box plans must contain exactly four skills")
    source = requests[0].source_column_number
    for index, item in enumerate(requests):
        batch = index // 2
        if (
            item.skill is not expected[index]
            or any(type(number) is not int for number in (
                item.batch_index, item.source_column_number, item.target_slot_number
            ))
            or item.batch_index != batch
            or item.source_column_number not in allowed_source_columns
            or item.source_column_number != source
            or item.target_slot_number != batch + 1
        ):
            raise ModelOutputError("plan violates the single-box skill sequence or column limits")


class MultimodalSkillPlanner:
    def __init__(
        self, backend: ChatBackend, *, allowed_source_columns: Sequence[int] = (1, 2, 3, 4)
    ) -> None:
        self.backend = backend
        self.allowed_source_columns = tuple(allowed_source_columns)
        if not self.allowed_source_columns or any(
            type(number) is not int or not 1 <= number <= 4
            for number in self.allowed_source_columns
        ):
            raise ValueError("allowed source columns must be a nonempty subset of 1..4")
        # Bounded diagnostics only; do not include model output in transport errors
        # or automatically log it. The planner never receives API credentials.
        self.last_response: str | None = None

    def plan(self, task: str, initial_images: Mapping[str, np.ndarray]) -> list[SkillRequest]:
        self.last_response = None
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be nonempty text")
        front = camera_view(initial_images, "front")
        schema_example = [
            {
                "skill": skill.value,
                "batch_index": batch,
                "source_column_number": self.allowed_source_columns[0],
                "target_slot_number": batch + 1,
            }
            for batch in range(2)
            for skill in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE)
        ]
        prompt = (
            "You plan a single-box A3 cookie packing task from the attached initial front image. "
            "The source tray has four columns numbered 1..4 in increasing workspace X. "
            "The target box has two columns numbered 1..2. Each batch has exactly five cookies. "
            "Atomic skill library: PICK_FIVE picks and lifts five cookies from one source column; "
            "PLACE_FIVE transports the held cookies, releases them into one target box column, "
            "and retracts the gripper. Do not invent other skills or low-level actions. "
            f"Allowed source columns: {list(self.allowed_source_columns)}. "
            "Choose one source column for both batches. Return ONLY a JSON array of exactly "
            "four objects with these exact fields: skill, batch_index, source_column_number, "
            "target_slot_number. Order: PICK_FIVE, PLACE_FIVE, PICK_FIVE, PLACE_FIVE. "
            "Use batch_index 0 for the first pair and 1 for the second; target_slot_number "
            "1 for the first pair and 2 for the second. All numeric fields must be integers. "
            'The skill field must contain exactly "PICK_FIVE" or "PLACE_FIVE", not a '
            "natural-language instruction. No Markdown or explanatory text.\n"
            "Example of the complete required JSON format (the source column in this "
            "example is not a prescribed choice; choose from the allowed columns for "
            "the actual task and image):\n" + json.dumps(schema_example)
            + "\nUser task: " + task
            + "\nReturn exactly four objects, not six. There are only two batches in total. "
            "Do not repeat the example as extra steps. Across the four objects, "
            "batch_index must be [0, 0, 1, 1] and target_slot_number must be [1, 1, 2, 2]."
        )
        raw = self.backend.complete(prompt, [front])
        if not isinstance(raw, str):
            raise ModelOutputError("planner response must be text containing the four-skill JSON schema")
        self.last_response = raw[:16384]
        if len(raw) > 65536:
            raise ModelOutputError("planner response exceeded the text size limit")
        # A code fence wrapping the entire response is only presentation, not a
        # different plan. Do not search for JSON buried in prose or repair values.
        lines = raw.strip().splitlines()
        if len(lines) >= 3 and lines[0].lower() in ("```", "```json") and lines[-1] == "```":
            raw = "\n".join(lines[1:-1])
        try:
            values = json.loads(raw)
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError
            result = []
            keys = {"skill", "batch_index", "source_column_number", "target_slot_number"}
            for value in values:
                if not isinstance(value, dict) or set(value) != keys:
                    raise ValueError
                if any(type(value[key]) is not int for key in keys - {"skill"}):
                    raise ValueError
                result.append(SkillRequest(
                    skill=CookieSkill(value["skill"]),
                    batch_index=value["batch_index"],
                    source_column_number=value["source_column_number"],
                    target_slot_number=value["target_slot_number"],
                ))
        except (ValueError, TypeError, KeyError):
            raise ModelOutputError("planner response must be the four-skill JSON schema") from None
        validate_single_box_plan(result, allowed_source_columns=self.allowed_source_columns)
        return result
