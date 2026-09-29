"""Lightweight skill contracts shared by agents, datasets and evaluation.

No simulation, recording, model-loading or training dependencies belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CookieSkill(str, Enum):
    PICK_FIVE = "PICK_FIVE"
    PLACE_FIVE = "PLACE_FIVE"


def skill_instruction(
    skill: CookieSkill, *, batch_index: int, source_column_number: int,
    target_slot_number: int,
) -> str:
    """Use one prompt template for training segments and agent execution."""
    if skill is CookieSkill.PICK_FIVE:
        return (
            f"Pick up five cookies from source column {source_column_number}, "
            f"batch {batch_index + 1}."
        )
    return f"Place the five held cookies into target box column {target_slot_number}."


@dataclass(frozen=True)
class SkillRequest:
    skill: CookieSkill
    batch_index: int
    source_column_number: int
    target_slot_number: int

    @property
    def instruction(self) -> str:
        return skill_instruction(
            self.skill,
            batch_index=self.batch_index,
            source_column_number=self.source_column_number,
            target_slot_number=self.target_slot_number,
        )


@dataclass(frozen=True)
class SkillOutcome:
    success: bool
    reason: str = ""
