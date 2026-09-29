from __future__ import annotations

import ast
import inspect
from dataclasses import asdict

from a3_dual_arm_sim.core import skills
from a3_dual_arm_sim.data.skills import CookieSkill as RecordedCookieSkill


def test_shared_skill_enum_has_one_identity() -> None:
    assert RecordedCookieSkill is skills.CookieSkill
    assert skills.CookieSkill.PICK_FIVE.value == "PICK_FIVE"
    assert skills.CookieSkill.PLACE_FIVE.value == "PLACE_FIVE"


def test_shared_contract_preserves_skill_instructions_and_serialization() -> None:
    pick = skills.SkillRequest(skills.CookieSkill.PICK_FIVE, 1, 4, 2)
    place = skills.SkillRequest(skills.CookieSkill.PLACE_FIVE, 1, 4, 2)
    assert pick.instruction == "Pick up five cookies from source column 4, batch 2."
    assert place.instruction == "Place the five held cookies into target box column 2."
    assert asdict(pick) == {
        "skill": skills.CookieSkill.PICK_FIVE,
        "batch_index": 1,
        "source_column_number": 4,
        "target_slot_number": 2,
    }
    assert asdict(skills.SkillOutcome(True)) == {"success": True, "reason": ""}


def test_shared_skill_contract_has_only_standard_library_dependencies() -> None:
    tree = ast.parse(inspect.getsource(skills))
    dependencies = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert dependencies <= {"__future__", "dataclasses", "enum"}
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))
