"""Versioned course benchmark contract and its task inventory."""

COURSE_PROTOCOL = "course-v1"
LEGACY_PROTOCOL = "legacy"
COURSE_PROMPT = "Fill the target box."

COURSE_BASE_TASKS = (
    "2x5", "2x10",
    "missing-columns", "2x10-missing-columns",
    "partial-gaps", "2x10-partial-gaps",
    "cross-column", "2x10-cross-column",
    "sparse-exact-ten", "2x10-sparse-exact-twenty",
    "tilted", "2x10-tilted",
    "whole-row-lean", "2x10-whole-row-lean",
)
COURSE_BONUS_TASKS = ("prefilled", "two-boxes", "rotated-target", "variable-prefill")
COURSE_TASKS = (*COURSE_BASE_TASKS, *COURSE_BONUS_TASKS)


def course_max_steps(task: str) -> int:
    return 12000 if task == "2x10" or task.startswith("2x10-") else 6000
