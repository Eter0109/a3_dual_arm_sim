"""Exact shared visual-verification prompts for inference and data preparation."""

from __future__ import annotations

from a3_dual_arm_sim.core.skills import CookieSkill


def verifier_completion_prompt(skill: CookieSkill, instruction: str, frame_count: int) -> str:
    """One shared inference/training prompt; it never receives truth metadata."""
    if skill not in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE):
        raise ValueError("unsupported single-box verification skill")
    if frame_count < 2 or not instruction.strip():
        raise ValueError("verifier prompt requires temporal frames and a subgoal")
    layout = (
        f"There are {frame_count} frame pairs in chronological order, oldest first. "
        "Each pair is front/global view followed by left wrist view. "
    )
    condition = (
        "All five requested cookies must be securely held together and clearly lifted "
        "off the source tray; contact alone or fewer than five is not success."
        if skill is CookieSkill.PICK_FIVE else
        "All five held cookies must be stably placed in the specified target box column, "
        "with the gripper empty and retracting or clear of the cookies."
    )
    return (
        layout + f"Current subgoal: {instruction}\n" + condition
        + " Combine both camera views; an occlusion in one view is not evidence of failure "
        "if the other clearly shows the required state. Based only on these images, "
        "is this subgoal complete at the last frame? "
        "Answer exactly Yes or No, without punctuation or explanations."
    )


def verifier_diagnosis_prompt(instruction: str, frame_count: int) -> str:
    """Distinguish actual failure from normal not-yet-complete manipulation."""
    if frame_count < 2 or not instruction.strip():
        raise ValueError("verifier prompt requires temporal frames and a subgoal")
    return (
        f"There are {frame_count} frame pairs in chronological order, oldest first. "
        "Each pair is front/global view followed by left wrist view. "
        f"Current subgoal: {instruction}\n"
        "The subgoal is not yet complete. Is execution stuck (sustained near-stationary "
        "gripper and cookies with no meaningful motion or progress across the full observed "
        "history), or still trying? Normal slow alignment, rotation, gripper closing, "
        "movement after a failed grasp and a brief pause are not by themselves Stuck. "
        "Choose Stuck only for sustained near-stationary/no-progress behavior; otherwise "
        "choose StillTrying. Base your diagnosis only "
        "on the temporal images. Answer exactly Stuck or StillTrying, without punctuation "
        "or explanations."
    )
