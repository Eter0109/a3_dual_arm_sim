"""Temporal visual skill verification and an explicit privileged oracle baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from a3_dual_arm_sim.agents.backends import ChatBackend, ModelOutputError, camera_view
from a3_dual_arm_sim.agents.prompts import (
    verifier_completion_prompt,
    verifier_diagnosis_prompt,
)
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest


@dataclass(frozen=True)
class VerificationDecision:
    status: Literal["completed", "continuing", "stuck"]
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ("completed", "continuing", "stuck"):
            raise ValueError("invalid verification status")


class TemporalSkillVerifier:
    """Two-stage visual assessment; no truth-state shortcuts or implicit success."""

    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    def check(
        self, request: SkillRequest, frames: Sequence[Mapping[str, np.ndarray]]
    ) -> VerificationDecision:
        if request.skill not in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE):
            raise ValueError("unsupported single-box verification skill")
        if len(frames) < 2:
            raise ValueError("temporal verification requires at least two camera frame pairs")
        images = []
        for frame in frames:
            images.extend((camera_view(frame, "front"), camera_view(frame, "left_wrist")))
        answer = self.backend.complete(
            verifier_completion_prompt(request.skill, request.instruction, len(frames)), images,
        ).strip().lower()
        if answer == "yes":
            return VerificationDecision("completed", "visual verifier confirmed subgoal completion")
        if answer != "no":
            raise ModelOutputError("completion verifier must return exactly Yes or No")
        diagnosis = self.backend.complete(
            verifier_diagnosis_prompt(request.instruction, len(frames)), images,
        ).strip().lower()
        if diagnosis == "stuck":
            return VerificationDecision("stuck", "visual verifier diagnosed stalled execution")
        if diagnosis == "stilltrying":
            return VerificationDecision("continuing", "visual verifier observed ongoing execution")
        raise ModelOutputError("diagnosis verifier must return exactly Stuck or StillTrying")


class OracleSkillVerifier:
    """Ground-truth baseline for integration tests, never a visual-verifier fallback."""

    uses_simulator_truth = True

    def check_truth(self, truth: Any) -> Any:
        status = "completed" if truth.complete else "continuing"
        return VerificationDecision(status=status, reason=truth.reason)
