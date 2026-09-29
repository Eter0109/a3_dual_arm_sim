"""Semantic-skill adapters for interchangeable policies and reference experts.

The physical expert stays in experts; these adapters only manage skill boundaries.
Expert modules load only when the expert adapter is actually used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from a3_dual_arm_sim.core.contracts import EpisodeContext
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest
from a3_dual_arm_sim.policies.base import Policy


class PolicySkillExecutor:
    """Pass the current skill prompt to any existing A3 policy plugin.

    Resetting at each skill boundary also clears queued VLA action chunks, so
    actions planned for PICK cannot leak into PLACE or the next batch.
    """

    def __init__(self, policy: Policy, *, seed: int) -> None:
        self.policy = policy
        self.seed = seed
        self.current: SkillRequest | None = None

    def start(self, request: SkillRequest) -> None:
        self.current = request
        self.policy.reset(
            EpisodeContext(
                seed=self.seed,
                task=request.instruction,
                action_mode=self.policy.action_mode,
            )
        )

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if self.current is None:
            raise RuntimeError("start a skill before requesting an action")
        return self.policy.act(observation, self.current.instruction)

    def close(self) -> None:
        self.policy.close()


class ExpertSkillExecutor:
    """Keep the original physical expert intact; gate its semantic boundaries.

    This privileged controller is only a benchmark executor.  It is never
    presented as a VLA policy or as evidence of a VLM's verification accuracy.
    """

    def __init__(self, env: Any, *, seed: int, source_column_number: int = 1) -> None:
        from a3_dual_arm_sim.experts.same_column import (
            A3SameColumnBatchExpert,
            A3VariedColumnBatchExpert,
        )

        self.env = env
        self.seed = seed
        self.source_column_number = source_column_number
        if source_column_number == 1:
            self.expert = A3SameColumnBatchExpert(env)
        else:
            self.expert = A3VariedColumnBatchExpert(env)
            self.expert.requested_source_column_index = source_column_number - 1
        self.expert.reset(EpisodeContext(seed, "single-box skill benchmark", "cartesian_delta"))
        self.current: SkillRequest | None = None

    def start(self, request: SkillRequest) -> None:
        from a3_dual_arm_sim.experts.single_cookie import CookiePhase

        if request.source_column_number != self.source_column_number:
            raise ValueError("expert source column differs from the generated plan")
        if self.current == request:
            # After recovery, retry a physical grasp without resetting the scene
            # or losing the already placed batch.  The controller checks this
            # before retreat; repeat the guard here for direct executor callers.
            refusal = self.recovery_refusal_reason()
            if refusal is not None:
                raise RuntimeError(refusal)
            self.expert.failure_reason = None
            self.expert.failure_phase = None
            self.expert._advance(CookiePhase.SELECT_COOKIE)
        if request.batch_index != self.expert.batch_index:
            raise RuntimeError("expert has not reached the requested batch boundary")
        if request.skill is CookieSkill.PLACE_FIVE:
            if self.expert.phase is not CookiePhase.MOVE_TO_SLOT:
                raise RuntimeError("PLACE requested before the physical expert finished PICK")
        elif (self.current is not None and self.current != request
              and self.expert.phase is not CookiePhase.SELECT_COOKIE):
            raise RuntimeError("next PICK requested before release and retreat completed")
        self.current = request

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        from a3_dual_arm_sim.experts.single_cookie import CookiePhase

        if self.current is None:
            raise RuntimeError("start a skill before requesting an action")
        if self.expert.failed:
            raise RuntimeError(self.expert.failure_reason or "expert failed")
        if self.current.skill is CookieSkill.PICK_FIVE:
            if self.expert.phase is CookiePhase.MOVE_TO_SLOT:
                return self.expert._hold_command(self.expert._opening)
        elif self.expert.batch_index > self.current.batch_index:
            return self.expert._hold_command(self.expert._opening)
        return self.expert.act(observation, self.current.instruction)

    def ready_to_switch(self) -> bool:
        """Executor boundary readiness, not a replacement completion verdict."""
        from a3_dual_arm_sim.experts.single_cookie import CookiePhase

        if self.current is None:
            return False
        if self.current.skill is CookieSkill.PICK_FIVE:
            return self.expert.phase is CookiePhase.MOVE_TO_SLOT
        return self.expert.batch_index > self.current.batch_index

    def verification_diagnostics(self) -> dict[str, Any]:
        """Read-only executor audit, never a verifier input or completion label.

        ``_stable`` is the expert's current-phase counter, not necessarily the
        independent evaluator's hold count. In particular PLACE completion may
        precede the expert's retreat and VERIFY_RELEASE boundary.
        """
        return {
            "executor_ready_to_switch": self.ready_to_switch(),
            "expert_phase": getattr(self.expert.phase, "name", None),
            "expert_batch_index": self.expert.batch_index,
            "expert_phase_steps": getattr(self.expert, "phase_steps", None),
            "expert_stable_steps": getattr(self.expert, "_stable", None),
            "expert_completed_cookie_count": len(
                getattr(self.expert, "completed_cookie_indices", ())
            ),
            "expert_failed": bool(getattr(self.expert, "failed", False)),
            "expert_failure_reason": getattr(self.expert, "failure_reason", None),
        }

    def recovery_refusal_reason(self) -> str | None:
        """Reject unsupported physical recovery before any retreat is applied.

        A held batch cannot be reselected from the source tray.  Likewise this
        reference expert has no safe PLACE replay: the cookies may already have
        been released.  Neither case is promoted to visual completion; stop and
        report the disagreement without disturbing the current physical state.
        """
        if self.current is None:
            return "expert has no active skill to recover"
        if self.current.skill is CookieSkill.PLACE_FIVE:
            return "expert PLACE recovery unsupported; holding position without retreat or replay"
        if self.ready_to_switch():
            return (
                "verifier/executor disagreement: expert PICK reached its held-batch boundary "
                "but verification did not complete; holding position without retreat or reselection"
            )
        return None

    def close(self) -> None:
        pass


def recovery_action(last_applied_action: np.ndarray) -> np.ndarray:
    """Small upward Cartesian retreat, preserving both commanded gripper states."""
    joint = np.asarray(last_applied_action)
    if joint.shape != (16,) or not np.isfinite(joint).all():
        raise ValueError("recovery requires the last finite 16D applied action")
    action = np.zeros(14)
    action[2] = 0.4
    action[6] = joint[7] * 2 - 1
    action[13] = joint[15] * 2 - 1
    return action
