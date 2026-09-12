from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .cookie_transfer import A3CookieTransferEnv
from .expert import A3CookieTransferExpert, CookiePhase


@dataclass(frozen=True)
class CookieTransferEpisodeResult:
    seed: int
    success: bool
    grasp_verified: bool
    lift_verified: bool
    dropped_during_transfer: bool
    completed_cookies: int
    expert_confirmed_cookies: int
    failed_cookie: int | None
    failure_phase: str | None
    reason: str | None
    max_gripper_force_n: float
    steps: int
    cookies_in_target: int
    cookies_in_source: int
    target_slot_occupancy: list[int]
    retries_per_slot: list[int]
    retry_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cookie_transfer_episode(
    seed: int,
    *,
    max_steps: int = 2500,
    randomize_cookies: bool = True,
) -> CookieTransferEpisodeResult:
    """Run one headless closed-loop Expert episode and return stable metrics."""
    env = A3CookieTransferEnv(render_cameras=False)
    expert = A3CookieTransferExpert(env)
    max_force = 0.0
    steps = 0
    truncated = False
    try:
        observation, info = env.reset(
            seed=seed, options={"randomize_cookies": randomize_cookies}
        )
        expert.reset()
        terminated = False
        while not terminated and not expert.failed and steps < max_steps:
            action = expert.act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            touches = env.privileged_left_touch_values()
            max_force = max(max_force, *touches)
            steps += 1
            if truncated:
                break

        phases = set(expert.transition_history)
        dropped = any("dropped" in reason.lower() for reason in expert.retry_reasons)
        success = bool(info.get("success", False))
        stopped_by_limit = not success and not expert.failed and steps >= max_steps
        failure_phase = expert.failure_phase
        failure_reason = expert.failure_reason
        if not success and failure_phase is None:
            failure_phase = expert.phase.name
        if not success and failure_reason is None:
            failure_reason = (
                "environment_truncated"
                if truncated
                else "max_steps_exceeded"
                if stopped_by_limit
                else "episode_ended_without_success"
            )
        return CookieTransferEpisodeResult(
            seed=seed,
            success=success,
            grasp_verified=CookiePhase.LIFT in phases,
            lift_verified=CookiePhase.MOVE_TO_SLOT in phases,
            dropped_during_transfer=dropped,
            completed_cookies=int(info.get("cookies_in_target", 0)),
            expert_confirmed_cookies=len(expert.completed_cookie_indices),
            failed_cookie=expert.current_cookie_index if not success else None,
            failure_phase=failure_phase,
            reason=failure_reason,
            max_gripper_force_n=float(max_force),
            steps=steps,
            cookies_in_target=int(info.get("cookies_in_target", 0)),
            cookies_in_source=int(info.get("cookies_in_source", 0)),
            target_slot_occupancy=[
                int(value) for value in info.get("target_slot_occupancy", ())
            ],
            retries_per_slot=list(expert.retry_counts),
            retry_reasons=list(expert.retry_reasons),
        )
    finally:
        env.close()


def evaluate_cookie_transfer(
    seeds: Iterable[int],
    *,
    max_steps: int = 2500,
    randomize_cookies: bool = True,
) -> dict[str, Any]:
    results = [
        run_cookie_transfer_episode(
            int(seed),
            max_steps=max_steps,
            randomize_cookies=randomize_cookies,
        )
        for seed in seeds
    ]
    successes = sum(result.success for result in results)
    failure_phases = Counter(
        result.failure_phase or "NONE" for result in results if not result.success
    )
    return {
        "summary": {
            "episodes": len(results),
            "successes": successes,
            "success_rate": successes / len(results) if results else 0.0,
            "mean_completed_cookies": float(
                np.mean([result.completed_cookies for result in results])
            )
            if results
            else 0.0,
            "mean_steps": float(np.mean([result.steps for result in results]))
            if results
            else 0.0,
            "failure_phase_counts": dict(sorted(failure_phases.items())),
        },
        "episodes": [result.to_dict() for result in results],
    }
