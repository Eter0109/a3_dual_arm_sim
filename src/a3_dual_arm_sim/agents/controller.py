"""Single-box semantic planning/action/verification control loop.

Simulator truth is an explicitly separate evaluation/labeling dependency. The
visual planner and verifier receive RGB only, never cookie coordinates or IDs.
Simulation pauses during model calls and their latency is reported separately.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from a3_dual_arm_sim.agents.executors import ExpertSkillExecutor, recovery_action
from a3_dual_arm_sim.agents.planning import validate_single_box_plan
from a3_dual_arm_sim.agents.verification import VerificationDecision
from a3_dual_arm_sim.core.contracts import FRONT_IMAGE, LEFT_WRIST_IMAGE
from a3_dual_arm_sim.core.skills import CookieSkill

_EXECUTOR_DIAGNOSTIC_KEYS = (
    "executor_ready_to_switch", "expert_phase", "expert_batch_index",
    "expert_phase_steps", "expert_stable_steps", "expert_completed_cookie_count",
    "expert_failed", "expert_failure_reason",
)


@dataclass(frozen=True)
class AgentLoopConfig:
    verify_every_steps: int = 20
    frame_interval: int = 5
    window_frames: int = 2
    skill_timeout_steps: int = 600
    max_steps: int = 2400
    max_retries: int = 2
    recovery_steps: int = 15
    # Control steps, not wall time: inference deliberately pauses simulation.
    min_stuck_steps: int = 40
    stuck_confirmations: int = 2

    def __post_init__(self) -> None:
        for key in (
            "verify_every_steps", "frame_interval", "window_frames",
            "skill_timeout_steps", "max_steps", "recovery_steps",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.window_frames < 2 or self.max_retries < 0:
            raise ValueError("use at least two temporal frames and a non-negative retry limit")
        if type(self.min_stuck_steps) is not int or self.min_stuck_steps < 0:
            raise ValueError("min_stuck_steps must be a non-negative integer")
        if type(self.stuck_confirmations) is not int or self.stuck_confirmations < 1:
            raise ValueError("stuck_confirmations must be a positive integer")


class TemporalFrameBuffer:
    """Owned, sampled RGB history, cleared at every skill/retry boundary."""

    def __init__(self, *, size: int = 2, interval: int = 5) -> None:
        if size < 2 or interval < 1:
            raise ValueError("invalid temporal buffer settings")
        self.frames: deque[dict[str, np.ndarray]] = deque(maxlen=size)
        self.interval = interval

    def clear(self) -> None:
        self.frames.clear()

    def observe(self, observation: Mapping[str, Any], step: int, *, force: bool = False) -> None:
        if not force and step % self.interval:
            return
        frame = {}
        for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE):
            image = np.asarray(observation[key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{key} must contain a real uint8 RGB image")
            frame[key] = image.copy()
        self.frames.append(frame)


class AgenticController:
    """Execute a validated plan with completion checks, bounded retry and recovery."""

    def __init__(self, planner: Any, verifier: Any, *, config: AgentLoopConfig | None = None):
        self.planner = planner
        self.verifier = verifier
        self.config = config or AgentLoopConfig()

    def run(
        self, env: Any, executor: Any, *, task: str, observation: dict[str, Any],
        seed: int, truth_evaluator: Any | None = None, verification_recorder: Any | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        started = time.monotonic()
        model_seconds = 0.0
        events: list[dict[str, Any]] = []
        plan = []
        frames = TemporalFrameBuffer(size=cfg.window_frames, interval=cfg.frame_interval)
        step = 0
        completed = 0
        info: dict[str, Any] = {}
        failure = ""
        active_truth = None
        try:
            if verification_recorder is not None and (
                getattr(verification_recorder, "max_frames", None) != cfg.window_frames
                or getattr(verification_recorder, "frame_stride", None) != cfg.frame_interval
            ):
                raise ValueError("verification recorder must use the controller's temporal window")
            tick = time.monotonic()
            plan = self.planner.plan(task, {FRONT_IMAGE: observation[FRONT_IMAGE]})
            validate_single_box_plan(plan)
            model_seconds += time.monotonic() - tick
            for skill_index, request in enumerate(plan):
                retries = 0
                # Initial heights must not be redefined after a failed lift.
                if truth_evaluator is not None:
                    truth_evaluator.reset(request)
                while True:
                    executor.start(request)
                    frames.clear()
                    frames.observe(observation, 0, force=True)
                    if verification_recorder is not None:
                        verification_recorder.start(request, seed=seed, attempt=retries)
                        verification_recorder.observe(observation, step=step)
                    events.append({"event": "start", "step": step, "skill_index": skill_index,
                                   "request": asdict(request), "attempt": retries + 1})
                    print(f"step={step} skill={request.skill.value} batch={request.batch_index + 1} "
                          f"attempt={retries + 1}", flush=True)
                    decision = None
                    stuck_streak = 0
                    for skill_step in range(1, cfg.skill_timeout_steps + 1):
                        try:
                            action = executor.act(observation)
                        except RuntimeError as exc:
                            if (isinstance(executor, ExpertSkillExecutor)
                                    and executor.expert.failed
                                    and request.skill is CookieSkill.PICK_FIVE):
                                decision = VerificationDecision("stuck", str(exc))
                                events.append({"event": "executor_failed", "step": step,
                                               "skill_index": skill_index, "reason": str(exc)})
                                break
                            raise
                        observation, _, terminated, truncated, info = env.step(action)
                        step += 1
                        frames.observe(observation, skill_step)
                        if verification_recorder is not None:
                            verification_recorder.observe(observation, step=step)
                        if truth_evaluator is not None:
                            active_truth = truth_evaluator.update()
                        if terminated or truncated:
                            failure = info.get("safety_reason") or "environment terminated/truncated"
                            break
                        if step >= cfg.max_steps:
                            failure = "total step limit reached"
                            break
                        if (skill_step % cfg.verify_every_steps
                                or skill_step % cfg.frame_interval
                                or len(frames.frames) < cfg.window_frames):
                            # Both the RGB window and its separate truth label
                            # must end on this simulator step, not an older image.
                            continue
                        tick = time.monotonic()
                        if getattr(self.verifier, "uses_simulator_truth", False):
                            if active_truth is None:
                                raise ValueError("oracle verifier requires a truth evaluator")
                            decision = self.verifier.check_truth(active_truth)
                        else:
                            decision = self.verifier.check(request, list(frames.frames))
                        model_seconds += time.monotonic() - tick
                        executor_audit = {}
                        diagnose = getattr(executor, "verification_diagnostics", None)
                        if callable(diagnose):
                            diagnostics = diagnose()
                            if isinstance(diagnostics, Mapping):
                                # Executor metadata is strictly a separate audit.
                                # It cannot replace raw model outputs or labels.
                                executor_audit = {
                                    key: diagnostics[key] for key in _EXECUTOR_DIAGNOSTIC_KEYS
                                    if key in diagnostics
                                }
                        events.append({"event": "verify", "step": step, "skill_index": skill_index,
                                       "status": decision.status, "reason": decision.reason,
                                       "truth_complete": (
                                           active_truth.complete if active_truth is not None else None
                                       ), **executor_audit})
                        if verification_recorder is not None and active_truth is not None:
                            verification_recorder.record(
                                active_truth, prediction=decision.status,
                            )
                        if (decision.status == "stuck"
                                and not getattr(self.verifier, "uses_simulator_truth", False)):
                            # Keep the raw model decision above for evaluation.
                            # A short stationary interval during alignment or
                            # finger closing is not sufficient reason to reset.
                            if skill_step < cfg.min_stuck_steps:
                                stuck_streak = 0
                            else:
                                stuck_streak += 1
                            if stuck_streak < cfg.stuck_confirmations:
                                events.append({
                                    "event": "stuck_deferred", "step": step,
                                    "skill_index": skill_index, "skill_step": skill_step,
                                    "stuck_streak": stuck_streak,
                                    "required_confirmations": cfg.stuck_confirmations,
                                    "min_stuck_steps": cfg.min_stuck_steps,
                                })
                                decision = None
                                continue
                        else:
                            # An intervening progress/completion decision breaks
                            # the streak; never delay a completed verdict by age.
                            stuck_streak = 0
                        if decision.status in ("completed", "stuck"):
                            if decision.status == "completed":
                                ready = getattr(executor, "ready_to_switch", lambda: True)()
                                if not ready:
                                    # The physical expert still needs its retreat.
                                    # Keep the raw verifier output in audit logs;
                                    # do not start a new skill inside an old phase.
                                    decision = None
                                    continue
                            break
                    if failure:
                        break
                    if decision is not None and decision.status == "completed":
                        completed += 1
                        events.append({"event": "complete", "step": step,
                                       "skill_index": skill_index})
                        break
                    refusal = getattr(executor, "recovery_refusal_reason", lambda: None)()
                    if refusal is not None:
                        failure = refusal
                        events.append({"event": "recovery_refused", "step": step,
                                       "skill_index": skill_index, "reason": refusal})
                        break
                    if retries >= cfg.max_retries:
                        failure = f"{request.skill.value}: retry limit reached"
                        break
                    retries += 1
                    events.append({"event": "recover", "step": step, "skill_index": skill_index})
                    retreat = recovery_action(env.last_applied_action)
                    for _ in range(cfg.recovery_steps):
                        if step >= cfg.max_steps:
                            failure = "total step limit reached during recovery"
                            break
                        observation, _, terminated, truncated, info = env.step(retreat)
                        step += 1
                        if terminated or truncated:
                            failure = info.get("safety_reason") or "recovery interrupted"
                            break
                    if failure:
                        break
                if failure:
                    break
            # Independent whole-box score, never used to promote a VLM decision.
            success = completed == len(plan) and bool(info.get("success", False))
            if completed == len(plan) and not success:
                failure = "verifier completed the plan but whole-box evaluation failed"
        except (RuntimeError, ValueError, KeyError) as exc:
            success = False
            failure = f"{type(exc).__name__}: {exc}"
        return {
            "seed": seed, "task": task, "success": success, "completed_skills": completed,
            "steps": step, "failure_reason": failure or None, "plan": [asdict(r) for r in plan],
            "events": events, "cookies_in_target": info.get("cookies_in_target", 0),
            "cookies_in_source": info.get("cookies_in_source"),
            "wall_seconds": round(time.monotonic() - started, 3),
            "model_call_seconds": round(model_seconds, 3),
            "verifier_mode": "oracle" if getattr(self.verifier, "uses_simulator_truth", False)
            else "visual", "simulation_pauses_during_model_calls": True,
        }
