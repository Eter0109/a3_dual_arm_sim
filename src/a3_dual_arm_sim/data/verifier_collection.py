"""Controlled simulation failures for temporal-verifier supervision only.

Case names identify interventions, NOT truth labels. All labels are computed by
the independent GroundTruthSkillEvaluator from the actual simulated outcome.
Injected controls and privileged diagnostics belong in audit files only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest

FAILURE_CASES = ("miss_grasp", "drop", "stuck", "natural")


def hold_cartesian_action(
    joint_action: np.ndarray, *, left_opening: float | None = None
) -> np.ndarray:
    """Hold both TCPs; preserve actual gripper commands unless explicitly opening."""
    joint = np.asarray(joint_action, dtype=np.float64)
    if joint.shape != (16,) or not np.all(np.isfinite(joint)):
        raise ValueError("hold requires a finite A3 16D joint action")
    if left_opening is not None and not 0 <= left_opening <= 1:
        raise ValueError("left_opening must be in [0, 1]")
    action = np.zeros(14, dtype=np.float64)
    action[6] = 2 * (float(joint[7]) if left_opening is None else left_opening) - 1
    action[13] = 2 * float(joint[15]) - 1
    return action


class SyntheticFailureInjection:
    """Deterministic command interventions; never assign completion/diagnosis."""

    def __init__(self, case: str) -> None:
        if case not in FAILURE_CASES:
            raise ValueError(f"case must be one of {FAILURE_CASES}")
        self.case = case
        self.synthetic = case != "natural"
        self.active = False
        self.trigger_step: int | None = None

    @property
    def bypass_expert(self) -> bool:
        # Once frozen or prematurely released, keep physics evolving without
        # advancing the expert's internal state as though its commands executed.
        return self.active and self.case in ("stuck", "drop")

    def apply(
        self,
        action: np.ndarray,
        *,
        request: SkillRequest,
        expert_phase: str,
        joint_action: np.ndarray,
        step: int,
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        command = np.asarray(action, dtype=np.float64).copy()
        if command.shape != (14,) or not np.all(np.isfinite(command)):
            raise ValueError("failure injections require finite cartesian_delta actions")
        trigger = (
            (
                self.case == "miss_grasp"
                and request.skill is CookieSkill.PICK_FIVE
                and expert_phase in ("CLOSE", "VERIFY_GRASP", "LIFT", "VERIFY_LIFT")
            )
            or (
                self.case == "stuck"
                and request.skill is CookieSkill.PICK_FIVE
                and expert_phase == "DESCEND"
            )
            or (self.case == "drop" and request.skill is CookieSkill.PLACE_FIVE)
        )
        event = None
        if trigger and not self.active:
            self.active = True
            self.trigger_step = step
            descriptions = {
                "miss_grasp": "keep left gripper open instead of closing",
                "stuck": "freeze both TCP commands before descent",
                "drop": "open left gripper after verified lift, before transport",
            }
            event = {
                "event": "synthetic_control_injection",
                "case": self.case,
                "synthetic": True,
                "step": step,
                "phase": expert_phase,
                "control_change": descriptions[self.case],
            }
        if self.active:
            if self.case == "miss_grasp":
                command[6] = 1.0
            elif self.case == "stuck":
                command = hold_cartesian_action(joint_action)
            elif self.case == "drop":
                command = hold_cartesian_action(joint_action, left_opening=1.0)
        return command, event


def run_failure_attempt(
    env: Any,
    executor: Any,
    evaluator: Any,
    recorder: Any,
    *,
    observation: dict[str, Any],
    case: str,
    seed: int,
    source_column_number: int = 1,
    max_steps: int = 600,
    record_every_steps: int = 20,
    post_fault_steps: int = 80,
    skill_hold_steps: int = 0,
    terminal_hold_steps: int = 0,
    render_recorded_frames_only: bool = False,
    audit_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Collect one attempt, optionally rendering only real recorder sample frames.

    The supplied reset observation must contain real RGB. Physics, expert
    commands and oracle updates still execute every step. Camera flags are
    restored even if simulation, labeling or writing raises an exception.
    """
    if type(render_recorded_frames_only) is not bool:
        raise TypeError("render_recorded_frames_only must be a boolean")
    kwargs = {
        "observation": observation,
        "case": case,
        "seed": seed,
        "source_column_number": source_column_number,
        "max_steps": max_steps,
        "record_every_steps": record_every_steps,
        "post_fault_steps": post_fault_steps,
        "skill_hold_steps": skill_hold_steps,
        "terminal_hold_steps": terminal_hold_steps,
        "render_recorded_frames_only": render_recorded_frames_only,
        "audit_sink": audit_sink,
    }
    if not render_recorded_frames_only:
        return _run_failure_attempt(env, executor, evaluator, recorder, **kwargs)
    original_render_cameras = getattr(env, "render_cameras", None)
    if type(original_render_cameras) is not bool:
        raise TypeError("strided camera rendering requires env.render_cameras to be a boolean")
    try:
        env.render_cameras = True
        return _run_failure_attempt(env, executor, evaluator, recorder, **kwargs)
    finally:
        env.render_cameras = original_render_cameras


def _run_failure_attempt(
    env: Any,
    executor: Any,
    evaluator: Any,
    recorder: Any,
    *,
    observation: dict[str, Any],
    case: str,
    seed: int,
    source_column_number: int = 1,
    max_steps: int = 600,
    record_every_steps: int = 20,
    post_fault_steps: int = 80,
    skill_hold_steps: int = 0,
    terminal_hold_steps: int = 0,
    render_recorded_frames_only: bool = False,
    audit_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Collect one bounded attempt with real RGB and oracle-derived labels.

    Synthetic cases target only the first batch. ``natural`` has no intervention
    and may complete the whole box or expose a naturally occurring expert error.
    An optional final skill hold records later stationary image windows without
    assuming success persists: each tail label still comes from the live oracle.
    Optional non-final skill holds stop the attempt if real completion is lost;
    the held skill is counted only after the requested, frame-aligned hold.
    The caller creates/resets the environment and owns all resource cleanup.
    """
    if min(max_steps, record_every_steps, post_fault_steps) < 1:
        raise ValueError("collection step limits must be positive")
    if type(terminal_hold_steps) is not int or terminal_hold_steps < 0:
        raise ValueError("terminal_hold_steps must be a non-negative integer")
    if type(skill_hold_steps) is not int or skill_hold_steps < 0:
        raise ValueError("skill_hold_steps must be a non-negative integer")
    injection = SyntheticFailureInjection(case)
    requests = [
        SkillRequest(skill, batch, source_column_number, batch + 1)
        for batch in range(2)
        for skill in (CookieSkill.PICK_FIVE, CookieSkill.PLACE_FIVE)
    ]
    skill_index = 0
    request = requests[skill_index]
    executor.start(request)
    evaluator.reset(request)
    recorder.start(request, seed=seed, attempt=0)
    recorder.observe(observation, step=0)
    samples = Counter()
    truth_reasons = Counter()
    truth_diagnoses = Counter()
    completed = 0
    natural_error_step = None
    executor_error = None
    stop_reason = "step_limit"
    fault_positions: np.ndarray | None = None
    height_loss_m = 0.0
    labels_after_injection = Counter()
    terminal_hold_started_step: int | None = None
    actual_terminal_hold_steps = 0
    terminal_hold_completed = False
    terminal_hold_labels = Counter()
    skill_hold_started_step: int | None = None
    current_skill_hold_steps = 0
    actual_skill_hold_steps = 0
    skill_hold_reports: list[dict[str, Any]] = []
    skill_hold_labels = Counter()
    started = time.monotonic()
    last_recorded_step: int | None = None
    latest_image_step = 0
    frame_stride = int(getattr(recorder, "frame_stride", 1))
    info: dict[str, Any] = {}

    def emit(event):
        if audit_sink is not None:
            audit_sink(
                {"parent_seed": seed, "case": case, "synthetic": injection.synthetic, **event}
            )

    def record_truth(truth, step):
        nonlocal last_recorded_step
        if last_recorded_step == step or latest_image_step != step:
            return
        sample_index = recorder.record(truth)  # Never override truth diagnosis.
        if sample_index is None:
            return
        last_recorded_step = step
        label = "Yes" if truth.complete else "No"
        samples[label] += 1
        if not truth.complete:
            samples[truth.diagnosis or "StillTrying"] += 1
        if injection.active:
            labels_after_injection[label] += 1
        if terminal_hold_started_step is not None and step > terminal_hold_started_step:
            terminal_hold_labels[label] += 1
            if not truth.complete:
                terminal_hold_labels[truth.diagnosis or "StillTrying"] += 1
        if skill_hold_started_step is not None and step > skill_hold_started_step:
            skill_hold_labels[label] += 1
            if not truth.complete:
                skill_hold_labels[truth.diagnosis or "StillTrying"] += 1
        emit(
            {
                "event": "sample_label",
                "step": step,
                "sample_index": sample_index,
                "request": asdict(request),
                "truth": asdict(truth),
                "injection_active": injection.active,
            }
        )

    step = 0
    for step in range(1, max_steps + 1):
        phase = executor.expert.phase.name
        if (
            terminal_hold_started_step is not None
            or skill_hold_started_step is not None
            or injection.bypass_expert
            or executor_error is not None
        ):
            action = hold_cartesian_action(env.last_applied_action)
        else:
            try:
                action = executor.act(observation)
            except RuntimeError as exc:
                executor_error = str(exc)
                natural_error_step = step
                emit(
                    {
                        "event": "expert_error",
                        "step": step,
                        "reason": executor_error,
                        "after_injection": injection.active,
                    }
                )
                action = hold_cartesian_action(env.last_applied_action)
        action, fault = injection.apply(
            action,
            request=request,
            expert_phase=phase,
            joint_action=env.last_applied_action,
            step=step,
        )
        if fault is not None:
            fault_positions = np.asarray(
                [env.privileged_cookie_position(index) for index in evaluator.cookie_indices],
                dtype=np.float64,
            )
            emit({**fault, "cookie_positions_at_injection_m": fault_positions.tolist()})
        if render_recorded_frames_only:
            env.render_cameras = step - latest_image_step >= frame_stride
        observation, _, terminated, truncated, info = env.step(action)
        if terminal_hold_started_step is not None:
            actual_terminal_hold_steps += 1
        if skill_hold_started_step is not None:
            current_skill_hold_steps += 1
            actual_skill_hold_steps += 1
        recorder.observe(observation, step=step)
        if step - latest_image_step >= frame_stride:
            latest_image_step = step
        truth = evaluator.update()
        truth_reasons[truth.reason] += 1
        truth_diagnoses[truth.diagnosis or "Completed"] += 1
        if fault_positions is not None and case == "drop":
            positions = np.asarray(
                [env.privileged_cookie_position(index) for index in evaluator.cookie_indices],
                dtype=np.float64,
            )
            # Median actual world-height loss is an audit observation, not a
            # verifier label. PLACE completion still comes from the full oracle.
            height_loss_m = max(
                height_loss_m, float(np.median(fault_positions[:, 2] - positions[:, 2]))
            )
        if (
            step % record_every_steps == 0
            or truth.complete
            or terminal_hold_started_step is not None
            or skill_hold_started_step is not None
        ):
            record_truth(truth, step)
        if terminated or truncated:
            record_truth(truth, step)
            stop_reason = info.get("safety_reason") or "environment_terminated_or_truncated"
            break
        if injection.active and step - int(injection.trigger_step) >= post_fault_steps:
            record_truth(truth, step)
            stop_reason = "post_injection_window_finished"
            break
        if natural_error_step is not None and step - natural_error_step >= post_fault_steps:
            record_truth(truth, step)
            stop_reason = "post_expert_error_window_finished"
            break
        skill_hold_finished_this_step = False
        if skill_hold_started_step is not None:
            if not truth.complete:
                record_truth(truth, step)
                stop_reason = "skill_hold_completion_lost"
                skill_hold_reports.append(
                    {
                        "request": asdict(request),
                        "started_step": skill_hold_started_step,
                        "finished_step": step,
                        "actual_hold_steps": current_skill_hold_steps,
                        "completed": False,
                        "stop_reason": stop_reason,
                        "final_truth_complete": False,
                        "final_truth_reason": truth.reason,
                    }
                )
                emit(
                    {
                        "event": "skill_hold_completion_lost",
                        "step": step,
                        "request": asdict(request),
                        "truth": asdict(truth),
                        "actual_hold_steps": current_skill_hold_steps,
                    }
                )
                break
            if current_skill_hold_steps < skill_hold_steps or latest_image_step != step:
                continue
            record_truth(truth, step)
            skill_hold_reports.append(
                {
                    "request": asdict(request),
                    "started_step": skill_hold_started_step,
                    "finished_step": step,
                    "actual_hold_steps": current_skill_hold_steps,
                    "completed": True,
                    "stop_reason": "skill_hold_window_finished",
                    "final_truth_complete": True,
                    "final_truth_reason": truth.reason,
                }
            )
            emit(
                {
                    "event": "skill_hold_finished",
                    "step": step,
                    "request": asdict(request),
                    "actual_hold_steps": current_skill_hold_steps,
                    "truth_complete": truth.complete,
                }
            )
            skill_hold_started_step = None
            skill_hold_finished_this_step = True
        if terminal_hold_started_step is not None:
            if actual_terminal_hold_steps >= terminal_hold_steps and latest_image_step == step:
                record_truth(truth, step)
                terminal_hold_completed = True
                stop_reason = "terminal_hold_window_finished"
                emit(
                    {
                        "event": "terminal_hold_finished",
                        "step": step,
                        "actual_terminal_hold_steps": actual_terminal_hold_steps,
                        "truth_complete": truth.complete,
                    }
                )
                break
            # The final skill was already counted once. Never switch, restart,
            # or count it again while physics settles under the safe hold.
            continue
        # Do not advance a faulted attempt on a scenario assumption: even an
        # unexpectedly successful fault remains labelled Yes by the evaluator.
        if (
            truth.complete
            and latest_image_step == step
            and not injection.active
            and executor.ready_to_switch()
        ):
            if (
                skill_hold_steps > 0
                and skill_index < len(requests) - 1
                and not skill_hold_finished_this_step
            ):
                skill_hold_started_step = step
                current_skill_hold_steps = 0
                emit(
                    {
                        "event": "skill_hold_started",
                        "step": step,
                        "request": asdict(request),
                        "requested_skill_hold_steps": skill_hold_steps,
                    }
                )
                continue
            completed += 1
            skill_index += 1
            if skill_index == len(requests):
                if terminal_hold_steps == 0:
                    stop_reason = "all_skills_completed"
                    break
                terminal_hold_started_step = step
                emit(
                    {
                        "event": "terminal_hold_started",
                        "step": step,
                        "requested_terminal_hold_steps": terminal_hold_steps,
                    }
                )
                continue
            request = requests[skill_index]
            executor.start(request)
            evaluator.reset(request)
            recorder.start(request, seed=seed, attempt=0)
            recorder.observe(observation, step=step)
            latest_image_step = step
            emit({"event": "skill_started", "step": step, "request": asdict(request)})
    if evaluator.latest is not None:
        record_truth(evaluator.latest, step)
    if skill_hold_started_step is not None and stop_reason != "skill_hold_completion_lost":
        skill_hold_reports.append(
            {
                "request": asdict(request),
                "started_step": skill_hold_started_step,
                "finished_step": step,
                "actual_hold_steps": current_skill_hold_steps,
                "completed": False,
                "stop_reason": stop_reason,
                "final_truth_complete": bool(evaluator.latest.complete)
                if evaluator.latest
                else None,
                "final_truth_reason": evaluator.latest.reason if evaluator.latest else None,
            }
        )
    result = {
        "seed": seed,
        "case": case,
        "synthetic": injection.synthetic,
        "source_column_number": source_column_number,
        "render_recorded_frames_only": render_recorded_frames_only,
        "steps": step,
        "completed_skills": completed,
        "initialization_failed": False,
        "visual_labels_recorded": bool(samples),
        "requested_skill_hold_steps": skill_hold_steps,
        "actual_skill_hold_steps": actual_skill_hold_steps,
        "skill_hold_completed_count": sum(report["completed"] for report in skill_hold_reports),
        "skill_hold_reports": skill_hold_reports,
        "skill_hold_label_counts": dict(skill_hold_labels),
        "requested_terminal_hold_steps": terminal_hold_steps,
        "actual_terminal_hold_steps": actual_terminal_hold_steps,
        "terminal_hold_started_step": terminal_hold_started_step,
        "terminal_hold_completed": terminal_hold_completed,
        "terminal_hold_label_counts": dict(terminal_hold_labels),
        "terminal_hold_final_truth_complete": (
            bool(evaluator.latest.complete)
            if terminal_hold_started_step is not None and evaluator.latest is not None
            else None
        ),
        "fault_triggered": injection.active,
        "fault_trigger_step": injection.trigger_step,
        "expert_error": executor_error,
        "stop_reason": stop_reason,
        "sample_counts": dict(samples),
        "post_injection_completion_labels": dict(labels_after_injection),
        "actual_truth_reasons": dict(truth_reasons),
        "actual_truth_diagnosis_counts": dict(truth_diagnoses),
        "observed_drop_median_height_loss_m": height_loss_m if case == "drop" else None,
        "observed_drop_height_loss_over_30mm": height_loss_m > 0.030 if case == "drop" else None,
        "cookies_in_target": info.get("cookies_in_target"),
        "whole_box_success": bool(info.get("success", False)),
        "wall_seconds": round(time.monotonic() - started, 3),
        "labels_from_actual_simulator_truth": True,
    }
    emit({"event": "attempt_finished", **result})
    return result


def write_collection_summary(root: Path, results: list[dict]) -> dict:
    """Persist collection results and real label counts, not just console output."""
    counts = {"completed": 0, "continuing": 0, "stuck": 0}
    samples_path = root / "samples.jsonl"
    if samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            label = json.loads(line)["label"]
            outcome = (
                "completed"
                if label["completion"] == "Yes"
                else ("stuck" if label["diagnosis"] == "Stuck" else "continuing")
            )
            counts[outcome] += 1
    summary = {
        "schema_version": 1,
        "collection_status": "complete",
        "root": str(root),
        "attempts": len(results),
        "training_started": False,
        "sample_class_counts": counts,
        "results": results,
    }
    (root / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def collect_verifier_dataset(args: argparse.Namespace) -> int:
    """Collect independently labeled verifier windows without parsing arguments."""
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    from a3_dual_arm_sim.agents.executors import ExpertSkillExecutor
    from a3_dual_arm_sim.data.verifier_recording import TemporalVerificationRecorder
    from a3_dual_arm_sim.envs.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
    from a3_dual_arm_sim.evaluation.benchmark import DIVERSE_RANDOMIZATION, CookieBatchBenchmark
    from a3_dual_arm_sim.evaluation.skill_truth import GroundTruthSkillEvaluator

    if min(args.max_steps, args.post_fault_steps, args.record_every, args.frame_stride) < 1:
        raise ValueError("step limits and frame stride must be positive")
    if args.window_frames < 2:
        raise ValueError("temporal verification requires at least two frames")
    if args.terminal_hold_steps < 0:
        raise ValueError("terminal_hold_steps must be non-negative")
    skill_hold_steps = getattr(args, "skill_hold_steps", 0)
    if type(skill_hold_steps) is not int or skill_hold_steps < 0:
        raise ValueError("skill_hold_steps must be a non-negative integer")
    render_recorded_frames_only = getattr(args, "render_recorded_frames_only", False)
    if type(render_recorded_frames_only) is not bool:
        raise TypeError("render_recorded_frames_only must be a boolean")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("do not repeat a case in one dataset root")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        raise ValueError("do not repeat a seed in one dataset root")
    root = args.root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(
            f"Use a new failure dataset root; refusing implicit append/overwrite: {root}"
        )
    root.mkdir(parents=True)
    benchmark = CookieBatchBenchmark(args.config, max_steps=args.max_steps)
    recorder = TemporalVerificationRecorder(
        root, max_frames=args.window_frames, frame_stride=args.frame_stride
    )
    manifest = {
        "schema_version": 1,
        "purpose": "temporal_verifier_failure_smoke_collection",
        "seeds": seeds,
        "cases": args.cases,
        "source_column_number": args.source_column,
        "profile": args.profile,
        "task": "Transfer ten cookies into the single small box in two batches of five.",
        "config": asdict(benchmark.config),
        "max_steps_per_case": args.max_steps,
        "post_fault_steps": args.post_fault_steps,
        "window_frames": args.window_frames,
        "terminal_hold_steps": args.terminal_hold_steps,
        "skill_hold_steps": skill_hold_steps,
        "render_recorded_frames_only": render_recorded_frames_only,
        "frame_stride": args.frame_stride,
        "record_every_steps": args.record_every,
        "model_input": "samples.jsonl.input: RGB image paths and skill subgoal only",
        "labels": "independent simulator truth; never inferred from case names",
        "synthetic_audit": "fault_audit.jsonl; do not use as model inputs",
        "action_dataset": False,
        "fine_tuning_started": False,
    }
    (root / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    env = None
    executor = None
    results = []
    with (root / "fault_audit.jsonl").open("a", encoding="utf-8") as audit:

        def write_audit(event):
            audit.write(json.dumps(event, ensure_ascii=False) + "\n")
            audit.flush()

        def save_attempt(result):
            results.append(result)
            with (root / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)

        try:
            env = A3CookieTransferEnv(
                benchmark.config,
                task_config=CookieTransferTaskConfig(
                    require_exact_slots=False,
                    require_released=True,
                    terminate_on_success=False,
                ),
                render_cameras=True,
            )
            for seed in seeds:
                for case in args.cases:
                    if render_recorded_frames_only:
                        # Every case starts with a freshly rendered reset frame.
                        env.render_cameras = True
                    reset_options = {"randomize_cookies": False, "randomize_boxes": False}
                    if args.profile == "diverse":
                        reset_options = {
                            "randomize_cookies": True,
                            "randomize_boxes": True,
                            **DIVERSE_RANDOMIZATION,
                        }
                    observation, _ = env.reset(
                        seed=seed,
                        options=reset_options,
                    )
                    try:
                        executor = ExpertSkillExecutor(
                            env,
                            seed=seed,
                            source_column_number=args.source_column,
                        )
                    except RuntimeError as exc:
                        if "unreachable with vertical grasp" not in str(exc):
                            raise
                        # Initialization never started a skill or observed a
                        # verifier window. Preserve the failed attempt, not a
                        # fabricated visual failure label, then reset next case.
                        result = {
                            "seed": seed,
                            "case": case,
                            "synthetic": case != "natural",
                            "source_column_number": args.source_column,
                            "render_recorded_frames_only": render_recorded_frames_only,
                            "steps": 0,
                            "completed_skills": 0,
                            "initialization_failed": True,
                            "stop_reason": "expert_initialization_failed",
                            "expert_error": str(exc),
                            "sample_counts": {},
                            "fault_triggered": False,
                            "fault_trigger_step": None,
                            "requested_skill_hold_steps": skill_hold_steps,
                            "actual_skill_hold_steps": 0,
                            "skill_hold_completed_count": 0,
                            "skill_hold_reports": [],
                            "skill_hold_label_counts": {},
                            "requested_terminal_hold_steps": args.terminal_hold_steps,
                            "actual_terminal_hold_steps": 0,
                            "terminal_hold_completed": False,
                            "terminal_hold_started_step": None,
                            "terminal_hold_label_counts": {},
                            "terminal_hold_final_truth_complete": None,
                            "cookies_in_target": None,
                            "whole_box_success": False,
                            "labels_from_actual_simulator_truth": True,
                            "visual_labels_recorded": False,
                        }
                        write_audit(
                            {"event": "expert_initialization_failed", "parent_seed": seed, **result}
                        )
                        save_attempt(result)
                        continue
                    evaluator = GroundTruthSkillEvaluator(env)
                    result = run_failure_attempt(
                        env,
                        executor,
                        evaluator,
                        recorder,
                        observation=observation,
                        case=case,
                        seed=seed,
                        source_column_number=args.source_column,
                        max_steps=args.max_steps,
                        record_every_steps=args.record_every,
                        post_fault_steps=args.post_fault_steps,
                        audit_sink=write_audit,
                        terminal_hold_steps=args.terminal_hold_steps,
                        skill_hold_steps=skill_hold_steps,
                        render_recorded_frames_only=render_recorded_frames_only,
                    )
                    save_attempt(result)
                    executor.close()
                    executor = None
        finally:
            if executor is not None:
                executor.close()
            recorder.close()
            if env is not None:
                env.close()
    summary = write_collection_summary(root, results)
    print(json.dumps(summary, indent=2), flush=True)
    return 0
