"""Benchmark suite for single-box batch cookie transfer.

Provides a standardized 20-episode benchmark evaluating policies or expert models
on the 10-cookie transfer task under slight randomization of source/target boxes
and cookies. Scoring is based on the number of cookies settled in the target box.
"""

from __future__ import annotations

import importlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from a3_dual_arm_sim.core.contracts import ActionMode, EpisodeContext
from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.envs.config import load_config
from a3_dual_arm_sim.envs.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.experts.cross_column import A3CookieBatchExpert
from a3_dual_arm_sim.experts.same_column import A3SameColumnBatchExpert, A3VariedColumnBatchExpert

DEFAULT_CONFIG_PATH = project_root() / "configs" / "envs" / "cookie_batch.yaml"
DEFAULT_TASK_INSTRUCTION = "transfer 10 cookies into target box"
DIVERSE_RANDOMIZATION = {
    "source_bin_noise_m": 0.020,
    "target_bin_noise_m": 0.020,
    "target_bin_yaw_noise_rad": 0.080,
    "cookie_noise_m": 0.0003,
    "cookie_yaw_noise_rad": 0.015,
    "camera_position_noise_m": 0.009,
    "camera_fovy_noise_deg": 2.0,
    "light_noise_fraction": 0.20,
    "color_noise_fraction": 0.15,
}


def column_task_instruction(column_index: int) -> str:
    return (
        f"Transfer 10 cookies from source column {column_index + 1} "
        "into the target box in two batches of five."
    )


@runtime_checkable
class BenchmarkPolicy(Protocol):
    """Protocol for any policy evaluated in the benchmark.

    Compatible with neural network policies (e.g. SmolVLA, OpenVLA, ACT, Diffusion Policy),
    RL agents, or rule-based expert controllers.
    """

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        """Produce an action given the current observation."""
        ...

    def reset(self, context: EpisodeContext | None = None) -> None:
        """Reset the policy state at the beginning of an episode."""
        ...


class ExpertPolicyAdapter:
    """Adapts an expert model to the standard BenchmarkPolicy interface."""

    def __init__(self, expert_factory: Callable[[A3CookieTransferEnv], Any], name: str = "expert"):
        self.expert_factory = expert_factory
        self.name = name
        self.expert: Any | None = None
        self.action_mode: ActionMode = "cartesian_delta"

    def bind_env(self, env: A3CookieTransferEnv) -> None:
        self.expert = self.expert_factory(env)
        self.expert.reset()

    def reset(self, context: EpisodeContext | None = None) -> None:
        if self.expert is not None:
            self.expert.reset(context)

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        if self.expert is None:
            raise RuntimeError("Expert policy must be bound to an environment before acting")
        return self.expert.act(observation, task)

    @property
    def finished(self) -> bool:
        return bool(getattr(self.expert, "finished", False))

    @property
    def phase_name(self) -> str:
        phase = getattr(self.expert, "phase", None)
        return phase.name if phase is not None else ""


def make_policy_adapter(policy_or_spec: Any, env: A3CookieTransferEnv | None = None) -> Any:
    """Create a policy runner from a string name, spec, or object."""
    if isinstance(policy_or_spec, str):
        spec = policy_or_spec.strip().lower()
        if spec in ("same_column", "same-column", "same"):
            adapter = ExpertPolicyAdapter(A3SameColumnBatchExpert, name="same_column")
            if env is not None:
                adapter.bind_env(env)
            return adapter
        if spec in ("varied_column", "varied-column"):
            adapter = ExpertPolicyAdapter(A3VariedColumnBatchExpert, name="varied_column")
            if env is not None:
                adapter.bind_env(env)
            return adapter
        if spec.startswith("column_") and spec[7:] in ("0", "1", "2", "3"):
            selected = int(spec[7:])

            def fixed_column_factory(e: A3CookieTransferEnv) -> A3VariedColumnBatchExpert:
                expert = A3VariedColumnBatchExpert(e)
                expert.requested_source_column_index = selected
                return expert

            adapter = ExpertPolicyAdapter(fixed_column_factory, name=spec)
            if env is not None:
                adapter.bind_env(env)
            return adapter
        if spec in ("cross_column", "cross-column", "cross", "batch"):
            adapter = ExpertPolicyAdapter(A3CookieBatchExpert, name="cross_column")
            if env is not None:
                adapter.bind_env(env)
            return adapter
        # Handle module:factory policy spec
        if ":" in policy_or_spec:
            mod_name, func_name = policy_or_spec.rsplit(":", 1)
            mod = importlib.import_module(mod_name)
            factory = getattr(mod, func_name)
            policy = factory()
            return policy
        raise ValueError(f"Unknown policy specifier: {policy_or_spec}")

    # Callable or class
    if isinstance(policy_or_spec, type):
        if issubclass(policy_or_spec, (A3CookieBatchExpert, A3SameColumnBatchExpert)):
            adapter = ExpertPolicyAdapter(policy_or_spec, name=policy_or_spec.__name__)
            if env is not None:
                adapter.bind_env(env)
            return adapter
        return policy_or_spec()

    # Already an expert instance
    if isinstance(policy_or_spec, (A3CookieBatchExpert, A3SameColumnBatchExpert)):
        adapter = ExpertPolicyAdapter(lambda e: policy_or_spec, name=type(policy_or_spec).__name__)
        adapter.expert = policy_or_spec
        return adapter

    return policy_or_spec


@dataclass
class EpisodeScore:
    """Result of a single benchmark episode."""

    episode: int
    seed: int
    score: int  # Number of cookies settled in small target box (0 - 10)
    max_score: int = 10
    success: bool = False
    steps: int = 0
    wall_seconds: float = 0.0
    cookies_in_target: int = 0
    cookies_in_source: int = 70
    max_cookies_in_target: int = 0
    min_cookies_in_source: int = 80
    target_bin_pos: list[float] = field(default_factory=list)
    source_bin_pos: list[float] = field(default_factory=list)
    source_column_index: int | None = None
    source_column_number: int | None = None
    task_instruction: str = ""
    target_slot_occupancy: list[int] = field(default_factory=list)
    source_initially_filled: bool = False
    target_touches_all_walls: bool = False
    success_hold_count: int = 0
    phase: str = ""
    failure_reason: str | None = None
    failure_phase: str | None = None
    safety_reason: str | None = None
    expert_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expert_failure_diagnostics(env: A3CookieTransferEnv, expert: Any) -> dict[str, Any]:
    """Keep a compact privileged trace for failed expert attempts, not policy input."""
    phase = getattr(expert, "failure_phase", None) or getattr(
        getattr(expert, "phase", None), "name", ""
    )
    result: dict[str, Any] = {
        "batch_index": int(getattr(expert, "batch_index", 0)),
        "batch_indices": [int(index) for index in getattr(expert, "batch_indices", [])],
        "phase_steps": int(
            getattr(expert, "failure_phase_steps", None)
            if getattr(expert, "failure_phase_steps", None) is not None
            else getattr(expert, "phase_steps", 0)
        ),
        "push_stage": getattr(expert, "_push_stage", None) if phase == "ALIGN" else None,
        "push_attempts": int(getattr(expert, "_base_push_attempts", 0)),
        "grasp_tilt_deg": float(getattr(expert, "_grasp_tilt_deg", 0.0)),
        "rear_clearance_m": None,
        "max_pad_force_n": float(getattr(expert, "max_pad_force", 0.0)),
    }
    rear_clearance = float(getattr(expert, "_rear_clearance_m", 0.0))
    if np.isfinite(rear_clearance):
        result["rear_clearance_m"] = rear_clearance
    if result["batch_indices"]:
        try:
            connected, forces = expert._contact_chain()
            result["contact_chain_complete"] = bool(connected)
            result["finger_contact_forces_n"] = np.asarray(forces).round(3).tolist()
            result["total_pad_forces_n"] = np.asarray(
                getattr(expert, "_total_pad_forces", np.zeros(2))
            ).round(3).tolist()
        except (AttributeError, RuntimeError, ValueError, FloatingPointError):
            pass

    target = None
    rotation = None
    if phase in ("APPROACH", "PRE_CLOSE"):
        target = getattr(expert, "_approach_eef", None)
        rotation = getattr(expert, "_grasp_rotation", None)
    elif phase == "ALIGN":
        stage = getattr(expert, "_push_stage", "")
        target = getattr(
            expert,
            {
                "approach": "_push_high",
                "preclose": "_push_high",
                "descend": "_push_start",
                "sweep": "_push_end",
                "retract": "_push_retract",
            }.get(stage, "_push_high"),
            None,
        )
        rotation = getattr(expert, "_push_rotation", getattr(expert, "_canonical", None))
    elif phase in ("DESCEND", "CLOSE"):
        target = getattr(expert, "_pick_eef", None)
        rotation = getattr(expert, "_grasp_rotation", None)
    elif phase == "LIFT":
        target = getattr(expert, "_high_eef", None)
        rotation = getattr(expert, "_grasp_rotation", None)
    elif phase == "RETRACT":
        target = getattr(expert, "_retract_pos", None)
        rotation = getattr(expert, "_retract_rotation", None)

    if target is None or rotation is None:
        return result
    target = np.asarray(target, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(rotation)):
        return result
    try:
        actual = env.data.site_xpos[expert._l_site]
        result["tcp_target_m"] = target.round(5).tolist()
        result["tcp_actual_m"] = actual.round(5).tolist()
        result["tcp_target_error_m"] = float(np.linalg.norm(target - actual))
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        q = env.solve_ik(target, quat, env.current_joint_action[:7])
        work = env._ik._work
        work.qpos[expert._l_qpos] = q
        mujoco.mj_kinematics(env.model, work)
        actual_quat = np.empty(4)
        mujoco.mju_mat2Quat(actual_quat, work.site_xmat[expert._l_site])
        orientation_error = np.empty(3)
        mujoco.mju_subQuat(orientation_error, quat, actual_quat)
        result["ik_position_error_m"] = float(
            np.linalg.norm(work.site_xpos[expert._l_site] - target)
        )
        result["ik_rotation_error_deg"] = float(np.rad2deg(np.linalg.norm(orientation_error)))
    except (AttributeError, RuntimeError, ValueError, FloatingPointError):
        pass
    return result


@dataclass
class BenchmarkResult:
    """Aggregated benchmark metrics across all episodes."""

    policy_name: str
    total_episodes: int
    total_score: int
    max_possible_score: int
    mean_score: float
    min_score: int
    max_score: int
    success_rate: float
    mean_steps: float
    mean_wall_seconds: float
    score_distribution: dict[int, int]
    episodes: list[EpisodeScore]
    policy_rgb_cameras: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "policy_name": self.policy_name,
                "total_episodes": self.total_episodes,
                "total_score": self.total_score,
                "max_possible_score": self.max_possible_score,
                "mean_score": round(self.mean_score, 2),
                "min_score": self.min_score,
                "max_score": self.max_score,
                "success_rate": round(self.success_rate, 4),
                "mean_steps": round(self.mean_steps, 1),
                "mean_wall_seconds": round(self.mean_wall_seconds, 2),
                "score_distribution": self.score_distribution,
                "policy_rgb_cameras": self.policy_rgb_cameras,
            },
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    def summary_table(self) -> str:
        lines = [
            f"=== Benchmark Summary: {self.policy_name} ===",
            f"Total Episodes      : {self.total_episodes}",
            f"Total Score         : {self.total_score} / {self.max_possible_score} ({self.total_score / self.max_possible_score * 100:.1f}%)",
            f"Mean Score / Ep     : {self.mean_score:.2f} / 10.00",
            f"Min / Max Score     : {self.min_score} / {self.max_score}",
            f"Success Rate (10/10): {self.success_rate * 100:.1f}%",
            f"Mean Steps / Ep     : {self.mean_steps:.1f}",
            f"Mean Wall Time / Ep : {self.mean_wall_seconds:.2f}s",
            f"Policy RGB Cameras  : {self.policy_rgb_cameras}",
            f"Score Distribution  : {dict(sorted(self.score_distribution.items(), reverse=True))}",
        ]
        return "\n".join(lines)


class CookieBatchBenchmark:
    """Benchmark runner for single-box batch transfer tasks."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        *,
        max_steps: int = 1000,
        randomize_boxes: bool = True,
        target_bin_noise_m: float = 0.010,
        target_bin_yaw_noise_rad: float = 0.030,
        source_bin_noise_m: float = 0.010,
        randomize_cookies: bool = True,
        cookie_noise_m: float = 0.0003,
        cookie_yaw_noise_rad: float = 0.015,
        camera_position_noise_m: float = 0.0,
        camera_fovy_noise_deg: float = 0.0,
        light_noise_fraction: float = 0.0,
        color_noise_fraction: float = 0.0,
        render: bool = False,
    ):
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.max_steps = max_steps
        self.randomize_boxes = randomize_boxes
        self.target_bin_noise_m = target_bin_noise_m
        self.target_bin_yaw_noise_rad = target_bin_yaw_noise_rad
        self.source_bin_noise_m = source_bin_noise_m
        self.randomize_cookies = randomize_cookies
        self.cookie_noise_m = cookie_noise_m
        self.cookie_yaw_noise_rad = cookie_yaw_noise_rad
        self.camera_position_noise_m = camera_position_noise_m
        self.camera_fovy_noise_deg = camera_fovy_noise_deg
        self.light_noise_fraction = light_noise_fraction
        self.color_noise_fraction = color_noise_fraction
        self.render = render

        self.config = load_config(self.config_path)

    def create_env(self, *, render_cameras: bool = False) -> A3CookieTransferEnv:
        return A3CookieTransferEnv(
            config=self.config,
            task_config=CookieTransferTaskConfig(
                cookie_count=80,
                required_cookies=10,
                position_noise_m=self.cookie_noise_m,
                yaw_noise_rad=self.cookie_yaw_noise_rad,
            ),
            render_mode="human" if self.render else None,
            render_cameras=render_cameras,
        )

    @staticmethod
    def policy_uses_cameras(policy: Any) -> bool:
        """Render RGB for external policies, while leaving expert rollouts fast."""
        if isinstance(policy, str):
            # Built-in names are privileged-state experts; module:factory is external.
            return ":" in policy
        if isinstance(policy, ExpertPolicyAdapter):
            return False
        if isinstance(policy, (A3CookieBatchExpert, A3SameColumnBatchExpert)):
            return False
        if isinstance(policy, type) and issubclass(
            policy, (A3CookieBatchExpert, A3SameColumnBatchExpert)
        ):
            return False
        return bool(getattr(policy, "requires_camera_rendering", True))

    def run_episode(
        self,
        policy: Any,
        seed: int,
        episode_idx: int = 1,
        env: A3CookieTransferEnv | None = None,
        recorder: Any | None = None,
    ) -> EpisodeScore:
        should_close_env = False
        needs_cameras = recorder is not None or self.policy_uses_cameras(policy)
        if env is None:
            env = self.create_env(render_cameras=needs_cameras)
            should_close_env = True
        elif needs_cameras and getattr(env, "render_cameras", True) is False:
            raise ValueError(
                "This policy or recorder requires RGB cameras, but the supplied environment "
                "has render_cameras=False. Create it with render_cameras=True."
            )

        reset_options = {
            "randomize_cookies": self.randomize_cookies,
            "randomize_boxes": self.randomize_boxes,
            "randomize_source_bin": self.randomize_boxes,
            "randomize_target_bin": self.randomize_boxes,
            "source_bin_noise_m": self.source_bin_noise_m,
            "target_bin_noise_m": self.target_bin_noise_m,
            "target_bin_yaw_noise_rad": self.target_bin_yaw_noise_rad,
            "camera_position_noise_m": self.camera_position_noise_m,
            "camera_fovy_noise_deg": self.camera_fovy_noise_deg,
            "light_noise_fraction": self.light_noise_fraction,
            "color_noise_fraction": self.color_noise_fraction,
        }

        observation, info = env.reset(seed=seed, options=reset_options)

        try:
            runner_policy = make_policy_adapter(policy, env=env)
            if hasattr(runner_policy, "bind_env") and getattr(runner_policy, "expert", None) is None:
                runner_policy.bind_env(env)
            if hasattr(runner_policy, "reset"):
                context = EpisodeContext(
                    seed=seed, task=DEFAULT_TASK_INSTRUCTION, action_mode=env.action_mode
                )
                runner_policy.reset(context)
        except RuntimeError as exc:
            if "unreachable with vertical grasp" not in str(exc):
                if should_close_env:
                    env.close()
                raise
            if should_close_env:
                env.close()
            return EpisodeScore(
                episode=episode_idx,
                seed=seed,
                score=0,
                success=False,
                cookies_in_target=int(info.get("cookies_in_target", 0)),
                cookies_in_source=int(info.get("cookies_in_source", 80)),
                task_instruction=DEFAULT_TASK_INSTRUCTION,
                failure_reason=f"expert_preflight: {exc}",
                failure_phase="PREFLIGHT",
            )

        selected_column = getattr(
            getattr(runner_policy, "expert", None), "selected_source_column_index", None
        )
        task_instruction = (
            DEFAULT_TASK_INSTRUCTION
            if selected_column is None
            else column_task_instruction(selected_column)
        )

        t_start = time.monotonic()
        if recorder is not None:
            recorder.start_episode(
                EpisodeContext(
                    seed=seed,
                    task=task_instruction,
                    action_mode=runner_policy.action_mode,
                ),
                getattr(runner_policy, "name", type(runner_policy).__name__),
            )
        steps = 0
        terminated = False
        truncated = False
        max_cookies_in_target = int(info.get("cookies_in_target", 0))
        min_cookies_in_source = int(info.get("cookies_in_source", 80))
        skill_recording = recorder is not None and hasattr(recorder, "add_expert_frame")
        expert = getattr(runner_policy, "expert", None)

        try:
            if skill_recording:
                if expert is None:
                    raise TypeError("skill recording requires an expert with phase annotations")
                from a3_dual_arm_sim.data.skills import label_expert_step
            while steps < self.max_steps:
                steps += 1
                if skill_recording:
                    phase_before = expert.phase
                    batch_index_before = expert.batch_index
                if hasattr(runner_policy, "act"):
                    try:
                        action = runner_policy.act(
                            observation, task_instruction
                        )
                    except TypeError:
                        action = runner_policy.act(observation)
                elif callable(runner_policy):
                    action = runner_policy(observation)
                else:
                    raise ValueError(
                        f"Policy {runner_policy} does not have act() and is not callable"
                    )

                previous_observation = observation
                observation, _, terminated, truncated, info = env.step(action)
                max_cookies_in_target = max(
                    max_cookies_in_target, int(info.get("cookies_in_target", 0))
                )
                min_cookies_in_source = min(
                    min_cookies_in_source, int(info.get("cookies_in_source", 80))
                )
                if recorder is not None:
                    if skill_recording:
                        label = label_expert_step(
                            phase_before,
                            batch_index_before,
                            expert.batch_indices,
                            env.SOURCE_POSITIONS,
                        )
                        recorder.add_expert_frame(
                            previous_observation, info["applied_action"], label
                        )
                    else:
                        recorder.add_frame(previous_observation, info["applied_action"])

                if getattr(runner_policy, "finished", False) or terminated or truncated:
                    break
        except BaseException:
            if recorder is not None:
                recorder.discard_episode()
            if should_close_env:
                env.close()
            raise

        wall_time = time.monotonic() - t_start
        cookies_in_target = int(info.get("cookies_in_target", 0))
        cookies_in_source = int(info.get("cookies_in_source", 0))
        env_success = bool(info.get("success", False))
        success = env_success and cookies_in_target == 10
        if recorder is not None:
            success = success and cookies_in_source == 70 and not info.get("safety_reason")
            if success:
                recorder.finish_episode(success=True)
            else:
                recorder.discard_episode()

        target_pos = (
            env.data.xpos[env._target_bin_body].round(4).tolist()
            if hasattr(env, "_target_bin_body")
            else []
        )
        source_pos = (
            env.data.xpos[env._source_bin_body][:2].round(4).tolist()
            if hasattr(env, "_source_bin_body")
            else []
        )

        phase = getattr(runner_policy, "phase_name", "")
        failure_reason = None
        failure_phase = None
        diagnostics: dict[str, Any] = {}
        if not success:
            if expert is not None:
                failure_phase = getattr(expert, "failure_phase", None) or phase
                diagnostics = _expert_failure_diagnostics(env, expert)
            if terminated:
                failure_reason = "environment_safety_terminated"
            elif truncated or steps >= self.max_steps:
                failure_reason = "max_steps_exceeded"
            elif getattr(runner_policy, "expert", None) is not None and getattr(
                runner_policy.expert, "failed", False
            ):
                failure_reason = getattr(runner_policy.expert, "failure_reason", "expert_failed")
            elif cookies_in_target == 10 and cookies_in_source != 70:
                failure_reason = f"source box contains {cookies_in_source}/70 remaining cookies"
            elif cookies_in_target == 10 and not info.get("exact_2x5_fill", False):
                failure_reason = (
                    "target box is not an exact stable 2x5 fill; "
                    f"slots={list(info.get('target_slot_occupancy', []))}, "
                    f"walls={info.get('target_touches_all_walls', False)}"
                )
            elif cookies_in_target == 10:
                failure_reason = (
                    "target fill did not remain stable for the required hold; "
                    f"hold={info.get('success_hold_count', 0)}"
                )
            else:
                failure_reason = f"only {cookies_in_target}/10 cookies in target box"

        if should_close_env:
            env.close()

        return EpisodeScore(
            episode=episode_idx,
            seed=seed,
            score=cookies_in_target,
            max_score=10,
            success=success,
            steps=steps,
            wall_seconds=round(wall_time, 2),
            cookies_in_target=cookies_in_target,
            cookies_in_source=cookies_in_source,
            max_cookies_in_target=max_cookies_in_target,
            min_cookies_in_source=min_cookies_in_source,
            target_bin_pos=target_pos,
            source_bin_pos=source_pos,
            source_column_index=selected_column,
            source_column_number=None if selected_column is None else selected_column + 1,
            task_instruction=task_instruction,
            target_slot_occupancy=list(info.get("target_slot_occupancy", [])),
            source_initially_filled=bool(info.get("source_initially_filled", False)),
            target_touches_all_walls=bool(info.get("target_touches_all_walls", False)),
            success_hold_count=int(info.get("success_hold_count", 0)),
            phase=phase,
            failure_reason=failure_reason,
            failure_phase=failure_phase,
            safety_reason=info.get("safety_reason"),
            expert_diagnostics=diagnostics,
        )

    def evaluate(
        self,
        policy: Any,
        num_episodes: int = 20,
        seed_start: int = 0,
        policy_name: str | None = None,
        verbose: bool = True,
        workers: int = 1,
        on_episode_end: Callable[[EpisodeScore], None] | None = None,
    ) -> BenchmarkResult:
        """Run standard benchmark for the specified number of episodes (default: 20)."""
        if policy_name is None:
            if isinstance(policy, str):
                policy_name = policy
            else:
                policy_name = getattr(policy, "name", type(policy).__name__)

        if verbose:
            print("\n=======================================================", flush=True)
            print(f" Starting Benchmark: {policy_name}", flush=True)
            print(
                f" Episodes: {num_episodes}, Seeds: {seed_start} to {seed_start + num_episodes - 1}",
                flush=True,
            )
            print(
                f" Randomization: boxes={self.randomize_boxes}, cookies={self.randomize_cookies}, workers={workers}",
                flush=True,
            )
            print(f" Policy RGB cameras: {self.policy_uses_cameras(policy)}", flush=True)
            print("=======================================================", flush=True)

        episode_results: list[EpisodeScore] = []

        if workers > 1 and num_episodes > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            policy_spec = policy if isinstance(policy, str) else policy_name
            worker_args = [
                {
                    "config_path": str(self.config_path),
                    "policy": policy_spec,
                    "seed": seed_start + ep_i,
                    "episode_idx": ep_i + 1,
                    "max_steps": self.max_steps,
                    "randomize_boxes": self.randomize_boxes,
                    "target_bin_noise_m": self.target_bin_noise_m,
                    "target_bin_yaw_noise_rad": self.target_bin_yaw_noise_rad,
                    "source_bin_noise_m": self.source_bin_noise_m,
                    "randomize_cookies": self.randomize_cookies,
                    "cookie_noise_m": self.cookie_noise_m,
                    "cookie_yaw_noise_rad": self.cookie_yaw_noise_rad,
                    "camera_position_noise_m": self.camera_position_noise_m,
                    "camera_fovy_noise_deg": self.camera_fovy_noise_deg,
                    "light_noise_fraction": self.light_noise_fraction,
                    "color_noise_fraction": self.color_noise_fraction,
                }
                for ep_i in range(num_episodes)
            ]
            with ProcessPoolExecutor(max_workers=min(workers, num_episodes)) as executor:
                future_to_idx = {
                    executor.submit(_run_single_episode_worker, arg): i
                    for i, arg in enumerate(worker_args)
                }
                results_by_idx: dict[int, EpisodeScore] = {}
                for future in as_completed(future_to_idx):
                    ep_res = future.result()
                    results_by_idx[ep_res.episode - 1] = ep_res
                    if on_episode_end is not None:
                        on_episode_end(ep_res)
                    if verbose:
                        status_str = "SUCCESS" if ep_res.success else "FAIL"
                        print(
                            f"[{len(results_by_idx):02d}/{num_episodes:02d}] Finished Seed {ep_res.seed:4d} | "
                            f"Score: {ep_res.score:2d}/10 | "
                            f"Steps: {ep_res.steps:4d} | "
                            f"Time: {ep_res.wall_seconds:5.2f}s | "
                            f"Status: {status_str}",
                            flush=True,
                        )
                episode_results = [results_by_idx[i] for i in range(num_episodes)]
        else:
            # A model checkpoint is expensive to load; share one instance across
            # serial episodes and let run_episode() reset its action queue each time.
            owns_policy = isinstance(policy, str) and ":" in policy
            episode_policy = make_policy_adapter(policy) if owns_policy else policy
            env = self.create_env(render_cameras=self.policy_uses_cameras(episode_policy))
            try:
                for ep_i in range(num_episodes):
                    seed = seed_start + ep_i
                    ep_result = self.run_episode(
                        policy=episode_policy,
                        seed=seed,
                        episode_idx=ep_i + 1,
                        env=env,
                    )
                    episode_results.append(ep_result)

                    if on_episode_end is not None:
                        on_episode_end(ep_result)

                    if verbose:
                        status_str = "SUCCESS" if ep_result.success else "FAIL"
                        print(
                            f"[{ep_result.episode:02d}/{num_episodes:02d}] Seed {seed:4d} | "
                            f"Score: {ep_result.score:2d}/10 | "
                            f"Steps: {ep_result.steps:4d} | "
                            f"Time: {ep_result.wall_seconds:5.2f}s | "
                            f"Status: {status_str}",
                            flush=True,
                        )
            finally:
                env.close()
                if owns_policy and hasattr(episode_policy, "close"):
                    episode_policy.close()

        total_score = sum(r.score for r in episode_results)
        max_possible = num_episodes * 10
        successes = sum(1 for r in episode_results if r.success)
        scores = [r.score for r in episode_results]
        steps = [r.steps for r in episode_results]
        wall_times = [r.wall_seconds for r in episode_results]
        score_dist = dict(Counter(scores))

        result = BenchmarkResult(
            policy_name=policy_name,
            total_episodes=num_episodes,
            total_score=total_score,
            max_possible_score=max_possible,
            mean_score=float(np.mean(scores)) if scores else 0.0,
            min_score=min(scores) if scores else 0,
            max_score=max(scores) if scores else 0,
            success_rate=successes / num_episodes if num_episodes > 0 else 0.0,
            mean_steps=float(np.mean(steps)) if steps else 0.0,
            mean_wall_seconds=float(np.mean(wall_times)) if wall_times else 0.0,
            score_distribution=score_dist,
            episodes=episode_results,
            policy_rgb_cameras=self.policy_uses_cameras(policy),
        )

        if verbose:
            print("\n" + result.summary_table() + "\n", flush=True)

        return result


def _run_single_episode_worker(args: dict[str, Any]) -> EpisodeScore:
    benchmark = CookieBatchBenchmark(
        config_path=args["config_path"],
        max_steps=args["max_steps"],
        randomize_boxes=args["randomize_boxes"],
        target_bin_noise_m=args["target_bin_noise_m"],
        target_bin_yaw_noise_rad=args["target_bin_yaw_noise_rad"],
        source_bin_noise_m=args["source_bin_noise_m"],
        randomize_cookies=args["randomize_cookies"],
        cookie_noise_m=args["cookie_noise_m"],
        cookie_yaw_noise_rad=args["cookie_yaw_noise_rad"],
        camera_position_noise_m=args["camera_position_noise_m"],
        camera_fovy_noise_deg=args["camera_fovy_noise_deg"],
        light_noise_fraction=args["light_noise_fraction"],
        color_noise_fraction=args["color_noise_fraction"],
        render=False,
    )
    return benchmark.run_episode(
        policy=args["policy"],
        seed=args["seed"],
        episode_idx=args["episode_idx"],
    )


def run_cookie_batch_benchmark(
    policy: Any = "same_column",
    *,
    num_episodes: int = 20,
    seed_start: int = 0,
    max_steps: int = 1000,
    randomize_boxes: bool = True,
    randomize_cookies: bool = True,
    workers: int = 1,
    output_path: Path | str | None = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Convenience function to run the benchmark and optionally save JSON."""
    benchmark = CookieBatchBenchmark(
        max_steps=max_steps,
        randomize_boxes=randomize_boxes,
        randomize_cookies=randomize_cookies,
    )
    result = benchmark.evaluate(
        policy=policy,
        num_episodes=num_episodes,
        seed_start=seed_start,
        workers=workers,
        verbose=verbose,
    )
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"Saved benchmark results to {p}", flush=True)
    return result
