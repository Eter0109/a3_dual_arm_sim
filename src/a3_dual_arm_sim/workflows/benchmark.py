# Explicit same-name imports preserve the legacy public API.
# ruff: noqa: PLC0414
"""Benchmark suite for single-box batch cookie transfer.

Provides a standardized 20-episode benchmark evaluating policies or expert models
on the 10-cookie transfer task under slight randomization of source/target boxes
and cookies. Scoring is based on the number of cookies settled in the target box.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.sim.randomization import (
    load_randomization_config,
    require_basic_expanded_profile,
)
from a3_dual_arm_sim.tasks.cookie_2x10 import SOURCE_TASKS, Cookie2x10TaskConfig
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_expanded_randomization
from a3_dual_arm_sim.tasks.cookie_scenarios import (
    SCENARIO_2X10_BASE,
    SCENARIO_2X10_TASKS,
    SCENARIO_PROMPTS,
    SCENARIO_TASKS,
    A3CookieScenarioEnv,
)
from a3_dual_arm_sim.tasks.cookie_transfer import CookieTransferTaskConfig

from .benchmark_results import BenchmarkResult as BenchmarkResult
from .benchmark_results import EpisodeScore as EpisodeScore
from .cookie_2x10_execution import Cookie2x10EpisodeExecution
from .course_protocol import COURSE_PROMPT, COURSE_PROTOCOL, LEGACY_PROTOCOL, course_max_steps
from .episode_execution import DEFAULT_CONFIG_PATH as DEFAULT_CONFIG_PATH
from .episode_execution import EpisodeExecution as EpisodeExecution
from .policy_adapters import ACTPolicyAdapter as ACTPolicyAdapter
from .policy_adapters import BenchmarkPolicy as BenchmarkPolicy
from .policy_adapters import ExpertPolicyAdapter as ExpertPolicyAdapter
from .policy_adapters import PI05PolicyAdapter as PI05PolicyAdapter
from .policy_adapters import SmolVLAPolicyAdapter as SmolVLAPolicyAdapter
from .policy_adapters import make_policy_adapter as make_policy_adapter
from .policy_adapters import policy_needs_cameras
from .policy_adapters import resolve_smolvla_checkpoint as resolve_smolvla_checkpoint


class CookieBatchBenchmark(EpisodeExecution):
    """Benchmark runner for single-box batch transfer tasks."""

    task = "2x5"
    # 2x5 training data and every evaluation CLI default to the original short
    # instruction; the per-column "task" prompt is opt-in via --prompt-style task.
    prompt_style = "training"
    diagnostic_dir = None

    ALIGNED_PI05_PROMPT = (
        "grasp five yellow blocks together from the source bin, insert them upright into "
        "the marked slots of the 2 by 5 box, then return to the source bin, grasp five "
        "more, and fill all ten slots"
    )

    def _episode_task(self, column, seed):
        if self.protocol == COURSE_PROTOCOL:
            return COURSE_PROMPT
        if self.prompt_style == "pi05_aligned":
            return self.ALIGNED_PI05_PROMPT
        if self.prompt_style in ("training", "short"):
            return "transfer 10 cookies into target box"
        return super()._episode_task(column, seed)

    def _episode_metadata(self, column, seed):
        return {
            **super()._episode_metadata(column, seed),
            "prompt_style": self.prompt_style,
            "task_instruction": self._episode_task(column, seed),
        }

    def run_episode(self, policy, seed, episode_idx=1, **kwargs):
        if policy == "expert":
            policy = ("course_expert" if self.protocol == COURSE_PROTOCOL and self.task in SCENARIO_TASKS
                      else "variable_grasp" if self.task in ("2x10", *SOURCE_TASKS) else "same_column")
        if self.diagnostic_dir is None:
            return super().run_episode(policy, seed, episode_idx, **kwargs)
        from .diagnostics import EpisodeDiagnostic

        trace = EpisodeDiagnostic(self.diagnostic_dir / f"seed_{seed}", self.config.control_hz)
        try:
            result = super().run_episode(policy, seed, episode_idx, diagnostic=trace, **kwargs)
            (trace.root / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
            return result
        finally:
            trace.close()

    def evaluate(
        self,
        policy: Any,
        num_episodes: int | None = None,
        seed_start: int = 0,
        policy_name: str | None = None,
        verbose: bool = True,
        workers: int = 1,
        on_episode_end: Callable[[EpisodeScore], None] | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark; course defaults to 5 episodes and legacy to 20."""
        if num_episodes is None:
            num_episodes = 5 if self.protocol == COURSE_PROTOCOL else 20
        if num_episodes < 1 or workers < 1:
            raise ValueError("num_episodes and workers must be positive")
        if policy == "expert":
            policy = ("course_expert" if self.protocol == COURSE_PROTOCOL and self.task in SCENARIO_TASKS
                      else "variable_grasp" if self.task in ("2x10", *SOURCE_TASKS) else "same_column")
        if not isinstance(policy, str) and workers > 1:
            raise ValueError("Policy instances require workers=1")
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
            print("=======================================================", flush=True)

        episode_results: list[EpisodeScore] = []

        if workers > 1 and num_episodes > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            policy_spec = policy if isinstance(policy, str) else policy_name
            worker_args = [
                {
                    "config_path": str(self.config_path),
                    "complete_demonstrations": self.complete_demonstrations,
                    "strict_policy_completion": self.strict_policy_completion,
                    "placement_order": self.placement_order,
                    "task": self.task,
                    "protocol": self.protocol,
                    "prompt_style": self.prompt_style,
                    "first_grasp": getattr(self, "first_grasp", None),
                    "tilt_level": getattr(self, "tilt_level", None),
                    "box_quota": getattr(self, "box_quota", None),
                    "success_hold_steps": self.success_hold_steps,
                    "slot_tolerance_m": self.config.cookie_transfer.target_slot_tolerance_m,
                    "wall_contact_tolerance_m": self.wall_contact_tolerance_m,
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
                    "profile": self.profile,
                    "source_column": self.source_column,
                    "randomization_settings": self.randomization_settings,
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
                            f"Score: {ep_res.score:2d}/{ep_res.max_score} | "
                            f"Steps: {ep_res.steps:4d} | "
                            f"Time: {ep_res.wall_seconds:5.2f}s | "
                            f"Status: {status_str}",
                            flush=True,
                        )
                episode_results = [results_by_idx[i] for i in range(num_episodes)]
        else:
            runner_policy = None if policy in ("variable_grasp", "course_expert") else make_policy_adapter(policy)
            needs_cameras = (
                self.diagnostic_dir is not None
                or policy_needs_cameras(runner_policy)
            )
            env = None
            try:
                env = self.create_env(render_cameras=needs_cameras)
                for ep_i in range(num_episodes):
                    seed = seed_start + ep_i
                    ep_result = self.run_episode(
                        policy=policy
                        if isinstance(policy, str) and policy in ("same_column", "variable_grasp", "course_expert")
                        else runner_policy,
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
                            f"Score: {ep_result.score:2d}/{ep_result.max_score} | "
                            f"Steps: {ep_result.steps:4d} | "
                            f"Time: {ep_result.wall_seconds:5.2f}s | "
                            f"Status: {status_str}",
                            flush=True,
                        )
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    if isinstance(policy, str) and hasattr(runner_policy, "close"):
                        runner_policy.close()

        total_score = sum(r.score for r in episode_results)
        max_possible = sum(r.max_score for r in episode_results)
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
        )

        if verbose:
            print("\n" + result.summary_table() + "\n", flush=True)

        return result


class Cookie2x10Benchmark(Cookie2x10EpisodeExecution, CookieBatchBenchmark):
    """The 20-cookie task with the same evaluate/policy interface as 2x5."""

    task = "2x10"
    # The 2x10 default stays the native four-grasp plan prompt; only the 2x5
    # default converges on the short training instruction.
    prompt_style = "task"

    def _episode_task(self, column, seed):
        if self.protocol == COURSE_PROTOCOL:
            return COURSE_PROMPT
        if self.prompt_style == "short":
            return "transfer 20 cookies into target box"
        return super()._episode_task(column, seed)


def _expanded_basic_kwargs(task, kwargs):
    """Select basic without mutating the caller's configuration snapshot."""
    kwargs = dict(kwargs)
    requested = kwargs.get("profile")
    if requested is not None:
        require_basic_expanded_profile(requested, task)
    kwargs["profile"] = "basic"

    path = kwargs.get("randomization_config")
    if path is not None and task in SCENARIO_TASKS:
        if kwargs.get("randomization_settings") is not None:
            raise ValueError("provide either randomization_config or randomization_settings")
        settings = (load_expanded_randomization(path) if task in SCENARIO_2X10_TASKS
                    else load_randomization_config(path))
        # EpisodeExecution normally takes the column from a config file.
        kwargs["source_column"] = settings["source_column"]
        kwargs.pop("randomization_config")
        kwargs["randomization_settings"] = settings
    if kwargs.get("randomization_settings") is not None:
        settings = load_randomization_config(config=kwargs["randomization_settings"])
        if "basic" not in settings["profiles"]:
            raise ValueError(f"expanded task {task!r} requires a basic randomization profile")
        settings["profile"] = "basic"
        kwargs["randomization_settings"] = settings
    return kwargs


class CookieScenarioBenchmark(CookieBatchBenchmark):
    """Additional initial-state and multi-target benchmark tasks."""

    prompt_style = "task"

    def __init__(self, scenario: str, *, tilt_level: str | None = None,
                 box_quota: int | None = None, **kwargs):
        self.task = scenario
        self.base_scenario = SCENARIO_2X10_BASE.get(scenario, scenario)
        if scenario in SCENARIO_2X10_TASKS:
            kwargs.setdefault("config_path", resource_root() / "configs" / "cookie_2x10.yaml")
        self.tilt_level = (tilt_level or "mild") if self.base_scenario == "tilted" else None
        self.box_quota = (5 if box_quota is None else box_quota) if scenario == "two-boxes" else None
        self.required_cookies = (2 * self.box_quota if self.box_quota is not None
                                 else 5 if scenario == "prefilled"
                                 else 20 if scenario in SCENARIO_2X10_TASKS else 10)
        super().__init__(**_expanded_basic_kwargs(scenario, kwargs))
        self.randomization_settings["profile"] = "basic"
        # Use the validated 2x10 source lattice, shifted 16 mm towards the
        # reachable side of the workspace, for default expanded 2x5 scenes.
        # A caller-supplied scene remains authoritative; its source geometry
        # must still satisfy the clearance checks at reset.
        scene = self.config.cookie_transfer
        default_scene = (resource_root() / "configs" / "cookie_2x10.yaml"
                         if scenario in SCENARIO_2X10_TASKS else DEFAULT_CONFIG_PATH)
        if self.config_path.resolve() == default_scene.resolve():
            source = load_config(resource_root() / "configs" / "cookie_2x10.yaml").cookie_transfer
            shift_x = -0.016
            scene = replace(
                scene,
                source_bin_center_m=(
                    source.source_bin_center_m[0] + shift_x,
                    source.source_bin_center_m[1],
                ),
                cookie_source_positions_m=tuple(
                    (x + shift_x, y) for x, y in source.cookie_source_positions_m
                ),
            )
            if self.base_scenario == "sparse-exact-ten":
                # All ten pieces are required but are distributed across four
                # columns. Leave room for individual placements and retreat
                # instead of requiring a five-piece grasp at the target.
                rows = sorted({y for _, y in scene.target_slots_local_m})
                expected_rows = 10 if scenario in SCENARIO_2X10_TASKS else 5
                if len(rows) != expected_rows:
                    raise ValueError(f"{scenario} requires {expected_rows} target rows")
                row_y = {y: (index - (expected_rows - 1) / 2) * 0.012
                         for index, y in enumerate(rows)}
                scene = replace(
                    scene,
                    target_bin_half_size_m=(scene.target_bin_half_size_m[0],
                                            0.070 if expected_rows == 10 else 0.040),
                    target_slots_local_m=tuple(
                        (x, row_y[y]) for x, y in scene.target_slots_local_m
                    ),
                )
            if scenario == "rotated-target":
                x, y, z = scene.target_bin_world_position_m
                scene = replace(scene, target_bin_world_position_m=(x - 0.09, y, z))
        if scenario == "two-boxes":
            x, y, z = scene.target_bin_world_position_m
            scene = replace(
                scene,
                target_bin_world_position_m=(x, y + 0.04, z),
                spare_target_bin_world_position_m=(x - 0.025, y - 0.055, z),
            )
        self.config = replace(self.config, cookie_transfer=scene)

    def create_env(self, *, render_cameras=False):
        task_config = Cookie2x10TaskConfig if self.task in SCENARIO_2X10_TASKS else CookieTransferTaskConfig
        return A3CookieScenarioEnv(
            self.config,
            scenario=self.task,
            tilt_level=self.tilt_level,
            box_quota=self.box_quota,
            task_config=task_config(
                cookie_count=len(self.config.cookie_transfer.cookie_source_positions_m),
                position_noise_m=self.cookie_noise_m,
                yaw_noise_rad=self.cookie_yaw_noise_rad,
                terminate_on_success=False,
                require_released=True,
                target_only=self.protocol == COURSE_PROTOCOL,
                success_hold_steps=self.success_hold_steps,
                wall_contact_tolerance_m=self.wall_contact_tolerance_m,
            ),
            render_mode="human" if self.render else None,
            render_cameras=render_cameras,
        )

    def _episode_task(self, column, seed):
        if self.protocol == COURSE_PROTOCOL:
            return COURSE_PROMPT
        if self.task == "tilted" and self.tilt_level == "fallen":
            return ("Some cookies are lying flat. Transfer 10 available cookies from the "
                    "source box into the target box, upright.")
        if self.task == "two-boxes" and self.box_quota == 10:
            return ("Transfer 20 cookies from the source box and fill both target boxes, "
                    "placing 10 upright in each 2 by 5 box.")
        if self.task in SCENARIO_2X10_TASKS:
            return "Transfer 20 cookies into the 2x10 target box."
        return SCENARIO_PROMPTS[self.task]

    def _episode_policy(self, policy, column, seed):
        if isinstance(policy, str) and policy == "course_expert":
            from a3_dual_arm_sim.controllers.course_expert import (
                A3Course2x5Expert,
                A3Course2x10Expert,
                A3CourseSparse2x10Expert,
            )
            if self.task == "2x10-sparse-exact-twenty":
                factory = lambda env: A3CourseSparse2x10Expert(env, first_grasp=5)
            elif self.base_scenario == "sparse-exact-ten":
                from a3_dual_arm_sim.controllers.course_sparse_expert import A3CourseSparseExpert
                factory = A3CourseSparseExpert
            else:
                factory = (lambda env: A3Course2x10Expert(env, first_grasp=5)) if (
                    self.task in SCENARIO_2X10_TASKS
                ) else A3Course2x5Expert
            return ExpertPolicyAdapter(factory, name="course_privileged_expert")
        return super()._episode_policy(policy, column, seed)

    def _expected_source_count(self, env):
        return env.initial_source_count - env.goal_count

    def _describe_incomplete_fill(self, info):
        metrics = info.get("scenario_metrics", {})
        return (f"scenario incomplete: targets={metrics.get('target_counts')}, "
                f"source={metrics.get('cookies_in_source')}/"
                f"{metrics.get('expected_source_count')}, "
                f"outside={metrics.get('cookies_outside_containers')}")


def make_cookie_benchmark(task="2x5", *, prompt_style=None, first_grasp=None, tilt_level=None,
                          box_quota=None, protocol=None, **kwargs):
    """Shared construction for the benchmark CLI and grid evaluations."""
    if task not in ("2x5", "2x10", *SOURCE_TASKS, *SCENARIO_TASKS):
        raise ValueError(f"unknown benchmark task: {task}")
    if protocol is None:
        protocol = (LEGACY_PROTOCOL if task in SOURCE_TASKS
                    or prompt_style not in (None, "course") or tilt_level == "fallen"
                    or box_quota == 10 else COURSE_PROTOCOL)
    if protocol not in (COURSE_PROTOCOL, LEGACY_PROTOCOL):
        raise ValueError(f"unknown benchmark protocol: {protocol}")
    if task in SOURCE_TASKS and protocol != LEGACY_PROTOCOL:
        raise ValueError("source quota tasks require the legacy protocol")
    if protocol == COURSE_PROTOCOL and prompt_style not in (None, "course"):
        raise ValueError("course-v1 requires the unified course prompt")
    if task in SOURCE_TASKS and prompt_style not in (None, "task"):
        raise ValueError("source quota tasks require the task prompt")
    if tilt_level is not None and (task not in ("tilted", "2x10-tilted") or tilt_level not in ("mild", "fallen")):
        raise ValueError("tilt_level must be mild or fallen and only applies to tilted")
    if box_quota is not None and (task != "two-boxes" or box_quota not in (5, 10)):
        raise ValueError("box_quota must be 5 or 10 and only applies to two-boxes")
    if protocol == COURSE_PROTOCOL:
        prompt_style = "course"
        kwargs.setdefault("max_steps", course_max_steps(task))
        if task == "2x5" and kwargs.get("profile") is None and kwargs.get("randomization_config") is None:
            settings = load_randomization_config(config=kwargs.get("randomization_settings"))
            kwargs["profile"] = settings["profile"]
    elif prompt_style is None:
        # Match the per-task CLI defaults: 2x5 evaluation and collection use the
        # short training instruction, 2x10 keeps its native plan prompt.
        prompt_style = "task" if task in ("2x10", *SOURCE_TASKS) or task in SCENARIO_TASKS else "training"
    if prompt_style not in ("task", "training", "short", "pi05_aligned", "course"):
        raise ValueError("prompt_style must be task, training, short or pi05_aligned")
    if task in SCENARIO_TASKS:
        if first_grasp is not None:
            raise ValueError("first_grasp only applies to 2x10")
        if prompt_style not in ("task", "course"):
            raise ValueError("scenario benchmarks require prompt_style='task'")
        benchmark = CookieScenarioBenchmark(
            scenario=task, tilt_level=tilt_level, box_quota=box_quota, protocol=protocol, **kwargs
        )
    elif task in ("2x10", *SOURCE_TASKS):
        if task in SOURCE_TASKS:
            kwargs = _expanded_basic_kwargs(task, kwargs)
        benchmark = Cookie2x10Benchmark(first_grasp=first_grasp, protocol=protocol, **kwargs)
        benchmark.task = task
        if task in SOURCE_TASKS:
            benchmark.randomization_settings["profile"] = "basic"
    else:
        if first_grasp is not None:
            raise ValueError("first_grasp only applies to 2x10")
        benchmark = CookieBatchBenchmark(protocol=protocol, **kwargs)
    benchmark.prompt_style = prompt_style
    return benchmark


def _run_single_episode_worker(args: dict[str, Any]) -> EpisodeScore:
    benchmark = make_cookie_benchmark(
        task=args.get("task", "2x5"),
        protocol=args.get("protocol", LEGACY_PROTOCOL),
        prompt_style=args.get("prompt_style"),
        first_grasp=args.get("first_grasp"),
        tilt_level=args.get("tilt_level"),
        box_quota=args.get("box_quota"),
        success_hold_steps=args.get("success_hold_steps", 5),
        slot_tolerance_m=args.get("slot_tolerance_m", (0.015, 0.005)),
        wall_contact_tolerance_m=args.get("wall_contact_tolerance_m", 0.030),
        config_path=args["config_path"],
        complete_demonstrations=args.get("complete_demonstrations", False),
        strict_policy_completion=args.get("strict_policy_completion", False),
        placement_order=args.get("placement_order", "canonical"),
        max_steps=args["max_steps"],
        randomize_boxes=args["randomize_boxes"],
        target_bin_noise_m=args["target_bin_noise_m"],
        target_bin_yaw_noise_rad=args["target_bin_yaw_noise_rad"],
        source_bin_noise_m=args["source_bin_noise_m"],
        randomize_cookies=args["randomize_cookies"],
        cookie_noise_m=args["cookie_noise_m"],
        cookie_yaw_noise_rad=args["cookie_yaw_noise_rad"],
        render=False,
        profile=args.get("profile"),
        source_column=args.get("source_column"),
        randomization_settings=args.get("randomization_settings"),
    )
    return benchmark.run_episode(
        policy=args["policy"],
        seed=args["seed"],
        episode_idx=args["episode_idx"],
    )


def run_cookie_batch_benchmark(
    policy: Any = "expert",
    *,
    task: str = "2x5",
    num_episodes: int | None = None,
    seed_start: int = 0,
    max_steps: int | None = None,
    randomize_boxes: bool = True,
    randomize_cookies: bool = True,
    workers: int = 1,
    output_path: Path | str | None = None,
    verbose: bool = True,
    **benchmark_options: Any,
) -> BenchmarkResult:
    """Convenience function to run the benchmark and optionally save JSON."""
    if max_steps is not None:
        benchmark_options["max_steps"] = max_steps
    benchmark = make_cookie_benchmark(
        task=task,
        **benchmark_options,
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
