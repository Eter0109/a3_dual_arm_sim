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
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_results import BenchmarkResult as BenchmarkResult
from .benchmark_results import EpisodeScore as EpisodeScore
from .cookie_2x10_execution import Cookie2x10EpisodeExecution
from .episode_execution import DEFAULT_CONFIG_PATH as DEFAULT_CONFIG_PATH
from .episode_execution import EpisodeExecution as EpisodeExecution
from .policy_adapters import ACTPolicyAdapter as ACTPolicyAdapter
from .policy_adapters import BenchmarkPolicy as BenchmarkPolicy
from .policy_adapters import ExpertPolicyAdapter as ExpertPolicyAdapter
from .policy_adapters import SmolVLAPolicyAdapter as SmolVLAPolicyAdapter
from .policy_adapters import make_policy_adapter as make_policy_adapter
from .policy_adapters import resolve_smolvla_checkpoint as resolve_smolvla_checkpoint


class CookieBatchBenchmark(EpisodeExecution):
    """Benchmark runner for single-box batch transfer tasks."""

    task = "2x5"
    # 2x5 training data and every evaluation CLI default to the original short
    # instruction; the per-column "task" prompt is opt-in via --prompt-style task.
    prompt_style = "training"
    diagnostic_dir = None

    def _episode_task(self, column, seed):
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
            policy = "variable_grasp" if self.task == "2x10" else "same_column"
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
        num_episodes: int = 20,
        seed_start: int = 0,
        policy_name: str | None = None,
        verbose: bool = True,
        workers: int = 1,
        on_episode_end: Callable[[EpisodeScore], None] | None = None,
    ) -> BenchmarkResult:
        """Run standard benchmark for the specified number of episodes (default: 20)."""
        if num_episodes < 1 or workers < 1:
            raise ValueError("num_episodes and workers must be positive")
        if policy == "expert":
            policy = "variable_grasp" if self.task == "2x10" else "same_column"
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
                    "task": self.task,
                    "prompt_style": self.prompt_style,
                    "first_grasp": getattr(self, "first_grasp", None),
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
            runner_policy = None if policy == "variable_grasp" else make_policy_adapter(policy)
            needs_cameras = (
                self.diagnostic_dir is not None
                or getattr(runner_policy, "action_mode", "") == "joint_position"
            )
            env = None
            try:
                env = self.create_env(render_cameras=needs_cameras)
                for ep_i in range(num_episodes):
                    seed = seed_start + ep_i
                    ep_result = self.run_episode(
                        policy=policy
                        if isinstance(policy, str) and policy in ("same_column", "variable_grasp")
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
        if self.prompt_style == "short":
            return "transfer 20 cookies into target box"
        return super()._episode_task(column, seed)


def make_cookie_benchmark(task="2x5", *, prompt_style=None, first_grasp=None, **kwargs):
    """Shared construction for the benchmark CLI and grid evaluations."""
    if task not in ("2x5", "2x10"):
        raise ValueError("task must be 2x5 or 2x10")
    if prompt_style is None:
        # Match the per-task CLI defaults: 2x5 evaluation and collection use the
        # short training instruction, 2x10 keeps its native plan prompt.
        prompt_style = "task" if task == "2x10" else "training"
    if prompt_style not in ("task", "training", "short"):
        raise ValueError("prompt_style must be task, training or short")
    if task == "2x10":
        benchmark = Cookie2x10Benchmark(first_grasp=first_grasp, **kwargs)
    else:
        if first_grasp is not None:
            raise ValueError("first_grasp only applies to 2x10")
        benchmark = CookieBatchBenchmark(**kwargs)
    benchmark.prompt_style = prompt_style
    return benchmark


def _run_single_episode_worker(args: dict[str, Any]) -> EpisodeScore:
    benchmark = make_cookie_benchmark(
        task=args.get("task", "2x5"),
        prompt_style=args.get("prompt_style"),
        first_grasp=args.get("first_grasp"),
        success_hold_steps=args.get("success_hold_steps", 5),
        slot_tolerance_m=args.get("slot_tolerance_m", (0.015, 0.005)),
        wall_contact_tolerance_m=args.get("wall_contact_tolerance_m", 0.030),
        config_path=args["config_path"],
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
    num_episodes: int = 20,
    seed_start: int = 0,
    max_steps: int = 1000,
    randomize_boxes: bool = True,
    randomize_cookies: bool = True,
    workers: int = 1,
    output_path: Path | str | None = None,
    verbose: bool = True,
    **benchmark_options: Any,
) -> BenchmarkResult:
    """Convenience function to run the benchmark and optionally save JSON."""
    benchmark = make_cookie_benchmark(
        task=task,
        **benchmark_options,
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
