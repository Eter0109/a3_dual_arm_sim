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
                    "profile": self.profile,
                    "source_column": self.source_column,
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
            runner_policy = make_policy_adapter(policy)
            needs_cameras = getattr(runner_policy, "action_mode", "") == "joint_position"
            env = None
            try:
                env = self.create_env(render_cameras=needs_cameras)
                for ep_i in range(num_episodes):
                    seed = seed_start + ep_i
                    ep_result = self.run_episode(
                        policy=runner_policy,
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
                try:
                    if env is not None:
                        env.close()
                finally:
                    if isinstance(policy, str) and hasattr(runner_policy, "close"):
                        runner_policy.close()

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
        render=False,
        profile=args.get("profile"),
        source_column=args.get("source_column"),
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
