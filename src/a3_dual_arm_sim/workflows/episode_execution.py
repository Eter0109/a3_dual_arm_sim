"""Benchmark suite for single-box batch cookie transfer.

Provides a standardized 20-episode benchmark evaluating policies or expert models
on the 10-cookie transfer task under slight randomization of source/target boxes
and cookies. Scoring is based on the number of cookies settled in the target box.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig

from .benchmark_results import EpisodeScore
from .policy_adapters import (
    make_policy_adapter,
)

DEFAULT_CONFIG_PATH = resource_root() / "configs" / "cookie_batch.yaml"


class EpisodeExecution:
    """Single-episode environment lifecycle and execution."""

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

    def run_episode(
        self,
        policy: Any,
        seed: int,
        episode_idx: int = 1,
        env: A3CookieTransferEnv | None = None,
        recorder: Any | None = None,
        diagnostic: Any | None = None,
    ) -> EpisodeScore:
        runner_policy = make_policy_adapter(policy, env=env)
        should_close_env = False
        if env is None:
            needs_cameras = (recorder is not None or diagnostic is not None) or (
                getattr(runner_policy, "action_mode", "") == "joint_position"
            )
            env = self.create_env(render_cameras=needs_cameras)
            should_close_env = True

        reset_options = {
            "randomize_cookies": self.randomize_cookies,
            "randomize_boxes": self.randomize_boxes,
            "randomize_source_bin": self.randomize_boxes,
            "randomize_target_bin": self.randomize_boxes,
            "source_bin_noise_m": self.source_bin_noise_m,
            "target_bin_noise_m": self.target_bin_noise_m,
            "target_bin_yaw_noise_rad": self.target_bin_yaw_noise_rad,
        }

        observation, info = env.reset(seed=seed, options=reset_options)

        if hasattr(runner_policy, "bind_env") and getattr(runner_policy, "expert", None) is None:
            runner_policy.bind_env(env)

        if hasattr(runner_policy, "reset"):
            context = EpisodeContext(
                seed=seed, task="transfer 10 cookies into target box", action_mode=env.action_mode
            )
            runner_policy.reset(context)

        t_start = time.monotonic()
        if recorder is not None:
            recorder.start_episode(
                EpisodeContext(
                    seed=seed,
                    task="transfer 10 cookies into target box",
                    action_mode=runner_policy.action_mode,
                ),
                getattr(runner_policy, "name", type(runner_policy).__name__),
            )
        steps = 0
        terminated = False
        truncated = False

        try:
            while steps < self.max_steps:
                steps += 1
                if hasattr(runner_policy, "act"):
                    try:
                        action = runner_policy.act(
                            observation, "transfer 10 cookies into target box"
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
                if diagnostic is not None:
                    plugin = getattr(runner_policy, "plugin", runner_policy)
                    diagnostic.add_frame(
                        previous_observation,
                        action,
                        info,
                        raw_action=getattr(plugin, "last_raw_action", None),
                        phase=getattr(runner_policy, "phase_name", ""),
                    )
                if recorder is not None:
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
        success = (
            env_success
            and cookies_in_target == 10
            and cookies_in_source == 70
            and not info.get("safety_reason")
        )
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
        if not success:
            if terminated:
                failure_reason = "environment_safety_terminated"
            elif truncated or steps >= self.max_steps:
                failure_reason = "max_steps_exceeded"
            elif getattr(runner_policy, "expert", None) is not None and getattr(
                runner_policy.expert, "failed", False
            ):
                failure_reason = getattr(runner_policy.expert, "failure_reason", "expert_failed")
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
            target_bin_pos=target_pos,
            source_bin_pos=source_pos,
            phase=phase,
            failure_reason=failure_reason,
        )
