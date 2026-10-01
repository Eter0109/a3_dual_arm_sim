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
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert
from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.sim.randomization import (
    RandomizationProfile,
    choose_column,
    load_randomization_config,
)
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig

from .benchmark_results import EpisodeScore
from .policy_adapters import (
    ExpertPolicyAdapter,
    make_policy_adapter,
)

DEFAULT_CONFIG_PATH = resource_root() / "configs" / "cookie_batch.yaml"


class EpisodeExecution:
    """Single-episode environment lifecycle and execution."""

    required_cookies = 10

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
        profile: str | None = None,
        source_column: str | None = None,
        randomization_config: Path | str | None = None,
        randomization_settings: dict | None = None,
    ):
        self.randomization_settings = load_randomization_config(
            randomization_config, config=randomization_settings
        )
        if randomization_config is not None:
            profile = self.randomization_settings["profile"]
            source_column = self.randomization_settings["source_column"]
        if profile is not None:
            selected_profile = RandomizationProfile(
                **self.randomization_settings["profiles"][profile]
            )
            source_bin_noise_m = selected_profile.source_bin_noise_m
            target_bin_noise_m = selected_profile.target_bin_noise_m
            target_bin_yaw_noise_rad = selected_profile.target_bin_yaw_noise_rad
            cookie_noise_m = selected_profile.cookie_noise_m
            cookie_yaw_noise_rad = selected_profile.cookie_yaw_noise_rad
        if source_column is not None:
            choose_column(source_column, 0)
        self.profile = profile
        self.source_column = source_column
        self.last_attempt_metadata = {}
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

    def _episode_task(self, column: int | None, seed: int) -> str:
        return (
            "transfer 10 cookies into target box"
            if column is None
            else f"Transfer 10 cookies from source column {column} into the target box in two batches of five."
        )

    def _episode_policy(self, policy: Any, column: int | None, seed: int):
        if column is not None and column > 1 and policy == "same_column":

            def expert_factory(environment):
                expert = A3VariedColumnBatchExpert(environment)
                expert.requested_source_column_index = column - 1
                return expert

            return ExpertPolicyAdapter(expert_factory, name="selected_column")
        return make_policy_adapter(policy)

    def _episode_metadata(self, column: int | None, seed: int) -> dict:
        return {"profile": self.profile or "basic", "source_column": column, "seed": seed}

    def _completed_episode_metadata(self, runner_policy: Any, info: dict) -> dict:
        return {}

    def run_episode(
        self,
        policy: Any,
        seed: int,
        episode_idx: int = 1,
        env: A3CookieTransferEnv | None = None,
        recorder: Any | None = None,
        diagnostic: Any | None = None,
    ) -> EpisodeScore:
        column = choose_column(self.source_column, seed) if self.source_column is not None else None
        task = self._episode_task(column, seed)
        self.last_attempt_metadata = self._episode_metadata(column, seed)
        runner_policy = self._episode_policy(policy, column, seed)
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
            "randomization_profile": self.profile or "basic",
            "appearance_seed": seed,
            "randomization_settings": self.randomization_settings,
        }

        observation, info = env.reset(seed=seed, options=reset_options)
        self.last_attempt_metadata.update(getattr(env, "randomization_metadata", {}))
        self.last_attempt_metadata["reset_options"] = reset_options

        try:
            if hasattr(runner_policy, "bind_env"):
                runner_policy.bind_env(env)
            if hasattr(runner_policy, "reset"):
                runner_policy.reset(
                    EpisodeContext(seed=seed, task=task, action_mode=env.action_mode)
                )
            if column is not None and getattr(runner_policy, "expert", None) is not None:
                import numpy as np

                columns = sorted(set(np.asarray(env.SOURCE_POSITIONS)[:, 0]))
                runner_policy.expert._column_order = [columns[column - 1]]
        except BaseException:
            if should_close_env:
                env.close()
            raise

        t_start = time.monotonic()
        if recorder is not None:
            recorder.start_episode(
                EpisodeContext(
                    seed=seed,
                    task=task,
                    action_mode=runner_policy.action_mode,
                ),
                getattr(runner_policy, "name", type(runner_policy).__name__),
            )
            if hasattr(recorder, "set_episode_metadata"):
                recorder.set_episode_metadata(self.last_attempt_metadata)
        steps = 0
        terminated = False
        truncated = False

        try:
            while steps < self.max_steps:
                steps += 1
                if hasattr(runner_policy, "act"):
                    try:
                        action = runner_policy.act(observation, task)
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
        expected_source = (
            len(self.config.cookie_transfer.cookie_source_positions_m) - self.required_cookies
        )
        success = (
            env_success
            and cookies_in_target == self.required_cookies
            and cookies_in_source == expected_source
            and not info.get("safety_reason")
        )
        self.last_attempt_metadata.update(self._completed_episode_metadata(runner_policy, info))
        if recorder is not None:
            success = (
                success and cookies_in_source == expected_source and not info.get("safety_reason")
            )
            if hasattr(recorder, "set_episode_metadata"):
                recorder.set_episode_metadata(self.last_attempt_metadata)
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
                failure_reason = (
                    f"only {cookies_in_target}/{self.required_cookies} cookies in target box"
                )

        if should_close_env:
            env.close()

        return EpisodeScore(
            episode=episode_idx,
            seed=seed,
            score=cookies_in_target,
            max_score=self.required_cookies,
            success=success,
            steps=steps,
            wall_seconds=round(wall_time, 2),
            cookies_in_target=cookies_in_target,
            cookies_in_source=cookies_in_source,
            target_bin_pos=target_pos,
            source_bin_pos=source_pos,
            phase=phase,
            failure_reason=failure_reason,
            randomization=self.last_attempt_metadata.copy(),
            source_column=column,
        )
