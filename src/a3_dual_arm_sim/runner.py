from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .contracts import EpisodeContext
from .env import A3DualArmEnv
from .policy import Policy
from .recording import Recorder


@dataclass(frozen=True)
class EpisodeResult:
    steps: int
    terminated: bool
    truncated: bool
    safety_reason: str | None
    success: bool = False
    discarded: bool = False


class EpisodeRunner:
    """One source-neutral rollout loop for policies, scripts, and teleoperation."""

    def __init__(
        self,
        env: A3DualArmEnv,
        policy: Policy,
        *,
        task: str,
        recorder: Recorder | None = None,
        realtime: bool = False,
        save_failed_episodes: bool = True,
    ) -> None:
        if policy.action_mode != env.action_mode:
            raise ValueError(
                f"policy action_mode={policy.action_mode} does not match env {env.action_mode}"
            )
        self.env = env
        self.policy = policy
        self.task = task
        self.recorder = recorder
        self.realtime = realtime
        self.save_failed_episodes = save_failed_episodes

    def run(self, *, seed: int = 0, max_steps: int | None = None) -> EpisodeResult:
        context = EpisodeContext(seed=seed, task=self.task, action_mode=self.env.action_mode)
        observation, _ = self.env.reset(seed=seed)
        self.policy.reset(context)
        if self.recorder is not None:
            self.recorder.start_episode(context, type(self.policy).__name__)
        terminated = truncated = False
        info: dict[str, Any] = {}
        limit = max_steps or self.env.config.horizon
        steps = 0
        discarded = False
        recorded_frames = 0
        try:
            for steps in range(1, limit + 1):
                started = time.monotonic()
                if bool(getattr(self.policy, "emergency_requested", False)):
                    self.env.emergency_stop("teleoperation emergency stop")
                action = self.policy.act(observation, self.task)
                next_observation, _, terminated, truncated, info = self.env.step(action)
                recording_enabled = bool(getattr(self.policy, "recording", True))
                if self.recorder is not None and recording_enabled:
                    self.recorder.add_frame(observation, info["applied_action"])
                    recorded_frames += 1
                observation = next_observation
                if terminated or truncated or bool(getattr(self.policy, "stop_requested", False)):
                    break
                if self.realtime:
                    remaining = 1.0 / self.env.config.control_hz - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            success = bool(info.get("success", False))
            recorded_success = success if "success" in info else None
            discarded = (
                bool(getattr(self.policy, "discard_requested", False))
                or (self.recorder is not None and recorded_frames == 0)
                or (not self.save_failed_episodes and not success)
            )
            if self.recorder is not None:
                if discarded:
                    self.recorder.discard_episode()
                else:
                    self.recorder.finish_episode(success=recorded_success)
        except BaseException:
            if self.recorder is not None:
                self.recorder.discard_episode()
            raise
        return EpisodeResult(
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            safety_reason=info.get("safety_reason"),
            success=success,
            discarded=discarded,
        )

    def close(self) -> None:
        self.policy.close()
        if self.recorder is not None:
            self.recorder.close()
        self.env.close()
