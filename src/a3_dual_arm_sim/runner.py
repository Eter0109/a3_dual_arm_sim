from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contracts import EpisodeContext, jsonable
from .env import A3DualArmEnv
from .policy import Policy
from .recording import Recorder


def _jsonable(value: Any) -> Any:
    """Coerce numpy to plain Python; see :func:`a3_dual_arm_sim.contracts.jsonable`."""

    return jsonable(value)


@dataclass(frozen=True)
class EpisodeResult:
    steps: int
    terminated: bool
    truncated: bool
    safety_reason: str | None
    success: bool = False
    discarded: bool = False
    #: Final environment info mapping. Task-specific metrics live here rather
    #: than as fields, so every task can be summarised without changing this type.
    final_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, with numpy converted to builtins."""

        payload = asdict(self)
        payload["final_info"] = _jsonable(dict(self.final_info))
        return payload


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
        reset_options: Mapping[str, Any] | None = None,
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
        #: Extra ``env.reset`` options, so a caller can control a scene's own
        #: randomisation without wrapping the environment.  The collection driver
        #: uses this to pass the two switches the batch scenes need set
        #: differently: boxes and start pose move, per-Cookie jitter stays off,
        #: because their insertion is aimed at 2.5 mm gaps a jitter would close.
        self.reset_options = dict(reset_options) if reset_options else None

    def run(self, *, seed: int = 0, max_steps: int | None = None) -> EpisodeResult:
        context = EpisodeContext(seed=seed, task=self.task, action_mode=self.env.action_mode)
        observation, _ = self.env.reset(seed=seed, options=self.reset_options)
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
            # A policy may know the task was achieved when the environment cannot
            # say so.  The two-box workflow is the case: its criterion is ten
            # Cookies in *each* box with sixty left in the source, which the
            # single-box task config has no way to express, so the environment
            # reports success=False for an episode that fully succeeded.  The
            # override is applied here rather than by the caller because it also
            # feeds the discard decision below -- deciding it afterwards would throw
            # away the very episodes it exists to keep.
            override = getattr(self.policy, "success_override", None)
            if override is not None:
                success = bool(override)
            recorded_success = success if ("success" in info or override is not None) else None
            # Task-specific numbers a policy knows and ``info`` does not.  Read
            # here, before the episode is written, so the dataset's own metadata
            # can carry them: a scene that runs several attempts per process
            # writes accepted and rejected episodes side by side, and a reader who
            # has to open the collection summary to tell them apart -- or to learn
            # *why* one was rejected -- cannot filter the dataset by itself.  A
            # reason a scripted expert failed is the motivating case: it lives on
            # the expert, and a metadata line without it cannot explain a
            # rejection.
            metrics = getattr(self.policy, "metrics", None)
            if metrics:
                info = {**info, **dict(metrics)}
            discarded = (
                bool(getattr(self.policy, "discard_requested", False))
                or (self.recorder is not None and recorded_frames == 0)
                or (not self.save_failed_episodes and not success)
            )
            if self.recorder is not None:
                if discarded:
                    self.recorder.discard_episode()
                else:
                    self.recorder.finish_episode(success=recorded_success, metrics=metrics)
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
            final_info=dict(info),
        )

    def close(self) -> None:
        self.policy.close()
        if self.recorder is not None:
            self.recorder.close()
        self.env.close()
