from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ActionMode, EpisodeContext
from .cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from .expert import A3CookieTransferExpert, A3GraspExpert, CookiePhase
from .grasp import A3GraspEnv
from .recording import LeRobotV3Recorder
from .runner import EpisodeRunner

GRASP_PROMPT = "pick up the red cube"
COOKIE_PROMPT = "transfer exactly ten upright square cookie blocks into the 2x5 box"


@dataclass
class CookieCollectionPolicy:
    """Drive the cookie expert and end the rollout once placing has finished.

    The expert's state machine reaches ``DONE`` after the final placement and then
    holds the pose, so waiting for the 2500-step horizon would record roughly a
    thousand frames of a stationary arm. Asking the runner to stop shortly after
    ``DONE`` halves the episode length while still leaving the environment's
    20-step stable-fill window intact, because that window is already counting
    down by the time the expert finishes its retract.
    """

    expert: A3CookieTransferExpert
    hold_steps: int = 40
    action_mode: ActionMode = "joint_position"
    _steps_since_done: int = 0

    def reset(self, context: EpisodeContext | None = None) -> None:
        self.expert.reset()
        self._steps_since_done = 0

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        if self.expert.phase is CookiePhase.DONE:
            self._steps_since_done += 1
        return self.expert.act(observation)

    @property
    def stop_requested(self) -> bool:
        """Let the runner finish once the hold window has been observed."""

        return self.expert.failed or self._steps_since_done >= self.hold_steps

    def close(self) -> None:
        self.expert.close()


def collect_cookie_dataset(
    root: str | Path,
    *,
    repo_id: str,
    episodes: int,
    start_seed: int = 0,
    max_attempts: int | None = None,
    position_noise_m: float = 0.002,
    yaw_noise_rad: float = 0.05,
    hold_steps: int = 40,
    shard_index: int = 0,
    shard_count: int = 1,
    config: str | Path | None = None,
    render_cameras: bool = True,
    fast_render: bool = False,
    use_videos: bool = True,
) -> dict[str, Any]:
    """Collect successful cookie-transfer demonstrations.

    Seeds are interleaved across shards (``shard_index``, ``shard_index +
    shard_count``, ...) so several processes can fill one logical dataset in
    parallel without sharing a root: LeRobot writes one parquet file and one set
    of videos per dataset, so concurrent writers need separate directories.
    Merge the shards afterwards.

    Small placement noise is on by default. It is the only source of variation
    available after the domain-randomisation module was removed, and the scripted
    expert performs slightly *better* with it than with the fully deterministic
    scene, where it tips the last cookie over.

    Know what this does *not* vary, because it bounds what a policy trained on the
    result can learn. ``A3CookieTransferEnv.reset`` writes ``DEPLOYMENT_HOME`` into
    ``qpos`` every episode, so the arm's initial pose is byte-identical across all
    of them, and the only other variation is ``position_noise_m`` /
    ``yaw_noise_rad``. At the default 2 mm and 256x256, a cookie moves by about one
    pixel between episodes, so there is almost no visual signal distinguishing one
    demonstration from another. A policy trained on that can reach a very low
    loss by memorising one joint trajectory, and measured rollouts confirm it
    does: they track the demonstration for the first few cookies and then fall
    apart, because they never learned to correct anything. Treat closed-loop
    robustness as out of reach until the initial pose and the scene are varied
    enough for the cameras to see a difference.
    """

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    attempt_limit = max_attempts or episodes * 5
    if attempt_limit < episodes:
        raise ValueError("max_attempts cannot be smaller than episodes")

    destination = Path(root)
    task_config = CookieTransferTaskConfig(
        position_noise_m=position_noise_m,
        yaw_noise_rad=yaw_noise_rad,
    )
    env = A3CookieTransferEnv(
        config, task_config=task_config, render_cameras=render_cameras
    )
    if fast_render:
        env.use_fast_render()
    expert = A3CookieTransferExpert(env)
    policy = CookieCollectionPolicy(expert=expert, hold_steps=hold_steps)
    recorder = LeRobotV3Recorder(
        destination,
        repo_id=repo_id,
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
        use_videos=use_videos,
    )
    runner = EpisodeRunner(
        env,
        policy,
        task=COOKIE_PROMPT,
        recorder=recorder,
        save_failed_episodes=False,
    )

    results: list[dict[str, Any]] = []
    accepted = 0
    try:
        for attempt in range(attempt_limit):
            seed = start_seed + shard_index + attempt * shard_count
            result = runner.run(seed=seed)
            results.append(
                {
                    "seed": seed,
                    "success": result.success,
                    "steps": result.steps,
                    "discarded": result.discarded,
                    "safety_reason": result.safety_reason,
                    "cookies_in_target": int(result.final_info.get("cookies_in_target", 0)),
                    "cookies_in_source": int(result.final_info.get("cookies_in_source", 0)),
                }
            )
            accepted += int(result.success)
            print(
                f"cookie_collection shard={shard_index}/{shard_count} "
                f"attempt={attempt + 1}/{attempt_limit} seed={seed} "
                f"success={result.success} accepted={accepted}/{episodes} "
                f"steps={result.steps}",
                flush=True,
            )
            if accepted >= episodes:
                break
    finally:
        runner.close()

    summary = {
        "schema_version": 2,
        "task": "a3_cookie_transfer",
        "success_contract": "exact_2x5_fill_touching_all_walls_upright_1s",
        "prompt": COOKIE_PROMPT,
        "repo_id": repo_id,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "position_noise_m": position_noise_m,
        "yaw_noise_rad": yaw_noise_rad,
        "requested_episodes": episodes,
        "accepted_episodes": accepted,
        "attempts": len(results),
        "results": results,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if accepted < episodes:
        raise RuntimeError(
            f"collected only {accepted}/{episodes} successful episodes "
            f"in {len(results)} attempts"
        )
    return summary


def collect_grasp_dataset(
    root: str | Path,
    *,
    repo_id: str,
    episodes: int,
    start_seed: int = 0,
    max_attempts: int | None = None,
    config: str | Path | None = None,
    render_cameras: bool = True,
    fast_render: bool = False,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    attempt_limit = max_attempts or episodes * 3
    if attempt_limit < episodes:
        raise ValueError("max_attempts cannot be smaller than episodes")
    destination = Path(root)
    env = A3GraspEnv(config, render_cameras=render_cameras)
    if fast_render:
        env.use_fast_render()
    expert = A3GraspExpert(env)
    recorder = LeRobotV3Recorder(
        destination,
        repo_id=repo_id,
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
    )
    runner = EpisodeRunner(
        env,
        expert,
        task=GRASP_PROMPT,
        recorder=recorder,
        save_failed_episodes=False,
    )
    results: list[dict[str, Any]] = []
    accepted = 0
    try:
        for attempt in range(attempt_limit):
            seed = start_seed + attempt
            result = runner.run(seed=seed)
            results.append(
                {
                    "seed": seed,
                    "success": result.success,
                    "steps": result.steps,
                    "discarded": result.discarded,
                    "safety_reason": result.safety_reason,
                }
            )
            accepted += int(result.success)
            print(
                f"grasp_collection attempt={attempt + 1}/{attempt_limit} seed={seed} "
                f"success={result.success} accepted={accepted}/{episodes}",
                flush=True,
            )
            if accepted >= episodes:
                break
    finally:
        runner.close()
    summary = {
        "schema_version": 2,
        "task": "a3_grasp",
        "success_contract": "dual_contact_centered_clear_lifted_low_motion_1s",
        "prompt": GRASP_PROMPT,
        "repo_id": repo_id,
        "requested_episodes": episodes,
        "accepted_episodes": accepted,
        "attempts": len(results),
        "results": results,
    }
    (destination / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if accepted < episodes:
        raise RuntimeError(
            f"collected only {accepted}/{episodes} successful episodes "
            f"in {len(results)} attempts"
        )
    return summary
