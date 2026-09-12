from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .expert import A3GraspExpert
from .grasp import A3GraspEnv
from .recording import LeRobotV3Recorder
from .runner import EpisodeRunner

GRASP_PROMPT = "pick up the red cube"


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
            f"collected only {accepted}/{episodes} successful episodes in {len(results)} attempts"
        )
    return summary
