from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .batch_expert import A3CookieBatchExpert
from .contracts import ActionMode, EpisodeContext
from .cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from .env import A3DualArmEnv
from .expert import A3CookieTransferExpert, A3GraspExpert, CookiePhase
from .grasp import A3GraspEnv, GraspTaskConfig
from .policy import Policy
from .recording import LeRobotV3Recorder
from .runner import EpisodeRunner
from .same_column_batch_expert import A3SameColumnBatchExpert
from .tasks import TASKS, TaskSpec


@dataclass
class CookieCollectionPolicy:
    """Drive the cookie expert and end the rollout once placing has finished.

    The expert's state machine reaches ``DONE`` after the final placement and then
    holds the pose, so waiting for the 2500-step horizon would record roughly a
    thousand frames of a stationary arm. Asking the runner to stop shortly after
    ``DONE`` halves the episode length while still leaving the environment's
    20-step stable-fill window intact, because that window is already counting
    down by the time the expert finishes its retract.

    This wraps any cookie expert, batch or single-Cookie: they share the phase
    enum and the ``reset``/``act`` interface, so the stopping rule needs no
    per-variant knowledge.
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


@dataclass(frozen=True)
class CollectionScene:
    """One collectable scene: what to build, what to drive it with, what it is.

    The rollout loop, the recorder, the sharding, the summary schema and the
    discard rule are identical for every scene -- they are the *collection
    interface* and they live in :func:`collect_dataset`.  A scene is only the part
    that genuinely differs, so registering a new one is a few lines rather than
    another copy of the driver.

    ``task`` carries the three pieces the dataset has to declare (name, prompt,
    success contract) and is shared with ``training``, which needs them without
    importing the simulator.
    """

    task: TaskSpec
    #: Build the environment.  Called with the run's placement noise; a scene that
    #: cannot vary its layout ignores the arguments (and should say so in its
    #: docstring, because it bounds what a policy trained on the result can learn).
    build_env: Callable[[float, float], A3DualArmEnv]
    #: Wrap the scripted expert in the policy the runner drives, including its
    #: stopping rule.
    build_policy: Callable[[A3DualArmEnv], Policy]
    #: ``info`` keys worth lifting into each attempt's row of the summary.  Task
    #: metrics stay task-specific while the row shape stays uniform.
    metric_keys: tuple[str, ...] = ()
    #: Whether the scene's own placement randomisation runs -- moving the boxes and
    #: the arm start pose.  Defaults to True because this is the variation a policy
    #: actually needs: with it off, every episode of a scene is the same picture and
    #: the same motion, and a policy can score well by memorising one trajectory.
    #: Sizes come from the config's `randomization` section; a scene whose config
    #: leaves that at zero is unaffected either way.
    #:
    #: It is safe for the batch scenes even though their insertion is aimed at
    #: measured gaps: moving the source box translates the whole Cookie layout, so
    #: the gaps are preserved exactly.  Only the per-Cookie jitter changes them, and
    #: that is the *other* switch, which the batch scenes turn off.
    randomize_placements: bool = True
    #: Whether each Cookie is jittered on its own, on top of the scene variation.
    #: Off for scenes built on measured clearances -- a 2 mm jitter is wider than
    #: the 2.5 mm gaps the five-Cookie insertion enters, so with it on both batch
    #: experts fail immediately (measured: 0 accepted in 2 attempts, against success
    #: at 1860 steps with it off).
    randomize_cookies: bool = True
    #: Placement jitter the scene really applies, when that is not the run's to
    #: choose.  ``None`` means "whatever the caller asked for", which is the normal
    #: case; a scene that owns its noise (the grasp cube, whose jitter lives in its
    #: own task config) reports it here so the summary states what was applied
    #: rather than what was requested.
    applied_noise: tuple[float, float] | None = None
    #: Extra scene facts for the summary (config path, noise, layout). Recorded so
    #: a dataset explains itself without reading the script that made it.
    details: dict[str, Any] = field(default_factory=dict)


def collect_dataset(
    root: str | Path,
    scene: CollectionScene,
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
    render_cameras: bool = True,
    fast_render: bool = False,
    use_videos: bool = True,
    save_failed_episodes: bool = False,
) -> dict[str, Any]:
    """Collect accepted episodes of one scene, and write ``collection_summary.json``.

    Seeds are interleaved across shards (``shard_index``, ``shard_index +
    shard_count``, ...) so several processes can fill one logical dataset in
    parallel without sharing a root: LeRobot writes one parquet file and one set
    of videos per dataset, so concurrent writers need separate directories.
    Merge the shards afterwards.

    ``save_failed_episodes`` is the "do we keep this data" decision.  It defaults
    to False, so an episode the environment did not call a success is discarded
    rather than written: the scripted expert is the ground truth here, and a
    demonstration of a failure teaches a policy the wrong thing.  Turning it on is
    for diagnosing a collection run, not for training.

    There are two kinds of variation and they are configured separately, because
    they cost different things:

    * *scene* variation moves the three boxes and jitters the arm's start pose.
      Sizes come from the config's ``randomization`` section, not from this
      call, and the summary records the ranges that were in force.  Every batch
      scene wants this on: moving the source box translates the whole Cookie
      layout rigidly, so the 2.5 mm gaps the five-Cookie insertion enters are
      preserved (measured: per-Cookie deviation from the applied offset is
      2.8e-17 m, float64 rounding of the same nominal numbers), and both batch
      experts still complete on randomized seeds (measured: 3/3 seeds,
      1855-1857 steps, against 1860 for the exact layout).

    * *Cookie* jitter (`position_noise_m` / `yaw_noise_rad`) moves each Cookie on
      its own, on top of the scene variation.  Only a scene with
      ``randomize_cookies`` receives it, and the five-Cookie scenes turn it off:
      a 2 mm jitter is wider than the 2.5 mm gaps their insertion is aimed at, so
      with it on both batch experts fail immediately (measured: 0 accepted in 2
      attempts, against success at 1860 steps with it off).  The single-Cookie
      scene uses it and does slightly *better* with it than with no variation at
      all, where it tips the last cookie over.

    Know what this still does *not* vary, because it bounds what a policy trained
    on the result can learn.  The jitter magnitudes are deliberately modest -- at
    the default 2 mm and 256x256 a Cookie moves about one pixel between episodes,
    and the scene ranges are single-digit millimetres -- so two episodes differ
    subtly rather than obviously.  A policy trained on that can reach a very low
    loss by memorising the trajectory it sees most, and measured rollouts confirm
    it does: they track the demonstration for the first few cookies and then fall
    apart, because they never learned to correct anything.  Widening the ranges
    is what would change that, and each widening has to be paid for by teaching
    the expert to cope: the batch insertion is aimed at measured clearances, and
    the placement is solved against the target box's live frame, so both are
    bounded by what those can absorb rather than by what the driver allows.
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
    env = scene.build_env(position_noise_m, yaw_noise_rad)
    if fast_render:
        env.use_fast_render()
    policy = scene.build_policy(env)
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
        task=scene.task.prompt,
        recorder=recorder,
        save_failed_episodes=save_failed_episodes,
        reset_options={
            "randomize_scene": scene.randomize_placements,
            "randomize_cookies": scene.randomize_cookies,
        },
    )

    results: list[dict[str, Any]] = []
    accepted = 0
    # Read the scene's actual variation before the rollout loop, while the
    # environment is definitely alive: `runner.close()` shuts it down, and the
    # summary has to state the ranges that were in force.  Each range is recorded
    # as [low, high] in metres or radians, so a dataset says how far its boxes
    # could move without anyone having to find the config.
    randomization = env.config.randomization
    scene_randomization = {
        name: [axis.low, axis.high]
        for name, axis in randomization.axes().items()
        if axis.movable
    }
    scene_randomization["min_box_clearance_m"] = randomization.min_box_clearance_m
    try:
        for attempt in range(attempt_limit):
            seed = start_seed + shard_index + attempt * shard_count
            result = runner.run(seed=seed)
            row: dict[str, Any] = {
                "seed": seed,
                "success": result.success,
                "steps": result.steps,
                "discarded": result.discarded,
                "safety_reason": result.safety_reason,
            }
            for key in scene.metric_keys:
                row[key] = result.final_info.get(key)
            results.append(row)
            accepted += int(result.success)
            metrics = "".join(f" {key}={row[key]}" for key in scene.metric_keys)
            print(
                f"{scene.task.name} shard={shard_index}/{shard_count} "
                f"attempt={attempt + 1}/{attempt_limit} seed={seed} "
                f"success={result.success} accepted={accepted}/{episodes} "
                f"steps={result.steps}{metrics}",
                flush=True,
            )
            if accepted >= episodes:
                break
    finally:
        runner.close()

    # Record what was actually applied, not what was asked for.  A scene that
    # turns its placement randomisation off ignores both noise values, and one
    # that owns its own jitter ignores them too; a dataset claiming 2 mm of jitter
    # it never had would be misleading about what a policy trained on it can be
    # expected to handle.
    if scene.applied_noise is not None:
        applied_position_noise, applied_yaw_noise = scene.applied_noise
    elif scene.randomize_cookies:
        applied_position_noise, applied_yaw_noise = position_noise_m, yaw_noise_rad
    else:
        applied_position_noise, applied_yaw_noise = 0.0, 0.0

    summary = {
        "schema_version": 2,
        "task": scene.task.name,
        "success_contract": scene.task.success_contract,
        "prompt": scene.task.prompt,
        "repo_id": repo_id,
        "shard_index": shard_index,
        "shard_count": shard_count,
        # Two different kinds of variation, recorded separately because they mean
        # different things for a policy: the boxes and start pose moving vs each
        # Cookie being jittered in place.  The magnitudes come from the config
        # that built this environment, so the summary states the ranges that were
        # in force rather than the ones a caller hoped for.
        "placements_randomized": scene.randomize_placements,
        "cookies_jittered": scene.randomize_cookies,
        "scene_randomization_m": scene_randomization,
        "position_noise_m": applied_position_noise,
        "yaw_noise_rad": applied_yaw_noise,
        "requested_episodes": episodes,
        "accepted_episodes": accepted,
        "attempts": len(results),
        **scene.details,
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

# --------------------------------------------------------------------------- scenes
#
# Each builder below is four facts and two closures.  They are grouped so a reader
# can see every registered scene at once, and so adding one is obviously additive.


def grasp_scene() -> CollectionScene:
    """The single-cube grasp, at a fixed object pose plus its own jitter.

    This scene owns its placement noise: ``GraspTaskConfig`` jitters the cube by
    ``position_noise_m`` (25 mm by default) through its own mechanism, so the
    run's noise arguments are not forwarded and the scene reports the value it
    really applies.  ``grasp_collection_noise_m`` below is that value, restated for
    the summary.
    """

    default_noise = GraspTaskConfig().position_noise_m

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        del position_noise_m, yaw_noise_rad
        return A3GraspEnv(render_cameras=False)

    def build_policy(env: A3DualArmEnv) -> Policy:
        return A3GraspExpert(env)  # type: ignore[arg-type]

    return CollectionScene(
        task=TASKS["a3_grasp"],
        build_env=build_env,
        build_policy=build_policy,
        applied_noise=(default_noise, 0.0),
        details={"noise_source": "GraspTaskConfig.position_noise_m"},
    )


def cookie_transfer_scene(
    config: str | Path | None = None, *, hold_steps: int = 40
) -> CollectionScene:
    """One Cookie at a time, moved by the grip that also holds the target box.

    This is the original single-Cookie workflow.  It is kept registered because its
    datasets are the ``a3_cookie_transfer`` contract, but be aware it no longer
    completes on the current dense 80-Cookie scene: its right arm cannot reach the
    tabletop target box, so collection stops at ``SUPPORT_BOX`` and accepts
    nothing.  Use one of the batch scenes below for new data.
    """

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        task_config = CookieTransferTaskConfig(
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
        )
        return A3CookieTransferEnv(config, task_config=task_config, render_cameras=True)

    def build_policy(env: A3DualArmEnv) -> Policy:
        return CookieCollectionPolicy(
            expert=A3CookieTransferExpert(env),  # type: ignore[arg-type]
            hold_steps=hold_steps,
        )

    return CollectionScene(
        task=TASKS["a3_cookie_transfer"],
        build_env=build_env,
        build_policy=build_policy,
        metric_keys=("cookies_in_target", "cookies_in_source"),
        details={"config": None if config is None else str(config), "hold_steps": hold_steps},
    )


def _batch_scene(
    task_name: str,
    config: str,
    make_expert: Callable[[A3DualArmEnv], A3CookieTransferExpert],
    *,
    hold_steps: int,
    details: dict[str, Any] | None = None,
) -> CollectionScene:
    """Shared body of the two five-at-a-time scenes.

    Both need the same task config, and it is not the default one:
    ``require_exact_slots=False`` because five Cookies placed from a single wide
    grip do not land on the ten single-Cookie slot centres, ``require_released=True``
    so nothing counts while still pinched, and ``terminate_on_success=False`` so the
    recording runs to the expert's own stopping rule instead of the environment
    cutting it off mid-retract.  ``success`` still comes from the environment, so
    the accepted-episode rule stays in one place.
    """

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        task_config = CookieTransferTaskConfig(
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
            require_exact_slots=False,
            require_released=True,
            terminate_on_success=False,
        )
        return A3CookieTransferEnv(config, task_config=task_config, render_cameras=True)

    def build_policy(env: A3DualArmEnv) -> Policy:
        return CookieCollectionPolicy(expert=make_expert(env), hold_steps=hold_steps)

    return CollectionScene(
        task=TASKS[task_name],
        build_env=build_env,
        build_policy=build_policy,
        metric_keys=("cookies_in_target", "cookies_in_source"),
        # The insertion is aimed at the layout's 2.5 mm gaps, so per-Cookie jitter
        # is off.  The scene *is* randomised -- the boxes and the arm start pose
        # move -- because translating the source box carries the whole layout with
        # it and leaves every gap exactly as measured.  See the field docs.
        randomize_placements=True,
        randomize_cookies=False,
        details={
            "config": config,
            "hold_steps": hold_steps,
            "require_exact_slots": False,
            "require_released": True,
            **(details or {}),
        },
    )


def cookie_batch_scene(*, hold_steps: int = 40) -> CollectionScene:
    """Five at a time, both batches from opposite ends of two columns (0-4, 20-24)."""

    return _batch_scene(
        "a3_cookie_batch",
        "configs/cookie_batch.yaml",
        A3CookieBatchExpert,
        hold_steps=hold_steps,
        details={"source_columns": "one exposed end of each of two columns"},
    )


def cookie_same_column_scene(*, hold_steps: int = 40) -> CollectionScene:
    """Five at a time, both batches from one column (0-4, then 5-9).

    This is the variant that succeeds on the second batch without moving to
    another column: its thin left fingertip descends into the 2.5 mm gaps instead
    of ramming the bevel, so the neighbours are not pushed over in the first place.
    """

    return _batch_scene(
        "a3_cookie_same_column",
        "configs/cookie_same_column.yaml",
        A3SameColumnBatchExpert,
        hold_steps=hold_steps,
        details={"source_columns": "both batches from one column, front end first"},
    )


def registered_scenes() -> dict[str, Callable[[], CollectionScene]]:
    """Scene name -> builder.  This is the list a CLI or a sweep iterates."""

    return {
        "a3_grasp": grasp_scene,
        "a3_cookie_transfer": cookie_transfer_scene,
        "a3_cookie_batch": cookie_batch_scene,
        "a3_cookie_same_column": cookie_same_column_scene,
    }


def scene_by_name(name: str, **kwargs: Any) -> CollectionScene:
    """Build a registered scene, with a message that lists the alternatives."""

    builders = registered_scenes()
    if name not in builders:
        raise ValueError(f"unknown scene {name!r}; registered scenes are {sorted(builders)}")
    return builders[name](**kwargs)


# --------------------------------------------------------------------------- compatibility
#
# The two original entry points, kept so existing scripts and the CLI keep working.
# They are wrappers now: the driver they used to duplicate lives in
# `collect_dataset`, and everything they configured by hand is a scene.


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
    """Collect successful single-Cookie transfer demonstrations.

    Prefer :func:`collect_dataset` with :func:`cookie_batch_scene` or
    :func:`cookie_same_column_scene`; this scene no longer completes on the dense
    layout (see :func:`cookie_transfer_scene`).
    """

    return collect_dataset(
        root,
        cookie_transfer_scene(config, hold_steps=hold_steps),
        repo_id=repo_id,
        episodes=episodes,
        start_seed=start_seed,
        max_attempts=max_attempts,
        position_noise_m=position_noise_m,
        yaw_noise_rad=yaw_noise_rad,
        hold_steps=hold_steps,
        shard_index=shard_index,
        shard_count=shard_count,
        render_cameras=render_cameras,
        fast_render=fast_render,
        use_videos=use_videos,
    )


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
    """Collect successful single-cube grasp demonstrations."""

    del config, render_cameras
    return collect_dataset(
        root,
        grasp_scene(),
        repo_id=repo_id,
        episodes=episodes,
        start_seed=start_seed,
        max_attempts=max_attempts,
        fast_render=fast_render,
    )
