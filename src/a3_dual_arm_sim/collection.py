from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .batch_expert import A3CookieBatchExpert
from .config import CameraConfig, load_config
from .contracts import ActionMode, EpisodeContext
from .cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from .env import A3DualArmEnv
from .expert import A3CookieTransferExpert, A3GraspExpert, CookiePhase
from .grasp import A3GraspEnv, GraspTaskConfig
from .policy import Policy
from .recording import LeRobotV3Recorder
from .runner import EpisodeRunner
from .same_column_batch_expert import A3SameColumnBatchExpert
from .single_piece_expert import A3SinglePieceExpert
from .tasks import TASKS, TaskSpec
from .two_box_batch import TwoBoxBatchExpert


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


@dataclass
class TwoBoxCollectionPolicy:
    """Drive the two-box relay, and declare success the way *it* defines it.

    The relay's own criterion is ten Cookies in each box, sixty left in the source,
    box A pushed clear and box B back in the station.  The environment cannot report
    that: its task config checks one target bin against a ten-Cookie fill, so a run
    that emptied two boxes would look like a failure to it.  ``success_override``
    therefore comes from the expert, which is the only thing that knows.

    ``act`` also bridges a signature difference: the relay expert takes no
    observation (it reads the simulator directly, like every other scripted expert
    here) while the runner passes one.
    """

    expert: TwoBoxBatchExpert
    hold_steps: int = 40
    action_mode: ActionMode = "joint_position"
    _steps_since_done: int = 0

    def reset(self, context: EpisodeContext | None = None) -> None:
        self.expert.reset()
        self._steps_since_done = 0

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        del observation, task
        if self.expert.done:
            self._steps_since_done += 1
        return self.expert.act()

    @property
    def stop_requested(self) -> bool:
        return bool(self.expert.failed) or self._steps_since_done >= self.hold_steps

    @property
    def success_override(self) -> bool:
        """The relay's criterion, not the environment's; see the class docstring."""

        return bool(self.expert.done and not self.expert.failed)

    @property
    def metrics(self) -> dict[str, Any]:
        """What the relay knows that ``info`` does not.

        The counts are per box, and the environment reports neither: it tracks one
        target bin.  The stage and the failure reason are what make a rejected
        episode diagnosable -- "0/1 accepted" says nothing about whether the fill,
        the push, or the carry went wrong.
        """

        try:
            box_a, box_b = self.expert.counts()
        except Exception:  # noqa: BLE001 - a torn-down env must not mask the result
            box_a = box_b = -1
        return {
            "stage": self.expert.stage,
            "failure_reason": self.expert.failed,
            "box_a_cookie_count": box_a,
            "box_b_cookie_count": box_b,
        }

    def close(self) -> None:
        return None


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
    rollover_mb: float | None = None,
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
        rollover_mb=rollover_mb,
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
            # An expert may refuse a layout outright rather than run it and fail.
            # The five-Cookie insertion checks its target column's reachability when
            # it plans, and raises when a randomised layout puts the column outside
            # the window it can place into.  That is a legitimate rejected attempt --
            # the same information a failed rollout carries, arrived at by planning
            # instead of by trying -- so it is recorded and the sweep moves on.
            #
            # Letting it escape is what this guard exists to prevent, and the cost
            # was measured: on a 20-shard overnight run, 6 shards died this way
            # within their first dozen attempts and took the rest of their 170-episode
            # quota with them.  Roughly 30% of the collection capacity was lost to an
            # exception that simply meant "this seed is not usable".
            #
            # ``EpisodeRunner.run`` already discards the episode buffer on any
            # exception, so the recorder is left clean and the next seed starts fresh.
            try:
                result = runner.run(seed=seed)
            except Exception as exc:  # noqa: BLE001 - see above: any refusal is a skip
                reason = f"{type(exc).__name__}: {exc}"
                row = {
                    "seed": seed,
                    "success": False,
                    "steps": 0,
                    "discarded": True,
                    "safety_reason": None,
                    "failure_reason": reason,
                }
                for key in scene.metric_keys:
                    row[key] = None
                results.append(row)
                print(
                    f"{scene.task.name} shard={shard_index}/{shard_count} "
                    f"attempt={attempt + 1}/{attempt_limit} seed={seed} "
                    f"success=False accepted={accepted}/{episodes} steps=0 "
                    f"reason={reason}",
                    flush=True,
                )
                continue
            row: dict[str, Any] = {
                "seed": seed,
                "success": result.success,
                "steps": result.steps,
                "discarded": result.discarded,
                "safety_reason": result.safety_reason,
                # Read from the episode, not from the metric keys: an expert that
                # fails explains itself here, and without this line the reason was
                # never in the row the progress line reads, so every rejection
                # printed as a bare ``success=False``.
                "failure_reason": result.final_info.get("failure_reason"),
            }
            for key in scene.metric_keys:
                row[key] = result.final_info.get(key)
            results.append(row)
            accepted += int(result.success)
            metrics = "".join(
                f" {key}={row[key]}" for key in scene.metric_keys if row[key] is not None
            )
            # A rejected episode's reason belongs on the progress line, not only in
            # the summary: a long collection is watched as it runs, and "success=
            # False" alone does not say whether to stop the run and fix something.
            reason = ""
            if not result.success and row.get("failure_reason"):
                reason = f" reason={row['failure_reason']}"
            print(
                f"{scene.task.name} shard={shard_index}/{shard_count} "
                f"attempt={attempt + 1}/{attempt_limit} seed={seed} "
                f"success={result.success} accepted={accepted}/{episodes} "
                f"steps={result.steps}{metrics}{reason}",
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


def grasp_scene(*, visual_style: str = "plain") -> CollectionScene:
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
        return A3GraspEnv(
            _styled_config(None, visual_style), render_cameras=False
        )

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
    config: str | Path | None = None, *, hold_steps: int = 40, visual_style: str = "plain"
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


def _styled_config(config: str | Path | None, visual_style: str) -> Any:
    """Load ``config`` and take on ``visual_style`` if it differs.

    The appearance is compiled into the model XML, so a scene that wants a
    different one has to hand the constructor a ``SimConfig`` rather than a path.
    """

    loaded = load_config(config)
    if visual_style == loaded.visual_style:
        return loaded
    return replace(loaded, visual_style=visual_style)


def _batch_scene(
    task_name: str,
    config: str,
    make_expert: Callable[[A3DualArmEnv], A3CookieTransferExpert],
    *,
    hold_steps: int,
    details: dict[str, Any] | None = None,
    visual_style: str = "plain",
) -> CollectionScene:
    """Shared body of the two five-at-a-time scenes.

    Both need the same task config, and it is not the default one:
    ``require_exact_slots=False`` because five Cookies placed from a single wide
    grip do not land on the ten single-Cookie slot centres, ``require_released=True``
    so nothing counts while still pinched, and ``terminate_on_success=False`` so the
    recording runs to the expert's own stopping rule instead of the environment
    cutting it off mid-retract.  ``success`` still comes from the environment, so
    the accepted-episode rule stays in one place.

    ``visual_style`` reaches the model builder, so it has to arrive as a
    ``SimConfig`` rather than a path: the appearance is compiled into the XML, not
    applied to a live scene.  It changes no physical parameter -- see
    :func:`a3_dual_arm_sim.model._add_detailed_visuals` for the measurements.
    """

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        task_config = CookieTransferTaskConfig(
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
            require_exact_slots=False,
            require_released=True,
            terminate_on_success=False,
        )
        loaded = load_config(config)
        if visual_style != loaded.visual_style:
            loaded = replace(loaded, visual_style=visual_style)
        return A3CookieTransferEnv(loaded, task_config=task_config, render_cameras=True)
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


def cookie_batch_scene(*, hold_steps: int = 40, visual_style: str = "plain") -> CollectionScene:
    """Five at a time, both batches from opposite ends of two columns (0-4, 20-24)."""

    return _batch_scene(
        "a3_cookie_batch",
        "configs/cookie_batch.yaml",
        A3CookieBatchExpert,
        hold_steps=hold_steps,
        details={"source_columns": "one exposed end of each of two columns"},
        visual_style=visual_style,
    )


def cookie_same_column_scene(
    *, hold_steps: int = 40, visual_style: str = "plain"
) -> CollectionScene:
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
        visual_style=visual_style,
    )


def cookie_two_box_scene(
    *, hold_steps: int = 40, visual_style: str = "plain"
) -> CollectionScene:
    """Two boxes filled by relay: A is filled, pushed clear, B takes its place.

    Twice the work of a single-box scene, so an episode is roughly twice as long
    and a run needs its own time budget; `configs/cookie_two_box_batch.yaml` raises
    the horizon to 12000 for the same reason.

    The success criterion belongs to the expert rather than the environment (see
    :class:`TwoBoxCollectionPolicy`), and the two-box randomisation is the one this
    scene owns: all three boxes move and yaw, bounded by their mutual clearance,
    because the two tabletop boxes start only 86 mm apart.
    """

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        task_config = CookieTransferTaskConfig(
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
            require_exact_slots=False,
            require_released=True,
            # The relay's subject is the box exchange, not the placement: a Cookie
            # shoved flat by the push is still in the box, and grading it as
            # missing would reject a run for succeeding at the harder half.
            require_upright=False,
            terminate_on_success=False,
        )
        return A3CookieTransferEnv(
            "configs/cookie_two_box_batch.yaml",
            task_config=task_config,
            render_cameras=True,
        )

    def build_policy(env: A3DualArmEnv) -> Policy:
        return TwoBoxCollectionPolicy(
            expert=TwoBoxBatchExpert(env),  # type: ignore[arg-type]
            hold_steps=hold_steps,
        )

    return CollectionScene(
        task=TASKS["a3_cookie_two_box"],
        build_env=build_env,
        build_policy=build_policy,
        # What a two-box episode is judged on, per box and per stage: a row that
        # says only "success: false" cannot distinguish a failed fill from a failed
        # carry, and those need different fixes.
        metric_keys=(
            "stage",
            "failure_reason",
            "box_a_cookie_count",
            "box_b_cookie_count",
        ),
        randomize_placements=True,
        randomize_cookies=False,
        details={
            "config": "configs/cookie_two_box_batch.yaml",
            "hold_steps": hold_steps,
            "require_exact_slots": False,
            "require_released": True,
            "require_upright": False,
            "workflow": "fill A, push A clear, carry B into the station, fill B",
            "success_decision": (
                "expert, not the environment: a two-box fill is not expressible as "
                "the single-bin task config"
            ),
        },
    )


def cookie_single_piece_scene(
    *, hold_steps: int = 40, visual_style: str = "plain"
) -> CollectionScene:
    """One Cookie in the source box, and an instruction that names one.

    The smallest arrangement that is still the task: take a piece out of the source bin and
    put it in the target box.  Nothing physical is re-derived -- same bins, same Cookie,
    same target box, and the piece sits exactly where the dense layout puts its first one,
    which is the one the five-at-a-time experts reach for first.  Only the number changes.

    That is the point.  A result here separates two questions the dense scene answers at
    once: whether a policy can approach, grasp and place at all, and whether it can do so
    ten times without disturbing the stack.  A failure here is about the first, and says
    nothing about the second.

    The layout is *replaced* rather than parameterised because the source positions **are**
    the Cookies: ``model._add_cookie_scene`` compiles one body per entry, so a one-entry list
    is a one-Cookie scene by construction rather than by a count that could disagree with it.
    The piece kept is the config's own first source position, which is where a policy
    fine-tuned on the dense scene has seen a Cookie before.

    ``require_exact_slots=False``: the ten-slot grid is the ten-piece fill's criterion, and
    one piece cannot fill it.  With a single piece the task is "counted in the box and let go
    of", which is what ``require_released`` already says.  ``terminate_on_success=True``,
    unlike the batch scenes, because there is no expert stopping rule to defer to and no
    second batch to reach.

    ``build_policy`` is :class:`~a3_dual_arm_sim.single_piece_expert.A3SinglePieceExpert`: the
    base single-Cookie state machine with its held-box opening removed.  That opening is what
    made this scene look expertless -- ``A3CookieTransferExpert`` spends its first phase
    having the right arm fetch the tabletop box, times out with "right box approach timeout;
    contacts=[0.0, 0.0]" (the same failure three pre-existing tests record), and never reaches
    the Cookies.  With the box left where the scene put it, the rest of that state machine --
    approach, close, verify, lift, place, release, retract -- runs as written, and it is the
    reference that says this scene can be solved at all.
    """

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        loaded = _styled_config("configs/cookie_same_column.yaml", visual_style)
        layout = loaded.cookie_transfer.cookie_source_positions_m
        loaded = replace(
            loaded,
            cookie_transfer=replace(
                loaded.cookie_transfer,
                # This config's **first** source position, which is the piece the verified
                # five-at-a-time expert grasps first: `_candidate_batches` walks the columns
                # in x order and takes the five lowest y, so index 0 is that expert's opening
                # move.  Reducing the layout to it changes the count and nothing else.
                #
                # The config matters, and getting it wrong is not obvious from a failure.  An
                # earlier version of this scene started from the default config, whose target
                # box sits at a different world position, and every expert placement failed --
                # including the batch expert's own reachability check, which reported "target
                # column 1 unreachable with vertical grasp: the closest pose is 6.39 mm from
                # the target".  Nothing about that message says "wrong config", and it was
                # read first as a Cookie-count problem.
                #
                # The module-level COOKIE_SOURCE_POSITIONS would not serve either: this config
                # has its own rows, and mixing one in would put the piece at a y the expert
                # never used.
                cookie_source_positions_m=(layout[0],),
            ),
        )
        task_config = CookieTransferTaskConfig(
            cookie_count=1,
            required_cookies=1,
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
            require_exact_slots=False,
            require_released=True,
            terminate_on_success=True,
        )
        return A3CookieTransferEnv(loaded, task_config=task_config, render_cameras=True)

    def build_policy(env: A3DualArmEnv) -> Policy:
        return CookieCollectionPolicy(
            expert=A3SinglePieceExpert(env),  # type: ignore[arg-type]
            hold_steps=hold_steps,
        )

    return CollectionScene(
        task=TASKS["a3_cookie_single_piece"],
        build_env=build_env,
        build_policy=build_policy,
        metric_keys=("cookies_in_target", "cookies_in_source"),
        randomize_placements=True,
        # No stack, so there is nothing for a per-Cookie jitter to disturb; the clearance
        # argument that turns it off for the batch scenes does not apply.
        randomize_cookies=True,
        details={
            "config": "configs/cookie_same_column.yaml, with cookie_source_positions_m reduced to one entry",
            "hold_steps": hold_steps,
            "source_positions": 1,
            "required_cookies": 1,
            "require_exact_slots": False,
            "require_released": True,
            "expert": "A3SinglePieceExpert (base state machine, box left on the table)",
        },
    )


#: LIBERO's agentview, **copied** rather than re-derived.  It is a *fixed* camera -- it
#: points one way for the whole episode -- so a camera reconstructed to aim *at* the bowl
#: reproduces the distance, the elevation, the bearing and the field of view and still
#: frames the shot differently: centred, where LIBERO's has the bowl off to one side.  The
#: previous version of this did exactly that, and its elevation was 2.6 degrees flatter and
#: 60 mm closer on top of the aim difference.  Measured off LIBERO's compiled model by
#: `scripts/compare_libero_environment.py`.
LIBERO_AGENTVIEW_POS_M = (0.6586, 0.0, 1.6104)
LIBERO_AGENTVIEW_QUAT_WXYZ = (0.6380, 0.3048, 0.3048, 0.6380)

#: LIBERO world -> this world, as **one translation**: measured at the centre of the bowl
#: the task picks up, which sits at (-0.0635, 0.2020, 0.9247) there and (0.1500, 0.1300,
#: 0.7747) here.  The two worlds share their axes -- LIBERO's bearing to its own camera
#: comes back within 0.9 degrees either way, and every object placed through this offset
#: lands on its LIBERO position to a millimetre -- so no rotation belongs in the map, and
#: the camera's quaternion transfers unchanged.  Everything LIBERO-side in this file is
#: expressed through it, which is what makes "restore LIBERO's scene" a single number.
LIBERO_WORLD_OFFSET_M = (0.2135, -0.0720, -0.1500)

LIBERO_AGENTVIEW_MATCHED_POS_M = tuple(
    measured + offset
    for measured, offset in zip(LIBERO_AGENTVIEW_POS_M, LIBERO_WORLD_OFFSET_M)
)
LIBERO_AGENTVIEW_FOVY_DEG = 45.0

#: Where the bowl actually is, after settling: the scene config's spawn position plus the
#: mesh's 100 mm offset along -y and the 6 mm it drops.  Measured, and equal to the 0.774 m
#: that `scripts/probe_libero_props_trace.py` reports.
LIBERO_MATCHED_FOCUS_M = (0.150, 0.130, 0.774)

#: LIBERO's `eye_in_hand` is `pos="0.05 0 0"` on `right_hand`, i.e. on the approach axis, at
#: fovy 75.  This arm's is 45 mm out along its own tool axis -- the same idea -- but 85 mm
#: *off* it, and 70 degrees.
LIBERO_MATCHED_WRIST_AXIAL_M = 0.045
LIBERO_MATCHED_WRIST_LOOKAHEAD_M = 0.200
LIBERO_MATCHED_WRIST_FOVY_DEG = 75.0

#: Where LIBERO's own policy expects to find the hand, on this arm's joints.  Solved by
#: `scripts/solve_libero_home.py`, which also prints why it is a *nearest* pose rather than
#: the pose itself: LIBERO's policy starts every episode with the end effector at
#: (-0.2112, -0.0111, 1.1742) -- at the height of the shoulder, 0.365 m in front of it, tool
#: hanging straight down -- and this arm cannot stand there.  The wrist pitch spans +/-40
#: degrees and that pose needs more bend than it has, so the closest reachable pose with the
#: tool still pointing down lands **26 mm** away, six degrees off vertical.  Held at the
#: full orientation instead, it lands 102 mm away.
#:
#: That residual is the measurement this is here to make.  The scene's camera, its objects
#: and its lighting are LIBERO's to a millimetre; the arm is the part that cannot be
#: reproduced, and this is how much of it cannot be, in metres, at the one pose an episode
#: starts from.
LIBERO_MATCHED_HOME_RAD = (
    1.028929, 1.569900, -1.365162, -1.198814, -0.596119, 0.228400, -1.489687,
)

#: LIBERO's own initial joint configuration for its Panda, read from `qpos` at the reset its
#: evaluation uses.  With LIBERO's robot in the scene this is the only home that makes sense:
#: the whole point is that the arm stops being a difference, and a home invented here would
#: put the hand somewhere the policy has never started from.
LIBERO_PANDA_HOME_RAD = (
    -0.000131, -0.170836, -0.024106, -2.452567, -0.006121, 2.225902, 0.767745,
)


def _libero_matched_cameras(cameras: CameraConfig) -> CameraConfig:
    """The camera geometry that reproduces LIBERO's own framing on this arm.

    Why this touches the cameras and nothing else
    ---------------------------------------------
    A LIBERO-trained policy is being asked to drive this arm, and the perturbation sweep
    over `libero_spatial` prices what such a policy needs from a camera: every material
    flattened to one grey and every texture reference dropped costs **2 points**, the camera
    headlight costs **0**, the front camera's field of view costs 12-14 either way, and a
    1.85x change in the wrist camera's field of view costs **all 50**.  The project's own
    `plain`-versus-`detailed` test on this scene agrees that style is free.

    So this matches the cameras and deliberately leaves the lighting and the materials
    alone.  Changing those would be work with a measured price of nought and a real chance
    of moving something else; the cameras carry the money.

    What is matched
    ---------------
    *Front.*  LIBERO's agentview is rebuilt around the bowl from its measured distance,
    elevation and bearing, at its own **45 degrees**, and the auto-aim target is moved to the
    bowl so the direction follows.  The previous version equalised millimetres-per-pixel by
    sliding the camera along its existing view direction to 0.722 m and left the field of
    view at 54: a scalar matched, with the perspective still wrong, and the camera still
    aiming at a point that is not the object.

    *Wrist.*  LIBERO's `eye_in_hand` is `pos="0.05 0 0"` on `right_hand`, whose +x is the
    tool axis, at fovy 75.  This arm's wrist camera is 45 mm out along its own tool axis --
    the same idea -- but **85 mm off it**, aimed at a target body that shares the offset, so
    its optical axis runs *parallel* to the approach axis instead of through it.  Removing
    the lateral term puts the axis through the grasp point, which is where LIBERO's points,
    and 75 degrees matches the field of view.
    """

    return replace(
        cameras,
        # Transferred, not reconstructed: position through the one world offset, orientation
        # unchanged because the offset is a translation.  Three attempts preceded this one --
        # a millimetres-per-pixel match, then a sphere at LIBERO's measured distance and
        # elevation -- and each pinned a scalar while leaving the direction wrong, because
        # LIBERO's camera is not aimed at the object at all.
        front_position_m=LIBERO_AGENTVIEW_MATCHED_POS_M,
        front_quat_wxyz=LIBERO_AGENTVIEW_QUAT_WXYZ,
        front_fovy_deg=LIBERO_AGENTVIEW_FOVY_DEG,
        # Inert while the camera is fixed -- MuJoCo reads a target only in `targetbody`
        # mode -- but left on the bowl so the scene still records what the shot is about.
        workspace_target_m=LIBERO_MATCHED_FOCUS_M,
        left_wrist_position_m=(0.0, LIBERO_MATCHED_WRIST_AXIAL_M, 0.0),
        left_wrist_target_m=(
            0.0,
            LIBERO_MATCHED_WRIST_AXIAL_M + LIBERO_MATCHED_WRIST_LOOKAHEAD_M,
            0.0,
        ),
        wrist_fovy_deg=LIBERO_MATCHED_WRIST_FOVY_DEG,
    )


#: Where LIBERO's ``akita_black_bowl`` mesh settles in this scene, in world XY.  Measured,
#: not configured: `configs/libero_props.yaml` drops the wrapper at (0.150, 0.230) and
#: LIBERO authors the object with an inner body offset 100 mm towards -y, so the mesh --
#: the thing a camera sees and a gripper touches -- lands here.
LIBERO_BOWL_MESH_XY_M = (0.150, 0.130)

#: The props axis 4 of the ladder removes.  Both are `akita_black_bowl`, the object the
#: checkpoints were trained on, and leaving either in place would leave a *easier* object
#: than the Cookie on the table for the policy to choose instead.
LIBERO_BOWL_PROP_NAMES = ("bowl", "bowl_near")

#: Axis 4's instruction: LIBERO's own sentence with the object swapped, and its
#: disambiguating clause dropped because this scene has one graspable object.  The target is
#: kept -- "the plate" is the same plate, in the same place, as the sentence the checkpoints
#: were trained on.
COOKIE_ON_TABLE_PROMPT = "pick up the yellow block and place it on the plate"


def libero_props_scene(
    *,
    visual_style: str = "plain",
    matched_cameras: bool = False,
    panda: bool = False,
    cookie_on_table: bool = False,
) -> CollectionScene:
    """The A3 table with LIBERO's own bowl and plate on it, and no Cookie task.

    Why this scene exists, and what its numbers mean
    ------------------------------------------------
    Nothing here can be scored.  `success` still asks whether the Cookie is in the target
    bin, the Cookie is off the table with the shrunk trays, and no expert can reach it --
    so every episode ends in failure by construction and a success rate of 0% here carries
    no information at all.  What the scene is for is **watching**: it is the one
    arrangement where a policy's failure has a meaning, because
    ``lerobot/pi05_libero_finetuned`` has picked up this exact bowl and placed it on this
    exact plate tens of thousands of times in training.  Whatever it does here is a
    statement about the arm and the renderer rather than about whether it can see the
    object.

    The scene is a thin wrapper over `configs/libero_props.yaml`, which holds the props and
    the explanation.  The task string is deliberately *not* overridden here: the scene
    declares the Cookie prompt like every other, and a caller that wants LIBERO's language
    passes it explicitly, so a report cannot silently pair LIBERO's words with A3's task.

    ``terminate_on_success=False`` because there is no success to terminate on; the
    episode should run its full budget so the recording shows what the policy does after
    the point where a Cookie task would have stopped.
    """

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        loaded = _styled_config("configs/libero_props.yaml", visual_style)
        if matched_cameras:
            loaded = replace(loaded, cameras=_libero_matched_cameras(loaded.cameras))
            # **The arm is deliberately left at the A3 deployment home, and that is a
            # measurement rather than an omission.**  LIBERO starts every episode with its
            # end effector at (-0.2112, -0.0111, 1.1742) and this arm starts 362 mm away, so
            # `scripts/solve_libero_home.py` was written to move it there -- and the closest
            # reachable pose is **26 mm** short with the tool 6 degrees off vertical (102 mm
            # short, if the orientation is held as well), because `L_WRIST_P` spans only
            # +/-40 degrees.  Doing it anyway cost the task: the fingers got over the bowl in
            # **4 of 40** episodes against **39 of 40** from the A3 home, and placements went
            # from 3 to 1.  A pose that rides the shoulder-roll limit is not a better place to
            # start from than a pose the arm was built for.
            #
            # `deployment_home` is what `A3CookieTransferEnv.reset` actually applies -- it
            # writes `config.home` first and then overwrites the arm's qpos from here, so an
            # edit to `home` is thrown away at the top of every episode.  That cost a run too.
            deployment = loaded.cookie_transfer.deployment_home
            if panda:
                # LIBERO's robot, LIBERO's start pose, and -- because the scene's own assets
                # arrive with it -- nothing else to do.  The arm's angles are LIBERO's, read
                # from its reset; the gripper entries are left as they are (1.0, fully open),
                # which is also how LIBERO starts.
                loaded = replace(loaded, left_arm="panda")
                loaded = replace(
                    loaded,
                    cookie_transfer=replace(
                        loaded.cookie_transfer,
                        deployment_home=tuple(LIBERO_PANDA_HOME_RAD) + tuple(deployment[7:]),
                    ),
                )
                # **The servo is deliberately left at the A3's tuning.**  A stiffer one was
                # tried and is worse than useless here: `arm_servo_bandwidth` at 4.0 -- kp
                # 2400 on a 2 ms step -- sends this arm 26.8 degrees off home over the
                # settling steps, because the A3's gains and the A3's integrator step were
                # chosen together.  At 1.0 the arm sags 3.75 degrees at the elbow instead,
                # 42 mm at the tool, and holds.
        if cookie_on_table:
            # **Axis 4 of the ladder: the object changes and nothing else.**  Same camera,
            # same robot, same physics, same `pruned_init` start pose, same plate in the
            # same place -- the reachable object goes from LIBERO's bowl, which the policy
            # was trained on, to A3's own Cookie, which appears in no public dataset.
            #
            # `cookie_source_positions_m` is a *world* position (`model._add_cookie_scene`
            # parents it straight onto `world`), so this does not move the source bin: the
            # bin stays parked past the table's corner and the Cookie is dropped onto the
            # table instead of falling with the tray.
            #
            # The two bowls come out.  "The object on the table" has to be the Cookie and
            # only the Cookie, or the policy can satisfy the instruction by picking up the
            # object it was trained on and the axis measures nothing.
            loaded = replace(
                loaded,
                cookie_transfer=replace(
                    loaded.cookie_transfer,
                    cookie_source_positions_m=(LIBERO_BOWL_MESH_XY_M,),
                    libero_props=tuple(
                        prop
                        for prop in loaded.cookie_transfer.libero_props
                        if prop.name not in LIBERO_BOWL_PROP_NAMES
                    ),
                ),
            )
        task_config = CookieTransferTaskConfig(
            cookie_count=1,
            required_cookies=1,
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
            require_exact_slots=False,
            require_released=True,
            terminate_on_success=False,
        )
        return A3CookieTransferEnv(loaded, task_config=task_config, render_cameras=True)

    def build_policy(env: A3DualArmEnv) -> Policy:
        # Unused by any recording this scene is for, but `CollectionScene` requires one,
        # and the single-piece expert is the honest choice: it is the expert the props
        # scene's Cookie layout was reduced for, and it ignores the props entirely.
        return CookieCollectionPolicy(
            expert=A3SinglePieceExpert(env),  # type: ignore[arg-type]
            hold_steps=0,
        )

    return CollectionScene(
        # The instruction has to name what is on the table.  `a3_libero_panda` keeps the
        # A3 Cookie prompt and takes LIBERO's words on the command line, because there the
        # instruction is a separate axis; here the object *is* the axis, so a sentence about
        # a bowl would be asking about the wrong thing.
        task=(
            replace(TASKS["a3_cookie_single_piece"], prompt=COOKIE_ON_TABLE_PROMPT)
            if cookie_on_table
            else TASKS["a3_cookie_single_piece"]
        ),
        build_env=build_env,
        build_policy=build_policy,
        metric_keys=("cookies_in_target", "cookies_in_source"),
        randomize_placements=False,
        randomize_cookies=False,
        details={
            "config": "configs/libero_props.yaml",
            "props": (
                "LIBERO `plate`/`ramekin`/`cookies` as scenery, with the A3 Cookie dropped "
                "where LIBERO's bowl mesh settles -- axis 4 of the ladder"
                if cookie_on_table
                else "LIBERO `akita_black_bowl` and `plate`, as passive scenery"
            ),
            "cameras": (
                "LIBERO-matched: the agentview rebuilt around the bowl at LIBERO's own "
                "0.96 m / 39.8 deg / 163.4 deg / fovy 45, wrist on the tool axis at fovy 75"
                if matched_cameras
                else "this branch's: front at 1.150 m (4.58 mm/px), wrist 85 mm off axis"
            ),
            "scorable": False,
            "why": "no reachable Cookie, so failure is by construction; use it for video",
        },
    )


def registered_scenes() -> dict[str, Callable[[], CollectionScene]]:
    """Scene name -> builder.  This is the list a CLI or a sweep iterates."""

    return {
        "a3_grasp": grasp_scene,
        "a3_cookie_transfer": cookie_transfer_scene,
        "a3_cookie_batch": cookie_batch_scene,
        "a3_cookie_same_column": cookie_same_column_scene,
        "a3_cookie_two_box": cookie_two_box_scene,
        "a3_cookie_single_piece": cookie_single_piece_scene,
        "a3_libero_props": libero_props_scene,
        # The same scene with the camera geometry a LIBERO policy was trained under.  A
        # separate name rather than a flag on the CLI because the two produce numbers that
        # must not be compared, and a name is the only thing that survives into a report.
        "a3_libero_props_matched": lambda **kwargs: libero_props_scene(
            matched_cameras=True, **kwargs
        ),
        # LIBERO's cameras, LIBERO's objects, LIBERO's fixtures -- and LIBERO's own robot on
        # the left arm.  The one scene where a failure cannot be about the arm, because the
        # arm is the reference one; see `model._add_libero_panda`.
        "a3_libero_panda": lambda **kwargs: libero_props_scene(
            matched_cameras=True, panda=True, **kwargs
        ),
        # Axis 4 of the ladder: `a3_libero_panda` with A3's own Cookie in the bowl's place
        # and the two bowls taken out.  The only axis that differs from `a3_libero_panda` is
        # the object, which is what makes the pair a measurement rather than two scenes.
        "a3_libero_panda_cookie": lambda **kwargs: libero_props_scene(
            matched_cameras=True, panda=True, cookie_on_table=True, **kwargs
        ),
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
