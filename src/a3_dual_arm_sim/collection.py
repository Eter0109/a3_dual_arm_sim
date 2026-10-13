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
from .runner import EpisodeRunner, RecordedAction, stored_action_mode
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
    #: Which of ``env.step``'s two arrays becomes the dataset's ``action``.
    #:
    #: ``"applied"`` is the environment's own canonical command, which is what every
    #: dataset collected before this was a choice contains.  ``"requested"`` is what the
    #: *policy* emitted, and it is the only honest choice when the two differ: under
    #: operational-space control the applied action is joint torque, so a scene with
    #: LIBERO's Panda on the left arm has to record the request or ship Newton-metres in a
    #: column a reader takes for joint angles.  The dataset's declared width follows from
    #: this -- see `runner.stored_action_mode` and `recording.ACTION_WIDTH` -- so it also
    #: decides whether the ``action`` column is 16 wide or 14.
    recorded_action: RecordedAction = "applied"
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
    # The dataset's schema is fixed when the file is created, so the space has to be known
    # here rather than discovered from the first frame.  Asked once, of the same function the
    # runner uses, and the recorder asserts the two agree before an episode starts.
    recorded_mode = stored_action_mode(env, scene.recorded_action)
    recorder = LeRobotV3Recorder(
        destination,
        repo_id=repo_id,
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
        action_mode=recorded_mode,
        use_videos=use_videos,
        rollover_mb=rollover_mb,
    )
    runner = EpisodeRunner(
        env,
        policy,
        task=scene.task.prompt,
        recorder=recorder,
        recorded_action=scene.recorded_action,
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


def cookie_batch_second_batch_scene(
    *, hold_steps: int = 40, visual_style: str = "plain"
) -> CollectionScene:
    """`a3_cookie_batch`, with the second batch trained out of its one state.

    The baseline's episodes are all "start empty, do batch 0, do batch 1", so a
    policy trained on them meets the second batch only ever from the state the
    *expert's* first batch leaves behind -- five perfectly upright Cookies at the
    slot centres, which is a state a policy never produces.  Its own first batch
    leaves five imperfectly-placed ones, and from there it is off the
    distribution it was trained on: the shape of every "first five land, second
    five don't" fine-tune so far.

    This scene draws the second-batch condition per episode, by seed, uniformly
    over four modes (see `configs/cookie_batch_second_batch.yaml`, which is
    `cookie_batch.yaml` plus that one list): the full baseline, an episode that
    *starts* at the second batch with five imperfect Cookies already in the box,
    the full episode with the just-placed five nudged at the moment of release,
    and the prefill with wider maxima.  The prompt and the success contract are
    the baseline's own, character for character -- the instruction is the same
    job, and the contract still asks for ten released and seventy remaining,
    which the prefill satisfies by construction (five prefilled plus five
    placed).

    Everything physical is `cookie_batch.yaml`'s, copied rather than restated, so
    the two scenes cannot drift apart on any parameter a grasp depends on.
    """

    return _batch_scene(
        "a3_cookie_batch_second_batch",
        "configs/cookie_batch_second_batch.yaml",
        A3CookieBatchExpert,
        hold_steps=hold_steps,
        details={
            "source_columns": "one exposed end of each of two columns",
            "second_batch_modes": "full / prefill / perturb / prefill_perturb, drawn by seed",
        },
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

#: A second home: the same arm with its hand **over the Cookie**, 120 mm above it, tool still
#: pointing straight down.  Solved on the live model by least squares against the eef site
#: (`scripts/solve_home_over_cookie.py`), starting from `LIBERO_PANDA_HOME_RAD`: residual
#: **0.00 mm** in position and the tool axis exactly `(0, 0, -1)`, which is the pose the
#: wrist camera needs to have the Cookie in shot.
#:
#: Why it exists: the wrist camera looks down the tool axis, so at LIBERO's home it is aimed
#: at the table under the hand, 330 mm to one side of the Cookie, and the policy only sees the
#: object once the arm happens to wander over the bin ~150 steps in.  This start pose puts it
#: there at step 0.
#:
#: **It is not a candidate home.**  Moving the start pose away from LIBERO's own has been
#: measured to cost the task once already -- on the A3 arm, "the fingers got over the bowl in
#: 4 of 40 episodes against 39 of 40 from the A3 home" -- so a result from this pose is a
#: result about the pose as much as about the camera.
LIBERO_PANDA_HOME_OVER_COOKIE_RAD = (
    0.106614, 0.392458, 0.072980, -2.129438, -0.047935, 2.520412, 0.434565,
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


def dense_cookie_layout(offset: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """The batch scene's 4 x 20 packing, translated into another scene's frame.

    Read from `configs/cookie_batch.yaml` rather than copied here: eighty positions copied
    is eighty chances to disagree with the packing they came from.

    The translation is the scene's own ``a3_layout_offset_m``, and that it is *only* a
    translation is a measured fact rather than a convention: the two-Cookie scene's pair of
    Cookies resolve to indices 2 and 10 of this grid once the offset is removed, which is
    what that scene's registration comment already says it chose ("rows 2 and 10 rather
    than 0 and 8").  The two scenes are the same arrangement in two places on the table,
    so the arrangement is stated once.
    """

    batch = _styled_config("configs/cookie_batch.yaml", "plain").cookie_transfer
    return tuple(
        (float(x) + float(offset[0]), float(y) + float(offset[1]))
        for x, y in batch.cookie_source_positions_m
    )


def libero_props_scene(
    *,
    task_name: str,
    visual_style: str = "plain",
    matched_cameras: bool = False,
    panda: bool = False,
    cookie_on_table: bool = False,
    a3_single_cookie: bool = False,
    a3_layout_offset_m: tuple[float, float] = (0.0, 0.0),
    #: Slide the **robot** instead of the task, by this much of world XY, in metres.
    #:
    #: The two offsets are not interchangeable and the difference is the whole reason this
    #: parameter exists.  ``a3_layout_offset_m`` moves the bins, the box and the Cookies, which
    #: is what puts them inside LIBERO's fixed frame -- and it moves them away from the arm that
    #: has to reach them.  Measured with the batch expert's own reset IK on the dense packing:
    #: the A3 arm solves the task with the layout where A3 puts it and misses the target by
    #: 5.71 / 7.73 / 11.61 mm at ``dy`` = -0.15 / -0.20 / -0.25, against a 4.00 mm tolerance,
    #: while the Panda solves every one of them.  So on the A3 arm the scene as registered
    #: cannot be done at all: the placement LIBERO's camera demands is outside its reach, and
    #: the placement it can reach is clipped into the frame's left edge (measured: the source
    #: bin and the box sit in the bottom-left corner with the packing half out of frame, and
    #: `pi05_libero_finetuned` answers that frame with 500 steps of ``motion mean 0.000`` --
    #: the model holds, because there is nothing in the picture it recognises).
    #:
    #: Sliding the robot by the amount the layout was slid restores the *relative* geometry the
    #: arm needs while leaving the task where LIBERO's camera already frames it, and it changes
    #: no camera at all -- which is the axis the scene's own history decided not to spend
    #: (`11f5ffb`, reversing `a99c876`'s re-aimed agentview).  Defaulted to the identity: a
    #: scene that does not pass it is byte-for-byte the scene it was.
    a3_robot_offset_m: tuple[float, float] = (0.0, 0.0),
    a3_cookie_position_m: tuple[float, float] | None = None,
    #: Several Cookies at once, for the stages that place more than one.  The singular
    #: ``a3_cookie_position_m`` is what the single-Cookie scenes use and stays as it is; this is
    #: only read when it is given, and the Cookie count follows its length rather than being one.
    a3_cookie_positions_m: tuple[tuple[float, float], ...] | None = None,
    a3_cookie_quat_wxyz: tuple[float, ...] = (),
    a3_target_box_position_m: tuple[float, float, float] | None = None,
    a3_home_over_cookie: bool = False,
    #: Drive the arm with Cartesian deltas rather than raw joint targets.  Defaulted from
    #: ``panda`` rather than to ``False``, because the two are one fact and leaving them
    #: independent is how they came apart: the policy a scene builds has to emit in the space
    #: its own arm is driven in, and LIBERO's Panda is driven by operational-space control, so a
    #: ``panda=True`` scene that reported ``False`` here built a *joint*-command policy for a
    #: *torque* arm.  All twenty gate and probe scripts passed both flags by hand and agreed;
    #: only the registry in this module passed ``panda`` alone, and so only `collect` -- which
    #: runs the registry -- built the broken pair.  The symptom was not an error: the arm simply
    #: never reached a Cookie, and twenty episodes of 750-1100 steps each ended with
    #: ``cookies_in_source = 2``.
    #:
    #: A caller may still force either value; this is a default, not a rule.
    cartesian: bool | None = None,
    hold_steps: int = 40,
    #: Run the scene with the **five-at-a-time** expert instead of the one-at-a-time one.
    #:
    #: The single-piece expert is the honest choice for a scene that offers one Cookie, and it
    #: is what every branch here used.  The dense packing is not that scene: its task asks for
    #: **ten** released and its prompt says "five ... then five more ... all ten slots", and a
    #: one-at-a-time state machine cannot express that.  Until this flag existed, the scene had
    #: never been run by the workflow it describes, so every failure recorded against it was a
    #: fact about the wrong controller.
    #:
    #: This changes the *expert* and nothing else -- the environment, the arm, the cameras and
    #: the control parameters are the ones the scene had.
    batch_expert: bool = False,
    #: Keep LIBERO's own physics block.  **On by default**, because it is what makes a
    #: zero-shot result mean something: `libero_fidelity` puts the Panda on LIBERO's contact
    #: model -- an explicit `Euler` integrator, `impratio` 20 and `noslip_iterations` 0, all
    #: set in `model.py` -- and every scene here keeps it.
    #:
    #: A scene that exists to *collect* the A3 task with a scripted expert is not that case,
    #: and the dense packing does not survive the block.  Measured with the arm commanded to
    #: hold and **no policy at all**, eighty Cookies in the 4 x 20 grid reach **5.60 m/s on the
    #: first control step** under `Euler`/`noslip 0`, against **0.54 m/s** on the packing's own
    #: block and **0.00 m/s** on `a3_cookie_batch`; they then jitter at ~1.3 m/s indefinitely
    #: with the arm perfectly still.  That jitter -- not a grasp -- is what throws a Cookie out
    #: of the bin at 100 m/s during `DESCEND`, and it is why the scripted expert could not
    #: solve the scene.
    libero_physics: bool = True,) -> CollectionScene:
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

    # Resolved once, up here, because ``build_env`` and ``build_policy`` both close over it and
    # the whole point is that they cannot end up disagreeing.  See the parameter's note.
    cartesian = panda if cartesian is None else cartesian

    def build_env(position_noise_m: float, yaw_noise_rad: float) -> A3DualArmEnv:
        loaded = _styled_config("configs/libero_props.yaml", visual_style)
        # See the parameter's note: this is the one scene here that exists to collect the A3
        # task rather than to reproduce LIBERO, and LIBERO's contact block is what makes its
        # eighty-Cookie packing unstable.
        if not libero_physics:
            loaded = replace(loaded, libero_fidelity=False)
        # A3's own block, read once and up here because both the camera and the props need
        # it and they are set in different branches below.
        a3_transfer = (
            _styled_config("configs/cookie_same_column.yaml", visual_style).cookie_transfer
            if a3_single_cookie
            else None
        )
        if matched_cameras:
            # **The camera is LIBERO's and does not move; the layout is what moves.**
            #
            # LIBERO's agentview is copied with its position *and* its orientation, and in
            # this branch that means it is aimed at the spot LIBERO's bowl used to occupy --
            # which here is bare table, with A3's bin and box clipped into the bottom-left
            # corner of the frame.  Something has to give, and it is not the camera: staying on
            # LIBERO's geometry is the point of the configuration, and the camera is the axis
            # with the *known* price (the front fovy costs 12-14 points of 50).
            #
            # An earlier version re-aimed the front camera at A3's layout instead, on the
            # argument that moving the layout breaks the *scripted expert* -- measured: A3's own
            # arm solves this scene 3 times in 3 with the layout where A3 puts it, and 0 in 3 at
            # offsets (0.02, -0.12) and (0.02, -0.20).  That argument is backwards.  The expert
            # is a reference kept for convenience, not the objective; if it cannot reach the
            # arrangement the scene needs, the thing to fix is the expert, and only when it is
            # actually needed -- for collecting data -- rather than bending the scene to it now.
            # The re-aim is in the history at `a99c876` if it is ever wanted.
            #
            # So: LIBERO's camera, position, aim and 45 degrees, with
            # `a3_layout_offset_m` sliding A3's arrangement into that frame.  What the offset
            # costs is now an accepted, recorded cost rather than a surprise.
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
        if a3_single_cookie:
            # **A3's own task, in the scene that is known to work.**  This is the direct
            # question the ladder is a slower way of asking: if the arm, the cameras, the
            # physics and the start poses are all the ones that place LIBERO's bowl 35 times
            # in 40, does the policy do *A3's* job when the objects and the sentence are
            # A3's?
            #
            # Unlike axis 4 this keeps A3's instruction **and** A3's target: the sentence is
            # `a3_cookie_single_piece`'s own ("pick up the one yellow block from the source
            # bin and put it in the target box"), and the source bin and the ten-slot box sit
            # where A3 puts them.  Axis 4 asked something no dataset contains -- "place it on
            # the plate" -- and answered it; this asks the question the task actually is.
            #
            # The props are taken from `configs/cookie_same_column.yaml` **by loading it**
            # rather than by copying numbers, so a later edit to A3's own scene cannot leave
            # this one quietly describing a different table.  The list below is the whole
            # difference: eleven fields of `cookie_transfer`.  Everything else stays as the
            # configuration that measured 35/40, because that is the part known to be right --
            # the LIBERO physics block (`physics_hz 500`, `libero_fidelity`, the 0.40 damping,
            # the 0.05 m control scale, the `world` rotation frame), LIBERO's per-episode
            # `pruned_init` start poses, the matched cameras and the mirror.
            #
            # Two of the eleven are worth naming because they are not A3's defaults seen from
            # this file.  `box_lift_m` and `box_tilt_deg` are 0.03 m and 8 degrees here -- the
            # box is lifted and canted -- which costs nothing while the box is parked past the
            # table's corner and would be a tilted ten-slot box the moment it is in reach.  A3
            # runs it flat, and so does this.
            a3 = _styled_config("configs/cookie_same_column.yaml", visual_style)
            a3_transfer = a3.cookie_transfer
            # **The offset exists to be measured, and it measured zero.**  Translating the
            # whole layout into the camera's frame looked like the smaller change than
            # re-aiming the camera -- same internal geometry, camera untouched -- and the
            # reference controller rejected it: A3's own arm solves this scene **3 times in 3**
            # with the layout where A3 puts it, and **0 in 3** at both (0.02, -0.12) and
            # (0.02, -0.20).  Where the arrangement sits is load-bearing, so what gives is the
            # camera's aim, in the `matched_cameras` branch above.  The parameter is kept so
            # that result stays reproducible rather than becoming a memory.
            #
            # Written as explicit tuple literals rather than by adding tuples together:
            # the first version of this added a shifted 2-tuple to a `(z,)` and left the
            # trailing comma *inside* the parentheses, which silently produced
            # `((x, y, z),)` -- a one-tuple where a three-tuple was meant -- and the same
            # slip nested `cookie_source_positions_m` one level too deep.  Both surfaced
            # only as `TypeError: unsupported format string passed to tuple.__format__`,
            # which names neither the field nor the shape.
            offset_x, offset_y = a3_layout_offset_m
            robot_x, robot_y = a3_robot_offset_m
            a3_source = a3_transfer.cookie_source_positions_m[0]
            cookie_xy = (
                (
                    a3_source[0] + offset_x,
                    a3_source[1] + offset_y,
                )
                if a3_cookie_position_m is None
                else a3_cookie_position_m
            )
            # One Cookie, or as many as the caller named.  Everything below that used to read
            # `cookie_xy` reads this instead, so the count is in one place.
            cookie_positions = (
                (cookie_xy,) if a3_cookie_positions_m is None else a3_cookie_positions_m
            )
            # The **bin's** row pitch, read off A3's full grid before it is reduced to one or two
            # positions.  The expert needs it to find the groove between two packed neighbours,
            # and taking it from the reduced list would describe the gap between the two pieces
            # this scene happens to place instead -- 70.7 mm rather than 8.83 mm, which made the
            # descent opening 0.51 instead of 0.08 and stopped the grasp verifying.
            full_rows = sorted({float(position[1]) for position in a3_transfer.cookie_source_positions_m})
            cookie_row_pitch = full_rows[1] - full_rows[0] if len(full_rows) > 1 else None
            loaded = replace(
                loaded,
                cookie_transfer=replace(
                    loaded.cookie_transfer,
                    libero_props=(),
                    box_lift_m=a3_transfer.box_lift_m,
                    box_tilt_deg=a3_transfer.box_tilt_deg,
                    cookie_source_positions_m=cookie_positions,
                    cookie_source_row_pitch_m=cookie_row_pitch,
                    cookie_source_quat_wxyz=a3_cookie_quat_wxyz,
                    source_bin_center_m=(
                        a3_transfer.source_bin_center_m[0] + offset_x,
                        a3_transfer.source_bin_center_m[1] + offset_y,
                    ),
                    source_bin_half_size_m=a3_transfer.source_bin_half_size_m,
                    source_bin_wall_height_m=a3_transfer.source_bin_wall_height_m,
                    target_bin_world_position_m=(
                        (
                            a3_transfer.target_bin_world_position_m[0] + offset_x,
                            a3_transfer.target_bin_world_position_m[1] + offset_y,
                            a3_transfer.target_bin_world_position_m[2],
                        )
                        if a3_target_box_position_m is None
                        else a3_target_box_position_m
                    ),
                    target_bin_half_size_m=a3_transfer.target_bin_half_size_m,
                    target_bin_wall_height_m=a3_transfer.target_bin_wall_height_m,
                    target_slots_local_m=a3_transfer.target_slots_local_m,
                    # The **robot**, not the task.  See `a3_robot_offset_m`: the arm has to
                    # travel with the layout when the layout travels, or the reach the scene
                    # needs is gone.  Read off the configuration rather than recomputed from a
                    # literal, so a config that already moved the base keeps its own value.
                    base_xy_m=(
                        loaded.cookie_transfer.base_xy_m[0] + robot_x,
                        loaded.cookie_transfer.base_xy_m[1] + robot_y,
                    ),
                    # ... and the expert's own reference poses have to travel too.  See the
                    # field: a fixed staging point left behind by a translated arrangement is
                    # what made this scene's reset IK refuse at 4.21 mm.
                    layout_offset_m=(offset_x, offset_y),
                    # LIBERO's per-episode start poses are *its* task's archive and put the
                    # arm at the pose for picking up its bowl.  A3's task is a different
                    # reach with a different first move.
                    libero_init_states="",
                    deployment_home=(
                        (
                            tuple(LIBERO_PANDA_HOME_OVER_COOKIE_RAD) if a3_home_over_cookie
                            else tuple(LIBERO_PANDA_HOME_RAD)
                        )
                        if panda
                        else tuple(a3_transfer.deployment_home[:7])
                    ) + tuple(loaded.cookie_transfer.deployment_home[7:]),
                ),
            )
        task_config = CookieTransferTaskConfig(
            cookie_count=len(cookie_positions) if a3_single_cookie else 1,
            required_cookies=len(cookie_positions) if a3_single_cookie else 1,
            position_noise_m=position_noise_m,
            yaw_noise_rad=yaw_noise_rad,
            require_exact_slots=False,
            require_released=True,
            # A3's own scene stops the episode once the Cookie is in and released, and the
            # branch that runs A3's own task has to as well -- otherwise a solved episode runs
            # to the 6000-step horizon and the recording is a stationary arm.  The other two
            # branches keep it off because there is no success for them to terminate on.
            terminate_on_success=bool(a3_single_cookie),
        )
        return A3CookieTransferEnv(
            loaded,
            task_config=task_config,
            action_mode="cartesian_delta" if cartesian else "joint_position",
            render_cameras=True,
        )

    def build_policy(env: A3DualArmEnv) -> Policy:
        # The single-piece expert, which is the honest choice for every branch here: it is the
        # expert the props scene's Cookie layout was reduced for and it ignores the props
        # entirely.
        #
        # **`hold_steps` must not be 0, and it was.**  `CookieCollectionPolicy.stop_requested`
        # is `expert.failed or self._steps_since_done >= self.hold_steps`, so 0 makes the
        # condition true on the very first step and the runner ends the episode at step 1 with
        # `terminated=False` and no safety reason -- a silent failure with no message, because
        # it is a *legitimate* way for a rollout to end.  It cost nothing while nothing here
        # ran the expert; `a3_libero_panda_single_cookie` does, and reported 0/20 with every
        # episode one step long.  The value is A3's own: `cookie_single_piece_scene`'s default.
        mode: ActionMode = "cartesian_delta" if cartesian else "joint_position"
        command_space = "cartesian" if cartesian else "joint"
        expert: A3CookieTransferExpert = (
            A3CookieBatchExpert(  # type: ignore[arg-type]
                env, action_mode=mode, command_space=command_space
            )
            if batch_expert
            else A3SinglePieceExpert(  # type: ignore[arg-type]
                env, action_mode=mode, command_space=command_space
            )
        )
        return CookieCollectionPolicy(
            expert=expert,
            hold_steps=hold_steps,
            action_mode=mode,
        )

    return CollectionScene(
        # The instruction has to name what is on the table.  `a3_libero_panda` keeps the
        # A3 Cookie prompt and takes LIBERO's words on the command line, because there the
        # instruction is a separate axis; here the object *is* the axis, so a sentence about
        # a bowl would be asking about the wrong thing.
        # The instruction has to name what is on the table, and the *acceptance rule* has to
        # be the one belonging to this scene.  `task_name` is required rather than defaulted
        # because this builder serves eight registered scenes and a default would silently
        # hand several of them one scene's rule -- which is what used to happen: every
        # `a3_libero_*` scene declared the task named `a3_cookie_single_piece`, so a summary
        # row could not be traced back to the scene that wrote it and the registry's own
        # consistency test was red.
        task=(
            replace(TASKS[task_name], prompt=COOKIE_ON_TABLE_PROMPT)
            if cookie_on_table
            else TASKS[task_name]
        ),
        build_env=build_env,
        build_policy=build_policy,
        metric_keys=("cookies_in_target", "cookies_in_source"),
        # A3's own scene randomizes both, and this branch has to as well: the per-episode
        # variation in `a3_libero_panda` comes from LIBERO's fifty `pruned_init` start poses,
        # which this branch drops, so with these off every seed would build the same table
        # and the only variation left would be the policy's own sampling.
        randomize_placements=bool(a3_single_cookie),
        randomize_cookies=bool(a3_single_cookie),
        # What the dataset may record, and it is not a preference.  `panda` selects LIBERO's arm,
        # which is driven by operational-space control, so the action the actuators were given is a
        # joint *torque*: it belongs to no action space a policy could emit or a checkpoint could be
        # trained against, and `stored_action_mode` refuses it by name.  The other registrations of
        # this builder (`a3_libero_props`, `a3_libero_props_matched`) drive A3's own position-servoed
        # arm, where the applied command *is* the space, so there `applied` is the honest answer.
        #
        # Not `cartesian`: that flag picks the action *mode* and is orthogonal to which arm is
        # mounted -- this scene sets it not at all while `panda` is True, and keying on `cartesian`
        # left `collect` declaring a schema it could never fill.
        #
        # Consequence of getting it wrong: no dataset at all, because the schema is declared before
        # the first episode.  The gate has always passed `requested` to `EpisodeRunner` by hand,
        # which is why `gate` never saw this and only `collect` did.
        recorded_action="requested" if panda else "applied",
        details={
            "config": (
                "configs/libero_props.yaml with A3's `cookie_transfer` props from "
                "configs/cookie_same_column.yaml -- the validated cameras, robot and "
                "physics with A3's own task on the table"
                if a3_single_cookie
                else "configs/libero_props.yaml"
            ),
            "layout_offset_m": (
                f"{a3_layout_offset_m} -- A3's arrangement slid rigidly into LIBERO's frame"
                if a3_single_cookie
                else None
            ),
            "robot_offset_m": (
                f"{a3_robot_offset_m} -- and the arm slid by the same amount, so the reach "
                "the layout left behind comes back"
                if a3_single_cookie and a3_robot_offset_m != (0.0, 0.0)
                else None
            ),
            "props": (
                "A3's single Cookie, A3's source bin and A3's ten-slot box, flat on the "
                "table, and no LIBERO object at all"
                if a3_single_cookie
                else (
                    "LIBERO `plate`/`ramekin`/`cookies` as scenery, with the A3 Cookie "
                    "dropped where LIBERO's bowl mesh settles -- axis 4 of the ladder"
                    if cookie_on_table
                    else "LIBERO `akita_black_bowl` and `plate`, as passive scenery"
                )
            ),
            "cameras": (
                "LIBERO-matched: the agentview rebuilt around the bowl at LIBERO's own "
                "0.96 m / 39.8 deg / 163.4 deg / fovy 45, wrist on the tool axis at fovy 75"
                if matched_cameras
                else "this branch's: front at 1.150 m (4.58 mm/px), wrist 85 mm off axis"
            ),
            "scorable": bool(a3_single_cookie),
            "why": (
                "A3's own success predicate is reachable here, so this one can be scored"
                if a3_single_cookie
                else "no reachable Cookie, so failure is by construction; use it for video"
            ),
        },
    )


def registered_scenes() -> dict[str, Callable[[], CollectionScene]]:
    """Scene name -> builder.  This is the list a CLI or a sweep iterates."""

    return {
        "a3_grasp": grasp_scene,
        "a3_cookie_transfer": cookie_transfer_scene,
        "a3_cookie_batch": cookie_batch_scene,
        "a3_cookie_batch_second_batch": cookie_batch_second_batch_scene,
        "a3_cookie_same_column": cookie_same_column_scene,
        "a3_cookie_two_box": cookie_two_box_scene,
        "a3_cookie_single_piece": cookie_single_piece_scene,
        "a3_libero_props": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_props", **kwargs
        ),
        # The same scene with the camera geometry a LIBERO policy was trained under.  A
        # separate name rather than a flag on the CLI because the two produce numbers that
        # must not be compared, and a name is the only thing that survives into a report.
        "a3_libero_props_matched": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_props_matched", matched_cameras=True, **kwargs
        ),
        # LIBERO's cameras, LIBERO's objects, LIBERO's fixtures -- and LIBERO's own robot on
        # the left arm.  The one scene where a failure cannot be about the arm, because the
        # arm is the reference one; see `model._add_libero_panda`.
        "a3_libero_panda": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_panda", matched_cameras=True, panda=True, **kwargs
        ),
        # Axis 4 of the ladder: `a3_libero_panda` with A3's own Cookie in the bowl's place
        # and the two bowls taken out.  The only axis that differs from `a3_libero_panda` is
        # the object, which is what makes the pair a measurement rather than two scenes.
        "a3_libero_panda_cookie": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_panda_cookie",
            matched_cameras=True, panda=True, cookie_on_table=True, **kwargs
        ),
        # A3's own single-Cookie task on the table that is known to work: the validated
        # cameras, robot, physics and start poses, with A3's props, A3's sentence and A3's
        # success predicate.  The only `libero_props` branch whose `scorable` is true.
        # `a3_layout_offset_m` slides A3's arrangement into LIBERO's frame, because LIBERO's
        # agentview is aimed at where LIBERO's bowl used to be and A3's layout is not there.
        # The camera is not touched -- see the branch -- so this offset is the price the
        # camera choice costs, and it is a rigid translation of the whole arrangement, which
        # keeps the internal geometry A3's task is about.  Chosen by rendering the first frame
        # at four candidates; (0.02, -0.20) puts both boxes fully inside with margin.
        #
        # `a3_cookie_position_m` and the quarter-turn move the Cookie out of the bin and
        # below the box, and turn it so its **50 x 25 mm face** is what the camera sees.  It
        # was 23 pixels in a 3 x 21 box -- a 6.4 mm edge, 0.035% of the frame -- and it is now
        # **101 pixels in an 18 x 6 box**, a factor of 4.4:
        #
        #   23 px standing in the bin's corner, edge to the camera
        #  101 px on the table below the box, long face to the camera, quarter-turned
        #
        # The position is the *second* of three candidates that all measured 96-102 px, and it
        # is chosen for reach rather than for pixels: it is 0.681 m from the Panda's base
        # against 0.793 m for the one nearest the camera, 0.721 m for the middle.  Nearer the
        # camera and within the arm's reach pull in opposite directions -- the camera sits at
        # x = +0.87 and the arm's base at x = -0.45 -- and the arm has to actually get there,
        # so the trade went to the arm, at a cost of 35 mm of camera distance.
        #
        # **This changes the task, and that is the point of the run.**  The Cookie is no longer
        # in the source bin, so `from the source bin` in the instruction is no longer true of
        # the scene; and a slab turned onto its long face is a different grasp from one
        # presented edge-on.  It is a visibility experiment, not a candidate configuration.
        "a3_libero_panda_single_cookie": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_panda_single_cookie",
            matched_cameras=True, panda=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            a3_cookie_position_m=(0.2343, -0.0932),
            a3_cookie_quat_wxyz=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            **kwargs
        ),
        # Stage 2: two Cookies, fixed layout, one pass each.  Both stand in the **same column**
        # -- the one the wall-clearance sweep measured, where A3's own x is reachable -- and ten
        # rows apart, so at 8.83 mm a row that is about 71 mm of separation.  That is well past
        # the plan's measured minimum and leaves the second Cookie undisturbed when the first
        # is lifted out, which is the hazard this stage exists to find.
        #
        # Rows 2 and 10 rather than 0 and 8: the sweep says a Cookie nearer than about 10 mm
        # beyond A3's placement costs 275 extra steps an episode, and row 0 is A3's placement.
        "a3_libero_panda_two_cookies": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_panda_two_cookies",
            matched_cameras=True, panda=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            a3_cookie_positions_m=((0.1194, 0.04875), (0.1194, 0.11941667)),
            **kwargs
        ),
        # **The same sparse scene on the A3 arm, with the robot slid into the task.**  The
        # two-Cookie scene is the one worth asking the arm question on: unlike the eighty-Cookie
        # packing, its task is one the *reference* arm completes, and the two Cookies have ~71 mm
        # of clear space between them, so a grasp that works here is a statement about the arm
        # rather than about the packing.  Everything else is the entry above, character for
        # character -- same placement, same absolute Cookie positions, same matched camera pair,
        # same LIBERO physics block, same prompt, same single-piece expert -- except `panda=False`
        # with its consequence `cartesian=True`, and `a3_robot_offset_m` equal to the layout
        # offset, which is what puts the arm where the task now is (see that parameter).
        "a3_libero_two_cookies_libero_mount": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_two_cookies_libero_mount",
            matched_cameras=True, panda=False, cartesian=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            a3_cookie_positions_m=((0.1194, 0.04875), (0.1194, 0.11941667)),
            a3_robot_offset_m=(0.020, -0.200),
            **kwargs
        ),
        # Stage 4: the dense packing, on the embodiment the rest of this pipeline is built
        # on.  Only the *scene* comes from `a3_cookie_batch` -- eighty Cookies in a 4 x 20
        # grid at 50.4 mm and 8.833 mm, five at a time into the 2 x 5 box, same bin, same
        # bevels.  Everything about *how the arm works* is this file's: LIBERO's Panda under
        # `OperationalSpaceAdapter`, the LIBERO-matched camera pair, and `physics_hz`,
        # `max_joint_step_rad`, the servo bandwidth and the fingertip shim exactly as the
        # two-Cookie scene takes them.
        #
        # That split is the point rather than a convenience.  A scene is also an embodiment,
        # and swapping the arm or the camera leaves the videos looking like Cookies while
        # every checkpoint trained on the old embodiment is being asked about a different
        # robot -- a mismatch that reads as a capability failure.  The packing is the part
        # that is genuinely new information.
        #
        # **The fingertip shim does not come along, and this scene is not solvable without it.**
        # This comment used to say the opposite -- that `left_finger_pad_half_thickness_m` (0.0005
        # m, already in `configs/cookie_same_column.yaml`) comes along with the packing.  It does
        # not: `model._add_gripper` applies that field to the gripper *it* builds, i.e. A3's own
        # Robotiq pads, and LIBERO's Panda brings its own fingers.  The scene was therefore
        # designed around a shim that was never installed, which is why every attempt to run the
        # workflow it describes ended in `DESCEND` at 20-130 N on a fingertip.
        #
        # What the Panda's fingers measure, on this scene, in the closing direction:
        #
        #   `gripper0_finger{1,2}_pad_collision`   an 8 mm box, contact face 12.45 mm from the
        #                                          finger geom origin the expert measures
        #                                          separation between -- so the hand is ~24.9 mm
        #                                          thick at the point of contact;
        #   `L_finger_{inner,outer}_geom`          the finger body's mesh, ~27 mm deep, and the
        #                                          part that actually meets a neighbour.
        #
        # and what the packing leaves for them:
        #
        #   groove between two packed Cookies            2.50 mm   (8.833 pitch - 6.33 thick)
        #   a column's high end, to the bin wall        15.1 mm
        #   a column's low end, to the bin wall          3.3 mm
        #
        # Neither placement exists -- the hand can neither enter a groove nor stand outside the
        # run -- and the failure is one-sided, which is the signature: measured on step 18 of
        # `DESCEND`, `L_finger_outer_geom <-> cookie_5` and `cookie_6` (the next two pieces of the
        # same twenty-long column) at `dist -0.36/-0.38 mm`, 64/83 N, with the other jaw at zero.
        #
        # Note that this is a fact about the *hand* and the *packing together*, not about batch
        # size: a single Cookie from inside a column needs the same 27 mm of y beside it and has
        # 2.50 mm.  `a3_libero_panda_single_cookie` and `a3_libero_panda_two_cookies` work because
        # their one or two Cookies have ~70 mm of clear space beside them, and that clear space is
        # exactly what this packing removes.  The same workflow on the **A3 arm** -- whose own pad
        # is 1.00 mm thick, so the grooves are 0.75 mm a side -- reaches `DONE` in 539 steps with
        # 10 in the target box and 70 in the source: `a3_cookie_batch`.
        "a3_libero_panda_dense_batch": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_panda_dense_batch",
            matched_cameras=True, panda=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            a3_cookie_positions_m=dense_cookie_layout((0.020, -0.200)),
            # The workflow this scene's prompt and contract describe: five at a time, twice.
            batch_expert=True,
            # ... and the packing's own physics block, not LIBERO's; see the parameter.
            libero_physics=False,
            **kwargs
        ),
        # **The same scene, the same packing, the same cameras, the same prompt -- on the A3 arm.**
        # This is the one-variable counterpart of the entry above, and it exists because that one
        # cannot be finished by any controller on its own embodiment (LIBERO's jaws are ~25 mm thick
        # against a 2.50 mm groove, and a column's ends have 3.3 / 15.1 mm), so a policy's behaviour
        # there is measured against a scene with no solution to find.  Here the packing is inside
        # the reach of a hand that can actually take five at a time.
        #
        # `panda=False` is the only change to the arm, and with it the deployment home
        # (`deployment_home` is LIBERO's Panda's angles in that branch, and A3's own pose is the
        # one this arm was built for) and the gripper.  `cartesian=True` is *not* a second change --
        # it is the arm's own consequence, kept explicit because `cartesian` defaults from `panda`
        # and a silent flip to `joint_position` would pair a `cartesian_delta` policy with a
        # joint-command environment, which is the exact failure the parameter's note describes.
        #
        # **The offset is zero, and that is a measurement rather than a preference.**  The entry
        # above slides A3's arrangement by `(0.020, -0.200)` so that it lands where LIBERO's fixed
        # camera is aimed -- which is in front of LIBERO's Panda.  A3's arm is not there: measured
        # with the batch expert's own reset IK, the transit pose at `-0.200` is **7.73 mm** out of
        # reach against a 4.00 mm tolerance, and at `-0.120` it is 5.44 mm, while `0.000` and
        # `-0.050` both solve.  A swept grid of the two arms over `dx` in {-0.02, 0, +0.02} and `dy`
        # in {-0.25 ... +0.05} puts the boundary in the same place: A3 fails for every `dy` at or
        # below -0.10 and works at or above -0.05, and the Panda works everywhere tried.  So **no
        # single placement lets LIBERO's camera frame the arrangement and A3 reach it**: at `dy`
        # >= -0.05 the layout sits at the frame's lower-left edge, with the source bin's left
        # margin clipped, which is the price of this scene existing at all.
        #
        # Cameras, prompt, contract, cookie positions and the physics block are otherwise read
        # from the entry above unchanged.
        "a3_libero_dense_batch": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_dense_batch",
            matched_cameras=True, panda=False, cartesian=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.0, 0.0),
            a3_cookie_positions_m=dense_cookie_layout((0.0, 0.0)),
            batch_expert=True,
            libero_physics=False,
            **kwargs
        ),
        # **The Panda at the A3 placement, for the controlled pair.**  The two entries together are
        # the same task, packing, camera, box and physics with one variable between them -- the arm
        # -- which is what the A3 entry above cannot be against the original dense scene, because
        # there the placement differs by 200 mm as well.  Each arm is driven through its own
        # state calibration (`configs/policies/pi05_libero_a3_cartesian.yaml` against
        # `pi05_libero_panda.yaml`), so the policy receives LIBERO's state convention on either.
        "a3_libero_panda_dense_batch_a3_placement": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_panda_dense_batch_a3_placement",
            matched_cameras=True, panda=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.0, 0.0),
            a3_cookie_positions_m=dense_cookie_layout((0.0, 0.0)),
            batch_expert=True,
            libero_physics=False,
            **kwargs
        ),
        # **The A3 arm at LIBERO's placement, with the robot slid to meet the task.**  The two
        # entries above are a dilemma and this is the way out of it: the Panda works at
        # `(0.020, -0.200)` because LIBERO's fixed camera frames that spot and the Panda reaches
        # it, the A3 arm works at `(0.000, 0.000)` because that is where its own workspace is --
        # and neither placement satisfies both.  Sliding the *arm* by the amount the layout was
        # slid removes the dilemma without touching the camera: the relative geometry is
        # `(0.000, 0.000)`'s, which the batch expert solves, and the frame is the one the Panda
        # entry was measured on, which is where the packing is legible.
        #
        # Why this matters beyond the expert: with the layout at LIBERO's spot and the arm left
        # at home, `pi05_libero_finetuned` does not move at all on the A3 arm -- 500 steps of
        # `motion mean 0.000` against `0.008` on the Panda, and the front frame says why (the
        # task is a clipped sliver at the frame's left edge and the model holds).  So the arm's
        # position is a *policy input*, not bookkeeping, and this is the scene that tests it
        # with everything else held: same camera, same prompt, same packing, same box, same
        # physics, same layout -- the arm and its mount are the one variable.
        "a3_libero_dense_batch_libero_mount": lambda **kwargs: libero_props_scene(
            task_name="a3_libero_dense_batch_libero_mount",
            matched_cameras=True, panda=False, cartesian=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            a3_cookie_positions_m=dense_cookie_layout((0.020, -0.200)),
            a3_robot_offset_m=(0.020, -0.200),
            batch_expert=True,
            libero_physics=False,
            **kwargs
        ),
        # ------------------------------------------------------------------ behaviour probes
        #
        # These two are **not** candidate configurations for A3's task.  They are probes of
        # what the policy does when the scene stops offering the thing it has been choosing,
        # and they both keep the Cookie in the source bin's corner, standing, exactly as A3
        # draws it -- the moved-and-turned Cookie of the visibility run above is specific to
        # that question.
        #
        # Why they exist: with the box on the table, ten seeds in two runs picked the box up
        # and carried it into the bin, and the Cookie was never touched.  The question the
        # next run asks is what the policy does when that move is **already done** -- either
        # because the destination is occupied, or because it cannot see the Cookie until the
        # arm has gone where the Cookie is.
        #
        # `a3_probe_box_in_bin`: the target box starts **inside** the source bin, placed at the
        # bin's centre so it neither touches nor covers the Cookie, which is at the bin's
        # near-left corner.  Nothing else changes -- same objects, same appearance, same
        # instruction.  If the policy reaches for the Cookie here, its choice depends on the
        # *state* rather than only on how the objects look, which would be worth knowing.
        "a3_probe_box_in_bin": lambda **kwargs: libero_props_scene(
            task_name="a3_probe_box_in_bin",
            matched_cameras=True, panda=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            # The bin's centre after the offset is (0.195, 0.115); its inner half size is
            # (0.1126, 0.0951), so a box centred here spans y 0.080..0.150 and the Cookie at
            # y = 0.031 is 49 mm clear of it.  Spawned at 0.78 and dropped: it rests on the
            # bin floor rather than the table.
            a3_target_box_position_m=(0.195, 0.115, 0.780),
            **kwargs
        ),
        # `a3_probe_home_over_cookie`: the arm starts with its hand over the source bin rather
        # than at LIBERO's home, so the wrist camera -- which looks straight down the tool
        # axis -- has the Cookie in shot at step 0 instead of at step 150.
        #
        # **This deliberately spends something the repository has already priced.**  Moving
        # the start pose away from LIBERO's own cost the bowl task badly the last time it was
        # tried: "the fingers got over the bowl in **4 of 40** episodes against **39 of 40**
        # from the A3 home".  That was the A3 arm, and this is the Panda, so it is not the
        # same measurement -- but the direction of the risk is known and it is recorded here
        # so the result is not read as a clean test of "the camera sees the object".
        "a3_probe_home_over_cookie": lambda **kwargs: libero_props_scene(
            task_name="a3_probe_home_over_cookie",
            matched_cameras=True, panda=True, a3_single_cookie=True,
            a3_layout_offset_m=(0.020, -0.200),
            a3_home_over_cookie=True,
            **kwargs
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
