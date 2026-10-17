from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .contracts import EpisodeContext, jsonable
from .env import A3DualArmEnv
from .policy import Policy
from .recording import Recorder

#: Which of the two arrays ``env.step`` returns is written as an episode's ``action``.
#: ``applied`` is the environment's canonical command and is what every scene recorded before
#: this was a choice; ``requested`` is what the policy emitted.
RecordedAction = Literal["applied", "requested"]


def stored_action_mode(env: A3DualArmEnv, recorded_action: str) -> str:
    """Name the action space the array we are about to record belongs to.

    One definition, because two callers need the same answer and a mismatch between them is
    exactly the defect this exists to prevent: ``collect_dataset`` has to declare the dataset's
    schema *before* the first episode and the runner has to declare what it wrote, and a schema
    of one width holding an action of another is caught by LeRobot only on the first frame.

    ``requested`` is the policy's own space by construction.  ``applied`` is the environment's,
    and where the left arm is torque-driven the environment refuses to put a name on it -- see
    `A3DualArmEnv.canonical_action_mode` -- because there is no action space whose entries are
    Newton-metres and inventing one is how torques end up in a jaw-angle column.
    """

    if recorded_action not in ("applied", "requested"):
        raise ValueError(
            f"recorded_action must be 'applied' or 'requested', got {recorded_action!r}"
        )
    if recorded_action == "requested":
        return str(env.action_mode)
    return env.canonical_action_mode


def _jsonable(value: Any) -> Any:
    """Coerce numpy to plain Python; see :func:`a3_dual_arm_sim.contracts.jsonable`."""

    return jsonable(value)


def _peakable(value: Any) -> float | None:
    """An info entry as a number a running maximum can be taken of, or ``None``.

    Deliberately narrow.  An environment's ``info`` carries action vectors and phase
    labels beside its task counters, and folding those into a maximum would either
    raise or invent a number.  ``bool`` is admitted on purpose: ``success`` is one,
    and "did this ever become true" is the single most useful peak in the mapping.
    """

    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, (np.ndarray, list, tuple, dict, set, frozenset)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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
    #: The largest value each scalar entry of ``final_info`` reached during the
    #: episode, keyed the same way.  Read it as "did this happen at all", which the
    #: final value cannot answer for an episode nothing cut short -- see
    #: :meth:`EpisodeRunner.run`.  Empty when the episode never stepped.
    peak_info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, with numpy converted to builtins."""

        payload = asdict(self)
        payload["final_info"] = _jsonable(dict(self.final_info))
        payload["peak_info"] = _jsonable(dict(self.peak_info))
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
        recorded_action: RecordedAction = "applied",
        post_reset: Callable[[Any, int], None] | None = None,
    ) -> None:
        if policy.action_mode != env.action_mode:
            raise ValueError(
                f"policy action_mode={policy.action_mode} does not match env {env.action_mode}"
            )
        if recorded_action not in ("applied", "requested"):
            raise ValueError(
                f"recorded_action must be 'applied' or 'requested', got {recorded_action!r}"
            )
        self.env = env
        self.policy = policy
        self.task = task
        self.recorder = recorder
        self.realtime = realtime
        self.save_failed_episodes = save_failed_episodes
        #: Which of the two arrays ``env.step`` returns is written as the dataset's
        #: ``action``.  ``applied`` is the environment's own canonical command and is what
        #: every scene recorded before this was a choice; ``requested`` is what the policy
        #: emitted, which is the space a fine-tune trains on and the only one that exists
        #: for a torque-driven arm.
        #:
        #: The distinction is not cosmetic.  Under `OperationalSpaceAdapter` the applied
        #: action's first seven entries are Newton-metres; recording them would put torques
        #: in a field a reader takes for joint angles, and the old hardcoded
        #: ``stored_action_mode = "joint_position"`` would have agreed with the reader.  So
        #: the mode is asked of the environment (`canonical_action_mode`) rather than
        #: assumed, and it raises for an arm whose command is not an action space at all.
        self.recorded_action = recorded_action
        #: Extra ``env.reset`` options, so a caller can control a scene's own
        #: randomisation without wrapping the environment.  The collection driver
        #: uses this to pass the two switches the batch scenes need set
        #: differently: boxes and start pose move, per-Cookie jitter stays off,
        #: because their insertion is aimed at 2.5 mm gaps a jitter would close.
        self.reset_options = dict(reset_options) if reset_options else None
        #: Called with ``(env, seed)`` right after ``env.reset``, and the observation is
        #: re-read afterwards.  Collection uses it to place the scene on an arrangement the
        #: reset would not produce by itself (see `main_alignment.apply_scene`): the scene
        #: has to be rewritten *after* reset, because reset re-derives the layout, and the
        #: observation has to be re-read, because the one reset returned describes the old
        #: layout.
        self.post_reset = post_reset

    def run(self, *, seed: int = 0, max_steps: int | None = None) -> EpisodeResult:
        context = EpisodeContext(seed=seed, task=self.task, action_mode=self.env.action_mode)
        # Asked before the first step when something is going to be *written*, because the
        # dataset's schema is created from this same answer, the recorder asserts the two
        # agree, and an impossible combination should fail on a scene that has not yet run
        # rather than 500 episodes into a collection.
        #
        # Not asked otherwise.  `stored_action_mode` names the space of the recorded column,
        # and on a torque arm it *refuses* to name the applied command -- there is no action
        # space whose entries are Newton-metres (see `A3DualArmEnv.canonical_action_mode`).
        # Resolving it unconditionally therefore failed every caller that never records, on
        # every LIBERO-Panda scene, with an error about a column that does not exist; `compare`
        # was one, and its report said `failure:` with no episodes rather than anything a
        # reader could connect to an action space.
        mode = (
            stored_action_mode(self.env, self.recorded_action)
            if self.recorder is not None
            else None
        )
        observation, _ = self.env.reset(seed=seed, options=self.reset_options)
        if self.post_reset is not None:
            self.post_reset(self.env, seed)
            observation = self.env._observation()  # noqa: SLF001 - see `post_reset`
        self.policy.reset(context)
        if self.recorder is not None:
            self.recorder.start_episode(context, type(self.policy).__name__, mode)
        terminated = truncated = False
        info: dict[str, Any] = {}
        # Running maximum of every scalar info entry.  The final value is not enough,
        # and the gap is not hypothetical: a scene built with
        # ``terminate_on_success=False`` -- which both cookie batch scenes are, so that
        # the policy keeps executing its second batch -- goes on stepping after the
        # task is done.  A policy that puts ten Cookies in the target box and then
        # wanders off ends the episode with ``success=False`` and
        # ``cookies_in_target=0``, which reads exactly like a policy that never
        # touched one.  The peak separates those, and both values are reported: "did
        # it happen" and "did it survive" are different questions and the report
        # should not silently answer one with the other.
        peaks: dict[str, float] = {}
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
                for key, value in info.items():
                    number = _peakable(value)
                    if number is not None and number > peaks.get(key, -np.inf):
                        peaks[key] = number
                # Optional per-step hook, read the same duck-typed way as
                # ``metrics`` and ``success_override`` below, so a caller that needs
                # the trajectory rather than just its outcome does not have to
                # reimplement the loop to get it.  Comparison is the caller that
                # needs it: whether a model's output could drive the robot at all is
                # a question about how far the environment had to correct each
                # command, and only the environment knows that.
                observe_step = getattr(self.policy, "observe_step", None)
                if observe_step is not None:
                    # `commanded_action`, not the policy's own vector: the trace asks how far
                    # the environment had to move the command before applying it, and that is
                    # a joint-space question.  For a Cartesian policy the two are different
                    # spaces entirely -- fourteen Cartesian deltas in, sixteen joint targets
                    # out -- and the runner used to pass the former, so every behaviour
                    # comparison on a `cartesian_delta` scene died on the shape before it
                    # could measure anything.  On a joint-target arm `commanded_action` *is*
                    # the policy's vector, so those runs are unchanged.
                    observe_step(
                        requested=info["commanded_action"],
                        applied=info["applied_action"],
                        info=info,
                    )
                recording_enabled = bool(getattr(self.policy, "recording", True))
                if self.recorder is not None and recording_enabled:
                    recorded = (
                        info["requested_action"]
                        if self.recorded_action == "requested"
                        else info["applied_action"]
                    )
                    self.recorder.add_frame(observation, recorded)
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
            peak_info=peaks,
        )

    def close(self) -> None:
        self.policy.close()
        if self.recorder is not None:
            self.recorder.close()
        self.env.close()
