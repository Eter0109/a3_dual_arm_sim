"""Compare external models on one scene, under one declared ruler.

The sizing guide's list has sixteen VLA and world-model projects, and "which is
best" cannot be answered by running them one at a time and eyeballing the results:
they differ in action space, in camera count, in prompt handling and in cost, and
each of those differences can look like a capability difference.

Three things make this module more than a loop over ``a3-sim run``:

**Behaviour is measured, not assumed.**  Under a declared ``zero_shot`` regime the
task metric is expected to be near zero -- none of these models was trained on A3 --
and a table of zeros cannot separate "not taught" from "cannot learn".  So each step
is also scored on whether the output could drive the robot at all: how far the
environment had to correct each command before applying it, whether joints sat on
their limits, whether the motion was smooth, and whether the episode was stopped for
safety.  That is a real result even when every episode fails the task.

**A regime is a property of the weights.**  ``finetuned`` and ``zero_shot`` results
are reported as separate groups and never averaged together, because the two answer
different questions.

**What differed is reported, not normalised away.**  How many cameras each model
saw, which prompt kind it accepts and which label was used all travel into the
report, so a reader can tell a wiring difference from a capability difference.

See ``docs/external-policy-bridge.md`` sections 7.2 to 7.4 for the measurements
these choices rest on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import ARM_JOINTS, JOINT_ACTION_DIM
from .manifest import PolicyManifest
from .protocol import BridgeError

#: A joint is called saturated when its applied command is this close to a limit,
#: in radians.  Non-zero because the environment clips and rate-limits before
#: applying, so a joint "at" its limit is usually a hair inside it.
SATURATION_TOLERANCE_RAD = 1e-3

#: Below this, an action is treated as unchanged.  The correction metric counts how
#: often the environment had to alter a command, and floating-point noise is not an
#: alteration.
CORRECTION_EPSILON_RAD = 1e-6


def _arm_indices() -> np.ndarray:
    """Indices of the fourteen arm joints in the sixteen-dimensional vector.

    Grippers sit at 7 and 15 and are excluded: they are commanded on a unit scale,
    so mixing them into a radian-valued metric would make the number meaningless.
    """

    return np.asarray([index for index in range(JOINT_ACTION_DIM) if index not in (7, 15)], dtype=np.int64)


@dataclass
class BehaviorTrace:
    """Per-step behaviour of one episode, accumulated through ``observe_step``."""

    steps: int = 0
    corrected_steps: int = 0
    saturated_steps: int = 0
    rate_limited_steps: int = 0
    correction_rad: list[float] = field(default_factory=list)
    motion_rad: list[float] = field(default_factory=list)
    _previous: np.ndarray | None = None
    #: Joint limits, supplied by the caller because they come from the model rather
    #: than from any single step.
    _limits: np.ndarray | None = None
    #: The environment's per-step joint cap.  Supplied for the same reason, and
    #: needed because a motion figure that equals it is otherwise unreadable: every
    #: model would appear to move at exactly the same speed.
    _rate_limit: float | None = None
    #: Whether the arm entries of the recorded vectors are joint targets at all.
    #: False under an operational-space adapter, where they are torques; see
    #: `Env.applied_action_is_joint_target` and `summary`.
    _joint_target: bool = True

    def bind_limits(
        self,
        limits: np.ndarray,
        rate_limit: float | None = None,
        *,
        joint_target: bool = True,
    ) -> None:
        """Attach the fourteen arm-joint ``(low, high)`` pairs and the step cap.

        ``joint_target=False`` says the vectors this trace is about to be fed are not
        joint targets -- which is the case on any scene whose arm is driven by
        `OperationalSpaceAdapter`, and is not something the trace can see for itself.
        """

        self._limits = np.asarray(limits, dtype=np.float64)
        self._rate_limit = None if rate_limit is None else float(rate_limit)
        self._joint_target = bool(joint_target)

    def record(self, requested: Any, applied: Any) -> None:
        """Fold one step in.  ``requested`` is what the policy asked for."""

        wanted = np.asarray(requested, dtype=np.float64).reshape(-1)
        got = np.asarray(applied, dtype=np.float64).reshape(-1)
        if wanted.shape != (JOINT_ACTION_DIM,) or got.shape != (JOINT_ACTION_DIM,):
            raise BridgeError(
                "behaviour tracing needs the environment's sixteen-dimensional action "
                f"vector, got requested={wanted.shape} applied={got.shape}"
            )
        self.steps += 1
        if not self._joint_target:
            # Every number below is a subtraction between two entries of these vectors
            # compared against a bound in radians, and on a torque-driven arm that is
            # a torque compared against a radian cap.  It is not a small error: on the
            # two-Cookie scene -- LIBERO's Panda under `OperationalSpaceAdapter` -- the
            # table reported 7.7 rad of mean "motion" per step against a 0.1 rad cap,
            # and `corrected` read exactly zero for every step because
            # `_limit_joint_action` deliberately neither clips nor rate-limits torques.
            # A reader told that `corrected` is "the clearest single sign that a model
            # is not driving this robot" would have read both as facts about the model.
            # Counted, and reported as unavailable; see `summary`.
            return
        arms = _arm_indices()
        correction = float(np.max(np.abs(got[arms] - wanted[arms])))
        self.correction_rad.append(correction)
        self.corrected_steps += int(correction > CORRECTION_EPSILON_RAD)
        if self._limits is not None:
            touched = np.any(
                np.abs(got[arms] - self._limits[:, 0]) <= SATURATION_TOLERANCE_RAD
            ) or np.any(np.abs(got[arms] - self._limits[:, 1]) <= SATURATION_TOLERANCE_RAD)
            self.saturated_steps += int(bool(touched))
        # Motion is measured on what was applied, not on what was asked for: the
        # question is whether the arm moved smoothly, and only the applied command
        # reaches the actuators.
        if self._previous is not None:
            step = float(np.max(np.abs(got[arms] - self._previous[arms])))
            self.motion_rad.append(step)
            if self._rate_limit is not None and step >= self._rate_limit - SATURATION_TOLERANCE_RAD:
                # The command was cut to the cap, so the model asked for more travel
                # than the robot permits.  Counted separately from ``corrected``
                # because it is the specific, actionable case: the model's output
                # scale is wrong for this robot, not merely clipped at a limit.
                self.rate_limited_steps += 1
        self._previous = got.copy()

    def reset(self) -> None:
        """Clear the per-episode counters, keeping the environment's constants.

        The limits and the rate cap are properties of the robot, not of an episode,
        so they survive a reset.  Replacing the whole trace instead would silently
        disable saturation and rate-limit detection from the second episode onward --
        and, because both would then read zero, it would look like a model that never
        touched a limit rather than like a broken measurement.
        """

        self.steps = 0
        self.corrected_steps = 0
        self.saturated_steps = 0
        self.rate_limited_steps = 0
        self.correction_rad = []
        self.motion_rad = []
        self._previous = None

    def summary(self) -> dict[str, Any]:
        """Behaviour numbers for one episode.  Empty when nothing was observed."""

        if self.steps == 0:
            return {}
        if not self._joint_target:
            # Keys present and null rather than absent: a consumer that looks them up
            # gets "unknown" instead of a default, and the table renders the em dash.
            # The reason travels with them, because a null column is otherwise read as
            # a missing measurement rather than as a measurement that means nothing here.
            return {
                "behaviour_steps": self.steps,
                "behaviour_joint_space": False,
                "behaviour_note": (
                    "the arm is driven by operational-space control, so the applied "
                    "action is joint torque: correction, saturation and motion are not "
                    "radians and are not reported"
                ),
                "corrected_fraction": None,
                "saturated_fraction": None,
                "rate_limited_fraction": None,
                "correction_mean_rad": None,
                "correction_max_rad": None,
                "motion_mean_rad": None,
                "motion_max_rad": None,
            }
        correction = np.asarray(self.correction_rad, dtype=np.float64)
        motion = np.asarray(self.motion_rad, dtype=np.float64) if self.motion_rad else np.zeros(1)
        samples = max(1, len(self.motion_rad))
        return {
            "behaviour_steps": self.steps,
            # How much the environment had to overrule the model.  A model whose
            # commands are accepted as issued is driving the robot; one whose every
            # command is rate-limited or clipped is not.
            "corrected_fraction": round(self.corrected_steps / self.steps, 4),
            "saturated_fraction": round(self.saturated_steps / self.steps, 4),
            # The subset of corrections caused by the per-step cap rather than by a
            # joint limit.  Reported because ``motion_max_rad`` equal to the cap is
            # otherwise indistinguishable from a coincidence.
            "rate_limited_fraction": round(self.rate_limited_steps / samples, 4),
            "correction_mean_rad": round(float(correction.mean()), 5),
            "correction_max_rad": round(float(correction.max()), 5),
            # Smoothness.  A high figure next to a low correction means the arm was
            # commanded to move hard rather than being overruled.
            "motion_mean_rad": round(float(motion.mean()), 5),
            "motion_max_rad": round(float(motion.max()), 5),
        }


class PolicyTrace:
    """Wrap a policy to record per-step behaviour, staying a policy itself.

    A wrapper rather than a runner option because the runner already reads its
    optional collaborators off the policy (``metrics``, ``success_override``,
    ``stop_requested``), so this needs no change to the loop and composes with
    anything the runner accepts.
    """

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self.action_mode = policy.action_mode
        self.trace = BehaviorTrace()

    def bind_limits(
        self,
        limits: np.ndarray,
        rate_limit: float | None = None,
        *,
        joint_target: bool = True,
    ) -> None:
        self.trace.bind_limits(limits, rate_limit, joint_target=joint_target)

    def observe_step(self, *, requested: Any, applied: Any, info: Mapping[str, Any]) -> None:
        self.trace.record(requested, applied)

    # -- forwarded policy protocol ------------------------------------------

    def reset(self, context: Any) -> None:
        self.trace.reset()
        self._policy.reset(context)

    def act(self, observation: Mapping[str, Any], task: str) -> np.ndarray:
        return self._policy.act(observation, task)

    def close(self) -> None:
        self._policy.close()

    def __getattr__(self, name: str) -> Any:
        # Everything the runner reads duck-typed -- metrics, success_override,
        # stop_requested, recording -- comes from the wrapped policy, so this
        # wrapper does not have to know the list.
        return getattr(self._policy, name)


@dataclass
class EpisodeMetrics:
    """One episode, in the four families the comparison reports."""

    seed: int
    steps: int
    terminated: bool
    truncated: bool
    behavior: Mapping[str, Any] = field(default_factory=dict)
    task: Mapping[str, Any] = field(default_factory=dict)
    self_report: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    safety_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "safety_reason": self.safety_reason,
            "behavior": dict(self.behavior),
            "task": dict(self.task),
            "self_report": dict(self.self_report),
            "cost": dict(self.cost),
        }


#: Task-side keys a scene may publish in its own info.  Listed rather than taken
#: wholesale because an episode's info also carries the action vectors, and copying
#: every array into a comparison report would bury the numbers that matter.
_TASK_KEYS = (
    "success",
    "cookies_in_target",
    "cookies_in_source",
    "required_cookies",
    "lifted",
    "released",
    "phase",
)


#: The task key whose peak reads as "how far into the task did this get".  One name
#: rather than a list because the column has room for one number and the counting
#: metric is the only one that means progress; a scene that publishes none of these
#: gets an em dash rather than a fabricated zero.
_PROGRESS_KEY = "cookies_in_target"


#: Model self-report keys, matched by suffix so a new integration that names its
#: estimate ``something_value`` is picked up without editing this list.
_SELF_REPORT_SUFFIXES = ("_value", "_calls", "_task")

#: Suffixes whose value is a number rather than a label.  ``_calls`` is a count and
#: ``_value`` is an estimate; ``_task`` is the label the model was actually driven
#: with, which is a string and the one that matters most -- a substituted prompt
#: changes what the model was asked to do.
_NUMERIC_SUFFIXES = ("_value", "_calls")


def _self_report(info: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in info.items():
        if key.endswith("_task") and isinstance(value, str):
            out[key] = value
        elif key.endswith(_NUMERIC_SUFFIXES) and isinstance(value, (int, float)):
            out[key] = round(float(value), 5)
    return out


def episode_metrics(result: Any, trace: BehaviorTrace) -> EpisodeMetrics:
    """Turn a runner result plus a trace into the report's four families."""

    info = result.final_info
    peaks: Mapping[str, Any] = getattr(result, "peak_info", None) or {}
    bridge = info.get("bridge") or {}
    # Task numbers travel twice, under the plain key and under a ``_peak`` suffix:
    # where the episode ended, and the best it ever reached.  Both are kept because
    # they disagree in the case this report exists to catch -- an episode that
    # finished the task and kept going -- and a reader who only wants the old number
    # is not moved by the addition.  Flat rather than nested because both
    # ``aggregate`` and the table treat this family as a mapping of numbers, and
    # nesting would make one of them special-case the other.
    task = {key: _plain(info[key]) for key in _TASK_KEYS if key in info}
    task.update({f"{key}_peak": _plain(peaks[key]) for key in _TASK_KEYS if key in peaks})
    return EpisodeMetrics(
        seed=int(info.get("seed", -1)),
        steps=int(result.steps),
        terminated=bool(result.terminated),
        truncated=bool(result.truncated),
        behavior=trace.summary(),
        task=task,
        self_report=_self_report(info),
        cost={
            "startup_s": bridge.get("startup_s"),
            "request_mean_ms": bridge.get("request_mean_ms"),
            "request_max_ms": bridge.get("request_max_ms"),
            "requests": bridge.get("requests"),
            "chunk_size": info.get("chunk_size"),
            "execute_steps": info.get("execute_steps"),
            "dropped_cameras": len(bridge.get("dropped_cameras") or ()),
        },
        safety_reason=result.safety_reason,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ManifestResult:
    """Every episode of one manifest, plus what it was wired up to see."""

    name: str
    path: str
    regime: str
    primary_metric: str
    prompt_kind: str
    prompt_labels: tuple[str, ...]
    notes: str
    dropped_cameras: tuple[str, ...]
    action_space: str
    action_dim: int
    episodes: list[EpisodeMetrics] = field(default_factory=list)
    failure: str | None = None

    def aggregate(self) -> dict[str, Any]:
        """Mean over episodes for each numeric family."""

        if not self.episodes:
            return {}
        payload: dict[str, Any] = {}
        for family in ("behavior", "task", "cost"):
            keys = sorted({key for ep in self.episodes for key in getattr(ep, family)})
            values: dict[str, Any] = {}
            for key in keys:
                numbers = [
                    getattr(ep, family)[key]
                    for ep in self.episodes
                    if isinstance(getattr(ep, family).get(key), (int, float))
                ]
                if numbers:
                    values[key] = round(float(np.mean(numbers)), 5)
            if values:
                payload[family] = values
        payload["episodes"] = len(self.episodes)
        payload["task_successes"] = sum(
            1 for ep in self.episodes if bool(ep.task.get("success"))
        )
        # Counted as well as, not instead of, the final-step count.  The two differ
        # exactly where it matters: ``success`` is read from the episode's last step,
        # so a policy that finishes the task and then keeps moving -- which any scene
        # with ``terminate_on_success=False`` permits -- scores zero in the first
        # column and one here.  A run where they disagree is a run whose headline
        # number is about the episode cap rather than about the policy.
        payload["task_successes_peak"] = sum(
            1 for ep in self.episodes if ep.task.get("success_peak")
        )
        payload["safety_stops"] = sum(1 for ep in self.episodes if ep.terminated)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "regime": self.regime,
            "primary_metric": self.primary_metric,
            "prompt_kind": self.prompt_kind,
            "prompt_labels": list(self.prompt_labels),
            "notes": self.notes,
            "dropped_cameras": list(self.dropped_cameras),
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "failure": self.failure,
            "aggregate": self.aggregate(),
            "episodes": [ep.to_dict() for ep in self.episodes],
        }


@dataclass
class ComparisonReport:
    """One scene, one set of seeds, several manifests."""

    scene: str
    seeds: tuple[int, ...]
    steps: int | None
    task: str | None
    results: list[ManifestResult] = field(default_factory=list)
    #: Whether each episode drew its layout from the seed, or every episode ran the
    #: scene's canonical one.  Recorded because the two produce tables that look
    #: identical and mean different things: with randomisation off, a multi-seed
    #: table is one configuration repeated and its episode count is not a sample size.
    #: Nothing in the numbers themselves says which of the two happened, so it is
    #: carried here and printed above the table.
    randomized: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "seeds": list(self.seeds),
            "steps": self.steps,
            "task": self.task,
            "randomized": self.randomized,
            "results": [result.to_dict() for result in self.results],
        }


def run_manifest(
    manifest: PolicyManifest,
    *,
    env_factory: Callable[[], Any],
    seeds: Sequence[int],
    steps: int | None,
    task: str | None,
    policy_factory: Callable[[PolicyManifest, Any], Any],
    reset_options: Mapping[str, Any] | None = None,
) -> ManifestResult:
    """Run one manifest over the seeds, building a fresh environment per episode.

    A fresh environment per episode because scenes randomise on reset from the seed,
    and reusing one would carry contact state across episodes -- two models would
    then be compared on different scene realisations.

    ``reset_options`` is passed through to every reset.  The CLI samples each
    episode's layout from its seed by default and pins the scene's canonical one only
    under ``--no-randomize``.  Pinning is for reproducing a rendered video, not for
    measuring a policy: with it on, every seed drew the identical scene and a
    multi-seed table was one configuration repeated while looking like a sample.
    """

    from ..runner import EpisodeRunner

    result = ManifestResult(
        name=manifest.name,
        path=str(manifest.source_path or ""),
        regime=manifest.evaluation.regime,
        primary_metric=manifest.evaluation.primary_metric,
        prompt_kind=manifest.evaluation.prompt_kind,
        prompt_labels=manifest.evaluation.prompt_labels,
        notes=manifest.notes.strip(),
        dropped_cameras=tuple(
            sorted(
                {
                    "observation.images.front",
                    "observation.images.left_wrist",
                    "observation.images.right_wrist",
                }
                - {spec.source for spec in manifest.observation.images.values()}
            )
        ),
        action_space=manifest.action.space,
        action_dim=manifest.action.dim,
    )

    for seed in seeds:
        env = env_factory()
        try:
            gateway = policy_factory(manifest, env)
            trace = PolicyTrace(gateway)
            # The behaviour numbers are subtractions between entries of the commanded
            # and applied action vectors, so they are only radians while those entries
            # are joint targets.  On a torque-driven arm they are not, and the trace is
            # told so rather than left to report them anyway -- see
            # `BehaviorTrace.record` for what that produced before.
            joint_target = bool(getattr(env, "applied_action_is_joint_target", True))
            trace.bind_limits(
                _arm_limits(env),
                getattr(env.config, "max_joint_step_rad", None) if joint_target else None,
                joint_target=joint_target,
            )
            runner = EpisodeRunner(
                env,
                trace,
                task=task or "move the A3 dual-arm robot",
                # Behaviour has to be judged on complete episodes: an episode the
                # environment cut short carries no information about what the model
                # would have done next.
                save_failed_episodes=False,
                reset_options=reset_options,
            )
            try:
                episode = runner.run(seed=seed, max_steps=steps)
            finally:
                runner.close()
            metrics = episode_metrics(episode, trace.trace)
            metrics.seed = seed
            result.episodes.append(metrics)
        except Exception as exc:  # noqa: BLE001 - one model must not sink the report
            result.failure = f"{type(exc).__name__}: {exc}"
            break

    return result


def _arm_limits(env: Any) -> np.ndarray:
    """The fourteen arm-joint limits, in action-vector order."""

    return np.asarray([env.model.jnt_range[env._joint_ids[name]] for name in ARM_JOINTS])


def compare(
    manifests: Sequence[PolicyManifest],
    *,
    env_factory: Callable[[], Any],
    policy_factory: Callable[[PolicyManifest, Any], Any],
    seeds: Sequence[int] = (0,),
    steps: int | None = None,
    task: str | None = None,
    scene: str = "sandbox",
    reset_options: Mapping[str, Any] | None = None,
    randomized: bool = True,
) -> ComparisonReport:
    """Run every manifest on the same scene and seeds."""

    report = ComparisonReport(
        scene=scene, seeds=tuple(seeds), steps=steps, task=task, randomized=randomized
    )
    for manifest in manifests:
        report.results.append(
            run_manifest(
                manifest,
                env_factory=env_factory,
                seeds=seeds,
                steps=steps,
                task=task,
                policy_factory=policy_factory,
                reset_options=reset_options,
            )
        )
    return report


def regime_groups(report: ComparisonReport) -> dict[str, list[ManifestResult]]:
    """Split results by regime, which is how they must be read.

    A single table mixing fine-tuned and zero-shot results answers nothing: a
    fine-tuned model winning says it was taught the task, and a zero-shot model
    losing says it was not, and neither statement is about the architecture.
    """

    groups: dict[str, list[ManifestResult]] = {}
    for result in report.results:
        groups.setdefault(result.regime, []).append(result)
    return groups


def _cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _progress_cell(result: ManifestResult) -> str:
    """``mean last → best`` for the scene's progress counter, or an em dash.

    Two numbers rather than one because they answer different questions and the gap
    between them is the diagnosis.  ``last`` is where the episode ran out of steps;
    ``best`` is how far it ever got.  A model whose best is well above its last is
    being cut off -- or is losing work it had already done -- and neither reading is
    available from a figure that has averaged the two together.
    """

    finals = [
        ep.task[_PROGRESS_KEY]
        for ep in result.episodes
        if isinstance(ep.task.get(_PROGRESS_KEY), (int, float))
    ]
    peaks = [
        ep.task[f"{_PROGRESS_KEY}_peak"]
        for ep in result.episodes
        if isinstance(ep.task.get(f"{_PROGRESS_KEY}_peak"), (int, float))
    ]
    if not peaks:
        return "—"
    last = _cell(sum(finals) / len(finals), 1) if finals else "—"
    # Cast to float before formatting.  ``_cell`` only pads decimal places for floats,
    # and the peak arrives as an int whenever the counter is one -- so the column would
    # read ``0.0 → 10`` and look like two different quantities.
    return f"{last} → {_cell(float(max(peaks)), 1)}"


def format_table(report: ComparisonReport) -> str:
    """A markdown summary, grouped by regime and led by the declared metric.

    Deliberately plain text: the report has to be readable in a terminal, in a pull
    request and in a chat log, and a table that needs a viewer to interpret is one
    nobody checks.
    """

    lines: list[str] = [
        f"# Policy comparison — scene `{report.scene}`, seeds {list(report.seeds)}",
        "",
    ]
    if report.steps is not None:
        lines.append(f"Episode cap: {report.steps} steps")
        lines.append("")
    lines.append(
        "Layout: sampled per episode from the seed (`--randomize`)."
        if report.randomized
        else "Layout: the scene's canonical one, the same for every seed (`--no-randomize`)."
    )
    lines.append("")
    if not report.randomized and len(report.seeds) > 1:
        # Printed rather than assumed, because the table cannot show it: a number that
        # is one episode repeated looks exactly like a number that is four episodes
        # averaged, and the difference decides whether any of it generalises.
        lines.append(
            f"> **All {len(report.seeds)} seeds ran the same scene.** Randomisation is "
            "off, so the seed does not reach the layout, the arm's start pose or the "
            "per-piece jitter; the episodes below are one configuration repeated and "
            "their count is not a sample size. Re-run with `--randomize` before "
            "reading a difference between two models out of this table."
        )
        lines.append("")
    if report.task:
        lines.append(f"Task string sent to every model: `{report.task}`")
        lines.append("")

    for regime, results in sorted(regime_groups(report).items()):
        lines.append(f"## Regime: `{regime}`")
        lines.append("")
        if regime == "zero_shot":
            lines.append(
                "> These weights were trained on other robots and other tasks. Task "
                "success is expected to be near zero and is **not** the informative "
                "column — behaviour is."
            )
            lines.append("")

        header = (
            "| model | prompt | cameras dropped | episodes | success | success ever "
            "| safety stops | corrected | rate limited | saturated "
            "| correction max (rad) | motion mean (rad) | progress last → best "
            "| infer mean (ms) | startup (s) |"
        )
        divider = "|" + "---|" * 15
        lines.extend([header, divider])
        for result in results:
            agg = result.aggregate()
            behavior = agg.get("behavior", {})
            cost = agg.get("cost", {})
            lines.append(
                "| {name} | {prompt} | {dropped} | {episodes} | {success} | {ever} "
                "| {safety} | {corrected} | {limited} | {saturated} | {cmax} | {motion} "
                "| {progress} | {infer} | {startup} |".format(
                    name=result.name,
                    prompt=result.prompt_kind,
                    dropped=", ".join(result.dropped_cameras) or "none",
                    episodes=agg.get("episodes", 0),
                    success=agg.get("task_successes", "—"),
                    ever=agg.get("task_successes_peak", "—"),
                    safety=agg.get("safety_stops", "—"),
                    corrected=_cell(behavior.get("corrected_fraction")),
                    limited=_cell(behavior.get("rate_limited_fraction")),
                    saturated=_cell(behavior.get("saturated_fraction")),
                    cmax=_cell(behavior.get("correction_max_rad")),
                    motion=_cell(behavior.get("motion_mean_rad")),
                    progress=_progress_cell(result),
                    infer=_cell(cost.get("request_mean_ms"), 0),
                    startup=_cell(cost.get("startup_s"), 1),
                )
            )
        lines.append("")
        if any(result.failure for result in results):
            lines.append("Failures:")
            for result in results:
                if result.failure:
                    lines.append(f"- `{result.name}`: {result.failure}")
            lines.append("")

    lines.extend(
        [
            "## Reading this table",
            "",
            (
                "- **corrected** — fraction of steps where the environment had to change "
                "the command before applying it (joint clipping or rate limiting). The "
                "clearest single sign that a model is not driving this robot."
            ),
            (
                "- **success / success ever** — the first is read from each episode's "
                "last step, the second from whether it was ever true. They differ "
                "whenever the scene does not stop on success, which is the normal case "
                "here: a policy that finishes the task and then keeps moving lands in "
                "the second column and not the first. Read together, not either alone."
            ),
            (
                "- **progress last → best** — mean count in the target box at the end of "
                "an episode, then the best any episode ever reached. A best far above "
                "the mean says the task does come together and then comes apart, or "
                "that the step cap is cutting episodes off mid-task."
            ),
            (
                "- **corrected, again** — it is met by a model that never moves: the "
                "environment cannot overrule a command to stand still. A low value "
                "beside a low **motion mean** is inaction, not competence, so the "
                "progress column is the one to lead with."
            ),
            (
                "- **an em dash in the five figures from `corrected` to `motion mean`** "
                "— the scene's arm is driven by operational-space control, so the applied "
                "action is joint **torque** and every one of those figures is a torque "
                "compared against a bound in radians. They are reported as `null` in the "
                "JSON (with `behaviour_note` saying why) and as `—` here, rather than as "
                "the `0.000` a reader would take for a model driving the arm cleanly."
            ),
            (
                "- **rate limited** — the subset of those corrections caused by the "
                "per-step joint cap rather than by a joint limit. A high value means "
                "the model asks for more travel per step than the robot permits, which "
                "is a scale mismatch rather than a bad decision."
            ),
            (
                "- **saturated** — fraction of steps with an arm joint at a limit. High "
                "values mean the model is pushing against the end of travel. Note that "
                "rate limiting can hold this at zero: an arm that may only move a little "
                "per step cannot reach a limit in a short episode."
            ),
            "- **correction max** — the largest single-step overrule, in radians.",
            (
                "- **motion mean** — mean per-step joint travel. High motion with high "
                "correction means the arm was commanded to move hard, not that it did."
            ),
            (
                "- A model with no episodes either failed to load or crashed; see its "
                "``failure`` entry in the JSON."
            ),
        ]
    )
    return "\n".join(lines)
