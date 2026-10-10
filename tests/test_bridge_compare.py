"""Tests for the comparison harness.

The harness exists to answer a question the task metric cannot: under a declared
zero-shot regime every model is expected to fail the task, so a table of zeros
would say nothing.  What it reports instead is whether a model's output could drive
the robot at all -- how far the environment had to correct each command, whether
joints sat on their limits, how hard the arm was commanded to move.

Those numbers are only meaningful if they are computed the same way for every
model, so the tests here are mostly about the arithmetic and about the report
refusing to blur distinctions that matter (regime, prompt kind, dropped cameras).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from a3_dual_arm_sim.bridge.compare import (
    CORRECTION_EPSILON_RAD,
    SATURATION_TOLERANCE_RAD,
    BehaviorTrace,
    ComparisonReport,
    ManifestResult,
    PolicyTrace,
    _cell,
    _self_report,
    format_table,
    regime_groups,
)
from a3_dual_arm_sim.bridge.manifest import (
    EVALUATION_REGIMES,
    PRIMARY_METRICS,
    PROMPT_KINDS,
    load_manifest,
)
from a3_dual_arm_sim.cli import _parse_seeds
from a3_dual_arm_sim.contracts import JOINT_ACTION_DIM

POLICY_DIR = Path(__file__).resolve().parents[1] / "configs" / "policies"


def limits() -> np.ndarray:
    """Fourteen arm-joint limits, wide enough that nothing saturates by default."""

    return np.tile(np.array([-3.0, 3.0]), (14, 1))


def action(**overrides: float) -> np.ndarray:
    """A sixteen-dimensional action, with named joints overridable."""

    value = np.zeros(JOINT_ACTION_DIM, dtype=np.float64)
    for name, number in overrides.items():
        value[int(name.lstrip("j"))] = number
    return value


# ---------------------------------------------------------------------------
# behaviour arithmetic
# ---------------------------------------------------------------------------


def test_an_uncorrected_step_is_not_counted_as_corrected() -> None:
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j0=0.5), applied=action(j0=0.5))

    summary = trace.summary()

    assert summary["behaviour_steps"] == 1
    assert summary["corrected_fraction"] == 0.0
    assert summary["correction_max_rad"] == 0.0


def test_a_clipped_command_is_counted_and_measured() -> None:
    # The environment clips and rate-limits before applying, so the gap between what
    # was asked for and what was applied is the environment overruling the model.
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j0=1.0), applied=action(j0=0.4))

    summary = trace.summary()

    assert summary["corrected_fraction"] == 1.0
    assert summary["correction_max_rad"] == pytest.approx(0.6)


def test_floating_point_noise_is_not_a_correction() -> None:
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(
        requested=action(j0=0.5),
        applied=action(j0=0.5 + CORRECTION_EPSILON_RAD / 10.0),
    )

    assert trace.summary()["corrected_fraction"] == 0.0


def test_grippers_are_excluded_from_the_radian_metrics() -> None:
    # Grippers are commanded on a unit scale; folding them into a radian-valued
    # metric would make the number meaningless.
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j7=0.0), applied=action(j7=1.0))

    summary = trace.summary()

    assert summary["corrected_fraction"] == 0.0
    assert summary["correction_max_rad"] == 0.0


def test_a_joint_at_its_limit_counts_as_saturated() -> None:
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j0=3.0), applied=action(j0=3.0))

    assert trace.summary()["saturated_fraction"] == 1.0


def test_a_joint_just_inside_its_limit_still_counts_as_saturated() -> None:
    # The environment clips before applying, so a joint "at" its limit is usually a
    # hair inside it; the tolerance is what makes the metric catch that.
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(
        requested=action(j0=3.0),
        applied=action(j0=3.0 - SATURATION_TOLERANCE_RAD / 2.0),
    )

    assert trace.summary()["saturated_fraction"] == 1.0


def test_motion_is_measured_on_what_was_applied() -> None:
    # The question is whether the arm moved smoothly, and only the applied command
    # reaches the actuators.  The first step has no predecessor, so it contributes
    # no motion sample -- which is why the mean here is one value, not two.
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j0=0.0), applied=action(j0=0.0))
    trace.record(requested=action(j0=9.0), applied=action(j0=0.25))

    summary = trace.summary()

    assert summary["motion_max_rad"] == pytest.approx(0.25)
    assert summary["motion_mean_rad"] == pytest.approx(0.25)


def test_motion_averages_over_the_steps_that_have_a_predecessor() -> None:
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j0=0.0), applied=action(j0=0.0))
    trace.record(requested=action(j0=0.0), applied=action(j0=0.10))
    trace.record(requested=action(j0=0.0), applied=action(j0=0.30))

    summary = trace.summary()

    # Two samples: 0.10 and 0.20.
    assert summary["motion_mean_rad"] == pytest.approx(0.15)
    assert summary["motion_max_rad"] == pytest.approx(0.20)


def test_an_empty_trace_summarises_to_nothing() -> None:
    # A model that failed to load has no steps, and reporting zeros for it would
    # read as "perfectly behaved" rather than "never ran".
    assert BehaviorTrace().summary() == {}


def test_a_wrong_shaped_action_is_refused() -> None:
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    with pytest.raises(Exception, match="sixteen-dimensional"):
        trace.record(requested=np.zeros(14), applied=np.zeros(JOINT_ACTION_DIM))


def test_saturation_is_not_reported_without_limits() -> None:
    # Limits come from the model, not from a step, so a trace that was never bound
    # must not invent them.
    trace = BehaviorTrace()
    trace.record(requested=action(j0=3.0), applied=action(j0=3.0))

    assert trace.summary()["saturated_fraction"] == 0.0


def test_a_step_at_the_rate_cap_is_counted_as_rate_limited() -> None:
    # The case this metric exists for: a model whose output scale is wrong for the
    # robot asks for more travel than the per-step cap allows, every step.  Without
    # this number, ``motion_max_rad`` equal to the cap looks like a coincidence.
    trace = BehaviorTrace()
    trace.bind_limits(limits(), rate_limit=0.05)
    trace.record(requested=action(j0=0.0), applied=action(j0=0.0))
    trace.record(requested=action(j0=9.0), applied=action(j0=0.05))

    summary = trace.summary()

    assert summary["rate_limited_fraction"] == 1.0
    assert summary["motion_max_rad"] == pytest.approx(0.05)


def test_a_step_below_the_rate_cap_is_not_counted() -> None:
    trace = BehaviorTrace()
    trace.bind_limits(limits(), rate_limit=0.05)
    trace.record(requested=action(j0=0.0), applied=action(j0=0.0))
    trace.record(requested=action(j0=0.01), applied=action(j0=0.01))

    assert trace.summary()["rate_limited_fraction"] == 0.0


def test_rate_limiting_is_not_reported_without_a_cap() -> None:
    # Same reasoning as saturation: the cap is a property of the environment, and a
    # trace that was not told it must not guess.
    trace = BehaviorTrace()
    trace.bind_limits(limits())
    trace.record(requested=action(j0=0.0), applied=action(j0=0.0))
    trace.record(requested=action(j0=9.0), applied=action(j0=0.05))

    assert trace.summary()["rate_limited_fraction"] == 0.0


def test_rate_limiting_is_measured_over_steps_that_have_a_predecessor() -> None:
    # The first step has no motion sample, so it cannot be rate limited either; the
    # denominator has to match the numerator's population.
    trace = BehaviorTrace()
    trace.bind_limits(limits(), rate_limit=0.05)
    trace.record(requested=action(j0=0.0), applied=action(j0=0.0))
    trace.record(requested=action(j0=9.0), applied=action(j0=0.05))
    trace.record(requested=action(j0=0.0), applied=action(j0=0.06))

    # Two motion samples: 0.05 (at the cap) and 0.01 (below it).
    assert trace.summary()["rate_limited_fraction"] == 0.5


# ---------------------------------------------------------------------------
# the wrapper
# ---------------------------------------------------------------------------


class FakePolicy:
    """A policy that does nothing, to check what the wrapper forwards."""

    action_mode = "joint_position"

    def __init__(self) -> None:
        self.resets = 0
        self.closed = False

    def reset(self, context: object) -> None:
        self.resets += 1

    def act(self, observation: object, task: str) -> np.ndarray:
        return np.zeros(JOINT_ACTION_DIM)

    def close(self) -> None:
        self.closed = True

    @property
    def metrics(self) -> dict[str, object]:
        return {"from": "wrapped"}

    @property
    def success_override(self) -> bool:
        return True


def test_the_wrapper_forwards_the_duck_typed_attributes() -> None:
    # The runner reads metrics, success_override and stop_requested off the policy,
    # so the wrapper must not swallow them.
    wrapped = PolicyTrace(FakePolicy())

    assert wrapped.metrics == {"from": "wrapped"}
    assert wrapped.success_override is True
    assert wrapped.action_mode == "joint_position"


def test_the_wrapper_forwards_the_protocol() -> None:
    inner = FakePolicy()
    wrapped = PolicyTrace(inner)

    wrapped.reset(object())
    assert inner.resets == 1
    assert wrapped.act({}, "task").shape == (JOINT_ACTION_DIM,)
    wrapped.close()
    assert inner.closed


def test_resetting_the_wrapper_clears_the_trace() -> None:
    # Otherwise a second episode would inherit the first one's behaviour numbers.
    wrapped = PolicyTrace(FakePolicy())
    wrapped.bind_limits(limits())
    wrapped.observe_step(requested=action(j0=1.0), applied=action(j0=0.0), info={})
    assert wrapped.trace.summary()["behaviour_steps"] == 1

    wrapped.reset(object())

    assert wrapped.trace.summary() == {}


def test_resetting_keeps_the_limits_and_the_rate_cap() -> None:
    """The robot's constants outlive an episode; only the counters are per-episode.

    Regression test for a bug that read as a result rather than as a fault: the
    wrapper replaced its whole trace on reset, so from the second episode onward the
    limits and the rate cap were gone.  Saturation and rate limiting then reported
    zero for every model -- which looks exactly like a model that never touched a
    limit, and is the kind of number a comparison would happily print.
    """

    wrapped = PolicyTrace(FakePolicy())
    wrapped.bind_limits(limits(), rate_limit=0.05)
    wrapped.reset(object())

    # Two steps: the first sits at zero, the second on the upper limit and far
    # beyond the rate cap.  Both facts must still be visible after the reset.
    wrapped.observe_step(requested=action(j0=0.0), applied=action(j0=0.0), info={})
    wrapped.observe_step(requested=action(j0=9.0), applied=action(j0=3.0), info={})

    summary = wrapped.trace.summary()

    # One of two steps is on a limit; the single motion sample is at the cap.
    assert summary["saturated_fraction"] == 0.5
    assert summary["rate_limited_fraction"] == 1.0


def test_a_second_episode_is_measured_with_the_same_constants() -> None:
    wrapped = PolicyTrace(FakePolicy())
    wrapped.bind_limits(limits(), rate_limit=0.05)
    for _ in range(2):
        wrapped.reset(object())
        wrapped.observe_step(requested=action(j0=0.0), applied=action(j0=0.0), info={})
        wrapped.observe_step(requested=action(j0=9.0), applied=action(j0=3.0), info={})
        summary = wrapped.trace.summary()
        assert summary["behaviour_steps"] == 2, "counters must restart per episode"
        assert summary["saturated_fraction"] == 0.5, "limits must survive the reset"
        assert summary["rate_limited_fraction"] == 1.0, "the cap must survive too"


def test_a_torque_arm_is_reported_as_unavailable_rather_than_in_radians() -> None:
    """The five radian figures are not numbers on an arm that is commanded in torques.

    Each one is a subtraction between entries of the commanded and applied action compared
    against a bound in radians, and on the Panda scenes those entries are the torques the
    operational-space controller solved for.  Reporting them anyway is how the two-Cookie
    table came to show 7.7 rad of mean per-step "motion" against a 0.1 rad cap, and to read
    ``corrected`` -- "the clearest single sign that a model is not driving this robot" -- as
    exactly zero, which is what a model driving the arm perfectly well reads as too.
    """

    trace = BehaviorTrace()
    trace.bind_limits(limits(), rate_limit=0.05, joint_target=False)
    trace.record(requested=action(j0=0.5), applied=action(j0=0.5))
    # A four-fold overrule between the two vectors.  On a joint arm this is a correction of
    # 2.5 rad and must be counted; here it is two torques being subtracted.
    trace.record(requested=action(j0=0.5), applied=action(j0=3.0))

    summary = trace.summary()

    assert summary["behaviour_steps"] == 2, "the steps were observed and are still counted"
    assert summary["behaviour_joint_space"] is False
    assert "operational-space" in summary["behaviour_note"]
    for key in (
        "corrected_fraction",
        "saturated_fraction",
        "rate_limited_fraction",
        "correction_mean_rad",
        "correction_max_rad",
        "motion_mean_rad",
        "motion_max_rad",
    ):
        assert summary[key] is None, f"{key} means nothing here and must not be a number"
        assert _cell(summary[key]) == "—", "the table must say so, not print 0.000"


def test_the_trace_is_told_which_kind_of_arm_it_is_measuring() -> None:
    """`PolicyTrace` forwards the flag, because the loop that drives it holds only a policy."""

    wrapped = PolicyTrace(FakePolicy())
    wrapped.bind_limits(limits(), rate_limit=0.05, joint_target=False)
    wrapped.observe_step(requested=action(j0=0.5), applied=action(j0=3.0), info={})

    summary = wrapped.trace.summary()

    assert summary["behaviour_steps"] == 1
    assert summary["corrected_fraction"] is None, "a torque difference is not a correction"


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def result(name: str, regime: str, **aggregate: object) -> ManifestResult:
    item = ManifestResult(
        name=name,
        path=f"/tmp/{name}.yaml",
        regime=regime,
        primary_metric="behavior",
        prompt_kind="free_text",
        prompt_labels=(),
        notes="",
        dropped_cameras=(),
        action_space="joint_position",
        action_dim=16,
    )
    item.aggregate = lambda: dict(aggregate)  # type: ignore[method-assign]
    return item


def test_regimes_are_grouped_and_never_mixed() -> None:
    # A fine-tuned model winning says it was taught the task; a zero-shot model
    # losing says it was not.  Neither statement is about the architecture, so the
    # two must not share a table.
    report = ComparisonReport(scene="sandbox", seeds=(0,), steps=None, task=None)
    report.results = [
        result("a", "zero_shot"),
        result("b", "finetuned"),
        result("c", "zero_shot"),
    ]

    groups = regime_groups(report)

    assert sorted(groups) == ["finetuned", "zero_shot"]
    assert [item.name for item in groups["zero_shot"]] == ["a", "c"]


def test_the_table_leads_with_behaviour_for_a_zero_shot_regime() -> None:
    report = ComparisonReport(scene="sandbox", seeds=(0,), steps=32, task="hold")
    report.results = [
        result(
            "model",
            "zero_shot",
            episodes=2,
            task_successes=0,
            safety_stops=0,
            behavior={"corrected_fraction": 0.5, "saturated_fraction": 0.25},
            cost={"request_mean_ms": 400.0, "startup_s": 18.0},
        )
    ]

    table = format_table(report)

    assert "Regime: `zero_shot`" in table
    assert "not** the informative" in table
    assert "| model |" in table
    assert "0.500" in table
    assert "0.250" in table


def test_the_table_reports_what_differed_rather_than_normalising_it() -> None:
    # Camera count and prompt kind are wiring facts.  A reader has to be able to
    # tell them from capability, so they travel into the report.
    report = ComparisonReport(scene="sandbox", seeds=(0,), steps=None, task=None)
    item = result("two_camera_model", "zero_shot", episodes=1)
    item.dropped_cameras = ("observation.images.right_wrist",)
    item.prompt_kind = "fixed_label"
    report.results = [item]

    table = format_table(report)

    assert "observation.images.right_wrist" in table
    assert "fixed_label" in table


def test_a_failed_model_is_named_with_its_error() -> None:
    report = ComparisonReport(scene="sandbox", seeds=(0,), steps=None, task=None)
    item = result("broken", "zero_shot")
    item.failure = "BridgeError: the model closed the connection"
    report.results = [item]

    table = format_table(report)

    assert "Failures:" in table
    assert "the model closed the connection" in table


def test_a_missing_number_renders_as_a_dash_not_a_zero() -> None:
    # A model that never reported a cost must not look like one that reported zero.
    report = ComparisonReport(scene="sandbox", seeds=(0,), steps=None, task=None)
    report.results = [result("silent", "zero_shot", episodes=1)]

    table = format_table(report)

    assert "—" in table


def test_the_report_serialises_to_json() -> None:
    report = ComparisonReport(scene="sandbox", seeds=(0, 1), steps=32, task="hold")
    report.results = [result("model", "zero_shot", episodes=2, task_successes=0)]

    payload = report.to_dict()

    assert payload["scene"] == "sandbox"
    assert payload["seeds"] == [0, 1]
    assert payload["results"][0]["name"] == "model"
    assert payload["results"][0]["regime"] == "zero_shot"


# ---------------------------------------------------------------------------
# model self-report
# ---------------------------------------------------------------------------


def test_a_numeric_self_estimate_is_carried_into_the_report() -> None:
    # Cosmos Policy publishes a value estimate; it is the only signal that says
    # anything about how the model judged its own rollout.
    assert _self_report({"cosmos_value": 0.8628678321838379}) == {"cosmos_value": 0.86287}


def test_a_call_count_is_carried_into_the_report() -> None:
    # Regression test: the suffix match used to require a string, so a count -- which
    # is always a number -- was silently dropped and the report showed no evidence
    # that the model had been called at all.
    assert _self_report({"gr00t_calls": 2}) == {"gr00t_calls": 2.0}


def test_the_prompt_label_actually_used_is_carried_into_the_report() -> None:
    # The most important self-report field: a substituted prompt changes what the
    # model was asked to do, so a comparison that hid it would be comparing
    # different questions.
    assert _self_report({"cosmos_task": "fold shirt"}) == {"cosmos_task": "fold shirt"}


def test_unrelated_keys_are_not_swept_into_the_report() -> None:
    # An episode's info also carries the action vectors and the bridge's own
    # bookkeeping; copying those would bury the numbers that matter.
    assert _self_report({"applied_action": [0.0], "infer_ms": 142.0, "bridge": {}}) == {}


def test_a_non_numeric_value_under_a_numeric_suffix_is_ignored() -> None:
    # A model that reports its estimate as a string is not reporting a number, and
    # coercing it would invent one.
    assert _self_report({"cosmos_value": "high"}) == {}


# ---------------------------------------------------------------------------
# seed parsing
# ---------------------------------------------------------------------------


def test_seeds_parse_as_a_list() -> None:
    assert _parse_seeds("0,1,2") == [0, 1, 2]


def test_seeds_parse_as_a_range() -> None:
    assert _parse_seeds("0-3") == [0, 1, 2, 3]


def test_seeds_parse_as_a_mixture() -> None:
    assert _parse_seeds("0,5-6,9") == [0, 5, 6, 9]


def test_blank_seed_entries_are_ignored() -> None:
    assert _parse_seeds("0,,1") == [0, 1]


# ---------------------------------------------------------------------------
# the shipped manifests declare a regime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    sorted(POLICY_DIR.glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_every_shipped_manifest_declares_its_evaluation_regime(path: Path) -> None:
    """A comparison cannot be read without knowing what the weights were trained on.

    The default is ``zero_shot``, which is the honest default for this list -- none
    of these models was trained on A3 -- but a manifest that means something else
    has to say so, and this test is what makes the declaration non-optional.

    The assertions compare against the parser's own vocabularies rather than spelling
    them out.  They used to be written out, and they drifted: the parser grew
    ``primary_metric: both`` -- the value a fine-tuned model wants, because behaviour is
    what makes it comparable to its zero-shot counterpart while task success is the
    claim the fine-tune is making -- and this test kept rejecting it.  A duplicate list
    can only ever be as current as the person who last edited the other one.
    """

    manifest = load_manifest(path)

    assert manifest.evaluation.regime in EVALUATION_REGIMES
    assert manifest.evaluation.primary_metric in PRIMARY_METRICS
    assert manifest.evaluation.prompt_kind in PROMPT_KINDS


def test_the_working_integrations_declare_zero_shot() -> None:
    # Both were trained on other robots and other tasks.  Declaring otherwise would
    # put their numbers in a group where they do not belong.
    for name in ("cosmos_policy_aloha.yaml", "gr00t_bimanual.yaml"):
        manifest = load_manifest(POLICY_DIR / name)
        assert manifest.evaluation.regime == "zero_shot", name
        assert manifest.evaluation.primary_metric == "behavior", name


def test_the_prompt_kinds_record_the_real_difference_between_the_two() -> None:
    """Cosmos Policy looks its prompt up in a table; GR00T encodes free text.

    That contrast is the kind of difference a comparison must not average away, so
    it is asserted rather than left to a comment.
    """

    cosmos = load_manifest(POLICY_DIR / "cosmos_policy_aloha.yaml")
    gr00t = load_manifest(POLICY_DIR / "gr00t_bimanual.yaml")

    assert cosmos.evaluation.prompt_kind == "fixed_label"
    assert gr00t.evaluation.prompt_kind == "free_text"
