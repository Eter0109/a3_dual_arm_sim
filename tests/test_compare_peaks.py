"""The peak half of the comparison report.

An episode's task numbers are read from its last step, which is the wrong end of the
episode for every scene that does not stop on success -- and both cookie batch scenes
are built with ``terminate_on_success=False`` so that a policy keeps executing its
second batch.  A policy that fills the target box and then wanders ends the episode
reporting ``success=False`` and ``cookies_in_target=0``, indistinguishable from one
that never moved.  These tests pin the fix: the runner keeps a running maximum, the
report carries both readings, and a table whose seeds all ran the same scene says so.
"""

from __future__ import annotations

import json

import numpy as np

from a3_dual_arm_sim.bridge.compare import (
    ComparisonReport,
    EpisodeMetrics,
    ManifestResult,
    _progress_cell,
    format_table,
)
from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.runner import EpisodeResult, EpisodeRunner, _peakable


class _CounterEnv:
    """Publishes a task counter that rises, peaks, then falls -- and never stops.

    The shape matters: a counter that only ever rose would be captured by the final
    value too, and the test would pass against the behaviour it exists to change.
    """

    action_mode = "joint_position"

    #: What the environment calls the command it actually applied.  The runner asks for this
    #: before it records a single frame (`stored_action_mode`), and a stub that stopped at
    #: ``action_mode`` was enough until it did.  A property rather than a second class attribute
    #: because on a joint-servo arm the two really are one fact and on a torque arm they are not;
    #: here they agree, and that agreement is the only thing a stub of this kind may assume.
    @property
    def canonical_action_mode(self) -> str:
        return "joint_position"

    def __init__(self, trajectory: list[int]) -> None:
        self._trajectory = trajectory
        self._index = 0
        self.steps_taken = 0
        self._open = True

    class _Config:
        horizon = 100

    config = _Config()

    def reset(self, *, seed: int = 0, options=None):
        self._index = 0
        self.steps_taken = 0
        return np.zeros(16, dtype=np.float64), {"seed": seed}

    def step(self, action):
        value = self._trajectory[min(self._index, len(self._trajectory) - 1)]
        self._index += 1
        self.steps_taken += 1
        info = {
            "cookies_in_target": value,
            "success": value >= 10,
            "applied_action": np.asarray(action, dtype=np.float64),
            # Both of these must be ignored rather than folded into a maximum: one
            # would raise, the other would invent a number.
            "phase": "inserting",
            "positions": np.zeros((3, 2)),
        }
        return np.zeros(16, dtype=np.float64), 0.0, False, False, info

    def close(self) -> None:
        self._open = False


class _HoldPolicy:
    action_mode = "joint_position"

    def reset(self, context: EpisodeContext) -> None:
        pass

    def act(self, observation, task: str) -> np.ndarray:
        return np.zeros(16, dtype=np.float64)

    def close(self) -> None:
        pass


def _run(trajectory: list[int]) -> tuple[object, _CounterEnv]:
    env = _CounterEnv(trajectory)
    runner = EpisodeRunner(env, _HoldPolicy(), task="test task")
    try:
        return runner.run(seed=0, max_steps=len(trajectory)), env
    finally:
        runner.close()


def test_peak_survives_the_fall_that_the_final_value_records():
    # Fills the box, then the policy wanders and knocks five out.
    result, _ = _run([0, 4, 8, 10, 10, 6, 3, 0])

    assert result.final_info["cookies_in_target"] == 0, "the final value is the problem"
    assert result.final_info["success"] is False
    assert result.peak_info["cookies_in_target"] == 10
    assert result.peak_info["success"] == 1.0


def test_peak_matches_the_final_value_when_the_episode_only_climbs():
    result, _ = _run([0, 2, 5])

    assert result.final_info["cookies_in_target"] == 5
    assert result.peak_info["cookies_in_target"] == 5


def test_peak_ignores_labels_and_arrays_and_keeps_scalars():
    result, _ = _run([0, 1, 2])

    assert "phase" not in result.peak_info, "a label has no maximum"
    assert "positions" not in result.peak_info, "an array has no maximum"
    assert "applied_action" not in result.peak_info
    assert "cookies_in_target" in result.peak_info


def test_an_episode_that_never_stepped_reports_no_peaks_rather_than_zeros():
    # ``max_steps=0`` would not exercise this: the runner reads ``max_steps or
    # horizon``, so a zero falls through to the scene's full budget.  A result built
    # without stepping is what the empty case actually looks like, and an empty
    # mapping is what it has to serialise to -- zeros would read as "counter was zero
    # every step", which is a measurement rather than an absence of one.
    payload = EpisodeResult(
        steps=0, terminated=False, truncated=False, safety_reason=None
    ).to_dict()

    assert payload["peak_info"] == {}
    json.dumps(payload)


def test_peakable_accepts_bools_and_rejects_non_numbers():
    assert _peakable(True) == 1.0
    assert _peakable(False) == 0.0
    assert _peakable(np.float32(0.5)) == 0.5
    assert _peakable(np.int64(3)) == 3.0
    assert _peakable(None) is None
    assert _peakable("inserting") is None
    assert _peakable([1, 2]) is None
    assert _peakable(np.array([1.0])) is None
    assert _peakable(float("nan")) is None
    assert _peakable(float("inf")) is None


def _metrics(seed: int, **task) -> EpisodeMetrics:
    return EpisodeMetrics(seed=seed, steps=10, terminated=False, truncated=False, task=task)


def _result(name: str, episodes: list[EpisodeMetrics]) -> ManifestResult:
    return ManifestResult(
        name=name,
        path=f"/tmp/{name}.yaml",
        regime="finetuned",
        primary_metric="both",
        prompt_kind="free_text",
        prompt_labels=(),
        notes="",
        dropped_cameras=(),
        action_space="joint_position",
        action_dim=16,
        episodes=episodes,
    )


def test_aggregate_separates_final_success_from_success_ever():
    result = _result(
        "wandered",
        [
            _metrics(0, success=False, success_peak=1.0, cookies_in_target=0, cookies_in_target_peak=10),
            _metrics(1, success=False, success_peak=1.0, cookies_in_target=0, cookies_in_target_peak=10),
            _metrics(2, success=False, success_peak=0.0, cookies_in_target=0, cookies_in_target_peak=5),
        ],
    )

    agg = result.aggregate()

    assert agg["task_successes"] == 0, "no episode ended in a success"
    assert agg["task_successes_peak"] == 2, "but two of them had one"


def test_progress_column_shows_the_mean_end_and_the_best_ever():
    result = _result(
        "mixed",
        [
            _metrics(0, cookies_in_target=0, cookies_in_target_peak=10),
            _metrics(1, cookies_in_target=0, cookies_in_target_peak=4),
        ],
    )

    assert _progress_cell(result) == "0.0 → 10.0"


def test_progress_column_is_absent_not_zero_when_unreported():
    result = _result("silent", [_metrics(0, success=False)])

    assert _progress_cell(result) == "—"


def _report(randomized: bool, seeds: tuple[int, ...] = (0, 1, 2)) -> ComparisonReport:
    return ComparisonReport(
        scene="a3_cookie_same_column",
        seeds=seeds,
        steps=400,
        task="fill the box",
        randomized=randomized,
        results=[
            _result("a", [_metrics(0, cookies_in_target=0, cookies_in_target_peak=10)]),
        ],
    )


def test_table_says_when_every_seed_ran_the_same_scene():
    text = format_table(_report(randomized=False))

    assert "All 3 seeds ran the same scene" in text
    assert "not a sample size" in text


def test_table_does_not_warn_when_the_layout_was_sampled():
    text = format_table(_report(randomized=True))

    assert "same scene" not in text
    assert "sampled per episode" in text


def test_single_seed_is_not_warned_about_because_there_is_nothing_to_repeat():
    text = format_table(_report(randomized=False, seeds=(0,)))

    assert "same scene" not in text


def test_table_carries_both_success_columns_and_the_progress_pair():
    text = format_table(_report(randomized=True))

    assert "success ever" in text
    assert "progress last → best" in text
    assert "0.0 → 10.0" in text
