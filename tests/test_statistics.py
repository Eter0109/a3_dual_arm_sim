"""Tests for normalization-statistics repair.

A dimension that never moves has zero variance, but dataset aggregation leaves a
spurious tiny ``std`` behind. Because the normaliser divides by ``std + eps``,
that artifact inflates float32 rounding noise into normalised values in the
hundreds -- enough to start training with a loss three orders of magnitude too
high. These tests pin the repair that prevents it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from a3_dual_arm_sim.statistics import (
    DEFAULT_STD_FLOOR,
    degenerate_dimensions,
    floor_normalization_std,
    recompute_quantiles,
)

# Synthetic statistics shaped like a real cookie dataset: one arm moving, the
# other parked, with LeRobot's aggregation artifacts in the parked dimensions.
# The values are fixtures, not a snapshot of any particular recording, so they do
# not need to track scene changes such as the right arm later learning to lift
# the target bin.
SPURIOUS_ACTION_STD = [
    0.1155, 0.1935, 0.2565, 0.2569, 0.2656, 0.2978, 0.0807, 0.0796,
    3.807e-07, 3.836e-08, 5.563e-03, 1.016e-03, 3.156e-07, 3.422e-04, 4.068e-10, 1.897e-05,
]
SPURIOUS_STATE_STD = [
    0.1161, 0.1961, 0.2528, 0.2548, 0.2653, 0.2979, 0.0808, 0.0408,
    0.0, 0.0033, 0.006, 0.0006, 0.0023, 0.0007, 0.0006, 0.0007,
]


def _dataset(tmp_path: Path, *, action_std: list[float], state_std: list[float]) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "stats.json").write_text(
        json.dumps(
            {
                "action": {"mean": [0.0] * 16, "std": action_std, "min": [-1.0] * 16, "max": [1.0] * 16},
                "observation.state": {
                    "mean": [0.0] * 16,
                    "std": state_std,
                    "min": [-1.0] * 16,
                    "max": [1.0] * 16,
                },
                "observation.images.front": {"mean": [[[0.5]]], "std": [[[0.2]]]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_degenerate_dimensions_finds_both_arms(tmp_path: Path) -> None:
    root = _dataset(tmp_path, action_std=SPURIOUS_ACTION_STD, state_std=SPURIOUS_STATE_STD)
    stats = json.loads((root / "meta" / "stats.json").read_text())

    found = degenerate_dimensions(stats)

    # Right arm joints and gripper only; the left arm's smallest std is 0.0807.
    assert found["action"] == [8, 9, 10, 11, 12, 13, 14, 15]
    assert found["observation.state"] == [8, 9, 10, 11, 12, 13, 14, 15]


def test_flooring_raises_only_degenerate_entries(tmp_path: Path) -> None:
    root = _dataset(tmp_path, action_std=SPURIOUS_ACTION_STD, state_std=SPURIOUS_STATE_STD)

    report = floor_normalization_std(root, floor=DEFAULT_STD_FLOOR)

    assert report["changed"] is True
    stats = json.loads((root / "meta" / "stats.json").read_text())
    std = stats["action"]["std"]
    # Constant dimensions become exactly the floor, so a constant input
    # normalises to zero instead of amplifying rounding noise.
    for index in range(8, 16):
        assert std[index] == DEFAULT_STD_FLOOR
    # Real motion is untouched, so no signal is compressed.
    assert std[:8] == SPURIOUS_ACTION_STD[:8]


def test_floored_stats_normalise_a_constant_dimension_to_zero() -> None:
    # What the normaliser computes: (x - mean) / (std + eps).
    eps = 1e-8
    x = -1.805  # the right shoulder holds one value for the whole dataset
    mean = -1.805

    before = (x - mean) / (3.807e-07 + eps)
    after = (x - mean) / (DEFAULT_STD_FLOOR + eps)

    assert after == 0.0
    assert before == 0.0  # exact equality still yields zero
    # The failure mode is float32 rounding, which the floor keeps negligible:
    # a 1e-4 error normalises to 1e-2 instead of 2.6e2.
    rounding_error = 1e-4
    assert rounding_error / (DEFAULT_STD_FLOOR + eps) < 0.02
    assert rounding_error / (3.807e-07 + eps) > 200


def test_a_dataset_without_degenerate_std_is_left_alone(tmp_path: Path) -> None:
    healthy = [0.1] * 16
    root = _dataset(tmp_path, action_std=healthy, state_std=healthy)

    report = floor_normalization_std(root)

    assert report["degenerate_dimensions"] == {}
    assert report["changed"] is False


def test_preflight_does_not_write_when_apply_is_false(tmp_path: Path) -> None:
    root = _dataset(tmp_path, action_std=SPURIOUS_ACTION_STD, state_std=SPURIOUS_STATE_STD)
    before = (root / "meta" / "stats.json").read_text()

    report = floor_normalization_std(root, apply=False)

    assert report["degenerate_dimensions"]
    assert report["changed"] is False
    assert (root / "meta" / "stats.json").read_text() == before


def test_missing_stats_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="stats.json"):
        floor_normalization_std(tmp_path / "absent")


def test_non_normalized_features_are_ignored(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "stats.json").write_text(
        json.dumps({"observation.force": {"mean": [0.0] * 18, "std": [0.0] * 18}}) + "\n",
        encoding="utf-8",
    )

    # The policy does not consume force, so its statistics are irrelevant here.
    assert floor_normalization_std(root)["degenerate_dimensions"] == {}


# ---------------------------------------------------------------------------
# Quantile repair
#
# The second defect, and the more dangerous one because it looks like data.  A
# dataset whose stored ``q01`` and ``q99`` are equal for a dimension that really
# moves leaves a QUANTILES normaliser dividing by ``eps``, so the loss starts at
# 1e12 rather than at a plausible number.  These tests use the shape the A3 cookie
# dataset actually had: the right arm holds one pose per episode, so each
# dimension is nearly constant *within* an episode while still moving 0.078 rad
# *between* them.
# ---------------------------------------------------------------------------

# 200 episodes, 16 dimensions, only the first two moving freely so the fixture
# stays small.  The right arm (8..14) is constant within an episode and drawn
# afresh per episode -- which is what makes its true quantiles span 0.078 while a
# per-episode aggregation collapses them onto the mean.
EPISODES = 200
FRAMES_PER_EPISODE = 8


def _recorded_rows() -> "np.ndarray":
    import numpy as np

    rng = np.random.default_rng(20260929)
    rows = []
    for episode in range(EPISODES):
        # One pose for this episode, shared by every frame in it.  The width matches the
        # scene's own `arm_home_rad` jitter (+/-0.04 rad), which is what gives the real
        # parked arm its ~0.078 rad span between episodes while it never moves within one.
        held = rng.uniform(-0.04, 0.04, size=7)
        for _ in range(FRAMES_PER_EPISODE):
            row = np.zeros(16, dtype=np.float64)
            # The left arm moves frame by frame, so only these dimensions have the
            # many-unique-values signature of genuine motion.
            row[:8] = rng.normal(1.0, 0.2, size=8)
            row[8:15] = held
            row[15] = 1.0  # the right gripper never moves at all
            rows.append(row)
    return np.stack(rows)


def _dataset_with_rows(tmp_path: Path) -> Path:
    """A v3 dataset tree with real parquet rows and the broken quantile pair."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet

    rows = _recorded_rows()
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    table = pa.table(
        {
            "action": pa.array(rows.tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(rows.tolist(), type=pa.list_(pa.float32())),
        }
    )
    parquet.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    # The defect, faithfully: the moving arm (0..7) keeps real quantiles, while the
    # parked arm (8..15) stores ``q01 == q99 == mean``.  Collapsing all sixteen would
    # also make the test pass, but it would not be the defect that occurred -- and the
    # distinction matters, because a repair that *recomputed everything* would then be
    # indistinguishable from one that fixed only what was broken.
    real_quantiles = {
        f"q{q:02d}": np.percentile(rows, q, axis=0).tolist() for q in (1, 10, 50, 90, 99)
    }
    collapsed = rows.mean(axis=0).tolist()
    stats = {}
    for key in ("action", "observation.state"):
        entry: dict[str, object] = {
            "min": rows.min(axis=0).tolist(),
            "max": rows.max(axis=0).tolist(),
            "mean": collapsed,
            "std": rows.std(axis=0).tolist(),
            "count": [len(rows)],
        }
        for name in ("q01", "q10", "q50", "q90", "q99"):
            entry[name] = [
                value if index < 8 else collapsed[index]
                for index, value in enumerate(real_quantiles[name])
            ]
        stats[key] = entry
    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return root


def test_recompute_finds_the_collapsed_quantiles(tmp_path: Path) -> None:
    import numpy as np

    root = _dataset_with_rows(tmp_path)

    report = recompute_quantiles(root, apply=False)

    # Dimensions 8..14 hold a real per-episode value, so their true span is
    # non-zero; the stored pair claims zero.  Dimension 15 is genuinely constant
    # and must NOT be reported as repaired -- the difference between "missing"
    # and "flat" is the whole point of the report.
    assert report["features"]["action"]["repaired_dimensions"] == [8, 9, 10, 11, 12, 13, 14]
    assert report["features"]["action"]["constant_dimensions"] == [15]
    assert report["changed"] is False

    detail = {row["dim"]: row for row in report["features"]["action"]["detail"]}
    for index in range(8, 15):
        assert detail[index]["stored_span"] == 0.0
        assert detail[index]["actual_span"] > 0.05
        assert detail[index]["unique_values"] == EPISODES
    assert detail[15]["unique_values"] == 1


def test_repaired_quantiles_normalise_a_real_dimension_into_range(tmp_path: Path) -> None:
    import numpy as np

    root = _dataset_with_rows(tmp_path)
    rows = _recorded_rows()

    recompute_quantiles(root)
    stats = json.loads((root / "meta" / "stats.json").read_text())["action"]
    q01 = np.asarray(stats["q01"])
    q99 = np.asarray(stats["q99"])

    # What the normaliser computes for QUANTILES, with its eps guard.
    eps = 1e-8
    before = 2.0 * (rows[:, 8] - rows[:, 8].mean()) / (0.0 + eps) - 1.0
    denominator = np.where(q99 - q01 == 0, eps, q99 - q01)
    after = 2.0 * (rows[:, 8] - q01[8]) / denominator[8] - 1.0

    # Before: the eps guard turns a 0.02 rad deviation into a magnitude of ~4e6,
    # which is the loss of 1e12 in squared form.
    assert np.abs(before).max() > 1e5
    # After: the same values land in the [-1, 1] range the discretizer assumes.
    assert np.abs(after).max() < 1.5


def test_recompute_leaves_a_backup_before_writing(tmp_path: Path) -> None:
    root = _dataset_with_rows(tmp_path)
    original = (root / "meta" / "stats.json").read_text()

    report = recompute_quantiles(root)

    assert report["changed"] is True
    backup = root / "meta" / "stats.json.bak"
    assert backup.is_file()
    assert backup.read_text() == original
    # The backup is written once and not overwritten by a second pass, so the
    # original survives re-running the repair.
    recompute_quantiles(root)
    assert backup.read_text() == original


def test_recompute_raises_a_degenerate_span_but_keeps_the_midpoint(tmp_path: Path) -> None:
    import numpy as np

    root = _dataset_with_rows(tmp_path)
    rows = _recorded_rows()

    recompute_quantiles(root)
    stats = json.loads((root / "meta" / "stats.json").read_text())["action"]
    q01 = np.asarray(stats["q01"])
    q99 = np.asarray(stats["q99"])

    # The repaired pair brackets the column it came from, as percentiles do: q01 is at
    # least the minimum (equal only when the minimum is not an outlier) and q99 is at most
    # the maximum.  Stated in that direction because the other is what a reader expects
    # from "raises the span", and it is wrong -- widening a *span* does not mean moving the
    # bounds outwards past the data.
    assert q01[8] >= rows[:, 8].min() - 1e-9
    assert q99[8] <= rows[:, 8].max() + 1e-9
    assert q99[8] - q01[8] > 0

    # Dimension 15 is genuinely constant, so its quantiles are left alone: both are the
    # single value, and the normaliser maps it to exactly -1 rather than dividing by eps.
    assert q01[15] == q99[15] == rows[:, 15][0]


def test_recompute_requires_a_data_directory(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "stats.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="parquet"):
        recompute_quantiles(root)
