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
)

# Mirrors the cookie dataset: the left arm moves, the right arm holds the bin.
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
