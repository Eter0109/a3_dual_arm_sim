"""Tests for the C51 value model and the value head.

Pure arithmetic -- no dataset, no checkpoint -- so they run in under a second and can gate every
change to the loss or the target projection.

The tests that matter most are the round-trip ones: `count_to_value` followed by `expected_value`
should return the count that went in, and `two_hot` should be a probability distribution for every
possible input.  Both were written because the first draft of the plan specified `count/10 - 1`,
which maps `[0, 20]` onto `[-1, 1]` and would have clipped half the distribution against a
`[-1, 0]` support.  A round-trip test catches that class of error immediately; reading the formula
does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from value_model import (  # noqa: E402
    DELTA_Z,
    NUM_ATOMS,
    V_MAX,
    V_MIN,
    HeadConfig,
    ValueHead,
    atoms,
    count_to_value,
    distributional_loss,
    expected_value,
    two_hot,
    value_to_count,
)


def test_atoms_span_the_support():
    grid = atoms()
    assert grid.shape == (NUM_ATOMS,)
    assert grid[0].item() == pytest.approx(V_MIN)
    assert grid[-1].item() == pytest.approx(V_MAX)
    # Evenly spaced: every gap equal to DELTA_Z.
    assert torch.allclose(grid[1:] - grid[:-1], torch.full((NUM_ATOMS - 1,), DELTA_Z))


@pytest.mark.parametrize("required", [10.0, 20.0])
def test_count_round_trip(required):
    """`count -> value -> count` must be the identity, which `x/10 - 1` would not be for 20."""

    counts = torch.arange(0, int(required) + 1, dtype=torch.float32)
    values = count_to_value(counts, required)
    assert values.min() >= V_MIN - 1e-6, "the value scale fell below the atom support"
    assert values.max() <= V_MAX + 1e-6, "the value scale rose above the atom support"
    assert torch.allclose(value_to_count(values, required), counts, atol=1e-4)


def test_count_maps_zero_to_minus_one_and_full_to_zero():
    assert count_to_value(torch.tensor([0.0]), 20.0).item() == pytest.approx(-1.0)
    assert count_to_value(torch.tensor([20.0]), 20.0).item() == pytest.approx(0.0)


@pytest.mark.parametrize("target", [-1.0, -0.5, 0.0, -0.35, -0.0625])
def test_two_hot_is_a_distribution(target):
    dist = two_hot(torch.tensor([target]))
    assert dist.shape == (1, NUM_ATOMS)
    assert dist.sum().item() == pytest.approx(1.0), "two-hot must sum to one"
    assert (dist >= 0).all()
    assert (dist <= 1).all()
    # At most two atoms are non-zero.
    assert int((dist[0] > 0).sum()) <= 2


def test_two_hot_handles_a_target_exactly_on_a_bin():
    """The `b_left == b_right` case: a naive two-hot writes 1.0 twice and sums to 2."""

    # `-1 + 130 * 0.005` is exactly a bin centre.
    exact = V_MIN + 130 * DELTA_Z
    dist = two_hot(torch.tensor([exact]))
    assert dist.sum().item() == pytest.approx(1.0)
    assert dist[0, 130].item() == pytest.approx(1.0)
    assert int((dist[0] > 0).sum()) == 1


def test_two_hot_clamps_out_of_range_targets():
    for target in (-5.0, 3.0):
        dist = two_hot(torch.tensor([target]))
        assert dist.sum().item() == pytest.approx(1.0)


def test_expected_value_reads_back_the_target():
    """A one-hot logit at the target's bin must decode to the target's count."""

    counts = torch.tensor([0.0, 5.0, 13.0, 20.0])
    values = count_to_value(counts, 20.0)
    bins = ((values - V_MIN) / DELTA_Z).round().long()
    logits = torch.full((len(counts), NUM_ATOMS), -20.0)
    logits[torch.arange(len(counts)), bins] = 20.0
    decoded = value_to_count(expected_value(logits), 20.0)
    assert torch.allclose(decoded, counts, atol=0.05), f"decoded {decoded} from {counts}"


def test_loss_is_minimised_by_the_correct_distribution():
    target = torch.tensor([-0.35])
    bins = ((target - V_MIN) / DELTA_Z).round().long()
    good = torch.full((1, NUM_ATOMS), -20.0)
    good[0, bins[0]] = 20.0
    wrong = torch.full((1, NUM_ATOMS), -20.0)
    wrong[0, 0] = 20.0
    assert distributional_loss(good, target).item() < distributional_loss(wrong, target).item()


def test_loss_has_no_reduction_argument():
    """Pinned deliberately: a `reduction` knob on an auxiliary loss has silently dropped it once
    (§9.81), so this loss always returns one value per sample and the caller reduces."""

    import inspect

    assert "reduction" not in inspect.signature(distributional_loss).parameters
    assert distributional_loss(torch.zeros(4, NUM_ATOMS), torch.zeros(4)).shape == (4,)


def test_head_shapes_and_gradient_reaches_every_branch():
    torch.manual_seed(0)
    config = HeadConfig(image_dim=64, language_dim=64, num_cameras=3)
    head = ValueHead(config)
    batch = 2
    logits = head(
        torch.randn(batch, 3, 64),
        torch.randn(batch, 64),
        torch.randn(batch, config.state_dim),
        torch.randn(batch, config.history_dim),
        torch.zeros(batch, config.progress_dim),
    )
    assert logits.shape == (batch, NUM_ATOMS)

    distributional_loss(logits, torch.tensor([-0.5, -0.2])).mean().backward()
    for name in ("image", "language", "state", "history", "progress", "trunk"):
        module = getattr(head, name)
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients, f"{name} received no gradient; that branch is not in the graph"
        assert any(float(g.abs().sum()) > 0 for g in gradients), f"{name} gradients are all zero"


def test_head_rejects_the_wrong_camera_count():
    """The image branch reshapes `[B, cameras, D]`; a mismatch must fail loudly at the reshape."""

    head = ValueHead(HeadConfig(image_dim=32, language_dim=32, num_cameras=3))
    with pytest.raises(RuntimeError):
        head(
            torch.randn(2, 5, 32),
            torch.randn(2, 32),
            torch.randn(2, 16),
            torch.randn(2, 128),
            torch.zeros(2, 2),
        )


def test_spearman_matches_a_known_ranking():
    from train_value import spearman

    assert spearman(np.arange(10, dtype=float), np.arange(10, dtype=float)) == pytest.approx(1.0)
    assert spearman(np.arange(10, dtype=float), -np.arange(10, dtype=float)) == pytest.approx(-1.0)
    # A monotone but non-linear relation is still a perfect rank correlation.
    assert spearman(np.arange(20, dtype=float), np.arange(20, dtype=float) ** 3) == pytest.approx(1.0)


import numpy as np  # noqa: E402  (after the pytest import block for readability)
