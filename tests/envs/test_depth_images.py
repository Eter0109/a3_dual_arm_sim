from __future__ import annotations

import numpy as np
import pytest

from a3_dual_arm_sim.envs.depth_images import depth_statistics, depth_to_grayscale


def test_fixed_range_near_white_far_black_and_three_equal_channels():
    depth = np.asarray([[0.0, 0.5, 1.5, 2.5, 3.0]], dtype=np.float32)
    preview, valid = depth_to_grayscale(depth, near_m=0.5, far_m=2.5)
    assert preview.shape == (1, 5, 3) and preview.dtype == np.uint8
    np.testing.assert_equal(preview[0, :, 0], [255, 255, 128, 0, 0])
    np.testing.assert_equal(preview[:, :, 0], preview[:, :, 1])
    np.testing.assert_equal(preview[:, :, 0], preview[:, :, 2])
    assert valid.all() and valid.dtype == np.bool_


def test_invalid_values_black_with_explicit_mask_no_input_mutation():
    depth = np.asarray([[np.nan, np.inf, -np.inf, -0.1, 0.02, 0.5]], dtype=np.float64)
    before = depth.copy()
    preview, valid = depth_to_grayscale(depth, near_m=0.02, far_m=0.5)
    np.testing.assert_equal(valid, [[False, False, False, False, True, True]])
    np.testing.assert_equal(preview[0, :4], np.zeros((4, 3), dtype=np.uint8))
    np.testing.assert_equal(depth, before)


def test_identical_metric_depth_keeps_same_intensity_despite_frame_extrema():
    first, _ = depth_to_grayscale(np.asarray([[1.5, 0.5]], dtype=np.float32), near_m=0.5, far_m=2.5)
    second, _ = depth_to_grayscale(np.asarray([[1.5, 20.0]], dtype=np.float32), near_m=0.5, far_m=2.5)
    np.testing.assert_equal(first[0, 0], second[0, 0])


@pytest.mark.parametrize("depth", [np.zeros((2, 2), dtype=np.int32), np.zeros((2, 2), dtype=np.uint16),
                                  np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=np.complex64)])
def test_nonfloating_depth_is_rejected(depth):
    with pytest.raises(TypeError, match="floating-point"):
        depth_to_grayscale(depth, near_m=0.02, far_m=0.5)


@pytest.mark.parametrize("shape", [(2,), (2, 2, 1), (0, 2), (2, 0)])
def test_non_hw_or_empty_depth_is_rejected(shape):
    with pytest.raises(ValueError, match="nonempty HW"):
        depth_to_grayscale(np.zeros(shape, dtype=np.float32), near_m=0.02, far_m=0.5)


@pytest.mark.parametrize("near,far", [(0.5, 0.5), (0.5, 0.1), (-0.1, 0.5),
                                     (np.nan, 0.5), (0.02, np.inf)])
def test_invalid_visualization_ranges_are_rejected(near, far):
    with pytest.raises(ValueError, match="visualization range"):
        depth_to_grayscale(np.ones((2, 2), dtype=np.float32), near_m=near, far_m=far)


def test_stats_exclude_invalid_depth_and_preserve_meter_units():
    values = np.asarray([[0.5, 1.5], [2.5, np.nan]], dtype=np.float32)
    stats = depth_statistics(values)
    assert stats["valid_pixels"] == 3 and stats["invalid_pixels"] == 1
    assert stats["valid_fraction"] == 0.75
    assert stats["min_m"] == 0.5 and stats["max_m"] == 2.5
    assert stats["quantiles_m"]["0.5"] == 1.5


def test_all_invalid_frame_has_black_preview_and_null_statistics():
    values = np.full((2, 2), np.nan, dtype=np.float32)
    preview, valid = depth_to_grayscale(values, near_m=0.02, far_m=0.5)
    assert not preview.any() and not valid.any()
    stats = depth_statistics(values)
    assert stats["min_m"] is None and stats["max_m"] is None and stats["quantiles_m"] is None
