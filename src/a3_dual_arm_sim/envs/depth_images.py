"""Fixed-distance depth previews, separate from policy observation contracts.

These three-channel grayscale images are optional diagnostics, not RGB and not
a silently introduced fourth LeRobot image channel. Raw depth stays in meters.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _depth_array(depth_m: np.ndarray) -> np.ndarray:
    array = np.asarray(depth_m)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("depth must be a nonempty HW array")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("depth must have a floating-point dtype with values in meters")
    return array


def depth_to_grayscale(
    depth_m: np.ndarray, *, near_m: float, far_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 HWC preview and explicit bool HW validity mask.

    Finite nonnegative values are valid. Nearer than ``near_m`` is white;
    ``far_m`` and farther is black. NaN, infinity and negative values are black
    with a False mask. No per-frame automatic normalization is performed.
    """
    array = _depth_array(depth_m)
    if (not np.isfinite(near_m) or not np.isfinite(far_m)
            or near_m < 0 or far_m <= near_m):
        raise ValueError("visualization range must satisfy finite 0 <= near_m < far_m")
    valid = np.isfinite(array) & (array >= 0)
    gray = np.zeros(array.shape, dtype=np.uint8)
    # Indexing valid entries avoids warnings and undefined NaN-to-integer casts.
    values = array[valid].astype(np.float64)
    normalized = np.clip((values - near_m) / (far_m - near_m), 0.0, 1.0)
    gray[valid] = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    preview = np.repeat(gray[:, :, None], 3, axis=2)
    return np.ascontiguousarray(preview), np.ascontiguousarray(valid)


def depth_statistics(depth_m: np.ndarray) -> dict[str, Any]:
    """Describe only finite nonnegative metric depth; never rescale it."""
    array = _depth_array(depth_m)
    valid = np.isfinite(array) & (array >= 0)
    values = array[valid].astype(np.float64)
    result: dict[str, Any] = {
        "shape_hw": list(array.shape), "dtype": str(array.dtype),
        "valid_pixels": int(np.count_nonzero(valid)),
        "invalid_pixels": int(np.count_nonzero(~valid)),
        "valid_fraction": float(np.mean(valid)),
    }
    if not values.size:
        return {**result, "min_m": None, "max_m": None, "quantiles_m": None}
    quantiles = (0.01, 0.05, 0.5, 0.95, 0.99)
    return {
        **result, "min_m": float(np.min(values)), "max_m": float(np.max(values)),
        "quantiles_m": {str(q): float(value)
                        for q, value in zip(quantiles, np.quantile(values, quantiles), strict=True)},
    }
