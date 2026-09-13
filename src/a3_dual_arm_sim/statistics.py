"""Normalization-statistics repair for datasets that are safe to train on.

A dimension whose values never change has zero variance by definition, but
LeRobot's aggregation across shards leaves a spurious tiny ``std`` there instead
of zero. The normaliser divides by ``std + 1e-8``, so a spurious 3.8e-7 turns
float32 rounding noise into a normalised magnitude of several hundred, and the
training loss starts around 3e3 instead of single digits.

Flooring ``std`` fixes that without touching real signal: a constant dimension
normalises to exactly zero, and any genuinely varying dimension has a spread far
above the floor. On the cookie dataset it only affects the right arm, which holds
the target bin and never moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Smallest standard deviation the normaliser is allowed to divide by.
#: Joint angles in this project move by tenths of a radian, so a hundredth
#: cannot mask real motion, while a constant dimension stays constant.
DEFAULT_STD_FLOOR = 0.01

#: Features the policy normalises. Visual features are left as-is.
NORMALIZED_KEYS = ("action", "observation.state")


def _feature_stats(stats: dict[str, Any], key: str) -> dict[str, Any] | None:
    entry = stats.get(key)
    if not isinstance(entry, dict) or "std" not in entry:
        return None
    return entry


def degenerate_dimensions(stats: dict[str, Any], *, floor: float = DEFAULT_STD_FLOOR) -> dict[str, list[int]]:
    """Indices whose stored ``std`` sits below ``floor``, per normalised feature."""

    degenerate: dict[str, list[int]] = {}
    for key in NORMALIZED_KEYS:
        entry = _feature_stats(stats, key)
        if entry is None:
            continue
        indices = [index for index, value in enumerate(entry["std"]) if float(value) < floor]
        if indices:
            degenerate[key] = indices
    return degenerate


def floor_normalization_std(
    dataset_root: Path | str,
    *,
    floor: float = DEFAULT_STD_FLOOR,
    apply: bool = True,
) -> dict[str, Any]:
    """Raise any ``std`` below ``floor`` in a dataset's ``meta/stats.json``.

    Returns a report describing what was found, so the caller can log it. With
    ``apply=False`` nothing is written, which makes the check usable as a
    pre-flight.
    """

    root = Path(dataset_root).expanduser().resolve()
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset has no meta/stats.json: {root}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    degenerate = degenerate_dimensions(stats, floor=floor)
    report: dict[str, Any] = {
        "root": str(root),
        "floor": floor,
        "degenerate_dimensions": degenerate,
        "changed": False,
    }
    if not degenerate:
        return report

    for key, indices in degenerate.items():
        std = list(_feature_stats(stats, key)["std"])
        for index in indices:
            std[index] = floor
        stats[key]["std"] = std

    if apply:
        temporary = stats_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        temporary.replace(stats_path)
        report["changed"] = True
    return report
