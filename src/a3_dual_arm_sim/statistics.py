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

import numpy as np

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


#: Quantiles LeRobot stores per feature, and what π0.5's QUANTILES mode reads.
#: ``q01``/``q99`` are the pair the normaliser divides by; the rest are carried along so
#: the file stays internally consistent with what a fresh aggregation would have written.
QUANTILES = ("q01", "q10", "q50", "q90", "q99")

#: A stored quantile span below this is treated as missing rather than as narrow.
#:
#: Not an exact-zero test, and the reason is visible in the A3 dataset: the action's
#: degenerate dimensions store exactly ``0.0`` while the state's store ``1.5e-9``, because
#: the two were aggregated by different code paths.  Both divide the normaliser by
#: ``eps = 1e-8``, so both are the same defect; a ``<= 0`` test catches only one.  A
#: thousandth of a radian is far below any real joint motion in this project -- the
#: smallest genuine span in this dataset is 0.075 rad, seventy-five times the threshold.
DEGENERATE_SPAN = 1e-3


def recompute_quantiles(
    dataset_root: Path | str,
    *,
    features: tuple[str, ...] = NORMALIZED_KEYS,
    apply: bool = True,
) -> dict[str, Any]:
    """Recompute ``q01``/``q99`` (and the rest) from the recorded rows.

    Why this exists
    ---------------
    ``floor_normalization_std`` above handles a *missing* scale.  This handles a *wrong*
    one, which is what the A3 cookie dataset turned out to have and which is worse because
    it looks plausible.

    A3's ``action`` statistics for the right arm store ``q01 == q99 == mean``: a span of
    exactly zero on a dimension that actually moves 0.077 rad.  So π0.5's QUANTILES
    normaliser divides by ``eps = 1e-8`` instead of by the range, turning a 0.04 rad
    deviation into a normalised magnitude of ~8e6, and the first training step reports a
    loss of **1.2e12** -- a number so far off that it reads as divergence rather than as a
    metadata defect.

    The tell is cheap to check and worth stating: **a stored quantile pair that is exactly
    equal is not a narrow distribution, it is a missing statistic.**  Real variation
    essentially never gives ``q01 == q99`` exactly.  Comparing the stored values against
    the raw column, as this function does, separates "this dimension is constant" (dim 15
    of the action: one unique value in 110,485 rows, span genuinely zero) from "this
    dimension's quantiles were never computed".

    Why not lerobot's own ``augment_dataset_quantile_stats``
    ------------------------------------------------------
    It does the same arithmetic, but it walks the whole dataset (including the videos) and
    ends by pushing to the Hub and re-tagging the repo -- neither of which works here, and
    both of which are more than the repair needs.  Reading two parquet columns is enough,
    and it finishes in seconds instead of over the image data.

    The original file is copied to ``meta/stats.json.bak`` before the first write, so the
    change is reversible and visible in the dataset directory rather than implied.
    """

    root = Path(dataset_root).expanduser().resolve()
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset has no meta/stats.json: {root}")

    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"Dataset has no data parquet files under {root / 'data'}")

    # Only the columns being repaired are read.  Selecting them explicitly keeps the
    # 14-value eef_pose, the 18-value force column and every image statistic out of the
    # working set -- and out of the diff.
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("pyarrow is required to recompute quantiles") from exc

    report: dict[str, Any] = {
        "root": str(root),
        "data_files": len(data_files),
        "features": {},
        "backup": None,
        "changed": False,
    }

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for feature in features:
        if feature not in stats:
            continue
        columns = [parquet.read_table(path, columns=[feature]) for path in data_files]
        values = np.concatenate(
            [np.stack(table.column(feature).to_pylist()) for table in columns], axis=0
        ).astype(np.float64)

        before_q01 = np.asarray(stats[feature].get("q01"), dtype=np.float64)
        before_q99 = np.asarray(stats[feature].get("q99"), dtype=np.float64)
        # `axis=0` is not optional.  `np.percentile` defaults to `axis=None`, which
        # *flattens* the array first -- so a per-dimension quantile silently becomes one
        # quantile over every dimension pooled together, and the result is a scalar that
        # then fails to index.  LeRobot's own stats are computed per dimension, so this
        # has to match.
        after = {
            name: np.percentile(values, int(name[1:]), axis=0).tolist() for name in QUANTILES
        }
        after_q01 = np.asarray(after["q01"])
        after_q99 = np.asarray(after["q99"])

        rows = []
        for index in range(values.shape[1]):
            rows.append(
                {
                    "dim": index,
                    "stored_span": float(before_q99[index] - before_q01[index]),
                    "actual_span": float(after_q99[index] - after_q01[index]),
                    "unique_values": int(np.unique(values[:, index]).size),
                }
            )

        # A dimension that really is constant: one unique value, and a stored span of zero
        # is then correct rather than missing.  Reported so that "no change" on such a
        # dimension is not mistaken for the repair having failed.
        constant = [row["dim"] for row in rows if row["unique_values"] == 1]
        repaired = [
            row["dim"]
            for row in rows
            if row["stored_span"] < DEGENERATE_SPAN and row["unique_values"] > 1
        ]
        report["features"][feature] = {
            "rows": len(values),
            "dimensions": values.shape[1],
            "constant_dimensions": constant,
            "repaired_dimensions": repaired,
            "detail": rows,
        }

        stats[feature] = {**stats[feature], **after}

    if apply:
        backup = stats_path.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_bytes(stats_path.read_bytes())
            report["backup"] = str(backup)
        temporary = stats_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        temporary.replace(stats_path)
        report["changed"] = True

    return report
