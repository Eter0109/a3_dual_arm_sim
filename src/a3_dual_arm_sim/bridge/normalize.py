"""Undo a model's own action normalisation.

The GPU sizing guide's issue list is largely a list of normalisation mistakes:
VITRA's outputs were wrong until the authors' pretrained statistics were used,
and Fast-WAM misbehaves when only ``.pt`` weights are downloaded without
``*_dataset_stats.json``.  Both are cases where a model runs happily and acts
wrongly, so the inverse is applied here, explicitly, from a declared stats file --
never inferred from the observed action range.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import NormalizeSpec
from .protocol import BridgeError

#: Candidate spellings for each statistic, tried in order.  Different projects
#: publish the same numbers under different names, and a stats file is not worth
#: a schema argument -- but a *missing* one is a hard failure, never a
#: pass-through.
_ALIASES: Mapping[str, tuple[str, ...]] = {
    "mean": ("mean", "loc", "shift"),
    "std": ("std", "scale", "sigma"),
    "min": ("min", "minimum", "low"),
    "max": ("max", "maximum", "high"),
    "q01": ("q01", "quantile_01", "percentile_01", "p01"),
    "q99": ("q99", "quantile_99", "percentile_99", "p99"),
}

#: Which statistics each mode needs, and how to spell the pair in messages.
_REQUIRED: Mapping[str, tuple[str, str]] = {
    "mean_std": ("mean", "std"),
    "min_max": ("min", "max"),
    "q01_q99": ("q01", "q99"),
}


def _load_stats(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise BridgeError(f"normalisation statistics not found: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            return {name: handle[name] for name in handle.files}
    if path.suffix in (".json", ".yaml", ".yml"):
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix == ".json" else __import__("yaml").safe_load(text)
        if not isinstance(payload, Mapping):
            raise BridgeError(f"{path} must hold a mapping of statistic name to value")
        return payload
    raise BridgeError(f"unsupported statistics format {path.suffix!r}; use .npz or .json")


def _lookup(stats: Mapping[str, Any], key: str, name: str) -> np.ndarray:
    """Find one statistic under any of its common spellings."""

    candidates: list[str] = []
    for alias in (name, *_ALIASES.get(name, ())):
        candidates.extend((f"{key}.{alias}", alias, f"{key}_{alias}"))
    for candidate in candidates:
        if candidate in stats:
            return np.asarray(stats[candidate], dtype=np.float64)
    raise BridgeError(
        f"statistics file has no entry for {name!r} of {key!r}; tried "
        f"{', '.join(candidates)}. Keys present: {sorted(map(str, stats))}"
    )


class Normalizer:
    """Applies the inverse of a declared normalisation mode."""

    def __init__(self, spec: NormalizeSpec) -> None:
        self._mode = spec.mode
        self._stats: Mapping[str, np.ndarray] = {}
        if spec.mode == "none":
            return
        if spec.stats is None:
            raise BridgeError(f"normalisation mode {spec.mode!r} needs a stats file")
        raw = _load_stats(Path(spec.stats))
        for name in _REQUIRED[spec.mode]:
            self._stats[name] = _lookup(raw, spec.key, name)

    @property
    def mode(self) -> str:
        return self._mode

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Map model units back to physical units.

        ``mean_std`` inverts ``(x - mean) / std``; the range modes invert a
        ``[-1, 1]`` rescaling, which is the convention the statistics in this
        family of projects are published for.
        """

        if self._mode == "none":
            return actions
        values = np.asarray(actions, dtype=np.float64)
        first, second = _REQUIRED[self._mode]
        low = self._stats[first]
        high = self._stats[second]
        if low.shape != high.shape:
            raise BridgeError(
                f"{first} and {second} statistics disagree: {low.shape} vs {high.shape}"
            )
        if self._mode == "mean_std":
            # ``first``/``second`` are mean and std here, not the ends of a range,
            # so the range modes below must not share this arithmetic.
            if np.any(high == 0):
                raise BridgeError("std statistics contain zero, which would divide by zero")
            return values * high + low
        width = high - low
        if np.any(width == 0):
            raise BridgeError(f"{first}/{second} statistics are degenerate (zero range)")
        return values * (width / 2.0) + (high + low) / 2.0

    def expected_width(self) -> int | None:
        """The dimension the statistics were collected for, when they carry one."""

        for value in self._stats.values():
            if value.ndim == 1:
                return int(value.shape[0])
        return None
