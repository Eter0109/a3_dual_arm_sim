"""Project an A3 observation onto the names a manifest declares.

The LeRobot adapter in ``lerobot_policy.py`` already learned why this cannot be
delegated: LeRobot accepts a missing camera silently, returns a well-shaped
action, warns about nothing, and quietly degrades the rollout (finding 8 in
``docs/policy-interface.md``).  An external bridge has the same failure mode plus
one of its own -- a key the model needs but the manifest misspelled -- so the
projection fails loudly instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from ..contracts import ContractError
from .manifest import ObservationSpec

#: Cameras are the only arrays that get resized; a resize on a state vector would
#: silently change its meaning rather than its resolution.
_RESIZE_INTERPOLATION = cv2.INTER_AREA


class ObservationView:
    """Turn an environment observation into the arrays a model expects."""

    def __init__(self, spec: ObservationSpec) -> None:
        self._spec = spec
        self._image_names = tuple(spec.images)
        self._vector_names = tuple(spec.vectors)

    @property
    def image_names(self) -> tuple[str, ...]:
        return self._image_names

    @property
    def vector_names(self) -> tuple[str, ...]:
        return self._vector_names

    @property
    def names(self) -> tuple[str, ...]:
        """Every array name this view will emit, in wire order."""

        return self._image_names + self._vector_names

    def missing_keys(self, observation: Mapping[str, Any]) -> list[str]:
        """A3 keys this view needs but the observation does not carry."""

        needed = {spec.source for spec in self._spec.images.values()}
        needed |= {spec.source for spec in self._spec.vectors.values()}
        return sorted(key for key in needed if key not in observation)

    def project(
        self, observation: Mapping[str, Any], task: str
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Return ``(arrays, fields)`` ready for :func:`protocol.encode_request`.

        Raises :class:`ContractError` naming every key this observation cannot
        supply, because a partially-wired bridge is worse than a failed one.
        """

        missing = self.missing_keys(observation)
        if missing:
            raise ContractError(
                f"manifest expects {missing}, which the observation does not carry. "
                f"Present keys: {sorted(observation)}"
            )
        arrays: dict[str, np.ndarray] = {}
        for name, spec in self._spec.images.items():
            arrays[name] = self._project_image(name, observation[spec.source], spec.resize)
        for name, spec in self._spec.vectors.items():
            arrays[name] = self._project_vector(name, observation[spec.source], spec)
        return arrays, {"task": task}

    def _project_image(
        self, name: str, image: Any, resize: tuple[int, int] | None
    ) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ContractError(
                f"{name}: an A3 camera must be HxWx3 RGB, got shape {value.shape}"
            )
        if value.dtype != np.uint8:
            raise ContractError(f"{name}: an A3 camera must be uint8, got {value.dtype}")
        if resize is not None and (value.shape[0], value.shape[1]) != resize:
            value = cv2.resize(value, (resize[1], resize[0]), interpolation=_RESIZE_INTERPOLATION)
        return np.ascontiguousarray(value, dtype=np.uint8)

    def _project_vector(self, name: str, vector: Any, spec: Any) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if value.ndim != 1:
            raise ContractError(f"{name}: expected a 1-D vector, got shape {value.shape}")
        if spec.slice is not None:
            start, stop = spec.slice
            if stop > value.shape[0]:
                raise ContractError(
                    f"{name}: slice {spec.slice} exceeds the {value.shape[0]}-dimensional "
                    f"source {spec.source!r}"
                )
            value = value[start:stop]
        elif spec.indices is not None:
            # Non-contiguous selection, which is what a checkpoint trained on a
            # different joint layout needs: ALOHA has six joints plus a gripper
            # per arm against A3's seven, so two of A3's sixteen values have no
            # counterpart and must be dropped rather than appended.
            highest = max(spec.indices, default=-1)
            if highest >= value.shape[0]:
                raise ContractError(
                    f"{name}: indices reach {highest} but the source {spec.source!r} is "
                    f"{value.shape[0]}-dimensional"
                )
            value = value[np.asarray(spec.indices, dtype=np.int64)]
        dtype = np.dtype(spec.dtype)
        if dtype.kind != "f":
            raise ContractError(f"{name}: observation dtype must be floating point, got {dtype}")
        return np.ascontiguousarray(value, dtype=dtype)
