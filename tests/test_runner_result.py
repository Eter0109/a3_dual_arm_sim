"""Tests for EpisodeResult serialisation.

Environment ``info`` carries numpy arrays (actions, object poses). Callers
serialise the result to JSON -- the CLI prints it, and evaluation scripts write
it to disk -- so the conversion has to happen in one place rather than in every
consumer.
"""

from __future__ import annotations

import json

import numpy as np

from a3_dual_arm_sim.runner import EpisodeResult


def test_to_dict_is_json_serialisable_with_array_info() -> None:
    result = EpisodeResult(
        steps=12,
        terminated=True,
        truncated=False,
        safety_reason=None,
        success=True,
        final_info={
            "success": True,
            "cookies_in_target": 10,
            "positions": np.zeros((3, 2), dtype=np.float64),
            "mask": np.array([True, False]),
            "scalar": np.float32(0.5),
        },
    )

    payload = result.to_dict()

    # Must not raise: this is exactly what broke the CLI and evaluation.
    encoded = json.dumps(payload)
    assert '"cookies_in_target": 10' in encoded


def test_arrays_become_lists_and_scalars_become_builtins() -> None:
    result = EpisodeResult(
        steps=1,
        terminated=False,
        truncated=True,
        safety_reason="none",
        final_info={
            "positions": np.array([[1.0, 2.0]]),
            "count": np.int64(7),
            "ratio": np.float32(0.25),
        },
    )

    info = result.to_dict()["final_info"]

    assert info["positions"] == [[1.0, 2.0]]
    assert info["count"] == 7
    assert isinstance(info["count"], int)
    assert isinstance(info["ratio"], float)
    assert info["ratio"] == 0.25


def test_nested_containers_are_converted_recursively() -> None:
    result = EpisodeResult(
        steps=1,
        terminated=False,
        truncated=False,
        safety_reason=None,
        final_info={
            "per_episode": [
                {"distance": np.float64(1.5), "path": np.array([0.0, 1.0])},
            ],
            "tuple_of_arrays": (np.array([1]), np.array([2])),
        },
    )

    info = result.to_dict()["final_info"]

    assert info["per_episode"][0]["distance"] == 1.5
    assert info["per_episode"][0]["path"] == [0.0, 1.0]
    assert info["tuple_of_arrays"] == [[1], [2]]
    json.dumps(info)


def test_scalar_fields_survive_unchanged() -> None:
    result = EpisodeResult(
        steps=99,
        terminated=True,
        truncated=False,
        safety_reason="joint velocity exceeded safety threshold",
        success=False,
        discarded=True,
    )

    payload = result.to_dict()

    assert payload["steps"] == 99
    assert payload["terminated"] is True
    assert payload["discarded"] is True
    assert payload["safety_reason"] == "joint velocity exceeded safety threshold"
    assert payload["final_info"] == {}


def test_default_result_serialises() -> None:
    # An empty final_info must not break the common teleop/hold path.
    json.dumps(EpisodeResult(1, False, False, None).to_dict())
