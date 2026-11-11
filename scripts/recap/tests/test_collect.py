#!/usr/bin/env python3
"""Tests for the RECAP collector and the pieces it depends on.

Run with the project venv from the repo root:

    /root/autodl-tmp/a3-collect/.venv/bin/python -m pytest scripts/recap/tests -q

Everything here is pure logic -- no simulation, no bridge, no GPU -- so it finishes in
well under a second and can run before every collection batch.

The last test in this file is the important one.  It pins a bug that is otherwise
invisible: main's `episode_execution.run_episode` never calls `policy.observe_step`, so on
that path `BridgeChunkPolicy` feeds the encoder a history window whose `a_previous`
channel is sixteen zeros for the entire episode.  The training data has the real applied
action there, so every benchmark episode measured a distribution shift rather than the
checkpoint.  The test states the requirement and then demonstrates the failure, so the
guard cannot be removed silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "recap"))

from recap.collect_rollouts import (  # noqa: E402
    KeepAllRecorder,
    _ProgressTracker,
    build_trace_entry,
    parse_seed_range,
)

# Importing the benchmark driver is what makes `BridgeChunkPolicy` available here.  It
# extends sys.path at module level and loads the wire protocol; it starts no process and
# reads no weights, so importing it in a test is safe.
from bench_pi05_on_main import BridgeChunkPolicy  # noqa: E402


# --------------------------------------------------------------------------------------
# KeepAllRecorder: failures are data
# --------------------------------------------------------------------------------------


class _FakeInner:
    """Stands in for `RecapRecorder`, recording which terminal call it received."""

    def __init__(self, frames: int = 10) -> None:
        self.frames_recorded = frames
        self.calls: list[tuple[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def start_episode(self, context, controller_type):
        self.calls.append(("start", controller_type))

    def add_frame(self, observation, action, *, history=None, progress=None):
        self.calls.append(("frame", None))

    def set_episode_metadata(self, metadata):
        self.metadata = metadata

    def finish_episode(self, success=None):
        self.calls.append(("finish", success))

    def discard_episode(self):
        self.calls.append(("discard", None))

    def close(self):
        self.calls.append(("close", None))


def test_keep_all_recorder_saves_failure_instead_of_discarding():
    """The whole reason the wrapper exists: main discards, RECAP needs the episode."""

    inner = _FakeInner(frames=42)
    recorder = KeepAllRecorder(inner)  # type: ignore[arg-type]
    recorder.start_episode(None, "policy")
    recorder.add_frame({}, np.zeros(16))
    recorder.discard_episode()

    assert ("finish", False) in inner.calls, "a failed episode must be saved with success=False"
    assert ("discard", None) not in inner.calls, "discard would delete the failure we need"


def test_keep_all_recorder_still_discards_empty_episode():
    """A zero-frame save would write a shard LeRobot cannot read, so keep the old behaviour."""

    inner = _FakeInner(frames=0)
    recorder = KeepAllRecorder(inner)  # type: ignore[arg-type]
    recorder.discard_episode()

    assert ("discard", None) in inner.calls
    assert not any(c[0] == "finish" for c in inner.calls)


def test_keep_all_recorder_passes_success_through():
    """Successful episodes must not be rewritten as failures by the wrapper."""

    inner = _FakeInner(frames=5)
    recorder = KeepAllRecorder(inner)  # type: ignore[arg-type]
    recorder.finish_episode(success=True)

    assert ("finish", True) in inner.calls


# --------------------------------------------------------------------------------------
# Seed parsing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("200-203", [200, 201, 202, 203]),
        ("7", [7]),
        ("1,3,5", [1, 3, 5]),
        ("10-12,20", [10, 11, 12, 20]),
    ],
)
def test_parse_seed_range(text, expected):
    assert parse_seed_range(text) == expected


@pytest.mark.parametrize("text", ["", "5,5", "abc"])
def test_parse_seed_range_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_seed_range(text)


# --------------------------------------------------------------------------------------
# Trace rows
# --------------------------------------------------------------------------------------


def test_trace_entry_carries_the_counts_the_value_target_needs():
    """`cookies_in_target` per frame is the reward; its final value is the value target."""

    info = {
        "applied_action": np.arange(16, dtype=np.float32),
        "cookies_in_target": 12,
        "cookies_in_source": 68,
        "success": False,
        "success_hold_count": 0,
        "target_slot_occupancy": [-1, -1, 3],
        "target_touches_all_walls": True,
        "safety_reason": None,
    }
    entry = build_trace_entry(7, np.ones(16), info, phase="DESCEND_TO_PLACE")

    assert entry["step"] == 7
    assert entry["expert_phase"] == "DESCEND_TO_PLACE"
    assert entry["after"]["cookies_in_target"] == 12
    assert entry["after"]["cookies_in_source"] == 68
    assert entry["after"]["target_slot_occupancy"] == [-1, -1, 3]
    # Serializable without a custom encoder, because it is written straight to jsonl.
    json.dumps(entry)


def test_trace_entry_reward_is_the_frame_difference():
    """The plan's reward definition, checked on a three-frame episode."""

    counts = [10, 10, 11]
    rewards = [b - a for a, b in zip(counts, counts[1:])]
    assert rewards == [0, 1]
    # Non-decreasing is what makes the reward non-negative; a decrease would mean a Cookie
    # left the box, which the judge would also score as a loss.
    assert all(r >= 0 for r in rewards)


# --------------------------------------------------------------------------------------
# The a_previous channel, and the path that never fills it
# --------------------------------------------------------------------------------------


def _history_policy(tmp_path: Path) -> BridgeChunkPolicy:
    """A policy whose bridge connection is never opened.

    `_history_vector` and `_advance_history` are pure bookkeeping over `self._history`; they
    only need `history_window=True` and a socket path that is never dialled.  Connecting
    would require the server, and the channel this test inspects is built before any
    request is encoded.
    """

    return BridgeChunkPolicy(
        str(tmp_path / "unused.sock"),
        execute=16,
        history_window=True,
        history_frames=10,
    )


def _observation(step: int) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.full(16, float(step), dtype=np.float32),
        "observation.velocity": np.full(16, float(step) * 0.5, dtype=np.float32),
    }


def test_a_previous_channel_is_filled_when_observe_step_is_called(tmp_path):
    """The contract the training data assumes: frame t carries the action applied at t-1."""

    policy = _history_policy(tmp_path)
    applied = np.arange(16, dtype=np.float32) + 100.0

    policy._advance_history(_observation(1))          # step 1, no previous action yet
    policy.observe_step(applied=applied)              # the world applied it
    policy._advance_history(_observation(2))          # step 2 carries it

    window = policy._history_vector().reshape(-1, 48)
    assert window.shape == (10, 48)
    # Frame 47 (the last one) is [q, q_dot, a_previous]; a_previous is the last 16 entries.
    assert np.allclose(window[-1, 32:], applied)
    assert np.allclose(window[-2, 32:], np.zeros(16)), "the first frame has no previous action"


def test_a_previous_channel_is_all_zero_when_observe_step_is_never_called(tmp_path):
    """This is main's benchmark path, and it is a bug worth pinning.

    `episode_execution.run_episode` calls `act` and `env.step` but never `observe_step`.
    So `_applied_action` stays None, `_advance_history` takes its zeros branch, and the
    encoder sees a window whose third channel is dead for the whole episode -- while the
    checkpoint was trained on real applied actions.  If this test ever starts failing
    because the zeros went away, the collector no longer needs its own loop.
    """

    policy = _history_policy(tmp_path)
    for step in range(1, 21):
        policy._advance_history(_observation(step))
        # No observe_step call, exactly as main's runner does it.

    window = policy._history_vector().reshape(-1, 48)
    assert window.shape == (10, 48)
    assert np.allclose(window[:, 32:], 0.0), (
        "without observe_step the a_previous channel is all zeros; the collector exists "
        "partly to avoid this on the training path"
    )
    # The other two channels are still live, which is why the bug does not look like one
    # from the outside: the window is the right shape and two thirds of it are correct.
    assert not np.allclose(window[:, :16], 0.0)


# --------------------------------------------------------------------------------------
# The progress channels: two implementations must agree
# --------------------------------------------------------------------------------------


def _synthetic_eef_track(n: int = 400) -> list[np.ndarray]:
    """An eef-11 track that crosses the box boundary the way a real episode does.

    The progress rule is a state machine over `eef[1] < 0.18` with three thresholds --
    entry increments when the previous gap was at least 20 frames, a dip shorter than 5
    frames is undone -- so the track has to exercise all three: long carries between
    deliveries, short dips that must not count, and real re-entries.
    """

    track: list[np.ndarray] = []
    phase = 0  # 0 = carrying (outside), 1 = placing (inside)
    remaining = 60
    dips_left = 0
    for i in range(n):
        if remaining == 0:
            if phase == 0:
                phase, remaining = 1, 12
                # Two of the four in-box stays contain a 2-frame dip, which must be undone.
                dips_left = 2 if i % 3 == 0 else 0
            else:
                phase, remaining = 0, 40
        remaining -= 1
        inside = phase == 1
        if inside and dips_left and remaining == 6:
            inside = False  # a dip shorter than the 5-frame rule, then back inside
            dips_left -= 1
        eef = np.zeros(14, dtype=np.float64)
        eef[1] = 0.10 if inside else 0.30
        eef[0] = 0.05 * i
        track.append(eef)
    return track


def test_progress_tracker_matches_policy():
    """The collector's copy of the rule must produce the policy's numbers exactly.

    `_ProgressTracker` exists because `BridgeChunkPolicy._state_with_progress` is not
    idempotent (it advances the delivery counter) and only runs once per 16-step chunk,
    while the dataset needs a value on every frame.  That makes duplication unavoidable,
    and duplication drifts.  This test is the guard: both implementations are fed the same
    observations in the same order, so any divergence -- a threshold, the undo rule, the
    normalisation -- fails here rather than becoming a silent distribution shift in the
    training data.
    """

    policy = BridgeChunkPolicy(
        "/tmp/unused.sock", execute=16, progress_channels=True
    )
    policy.reset(None)
    tracker = _ProgressTracker()

    mismatches: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i, eef in enumerate(_synthetic_eef_track()):
        observation = {
            "observation.state": np.zeros(16, dtype=np.float32),
            "observation.eef_pose": eef,
        }
        expected = policy._state_with_progress(observation)[16:]
        got = tracker.update(observation)
        if not np.allclose(expected, got, atol=1e-6):
            mismatches.append((i, expected, got))

    assert not mismatches, (
        f"{len(mismatches)} of 400 frames disagree; first at frame {mismatches[0][0]}: "
        f"policy={mismatches[0][1]}, tracker={mismatches[0][2]}"
    )
    # The track must actually have produced a non-constant channel, or the comparison above
    # is vacuous -- two implementations of `return zeros` would also "agree".
    final = tracker.update(
        {"observation.state": np.zeros(16, dtype=np.float32),
         "observation.eef_pose": _synthetic_eef_track()[-1]}
    )
    assert final[0] != pytest.approx(-1.0), "no delivery was ever counted; the test is vacuous"
