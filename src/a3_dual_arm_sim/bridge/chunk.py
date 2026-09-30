"""Spread a predicted action chunk over control steps.

A VLA runs at 3-10 Hz while the environment steps at 20 Hz, so a model that
returned one action per observation would either be asked to infer faster than it
can or would slow the rollout to its own rate.  Chunks are how the existing
LeRobot path already works -- ``select_action`` keeps its own queue, and
``docs/policy-interface.md`` finding 12 measured that executing a whole chunk is
both cheaper and no worse than re-planning every step.

This class is that queue, without the framework attached.
"""

from __future__ import annotations

import numpy as np

from .protocol import BridgeError


class ChunkBuffer:
    """Hold one predicted chunk and hand out one action per step.

    ``execute`` may be shorter than the chunk: the tail is then dropped rather
    than run open loop, which is how a caller trades inference cost against
    reactivity without changing the model.
    """

    def __init__(self, chunk: int, execute: int) -> None:
        if chunk < 1:
            raise BridgeError("chunk must be at least 1")
        if execute < 1:
            raise BridgeError("execute must be at least 1")
        if execute > chunk:
            raise BridgeError(f"execute ({execute}) cannot exceed chunk ({chunk})")
        self._chunk = chunk
        self._execute = execute
        self._actions: np.ndarray | None = None
        self._cursor = 0
        #: Total chunks loaded; the report uses it to show the inference rate.
        self.chunks = 0

    @property
    def chunk(self) -> int:
        return self._chunk

    @property
    def execute(self) -> int:
        return self._execute

    @property
    def needs_replan(self) -> bool:
        """Whether the next :meth:`pop` would come up empty."""

        return self._actions is None or self._cursor >= self._queue_length

    @property
    def steps_until_replan(self) -> int:
        if self.needs_replan:
            return 0
        return self._queue_length - self._cursor

    @property
    def _queue_length(self) -> int:
        if self._actions is None:
            return 0
        return int(min(self._execute, self._actions.shape[0]))

    def load(self, actions: np.ndarray) -> int:
        """Queue a chunk and return how many steps it will supply.

        A short chunk is accepted -- a model may legitimately predict fewer steps
        than requested -- but an empty one is not, because silently repeating the
        previous chunk would hide a broken bridge behind plausible motion.
        """

        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise BridgeError(f"expected a non-empty (horizon, dim) chunk, got {values.shape}")
        if values.shape[1] != self.expected_dim:
            raise BridgeError(
                f"chunk is {values.shape[1]}-dimensional but the manifest declares "
                f"{self.expected_dim}"
            )
        self._actions = values
        self._cursor = 0
        self.chunks += 1
        return self._queue_length

    #: Set by the gateway once the action width is known.
    expected_dim: int = 0

    def pop(self) -> np.ndarray | None:
        """Return the next action, or ``None`` when a fresh chunk is needed."""

        if self.needs_replan:
            return None
        assert self._actions is not None  # narrowed by needs_replan
        action = self._actions[self._cursor].copy()
        self._cursor += 1
        return action

    def reset(self) -> None:
        self._actions = None
        self._cursor = 0
        self.chunks = 0

    def metrics(self) -> dict[str, float | int]:
        return {
            "chunk_size": self._chunk,
            "execute_steps": self._execute,
            "chunks_consumed": self.chunks,
        }
