"""Wire protocol for the external-policy bridge.

The bridge boundary carries numpy arrays and plain scalars and nothing else, so
the A3 process never imports an external model's framework.  That is what lets a
model run in its own conda environment: the version matrix in the GPU sizing
guide spans torch 2.1.0 to 2.9.1 and Python 3.10 to 3.13, which no single
interpreter can hold.

Framing is deliberately boring -- a magic word, a JSON header, then
length-prefixed blobs -- so a conforming bridge needs nothing beyond numpy, and a
captured stream stays readable in a hex viewer when something goes wrong.

See ``docs/external-policy-bridge.md`` for why the boundary sits here.
"""

from __future__ import annotations

import json
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Version marker carried in every frame; a mismatch is a hard failure.
PROTOCOL = "a3-bridge/1"

_MAGIC = b"A3BR"
_U32 = struct.Struct(">I")

#: dtypes allowed across the boundary.  Restricted on purpose: a bridge that
#: reinterpreted bytes as another width would corrupt actions without raising
#: anything, and that is the failure mode hardest to notice from the A3 side.
ALLOWED_DTYPES = frozenset({"uint8", "int32", "int64", "float32", "float64"})

#: Action spaces the A3 side can translate into its own contract.
ACTION_SPACES = ("joint_position", "cartesian_delta")

#: How a model's raw output is turned back into physical units.
NORMALIZE_MODES = ("none", "mean_std", "min_max", "q01_q99")


class BridgeError(RuntimeError):
    """A framing, handshake, or transport failure."""


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def _transferable(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous little-endian view, which is what the wire assumes."""

    value = np.ascontiguousarray(array)
    if value.dtype.byteorder == ">" or (
        value.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        value = value.astype(value.dtype.newbyteorder("<"))
    return value


def encode_message(
    kind: str,
    fields: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> bytes:
    """Serialise one frame.

    ``arrays`` are carried in sorted key order, so the header alone is enough to
    rebuild them.  Keys are the model-visible names a manifest declares.
    """

    names = tuple(sorted(arrays))
    header = {
        "protocol": PROTOCOL,
        "kind": kind,
        "fields": dict(fields),
        "arrays": {
            name: {
                "dtype": str(np.asarray(arrays[name]).dtype),
                "shape": list(np.asarray(arrays[name]).shape),
            }
            for name in names
        },
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    parts = [
        _MAGIC,
        _U32.pack(len(header_bytes)),
        header_bytes,
        _U32.pack(len(names)),
    ]
    for name in names:
        blob = _transferable(np.asarray(arrays[name])).tobytes()
        parts.append(_U32.pack(len(blob)))
        parts.append(blob)
    return b"".join(parts)


def decode_message(payload: bytes) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """Rebuild ``(kind, fields, arrays)`` from a frame."""

    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        if offset + size > len(payload):
            raise BridgeError(
                f"frame truncated: wanted {size} bytes at offset {offset}, "
                f"only {len(payload) - offset} left"
            )
        chunk = payload[offset : offset + size]
        offset += size
        return chunk

    if take(4) != _MAGIC:
        raise BridgeError("frame does not start with the a3-bridge magic word")
    header = json.loads(take(_U32.unpack(take(4))[0]).decode("utf-8"))
    if header.get("protocol") != PROTOCOL:
        raise BridgeError(
            f"protocol mismatch: frame says {header.get('protocol')!r}, "
            f"this build speaks {PROTOCOL!r}"
        )
    declared: Mapping[str, Any] = header.get("arrays") or {}
    count = _U32.unpack(take(4))[0]
    if count != len(declared):
        raise BridgeError(f"header declares {len(declared)} arrays but the frame carries {count}")

    arrays: dict[str, np.ndarray] = {}
    for name in sorted(declared):
        meta = declared[name]
        dtype = np.dtype(meta["dtype"])
        if dtype.name not in ALLOWED_DTYPES:
            raise BridgeError(f"array {name!r} uses a disallowed dtype {dtype.name!r}")
        blob = take(_U32.unpack(take(4))[0])
        expected = int(np.prod(meta["shape"], dtype=np.int64)) * dtype.itemsize
        if len(blob) != expected:
            raise BridgeError(
                f"array {name!r} declares shape {meta['shape']} of {dtype.name} "
                f"({expected} bytes) but carries {len(blob)}"
            )
        arrays[name] = np.frombuffer(blob, dtype=dtype).reshape(meta["shape"]).copy()
    return header["kind"], dict(header.get("fields") or {}), arrays


# ---------------------------------------------------------------------------
# stream framing (subprocess pipes and sockets)
# ---------------------------------------------------------------------------


def write_frame(stream: Any, payload: bytes) -> None:
    """Write one length-prefixed frame and flush it."""

    stream.write(_U32.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_exactly(stream: Any, size: int) -> bytes:
    """Read exactly ``size`` bytes, raising ``EOFError`` on a short or clean end."""

    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            raise EOFError(f"bridge stream ended after {len(buffer)} of {size} bytes")
        buffer += chunk
    return bytes(buffer)


def read_frame(stream: Any) -> bytes:
    """Read one length-prefixed frame, returning ``b""`` at a clean end of stream."""

    head = stream.read(4)
    if not head:
        return b""
    if len(head) < 4:
        raise BridgeError("bridge stream ended inside a frame length prefix")
    return read_exactly(stream, _U32.unpack(head)[0])


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionChunk:
    """What a model returns: a block of future actions plus its own description.

    ``units`` and ``space`` are mandatory because scale errors are the quietest
    failure in this whole design -- the GPU sizing guide's issue list records
    several (VITRA's normalisation statistics, Fast-WAM's dataset stats) where a
    model runs happily and simply acts wrongly.
    """

    actions: np.ndarray
    #: One of :data:`ACTION_SPACES`.
    space: str = "joint_position"
    #: ``"raw"`` when the values are already physical, otherwise the mode that
    #: produced them so the A3 side can apply the inverse.
    units: str = "raw"
    latency_ms: float | None = None
    #: Model-reported numbers merged into the episode info mapping.
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.actions, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise BridgeError(f"action chunk must be 2-D (horizon, dim), got {array.shape}")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise BridgeError(f"action chunk is empty: {array.shape}")
        if not np.all(np.isfinite(array)):
            raise BridgeError("action chunk contains NaN or infinity")
        object.__setattr__(self, "actions", np.ascontiguousarray(array))

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def dim(self) -> int:
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class Request:
    """A decoded inference request, as a bridge server sees it."""

    episode_id: str
    step: int
    task: str
    arrays: Mapping[str, np.ndarray]
    timestamp: float | None
    dropped_images: tuple[str, ...]
    #: How many future actions the caller wants.  Sent explicitly because a
    #: bridge that guesses would return one action per call and quietly turn a
    #: chunked rollout into a per-step re-plan.
    chunk: int = 1

    def image(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def vector(self, name: str) -> np.ndarray:
        return self.arrays[name]


def encode_request(
    *,
    episode_id: str,
    step: int,
    task: str,
    arrays: Mapping[str, np.ndarray],
    dropped_images: tuple[str, ...] = (),
    timestamp: float | None = None,
    chunk: int = 1,
) -> bytes:
    """Build an ``infer`` frame.  Only the A3 side calls this."""

    fields = {
        "episode_id": episode_id,
        "step": int(step),
        "task": task,
        "dropped_images": list(dropped_images),
        "timestamp": timestamp,
        "chunk": int(chunk),
    }
    return encode_message("infer", fields, arrays)


def decode_request(payload: bytes) -> Request:
    """Read an ``infer`` frame.  Only a bridge server calls this."""

    kind, fields, arrays = decode_message(payload)
    if kind != "infer":
        raise BridgeError(f"expected an 'infer' frame, got {kind!r}")
    return Request(
        episode_id=str(fields.get("episode_id", "")),
        step=int(fields.get("step", 0)),
        task=str(fields.get("task", "")),
        arrays=arrays,
        timestamp=fields.get("timestamp"),
        dropped_images=tuple(fields.get("dropped_images") or ()),
        chunk=int(fields.get("chunk", 1)),
    )


def encode_response(chunk: ActionChunk) -> bytes:
    """Build an ``actions`` frame.  Only a bridge server calls this."""

    fields = {
        "space": chunk.space,
        "units": chunk.units,
        "latency_ms": chunk.latency_ms,
        "info": dict(chunk.info),
    }
    return encode_message("actions", fields, {"actions": chunk.actions})


def decode_response(payload: bytes) -> ActionChunk:
    """Read an ``actions`` frame.  Only the A3 side calls this."""

    kind, fields, arrays = decode_message(payload)
    if kind != "actions":
        raise BridgeError(f"expected an 'actions' frame, got {kind!r}")
    if "actions" not in arrays:
        raise BridgeError("'actions' frame carries no action array")
    return ActionChunk(
        actions=arrays["actions"],
        space=str(fields.get("space", "joint_position")),
        units=str(fields.get("units", "raw")),
        latency_ms=fields.get("latency_ms"),
        info=dict(fields.get("info") or {}),
    )


def encode_ready(side: str = "model", detail: str = "") -> bytes:
    """Handshake frame a bridge server sends once its model is loaded."""

    return encode_message("ready", {"side": side, "detail": detail}, {})


def encode_error(message: str, *, traceback_text: str = "") -> bytes:
    """Handshake or per-request failure a bridge server reports to the A3 side."""

    return encode_message("error", {"message": message, "traceback": traceback_text}, {})


def encode_reset(episode_id: str) -> bytes:
    """Tell a bridge server to clear per-episode state."""

    return encode_message("reset", {"episode_id": episode_id}, {})
