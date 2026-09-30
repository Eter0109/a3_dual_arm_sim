#!/usr/bin/env python
"""Bridge server: run an external VLA or world model behind the A3 wire protocol.

Copy this file into the external project's environment.  It imports nothing from
``a3_dual_arm_sim`` and nothing beyond numpy and the standard library, because the
whole point of the boundary is that the model's interpreter is not this one -- the
GPU sizing guide's version matrix spans torch 2.1.0 to 2.9.1 and Python 3.10 to
3.13, and no single environment holds both ends.

Usage, from inside the model's own conda environment::

    python a3_bridge_server.py --adapter adapters/my_model.py --config ckpt=/path/to/weights

The adapter module supplies two functions:

.. code-block:: python

    def load(config: dict) -> object:
        \"\"\"Build the model.  Called once.  ``config`` comes from --config.\"\"\"

    def infer(model: object, request: Request) -> dict:
        \"\"\"Return {\"actions\": ndarray (H, D), \"space\": str, \"units\": str}.\"\"\"

``Request`` exposes ``.task``, ``.step``, ``.episode_id``, ``.chunk``,
``.image(name)`` and ``.vector(name)``.  The adapter is the only file a new model
needs: everything else -- normalisation, dimensionality, chunk pacing -- is
declared in the A3-side manifest and applied there, so every model is measured on
one ruler.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL = "a3-bridge/1"
_MAGIC = b"A3BR"
_U32 = struct.Struct(">I")


class ProtocolError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# framing (kept byte-compatible with a3_dual_arm_sim.bridge.protocol)
# ---------------------------------------------------------------------------


def read_exactly(stream: Any, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            raise EOFError(f"stream ended after {len(buffer)} of {size} bytes")
        buffer += chunk
    return bytes(buffer)


def read_frame(stream: Any) -> bytes:
    head = stream.read(4)
    if not head:
        return b""
    if len(head) < 4:
        raise ProtocolError("stream ended inside a frame length prefix")
    return read_exactly(stream, _U32.unpack(head)[0])


def write_frame(stream: Any, payload: bytes) -> None:
    stream.write(_U32.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def encode_message(kind: str, fields: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bytes:
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
    parts = [_MAGIC, _U32.pack(len(header_bytes)), header_bytes, _U32.pack(len(names))]
    for name in names:
        blob = np.ascontiguousarray(np.asarray(arrays[name])).tobytes()
        parts.append(_U32.pack(len(blob)))
        parts.append(blob)
    return b"".join(parts)


def decode_message(payload: bytes) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        if offset + size > len(payload):
            raise ProtocolError(f"frame truncated: wanted {size} bytes at {offset}")
        chunk = payload[offset : offset + size]
        offset += size
        return chunk

    if take(4) != _MAGIC:
        raise ProtocolError("frame does not start with the a3-bridge magic word")
    header = json.loads(take(_U32.unpack(take(4))[0]).decode("utf-8"))
    if header.get("protocol") != PROTOCOL:
        raise ProtocolError(
            f"protocol mismatch: frame says {header.get('protocol')!r}, this server speaks "
            f"{PROTOCOL!r}"
        )
    declared = header.get("arrays") or {}
    count = _U32.unpack(take(4))[0]
    if count != len(declared):
        raise ProtocolError(f"header declares {len(declared)} arrays, frame carries {count}")
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(declared):
        meta = declared[name]
        dtype = np.dtype(meta["dtype"])
        blob = take(_U32.unpack(take(4))[0])
        arrays[name] = np.frombuffer(blob, dtype=dtype).reshape(meta["shape"]).copy()
    return header["kind"], dict(header.get("fields") or {}), arrays


# ---------------------------------------------------------------------------
# request view handed to adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """What an adapter receives.  Deliberately framework-free."""

    episode_id: str
    step: int
    task: str
    chunk: int
    timestamp: float | None
    arrays: Mapping[str, np.ndarray]

    def image(self, name: str) -> np.ndarray:
        """A uint8 HxWx3 RGB frame."""

        try:
            return self.arrays[name]
        except KeyError:
            raise KeyError(
                f"no image {name!r} in this request; present: {sorted(self.arrays)}"
            ) from None

    def vector(self, name: str) -> np.ndarray:
        """A float vector such as ``state`` or ``velocity``."""

        try:
            return self.arrays[name]
        except KeyError:
            raise KeyError(
                f"no vector {name!r} in this request; present: {sorted(self.arrays)}"
            ) from None


#: The A3 transport names the two protocol descriptors in the environment.  Using
#: descriptors other than stdin/stdout is what makes the bridge safe to run next to
#: a model that prints: those two streams belong to the model's own logging.
REQUEST_FD_ENV = "A3_BRIDGE_REQ_FD"
RESPONSE_FD_ENV = "A3_BRIDGE_RESP_FD"


def protocol_streams() -> tuple[Any, Any, bool]:
    """Return ``(stream_in, stream_out, dedicated)`` for the request loop.

    With the two environment variables set -- always the case under
    ``transport.kind: subprocess`` -- the protocol runs on its own descriptors and
    the model keeps stdout and stderr entirely for itself.

    Without them the server falls back to stdin/stdout so it can be driven by hand,
    and in that mode stdout is *replaced* by stderr before anything else runs: a
    model that prints to stdout would otherwise interleave its text with frame
    bytes, and the reader on the other end would read a log line as a length prefix
    and wait forever.  A fallback cannot be fully safe -- a C library writing
    straight to descriptor 1 bypasses the reassignment -- so the dedicated
    descriptors are the supported path for real runs.
    """

    request_fd = os.environ.get(REQUEST_FD_ENV)
    response_fd = os.environ.get(RESPONSE_FD_ENV)
    if request_fd and response_fd:
        return (
            os.fdopen(int(request_fd), "rb", buffering=0),
            os.fdopen(int(response_fd), "wb", buffering=0),
            True,
        )
    stream_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    return sys.stdin.buffer, stream_out, False


def _adapter(module: Any) -> tuple[Any, Any]:
    load = getattr(module, "load", None)
    infer = getattr(module, "infer", None)
    if load is None or infer is None:
        raise ProtocolError(
            f"adapter {getattr(module, '__file__', '?')} must define both load(config) and "
            "infer(model, request)"
        )
    return load, infer


def _load_adapter(path: Path) -> Any:
    if not path.is_file():
        raise ProtocolError(f"adapter module not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"could not import adapter module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_config(pairs: list[str]) -> dict[str, Any]:
    """``--config key=value`` pairs; a bare value becomes ``"1"`` is rejected."""

    config: dict[str, Any] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise ProtocolError(f"--config expects key=value, got {pair!r}")
        config[key] = value
    return config


def _coerce_chunk(result: Any) -> tuple[np.ndarray, str, str, dict[str, Any]]:
    """Accept an adapter's return value and normalise it to the wire shape.

    An adapter is allowed to return a bare array, which is the common case, and
    the space and units then come from the manifest on the A3 side.
    """

    if isinstance(result, Mapping):
        actions = result.get("actions")
        if actions is None:
            raise ProtocolError("adapter result has no 'actions' entry")
        space = str(result.get("space", "joint_position"))
        units = str(result.get("units", "raw"))
        info = dict(result.get("info") or {})
    else:
        actions = result
        space, units, info = "joint_position", "raw", {}
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ProtocolError(f"adapter returned actions of shape {array.shape}, expected (H, D)")
    return np.ascontiguousarray(array), space, units, info


def serve(
    adapter_path: Path,
    config: dict[str, Any],
    stream_in: Any,
    stream_out: Any,
    dedicated: bool = True,
) -> int:
    """Run the request loop until the A3 side closes the stream."""

    try:
        module = _load_adapter(adapter_path)
        load, infer = _adapter(module)
        model = load(config)
    except BaseException as exc:  # noqa: BLE001 - reported over the wire, then exit
        write_frame(
            stream_out,
            encode_message(
                "error",
                {"message": f"adapter failed to load: {exc}", "traceback": traceback.format_exc()},
                {},
            ),
        )
        return 1

    write_frame(
        stream_out,
        encode_message(
            "ready",
            {
                "side": "model",
                "detail": (
                    f"{adapter_path.name} protocol={PROTOCOL} "
                    f"python={sys.version.split()[0]} fds={'dedicated' if dedicated else 'stdio'}"
                ),
            },
            {},
        ),
    )

    while True:
        try:
            payload = read_frame(stream_in)
        except EOFError:
            return 0
        if not payload:
            return 0
        try:
            kind, fields, arrays = decode_message(payload)
            if kind == "reset":
                reset = getattr(model, "reset", None)
                if callable(reset):
                    reset()
                write_frame(
                    stream_out,
                    encode_message("ready", {"side": "model", "detail": "reset"}, {}),
                )
                continue
            if kind != "infer":
                raise ProtocolError(f"expected an 'infer' frame, got {kind!r}")
            request = Request(
                episode_id=str(fields.get("episode_id", "")),
                step=int(fields.get("step", 0)),
                task=str(fields.get("task", "")),
                chunk=int(fields.get("chunk", 1)),
                timestamp=fields.get("timestamp"),
                arrays=arrays,
            )
            actions, space, units, info = _coerce_chunk(infer(model, request))
            write_frame(
                stream_out,
                encode_message(
                    "actions",
                    {"space": space, "units": units, "info": info},
                    {"actions": actions},
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - one bad request must not kill the server
            write_frame(
                stream_out,
                encode_message(
                    "error",
                    {"message": str(exc), "traceback": traceback.format_exc()},
                    {},
                ),
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="a3_bridge_server", description="Serve an external model over the A3 wire protocol"
    )
    parser.add_argument("--adapter", type=Path, required=True, help="Path to an adapter module")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Adapter configuration; repeatable",
    )
    args = parser.parse_args(argv)
    try:
        config = _parse_config(args.config)
    except ProtocolError as exc:
        parser.error(str(exc))
        return 2
    stream_in, stream_out, dedicated = protocol_streams()
    return serve(args.adapter, config, stream_in, stream_out, dedicated)


if __name__ == "__main__":
    raise SystemExit(main())
