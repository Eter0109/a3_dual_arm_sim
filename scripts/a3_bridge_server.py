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
import hashlib
import importlib.util
import json
import os
import socket
import struct
import sys
import threading
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


def load_inference(adapter_path: Path, config: dict[str, Any]) -> tuple[Any, Any]:
    """Load an adapter module and its model once, returning ``(model, infer)``.

    Split out of `serve` so the listening server can load one model and hand it to many
    connections, which is the whole point of that mode.
    """

    module = _load_adapter(adapter_path)
    load, infer = _adapter(module)
    return load(config), infer


#: Print a hash of every inference's input and output.  Off by default: it is a diagnostic,
#: and the log has one line per chunk rather than one per episode.
_LOG_IO_SHA = os.environ.get("A3_LOG_IO_SHA", "") not in ("", "0")


def serve_stream(
    model: Any,
    infer: Any,
    stream_in: Any,
    stream_out: Any,
    *,
    lock: "threading.Lock | None" = None,
    ready_detail: str = "",
    allow_reset: bool = True,
) -> int:
    """Answer requests on one pair of streams until the far end closes them.

    ``lock`` serialises the calls into a model that several connections share: the adapter
    interface is a plain function over a loaded object and nothing about it promises to be
    re-entrant, so the server holds a lock rather than assuming.

    ``allow_reset`` is off for a shared server.  A reset is a per-episode instruction from
    one client, and a shared model has no way to apply it to that client alone -- forwarding
    it would silently discard another rollout's state mid-episode, which is the kind of
    quiet wrong answer this bridge exists to prevent.  An adapter whose model carries
    state that must be reset per episode therefore needs one process per rollout, and its
    manifest should use `subprocess` rather than a shared socket.
    """

    write_frame(
        stream_out,
        encode_message("ready", {"side": "model", "detail": ready_detail}, {}),
    )

    while True:
        try:
            payload = read_frame(stream_in)
        except EOFError:
            return 0
        except OSError:
            return 0
        if not payload:
            return 0
        try:
            kind, fields, arrays = decode_message(payload)
            if kind == "reset":
                if allow_reset:
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
            if _LOG_IO_SHA:
                # The one place both stacks pass through.  A hash of exactly what the model is
                # handed separates "the upstream observations differ" from "the model answers
                # differently to the same bytes" -- which no rollout can tell apart.
                digest = hashlib.sha256()
                parts = []
                for name in sorted(request.arrays):
                    block = np.ascontiguousarray(request.arrays[name])
                    digest.update(name.encode())
                    digest.update(block.tobytes())
                    parts.append(
                        f"{name}={block.shape}/{block.dtype}/"
                        f"{float(np.asarray(block, dtype=np.float64).mean()):.6f}/"
                        f"{hashlib.sha256(block.tobytes()).hexdigest()[:8]}")
                digest.update(request.task.encode())
                print(f"[in-sha] step={request.step} {digest.hexdigest()[:16]} "
                      + " ".join(parts), file=sys.stderr, flush=True)
            if lock is None:
                actions, space, units, info = _coerce_chunk(infer(model, request))
            else:
                with lock:
                    actions, space, units, info = _coerce_chunk(infer(model, request))
            if _LOG_IO_SHA:
                out = hashlib.sha256(
                    np.ascontiguousarray(np.asarray(actions)).tobytes()
                ).hexdigest()[:16]
                print(f"[out-sha] step={request.step} {out}", file=sys.stderr, flush=True)
            write_frame(
                stream_out,
                encode_message(
                    "actions",
                    {"space": space, "units": units, "info": info},
                    {"actions": actions},
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - one bad request must not kill the server
            try:
                write_frame(
                    stream_out,
                    encode_message(
                        "error",
                        {"message": str(exc), "traceback": traceback.format_exc()},
                        {},
                    ),
                )
            except OSError:
                # The client went away while its own error was being reported.  Nothing left
                # to report it to, and this is not the server's failure.
                return 0
    return 0


def serve(
    adapter_path: Path,
    config: dict[str, Any],
    stream_in: Any,
    stream_out: Any,
    dedicated: bool = True,
) -> int:
    """Run the request loop until the A3 side closes the stream."""

    try:
        model, infer = load_inference(adapter_path, config)
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

    return serve_stream(
        model,
        infer,
        stream_in,
        stream_out,
        ready_detail=(
            f"{adapter_path.name} protocol={PROTOCOL} "
            f"python={sys.version.split()[0]} fds={'dedicated' if dedicated else 'stdio'}"
        ),
    )


def _parse_listen_url(url: str) -> tuple[int, Any]:
    """``unix:///path`` or ``tcp://host:port`` to a socket family and address."""

    if url.startswith("unix://"):
        return socket.AF_UNIX, url[len("unix://") :]
    if url.startswith("tcp://"):
        host, _, port = url[len("tcp://") :].partition(":")
        if not port:
            raise ProtocolError(f"--listen needs a port: {url!r}")
        return socket.AF_INET, (host or "127.0.0.1", int(port))
    raise ProtocolError(f"--listen must start with tcp:// or unix://, got {url!r}")


def _serve_connection(
    connection: Any,
    model: Any,
    infer: Any,
    lock: "threading.Lock",
    adapter_name: str,
) -> None:
    """One client, on its own thread, over the same loaded model."""

    with connection:
        reader = connection.makefile("rb")
        writer = connection.makefile("wb")
        try:
            serve_stream(
                model,
                infer,
                reader,
                writer,
                lock=lock,
                ready_detail=f"{adapter_name} protocol={PROTOCOL} shared",
                allow_reset=False,
            )
        except (OSError, EOFError):
            # A client that vanished is a client that is done; the server keeps serving.
            pass
        finally:
            for handle in (writer, reader):
                try:
                    handle.close()
                except OSError:
                    pass


def serve_listening(
    url: str,
    adapter_path: Path,
    config: dict[str, Any],
    *,
    backlog: int = 32,
) -> int:
    """Serve one loaded model to many rollout processes at once.

    Why this exists: a rollout is a CPU-bound simulation that asks the model for a chunk
    once every few control steps, so a model process spends most of its life waiting.  Measured
    on this machine: the physics is ~80% of a step and the inference round trip ~5%, with the
    GPU busy under a tenth of the time.  Running eight rollouts against one loaded model is
    therefore eight times the throughput for one card's worth of weights -- which is the
    difference between two parallel seeds and eight, because the alternative (one model process
    per rollout) is capped by GPU memory at about three per card however idle the GPU is.

    Each connection gets its own thread; the calls into the model are serialised by one lock,
    because ``infer(model, request)`` is a plain function over a shared object and nothing
    about the adapter contract promises otherwise.
    """

    family, address = _parse_listen_url(url)
    try:
        model, infer = load_inference(adapter_path, config)
    except BaseException as exc:  # noqa: BLE001 - nothing to report over the wire yet
        print(f"adapter failed to load: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return 1

    server_socket = socket.socket(family, socket.SOCK_STREAM)
    if family == socket.AF_UNIX:
        # A crashed server leaves its socket file behind and bind() then fails with
        # "address already in use", which reads like a live server on the other end.
        try:
            os.unlink(address)
        except FileNotFoundError:
            pass
    try:
        server_socket.bind(address)
        server_socket.listen(backlog)
    except OSError as exc:
        print(f"could not listen on {url}: {exc}", file=sys.stderr)
        return 1

    lock = threading.Lock()
    print(f"listening on {url} (model loaded once)", file=sys.stderr, flush=True)
    try:
        while True:
            try:
                connection, _ = server_socket.accept()
            except OSError as exc:
                print(f"accept failed: {exc}", file=sys.stderr, flush=True)
                return 1
            threading.Thread(
                target=_serve_connection,
                args=(connection, model, infer, lock, adapter_path.name),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        return 0
    finally:
        server_socket.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="a3_bridge_server", description="Serve an external model over the A3 wire protocol"
    )
    parser.add_argument("--adapter", type=Path, required=True, help="Path to an adapter module")
    parser.add_argument(
        "--listen",
        default=None,
        metavar="URL",
        help=(
            "Serve on unix:///path or tcp://host:port, loading the model once for every "
            "client that connects, instead of the single-stream mode the A3 side drives "
            "through a pair of file descriptors.  One loaded model then backs many rollouts "
            "at once, which is the only way past the GPU-memory ceiling on how many model "
            "processes fit on a card.  Resets are not forwarded in this mode: a shared model "
            "cannot apply one client's per-episode reset to that client alone.  Manifests "
            "reach it with transport.kind: socket."
        ),
    )
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
    if args.listen:
        return serve_listening(args.listen, args.adapter, config)
    stream_in, stream_out, dedicated = protocol_streams()
    return serve(args.adapter, config, stream_in, stream_out, dedicated)


if __name__ == "__main__":
    raise SystemExit(main())
