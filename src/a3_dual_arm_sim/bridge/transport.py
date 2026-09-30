"""How a request reaches a model, and how its actions come back.

Four kinds, and the reason there are four rather than one:

* ``in_process`` -- the model shares this interpreter, so the boundary costs
  nothing.  Right for the LeRobot family, whose torch version matches the A3
  environment's.
* ``subprocess`` -- the model runs under another interpreter, usually another
  conda environment.  This is not a convenience: the GPU sizing guide's matrix
  spans torch 2.1.0 to 2.9.1 and Python 3.10 to 3.13, so most of the models on
  the list *cannot* be imported here at all.
* ``socket`` -- the model is already serving somewhere else, possibly on another
  host.  openpi's websocket policy server is the motivating case.
* ``replay`` -- no model at all.  Recorded actions are fed back through the same
  gateway, which is the only way to test the observation projection, the action
  bridge, the chunk queue and the dataset write without installing anything.

Every transport speaks the same protocol, so the gateway above them is unaware of
which one it has.
"""

from __future__ import annotations

import abc
import json
import os
import select
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import TransportSpec
from .protocol import (
    ActionChunk,
    BridgeError,
    Request,
    decode_message,
    encode_request,
    encode_reset,
    read_frame,
    write_frame,
)


@dataclass
class TransportReport:
    """What actually happened, for the episode result and for ``policy check``."""

    kind: str = ""
    detail: str = ""
    startup_s: float = 0.0
    stderr_log: Path | None = None
    requests: int = 0
    request_seconds: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        measured = self.request_seconds
        return {
            "kind": self.kind,
            "detail": self.detail,
            "startup_s": round(self.startup_s, 3),
            "stderr_log": str(self.stderr_log) if self.stderr_log else None,
            "requests": self.requests,
            "request_mean_ms": round(1000.0 * float(np.mean(measured)), 2) if measured else None,
            "request_max_ms": round(1000.0 * float(np.max(measured)), 2) if measured else None,
        }


class Transport(abc.ABC):
    """Send a request, receive actions (or raise)."""

    def __init__(self, spec: TransportSpec) -> None:
        self.spec = spec
        self.report = TransportReport(kind=self.kind)
        self._connected = False

    @property
    @abc.abstractmethod
    def kind(self) -> str: ...

    def connect(self) -> None:
        """Start or reach the model.  Must be idempotent."""

        self._connected = True

    @abc.abstractmethod
    def infer(self, request: Request) -> ActionChunk: ...

    def reset(self, episode_id: str) -> None:
        """Clear per-episode state on the model side."""

    def close(self) -> None:
        self._connected = False

    def _time(self, started: float) -> None:
        self.report.requests += 1
        self.report.request_seconds.append(time.monotonic() - started)


# ---------------------------------------------------------------------------
# in-process
# ---------------------------------------------------------------------------

#: A model loaded in this interpreter: takes a request, returns actions.
InProcessHandler = Callable[[Request], ActionChunk]


class InProcessTransport(Transport):
    """Call a Python handler directly.

    The handler is usually a bridge server module imported for its side effect of
    loading a model once and keeping it warm.
    """

    def __init__(self, spec: TransportSpec, handler: InProcessHandler) -> None:
        super().__init__(spec)
        self._handler = handler
        self.report.detail = getattr(handler, "__qualname__", type(handler).__name__)

    @property
    def kind(self) -> str:
        return "in_process"

    def infer(self, request: Request) -> ActionChunk:
        started = time.monotonic()
        try:
            chunk = self._handler(request)
        except Exception as exc:
            raise BridgeError(f"in-process model failed: {exc}") from exc
        self._time(started)
        return chunk


# ---------------------------------------------------------------------------
# framed transports (subprocess and socket)
# ---------------------------------------------------------------------------


class _FramedTransport(Transport):
    """Shared request/response logic for the two transports that use byte frames."""

    def _write_frame(self, payload: bytes) -> None:
        raise NotImplementedError

    def _read_frame(self) -> bytes:
        raise NotImplementedError

    def _terminate(self) -> None:
        raise NotImplementedError

    def _exchange(self, payload: bytes) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
        started = time.monotonic()
        self._write_frame(payload)
        response = self._read_frame()
        if not response:
            raise BridgeError(
                f"the model closed the connection without answering; see "
                f"{self.report.stderr_log or 'its logs'}"
            )
        kind, fields, arrays = decode_message(response)
        if kind == "error":
            raise BridgeError(
                f"the model reported an error: {fields.get('message', '')}\n"
                f"{fields.get('traceback', '')}".strip()
            )
        if kind != "actions":
            raise BridgeError(f"expected an 'actions' frame, got {kind!r}")
        self._time(started)
        return kind, fields, arrays

    def infer(self, request: Request) -> ActionChunk:
        payload = encode_request(
            episode_id=request.episode_id,
            step=request.step,
            task=request.task,
            arrays=request.arrays,
            dropped_images=request.dropped_images,
            timestamp=request.timestamp,
            # Without this the server would fall back to its default of one
            # action per call and a chunked rollout would silently become a
            # per-step re-plan -- 50x the inference for no benefit.
            chunk=request.chunk,
        )
        _, fields, arrays = self._exchange(payload)
        if "actions" not in arrays:
            raise BridgeError("'actions' frame carries no action array")
        return ActionChunk(
            actions=arrays["actions"],
            space=str(fields.get("space", "joint_position")),
            units=str(fields.get("units", "raw")),
            latency_ms=fields.get("latency_ms"),
            info=dict(fields.get("info") or {}),
        )

    def reset(self, episode_id: str) -> None:
        """Ask the model to clear its state, and wait for its acknowledgement.

        The acknowledgement is not optional.  A reset frame that is written and
        not read leaves one unread reply in the pipe, and the *next* request then
        reads that reply instead of its own -- which looks like a model returning
        the wrong kind of frame rather than a desynchronised stream.
        """

        self._write_frame(encode_reset(episode_id))
        response = self._read_frame()
        if not response:
            raise BridgeError(
                "the model closed the connection while being reset"
                + (f"; see {self.report.stderr_log}" if self.report.stderr_log else "")
            )
        kind, fields, _ = decode_message(response)
        if kind == "error":
            raise BridgeError(
                f"the model failed to reset: {fields.get('message', '')}\n"
                f"{fields.get('traceback', '')}".strip()
            )
        if kind != "ready":
            raise BridgeError(f"expected a 'ready' acknowledgement to a reset, got {kind!r}")

    def _handshake(self, timeout: float) -> None:
        """Wait for the model's ``ready`` frame, failing with the log path."""

        started = time.monotonic()
        deadline = started + timeout
        payload = self._read_frame_with_deadline(deadline)
        if not payload:
            raise BridgeError(
                f"the bridge exited before signalling readiness after "
                f"{time.monotonic() - started:.1f}s"
                + (f"; see {self.report.stderr_log}" if self.report.stderr_log else "")
            )
        kind, fields, _ = decode_message(payload)
        if kind == "error":
            raise BridgeError(
                f"the bridge failed to start: {fields.get('message', '')}\n"
                f"{fields.get('traceback', '')}".strip()
            )
        if kind != "ready":
            raise BridgeError(f"expected a 'ready' frame at startup, got {kind!r}")
        self.report.startup_s = time.monotonic() - started
        self.report.detail = str(fields.get("detail", "") or self.report.detail)

    def _read_frame_with_deadline(self, deadline: float) -> bytes:
        raise NotImplementedError


class SubprocessTransport(_FramedTransport):
    """Run the model in its own process, under its own interpreter.

    The protocol travels on dedicated file descriptors, never on the child's
    stdin or stdout.  That is not tidiness: a model is free to print, and this one
    does -- Cosmos Policy announces "Using ALOHA constants: ..." on stdout before
    it has loaded anything.  Sharing that stream would interleave prose with the
    frame bytes, and the reader would take four characters of a log line for a
    length prefix and wait for roughly a gigabyte that never arrives.  Requests
    therefore go to one extra descriptor and responses come back on another, while
    the child's stdout and stderr both go to the log file.

    ``conda run`` is how a second environment is reached without activating
    anything in the A3 process.  ``--no-capture-output`` matters: without it conda
    buffers the child's output and the handshake never arrives.
    """

    #: Environment variables naming the two protocol descriptors inside the child.
    REQUEST_FD_ENV = "A3_BRIDGE_REQ_FD"
    RESPONSE_FD_ENV = "A3_BRIDGE_RESP_FD"

    def __init__(self, spec: TransportSpec) -> None:
        super().__init__(spec)
        self._process: subprocess.Popen[bytes] | None = None
        self._to_child: Any = None
        self._from_child: Any = None
        self._stderr_file: Any = None

    @property
    def kind(self) -> str:
        return "subprocess"

    def _command(self) -> list[str]:
        runtime = self.spec.runtime
        command = list(self.spec.command)
        if runtime.kind == "conda":
            launcher = shutil.which("conda") or os.environ.get("CONDA_EXE")
            if not launcher:
                raise BridgeError(
                    "transport.runtime.kind is conda but no conda executable was found; "
                    "set CONDA_EXE or use runtime.kind local"
                )
            return [launcher, "run", "--no-capture-output", "-n", str(runtime.env), *command]
        if runtime.kind == "docker":
            return [
                "docker", "run", "--rm", "-i", "--network=host", str(runtime.image), *command,
            ]
        return command

    def connect(self) -> None:
        if self._process is not None:
            return
        command = self._command()
        if not command:
            raise BridgeError("transport.command is empty")
        log_path = Path(tempfile.gettempdir()) / f"a3-bridge-{_slug(self.spec, self.kind)}.log"
        self.report.stderr_log = log_path
        self._stderr_file = log_path.open("wb")

        # Two pipes: A3 to child (requests), child to A3 (responses).
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        environment = {**os.environ, **{str(k): str(v) for k, v in self.spec.env.items()}}
        environment[self.REQUEST_FD_ENV] = str(request_read)
        environment[self.RESPONSE_FD_ENV] = str(response_write)
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.spec.cwd) if self.spec.cwd else None,
                # The child's own output all goes to the log, so nothing a model
                # prints can reach the protocol.
                stdin=subprocess.DEVNULL,
                stdout=self._stderr_file,
                stderr=subprocess.STDOUT,
                env=environment,
                pass_fds=(request_read, response_write),
                bufsize=0,
            )
        except OSError as exc:
            for handle in (request_read, request_write, response_read, response_write):
                os.close(handle)
            raise BridgeError(f"could not start the bridge command {command[0]!r}: {exc}") from exc
        # Keep only this side's ends.  Holding the others open would keep the pipes
        # alive after the child exits, so a dead bridge would look like a slow one
        # instead of a closed stream.
        os.close(request_read)
        os.close(response_write)
        self._to_child = os.fdopen(request_write, "wb", buffering=0)
        self._from_child = os.fdopen(response_read, "rb", buffering=0)

        self.report.detail = " ".join(command)
        try:
            self._handshake(self.spec.startup_timeout_s)
        except BaseException:
            # A failed handshake must not leave an orphan holding the GPU.
            self._terminate()
            raise
        self._connected = True

    def _write_frame(self, payload: bytes) -> None:
        if self._to_child is None:
            raise BridgeError("bridge process is not running")
        try:
            write_frame(self._to_child, payload)
        except (BrokenPipeError, OSError) as exc:
            raise BridgeError(self._died("writing a request", exc)) from exc

    def _read_frame_with_deadline(self, deadline: float) -> bytes:
        if self._from_child is None:
            raise BridgeError("bridge process is not running")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BridgeError(
                "bridge did not answer within the deadline"
                + (f"; see {self.report.stderr_log}" if self.report.stderr_log else "")
            )
        ready, _, _ = select.select([self._from_child], [], [], remaining)
        if not ready:
            raise BridgeError(
                f"bridge did not answer within {self.spec.request_timeout_s:g}s"
                + (f"; see {self.report.stderr_log}" if self.report.stderr_log else "")
            )
        return read_frame(self._from_child)

    def _read_frame(self) -> bytes:
        return self._read_frame_with_deadline(time.monotonic() + self.spec.request_timeout_s)

    def _died(self, doing: str, exc: BaseException) -> str:
        code = self._process.poll() if self._process else None
        log = f"; see {self.report.stderr_log}" if self.report.stderr_log else ""
        if code is not None:
            return f"the bridge exited with code {code} while {doing}{log}"
        return f"the bridge failed while {doing}: {exc}{log}"

    def _terminate(self) -> None:
        if self._process is None:
            return
        # Closing this side first lets the child see end-of-stream and stop.
        for handle in (self._to_child, self._from_child):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        self._to_child = None
        self._from_child = None
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        self._process = None

    def close(self) -> None:
        self._terminate()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        super().close()


class SocketTransport(_FramedTransport):
    """Reach a model that is already serving, possibly on another host."""

    def __init__(self, spec: TransportSpec) -> None:
        super().__init__(spec)
        self._socket: socket.socket | None = None
        self._reader: Any = None
        self._writer: Any = None

    @property
    def kind(self) -> str:
        return "socket"

    def _parse_url(self) -> tuple[int, Any]:
        url = str(self.spec.url or "")
        if url.startswith("unix://"):
            return socket.AF_UNIX, url[len("unix://") :]
        if url.startswith("tcp://"):
            host, _, port = url[len("tcp://") :].partition(":")
            if not port:
                raise BridgeError(f"transport.url needs a port: {url!r}")
            return socket.AF_INET, (host or "127.0.0.1", int(port))
        raise BridgeError(f"transport.url must start with tcp:// or unix://, got {url!r}")

    def connect(self) -> None:
        if self._socket is not None:
            return
        family, address = self._parse_url()
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(self.spec.startup_timeout_s)
        try:
            sock.connect(address)
        except OSError as exc:
            sock.close()
            raise BridgeError(f"could not reach the model at {self.spec.url}: {exc}") from exc
        self._socket = sock
        self._reader = sock.makefile("rb")
        self._writer = sock.makefile("wb")
        self.report.detail = str(self.spec.url)
        self._handshake(self.spec.startup_timeout_s)
        self._connected = True

    def _write_frame(self, payload: bytes) -> None:
        if self._writer is None:
            raise BridgeError("socket transport is not connected")
        try:
            write_frame(self._writer, payload)
        except OSError as exc:
            raise BridgeError(f"lost the connection to {self.spec.url}: {exc}") from exc

    def _read_frame_with_deadline(self, deadline: float) -> bytes:
        if self._socket is None or self._reader is None:
            raise BridgeError("socket transport is not connected")
        remaining = max(0.0, deadline - time.monotonic())
        self._socket.settimeout(remaining if remaining > 0 else 1e-6)
        try:
            return read_frame(self._reader)
        except TimeoutError as exc:
            raise BridgeError(f"the model at {self.spec.url} did not answer in time") from exc
        except OSError as exc:
            raise BridgeError(f"lost the connection to {self.spec.url}: {exc}") from exc

    def _read_frame(self) -> bytes:
        return self._read_frame_with_deadline(time.monotonic() + self.spec.request_timeout_s)

    def _terminate(self) -> None:
        for handle in (self._reader, self._writer):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None
        self._reader = None
        self._writer = None

    def close(self) -> None:
        self._terminate()
        super().close()


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


class ReplayTransport(Transport):
    """Feed recorded actions back through the gateway.

    This exists so the whole path -- projection, normalisation, dimensionality,
    chunking, validation, dataset write -- is testable with no model installed.
    Without it, every one of those steps would only ever be exercised by a
    multi-gigabyte download.
    """

    def __init__(self, spec: TransportSpec) -> None:
        super().__init__(spec)
        if spec.replay_path is None:
            raise BridgeError("replay transport needs transport.replay_path")
        payload = json.loads(Path(spec.replay_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise BridgeError(f"{spec.replay_path} must hold a mapping at the top level")
        raw = payload.get("actions")
        if not raw:
            raise BridgeError(f"{spec.replay_path} carries no 'actions'")
        self._actions = np.asarray(raw, dtype=np.float64)
        if self._actions.ndim != 2:
            raise BridgeError(
                f"replay actions must be a (frames, dim) list, got shape {self._actions.shape}"
            )
        #: Recording metadata is carried into the episode info, so a replayed
        #: rollout can say which model and which manifest produced the actions.
        self._source = str(payload.get("name", Path(spec.replay_path).name))
        self._space = str(payload.get("space", "joint_position"))
        self._cursor = 0

    @property
    def kind(self) -> str:
        return "replay"

    def connect(self) -> None:
        self.report.detail = self._source
        self._connected = True

    def infer(self, request: Request) -> ActionChunk:
        started = time.monotonic()
        chunk_size = max(1, int(request.chunk))
        start = self._cursor
        stop = min(start + chunk_size, self._actions.shape[0])
        if start >= self._actions.shape[0]:
            raise BridgeError(
                f"replay recording ran out after {self._actions.shape[0]} frames; the "
                "episode is longer than the recording"
            )
        self._cursor = stop
        self._time(started)
        return ActionChunk(
            actions=self._actions[start:stop].copy(),
            space=self._space,
            units="raw",
            info={"replay_source": self._source, "replay_frame": start},
        )

    def reset(self, episode_id: str) -> None:
        self._cursor = 0

    def close(self) -> None:
        self._cursor = 0
        super().close()


def _slug(spec: TransportSpec, kind: str) -> str:
    """A filesystem-safe tag for a subprocess log file."""

    stem = spec.cwd.name if spec.cwd else "bridge"
    return f"{stem}-{os.getpid()}"


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_transport(
    spec: TransportSpec,
    *,
    handler: InProcessHandler | None = None,
) -> Transport:
    """Instantiate the transport a manifest asks for."""

    if spec.kind == "in_process":
        if handler is None:
            raise BridgeError(
                "transport.kind is in_process, which needs a handler object; load the model "
                "in this interpreter and pass it in, or use transport.kind subprocess"
            )
        return InProcessTransport(spec, handler)
    if spec.kind == "subprocess":
        return SubprocessTransport(spec)
    if spec.kind == "socket":
        return SocketTransport(spec)
    if spec.kind == "replay":
        return ReplayTransport(spec)
    raise BridgeError(f"unsupported transport kind {spec.kind!r}")
