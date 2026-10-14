"""One loaded model, several rollouts at once: the bridge's shared-server mode.

Why this exists: a rollout is a CPU-bound simulation that asks the model for a chunk once
every few control steps, so a model process spends most of its life waiting.  Measured on
this machine, the physics is ~80% of a step and the inference round trip ~5%, with the GPU
busy under a tenth of the time -- so the number of *model processes* is the worst possible
thing to scale with, and GPU memory (about 10 GB each) caps that at three per card however
idle the GPU is.  One model serving many rollouts is the way past that ceiling.

Two things can go wrong with the arrangement, and these tests pin both:

* a second client must not need a second model -- the load happens once, per server;
* a reset from one client must not reach the others -- a shared model cannot apply one
  episode's reset to one client alone, so the shared mode does not forward resets at all,
  and the single-stream mode (which does) is pinned here too so the difference stays a
  decision rather than an accident.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
from a3_dual_arm_sim.bridge.manifest import load_manifest
from a3_dual_arm_sim.contracts import FRONT_IMAGE, STATE
from a3_dual_arm_sim.env import A3DualArmEnv
from a3_dual_arm_sim.runner import EpisodeRunner

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SERVER = ROOT / "scripts" / "a3_bridge_server.py"
#: Enough steps to need more than one chunk from the model, short enough that the test is
#: about the wiring rather than a rollout.
STEPS = 16
CHUNK = 8


#: The adapter records every call in a file, because that is the only way to ask the two
#: questions here -- how many times a model was loaded, and whether a reset reached it.
ADAPTER = '''
"""A bridge adapter that writes down what the server did with it."""

import numpy as np

HOME = np.array([{home}], dtype=np.float64)
LOG = {log!r}


def _note(line):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\\n")


class Model:
    """An object rather than a dict, so `getattr(model, "reset")` finds something."""

    def __init__(self):
        self.action = HOME
        self.calls = 0

    def reset(self):
        _note("reset")


def load(config):
    _note("load")
    return Model()


def infer(model, request):
    model.calls += 1
    _note("infer " + request.episode_id)
    return {{
        "actions": np.tile(model.action, (request.chunk, 1)),
        "space": "joint_position",
        "units": "raw",
        "info": {{"calls": model.calls}},
    }}
'''


@pytest.fixture(scope="module")
def home_action() -> np.ndarray:
    from a3_dual_arm_sim.config import load_config

    config = load_config(None)
    return np.array(
        [*config.home.left, config.home.grippers[0], *config.home.right, config.home.grippers[1]],
        dtype=np.float64,
    )


def _adapter(tmp_path: Path, home_action: np.ndarray) -> tuple[Path, Path]:
    """Write the recording adapter; return it and the file it records into."""

    log = tmp_path / "calls.txt"
    log.write_text("", encoding="utf-8")
    path = tmp_path / "recording_adapter.py"
    path.write_text(
        ADAPTER.format(home=", ".join(repr(float(value)) for value in home_action), log=str(log)),
        encoding="utf-8",
    )
    return path, log


def _socket_manifest(tmp_path: Path, url: str, name: str = "shared.yaml") -> Path:
    payload = {
        "name": "shared-server-test",
        "transport": {
            "kind": "socket",
            "url": url,
            "startup_timeout_s": 120,
            "request_timeout_s": 120,
        },
        "observation": {
            "images": {"front": {"source": FRONT_IMAGE, "resize": [32, 32]}},
            "vectors": {"state": {"source": STATE}},
        },
        "action": {
            "space": "joint_position",
            "dim": 16,
            "chunk": CHUNK,
            "execute": CHUNK,
            "normalize": {"mode": "none"},
        },
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _wait_for_port(port: int, timeout: float = 120.0) -> None:
    """The server loads the model before it listens, so a successful connect means loaded."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket()
        probe.settimeout(0.5)
        try:
            probe.connect(("127.0.0.1", port))
            probe.close()
            return
        except OSError:
            probe.close()
            time.sleep(0.2)
    raise AssertionError(f"the bridge never started listening on port {port}")


def _notes(log: Path) -> list[str]:
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line]


def test_two_rollouts_share_one_loaded_model(tmp_path: Path, home_action: np.ndarray) -> None:
    """Two clients, one load, one model -- and both episodes still run to the end.

    The threads are here for the concurrency, not for speed: what is being checked is that
    the server stays correct with two connections open, which is the arrangement the sweeps
    use.  The rollouts themselves run in separate processes in practice -- this test is about
    the server's side of it.
    """

    adapter, log = _adapter(tmp_path, home_action)
    port = _free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            str(BRIDGE_SERVER),
            "--adapter",
            str(adapter),
            "--listen",
            f"tcp://127.0.0.1:{port}",
        ],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port(port)
        manifest = load_manifest(_socket_manifest(tmp_path, f"tcp://127.0.0.1:{port}"))
        # Built here rather than inside the threads: two MuJoCo models compiled at once is a
        # load-time race this test has no business creating.
        envs = [A3DualArmEnv(render_cameras=False) for _ in range(2)]
        results: list[dict] = []
        failures: list[BaseException] = []

        def drive(env: A3DualArmEnv) -> None:
            try:
                runner = EpisodeRunner(
                    env, ExternalPolicyGateway(manifest, env=env), task="hold the pose"
                )
                try:
                    results.append(dict(runner.run(seed=0, max_steps=STEPS).final_info))
                finally:
                    runner.close()
            except BaseException as exc:  # noqa: BLE001 - reported by the assert below
                failures.append(exc)

        threads = [threading.Thread(target=drive, args=(env,)) for env in envs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)

        assert not failures, f"a shared-server rollout failed: {failures}"
        assert len(results) == 2
        for info in results:
            assert info["bridge"]["transport"] == "socket"
            assert info["bridge"]["manifest"] == "shared-server-test"
        notes = _notes(log)
        assert notes.count("load") == 1, f"the model must be loaded once, got {notes}"
        # Both clients were served by that one model: an episode id per rollout, and at least
        # one inference each.
        episodes = {line.split()[1] for line in notes if line.startswith("infer ")}
        assert len(episodes) == 2, f"both rollouts must reach the model, saw {episodes}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        if server.stdout is not None:
            server.stdout.close()


def test_a_shared_server_does_not_forward_a_reset(tmp_path: Path, home_action: np.ndarray) -> None:
    """The shared mode drops resets; the single-stream mode still forwards them.

    A reset is one client saying "this episode is over".  With one model behind several
    clients there is no way to apply that to the caller alone, and forwarding it would reach
    into another rollout's episode -- silently, mid-run.  So the shared server does not
    forward it, and the single-stream path here proves the reset would otherwise have landed,
    which is what makes the difference a decision rather than a missing feature.
    """

    adapter, log = _adapter(tmp_path, home_action)

    # Single stream first: the same adapter, driven the way every existing manifest drives
    # it, and the reset does reach the model.
    payload = {
        "name": "single-stream-test",
        "transport": {
            "kind": "subprocess",
            "cwd": str(tmp_path),
            "command": [sys.executable, str(BRIDGE_SERVER), "--adapter", str(adapter)],
            "startup_timeout_s": 120,
            "request_timeout_s": 120,
        },
        "observation": {
            "images": {"front": {"source": FRONT_IMAGE, "resize": [32, 32]}},
            "vectors": {"state": {"source": STATE}},
        },
        "action": {
            "space": "joint_position",
            "dim": 16,
            "chunk": CHUNK,
            "execute": CHUNK,
            "normalize": {"mode": "none"},
        },
    }
    manifest_path = tmp_path / "single.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(
        env, ExternalPolicyGateway(load_manifest(manifest_path), env=env), task="hold"
    )
    try:
        runner.run(seed=0, max_steps=STEPS)
    finally:
        runner.close()
    assert _notes(log).count("reset") == 1, "the single-stream mode forwards the episode reset"

    # Then the shared mode, same adapter, same episode -- and the reset stays where it was.
    port = _free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            str(BRIDGE_SERVER),
            "--adapter",
            str(adapter),
            "--listen",
            f"tcp://127.0.0.1:{port}",
        ],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port(port)
        shared_env = A3DualArmEnv(render_cameras=False)
        runner = EpisodeRunner(
            shared_env,
            ExternalPolicyGateway(
                load_manifest(_socket_manifest(tmp_path, f"tcp://127.0.0.1:{port}")),
                env=shared_env,
            ),
            task="hold",
        )
        try:
            result = runner.run(seed=0, max_steps=STEPS)
        finally:
            runner.close()
        assert result.steps == STEPS, "the shared path still runs the episode"
        assert _notes(log).count("reset") == 1, (
            "the shared mode must not let one client's reset reach the model"
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        if server.stdout is not None:
            server.stdout.close()
