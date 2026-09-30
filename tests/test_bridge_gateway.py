"""End-to-end tests: a manifest drives a real rollout.

These are the tests that make Phase 0 answerable.  A replay recording is fed
through the gateway into a live MuJoCo environment, so the observation projection,
the action bridge, the chunk queue, the policy protocol and the runner are all
exercised together with no model installed.

The subprocess case then does the same thing across a real process boundary, with
the shipping bridge server on the other side -- which is the only way to know the
two ends of the wire protocol still agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
from a3_dual_arm_sim.bridge.manifest import load_manifest
from a3_dual_arm_sim.contracts import FRONT_IMAGE, STATE, EpisodeContext
from a3_dual_arm_sim.env import A3DualArmEnv
from a3_dual_arm_sim.runner import EpisodeRunner

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SERVER = ROOT / "scripts" / "a3_bridge_server.py"
#: Steps kept short: these tests prove the wiring, and every extra step costs a
#: MuJoCo rollout rather than adding coverage.
STEPS = 24


@pytest.fixture(scope="module")
def home_action() -> np.ndarray:
    """The environment's own reset pose, as a 16-dimensional action."""

    from a3_dual_arm_sim.config import load_config

    config = load_config(None)
    return np.array(
        [*config.home.left, config.home.grippers[0], *config.home.right, config.home.grippers[1]],
        dtype=np.float64,
    )


def write_replay(tmp_path: Path, actions: np.ndarray, name: str = "actions.json") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {"name": "test recording", "space": "joint_position", "actions": actions.tolist()}
        ),
        encoding="utf-8",
    )
    return path


def write_manifest(tmp_path: Path, payload: dict, name: str = "policy.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def replay_payload(tmp_path: Path, replay_name: str = "actions.json", **action: object) -> dict:
    return {
        "name": "replay-test",
        "transport": {"kind": "replay", "replay_path": replay_name},
        "observation": {
            "images": {
                "front": {"source": FRONT_IMAGE, "resize": [32, 32]},
            },
            "vectors": {"state": {"source": STATE}},
        },
        "action": {"space": "joint_position", "dim": 16, "chunk": 8, "execute": 8, **action},
    }


def test_a_replay_manifest_drives_a_real_episode(tmp_path: Path, home_action: np.ndarray) -> None:
    write_replay(tmp_path, np.tile(home_action, (STEPS, 1)))
    manifest = load_manifest(write_manifest(tmp_path, replay_payload(tmp_path)))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold the pose")

    result = runner.run(seed=0, max_steps=STEPS)
    runner.close()

    assert result.steps == STEPS
    assert not result.terminated
    # The bridge's own numbers reach the episode result, which is what lets a
    # dataset say which model produced a trajectory and what it cost to run.
    assert result.final_info["bridge"]["manifest"] == "replay-test"
    assert result.final_info["bridge"]["transport"] == "replay"
    assert result.final_info["bridge"]["dropped_cameras"] == [
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]


def test_the_chunk_cadence_is_visible_in_the_result(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    write_replay(tmp_path, np.tile(home_action, (STEPS, 1)))
    manifest = load_manifest(write_manifest(tmp_path, replay_payload(tmp_path)))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    result = runner.run(seed=0, max_steps=STEPS)
    runner.close()

    # 24 steps at 8 executed per inference is 3 requests, and the inference rate
    # is the number an operator needs when deciding whether a model can run here.
    assert result.final_info["chunks_consumed"] == 3
    assert result.final_info["bridge"]["requests"] == 3
    assert result.final_info["chunk_size"] == 8


def test_executing_one_step_per_chunk_replans_every_step(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    # A replay recording stands in for a real model: it predicts ``chunk`` rows per
    # request, and the gateway only executes ``execute`` of them.  So the recording
    # is consumed at the prediction rate, not the control rate -- exactly as a real
    # model's output would be discarded.
    requests = 24
    write_replay(tmp_path, np.tile(home_action, (requests * 16, 1)))
    payload = replay_payload(tmp_path, chunk=16, execute=1)
    manifest = load_manifest(write_manifest(tmp_path, payload))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    result = runner.run(seed=0, max_steps=requests)
    runner.close()

    assert result.final_info["chunks_consumed"] == requests
    assert result.final_info["bridge"]["predicted_rows"] == requests * 16
    assert result.final_info["bridge"]["requests"] == requests


def test_the_model_view_reaches_the_environment_intact(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    """A recording shorter than the episode must fail, not silently repeat.

    Repeating the last action would look like a model holding still, which is a
    behaviour, not an error -- so the bridge refuses instead.
    """

    write_replay(tmp_path, np.tile(home_action, (4, 1)))
    manifest = load_manifest(write_manifest(tmp_path, replay_payload(tmp_path)))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    with pytest.raises(RuntimeError, match="ran out"):
        runner.run(seed=0, max_steps=STEPS)
    runner.close()


def test_a_narrower_recording_is_refused_by_the_declared_width(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    write_replay(tmp_path, np.tile(home_action[:14], (STEPS, 1)))
    manifest = load_manifest(write_manifest(tmp_path, replay_payload(tmp_path)))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    with pytest.raises(RuntimeError, match="declares 16"):
        runner.run(seed=0, max_steps=STEPS)
    runner.close()


def test_normalised_actions_are_undone_before_they_reach_the_environment(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    """The declared statistics must be applied, and applied first."""

    stats_path = tmp_path / "stats.npz"
    np.savez(
        stats_path,
        **{"action.mean": np.zeros(16), "action.std": np.full(16, 0.5)},
    )
    write_replay(tmp_path, np.full((STEPS, 16), 0.0))
    payload = replay_payload(tmp_path, normalize={"mode": "mean_std", "stats": "stats.npz"})
    manifest = load_manifest(write_manifest(tmp_path, payload))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    result = runner.run(seed=0, max_steps=4)
    runner.close()

    assert result.steps == 4
    assert result.final_info["bridge"]["requests"] == 1


def test_reset_gives_each_episode_a_fresh_identity(tmp_path: Path, home_action: np.ndarray) -> None:
    write_replay(tmp_path, np.tile(home_action, (STEPS, 1)))
    manifest = load_manifest(write_manifest(tmp_path, replay_payload(tmp_path)))
    env = A3DualArmEnv(render_cameras=False)
    gateway = ExternalPolicyGateway(manifest, env=env)

    gateway.reset(EpisodeContext(seed=0, task="hold", action_mode="joint_position"))
    first = gateway.act(env.reset(seed=0)[0], "hold")
    first_id = gateway._episode_id
    gateway.reset(EpisodeContext(seed=1, task="hold", action_mode="joint_position"))
    second_id = gateway._episode_id

    assert first.shape == (16,)
    assert first_id != second_id
    gateway.close()
    env.close()


def test_a_success_verdict_is_only_overridden_when_the_manifest_asks(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    write_replay(tmp_path, np.tile(home_action, (STEPS, 1)))
    env_manifest = load_manifest(write_manifest(tmp_path, replay_payload(tmp_path)))
    policy_manifest = load_manifest(
        write_manifest(tmp_path, {**replay_payload(tmp_path), "success": {"probe": "policy"}})
    )

    # The default keeps every model on the environment's ruler.
    assert ExternalPolicyGateway(env_manifest).success_override is None
    # A model with its own criterion reports it, but only because it was asked to.
    assert ExternalPolicyGateway(policy_manifest).success_override is None


# ---------------------------------------------------------------------------
# across a real process boundary
# ---------------------------------------------------------------------------

TOY_ADAPTER = '''
"""A bridge adapter with no model at all: returns a fixed pose."""

import numpy as np

HOME = np.array([{home}], dtype=np.float64)


def load(config):
    return {{"action": HOME, "calls": 0}}


def infer(model, request):
    model["calls"] += 1
    assert request.vector("state").shape == (16,), "state did not survive the wire"
    assert request.image("front").dtype == np.uint8, "images must arrive as uint8"
    return {{
        "actions": np.tile(model["action"], (request.chunk, 1)),
        "space": "joint_position",
        "units": "raw",
        "info": {{"toy_calls": model["calls"], "saw_task": request.task}},
    }}
'''


#: An adapter that behaves like most real ones: it prints while it loads.
#: Cosmos Policy announces "Using ALOHA constants: ..." on stdout before it has
#: touched a weight, which is exactly what this reproduces.  The protocol must not
#: share that stream -- if it does, the reader takes four characters of prose for a
#: frame length and waits for about a gigabyte that never comes.
CHATTY_ADAPTER = '''
"""An adapter that prints to stdout, the way real models do."""

import sys

import numpy as np

HOME = np.array([{home}], dtype=np.float64)

print("Using SOME constants:")
print("  ACTION_DIM = 14")
print("  PROPRIO_DIM = 14")
sys.stdout.flush()


def load(config):
    print("loading a model that is going to say things")
    for index in range(50):
        print(f"[INFO] pipeline stage {{index}} ready")
    sys.stdout.flush()
    return {{"action": HOME, "calls": 0}}


def infer(model, request):
    model["calls"] += 1
    print(f"[INFO] inference {{model['calls']}} starting")
    sys.stdout.flush()
    return {{
        "actions": np.tile(model["action"], (request.chunk, 1)),
        "space": "joint_position",
        "units": "raw",
        "info": {{"chatty_calls": model["calls"]}},
    }}
'''


def _run_adapter_over_subprocess(
    tmp_path: Path,
    home_action: np.ndarray,
    source: str,
    name: str,
    steps: int = STEPS,
) -> dict:
    """Drive one adapter through a real subprocess and return the episode info."""

    adapter = tmp_path / f"{name}_adapter.py"
    adapter.write_text(
        source.format(home=", ".join(repr(float(value)) for value in home_action)),
        encoding="utf-8",
    )
    payload = replay_payload(tmp_path)
    payload["transport"] = {
        "kind": "subprocess",
        "cwd": str(tmp_path),
        "command": [sys.executable, str(BRIDGE_SERVER), "--adapter", str(adapter)],
        "startup_timeout_s": 60,
        "request_timeout_s": 60,
    }
    manifest = load_manifest(write_manifest(tmp_path, payload, name=f"{name}.yaml"))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold the pose")
    try:
        result = runner.run(seed=0, max_steps=steps)
    finally:
        runner.close()
    return dict(result.final_info)


def test_a_chatty_model_cannot_corrupt_the_protocol(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    """stdout belongs to the model, not to the bridge.

    Regression test for a deadlock rather than a wrong answer: with the protocol on
    stdout, a model that prints pushed its log text into the frame stream.  The A3
    side read four bytes of a log line as a length prefix -- around 1.4 GB -- and
    blocked forever while the model waited for a request that could never be
    parsed.  Both processes sat idle, the GPU was loaded, and nothing was raised.
    """

    info = _run_adapter_over_subprocess(tmp_path, home_action, CHATTY_ADAPTER, "chatty")

    assert info["chatty_calls"] >= 1, "the chatty model never answered"
    assert info["chunks_consumed"] == 3
    assert info["bridge"]["transport"] == "subprocess"


def test_the_child_log_receives_what_the_model_prints(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    """The model's output must be readable afterwards, not discarded.

    Sending stdout to the log is what keeps it out of the protocol; this asserts the
    log actually contains it, so a silent model and a broken redirect stay
    distinguishable.
    """

    info = _run_adapter_over_subprocess(tmp_path, home_action, CHATTY_ADAPTER, "chatty_log")

    log_path = Path(info["bridge"]["stderr_log"])
    assert log_path.is_file(), f"no log was written at {log_path}"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    assert "pipeline stage" in text
    assert "inference" in text


def test_a_subprocess_bridge_speaks_the_same_protocol(
    tmp_path: Path, home_action: np.ndarray
) -> None:
    info = _run_adapter_over_subprocess(tmp_path, home_action, TOY_ADAPTER, "toy")

    assert info["bridge"]["transport"] == "subprocess"
    assert info["chunks_consumed"] == 3
    # The adapter's own numbers crossed the boundary and reached the episode info.
    assert info["toy_calls"] == 3
    assert info["saw_task"] == "hold the pose"
    assert info["bridge"]["startup_s"] >= 0.0
    assert info["bridge"]["request_mean_ms"] > 0.0


def test_a_bridge_that_cannot_start_reports_its_log(tmp_path: Path) -> None:
    payload = replay_payload(tmp_path)
    payload["transport"] = {
        "kind": "subprocess",
        "cwd": str(tmp_path),
        "command": [sys.executable, "-c", "raise SystemExit('no model here')"],
        "startup_timeout_s": 30,
    }
    manifest = load_manifest(write_manifest(tmp_path, payload, name="broken.yaml"))
    env = A3DualArmEnv(render_cameras=False)
    gateway = ExternalPolicyGateway(manifest, env=env)

    with pytest.raises(RuntimeError, match="exited before signalling readiness"):
        gateway.reset(EpisodeContext(seed=0, task="hold", action_mode="joint_position"))
    gateway.close()
    env.close()


def test_a_bridge_that_misses_a_dependency_reports_the_traceback(tmp_path: Path) -> None:
    """The server must survive an adapter that fails to import.

    This is the common real case: the model's own dependencies are installed in
    its environment and the A3 side has never heard of them.  The traceback has to
    come back over the wire, because the alternative is a silent hang.
    """

    adapter = tmp_path / "missing_dep_adapter.py"
    adapter.write_text(
        "import a_package_that_does_not_exist\n\n\ndef load(config):\n    return None\n",
        encoding="utf-8",
    )
    payload = replay_payload(tmp_path)
    payload["transport"] = {
        "kind": "subprocess",
        "cwd": str(tmp_path),
        "command": [sys.executable, str(BRIDGE_SERVER), "--adapter", str(adapter)],
        "startup_timeout_s": 60,
    }
    manifest = load_manifest(write_manifest(tmp_path, payload, name="missing.yaml"))
    env = A3DualArmEnv(render_cameras=False)
    gateway = ExternalPolicyGateway(manifest, env=env)

    with pytest.raises(RuntimeError) as error:
        gateway.reset(EpisodeContext(seed=0, task="hold", action_mode="joint_position"))

    assert "a_package_that_does_not_exist" in str(error.value)
    gateway.close()
    env.close()
