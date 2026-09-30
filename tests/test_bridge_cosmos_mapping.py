"""Prove the Cosmos Policy ALOHA mapping against A3, without the model.

Cosmos Policy is blocked on this machine by a CUDA-extension build (see
``docs/external-policy-bridge.md``), but the part of the integration that A3 owns
-- the joint mapping between a 14-dimensional ALOHA robot and A3's 16 -- is
testable on its own.  That is what this module does: it builds the *same*
manifest shape the Cosmos manifest declares, feeds it 14-dimensional ALOHA-shaped
actions through the replay transport, and asserts the resulting A3 actions.

If the mapping is wrong, it is wrong here, before any weights are downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
from a3_dual_arm_sim.bridge.manifest import load_manifest, validate_manifest
from a3_dual_arm_sim.contracts import STATE
from a3_dual_arm_sim.env import A3DualArmEnv
from a3_dual_arm_sim.runner import EpisodeRunner

#: A3's layout: seven joints plus a gripper per arm.
A3_LAYOUT = ["L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P", "L_WRIST_R", "L_WRIST_P", "L_WRIST_Y", "L_gripper", "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", "R_WRIST_R", "R_WRIST_P", "R_WRIST_Y", "R_gripper"]

#: ALOHA's layout: six joints plus a gripper per arm.  The seventh wrist joint is
#: absent, which is the whole reason the mapping is not an identity.
DROPPED = (6, 14)
ALOHA_MAP = tuple(index for index in range(len(A3_LAYOUT)) if index not in DROPPED)

COSMOS_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "policies" / "cosmos_policy_aloha.yaml"
)


def test_the_mapping_is_the_one_the_cosmos_manifest_declares() -> None:
    """Guard the constant above against the manifest drifting away from it."""

    manifest = load_manifest(COSMOS_MANIFEST)

    assert tuple(manifest.action.joint_map) == ALOHA_MAP
    proprio = manifest.observation.vectors["proprio"]
    assert tuple(proprio.indices) == ALOHA_MAP
    assert manifest.action.default == "hold"
    assert manifest.action.dim == len(ALOHA_MAP) == 14


def test_the_dropped_joints_are_the_seventh_wrist_joint_of_each_arm() -> None:
    dropped_names = [A3_LAYOUT[index] for index in DROPPED]

    assert dropped_names == ["L_WRIST_Y", "R_WRIST_Y"]


def write_replay(tmp_path: Path, actions: np.ndarray) -> Path:
    path = tmp_path / "aloha_actions.json"
    path.write_text(
        json.dumps({"name": "aloha-shaped replay", "space": "joint_position", "actions": actions.tolist()}),
        encoding="utf-8",
    )
    return path


def cosmos_like_manifest(tmp_path: Path, actions: np.ndarray) -> Path:
    """The Cosmos manifest's observation and action spec, on the replay transport.

    Only the transport differs.  The indices, joint map, hold default and chunk
    settings are copied from the real manifest, so this exercises the same mapping
    the model integration will use.
    """

    path = tmp_path / "cosmos_replay.yaml"
    payload = {
        "name": "cosmos-aloha-mapping",
        "transport": {"kind": "replay", "replay_path": str(write_replay(tmp_path, actions))},
        "observation": {
            "images": {
                "primary_image": {"source": "observation.images.front", "resize": [224, 224]},
            },
            "vectors": {"proprio": {"source": STATE, "indices": list(ALOHA_MAP)}},
        },
        "action": {
            "space": "joint_position",
            "dim": 14,
            "joint_map": list(ALOHA_MAP),
            "default": "hold",
            "chunk": 50,
            "execute": 16,
            "dt": 0.02,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_a_14_dimensional_aloha_chunk_drives_a_16_dimensional_a3_action(
    tmp_path: Path,
) -> None:
    """The core claim: 14 ALOHA values become a valid A3 action, per joint."""

    # Distinct per-dimension values, so a misplaced index is visible in the
    # assertion rather than hidden by repeated numbers.
    chunk = np.linspace(-0.6, 0.6, 14 * 50, dtype=np.float64).reshape(50, 14)
    manifest = load_manifest(cosmos_like_manifest(tmp_path, chunk))
    assert validate_manifest(manifest) == []

    env = A3DualArmEnv(render_cameras=False)
    env.reset(seed=0)
    gateway = ExternalPolicyGateway(manifest, env=env)

    from a3_dual_arm_sim.contracts import EpisodeContext

    gateway.reset(EpisodeContext(seed=0, task="hold", action_mode="joint_position"))
    observation = env.reset(seed=0)[0]
    action = gateway.act(observation, "hold")

    assert action.shape == (16,)
    # The 14 model dimensions land on the 14 A3 joints they name...
    for model_index, native_index in enumerate(ALOHA_MAP):
        assert action[native_index] == pytest.approx(chunk[0, model_index])
    # ...and the two A3 joints ALOHA has no counterpart for keep the pose the arm
    # actually reported, rather than being commanded to zero.
    measured = env.current_joint_action
    for native_index in DROPPED:
        assert action[native_index] == pytest.approx(measured[native_index])
        assert A3_LAYOUT[native_index].endswith("WRIST_Y")

    gateway.close()
    env.close()


def test_the_chunk_cadence_follows_execute_not_the_predicted_length(
    tmp_path: Path,
) -> None:
    """50 predicted, 16 executed: the tail is dropped, so inference is 1 per 16 steps."""

    steps = 48
    # The replay transport stands in for a model: it serves ``chunk`` rows per
    # request, so the recording is consumed at the prediction rate, not the
    # execution rate.  Three requests of 50 rows cover 48 executed steps.
    recording = np.zeros((3 * 50, 14), dtype=np.float64)
    manifest = load_manifest(cosmos_like_manifest(tmp_path, recording))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    result = runner.run(seed=0, max_steps=steps)
    runner.close()

    assert result.steps == steps
    assert result.final_info["chunks_consumed"] == 3
    assert result.final_info["bridge"]["predicted_rows"] == 3 * 50
    assert result.final_info["execute_steps"] == 16
    assert result.final_info["chunk_size"] == 50


def test_a_16_dimensional_chunk_is_refused_rather_than_truncated(
    tmp_path: Path,
) -> None:
    """A model that ignores its own constants must fail, not be silently trimmed.

    Sixteen values would be the signature of LIBERO constants being resolved
    instead of ALOHA -- which is the exact failure the platform check in the
    adapter exists to catch.
    """

    chunk = np.zeros((50, 16), dtype=np.float64)
    manifest = load_manifest(cosmos_like_manifest(tmp_path, chunk))
    env = A3DualArmEnv(render_cameras=False)
    runner = EpisodeRunner(env, ExternalPolicyGateway(manifest, env=env), task="hold")

    with pytest.raises(RuntimeError, match="declares 14"):
        runner.run(seed=0, max_steps=8)
    runner.close()
