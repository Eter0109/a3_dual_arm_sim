"""Manifest parsing and audit tests.

The audit exists to move failures from mid-rollout to configuration time, so the
cases here are the ones that would otherwise cost a model download to discover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from a3_dual_arm_sim.bridge.manifest import (
    ManifestError,
    dropped_cameras,
    load_manifest,
    validate_manifest,
)

MINIMAL = {
    "name": "demo",
    "transport": {"kind": "replay", "replay_path": "actions.json"},
    "observation": {"images": {"front": {"source": "observation.images.front"}}},
    "action": {"space": "joint_position", "dim": 16, "chunk": 4, "execute": 4},
}

#: Where the runnable manifests live, for tests that guard what is shipped.
COSMOS_DIR = Path(__file__).resolve().parents[1] / "configs" / "policies"


def write_manifest(
    tmp_path: Path, payload: dict, name: str = "policy.yaml", *, create_replay: bool = True
) -> Path:
    """Write a manifest, materialising its replay recording unless asked not to.

    A replay transport needs a file on disk to be a valid audit target, and most
    cases here are about some other part of the wiring.
    """

    transport = payload.get("transport") or {}
    recording = transport.get("replay_path")
    if create_replay and transport.get("kind") == "replay" and recording:
        target = tmp_path / recording
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"actions": [[0]]}', encoding="utf-8")
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_a_minimal_manifest_parses(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path, MINIMAL))

    assert manifest.name == "demo"
    assert manifest.transport.kind == "replay"
    assert tuple(manifest.observation.images) == ("front",)
    assert manifest.action.dim == 16


def test_a_missing_manifest_file_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported_as_such(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="YAML"):
        load_manifest(path)


def test_a_manifest_without_a_name_is_refused(tmp_path: Path) -> None:
    payload = {key: value for key, value in MINIMAL.items() if key != "name"}
    with pytest.raises(ManifestError, match="name"):
        load_manifest(write_manifest(tmp_path, payload))


def test_command_must_be_a_list_not_a_shell_line(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "transport": {"kind": "subprocess", "command": "python -m bridge --serve"},
    }
    with pytest.raises(ManifestError, match="must be a list"):
        load_manifest(write_manifest(tmp_path, payload))


def test_relative_paths_resolve_against_the_manifest_directory(tmp_path: Path) -> None:
    nested = tmp_path / "configs"
    nested.mkdir()
    payload = {
        **MINIMAL,
        "transport": {"kind": "replay", "replay_path": "recordings/actions.json"},
    }
    manifest = load_manifest(write_manifest(nested, payload))

    assert manifest.transport.replay_path == (nested / "recordings/actions.json").resolve()


def test_a_relative_cwd_resolves_against_the_manifest_directory(tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    payload = {
        **MINIMAL,
        "transport": {
            "kind": "subprocess",
            "command": ["python", "-m", "bridge"],
            "cwd": "model",
        },
    }
    manifest = load_manifest(write_manifest(tmp_path, payload))

    assert manifest.transport.cwd == (tmp_path / "model").resolve()


def test_an_unknown_action_space_is_refused_with_advice(tmp_path: Path) -> None:
    payload = {**MINIMAL, "action": {**MINIMAL["action"], "space": "eef_pose"}}
    with pytest.raises(ManifestError, match="adapter on its own side"):
        load_manifest(write_manifest(tmp_path, payload))


# -- audit -----------------------------------------------------------------


def test_the_minimal_manifest_audits_clean(tmp_path: Path) -> None:
    assert validate_manifest(load_manifest(write_manifest(tmp_path, MINIMAL))) == []


def test_a_width_mismatch_is_caught_and_explained(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {"space": "joint_position", "dim": 14, "chunk": 1, "execute": 1},
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("14" in problem and "16" in problem for problem in problems)
    assert any("joint_map" in problem for problem in problems)


def test_a_joint_map_cannot_send_two_outputs_to_one_dimension(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {
            "space": "joint_position",
            "dim": 3,
            "joint_map": [0, 1, 1],
            "chunk": 1,
            "execute": 1,
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("repeats a destination" in problem for problem in problems)


def test_a_joint_map_cannot_index_past_the_native_width(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {
            "space": "joint_position",
            "dim": 2,
            "joint_map": [0, 99],
            "chunk": 1,
            "execute": 1,
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("must index the 16-dimensional" in problem for problem in problems)


def test_a_valid_joint_map_rearranges_a_narrower_output(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {
            "space": "joint_position",
            "dim": 2,
            "joint_map": [0, 8],
            "default": 1.0,
            "chunk": 1,
            "execute": 1,
        },
    }
    assert validate_manifest(load_manifest(write_manifest(tmp_path, payload))) == []


def test_execute_longer_than_the_chunk_is_caught(tmp_path: Path) -> None:
    payload = {**MINIMAL, "action": {**MINIMAL["action"], "chunk": 4, "execute": 10}}
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("exceeds" in problem for problem in problems)


def test_normalisation_without_statistics_is_caught(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {**MINIMAL["action"], "normalize": {"mode": "mean_std"}},
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    # Without this check the raw model output would be used as if physical, which
    # is exactly the silent failure the guide's issue list keeps recording.
    assert any("no stats file" in problem for problem in problems)


def test_normalisation_with_a_missing_statistics_file_is_caught(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {
            **MINIMAL["action"],
            "normalize": {"mode": "mean_std", "stats": "absent.npz"},
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("not found" in problem for problem in problems)


def test_an_observation_key_a3_cannot_supply_is_caught(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "observation": {"images": {"front": {"source": "observation.images.cam_high"}}},
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("cam_high" in problem and "cannot supply" in problem for problem in problems)


def test_a_reversed_slice_is_caught(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "observation": {
            "images": MINIMAL["observation"]["images"],
            "vectors": {"state": {"source": "observation.state", "slice": [16, 0]}},
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("reversed" in problem for problem in problems)


def test_slice_and_indices_together_are_refused(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "observation": {
            "images": MINIMAL["observation"]["images"],
            "vectors": {
                "state": {
                    "source": "observation.state",
                    "slice": [0, 14],
                    "indices": [0, 1, 2],
                }
            },
        },
    }
    with pytest.raises(ManifestError, match="alternatives"):
        load_manifest(write_manifest(tmp_path, payload))


def test_descending_indices_are_refused(tmp_path: Path) -> None:
    # The model's joint order is fixed by its checkpoint, so a permutation here
    # would silently reorder the robot rather than fail.
    payload = {
        **MINIMAL,
        "observation": {
            "images": MINIMAL["observation"]["images"],
            "vectors": {"state": {"source": "observation.state", "indices": [3, 1, 2]}},
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("ascending" in problem for problem in problems)


def test_repeated_indices_are_refused(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "observation": {
            "images": MINIMAL["observation"]["images"],
            "vectors": {"state": {"source": "observation.state", "indices": [0, 1, 1]}},
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("repeats an index" in problem for problem in problems)


def test_ascending_indices_audit_clean(tmp_path: Path) -> None:
    # The real ALOHA mapping: drop each arm's seventh wrist joint.
    payload = {
        **MINIMAL,
        "observation": {
            "images": MINIMAL["observation"]["images"],
            "vectors": {
                "state": {
                    "source": "observation.state",
                    "indices": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15],
                }
            },
        },
    }
    assert validate_manifest(load_manifest(write_manifest(tmp_path, payload))) == []


def test_a_hold_default_is_accepted_for_joint_actions(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {
            "space": "joint_position",
            "dim": 2,
            "joint_map": [0, 8],
            "default": "hold",
            "chunk": 1,
            "execute": 1,
        },
    }
    assert validate_manifest(load_manifest(write_manifest(tmp_path, payload))) == []


def test_an_unknown_default_string_is_refused(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {**MINIMAL["action"], "default": "freeze"},
    }
    with pytest.raises(ManifestError, match="'hold'"):
        load_manifest(write_manifest(tmp_path, payload))


def test_a_scalar_affine_is_broadcast_to_every_dimension(tmp_path: Path) -> None:
    # "scale everything by 2" is a real statement, so a single value is allowed and
    # expanded at parse time rather than left to numpy's broadcasting rules.
    payload = {
        **MINIMAL,
        "action": {**MINIMAL["action"], "affine": {"scale": 2.0, "offset": 1.0}},
    }
    manifest = load_manifest(write_manifest(tmp_path, payload))

    assert manifest.action.affine.scale == (2.0,) * 16
    assert manifest.action.affine.offset == (1.0,) * 16


def test_a_per_dimension_affine_is_kept_verbatim(tmp_path: Path) -> None:
    scale = [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, -2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, -2.0]
    offset = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
    payload = {
        **MINIMAL,
        "action": {
            "space": "cartesian_delta",
            "dim": 14,
            "chunk": 16,
            "execute": 16,
            "affine": {"scale": scale, "offset": offset},
        },
    }
    manifest = load_manifest(write_manifest(tmp_path, payload))

    assert manifest.action.affine.scale == tuple(scale)
    assert manifest.action.affine.offset == tuple(offset)
    assert not manifest.action.affine.is_identity


def test_an_affine_of_the_wrong_width_is_refused(tmp_path: Path) -> None:
    # A short list would broadcast against the action row in numpy and rescale the
    # wrong dimensions rather than fail.
    payload = {
        **MINIMAL,
        "action": {**MINIMAL["action"], "affine": {"scale": [1.0, 2.0]}},
    }
    with pytest.raises(ManifestError, match="one value per dimension"):
        load_manifest(write_manifest(tmp_path, payload))


def test_an_affine_on_a_joint_space_is_refused(tmp_path: Path) -> None:
    # A unit conversion is a property of a controller, not of a robot's joints; on a
    # joint space it would read as a calibration and silently rescale angles.
    payload = {
        **MINIMAL,
        "action": {**MINIMAL["action"], "affine": {"scale": 2.0}},
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("joint_position model" in problem for problem in problems)


def test_a_narrow_clip_on_a_cartesian_space_is_refused(tmp_path: Path) -> None:
    # A3 requires cartesian_delta in [-1, 1], so a narrower clip only hides a
    # conversion that has not been done properly.
    payload = {
        **MINIMAL,
        "action": {
            "space": "cartesian_delta",
            "dim": 14,
            "clip": [-0.5, 0.5],
            "chunk": 1,
            "execute": 1,
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("must lie in" in problem for problem in problems)


def test_the_shipped_gr00t_manifest_pairs_arms_by_side() -> None:
    """The arm GR00T observes as "right" must be the one its right-arm actions drive.

    DexMimicGen labels its arms from the viewer's side, so the metadata's
    ``left_arm`` is the one at -y (its position statistics average -0.235); A3
    labels anatomically, so its left arm is the +y one (measured at +0.25 while the
    right arm works between -0.25 and +0.10).  Both robots face +x, so this is a
    naming difference rather than a frame difference and nothing needs mirroring --
    but pairing by *name* would send each arm's commands to the other arm, and every
    number along the way would still look plausible.

    The check is structural rather than a golden map: whichever A3 arm supplies
    ``right_arm_eef_pos`` has to be the arm GR00T's first seven action dimensions
    land on.  Stating it that way means the test still holds if the mapping is
    re-derived, and it fails the moment the two halves disagree.
    """

    manifest = load_manifest(COSMOS_DIR / "gr00t_bimanual.yaml")
    eef = manifest.observation.vectors["right_arm_eef_pos"]
    joint_map = manifest.action.joint_map
    assert joint_map is not None

    # A3's eef_pose is left arm then right arm, so a slice starting below 7 is the
    # left arm and the action block must target the native left-arm indices.
    reads_native_left = eef.slice is not None and eef.slice[0] < 7
    driven = list(joint_map[:7])
    expected = list(range(7)) if reads_native_left else list(range(7, 14))
    assert driven == expected, (
        f"GR00T's right-arm observation reads A3's {'left' if reads_native_left else 'right'} "
        f"arm but its right-arm actions target native dimensions {driven}, which is the "
        f"{'right' if reads_native_left else 'left'} arm. The two halves of the mapping "
        "disagree about which side of the workspace they refer to."
    )

    # The gripper must agree with the arm it belongs to.
    gripper = manifest.observation.vectors["right_gripper"]
    assert gripper.slice is not None
    reads_left_gripper = gripper.slice[0] == 7
    assert reads_left_gripper == reads_native_left, (
        "GR00T's right gripper reads native dimension "
        f"{gripper.slice[0]}, which is a different arm from the one its right eef "
        "position reads"
    )

    # And the left arm must be the complement, so a one-sided edit cannot slip in.
    left_eef = manifest.observation.vectors["left_arm_eef_pos"]
    assert left_eef.slice is not None
    assert (left_eef.slice[0] < 7) != reads_native_left
    assert list(joint_map[7:]) == (
        list(range(7, 14)) if reads_native_left else list(range(7))
    )


def test_hold_is_refused_for_a_non_joint_action_space(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "action": {
            "space": "cartesian_delta",
            "dim": 14,
            "joint_map": [0, 1],
            "default": "hold",
            "chunk": 1,
            "execute": 1,
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("only defined for joint actions" in problem for problem in problems)


def test_a_subprocess_without_a_command_is_caught(tmp_path: Path) -> None:
    payload = {**MINIMAL, "transport": {"kind": "subprocess"}}
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("command is empty" in problem for problem in problems)


def test_a_conda_runtime_without_an_environment_is_caught(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "transport": {
            "kind": "subprocess",
            "command": ["python", "-m", "bridge"],
            "runtime": {"kind": "conda"},
        },
    }
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("environment name" in problem for problem in problems)


def test_a_socket_url_without_a_scheme_is_caught(tmp_path: Path) -> None:
    payload = {**MINIMAL, "transport": {"kind": "socket", "url": "127.0.0.1:8000"}}
    problems = validate_manifest(load_manifest(write_manifest(tmp_path, payload)))

    assert any("tcp://" in problem for problem in problems)


def test_a_missing_replay_recording_is_caught(tmp_path: Path) -> None:
    problems = validate_manifest(
        load_manifest(write_manifest(tmp_path, MINIMAL, create_replay=False))
    )

    assert any("replay_path not found" in problem for problem in problems)


def test_dropped_cameras_are_reported_not_ignored(tmp_path: Path) -> None:
    # A model that sees one camera is not the same as one that sees three, and a
    # comparison has to say so rather than let the difference look like capability.
    manifest = load_manifest(write_manifest(tmp_path, MINIMAL))

    assert set(dropped_cameras(manifest)) == {
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }


def test_a_manifest_using_every_camera_drops_none(tmp_path: Path) -> None:
    payload = {
        **MINIMAL,
        "observation": {
            "images": {
                "front": {"source": "observation.images.front"},
                "left_wrist": {"source": "observation.images.left_wrist"},
                "right_wrist": {"source": "observation.images.right_wrist"},
            }
        },
    }
    manifest = load_manifest(write_manifest(tmp_path, payload))

    assert dropped_cameras(manifest) == ()


def test_a_manifest_with_no_observation_at_all_is_refused(tmp_path: Path) -> None:
    payload = {**MINIMAL, "observation": {}}
    with pytest.raises(ManifestError, match="neither images nor vectors"):
        load_manifest(write_manifest(tmp_path, payload))


def test_statistics_paths_are_recorded_for_the_report(tmp_path: Path) -> None:
    stats = tmp_path / "stats.npz"
    np.savez(stats, **{"action.mean": np.zeros(16), "action.std": np.ones(16)})
    payload = {
        **MINIMAL,
        "action": {
            **MINIMAL["action"],
            "normalize": {"mode": "mean_std", "stats": "stats.npz"},
        },
    }
    manifest = load_manifest(write_manifest(tmp_path, payload))

    assert manifest.action.normalize.stats == stats.resolve()
