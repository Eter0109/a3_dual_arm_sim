"""The dataset schema must not drift between scenes.

Collection is now scene-independent -- one driver, one registry -- so the only way
two datasets can disagree is through the recorder's schema, which is built from the
environment's config (`image_height`, `image_width`) and the shared observation
contract.  Every cookie config happens to leave the image size at its 256 default,
which is why the datasets merge today; nothing was enforcing it.

That matters because a mismatch is not caught where it is created.  LeRobot writes
whatever shapes it is given, the merge step concatenates, and the trainer is the
first thing to notice -- reporting it as a feature-shape error against a hard-coded
256, long after the episodes were collected.  These checks move the failure to the
scene that would cause it.
"""

from __future__ import annotations

import json

import pytest

from a3_dual_arm_sim.collection import registered_scenes
from a3_dual_arm_sim.contracts import FORCE, STATE, VELOCITY
from a3_dual_arm_sim.recording import ACTION, lerobot_features

# What the trainer's audit insists on, restated here so a change to either side
# fails in one place rather than silently passing both.
TRAINED_IMAGE_SIZE = 256


def _schema(scene_name: str) -> str:
    """The recorder's feature declaration for one scene, as comparable JSON."""

    scene = registered_scenes()[scene_name]()
    env = scene.build_env(0.0, 0.0)
    try:
        features = lerobot_features(
            env.config.image_height,
            env.config.image_width,
            use_videos=True,
        )
    finally:
        env.close()
    return json.dumps(features, sort_keys=True, default=str)


def test_every_scene_declares_the_same_features():
    """One schema across all scenes, which is what makes their data mergeable."""

    schemas = {name: _schema(name) for name in registered_scenes()}
    first, reference = next(iter(schemas.items()))
    for name, schema in schemas.items():
        assert schema == reference, (
            f"scene {name!r} declares a different dataset schema than {first!r}; "
            f"datasets from the two cannot be merged or trained on together"
        )


def test_scenes_use_the_image_size_the_trainer_expects():
    """The audit hard-codes 256x256, so a scene must not pick another size.

    Checked against the scene's own config rather than the recorder, because the
    config is what a new scene would change.
    """

    for name in registered_scenes():
        scene = registered_scenes()[name]()
        env = scene.build_env(0.0, 0.0)
        try:
            assert env.config.image_height == TRAINED_IMAGE_SIZE, name
            assert env.config.image_width == TRAINED_IMAGE_SIZE, name
        finally:
            env.close()


def test_recorded_vectors_keep_the_contract_shapes():
    """State, velocity, force, and action shapes come from `contracts`, not a scene.

    This is why a new scene cannot invent a different observation width: it would
    have to change the contract, and the environment validates against it on every
    step.
    """

    features = lerobot_features(256, 256, use_videos=True)
    for key in (STATE, VELOCITY, FORCE, ACTION):
        assert key in features, f"{key} is missing from the recorded schema"
    assert features[STATE]["shape"] == (16,)
    assert features[ACTION]["shape"] == (16,)
    assert features[FORCE]["shape"] == (18,)


def test_video_and_image_modes_share_the_camera_shapes():
    """`use_videos` selects the storage, not the shape, so both stay compatible."""

    video = lerobot_features(256, 256, use_videos=True)
    image = lerobot_features(256, 256, use_videos=False)
    for key, feature in video.items():
        if feature.get("dtype") in ("video", "image"):
            assert image[key]["shape"] == feature["shape"]
            assert image[key]["dtype"] != feature["dtype"]


@pytest.mark.parametrize("size", [128, 512])
def test_a_second_image_size_would_break_the_schema(size: int) -> None:
    """Documents the failure this file exists to prevent.

    A scene that changed the image size would produce a dataset the trainer
    refuses.  Pinning that here means the guard above is known to be load-bearing
    rather than decorative.
    """

    assert lerobot_features(size, size, use_videos=True) != lerobot_features(
        TRAINED_IMAGE_SIZE, TRAINED_IMAGE_SIZE, use_videos=True
    )
