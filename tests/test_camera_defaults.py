"""The camera pose must have exactly one source, and every path must reach it.

``front`` is a policy camera: it is fed to models as ``observation.images.front``.
So when two ways of building a config disagree about its pose, two models are shown
different pictures of the same scene -- and nothing raises, because both configs are
individually valid.

That is what had happened.  The pose was written out three times: in
``CameraConfig``, in ``load_config``'s fallbacks, and in ``configs/default.yaml``.
The dataclass copy was dead -- the loader's literals shadowed it -- so a caller using
``CameraConfig()`` and a caller loading a scene config that omitted the ``cameras``
block got front cameras half a metre apart.  ``origin/main`` fixed the same fault in
``ff552c4``.

These tests hold the remaining copies together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from a3_dual_arm_sim import contracts
from a3_dual_arm_sim.config import CameraConfig, load_config

#: Every shipped scene config.  Only ``default.yaml`` declares a ``cameras`` block;
#: the rest inherit, which is precisely why the loader fallbacks have to agree with
#: the dataclass.
SCENE_CONFIGS = tuple(sorted(Path("configs").glob("*.yaml")))


def _inheriting() -> list[Path]:
    return [
        path
        for path in SCENE_CONFIGS
        if "cameras:" not in path.read_text(encoding="utf-8")
    ]


def test_there_are_scene_configs_that_inherit_the_pose() -> None:
    """The premise of the next test; without it that test proves nothing."""

    assert _inheriting(), "no config inherits the camera defaults any more"


def test_every_inheriting_config_gets_the_dataclass_pose() -> None:
    """``load_config``'s fallbacks must not be a second copy of the defaults."""

    for path in _inheriting():
        cameras = load_config(path).cameras
        assert cameras.front_position_m == CameraConfig().front_position_m, path
        assert cameras.workspace_target_m == CameraConfig().workspace_target_m, path
        assert cameras.front_fovy_deg == CameraConfig().front_fovy_deg, path


def test_the_shipped_default_config_states_the_same_pose() -> None:
    """``configs/default.yaml`` writes the pose out, so it is a real second copy."""

    written: dict[str, object] = {}
    text = (Path("configs") / "default.yaml").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        for key in ("workspace_target_m", "front_position_m", "front_fovy_deg"):
            if stripped.startswith(f"{key}:"):
                raw = stripped.split(":", 1)[1].strip()
                written[key] = (
                    float(raw)
                    if key == "front_fovy_deg"
                    else tuple(float(value) for value in raw.strip("[]").split(","))
                )

    expected = CameraConfig()
    assert written["workspace_target_m"] == expected.workspace_target_m
    assert written["front_position_m"] == expected.front_position_m
    assert written["front_fovy_deg"] == expected.front_fovy_deg


def test_the_front_pose_is_steeper_and_closer_than_the_one_it_replaced() -> None:
    """A regression guard against drifting back to the flat, distant pose.

    The measurements behind the change, taken by differential rendering at 640x640
    and counting the pixels that move when one slot tile is lifted:

    ===============  =========  =========  =============  ===============
    pose             elevation  distance   slot pixels    arms in frame
    ===============  =========  =========  =============  ===============
    previous          21.9 deg   1.289 m          339      both, fully
    this one          41.7 deg   1.150 m          839      both, fully
    origin/main       41.7 deg   0.722 m         2740      none
    ===============  =========  =========  =============  ===============

    Only the pose itself is asserted here.  The pixel counts measure one scene at one
    size, so pinning them would fail the first time the palette or the render size
    changed, without anything having actually gone wrong.
    """

    pose = np.array(CameraConfig().front_position_m, dtype=float)
    target = np.array(CameraConfig().workspace_target_m, dtype=float)
    offset = target - pose
    elevation = float(np.degrees(np.arctan2(abs(offset[2]), np.linalg.norm(offset[:2]))))
    distance = float(np.linalg.norm(offset))

    assert elevation > 35.0, (
        f"front camera is only {elevation:.1f} deg above the table; at the previous "
        "21.9 deg the target bin's own wall hid all but 339 pixels of the placement "
        "lattice"
    )
    assert 1.0 < distance < 1.30, (
        f"front camera is {distance:.3f} m out; below 1.0 m the arms leave the frame "
        "and above 1.30 m the lattice stops reading (see CameraConfig's docstring)"
    )
    # And it is aimed at the work surface, not somewhere else that merely happens to
    # sit at the right angle and distance.
    assert -0.10 < target[0] < 0.40, target
    assert 0.10 < target[1] < 0.40, target
    assert 0.70 < target[2] < 0.95, target


def test_the_front_image_key_a_policy_reads_comes_from_the_front_camera() -> None:
    """The premise that makes the pose a policy decision rather than a cosmetic one.

    If ``front`` ever stopped being an observation key, the tests above would be
    arguing about a camera nobody is shown.  This walks the real path instead of
    asserting the coupling in prose: the shipped manifests name
    ``observation.images.front`` as a model input, so the camera's pose reaches the
    model, and changing it changes what the model sees.
    """

    from a3_dual_arm_sim.bridge.manifest import load_manifest

    assert contracts.FRONT_IMAGE == "observation.images.front"

    policy_dir = Path("configs") / "policies"
    manifests = sorted(policy_dir.glob("*.yaml"))
    assert manifests, "no policy manifests to check"

    # Every shipped manifest has to load.  An earlier version of this test caught
    # exceptions here on the theory that the ``*_template`` files might be
    # incomplete; they are not, and swallowing a failure would have hidden a real
    # one, so the manifests are loaded the same way the runner loads them.
    feeds_front = {}
    for path in manifests:
        manifest = load_manifest(path)
        sources = {
            spec.source
            for spec in manifest.observation.images.values()
            if spec.source is not None
        }
        if sources:
            feeds_front[path.name] = contracts.FRONT_IMAGE in sources

    assert feeds_front, "no manifest declares image inputs"
    assert any(feeds_front.values()), (
        "no shipped manifest feeds the front camera to a model, so its pose is no "
        f"longer a policy input: {feeds_front}"
    )
