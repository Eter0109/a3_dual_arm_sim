"""What the dataset's ``action`` column *is*, and what its metadata calls it.

The two have to agree, and until this existed they did not have to.  ``runner.py`` recorded
``info["applied_action"]`` and ``recording.py`` wrote the literal
``stored_action_mode = "joint_position"``.  On a joint-servo arm that pair is correct, which is
why it survived: the environment's canonical command really is a joint position there.

Under operational-space control it is not.  `OperationalSpaceAdapter` solves for the torques that
drive the arm towards a latched Cartesian goal, and `_limit_joint_action` deliberately neither
clips nor rate-limits those seven numbers because clipping a torque is meaningless.  So a scene
with LIBERO's Panda on the left arm -- which is the scene this repository is now collecting on --
would have shipped Newton-metres inside a field whose own metadata said they were joint angles,
and nothing anywhere would have objected.  That is the F5-F13 failure mode in
``docs/bug-catalogue.md``: not a crash, a silent lie in the instrument.

These tests cover both halves.  The environment refuses to put a name on a vector that is not an
action, and the runner declares the space it actually recorded -- which for a policy-space
dataset is the space the policy emits, the one a fine-tune can use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from a3_dual_arm_sim.collection import registered_scenes, scene_by_name
from a3_dual_arm_sim.env import A3DualArmEnv
from a3_dual_arm_sim.recording import LeRobotV3Recorder, MemoryRecorder
from a3_dual_arm_sim.runner import EpisodeRunner, stored_action_mode

#: A scene whose left arm is A3's own, driven by a joint-position servo.
JOINT_ARM_SCENE = "a3_cookie_single_piece"
#: A scene whose left arm is robosuite's Panda, driven by operational-space control.
TORQUE_ARM_SCENE = "a3_libero_panda_single_cookie"

#: Every registered scene that mounts robosuite's Panda.  Written out rather than filtered
#: from the registry because the registry records no such field, and a list that has to be
#: kept right by hand is exactly what the test below it is for.
PANDA_ARM_SCENES = (
    "a3_libero_panda",
    "a3_libero_panda_cookie",
    "a3_libero_panda_single_cookie",
    "a3_libero_panda_two_cookies",
)


class HoldPolicy:
    """Emit a still command of whatever width the environment asked for.

    The subject here is the *recording* of an action, not the action, so the policy that produces
    it should be incapable of being interesting.
    """

    def __init__(self, action_mode: str) -> None:
        self.action_mode = action_mode

    def reset(self, context: Any = None) -> None:
        return None

    def act(self, observation: dict[str, Any], task: str = "") -> np.ndarray:
        del observation, task
        width = 16 if self.action_mode == "joint_position" else 14
        return np.zeros(width, dtype=np.float64)

    def close(self) -> None:
        return None


def test_a_joint_servo_arm_names_its_command() -> None:
    """The applied action of a position-servo arm really is a joint position."""

    env = scene_by_name(JOINT_ARM_SCENE).build_env(0.0, 0.0)
    try:
        assert env.canonical_action_mode == "joint_position"
    finally:
        env.close()


def test_a_torque_arm_refuses_to_name_its_command() -> None:
    """There is no action space whose entries are Newton-metres, so do not invent one."""

    env = scene_by_name(TORQUE_ARM_SCENE).build_env(0.0, 0.0)
    try:
        with pytest.raises(ValueError, match="operational-space"):
            _ = env.canonical_action_mode
    finally:
        env.close()


def test_the_runner_refuses_to_store_torque_as_joint_positions() -> None:
    """The default recording choice must fail loudly rather than write the wrong numbers.

    ``recorded_action`` defaults to ``"applied"``, which is what every scene recorded before
    the choice existed.  On this scene that default is a mistake, and the mistake is the one
    that would never be noticed -- so it has to raise, and the message has to name the fix.
    """

    env = scene_by_name(TORQUE_ARM_SCENE).build_env(0.0, 0.0)
    runner = EpisodeRunner(
        env, HoldPolicy(env.action_mode), task="torque refusal", recorder=MemoryRecorder()
    )
    try:
        with pytest.raises(ValueError, match="requested_action"):
            runner.run(seed=0, max_steps=1)
    finally:
        runner.close()


def test_the_runner_declares_the_space_it_recorded() -> None:
    """Recording the policy's own action means declaring the policy's own space.

    This is the shape the collection runs use: a `cartesian_delta` environment -- the space
    `scripts/adapter_libero_iface.py` declares for the policy -- with ``recorded_action`` set to
    ``"requested"``.  What lands in the frames has to be that 14-wide vector, bounded as the
    contract requires, and the metadata has to say so.
    """

    env = A3DualArmEnv(render_cameras=False)
    # The space a policy emits, which is `cartesian_delta` for the LIBERO interface.
    env.action_mode = "cartesian_delta"
    recorder = MemoryRecorder()
    runner = EpisodeRunner(
        env,
        HoldPolicy("cartesian_delta"),
        task="requested recording",
        recorder=recorder,
        recorded_action="requested",
    )
    try:
        result = runner.run(seed=0, max_steps=3)
        assert result.steps == 3
    finally:
        runner.close()

    assert recorder.stored_action_mode == "cartesian_delta"
    frames = recorder.episodes[0]
    assert len(frames) == 3
    for frame in frames:
        assert frame["action"].shape == (14,), frame["action"].shape
        assert np.all(np.abs(frame["action"]) <= 1.0)


def test_the_default_still_records_the_applied_joint_command() -> None:
    """A joint-servo scene's metadata must be byte-identical to what it was.

    The fix is allowed to add a choice; it is not allowed to change what every existing scene
    already produced.  Both modes name the same space here -- that is the point -- and the width
    is the joint one, 16, not 14.
    """

    env = A3DualArmEnv(render_cameras=False)
    recorder = MemoryRecorder()
    runner = EpisodeRunner(
        env, HoldPolicy("joint_position"), task="default recording", recorder=recorder
    )
    try:
        runner.run(seed=0, max_steps=2)
    finally:
        runner.close()

    assert recorder.stored_action_mode == "joint_position"
    for frame in recorder.episodes[0]:
        assert frame["action"].shape == (16,)


def test_the_written_metadata_carries_the_stored_mode(tmp_path: Path) -> None:
    """The line a reader trusts is the one that has to be checked on disk.

    Everything above is about the value being computed correctly; this is about it reaching the
    file a dataset ships with, because the defect being repaired lived in exactly that literal.
    """

    pytest.importorskip("lerobot")
    root = tmp_path / "dataset"
    env = A3DualArmEnv(render_cameras=False)
    env.action_mode = "cartesian_delta"
    recorder = LeRobotV3Recorder(
        root,
        repo_id="local/a3-recorded-action-mode",
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
        # The schema's width follows the space, and LeRobot refuses a frame of another shape
        # rather than coercing it: measured, a 14-wide action into a 16-wide schema raises
        # "does not have the expected shape".  So this argument is not optional for a
        # policy-space dataset, and the recorder asserts it matches what the runner records.
        action_mode="cartesian_delta",
        use_videos=False,
    )
    runner = EpisodeRunner(
        env,
        HoldPolicy("cartesian_delta"),
        task="metadata carried",
        recorder=recorder,
        recorded_action="requested",
    )
    try:
        runner.run(seed=3, max_steps=2)
    finally:
        runner.close()

    lines = (root / "a3_episode_metadata.jsonl").read_text().splitlines()
    metadata = [json.loads(line) for line in lines]
    assert len(metadata) == 1
    assert metadata[0]["stored_action_mode"] == "cartesian_delta"
    assert metadata[0]["source_action_mode"] == "cartesian_delta"
    assert metadata[0]["frames"] == 2


def test_a_schema_of_the_wrong_width_is_refused_before_any_frame(tmp_path: Path) -> None:
    """The dataset's declared shape is fixed at creation, so the mismatch has to be caught.

    A LeRobot dataset is created with the ``action`` width written into its ``info.json`` and
    every frame is validated against it.  That validation is a good failure -- it names the
    shapes -- but it arrives on the first frame, after the recorder has been handed a run and
    after any earlier episodes have been written.  This makes the same mistake arrive at
    ``start_episode``, which is the first moment the two declarations meet.
    """

    pytest.importorskip("lerobot")
    env = A3DualArmEnv(render_cameras=False)
    env.action_mode = "cartesian_delta"
    recorder = LeRobotV3Recorder(
        tmp_path / "dataset",
        repo_id="local/a3-recorded-action-mode-mismatch",
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
        use_videos=False,
    )
    runner = EpisodeRunner(
        env,
        HoldPolicy("cartesian_delta"),
        task="schema mismatch",
        recorder=recorder,
        recorded_action="requested",
    )
    try:
        with pytest.raises(ValueError, match="was created for"):
            runner.run(seed=0, max_steps=1)
    finally:
        runner.close()


def test_every_registered_scene_declares_a_space_that_can_be_named() -> None:
    """The declaration is a closed set, and only two values are in it.

    ``collect_dataset`` builds the dataset's schema from this field *before* the first episode,
    so a value outside the set is not a wrong column -- it is no dataset at all, raised from
    ``stored_action_mode`` and traceable to this line and nothing else.  Checked over the whole
    registry because the cost is nil: the scene objects are closures, so naming every one builds
    nothing and runs no physics.
    """

    for name in registered_scenes():
        scene = scene_by_name(name)
        assert scene.recorded_action in ("applied", "requested"), (name, scene.recorded_action)


@pytest.mark.parametrize("name", PANDA_ARM_SCENES)
def test_a_panda_arm_scene_records_the_space_its_own_policy_emits(name: str) -> None:
    """A Panda scene's declared action space has to be one its own environment will name.

    This is the defect that stopped the first two-Cookie collection, and it is worth stating in
    full because *every* message it produced pointed somewhere else.  LIBERO's Panda is driven by
    operational-space control, so the vector the actuators receive is a joint torque and
    ``canonical_action_mode`` refuses to call it an action; and ``collect`` -- which reaches its
    scenes through the registry -- therefore declared a schema it could never fill and died
    before the first episode.  ``gate``, which builds its scenes by hand and passes
    ``recorded_action="requested"`` itself, was green throughout.

    Naming the space is only half of it.  The other half is ``cartesian``: the policy a scene
    builds has to emit in the space its arm is driven in, and the registry was passing ``panda``
    without ``cartesian``, so a *joint*-command policy was wired to a *torque* arm.  That one had
    no error at all -- twenty episodes of 750-1100 steps each, ending with ``cookies_in_source``
    still at 2, because the arm never reached a Cookie.  Both are asserted together here because
    both are consequences of the same missing word in the same registry entry.
    """

    scene = scene_by_name(name)
    env = scene.build_env(0.0, 0.0)
    try:
        assert str(env.action_mode) == "cartesian_delta"
        assert stored_action_mode(env, scene.recorded_action) == "cartesian_delta"
        # The policy's own space, read off the expert it built rather than assumed: a policy
        # emitting joint targets into this environment is the silent half of the failure.
        policy = scene.build_policy(env)
        try:
            assert str(policy.action_mode) == "cartesian_delta"
        finally:
            policy.close()
    finally:
        env.close()
