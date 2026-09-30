"""Tests for the four mechanical stages between a model and the environment.

Each stage is tested on its own because each has a failure mode that looks like
nothing at all from the outside: a dropped camera, a missing denormalisation, a
dimension placed in the wrong slot, a chunk re-planned every step.  All four
produce plausible motion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from a3_dual_arm_sim.bridge.action_bridge import ActionBridge
from a3_dual_arm_sim.bridge.chunk import ChunkBuffer
from a3_dual_arm_sim.bridge.manifest import (
    ActionSpec,
    AffineSpec,
    ImageSpec,
    NormalizeSpec,
    ObservationSpec,
    VectorSpec,
)
from a3_dual_arm_sim.bridge.normalize import Normalizer
from a3_dual_arm_sim.bridge.observation import ObservationView
from a3_dual_arm_sim.bridge.protocol import BridgeError
from a3_dual_arm_sim.contracts import (
    EEF_POSE,
    FORCE,
    FRONT_IMAGE,
    LEFT_WRIST_IMAGE,
    RIGHT_WRIST_IMAGE,
    STATE,
    VELOCITY,
    ContractError,
)

# ---------------------------------------------------------------------------
# observation projection
# ---------------------------------------------------------------------------


def observation(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        FRONT_IMAGE: np.zeros((64, 48, 3), dtype=np.uint8),
        LEFT_WRIST_IMAGE: np.zeros((64, 48, 3), dtype=np.uint8),
        RIGHT_WRIST_IMAGE: np.zeros((64, 48, 3), dtype=np.uint8),
        STATE: np.arange(16, dtype=np.float32),
        VELOCITY: np.zeros(16, dtype=np.float32),
        EEF_POSE: np.zeros(14, dtype=np.float32),
        FORCE: np.zeros(18, dtype=np.float32),
        "time": 1.5,
        "safety_stop": False,
    }
    base.update(overrides)
    return base


def test_images_are_resized_and_carried_as_uint8() -> None:
    view = ObservationView(
        ObservationSpec(
            images={"camera": ImageSpec(source=FRONT_IMAGE, resize=(224, 224))}
        )
    )
    arrays, _ = view.project(observation(), "task")

    assert arrays["camera"].shape == (224, 224, 3)
    assert arrays["camera"].dtype == np.uint8


def test_a_camera_at_the_requested_size_is_not_resampled() -> None:
    view = ObservationView(
        ObservationSpec(images={"camera": ImageSpec(source=FRONT_IMAGE, resize=(64, 48))})
    )
    source = observation()
    arrays, _ = view.project(source, "task")

    assert arrays["camera"] is not None
    assert np.array_equal(arrays["camera"], source[FRONT_IMAGE])


def test_native_resolution_is_kept_when_no_resize_is_declared() -> None:
    view = ObservationView(ObservationSpec(images={"camera": ImageSpec(source=FRONT_IMAGE)}))
    arrays, _ = view.project(observation(), "task")

    assert arrays["camera"].shape == (64, 48, 3)


def test_a_vector_slice_selects_part_of_a_source() -> None:
    view = ObservationView(
        ObservationSpec(vectors={"arm": VectorSpec(source=STATE, slice=(0, 14))})
    )
    arrays, _ = view.project(observation(), "task")

    assert arrays["arm"].shape == (14,)
    assert arrays["arm"].dtype == np.float32
    assert np.array_equal(arrays["arm"], np.arange(14, dtype=np.float32))


def test_indices_select_a_non_contiguous_subset() -> None:
    # The case a real checkpoint forces: A3 has seven joints per arm, ALOHA has
    # six, so the values that must be dropped are in the middle of each arm's
    # block (indices 6 and 14) and no slice can express that.
    indices = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)
    view = ObservationView(
        ObservationSpec(vectors={"proprio": VectorSpec(source=STATE, indices=indices)})
    )
    arrays, _ = view.project(observation(), "task")

    expected = np.arange(16, dtype=np.float32)[np.asarray(indices)]
    assert arrays["proprio"].shape == (14,)
    assert np.array_equal(arrays["proprio"], expected)


def test_indices_past_the_end_of_a_source_are_refused() -> None:
    view = ObservationView(
        ObservationSpec(vectors={"proprio": VectorSpec(source=STATE, indices=(0, 99))})
    )
    with pytest.raises(ContractError, match="indices reach 99"):
        view.project(observation(), "task")


def test_a_missing_source_key_is_refused_and_names_what_is_present() -> None:
    view = ObservationView(
        ObservationSpec(images={"camera": ImageSpec(source="observation.images.top")})
    )
    with pytest.raises(ContractError) as error:
        view.project(observation(), "task")

    message = str(error.value)
    assert "observation.images.top" in message
    assert FRONT_IMAGE in message


def test_a_slice_past_the_end_of_a_vector_is_refused() -> None:
    view = ObservationView(
        ObservationSpec(vectors={"everything": VectorSpec(source=STATE, slice=(0, 32))})
    )
    with pytest.raises(ContractError, match="exceeds"):
        view.project(observation(), "task")


def test_a_non_uint8_camera_is_refused() -> None:
    view = ObservationView(ObservationSpec(images={"camera": ImageSpec(source=FRONT_IMAGE)}))
    with pytest.raises(ContractError, match="uint8"):
        view.project(observation(**{FRONT_IMAGE: np.zeros((8, 8, 3), dtype=np.float32)}), "t")


def test_image_names_keep_manifest_order() -> None:
    view = ObservationView(
        ObservationSpec(
            images={
                "third": ImageSpec(source=RIGHT_WRIST_IMAGE),
                "first": ImageSpec(source=FRONT_IMAGE),
            }
        )
    )

    assert view.image_names == ("third", "first")
    assert view.names == ("third", "first")


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def write_stats(path: Path, prefix: str, **values: np.ndarray) -> Path:
    np.savez(path, **{f"{prefix}.{name}": value for name, value in values.items()})
    return path


def test_mean_std_inverse_restores_physical_units(tmp_path: Path) -> None:
    mean = np.array([1.0, -2.0, 0.5])
    std = np.array([0.5, 2.0, 0.25])
    stats = write_stats(tmp_path / "stats.npz", "action", mean=mean, std=std)
    normalizer = Normalizer(NormalizeSpec(mode="mean_std", stats=stats))

    # A normalised zero is the mean, and a normalised one is one std above it.
    restored = normalizer.denormalize(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    assert np.allclose(restored[0], mean)
    assert np.allclose(restored[1], mean + std)


def test_min_max_inverse_maps_the_unit_range_onto_the_observed_range(tmp_path: Path) -> None:
    low = np.array([-1.0, 0.0])
    high = np.array([1.0, 10.0])
    stats = write_stats(tmp_path / "stats.npz", "action", min=low, max=high)
    normalizer = Normalizer(NormalizeSpec(mode="min_max", stats=stats))

    restored = normalizer.denormalize(np.array([[-1.0, -1.0], [1.0, 1.0]]))

    assert np.allclose(restored[0], low)
    assert np.allclose(restored[1], high)


def test_q01_q99_uses_the_published_quantiles(tmp_path: Path) -> None:
    stats = write_stats(
        tmp_path / "stats.npz",
        "action",
        q01=np.array([0.0, 0.0]),
        q99=np.array([2.0, 4.0]),
    )
    normalizer = Normalizer(NormalizeSpec(mode="q01_q99", stats=stats))

    assert np.allclose(normalizer.denormalize(np.zeros((1, 2))), [[1.0, 2.0]])


def test_none_mode_passes_values_through_untouched() -> None:
    normalizer = Normalizer(NormalizeSpec(mode="none"))
    values = np.array([[0.3, -0.7]])

    assert np.array_equal(normalizer.denormalize(values), values)
    assert normalizer.mode == "none"


def test_statistics_are_found_under_common_aliases(tmp_path: Path) -> None:
    # Projects publish the same numbers as mean/std, loc/scale, or with a
    # dataset prefix; the file is not worth a schema argument, but a missing one
    # is a hard failure.
    np.savez(tmp_path / "aliased.npz", loc=np.zeros(3), scale=np.ones(3))
    normalizer = Normalizer(NormalizeSpec(mode="mean_std", stats=tmp_path / "aliased.npz"))

    assert np.allclose(normalizer.denormalize(np.array([[2.0, 2.0, 2.0]])), 2.0)


def test_missing_statistics_are_reported_with_what_was_tried(tmp_path: Path) -> None:
    np.savez(tmp_path / "partial.npz", **{"action.mean": np.zeros(3)})
    with pytest.raises(BridgeError) as error:
        Normalizer(NormalizeSpec(mode="mean_std", stats=tmp_path / "partial.npz"))

    assert "std" in str(error.value)
    assert "action.mean" in str(error.value)


def test_degenerate_statistics_are_refused_rather_than_dividing_by_zero(tmp_path: Path) -> None:
    stats = write_stats(
        tmp_path / "flat.npz", "action", mean=np.zeros(2), std=np.zeros(2)
    )
    normalizer = Normalizer(NormalizeSpec(mode="mean_std", stats=stats))

    with pytest.raises(BridgeError, match="divide by zero"):
        normalizer.denormalize(np.zeros((1, 2)))


def test_statistics_can_be_a_json_object(tmp_path: Path) -> None:
    import json

    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"action": {"mean": [1.0], "std": [2.0]}}), encoding="utf-8")
    # The nested form is not one of the accepted spellings; the flat one is.
    with pytest.raises(BridgeError):
        Normalizer(NormalizeSpec(mode="mean_std", stats=path))

    path.write_text(json.dumps({"action.mean": [1.0], "action.std": [2.0]}), encoding="utf-8")
    normalizer = Normalizer(NormalizeSpec(mode="mean_std", stats=path))
    assert np.allclose(normalizer.denormalize(np.zeros((1, 1))), 1.0)


# ---------------------------------------------------------------------------
# chunk pacing
# ---------------------------------------------------------------------------


def test_a_chunk_is_drained_one_row_per_call() -> None:
    buffer = ChunkBuffer(chunk=4, execute=4)
    buffer.expected_dim = 3
    assert buffer.load(np.arange(12, dtype=float).reshape(4, 3)) == 4

    rows = [buffer.pop() for _ in range(4)]

    assert [row[0] for row in rows] == [0.0, 3.0, 6.0, 9.0]
    assert buffer.pop() is None
    assert buffer.needs_replan


def test_executing_fewer_steps_than_predicted_drops_the_tail() -> None:
    buffer = ChunkBuffer(chunk=50, execute=10)
    buffer.expected_dim = 2
    buffer.load(np.ones((50, 2)))

    for _ in range(10):
        assert buffer.pop() is not None
    # The tail is dropped rather than run open loop, which is how reactivity is
    # bought without changing the model.
    assert buffer.needs_replan
    assert buffer.steps_until_replan == 0


def test_a_short_chunk_is_accepted_and_reported() -> None:
    buffer = ChunkBuffer(chunk=50, execute=50)
    buffer.expected_dim = 2

    assert buffer.load(np.ones((7, 2))) == 7
    assert buffer.steps_until_replan == 7


def test_chunk_counts_are_exposed_for_the_report() -> None:
    buffer = ChunkBuffer(chunk=2, execute=2)
    buffer.expected_dim = 1
    buffer.load(np.ones((2, 1)))
    buffer.pop()
    buffer.pop()
    buffer.load(np.ones((2, 1)))

    assert buffer.metrics() == {
        "chunk_size": 2,
        "execute_steps": 2,
        "chunks_consumed": 2,
    }


def test_a_width_mismatch_is_refused_at_load() -> None:
    buffer = ChunkBuffer(chunk=4, execute=4)
    buffer.expected_dim = 16
    with pytest.raises(BridgeError, match="declares 16"):
        buffer.load(np.ones((4, 14)))


def test_execute_longer_than_chunk_is_refused_at_construction() -> None:
    with pytest.raises(BridgeError, match="cannot exceed"):
        ChunkBuffer(chunk=4, execute=8)


def test_reset_clears_the_queue_and_the_counter() -> None:
    buffer = ChunkBuffer(chunk=2, execute=2)
    buffer.expected_dim = 1
    buffer.load(np.ones((2, 1)))
    buffer.pop()
    buffer.reset()

    assert buffer.needs_replan
    assert buffer.chunks == 0


# ---------------------------------------------------------------------------
# action bridge
# ---------------------------------------------------------------------------


def test_a_native_width_joint_action_passes_straight_through() -> None:
    bridge = ActionBridge(ActionSpec(space="joint_position", dim=16), target_mode="joint_position")
    row = np.linspace(-0.5, 0.5, 16)

    assert np.allclose(bridge.convert_row(row), row)
    assert not bridge.needs_env


def test_a_narrower_output_is_placed_by_the_joint_map_and_padded() -> None:
    # A model that drives one arm: two gutters at indices 0 and 8, everything else
    # left at the declared default so a gripper is never silently left closed.
    bridge = ActionBridge(
        ActionSpec(space="joint_position", dim=2, joint_map=(0, 8), default=1.0),
        target_mode="joint_position",
    )
    action = bridge.convert_row(np.array([0.25, -0.25]))

    assert action.shape == (16,)
    assert action[0] == pytest.approx(0.25)
    assert action[8] == pytest.approx(-0.25)
    assert np.allclose(np.delete(action, [0, 8]), 1.0)


def test_normalisation_is_applied_before_the_map(tmp_path: Path) -> None:
    # Order matters: mapping first would place normalised values in the wrong
    # slots and produce plausible motion from wrong numbers.
    mean = np.array([0.0, 0.0])
    std = np.array([2.0, 4.0])
    stats = write_stats(tmp_path / "stats.npz", "action", mean=mean, std=std)
    bridge = ActionBridge(
        ActionSpec(
            space="joint_position",
            dim=2,
            joint_map=(0, 8),
            normalize=NormalizeSpec(mode="mean_std", stats=stats),
        ),
        target_mode="joint_position",
    )
    action = bridge.convert_row(np.array([1.0, 1.0]))

    assert action[0] == pytest.approx(2.0)
    assert action[8] == pytest.approx(4.0)


def test_hold_leaves_unclaimed_joints_at_their_measured_positions() -> None:
    # The ALOHA-to-A3 case: 14 model dimensions driving 16 native ones.  The two
    # joints the model has no outputs for must stay where they are, not be
    # commanded to an arbitrary constant.
    measured = np.arange(16, dtype=np.float64) * 0.1

    class FakeEnv:
        current_joint_action = measured

    bridge = ActionBridge(
        ActionSpec(
            space="joint_position",
            dim=2,
            joint_map=(0, 8),
            default="hold",
        ),
        target_mode="joint_position",
        env=FakeEnv(),
    )
    action = bridge.convert_row(np.array([0.5, -0.5]))

    assert bridge.holds_uncontrolled_joints
    assert action[0] == pytest.approx(0.5)
    assert action[8] == pytest.approx(-0.5)
    # Everything else is exactly what the arm reported, including the two wrist
    # joints whose zero happens to match A3's home pose.
    assert np.allclose(np.delete(action, [0, 8]), np.delete(measured, [0, 8]))


def test_held_joints_do_not_follow_the_arm_once_latched() -> None:
    """A held joint must stay put, not be re-targeted to wherever it drifted.

    Re-reading the position every step looks equivalent and is not: the arm's own
    motion perturbs a held joint, and commanding it back to the *new* position
    ratifies the perturbation instead of resisting it.  Measured on the real ALOHA
    mapping, the two unclaimed wrist joints wandered about 0.1 rad in 32 steps that
    way.  Latching at reset is what stops a joint the model cannot command from
    confounding an evaluation.
    """

    start = np.arange(16, dtype=np.float64) * 0.1

    class DriftingEnv:
        """Reports a different pose each time, as a simulator under load would."""

        def __init__(self) -> None:
            self.reads = 0

        @property
        def current_joint_action(self) -> np.ndarray:
            value = start + 0.01 * self.reads
            self.reads += 1
            return value

    env = DriftingEnv()
    bridge = ActionBridge(
        ActionSpec(space="joint_position", dim=2, joint_map=(0, 8), default="hold"),
        target_mode="joint_position",
        env=env,
    )
    latched = bridge.latch_hold_values()
    assert latched is not None
    assert env.reads == 1, "latching must read the pose exactly once"
    assert np.allclose(latched, start)

    first = bridge.convert_row(np.array([0.5, -0.5]))
    second = bridge.convert_row(np.array([0.5, -0.5]))

    # The commanded model dimensions are unaffected, and the held ones are identical
    # across steps even though the environment kept reporting new positions.
    assert first[0] == pytest.approx(0.5)
    assert np.allclose(np.delete(first, [0, 8]), np.delete(latched, [0, 8]))
    assert np.allclose(first[6], second[6])
    assert np.allclose(first[14], second[14])
    assert env.reads == 1, "a latched bridge must not keep re-reading"


def test_hold_latches_even_without_an_explicit_reset() -> None:
    """A caller driving the bridge directly gets the same frozen behaviour."""

    start = np.full(16, 0.25)

    class DriftingEnv:
        def __init__(self) -> None:
            self.reads = 0

        @property
        def current_joint_action(self) -> np.ndarray:
            value = start + 0.5 * self.reads
            self.reads += 1
            return value

    env = DriftingEnv()
    bridge = ActionBridge(
        ActionSpec(space="joint_position", dim=2, joint_map=(0, 8), default="hold"),
        target_mode="joint_position",
        env=env,
    )
    first = bridge.convert_row(np.array([0.0, 0.0]))
    second = bridge.convert_row(np.array([0.0, 0.0]))

    assert np.allclose(first[6], start[6])
    assert np.allclose(second[6], start[6])
    assert env.reads == 1


def test_hold_without_an_environment_is_refused() -> None:
    bridge = ActionBridge(
        ActionSpec(space="joint_position", dim=2, joint_map=(0, 8), default="hold"),
        target_mode="joint_position",
    )
    with pytest.raises(BridgeError, match="needs the live environment"):
        bridge.convert_row(np.array([0.0, 0.0]))


def test_a_constant_default_still_works() -> None:
    bridge = ActionBridge(
        ActionSpec(space="joint_position", dim=2, joint_map=(0, 8), default=0.75),
        target_mode="joint_position",
    )
    action = bridge.convert_row(np.array([0.25, -0.25]))

    assert not bridge.holds_uncontrolled_joints
    assert np.allclose(np.delete(action, [0, 8]), 0.75)


def test_clip_bounds_the_result(tmp_path: Path) -> None:
    bridge = ActionBridge(
        ActionSpec(space="joint_position", dim=16, clip=(-0.1, 0.1)),
        target_mode="joint_position",
    )
    action = bridge.convert_row(np.full(16, 5.0))

    assert np.allclose(action, 0.1)


def test_affine_converts_units_and_inverts_the_gripper() -> None:
    """The GR00T case, stated as arithmetic.

    GR00T's cartesian deltas are robosuite OSC-normalized (1.0 = 0.05 m, 0.5 rad)
    while A3 normalizes against 0.025 m and 0.10 rad, so translation scales by 2 and
    rotation by 5.  Its gripper also means the opposite: +1 closes, where A3's +1
    opens -- a negative scale with an offset, not a rescale.
    """

    scale = [2.0, 2.0, 2.0, 5.0, 5.0, 5.0, -2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, -2.0]
    offset = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
    bridge = ActionBridge(
        ActionSpec(
            space="cartesian_delta",
            dim=14,
            affine=AffineSpec(scale=tuple(scale), offset=tuple(offset)),
        ),
        target_mode="cartesian_delta",
    )
    row = np.array([0.5, -0.5, 0.25, 0.2, -0.2, 0.1, 1.0, 0.5, -0.5, 0.25, 0.2, -0.2, 0.1, 0.0])
    action = bridge.convert_row(row)

    # Position scales by two, rotation by five.
    assert action[0] == pytest.approx(1.0)
    assert action[1] == pytest.approx(-1.0)
    assert action[3] == pytest.approx(1.0)
    # Gripper: +1 (GR00T closed) becomes -1 (A3 closed), 0 becomes +1 (A3 open).
    assert action[6] == pytest.approx(-1.0)
    assert action[13] == pytest.approx(1.0)


def test_affine_applies_before_the_joint_map() -> None:
    """Order matters, and getting it wrong gives right-shaped wrong numbers.

    The unit conversion belongs to the model's own dimension order, so it has to
    happen while the values are still in that order.  Mapping first would convert
    units per *destination* slot -- swapping which arm gets which scale whenever the
    two arms are scaled differently, as GR00T's are not but another model's could be.
    """

    bridge = ActionBridge(
        ActionSpec(
            space="joint_position",
            dim=2,
            joint_map=(0, 8),
            default=0.0,
            affine=AffineSpec(scale=(10.0, 100.0)),
        ),
        target_mode="joint_position",
    )
    # dims 0 and 1 are invalid here: affine is a cartesian-only feature, but the
    # bridge itself must still apply it in the model's order.  The audit rejects
    # this combination; the arithmetic is what is under test.
    action = bridge.convert_row(np.array([1.0, 1.0]))

    assert action[0] == pytest.approx(10.0)
    assert action[8] == pytest.approx(100.0)


def test_an_identity_affine_changes_nothing() -> None:
    plain = ActionBridge(
        ActionSpec(space="cartesian_delta", dim=14), target_mode="cartesian_delta"
    )
    staged = ActionBridge(
        ActionSpec(
            space="cartesian_delta",
            dim=14,
            affine=AffineSpec(scale=(1.0,), offset=(0.0,)),
        ),
        target_mode="cartesian_delta",
    )
    row = np.linspace(-0.5, 0.5, 14)

    assert np.allclose(plain.convert_row(row), staged.convert_row(row))


def test_a_row_of_the_wrong_width_is_refused_with_both_numbers() -> None:
    bridge = ActionBridge(ActionSpec(space="joint_position", dim=16), target_mode="joint_position")
    with pytest.raises(ContractError) as error:
        bridge.convert_row(np.zeros(14))

    # Both numbers appear, because the reader has to decide which end is wrong.
    assert "14 action values" in str(error.value)
    assert "dim=16" in str(error.value)


def test_non_finite_values_are_refused() -> None:
    bridge = ActionBridge(ActionSpec(space="joint_position", dim=16), target_mode="joint_position")
    with pytest.raises(ContractError, match="NaN"):
        bridge.convert_row(np.full(16, np.nan))


def test_a_narrow_model_without_a_map_is_refused_at_construction() -> None:
    with pytest.raises(BridgeError, match="joint_map"):
        ActionBridge(ActionSpec(space="joint_position", dim=14), target_mode="joint_position")


def test_cartesian_actions_need_the_live_environment_for_the_ik() -> None:
    with pytest.raises(BridgeError, match="live environment"):
        ActionBridge(ActionSpec(space="cartesian_delta", dim=14), target_mode="joint_position")


def test_cartesian_actions_cannot_drive_a_joint_only_model() -> None:
    with pytest.raises(BridgeError, match="forward mapping"):
        ActionBridge(ActionSpec(space="joint_position", dim=16), target_mode="cartesian_delta")


def test_cartesian_actions_pass_through_when_the_environment_speaks_them() -> None:
    bridge = ActionBridge(
        ActionSpec(space="cartesian_delta", dim=14), target_mode="cartesian_delta"
    )
    row = np.zeros(14)

    assert np.allclose(bridge.convert_row(row), row)
    assert not bridge.needs_env
