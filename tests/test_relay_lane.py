"""The relay as a lane of N boxes, and what a lane costs.

Phase 4 of `docs/configurable-scenes.md`.  What is pinned here is the coordinator's
*structure* -- the stages a lane goes through, what each one waits for, and what the
final check demands -- because that is what the generalisation changed.  The two-box
behaviour is checked separately and more strongly, by hashing a whole episode's action
stream against the previous commit: for a two-box config nothing is meant to change, and
the method is recorded in `docs/configurable-scenes.md`.

Nothing here drives a full relay episode.  One takes about 4800 steps at the baseline
profile -- over an hour -- so these tests place the boxes and step the stages directly,
which is the same approach the two-box tests take.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.relay_batch_expert import RelayBatchExpert

ROOT = Path(__file__).resolve().parents[1]
RELAY_CONFIG = ROOT / "configs" / "cookie_two_box_batch.yaml"
THREE_BOX_CONFIG = ROOT / "configs" / "generated" / "cookie_three_box.yaml"
FIXED_SCENE = {"randomize_cookies": False, "randomize_scene": False}
RELAY_TASK = CookieTransferTaskConfig(
    require_exact_slots=False,
    require_released=True,
    require_upright=False,
    terminate_on_success=False,
)


def _relay_env(config_path: Path) -> A3CookieTransferEnv:
    return A3CookieTransferEnv(config_path, render_cameras=False, task_config=RELAY_TASK)


def _place(env: A3CookieTransferEnv, body: int, x: float, y: float) -> None:
    """Write a box's pose, keeping its height, and settle the model on it."""

    joint = env.model.body_jntadr[body]
    address = env.model.jnt_qposadr[joint]
    env.data.qpos[address] = x
    env.data.qpos[address + 1] = y
    env.data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)


def _lay_cookies(env: A3CookieTransferEnv, body: int, indices: list[int]) -> None:
    """Put ``indices`` Cookies into ``body`` where it currently sits.

    The box's own frame moves with it, so this has to run *after* the box is placed --
    and again after any move, since moving the box carries its Cookies out of it.

    The Cookies are laid into the box's *slots*, and the slot lattice has ten of them,
    so a box holding Cookies 10..19 uses the same ten slots with the later ids.  That is
    what a real lane does too: every box is filled the same way, with different Cookies.
    """

    previous = env._target_bin_body
    env._target_bin_body = body
    try:
        for offset, index in enumerate(indices):
            position = env.privileged_target_slot_world(
                offset % len(env.TARGET_SLOTS_LOCAL), env._target_cookie_center_z
            )
            env.set_cookie_pose(index, tuple(np.asarray(position[:3], dtype=float)))
    finally:
        env._target_bin_body = previous
    mujoco.mj_forward(env.model, env.data)


def _finish_fill(expert: RelayBatchExpert, indices: list[int]) -> None:
    """Make the current fill look finished, without running it.

    The fill's own ``finished`` is derived from its phase, so this sets the phase rather
    than the flag -- and the Cookie ids it reports are the ones the coordinator will
    record.
    """

    expert.fill.phase = expert.fill.phase.DONE
    expert.fill.completed_cookie_indices = list(indices)


# ------------------------------------------------------------------ the structure


def test_a_lane_starts_at_its_first_box_with_the_station_targeted():
    """The coordinator begins by filling the box that is already at the station."""

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        assert len(expert.boxes) == 3
        assert expert.names == ("target_bin", "spare_target_bin", "queue_target_bin_2")
        assert expert.index == 0
        assert expert.stage == "FILL"
        assert expert.push_distance == pytest.approx(0.090)
        # The fill grades the box the coordinator points it at.
        assert env._target_bin_body == expert.boxes[0]
        assert expert.indices == [[], [], []]
    finally:
        env.close()


def test_the_last_box_is_not_pushed():
    """A lane's last fill goes straight to the final check.

    There is nothing left to make room for, so pushing it would only move a filled box
    for no reason -- which is how a box gets spilled.  For two boxes this is the
    shipped behaviour: the second fill is followed by the dual check, not by a push.
    """

    for config_path, boxes in ((RELAY_CONFIG, 2), (THREE_BOX_CONFIG, 3)):
        env = _relay_env(config_path)
        try:
            env.reset(seed=0, options=FIXED_SCENE)
            expert = RelayBatchExpert(env)
            expert.reset()
            expert.index = boxes - 1
            expert.stage = "FILL"
            # Pretend the last fill just finished.
            _finish_fill(expert, list(range(10)))
            expert.act()
            assert expert.stage == "VERIFY_ALL", config_path.name
        finally:
            env.close()


def test_a_fill_that_is_not_the_last_starts_a_push_one_distance_out():
    """Every other fill is followed by a push, and the destination is the parameter.

    The distance is ``relay_push_distance_m`` rather than a literal: it is what the
    source bin's clearance is computed from, so a lane that changed one without the
    other would park its boxes somewhere the bin is not clear of.
    """

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        station_y = expert.station[1]
        for index in range(2):
            expert.index = index
            expert.stage = "FILL"
            _finish_fill(expert, list(range(10)))
            expert.act()
            assert expert.stage == "PUSH"
            assert expert.pusher.destination_y == pytest.approx(station_y + expert.push_distance)
            assert expert.pusher.box_id == expert.boxes[index]
            # A push uses the pad controller; a carry uses the pinch one.
            assert type(expert.pusher).__name__ == "RightBoxPushController"
    finally:
        env.close()


def test_the_carry_always_aims_at_the_station_and_uses_the_pinch_controller():
    """A carry's destination is the station, and it is the box *behind* the one filled."""

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        station = expert.station
        for index in range(2):
            # Each push has to have *worked* before the carry starts, so the box is
            # placed clear of the station -- the verify grades what it measures, not
            # what a controller reported.
            _place(env, expert.boxes[index], station[0], station[1] + 0.090 * (index + 1))
            _lay_cookies(env, expert.boxes[index], list(range(index * 10, index * 10 + 10)))
            expert.index = index
            expert.stage = "VERIFY_PUSH"
            expert._settled_steps = 14
            expert.indices[index] = list(range(index * 10, index * 10 + 10))
            expert.act()
            assert expert.stage == "CARRY"
            assert expert.pusher.destination_y == pytest.approx(expert.station[1])
            assert expert.pusher.box_id == expert.boxes[index + 1]
            assert type(expert.pusher).__name__ == "RightBoxCarryController"
    finally:
        env.close()


def test_a_carry_that_lands_advances_the_lane_by_one_box():
    """The verify's job is to move ``index`` on, so the next fill targets the right box."""

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        station = expert.station
        # Put box 1 at the station and fill box 0 where it is, clear of it.
        _place(env, expert.boxes[0], station[0], station[1] + 0.090)
        _lay_cookies(env, expert.boxes[0], list(range(10)))
        _place(env, expert.boxes[1], station[0], station[1])
        _lay_cookies(env, expert.boxes[1], [])

        expert.index = 0
        expert.indices[0] = list(range(10))
        expert.stage = "VERIFY_CARRY"
        expert._settled_steps = 14
        expert._verify_steps = 0
        expert.act()
        assert expert.stage == "FILL"
        assert expert.index == 1
        assert env._target_bin_body == expert.boxes[1]
    finally:
        env.close()


def test_a_carry_that_stops_short_is_reported_with_the_distance():
    """A box that does not arrive has to say how far out it is, and which box it is."""

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        station = expert.station
        _place(env, expert.boxes[0], station[0], station[1] + 0.090)
        _lay_cookies(env, expert.boxes[0], list(range(10)))
        _place(env, expert.boxes[1], station[0], station[1] + 0.020)
        expert.index = 0
        expert.indices[0] = list(range(10))
        expert.stage = "VERIFY_CARRY"
        expert._settled_steps = 0
        expert._verify_steps = RelayBatchExpert.VERIFY_SETTLE_STEPS - 1
        expert.act()
        assert expert.failed is not None
        assert "box 2 (spare_target_bin) did not reach" in expert.failed
        assert "20.0 mm" in expert.failed
    finally:
        env.close()


# ------------------------------------------------------------------- the final check


def test_the_final_check_grades_each_box_by_where_it_is():
    """The last box is at the station; the ones behind it were pushed clear of it.

    A lane's boxes are not interchangeable at the end: one is *in* the station and the
    rest are parked outside it, so a single tolerance for all of them would be wrong for
    every one of them.
    """

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        station = expert.station
        # Box 2 at the station (the last fill), boxes 1 and 0 parked clear of it, one
        # push distance apart.
        _place(env, expert.boxes[2], station[0], station[1])
        _place(env, expert.boxes[1], station[0], station[1] + expert.push_distance)
        _place(env, expert.boxes[0], station[0], station[1] + 2 * expert.push_distance)
        for index, body in enumerate(expert.boxes):
            _lay_cookies(env, body, list(range(index * 10, index * 10 + 10)))
            expert.indices[index] = list(range(index * 10, index * 10 + 10))

        expert.index = 2
        expert.stage = "VERIFY_ALL"
        expert._settled_steps = 19
        expert._verify_steps = 0
        expert.act()
        assert expert.failed is None, expert.failed
        assert expert.done
        assert expert.stage == "DONE"
        assert expert.all_counts() == [10, 10, 10]
    finally:
        env.close()


def test_the_final_check_names_every_box_when_it_fails():
    """A lane of three reporting only "the criteria failed" would not be diagnosable.

    The message has to say which box and by how much, for the same reason the two-box
    one does: a rejected episode is only useful if it says what to fix.
    """

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        station = expert.station
        _place(env, expert.boxes[2], station[0], station[1])
        _place(env, expert.boxes[1], station[0], station[1] + expert.push_distance)
        # Box 0 only half a push out: inside the clearance it has to keep.
        _place(env, expert.boxes[0], station[0], station[1] + 0.5 * expert.push_distance)
        for index, body in enumerate(expert.boxes):
            _lay_cookies(env, body, list(range(index * 10, index * 10 + 10)))
            expert.indices[index] = list(range(index * 10, index * 10 + 10))

        expert.index = 2
        expert.stage = "VERIFY_ALL"
        expert._settled_steps = 0
        expert._verify_steps = RelayBatchExpert.VERIFY_SETTLE_STEPS - 1
        expert.act()
        assert expert.failed is not None
        for name in expert.names:
            assert name in expert.failed, f"{name} is missing from: {expert.failed}"
        assert "45.0 mm clear (needs 85.0)" in expert.failed
        assert "source=50 (needs 50)" in expert.failed
    finally:
        env.close()


def test_the_final_check_wants_the_source_to_hold_what_was_not_moved():
    """``n - capacity * boxes``, computed rather than written down.

    The two-box scene's contract is 60 left in the source, and a lane of three needs 50;
    a literal 20 would have been wrong for every other lane length.
    """

    env = _relay_env(THREE_BOX_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        assert len(env._cookie_bodies) == 80
        assert env.batch_plan.capacity == 10
        # 80 - 10 * 3 = 50, which is what the failure above quotes.
        assert len(env._cookie_bodies) - env.batch_plan.capacity * len(expert.boxes) == 50
    finally:
        env.close()


def test_the_two_box_lane_reproduces_the_shipped_push_and_final_check():
    """The generalisation's anchor, in the two places a formula cannot check it.

    The action stream of a whole episode is compared against the previous commit by
    hashing (see `docs/configurable-scenes.md`), which is the strong statement.  These
    are the two *decisions* that hash cannot localise if it ever differs, so they are
    pinned directly as well: the push destination for a two-box lane is the 90 mm the
    shipped relay was measured with, and its final check reduces to the shipped pair of
    statements -- one box clear by 85 mm, one box at the station within 8 mm.
    """

    env = _relay_env(RELAY_CONFIG)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        assert len(expert.boxes) == 2
        assert expert.push_distance == pytest.approx(0.090)

        # The push destination, which the shipped coordinator wrote as `+ 0.090`.
        expert.index = 0
        expert.stage = "FILL"
        _finish_fill(expert, list(range(10)))
        expert.act()
        assert expert.stage == "PUSH"
        assert expert.pusher.destination_y == pytest.approx(expert.station[1] + 0.090)

        # The final check, on the layout a successful relay leaves behind.
        station = expert.station
        _place(env, expert.boxes[0], station[0], station[1] + 0.090)
        _place(env, expert.boxes[1], station[0], station[1])
        for index, body in enumerate(expert.boxes):
            _lay_cookies(env, body, list(range(index * 10, index * 10 + 10)))
            expert.indices[index] = list(range(index * 10, index * 10 + 10))
        expert.index = 1
        expert.stage = "VERIFY_ALL"
        expert._settled_steps = 19
        expert._verify_steps = 0
        expert.act()
        assert expert.failed is None, expert.failed
        assert expert.done
        assert expert.counts() == (10, 10)
    finally:
        env.close()


def test_a_single_box_scene_is_a_lane_of_one():
    """Degenerate but not special: one fill, no push, no carry, then the final check.

    Worth pinning because it is where the generalisation is easiest to get wrong -- the
    code that assumes there is always a box behind the station would index past the end.
    """

    env = _relay_env(ROOT / "configs" / "cookie_same_column.yaml")
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = RelayBatchExpert(env)
        expert.reset()
        assert len(expert.boxes) == 1
        assert expert.stage == "FILL"
        _finish_fill(expert, list(range(10)))
        expert.act()
        assert expert.stage == "VERIFY_ALL", "a lone box has nothing to make room for"
        with pytest.raises(IndexError, match="no box B"):
            _ = expert.box_b
    finally:
        env.close()


def test_the_push_distance_is_a_parameter_with_the_measured_default():
    """It is not the derived lane pitch, and the reason is on the field.

    90 mm is what the push and the carry were calibrated against; the derived pitch for
    this box is 87 mm, which would also satisfy the 85 mm clearance but would invalidate
    a yield that took an 80-minute run to measure.
    """

    config = load_config(RELAY_CONFIG)
    assert config.relay_push_distance_m == pytest.approx(0.090)
    assert config.relay_push_distance_m > RelayBatchExpert.PUSHED_CLEAR_M
    with pytest.raises(ValueError, match="relay_push_distance_m must be positive"):
        load_config(RELAY_CONFIG).__class__(
            **{**load_config(RELAY_CONFIG).__dict__, "relay_push_distance_m": 0.0}
        )
