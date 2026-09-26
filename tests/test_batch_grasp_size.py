"""A fill whose grasp size is a parameter, and the plan that follows from it.

Phase 3 of `docs/configurable-scenes.md`.  What is pinned here is the *plan* the
fill follows and the checks that keep it honest: a grasp cannot be longer than the
column it is placed into, a batch cannot span two columns, and the shipped
``per_grasp`` is what the shipped scenes have always done.

The simulator-level claim -- that ``per_grasp = 5`` is step-identical to the fill
before this existed -- is not a unit test.  It is checked by hashing the action
stream of a whole episode against the previous commit, and the method is recorded
in `docs/configurable-scenes.md`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.scene_spec import two_box_spec

ROOT = Path(__file__).resolve().parents[1]
SINGLE_CONFIG = ROOT / "configs" / "cookie_same_column.yaml"
RELAY_CONFIG = ROOT / "configs" / "cookie_two_box_batch.yaml"
FIXED_SCENE = {"randomize_cookies": False, "randomize_scene": False}


def _plan_for(config_path: Path, per_grasp: int):
    """The plan a scene derives at a given grasp size, without building the model.

    The verified-grasp floor is lowered on purpose: these tests are about the plan,
    which derives correctly for every size, and the floor exists because the *grasp*
    has only been measured at five.  Saying so here keeps the two claims separate --
    see `SimConfig.min_verified_batch_expert_per_grasp`.
    """

    config = replace(
        load_config(config_path),
        batch_expert_per_grasp=per_grasp,
        min_verified_batch_expert_per_grasp=1,
    )
    env = A3CookieTransferEnv(config, render_cameras=False)
    try:
        return env.batch_plan
    finally:
        env.close()


def _env_for(config_path: Path, per_grasp: int, **overrides) -> A3CookieTransferEnv:
    """A scene at a grasp size whose *plan* is under test rather than its grasp."""

    config = replace(
        load_config(config_path),
        batch_expert_per_grasp=per_grasp,
        min_verified_batch_expert_per_grasp=1,
        **overrides,
    )
    return A3CookieTransferEnv(config, render_cameras=False)


# ------------------------------------------------------------------- the config


def test_the_shipped_grasp_size_is_five_and_every_scene_agrees():
    """Five is what both batch experts were calibrated on, so it stays the default."""

    for path in (SINGLE_CONFIG, RELAY_CONFIG):
        config = load_config(path)
        assert config.batch_expert_per_grasp == 5


def test_a_grasp_longer_than_its_column_is_refused_at_load():
    """A batch goes into one column, so it cannot be longer than that column.

    Checked in `SimConfig.__post_init__` rather than in the expert because it is a
    property of the box and the grasp size together, and it is knowable from the
    config alone -- the same reason the nominal box gap is checked there.  The
    message names both numbers, because which of them to change is the reader's
    call.
    """

    raw = SINGLE_CONFIG.read_text(encoding="utf-8")
    broken = raw + "\nbatch_expert_per_grasp: 6\n"
    path = SINGLE_CONFIG.parent / "tmp_too_many_per_grasp.yaml"
    path.write_text(broken, encoding="utf-8")
    try:
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        message = str(excinfo.value)
        assert "batch_expert_per_grasp is 6" in message
        assert "holds 5 rows" in message
        assert "use at most 5" in message
    finally:
        path.unlink()


def test_a_grasp_of_zero_is_refused():
    from a3_dual_arm_sim.config import SimConfig

    with pytest.raises(ValueError, match="must be at least 1"):
        SimConfig(batch_expert_per_grasp=0)


# ---------------------------------------------------------------------- the plan


def test_the_shipped_plan_is_one_batch_per_column_at_the_shipped_size():
    """The anchor, in the form the fill reads it."""

    plan = _plan_for(SINGLE_CONFIG, 5)
    assert plan.rows == 5
    assert plan.columns == 2
    assert plan.capacity == 10
    assert [(group.column, group.size) for group in plan.groups] == [(0, 5), (1, 5)]
    # And the slots are the ones the expert places into: column-major, so column 0
    # is the even indices of the row-major lattice.
    assert plan.groups[0].slot_indices == (0, 2, 4, 6, 8)
    assert plan.groups[1].slot_indices == (1, 3, 5, 7, 9)


def test_a_smaller_grasp_makes_more_batches_within_the_same_columns():
    """Ten Cookies at three per grasp is four batches, not three and a remainder.

    The remainder is per column, which is the whole point: a batch cannot span two
    columns, so column 0 takes a three and a two and column 1 does the same.  A plan
    that spread the remainder across the box would give ``[3, 3, 3, 1]`` and leave
    the second batch of three with nowhere to go.
    """

    plan = _plan_for(SINGLE_CONFIG, 3)
    assert [(group.column, group.size) for group in plan.groups] == [
        (0, 3),
        (0, 2),
        (1, 3),
        (1, 2),
    ]
    assert plan.placed == plan.capacity == 10
    for group in plan.groups:
        assert {index % plan.columns for index in group.slot_indices} == {group.column}


def test_a_smaller_grasp_is_refused_until_its_grasp_is_measured():
    """The bound that makes the parameter honest, with the measurement in the message.

    The plan derives correctly for every size from 1 to 10 -- a test partitions all of
    them -- so this is not a statement about the plan.  It is a statement about the
    grasp: measured at three seeds each, five places all ten Cookies in 483-543 steps
    while one, two and three all fail at their first batch with the pads closed to the
    floor and at most 0.06 N of contact.  Declaring the parameter configurable down
    to one would mean collecting episodes at a yield nobody has measured, which is
    exactly what the bounds are for.
    """

    for per_grasp in (1, 2, 3, 4):
        with pytest.raises(ValueError) as excinfo:
            replace(load_config(SINGLE_CONFIG), batch_expert_per_grasp=per_grasp)
        message = str(excinfo.value)
        assert f"batch_expert_per_grasp is {per_grasp}" in message
        assert "0.06 N of pad contact" in message
        assert "483-543" in message
    # Five and above pass this check (a higher value may still fail the column bound).
    replace(load_config(SINGLE_CONFIG), batch_expert_per_grasp=5)


def test_the_verified_floor_is_what_the_shipped_scenes_use():
    """No shipped scene asks for a grasp size below the measured floor."""

    for path in (SINGLE_CONFIG, RELAY_CONFIG):
        config = load_config(path)
        assert config.batch_expert_per_grasp >= config.min_verified_batch_expert_per_grasp


def test_a_remainder_batch_is_refused_because_it_is_below_the_floor():
    """The floor applies to the plan's groups, not only to the parameter.

    A column whose row count is not a multiple of the grasp size ends in a shorter
    batch, and that remainder is below the grasp size by construction -- so with a
    measured floor of five it is always below the floor, whatever the parameter says.
    The check used to look only at the parameter, which let a box of capacity 14
    through with the plan ``[5, 2, 5, 2]``.

    Measured rather than inferred, by the Phase 7 matrix: that box was run and died at
    "batch 2 CLOSE timed out after 421 steps; pad forces=[0.0, 0.0] N", the same
    no-contact failure the floor's message describes for small grasps.
    """

    base = load_config(RELAY_CONFIG)
    for capacity, plan in ((11, "[5, 1, 5, 1]"), (14, "[5, 2, 5, 2]")):
        slots = replace(two_box_spec(), box_capacity=capacity).target_slots_local_m
        with pytest.raises(ValueError) as excinfo:
            replace(
                base,
                cookie_transfer=replace(
                    base.cookie_transfer,
                    target_slots_local_m=slots,
                ),
            )
        message = str(excinfo.value)
        assert f"is {plan}" in message
        assert "below the 5 that have a measured grasp" in message
        # The message has to say what would work, or it is only a refusal.
        assert "a multiple of the grasp size" in message
        assert "10 slots for this column count" in message
    # A box whose rows *are* a multiple of the grasp size passes this check.
    slots = replace(two_box_spec(), box_capacity=10).target_slots_local_m
    replace(
        base,
        cookie_transfer=replace(base.cookie_transfer, target_slots_local_m=slots),
    )


def test_a_batch_of_one_is_a_plan_the_config_derives():
    """`per_grasp = 1` derives a legal plan, whatever the grasp makes of it.

    Ten batches of one, five in each column, and every batch inside its own column.
    Whether a single Cookie can be grasped this way is the separate question the floor
    above answers, and the two claims are deliberately kept apart: this one is about
    the plan, and the floor is about the physics.
    """

    plan = _plan_for(SINGLE_CONFIG, 1)
    assert [group.size for group in plan.groups] == [1] * 10
    assert plan.rows == 5
    assert plan.placed == 10
    for group in plan.groups:
        assert {index % plan.columns for index in group.slot_indices} == {group.column}


def test_the_plan_is_read_off_the_lattice_rather_than_configured():
    """The column count comes from the slots, so it cannot drift from them.

    `batch_plan.columns` is `len({slot[0] for slot in TARGET_SLOTS_LOCAL})`.  A
    config that changed its lattice would change the plan with it, and there is no
    second number to keep in step.
    """

    env = A3CookieTransferEnv(SINGLE_CONFIG, render_cameras=False)
    try:
        lattice_columns = len({slot[0] for slot in env.TARGET_SLOTS_LOCAL})
        assert env.batch_plan.columns == lattice_columns == 2
        assert env.batch_plan.capacity == len(env.TARGET_SLOTS_LOCAL) == 10
    finally:
        env.close()


# ------------------------------------------------------- what the expert reads


def test_the_expert_reads_the_scene_s_plan_and_grasp_size():
    """One source of truth: the expert takes both from the environment."""

    env = A3CookieTransferEnv(SINGLE_CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        assert expert.per_grasp == env.config.batch_expert_per_grasp == 5
        assert expert.plan is env.batch_plan
        assert expert._final_batch == len(env.batch_plan.groups) - 1 == 1
        assert expert._is_later_batch() is False, "the first batch is not a later one"
    finally:
        env.close()


def test_a_later_batch_is_told_apart_from_the_first_one():
    """The split the same-column tuning is written against, named rather than spelled.

    It used to be ``batch_index == 0`` in one place and ``batch_index == 1`` in four,
    which for the shipped two-batch plan are the same two statements.  Naming it is
    what lets a four-batch plan say which batches get the neighbour logic -- and the
    docstring records that *that* generalisation is untested, because the anchor only
    pins the two-batch case.
    """

    env = A3CookieTransferEnv(SINGLE_CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        assert expert._is_later_batch() is False
        expert.batch_index = 1
        assert expert._is_later_batch() is True
    finally:
        env.close()


def test_each_batch_gets_its_own_size_from_the_plan():
    """A short batch takes fewer Cookies, which is what keeps the plan placeable."""

    env = _env_for(SINGLE_CONFIG, 3)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        sizes = []
        for batch_index in range(len(env.batch_plan.groups)):
            expert.batch_index = batch_index
            sizes.append(expert._next_batch_size())
        assert sizes == [3, 2, 3, 2]
    finally:
        env.close()


def test_the_place_pose_aims_at_the_batch_s_own_slots():
    """Not at the column's centre, which only coincides when a batch fills a column.

    For the shipped plan the two are the same point -- a batch that fills a whole
    column has that column's centre as its own mean -- so this change is invisible to
    the shipped scenes.  For a two-batch column they differ, and using the centre for
    both would drop the second batch in the wrong place.
    """

    import numpy as np

    env = _env_for(SINGLE_CONFIG, 3)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        box = env.data.xpos[expert._tb_id]
        rotation = env.data.xmat[expert._tb_id].reshape(3, 3)

        aims = []
        for group in expert.plan.groups:
            position, _ = expert._place_pose(0.0, group)
            # Undo the tool offset to recover the slot point the pose aims at.
            tool_r = rotation @ expert._canonical
            local = rotation.T @ (position + tool_r @ expert._group_offset - box)
            aims.append(local[:2])

        # The two batches of column 0 aim at different y, which is the property.
        assert aims[0][0] == pytest.approx(aims[1][0]), "same column, same x"
        assert aims[0][1] != pytest.approx(aims[1][1])
        for group, aim in zip(expert.plan.groups, aims, strict=True):
            batch_slots = slots[np.asarray(group.slot_indices)]
            assert aim[0] == pytest.approx(batch_slots[:, 0].mean())
            assert aim[1] == pytest.approx(batch_slots[:, 1].mean())
    finally:
        env.close()


def test_the_shipped_place_pose_is_the_column_centre_it_always_was():
    """The anchor for `_place_pose`, computed both ways.

    A batch that fills a whole column has that column's centre as the mean of its own
    slots, so the old per-column formula and the new per-batch one are the same
    number.  This is the check that says so, rather than a comment claiming it.
    """

    import numpy as np

    env = A3CookieTransferEnv(SINGLE_CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        for group in expert.plan.groups:
            # What the expert used to do: the column's x, the column's mean y.
            column_x = np.unique(slots[:, 0])[group.column]
            old_center = np.array(
                [
                    column_x,
                    slots[np.isclose(slots[:, 0], column_x), 1].mean(),
                ]
            )
            batch_slots = slots[np.asarray(group.slot_indices)]
            new_center = np.array([batch_slots[:, 0].mean(), batch_slots[:, 1].mean()])
            np.testing.assert_allclose(new_center, old_center, atol=1e-12)
    finally:
        env.close()


def test_the_reachability_check_covers_every_batch_not_every_column():
    """A plan with two batches in one column has two poses to check, not one.

    For the shipped plan this is the two columns at both clearances it has always
    been -- which is why the anchor holds here too -- and the error message names the
    batch as well as the column, so a refusal says which grasp could not be placed.
    """

    env = _env_for(SINGLE_CONFIG, 3)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3SameColumnBatchExpert(env)
        expert.reset()  # runs the check; four batches must all be reachable
        assert len(expert.plan.groups) == 4
    finally:
        env.close()


def test_an_unreachable_layout_names_the_batch_it_refused():
    """The message has to say which grasp could not be placed.

    A layout can be reachable for one batch of a column and not for another, so
    "target column 2 unreachable" would leave the reader guessing which of its two
    poses was the problem.
    """

    import re

    base = load_config(SINGLE_CONFIG)
    config = replace(
        base,
        cookie_transfer=replace(
            base.cookie_transfer, target_bin_world_position_m=(0.135, -0.075, 0.753)
        ),
    )
    env = A3CookieTransferEnv(config, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        with pytest.raises(RuntimeError) as excinfo:
            A3SameColumnBatchExpert(env).reset()
        message = str(excinfo.value)
        assert "unreachable" in message
        assert re.search(r"target column [12] \(batch [12] of 2\)", message), message
    finally:
        env.close()
