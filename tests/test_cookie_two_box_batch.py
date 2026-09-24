"""Independent dual-box layout and coordinator checks."""

from dataclasses import replace
from pathlib import Path

from types import SimpleNamespace

import mujoco
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.two_box_batch import (
    RightBoxCarryController,
    RightBoxPushController,
    TwoBoxBatchExpert,
)


ROOT = Path(__file__).resolve().parents[1]
DUAL_CONFIG = ROOT / "configs" / "cookie_two_box_batch.yaml"

# These tests isolate a single controller against a known layout.  The two-box
# scene randomises its boxes and the spare box's yaw between episodes, which would
# make an assertion about absolute pose or reach depend on the seed, so every reset
# here turns both switches off and gets the nominal layout back.
FIXED_SCENE = {"randomize_cookies": False, "randomize_scene": False}
SINGLE_CONFIG = ROOT / "configs" / "cookie_batch.yaml"


def test_dual_box_config_is_separate_from_single_box_reference():
    single = load_config(SINGLE_CONFIG).cookie_transfer
    dual = load_config(DUAL_CONFIG).cookie_transfer
    assert single.spare_target_bin_world_position_m is None
    assert dual.spare_target_bin_world_position_m is not None
    assert dual.target_bin_half_size_m[0] == single.target_bin_half_size_m[0]
    assert dual.target_bin_half_size_m[1] < single.target_bin_half_size_m[1]
    assert dual.cookie_source_positions_m == single.cookie_source_positions_m
    assert dual.cookie_edge_bevel_m == single.cookie_edge_bevel_m


def test_dual_box_coordinator_starts_with_a_and_counts_boxes_independently():
    env = A3CookieTransferEnv(DUAL_CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = TwoBoxBatchExpert(env)
        expert.reset()
        assert expert.stage == "FILL_A"
        assert expert.box_a != expert.box_b
        assert env._target_bin_body == expert.box_a
        assert expert.counts() == (0, 0)
        assert env._target_bin_body == expert.box_a
        for name in ("target_bin_free", "spare_target_bin_free"):
            assert env.model.joint(name).type == mujoco.mjtJoint.mjJNT_FREE
    finally:
        env.close()


def test_right_pusher_can_reach_both_ends_of_both_lanes():
    env = A3CookieTransferEnv(DUAL_CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        for name, destination in (("target_bin", 0.120), ("spare_target_bin", 0.030)):
            pusher = RightBoxPushController(env, env.model.body(name).id, destination)
            for y in (pusher.start_y, destination - 0.040):
                q = pusher.helper.solve(np.array([pusher.x, y, 0.755]), pusher.quat)
                assert np.all(np.isfinite(q))
    finally:
        env.close()


def test_right_pusher_clears_a_without_moving_spare_box():
    config = load_config(DUAL_CONFIG)
    config = replace(
        config,
        cookie_transfer=replace(
            config.cookie_transfer,
            cookie_source_positions_m=config.cookie_transfer.cookie_source_positions_m[:20],
        ),
    )
    env = A3CookieTransferEnv(
        config,
        render_cameras=False,
        task_config=CookieTransferTaskConfig(cookie_count=20, terminate_on_success=False),
    )
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        b_id = env.model.body("spare_target_bin").id
        b_initial = env.data.xpos[b_id].copy()
        box_id = env.model.body("target_bin").id
        pusher = RightBoxPushController(env, box_id, 0.120)
        for _ in range(850):
            _, _, terminated, truncated, info = env.step(pusher.act())
            if pusher.done or pusher.failed or terminated or truncated:
                break
        assert not terminated, info
        assert not truncated, info
        assert pusher.failed is None, pusher.failed
        assert pusher.done
        assert env.data.xpos[box_id, 1] >= 0.115
        np.testing.assert_allclose(env.data.xpos[b_id, :2], b_initial[:2], atol=1e-4)
    finally:
        env.close()


def test_right_gripper_carries_spare_box_into_station_without_rotation():
    """Isolate the physical B carry; A is pre-cleared only in this test setup."""
    config = load_config(DUAL_CONFIG)
    config = replace(
        config,
        physics_hz=1000,
        cookie_transfer=replace(
            config.cookie_transfer,
            cookie_source_positions_m=config.cookie_transfer.cookie_source_positions_m[:20],
        ),
    )
    env = A3CookieTransferEnv(
        config,
        render_cameras=False,
        task_config=CookieTransferTaskConfig(cookie_count=20, terminate_on_success=False),
    )
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        a_joint = env.model.joint("target_bin_free")
        env.data.qpos[a_joint.qposadr + 1] = 0.105
        mujoco.mj_forward(env.model, env.data)
        pusher = RightBoxCarryController(
            env, env.model.body("spare_target_bin").id, 0.030
        )
        for y in (pusher.anchor[1], pusher.anchor[1] + 0.160):
            target = pusher.anchor.copy()
            target[1] = y
            assert np.all(np.isfinite(pusher.helper.solve(target, pusher.quat)))
        for step in range(850):
            _, _, terminated, truncated, info = env.step(pusher.act())
            if pusher.done or pusher.failed or terminated or truncated:
                break
        assert not terminated, info
        assert not truncated, info
        assert pusher.failed is None, pusher.failed
        assert pusher.done
        assert np.linalg.norm(pusher.box_position[:2] - [0.075, 0.030]) < 0.008
        # The box must arrive facing the station, not spun.  The bound is 5 deg
        # because the measured residual depends on how rigidly the arm holds its
        # commanded pose: without actuator `kv` damping the carry left 2.1 deg of
        # yaw and 7.3 mm of x error, and with it 4.0 deg and 1.5 mm.  The pinch
        # is what trades one for the other -- a grip that resists the box's twist
        # also stops it sliding sideways off the station -- so the angular bound
        # is set from the measurement that produced the better placement.
        assert abs(np.rad2deg(pusher.box_yaw_rad)) < 5.0
        env._target_bin_body = pusher.box_id
        filler = A3SameColumnBatchExpert(env)
        filler.reset()  # The left arm must be able to work at the pushed-in pose.
    finally:
        env.close()


def test_fast_exchange_geometry_has_clearance_and_reachable_b_slots():
    """A synthetic pose check in seconds; it is not an end-to-end rollout."""
    env = A3CookieTransferEnv(DUAL_CONFIG, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        a_id = env.model.body("target_bin").id
        b_id = env.model.body("spare_target_bin").id
        a_addr = int(env.model.joint("target_bin_free").qposadr[0])
        b_addr = int(env.model.joint("spare_target_bin_free").qposadr[0])
        env.data.qpos[a_addr:a_addr + 3] = [0.060, 0.120, 0.7493]
        env.data.qpos[b_addr:b_addr + 3] = [0.0732, 0.0251, 0.7494]
        yaw = np.deg2rad(0.5)
        env.data.qpos[b_addr + 3:b_addr + 7] = [
            np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
        ]
        mujoco.mj_forward(env.model, env.data)
        env._target_bin_body = b_id
        half_y = env.config.cookie_transfer.target_bin_half_size_m[1]
        wall = env.config.cookie_transfer.bin_wall_thickness_m
        edge_gap = env.data.xpos[a_id, 1] - env.data.xpos[b_id, 1] - 2 * (half_y + wall)
        assert edge_gap > 0.015
        assert mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "box_exchange_guide_left"
        ) == -1
        filler = A3SameColumnBatchExpert(env)
        filler.reset()
        work = mujoco.MjData(env.model)
        work.qpos[:] = env.data.qpos
        # The batch that fills the *far* column: the place pose is per batch now, so
        # asking for it by column means asking for the group whose column that is.
        far_column_batch = next(
            group for group in filler.plan.groups if group.column == 1
        )
        for clearance in (0.0, 0.085):
            pos, rot = filler._place_pose(clearance, far_column_batch)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rot.ravel())
            work.qpos[filler._l_qpos] = filler._solve_l(pos, quat, filler.q_transit)
            mujoco.mj_forward(env.model, work)
            assert not any(
                a_id in (
                    env.model.geom_bodyid[contact.geom1],
                    env.model.geom_bodyid[contact.geom2],
                )
                and any(
                    env.model.geom(geom_id).name.startswith("L_")
                    for geom_id in (contact.geom1, contact.geom2)
                )
                for contact in work.contact[:work.ncon]
            )
    finally:
        env.close()


def _push_a_clear_and_fill(env, expert) -> list[int]:
    """Put the relay in the state `VERIFY_B` is meant to see.

    Box A is placed clear of the station rather than pushed there -- the push is not
    what these tests are about, and doing it physically would cost minutes.  The
    order matters: the box moves first and the Cookies are laid into it afterwards,
    because placing them first leaves them behind at the station, which both empties
    A and gives box B something to collide with.
    """

    station = np.asarray(env.config.cookie_transfer.target_bin_world_position_m[:2])
    joint = env.model.body_jntadr[expert.box_a]
    address = env.model.jnt_qposadr[joint]
    env.data.qpos[address] = station[0]
    env.data.qpos[address + 1] = station[1] + 0.090
    env.data.qpos[address + 2] = 0.753
    env.data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)

    indices = list(range(10))
    _fill_box(env, expert, indices)
    return indices


def _fill_box(env, expert, indices: list[int]) -> None:
    """Lay `indices` Cookies into box A where it currently sits.

    Separate from the placement above because anything that moves the box also
    carries its Cookies out of it, so the two steps have to be repeatable in the
    right order rather than fused into one helper.
    """

    previous = env._target_bin_body
    env._target_bin_body = expert.box_a
    try:
        for index in indices:
            position = env.privileged_target_slot_world(index, env._target_cookie_center_z)
            env.set_cookie_pose(index, tuple(np.asarray(position[:3], dtype=float)))
    finally:
        env._target_bin_body = previous
    mujoco.mj_forward(env.model, env.data)


def _place_box_b(env, station: np.ndarray, offset_m: float) -> None:
    joint = env.model.body_jntadr[env._spare_bin_body]
    address = env.model.jnt_qposadr[joint]
    env.data.qpos[address] = station[0]
    env.data.qpos[address + 1] = station[1] + offset_m
    env.data.qpos[address + 2] = 0.753
    env.data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_verify_b_never_parks_between_its_pass_and_fail_thresholds():
    """A box between 8 mm and 12 mm from the station must not stall the relay.

    `VERIFY_B` used to pass within 8 mm and fail only beyond 12 mm, so a carry that
    stopped in between satisfied neither branch and the stage waited until the
    horizon ran out -- seen as a run reaching 12000 steps where a successful one
    takes 4798.  `VERIFY_A` had the same shape (pass at 85 mm of clearance, fail
    below 50 mm).  A bounded settle window replaces the second threshold, so every
    state now either settles or times out.

    Swept across the old dead zone and the ranges on either side of it, since only
    the middle distances used to stall.
    """

    for offset_mm in (2.0, 7.0, 8.5, 10.0, 11.5, 16.0):
        env = A3CookieTransferEnv(
            DUAL_CONFIG,
            render_cameras=False,
            task_config=CookieTransferTaskConfig(
                require_exact_slots=False, require_released=True, terminate_on_success=False
            ),
        )
        try:
            env.reset(seed=0, options=FIXED_SCENE)
            expert = TwoBoxBatchExpert(env)
            expert.reset()
            station = np.asarray(env.config.cookie_transfer.target_bin_world_position_m[:2])
            indices = _push_a_clear_and_fill(env, expert)
            _place_box_b(env, station, offset_mm / 1000.0)

            expert.stage = "VERIFY_B"
            expert._settled_steps = 0
            expert._verify_steps = 0
            expert._a_indices = indices
            expert.pusher = SimpleNamespace(done=True)

            resolved = False
            for _ in range(TwoBoxBatchExpert.VERIFY_SETTLE_STEPS + 10):
                env.step(expert.act())
                if expert.failed or expert.stage != "VERIFY_B":
                    resolved = True
                    break
            assert resolved, (
                f"box B at {offset_mm} mm parked in VERIFY_B: neither the pass "
                f"condition nor the failure fired"
            )
            if offset_mm >= 8.0:
                assert expert.failed is not None, (
                    "a box beyond the pass threshold has to be reported as a failure"
                )
                assert f"{offset_mm:.1f} mm" in expert.failed, (
                    f"the failure should quote the measured distance, got: {expert.failed}"
                )
            else:
                assert expert.stage == "FILL_B", (
                    f"a box inside the pass threshold should advance, got {expert.stage}"
                )
        finally:
            env.close()


def test_verify_a_never_parks_when_a_overshoots_its_failure_threshold():
    """Box A pushed 50-85 mm used to stall `VERIFY_A` the same way.

    The pass condition was 85 mm of clearance and the failure fired only below
    50 mm, so an A that ended up in between parked the stage.
    """

    env = A3CookieTransferEnv(
        DUAL_CONFIG,
        render_cameras=False,
        task_config=CookieTransferTaskConfig(
            require_exact_slots=False, require_released=True, terminate_on_success=False
        ),
    )
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = TwoBoxBatchExpert(env)
        expert.reset()
        station = np.asarray(env.config.cookie_transfer.target_bin_world_position_m[:2])
        indices = _push_a_clear_and_fill(env, expert)
        # Move A into the old dead band -- clear, but not clear enough -- and put
        # its Cookies back in place afterwards, since moving the box carries them
        # out of it again.
        joint = env.model.body_jntadr[expert.box_a]
        address = env.model.jnt_qposadr[joint]
        env.data.qpos[address + 1] = station[1] + 0.065
        mujoco.mj_forward(env.model, env.data)
        _fill_box(env, expert, indices)

        expert.stage = "VERIFY_A"
        expert._settled_steps = 0
        expert._verify_steps = 0
        expert._a_indices = indices
        expert.pusher = SimpleNamespace(done=True)

        for _ in range(TwoBoxBatchExpert.VERIFY_SETTLE_STEPS + 10):
            env.step(expert.act())
            if expert.failed:
                break
        assert expert.failed is not None, (
            "box A at 65 mm of clearance parked in VERIFY_A instead of being reported"
        )
        assert "65.0 mm" in expert.failed, (
            f"the failure should quote the measured clearance, got: {expert.failed}"
        )
    finally:
        env.close()
