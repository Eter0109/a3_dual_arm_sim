"""Independent dual-box layout and coordinator checks."""

from dataclasses import replace
from pathlib import Path

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
        assert abs(np.rad2deg(pusher.box_yaw_rad)) < 3.0
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
        for clearance in (0.0, 0.085):
            pos, rot = filler._place_pose(clearance, column_index=1)
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
