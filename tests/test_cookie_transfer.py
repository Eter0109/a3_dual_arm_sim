from __future__ import annotations

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv


def test_cookie_scene_has_central_stand_bins_and_eighty_upright_cookies() -> None:
    env = A3CookieTransferEnv(render_cameras=False)
    try:
        for name in ("stand_mast", "source_bin_floor", "target_bin_floor"):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        for index in range(80):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"cookie_{index}") >= 0
        # Adjacent walls overlap at every corner, rather than merely meeting edge-to-edge.
        for bin_name in ("source_bin", "target_bin"):
            front = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{bin_name}_front_wall")
            left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{bin_name}_left_wall")
            front_center = env.model.geom_pos[front]
            front_half = env.model.geom_size[front]
            left_center = env.model.geom_pos[left]
            left_half = env.model.geom_size[left]
            assert np.all(
                np.minimum(front_center[:2] + front_half[:2], left_center[:2] + left_half[:2])
                > np.maximum(front_center[:2] - front_half[:2], left_center[:2] - left_half[:2])
            )
            front_visual = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{bin_name}_front_wall_visual"
            )
            left_visual = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{bin_name}_left_wall_visual"
            )
            visual_overlap = np.minimum(
                env.model.geom_pos[front_visual, :2] + env.model.geom_size[front_visual, :2],
                env.model.geom_pos[left_visual, :2] + env.model.geom_size[left_visual, :2],
            ) - np.maximum(
                env.model.geom_pos[front_visual, :2] - env.model.geom_size[front_visual, :2],
                env.model.geom_pos[left_visual, :2] - env.model.geom_size[left_visual, :2],
            )
            assert np.any(visual_overlap <= 1e-12)
            np.testing.assert_allclose(
                env.model.geom_rgba[front_visual], env.model.geom_rgba[left_visual]
            )
        collision = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_0_geom")
        first_visual = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_0_visual")
        second_visual = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_1_visual")
        assert env.model.geom_type[collision] == mujoco.mjtGeom.mjGEOM_MESH
        assert env.model.geom_type[first_visual] == mujoco.mjtGeom.mjGEOM_MESH
        assert env.model.geom_dataid[collision] != env.model.geom_dataid[first_visual]
        assert not np.allclose(
            env.model.geom_rgba[first_visual], env.model.geom_rgba[second_visual]
        )
    finally:
        env.close()


def test_cookie_reset_is_deterministic_and_starts_outside_target() -> None:
    env = A3CookieTransferEnv(render_cameras=False)
    try:
        _, first = env.reset(seed=11)
        positions = env.cookie_positions.copy()
        _, second = env.reset(seed=11)
        np.testing.assert_allclose(env.cookie_positions, positions)
        assert first["cookies_in_target"] == second["cookies_in_target"] == 0
        assert first["cookies_in_source"] == second["cookies_in_source"] == 80
        assert first["source_initially_filled"]
        assert second["source_initially_filled"]
        assert np.all(
            env.cookie_positions[:, 2] - env.COOKIE_HALF_SIZE[2]
            >= env.SOURCE_FLOOR_TOP_Z - 0.002
        )
        assert all(
            abs(float(env.data.xmat[env._cookie_bodies[index]].reshape(3, 3)[2, 2]))
            >= np.cos(env.task_config.max_tilt_rad)
            for index in range(env.task_config.cookie_count)
        )
    finally:
        env.close()




def test_dense_dimensions_capacity_walls_and_mirrored_home() -> None:
    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        scene = env.config.cookie_transfer
        np.testing.assert_allclose(2 * env.COOKIE_HALF_SIZE, [0.05, 0.019 / 3, 0.025])
        positions = np.asarray(env.SOURCE_POSITIONS)
        xs, ys = np.unique(positions[:, 0]), np.unique(positions[:, 1])
        assert (len(xs), len(ys), len(positions)) == (4, 20, 80)
        np.testing.assert_allclose(np.diff(xs) - 0.05, 0.0004)
        np.testing.assert_allclose(np.diff(ys) - 0.019 / 3, 0.0004)
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        assert (len(np.unique(slots[:, 0])), len(np.unique(slots[:, 1]))) == (2, 5)
        np.testing.assert_allclose(
            np.ptp(positions, axis=0) + 2 * env.COOKIE_HALF_SIZE[:2] + 0.006,
            2 * env.SOURCE_INNER_HALF_SIZE,
        )
        np.testing.assert_allclose(
            np.ptp(slots, axis=0) + 2 * env.COOKIE_HALF_SIZE[:2],
            2 * env.TARGET_INNER_HALF_SIZE,
        )
        floor_top = scene.source_floor_z_m + scene.bin_wall_thickness_m / 2
        assert np.isclose(env.SOURCE_WALL_TOP_Z - floor_top, 1.2 * 0.025)
        assert np.isclose(
            scene.target_bin_wall_height_m
            - (scene.target_floor_z_m + scene.bin_wall_thickness_m / 2),
            0.025,
        )
        for name in ("SHOULDER_Y_S", "ELBOW_P_S", "WRIST_P_S", "flange"):
            np.testing.assert_allclose(
                env.data.body(f"R_{name}").xpos,
                env.data.body(f"L_{name}").xpos * [1, -1, 1],
                atol=1e-5,
            )
        np.testing.assert_allclose(scene.source_bin_center_m, [0.175, 0.315])
        target_world = slots + np.asarray(scene.target_bin_world_position_m[:2])
        np.testing.assert_allclose(np.unique(target_world[:, 0]), xs[:2])
        row_offsets = (target_world[:, 1] - ys[0]) / (ys[1] - ys[0])
        np.testing.assert_allclose(row_offsets, np.round(row_offsets), atol=1e-10)
    finally:
        env.close()


def test_success_requires_exact_settled_upright_two_by_five_fill() -> None:
    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        for index, (x, y) in enumerate(env.TARGET_SLOT_CENTERS):
            env.set_cookie_pose(index, (float(x), float(y), 0.791))
        for _ in range(60):
            mujoco.mj_step(env.model, env.data)
        terminated = False
        info = {}
        for _ in range(env.task_config.success_hold_steps):
            _, _, terminated, _, info = env.step(env.DEPLOYMENT_HOME)
        assert terminated
        assert info["success"]
        assert info["cookies_in_target"] == 10
        assert info["cookies_in_source"] == 70
        assert info["exact_2x5_fill"]
        assert info["target_touches_all_walls"]
        assert all(index >= 0 for index in info["target_slot_occupancy"])
        assert info["success_hold_count"] >= env.task_config.success_hold_steps
    finally:
        env.close()


@pytest.mark.xfail(
    reason="Single-cookie sequential expert superseded by batch experts on dense 80-cookie layout"
)
def test_cookie_transfer_expert_runs_feedback_state_machine() -> None:
    from a3_dual_arm_sim.expert import A3CookieTransferExpert, CookiePhase

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        expert = A3CookieTransferExpert(env)
        obs, info = env.reset(seed=0, options={"randomize_cookies": False})
        expert.reset()
        step = 0
        while not expert.failed and not expert.completed_cookie_indices and step < 700:
            action = expert.act(obs)
            obs, _, _, _, info = env.step(action)
            step += 1
        assert not hasattr(expert, "actions")
        assert not expert.failed, expert.failure_reason
        assert info["cookies_in_target"] >= 1
        assert info["cookies_in_source"] <= 79
        for phase in (
            CookiePhase.VERIFY_GRASP,
            CookiePhase.VERIFY_LIFT,
            CookiePhase.VERIFY_RELEASE,
        ):
            assert phase in expert.transition_history
        assert len(expert.completed_cookie_indices) >= 1
    finally:
        env.close()


@pytest.mark.xfail(
    reason="Single-cookie sequential expert superseded by batch experts on dense 80-cookie layout"
)
def test_cookie_transfer_expert_replans_after_failed_lift_verification() -> None:
    from a3_dual_arm_sim.expert import A3CookieTransferExpert, CookiePhase

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        expert = A3CookieTransferExpert(env)
        obs, _ = env.reset(seed=0, options={"randomize_cookies": False})
        expert.reset()
        for _ in range(700):
            action = expert.act(obs)
            obs, _, _, _, _ = env.step(action)
            if expert.phase is CookiePhase.VERIFY_LIFT:
                break
        assert expert.phase is CookiePhase.VERIFY_LIFT

        cookie_index = expert.current_cookie_index
        assert cookie_index is not None
        source_position = expert._initial_cookie_positions[cookie_index].copy()
        env.set_cookie_pose(cookie_index, tuple(source_position))
        expert.phase_steps = 29

        expert.act(obs)

        assert expert.phase is CookiePhase.APPROACH
        assert expert.retry_counts[expert.target_slot_index] == 1
        assert "did not follow" in expert.retry_reasons[-1]
    finally:
        env.close()


def test_cookie_transfer_evaluation_labels_step_limit() -> None:
    from a3_dual_arm_sim.evaluation import run_cookie_transfer_episode

    result = run_cookie_transfer_episode(0, max_steps=1, randomize_cookies=False)

    assert not result.success
    assert result.failure_phase == "SUPPORT_BOX"
    assert result.reason == "max_steps_exceeded"
    assert result.failed_cookie is None


def test_idle_cartesian_hold_and_independent_target_bin() -> None:
    env = A3CookieTransferEnv(action_mode="cartesian_delta", render_cameras=False)
    try:
        env.reset(seed=0)
        target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_bin")
        position = env.data.xpos[target_id].copy()
        initial = env.last_applied_action.copy()
        action = np.zeros(14)
        action[6] = 2 * initial[7] - 1
        action[13] = 2 * initial[15] - 1
        for _ in range(500):
            _, _, terminated, _, info = env.step(action)
            assert not terminated
            np.testing.assert_allclose(info["applied_action"], initial, atol=1e-12)
        settled = env.current_joint_action.copy()
        for _ in range(100):
            env.step(action)
        np.testing.assert_allclose(env.current_joint_action, settled, atol=0.002)
        action[7] = 0.2
        for _ in range(10):
            env.step(action)
        assert env.model.body_parentid[target_id] == 0
        assert env.model.body_dofnum[target_id] == 6
        np.testing.assert_allclose(env.data.xpos[target_id], position, atol=0.001)
        np.testing.assert_allclose(env.last_applied_action[:7], initial[:7], atol=1e-12)
    finally:
        env.close()


def test_vertical_home_and_physical_right_box_support() -> None:
    from a3_dual_arm_sim.box_support import BoxSupportController

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        for side in ("L", "R"):
            tool_axis = env.data.body(f"{side}_robotiq_2f85").xmat.reshape(3, 3)[:, 2]
            assert tool_axis @ np.array([0.0, 0.0, -1.0]) > np.cos(np.deg2rad(2))
        support = BoxSupportController(env)
        for _ in range(700):
            env.step(support.act())
            assert support.failed is None
            if support.ready:
                break
        assert support.ready
        for _ in range(300):
            env.step(support.act())
        assert np.all(support.contact_forces() > 0.1)
        assert env.data.body("target_bin").xpos[2] > support.initial_height + 0.01
        tilt = np.arccos(np.clip(env.data.body("target_bin").xmat.reshape(3, 3)[2, 2], -1, 1))
        assert np.deg2rad(3) < tilt < np.deg2rad(18)
    finally:
        env.close()


def test_placement_timeout_holds_instead_of_sweeping_loaded_bin(monkeypatch) -> None:
    from a3_dual_arm_sim.expert import A3CookieTransferExpert, CookiePhase

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        expert = A3CookieTransferExpert(env)
        expert.reset()
        monkeypatch.setattr(expert.box_support, "contact_forces", lambda: np.ones(2))
        monkeypatch.setattr(expert, "_cookie_dropped", lambda: False)
        expert.phase = CookiePhase.DESCEND_TO_PLACE
        expert.phase_steps = expert.phase_timeout_steps
        expert.current_cookie_index = expert.target_slot_index = 0
        expert._placement_height = 0.042
        previous = env.last_applied_action.copy()
        np.testing.assert_array_equal(expert.act(), previous)
        assert expert.failed
        assert "placement descent timed out" in expert.failure_reason
        assert not expert.retry_reasons
        np.testing.assert_array_equal(expert.act(), previous)
    finally:
        env.close()


def test_viewer_starts_with_rotatable_front_camera() -> None:
    env = A3CookieTransferEnv(render_cameras=False)
    try:
        camera = mujoco.MjvCamera()
        env._configure_viewer_camera(camera)
        assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
        assert camera.fixedcamid == -1
        assert 120 < camera.azimuth < 150
        assert camera.elevation < 0
        np.testing.assert_allclose(camera.lookat, env.config.cameras.workspace_target_m)
    finally:
        env.close()


def test_expert_stops_if_a_previously_packed_cookie_moves(monkeypatch) -> None:
    from a3_dual_arm_sim.expert import A3CookieTransferExpert, CookiePhase

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0)
        expert = A3CookieTransferExpert(env)
        expert.reset()
        monkeypatch.setattr(expert.box_support, "contact_forces", lambda: np.ones(2))
        expert.phase = CookiePhase.APPROACH
        expert.completed_cookie_indices = [0]
        expert._packed_positions = {
            0: env.privileged_cookie_target_position(0) + np.array([0.01, 0.0, 0.0])
        }
        previous = env.last_applied_action.copy()
        np.testing.assert_array_equal(expert.act(), previous)
        assert expert.failed
        assert "packed Cookie 0 disturbed" in expert.failure_reason
    finally:
        env.close()


def test_pinch_offset_tracks_the_cookie_half_size() -> None:
    """The jaws pinch a fixed depth below the top face, whatever the Cookie size.

    The offsets here reproduce the historical 50 mm-Cookie value (``0.023``)
    exactly, which is the point: behaviour for the old Cookie is unchanged, while
    a resized Cookie now gets a proportionally corrected offset instead of the
    jaws closing at the same absolute height regardless of what they are aiming
    at.
    """
    from a3_dual_arm_sim.expert import _pinch_offset_from_cookie_centre

    assert _pinch_offset_from_cookie_centre(0.025) == pytest.approx(0.023)
    assert _pinch_offset_from_cookie_centre(0.0125) == pytest.approx(0.0105)
    # A Cookie respanned in z resizes the offset by the same amount.
    assert _pinch_offset_from_cookie_centre(0.025) - _pinch_offset_from_cookie_centre(
        0.0125
    ) == pytest.approx(0.0125)


def test_expert_plans_the_pinch_inside_the_cookie() -> None:
    """The planned grasp waypoint has to fall inside the Cookie's height.

    This is the scene-level half of the same invariant: the commanded tool height
    has to overlap the Cookie it is aiming at, not hover above it.  The pinch
    height used to be the literal ``0.023``, correct only while the Cookie was
    50 mm tall; once the Cookie was shortened to 25 mm the jaws were commanded to
    ~10.5 mm *above* the top face -- aiming at empty space.  Resizing the Cookie
    again must not break this.
    """
    from a3_dual_arm_sim.expert import A3CookieTransferExpert

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        expert = A3CookieTransferExpert(env)
        expert.reset()
        expert._select_cookie()

        cookie_index = expert.current_cookie_index
        assert cookie_index is not None, expert.failure_reason
        cookie_z = float(env.privileged_cookie_position(cookie_index)[2])
        half_z = float(env.COOKIE_HALF_SIZE[2])

        # Forward-kinematics the commanded waypoint rather than stepping, so the
        # assertion is about what the planner asked for and not about how far the
        # arm managed to track it.
        env.data.qpos[expert._l_qpos] = expert._waypoints["grasp"]
        mujoco.mj_kinematics(env.model, env.data)
        tool_z = float(env.data.site_xpos[expert._l_site][2])

        assert cookie_z - half_z < tool_z < cookie_z + half_z, (
            f"grasp waypoint z={tool_z * 1000:.3f} mm is outside the Cookie span "
            f"[{((cookie_z - half_z) * 1000):.3f}, {((cookie_z + half_z) * 1000):.3f}] mm"
        )
    finally:
        env.close()


def test_drop_check_measures_against_the_real_rest_height() -> None:
    """The "back on the bin floor" threshold has to track the Cookie height too.

    Same root cause as the grasp height: ``0.025`` was the Cookie's *half height*
    when the Cookie was 50 mm tall, so it happened to be the height of an upright
    Cookie's centre.  Left stale after the Cookie was shortened to 25 mm, it sat
    12.5 mm above the real rest height, so any Cookie that rose by less than that
    -- a marginal pinch, or one nudged by a neighbour -- would still read as
    "back on the bin floor" and be reported as dropped.
    """
    from a3_dual_arm_sim.expert import A3CookieTransferExpert

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False})
        expert = A3CookieTransferExpert(env)
        expert.reset()

        half_z = float(env.COOKIE_HALF_SIZE[2])
        threshold = expert._cookie_rest_height()
        assert threshold == pytest.approx(env.SOURCE_FLOOR_TOP_Z + half_z)

        for _ in range(200):
            mujoco.mj_step(env.model, env.data)
        settled = float(env.privileged_cookie_position(0)[2])

        # An un-lifted Cookie sits *below* the threshold (so a failed grasp is
        # still reported), but only just below it (so a real lift clears it
        # immediately and does not get mistaken for a drop).
        assert settled < threshold <= settled + 0.005, (
            f"rest-height threshold {threshold * 1000:.3f} mm vs settled Cookie "
            f"centre {settled * 1000:.3f} mm"
        )
    finally:
        env.close()
