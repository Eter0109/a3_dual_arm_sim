from __future__ import annotations

import mujoco
import numpy as np

from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv


def test_cookie_scene_has_central_stand_bins_and_thirty_upright_cookies() -> None:
    env = A3CookieTransferEnv(render_cameras=False)
    try:
        for name in ("stand_mast", "source_bin_floor", "target_bin_floor"):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        for index in range(30):
            assert (
                mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_BODY, f"cookie_{index}"
                )
                >= 0
            )
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
        collision = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_0_geom"
        )
        first_visual = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_0_visual"
        )
        second_visual = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "cookie_1_visual"
        )
        assert np.all(env.model.geom_size[first_visual] < env.model.geom_size[collision])
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
        assert first["cookies_in_source"] == second["cookies_in_source"] == 30
        assert first["source_initially_filled"]
        assert second["source_initially_filled"]
        assert all(
            abs(float(env.data.geom_xmat[geom_id].reshape(3, 3)[2, 2]))
            >= np.cos(env.task_config.max_tilt_rad)
            for geom_id in env._cookie_geoms
        )
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
        assert info["cookies_in_source"] == 20
        assert info["exact_2x5_fill"]
        assert info["target_touches_all_walls"]
        assert all(index >= 0 for index in info["target_slot_occupancy"])
        assert info["success_hold_count"] >= env.task_config.success_hold_steps
    finally:
        env.close()


def test_cookie_transfer_expert_transfers_ten_cookies_successfully() -> None:
    from a3_dual_arm_sim.expert import A3CookieTransferExpert

    env = A3CookieTransferEnv(render_cameras=False)
    try:
        expert = A3CookieTransferExpert(env)
        obs, info = env.reset(seed=0, options={"randomize_cookies": False})
        expert.reset()
        terminated = False
        step = 0
        while not terminated and step < len(expert.actions) + 80:
            action = expert.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
        assert terminated
        assert info["success"]
        assert info["cookies_in_target"] == 10
        assert info["cookies_in_source"] == 20
        assert info["exact_2x5_fill"]
        assert info["target_touches_all_walls"]
        assert all(index >= 0 for index in info["target_slot_occupancy"])
    finally:
        env.close()

