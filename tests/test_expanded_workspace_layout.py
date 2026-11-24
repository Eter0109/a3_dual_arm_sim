"""Geometry and IK regressions for expanded benchmark scene layouts."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.contracts import LEFT_JOINTS
from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.tasks.cookie_scenarios import SCENARIO_TASKS
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


def reset_options(benchmark):
    return {
        "randomize_cookies": True,
        "randomize_boxes": True,
        "source_bin_noise_m": benchmark.source_bin_noise_m,
        "target_bin_noise_m": benchmark.target_bin_noise_m,
        "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
        "randomization_profile": "basic",
        "randomization_settings": benchmark.randomization_settings,
    }


def test_expanded_source_layout_is_separate_from_original_tasks():
    basic = make_cookie_benchmark("2x5").config.cookie_transfer
    ten = make_cookie_benchmark("2x10").config.cookie_transfer
    expanded = make_cookie_benchmark("sparse-exact-ten").config.cookie_transfer
    np.testing.assert_allclose(basic.source_bin_center_m, (0.175, 0.315))
    np.testing.assert_allclose(ten.source_bin_center_m, (0.171, 0.315))
    np.testing.assert_allclose(expanded.source_bin_center_m, (0.155, 0.315))
    old_positions = np.asarray(ten.cookie_source_positions_m)
    new_positions = np.asarray(expanded.cookie_source_positions_m)
    np.testing.assert_allclose(new_positions[:, 0], old_positions[:, 0] - 0.016)
    np.testing.assert_allclose(new_positions[:, 1], old_positions[:, 1])
    np.testing.assert_allclose(np.diff(np.unique(new_positions[:, 0])), 0.0544)
    np.testing.assert_allclose(expanded.target_bin_half_size_m, (0.063, 0.040))
    np.testing.assert_allclose(np.diff(np.unique(np.asarray(expanded.target_slots_local_m)[:, 1])), 0.012)
    np.testing.assert_allclose(basic.target_bin_half_size_m, (0.063, 0.035))
    assert make_cookie_benchmark("rotated-target").config.cookie_transfer.target_bin_world_position_m[:2] == pytest.approx((0.005, 0.1))
    two = make_cookie_benchmark("two-boxes").config.cookie_transfer
    assert two.target_bin_world_position_m[:2] == pytest.approx((0.095, 0.14))
    assert two.spare_target_bin_world_position_m[:2] == pytest.approx((0.07, 0.045))


@pytest.mark.parametrize("task,kwargs", [(task, {}) for task in SCENARIO_TASKS] + [("tilted", {"tilt_level": "fallen"}), ("two-boxes", {"box_quota": 10})])
def test_settled_clearance_checks_actual_active_source_inventory(task, kwargs):
    benchmark = make_cookie_benchmark(task, **kwargs)
    env = benchmark.create_env(render_cameras=False)
    try:
        for seed in (13000, 13008):
            _, info = env.reset(seed=seed, options=reset_options(benchmark))
            assert env.scenario_metadata["version"] == 2
            mask = np.asarray(info["cookies_in_source_mask"], dtype=bool)
            nominal = np.asarray(env.SOURCE_POSITIONS)[:, 0]
            centers = env.cookie_positions[:, 0]
            rotations = env.data.xmat[list(env._cookie_bodies)].reshape(-1, 3, 3)
            extents = (np.abs(rotations) @ env.COOKIE_HALF_SIZE)[:, 0]
            columns = np.unique(nominal)
            clearances = env.randomization_metadata["source_column_clearances_m"]
            pairs = env.randomization_metadata["source_column_clearance_pairs"]
            assert len(clearances) == len(pairs)
            for gap, (left, right) in zip(clearances, pairs, strict=True):
                li = mask & (nominal == columns[left - 1])
                ri = mask & (nominal == columns[right - 1])
                assert li.any() and ri.any()
                measured = np.min(centers[ri] - extents[ri]) - np.max(centers[li] + extents[li])
                assert gap == pytest.approx(measured, abs=1e-12)
                assert gap >= 0.003 - 1e-9
            env._check_container_contacts()
    finally:
        env.close()


def test_interleaved_custom_source_is_rejected_instead_of_silently_rewritten(tmp_path):
    source = resource_root() / "configs" / "cookie_batch.yaml"
    custom = tmp_path / "custom_scene.yaml"
    custom.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    benchmark = make_cookie_benchmark("sparse-exact-ten", config_path=custom)
    np.testing.assert_allclose(benchmark.config.cookie_transfer.source_bin_center_m, (0.175, 0.315))
    env = benchmark.create_env(render_cameras=False)
    try:
        with pytest.raises(RuntimeError, match="cross-column clearance"):
            env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
    finally:
        env.close()


def test_overlapping_containers_are_rejected():
    env = make_cookie_benchmark("two-boxes").create_env(render_cameras=False)
    try:
        env.reset(seed=13000, options={"randomize_boxes": False, "randomize_cookies": False})
        joint = env.model.joint("spare_target_bin_free")
        adr = int(joint.qposadr[0])
        env.data.qpos[adr:adr + 3] = env.data.xpos[env._target_bin_body]
        env.data.qpos[adr + 3:adr + 7] = env.data.xquat[env._target_bin_body]
        mujoco.mj_forward(env.model, env.data)
        with pytest.raises(RuntimeError, match="overlap"):
            env._check_container_contacts()
    finally:
        env.close()


def _has_clear_source_ik(env, cookie):
    """Independently check the previously blocked source poses at two endpoints."""
    site = env._eef_sites[0]
    qids = [env._qpos_ids[name] for name in LEFT_JOINTS]
    current = env.data.site_xmat[site].reshape(3, 3)
    pad_offset = current.T @ (
        env.data.geom_xpos[list(env._left_finger_geoms)].mean(axis=0) - env.data.site_xpos[site]
    )
    pad = env.cookie_positions[cookie].copy()
    pad[2] += env.COOKIE_HALF_SIZE[2] - 0.002
    canonical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    initial = np.array([1.28, 1.119, -1.127, -0.999, -0.252, 0.478, -1.535])
    work = mujoco.MjData(env.model)
    for pitch in (0, -15, 15, -35, 35, -40, 40, -45, 45, -50, 50, -55, 55):
        angle = np.deg2rad(pitch)
        c, s = np.cos(angle), np.sin(angle)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        for turn in (np.eye(3), np.diag([-1., 1., -1.])):
            rot = ry @ canonical @ turn
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rot.ravel())
            pick = pad - rot @ pad_offset
            pre = pick + np.array([0, 0, 0.082])
            qpre = env.solve_ik(pre, quat, initial)
            qpick = env.solve_ik(pick, quat, qpre)
            valid = True
            for q, desired in ((qpre, pre), (qpick, pick)):
                work.qpos[:] = env.data.qpos
                work.qpos[qids] = q
                for jid in env._finger_joints["L"]:
                    work.qpos[int(env.model.jnt_qposadr[jid])] = (1 - 0.02) * env.GRIPPER_RANGE_M
                mujoco.mj_forward(env.model, work)
                actual_q, rotation_error = np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(actual_q, work.site_xmat[site])
                mujoco.mju_subQuat(rotation_error, quat, actual_q)
                if np.linalg.norm(work.site_xpos[site] - desired) > 0.0008 or np.linalg.norm(rotation_error) > 0.015:
                    valid = False
                    break
                for contact in work.contact:
                    if contact.dist >= -1e-5:
                        continue
                    first = env.model.geom(int(contact.geom1)).name or ""
                    second = env.model.geom(int(contact.geom2)).name or ""
                    if first.startswith("L_") or second.startswith("L_"):
                        other = second if first.startswith("L_") else first
                        if other.startswith(("L_", "R_", "cookie_", "source_bin", "target_bin", "spare_target_bin")) or other == "table_top":
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return True
    return False


def test_sparse_seed_13008_previously_blocked_required_cookies_have_clear_endpoints():
    benchmark = make_cookie_benchmark("sparse-exact-ten")
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=13008, options=reset_options(benchmark))
        for cookie in (76, 77, 78):
            assert env._active[cookie]
            assert _has_clear_source_ik(env, cookie), f"no clear endpoint for cookie {cookie}"
    finally:
        env.close()


def test_expanded_absolute_ik_retries_a_bad_branch_for_rotated_target():
    from a3_dual_arm_sim.sim.env import A3DualArmEnv

    benchmark = make_cookie_benchmark("rotated-target")
    scene = benchmark.config.cookie_transfer
    benchmark.config = replace(benchmark.config, cookie_transfer=replace(
        scene, target_bin_world_position_m=(0.025, 0.100, scene.target_bin_world_position_m[2])
    ))
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=13000, options=reset_options(benchmark))
        site = env._eef_sites[0]
        qids = [env._qpos_ids[name] for name in LEFT_JOINTS]
        tool = env.data.site_xmat[site].reshape(3, 3)
        offset = tool.T @ (
            env.data.geom_xpos[list(env._left_finger_geoms)].mean(axis=0)
            - env.data.site_xpos[site]
        )
        body = env._target_bin_body
        box = env.data.xmat[body].reshape(3, 3)
        scene = env.config.cookie_transfer
        z = (scene.target_floor_z_m + scene.bin_wall_thickness_m / 2
             + 2 * env.COOKIE_HALF_SIZE[2] - 0.002 + 0.008)
        pad = env.data.xpos[body] + box @ np.array([*env.TARGET_SLOTS_LOCAL[1], z])
        angle = np.deg2rad(40)
        c, s = np.cos(angle), np.sin(angle)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        canonical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        rotation = box @ ry @ canonical
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        pick = pad - rotation @ offset
        pre = pick + box[:, 2] * 0.082
        initial = np.array([1.28, 1.119, -1.127, -0.999, -0.252, 0.478, -1.535])
        work = mujoco.MjData(env.model)
        work.qpos[:] = env.data.qpos

        def errors(solver):
            qpre = solver(pre, quaternion, initial)
            qpick = solver(pick, quaternion, qpre)
            result = []
            for q, desired in ((qpre, pre), (qpick, pick)):
                work.qpos[qids] = q
                mujoco.mj_kinematics(env.model, work)
                actual, angular = np.empty(4), np.empty(3)
                mujoco.mju_mat2Quat(actual, work.site_xmat[site])
                mujoco.mju_subQuat(angular, quaternion, actual)
                result.append((np.linalg.norm(work.site_xpos[site] - desired),
                               np.linalg.norm(angular)))
            return np.max(result, axis=0)

        single = errors(lambda p, q, guess: A3DualArmEnv.solve_ik(env, p, q, guess))
        retried = errors(env.solve_ik)
        assert single[0] > 0.0008 or single[1] > 0.015
        assert retried[0] <= 0.0008 and retried[1] <= 0.015
    finally:
        env.close()


@pytest.mark.parametrize("seed", [13000, 13001, 13002])
def test_default_rotated_target_passes_existing_fixed_pose_preflight(seed):
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert

    benchmark = make_cookie_benchmark("rotated-target")
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(seed=seed, options=reset_options(benchmark))
        A3SameColumnBatchExpert(env)
    finally:
        env.close()


@pytest.mark.parametrize("task", ["sparse-exact-ten", "2x10-four-source"])
def test_expanded_run_records_layout_and_ik_protocol(task):
    from types import SimpleNamespace

    from a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli import run_metadata

    benchmark = make_cookie_benchmark(task)
    metadata = run_metadata(SimpleNamespace(policy="expert"), benchmark)
    protocol = metadata["scenario_protocol"]
    assert protocol["version"] == 2
    assert "tasks/source_clearance.py" in protocol["source_sha256"]
    assert "tasks/expanded_ik.py" in protocol["source_sha256"]
    assert metadata["config"]["cookie_transfer"]["source_bin_center_m"] == benchmark.config.cookie_transfer.source_bin_center_m
