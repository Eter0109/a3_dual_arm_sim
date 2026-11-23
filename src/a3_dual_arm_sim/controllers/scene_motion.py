"""Clearance-checked joint paths for physical multi-box expert transfers."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ClearancePath:
    joints: list[np.ndarray]
    route: str
    obstacle_z: float
    minimum_z: float


def obstacle_top(expert, *, loaded):
    env = expert.env
    top = max(env.SOURCE_WALL_TOP_Z, *(env.data.xpos[b, 2] + env.TARGET_WALL_HEIGHT
                                      for b in env.target_box_bodies))
    excluded = set(expert.batch_indices) if loaded else set()
    for i in np.flatnonzero(env._scene_randomizer.active):
        if i in excluded:
            continue
        b = env._cookie_bodies[i]
        half_z = np.abs(env.data.xmat[b].reshape(3, 3)[2]) @ env.COOKIE_HALF_SIZE
        top = max(top, float(env.data.xpos[b, 2] + half_z))
    return float(top)


def _gripper_bottom(expert, data):
    model = expert.model
    result = []
    for geom in expert.env._left_finger_geoms:
        half_z = np.abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom]
        result.append(data.geom_xpos[geom, 2] - half_z)
    base = model.geom("L_2f85_base_collision").id
    axis = data.geom_xmat[base].reshape(3, 3)[:, 2]
    radius, half_length = model.geom_size[base, :2]
    extent = radius * np.sqrt(max(0.0, 1 - axis[2] ** 2)) + half_length * abs(axis[2])
    result.append(data.geom_xpos[base, 2] - extent)
    return float(min(result))


def actual_bottom(expert, *, loaded):
    bottom = _gripper_bottom(expert, expert.data)
    if loaded:
        for i in expert.batch_indices:
            body = expert.env._cookie_bodies[i]
            extent = np.abs(expert.data.xmat[body].reshape(3, 3)[2]) @ expert.env.COOKIE_HALF_SIZE
            bottom = min(bottom, expert.data.xpos[body, 2] - extent)
    return float(bottom)


def plan_clearance_path(expert, goal_position, goal_rotation, *, loaded, allow_intermediate_relax=False):
    """Plan in a separate kinematic workspace; never modify live physics.

    A Cartesian rate limit followed by independently clipped joint commands
    can dip the real wrist during a large yaw change. These waypoints are
    checked at intermediate joint configurations and executed proportionally.
    """
    env, work = expert.env, expert.env._ik._work
    work.qpos[:] = env.data.qpos
    q_start = env.last_applied_action[:7]
    work.qpos[expert._l_qpos] = q_start
    mujoco.mj_kinematics(expert.model, work)
    source_r = work.site_xmat[expert._l_site].reshape(3, 3).copy()
    source_p = work.site_xpos[expert._l_site].copy()
    offset = expert._group_offset if loaded else np.zeros(3)
    source_center = source_p + source_r @ offset
    goal_center = np.asarray(goal_position) + goal_rotation @ offset
    obstacle_z = obstacle_top(expert, loaded=loaded)
    min_required = obstacle_z + 0.010
    failures = []

    def geometry(q):
        work.qpos[expert._l_qpos] = q
        mujoco.mj_kinematics(expert.model, work)
        p = work.site_xpos[expert._l_site].copy()
        r = work.site_xmat[expert._l_site].reshape(3, 3).copy()
        bottom = _gripper_bottom(expert, work)
        if loaded:
            centers = p + expert._held_offsets @ r.T
            rotations = np.einsum("ij,njk->nik", r, expert._scene_held_rotations)
            heights = np.abs(rotations[:, 2, :]) @ env.COOKIE_HALF_SIZE
            bottom = min(bottom, float(np.min(centers[:, 2] - heights)))
        return p, r, bottom

    def segment_plan(waypoints):
        q = q_start.copy()
        previous_center, previous_r = source_center.copy(), source_r.copy()
        joints, min_z = [], float("inf")
        for segment, (center, r) in enumerate(waypoints):
            turn = Rotation.from_matrix(previous_r.T @ r).as_rotvec()
            distance = np.linalg.norm(center - previous_center)
            n = max(1, int(np.ceil(distance / 0.014)), int(np.ceil(np.linalg.norm(turn) / 0.10)))
            for fraction in np.linspace(1 / n, 1, n):
                interpolated_r = previous_r @ Rotation.from_rotvec(turn * fraction).as_matrix()
                interpolated_center = previous_center + fraction * (center - previous_center)
                p = interpolated_center - interpolated_r @ offset
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, interpolated_r.ravel())
                next_q = env.solve_ik(p, quat, q)
                actual_p, actual_r, bottom = geometry(next_q)
                angle = np.linalg.norm(Rotation.from_matrix(actual_r.T @ interpolated_r).as_rotvec())
                position_error = np.linalg.norm(actual_p - p)
                final_pose = segment == len(waypoints) - 1 and fraction >= 1 - 1e-9
                # The path's intermediate ideal Cartesian samples are not
                # task targets. A slight deviation is harmless if actual FK,
                # joint continuity and clearance are all checked. The true
                # pre-place/approach endpoint keeps its original tolerance.
                approximate = allow_intermediate_relax and not final_pose
                position_tol = 0.009 if approximate else 0.0020
                angle_tol = 0.12 if approximate else 0.045
                if position_error > position_tol or angle > angle_tol:
                    failures.append({"reason": "IK_error", "position_m": p.tolist(),
                                     "position_error_m": float(position_error), "angle_error_rad": float(angle)})
                    return None
                if np.max(np.abs(next_q - q)) > 0.36 or bottom < min_required:
                    failures.append({"reason": "joint_jump_or_height", "position_m": p.tolist(),
                                     "bottom_m": bottom, "required_m": min_required,
                                     "joint_delta": float(np.max(np.abs(next_q - q)))})
                    return None
                min_z = min(min_z, bottom)
                for t in (0.25, 0.5, 0.75):
                    _, _, between_bottom = geometry(q + t * (next_q - q))
                    min_z = min(min_z, between_bottom)
                    if between_bottom < min_required:
                        failures.append({"reason": "interpolation_height", "bottom_m": between_bottom,
                                         "required_m": min_required})
                        return None
                joints.append(next_q.copy())
                q = next_q
            previous_center, previous_r = center, r
        return joints, min_z

    turn = Rotation.from_matrix(source_r.T @ goal_rotation).as_rotvec()
    mid_r = source_r @ Rotation.from_rotvec(turn * 0.5).as_matrix()
    heights = [max(source_center[2], goal_center[2], obstacle_z + 0.070),
               max(source_center[2], goal_center[2], obstacle_z + 0.090)]
    for high in heights:
        src_high, dst_high = source_center.copy(), goal_center.copy()
        src_high[2] = dst_high[2] = high
        middle = np.array([0.08, 0.32, high])
        routes = (
            ("high-linear", [(src_high, source_r), (dst_high, goal_rotation)]),
            ("turn-then-travel", [(src_high, source_r), (src_high, goal_rotation), (dst_high, goal_rotation)]),
            ("travel-then-turn", [(src_high, source_r), (dst_high, source_r), (dst_high, goal_rotation)]),
            ("workspace-waypoint", [(src_high, source_r), (middle, mid_r), (dst_high, goal_rotation)]),
        )
        if allow_intermediate_relax:
            # Far source rows can be near the arm's wrist limits. Withdraw
            # inward while preserving the secure grasp orientation before
            # rotating, instead of requiring the turn at the outer reach.
            inward = np.array([0.115, np.clip(source_center[1], 0.26, 0.36), high])
            routes = (*routes,
                      ("withdraw-before-turn", [(src_high, source_r), (inward, source_r),
                                               (inward, mid_r), (dst_high, goal_rotation)]))
        for name, route in routes:
            planned = segment_plan([*route, (goal_center, goal_rotation)])
            if planned is not None:
                joints, minimum_z = planned
                expert._last_path_failures = failures
                suffix = "-approx-intermediate" if allow_intermediate_relax else ""
                return ClearancePath(joints, name + suffix, obstacle_z, minimum_z)
    expert._last_path_failures = failures
    return None


def follow_clearance_path(expert, path, index, opening, *, loaded):
    """Return a rate-limited physical joint command and the next waypoint index."""
    actual = expert.data.qpos[expert._l_qpos]
    last = expert.env.last_applied_action
    if actual_bottom(expert, loaded=loaded) < path.obstacle_z + 0.003:
        raise RuntimeError("transport lost verified obstacle clearance")
    if index >= len(path.joints):
        return last, index
    delta = path.joints[index] - last[:7]
    largest = float(np.max(np.abs(delta)))
    # Avoid independently clipping the IK output: follow the tested joint
    # interpolation, and slow down if the measured arm lags the command.
    speed = 0.030 if loaded else 0.044
    if np.max(np.abs(actual - last[:7])) <= 0.090:
        last[:7] += delta * min(1.0, speed / max(largest, 1e-12))
        if largest <= speed:
            index += 1
    last[7] = opening
    return last, index


def insertion_reachable(expert, start_q, start_pos, end_pos, rotation):
    """Check the vertical approach's IK branch before committing the transfer."""
    work = expert.env._ik._work
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    q = start_q.copy()
    for t in (0.25, 0.5, 0.75, 1.0):
        p = np.asarray(start_pos) + t * (np.asarray(end_pos) - start_pos)
        next_q = expert.env.solve_ik(p, quat, q)
        work.qpos[expert._l_qpos] = next_q
        mujoco.mj_kinematics(expert.model, work)
        r = work.site_xmat[expert._l_site].reshape(3, 3)
        angle = np.linalg.norm(Rotation.from_matrix(r.T @ rotation).as_rotvec())
        if (np.linalg.norm(work.site_xpos[expert._l_site] - p) > 0.0025 or angle > 0.05
                or np.max(np.abs(next_q - q)) > 0.40):
            return False
        q = next_q
    return True
