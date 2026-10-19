"""Rate-limited IK commands. No method writes the live physics state."""
import numpy as np


class LeftArmServo:
    def __init__(self, env):
        self.env = env
        self.right_hold = np.asarray(env.last_applied_action[8:], dtype=float).copy()
        self.site = env._eef_sites[0]

    def canonical(self, action):
        value = np.asarray(action, dtype=float)
        if not np.isfinite(value).all():
            raise ValueError("Action contains non-finite values")
        if value.shape == (14,):
            value = self.env._ik.convert(self.env.data, value, self.env.last_applied_action)
        if value.shape not in ((8,), (16,)):
            raise ValueError("Expected left8, canonical16 or Cartesian14 action")
        left = value[:8].copy()
        left[7] = np.clip(left[7], 0, 1)
        return np.r_[left, self.right_hold]

    def hold(self, opening=None):
        left = self.env.last_applied_action[:8].copy()
        if opening is not None:
            left[7] = opening
        return self.canonical(left)

    def move(self, position, rotation, opening, speed=.003):
        import mujoco
        env, work = self.env, self.env._ik._work
        work.qpos[:] = env.data.qpos
        work.qpos[list(env._ik._qpos_ids[0])] = env.last_applied_action[:7]
        work.qpos[list(env._ik._qpos_ids[1])] = self.right_hold[:7]
        mujoco.mj_kinematics(env.model, work)
        delta = np.asarray(position) - work.site_xpos[self.site]
        distance = np.linalg.norm(delta)
        delta *= min(1., speed / max(distance, 1e-10))
        desired, current, error = np.empty(4), np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(desired, np.asarray(rotation).ravel())
        mujoco.mju_mat2Quat(current, work.site_xmat[self.site])
        mujoco.mju_subQuat(error, desired, current)
        error *= min(1., .04 / max(np.linalg.norm(error), 1e-10))
        action = np.zeros(14)
        action[:3] = np.clip(delta / env.config.cartesian_translation_scale_m, -1, 1)
        action[3:6] = np.clip(error / env.config.cartesian_rotation_scale_rad, -1, 1)
        action[6] = np.clip(2 * opening - 1, -1, 1)
        action[13] = 2 * self.right_hold[7] - 1
        measured = env.data.site_xpos[self.site]
        mujoco.mju_mat2Quat(current, env.data.site_xmat[self.site])
        mujoco.mju_subQuat(error, desired, current)
        reached = (np.linalg.norm(np.asarray(position) - measured) < .0025
                   and np.linalg.norm(error) < .045)
        return self.canonical(action), bool(reached)

    def pad_offset(self):
        env = self.env
        rotation = env.data.site_xmat[self.site].reshape(3, 3)
        midpoint = env.data.geom_xpos[list(env._left_finger_geoms)].mean(0)
        return rotation.T @ (midpoint - env.data.site_xpos[self.site])

    def opening_for_width(self, width, margin=.006):
        env = self.env
        pads = env.data.geom_xpos[list(env._left_finger_geoms)]
        distance = np.linalg.norm(pads[1] - pads[0])
        pad_half = env.config.cookie_transfer.left_finger_pad_half_thickness_m
        offset = distance - 2 * pad_half - env.current_joint_action[7] * .085
        return float(np.clip((width + margin - offset) / .085, 0, 1))


def average_rotation(rotations):
    u, _, vt = np.linalg.svd(np.asarray(rotations).mean(0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def group_width(scene, ids, jaw_axis):
    centers = np.asarray([scene.cookies[i].position @ jaw_axis for i in ids])
    radii = np.asarray([np.abs(jaw_axis @ scene.cookies[i].rotation) @ scene.half_size
                        for i in ids])
    return float(max(centers + radii) - min(centers - radii))
