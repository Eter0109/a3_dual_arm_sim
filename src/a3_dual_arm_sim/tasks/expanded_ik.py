"""Bounded multistart IK for absolute poses in expanded benchmark tasks.

An IK solution only certifies the tested pose residual. It does not certify a
collision-free path or a feasible grasp. The Cartesian action adapter is not
changed by this helper.
"""

import mujoco
import numpy as np


class ExpandedAbsoluteIKMixin:
    """Try alternative deterministic seeds when the initial solve is inaccurate."""

    def solve_ik(self, position, quaternion, initial_q=None, arm="L"):
        arm_index = 0 if arm in ("L", 0) else 1
        target_position = np.asarray(position, dtype=np.float64)
        target_quaternion = np.asarray(quaternion, dtype=np.float64)
        first = super().solve_ik(target_position, target_quaternion, initial_q, arm=arm)

        if not hasattr(self, "_expanded_ik_work"):
            self._expanded_ik_work = mujoco.MjData(self.model)
        work = self._expanded_ik_work
        work.qpos[:] = self.data.qpos
        qids = list(self._ik._qpos_ids[arm_index])
        site = self._eef_sites[arm_index]

        def residual(q):
            work.qpos[qids] = q
            mujoco.mj_kinematics(self.model, work)
            actual, rotation_error = np.empty(4), np.empty(3)
            mujoco.mju_mat2Quat(actual, work.site_xmat[site])
            mujoco.mju_subQuat(rotation_error, target_quaternion, actual)
            return (
                float(np.linalg.norm(work.site_xpos[site] - target_position)),
                float(np.linalg.norm(rotation_error)),
            )

        best = np.asarray(first, dtype=np.float64)
        distance, angle = residual(best)
        if distance <= 0.0008 and angle <= 0.015:
            return best
        best_error = distance + 0.12 * angle
        limits = self._ik._ranges[arm_index]
        home = self.DEPLOYMENT_HOME[:7] if arm_index == 0 else self.DEPLOYMENT_HOME[8:15]
        # This generator is local to the solver. It does not advance the scene
        # randomization stream or add a new benchmark randomization profile.
        rng = np.random.default_rng(0xA3C01 + arm_index)
        seeds = [self.data.qpos[qids].copy(), home, np.mean(limits, axis=1)]
        seeds.extend(rng.uniform(limits[:, 0], limits[:, 1], size=(4, len(qids))))
        for seed in seeds:
            candidate = super().solve_ik(target_position, target_quaternion, seed, arm=arm)
            distance, angle = residual(candidate)
            error = distance + 0.12 * angle
            if error < best_error:
                best, best_error = np.asarray(candidate, dtype=np.float64), error
            if distance <= 0.0008 and angle <= 0.015:
                return np.asarray(candidate, dtype=np.float64)
        return best
