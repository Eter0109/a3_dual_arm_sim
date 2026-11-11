"""Additional, reproducible initial states for the cookie benchmark.

The six tasks share the 2x5 robot, camera and object geometry. Removed cookies
are physically hidden and collision-disabled; prefilled cookies remain real
rigid bodies. Their initial identities are excluded from the transfer score.
"""

from __future__ import annotations

import mujoco
import numpy as np

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

SCENARIO_TASKS = (
    "missing-columns",
    "partial-gaps",
    "cross-column",
    "tilted",
    "prefilled",
    "two-boxes",
)

SCENARIO_PROMPTS = {
    "missing-columns": "Transfer 10 cookies from the available source columns into the target box.",
    "partial-gaps": "Transfer 10 available cookies from the source box into the target box.",
    "cross-column": "Transfer 10 cookies from the available source columns into the target box.",
    "tilted": "Transfer 10 cookies from the source box into the target box, upright.",
    "prefilled": "Add 5 cookies to the partially filled target box to complete the 2 by 5 arrangement.",
    "two-boxes": "Transfer 10 cookies from the source box, placing 5 upright in each of the two target boxes.",
}


class A3CookieScenarioEnv(A3CookieTransferEnv):
    """A benchmark-only environment with scenario-aware counting and success."""

    def __init__(
        self,
        config,
        *,
        scenario: str,
        tilt_level: str | None = None,
        box_quota: int | None = None,
        **kwargs,
    ):
        if scenario not in SCENARIO_TASKS:
            raise ValueError(f"unknown cookie scenario: {scenario}")
        self.scenario = scenario
        if scenario == "tilted":
            self.tilt_level = tilt_level or "mild"
            if self.tilt_level not in ("mild", "fallen"):
                raise ValueError("tilt_level must be mild or fallen")
        elif tilt_level is not None:
            raise ValueError("tilt_level only applies to the tilted scenario")
        else:
            self.tilt_level = None
        if scenario == "two-boxes":
            self.box_quota = 5 if box_quota is None else box_quota
            if self.box_quota not in (5, 10):
                raise ValueError("box_quota must be 5 or 10")
        elif box_quota is not None:
            raise ValueError("box_quota only applies to the two-boxes scenario")
        else:
            self.box_quota = None
        self.goal_count = (
            2 * self.box_quota
            if self.box_quota is not None
            else 5
            if scenario == "prefilled"
            else 10
        )
        super().__init__(config, **kwargs)
        self._cookie_all_geoms = tuple(
            np.flatnonzero(self.model.geom_bodyid == body) for body in self._cookie_bodies
        )
        self._original_contype = self.model.geom_contype.copy()
        self._original_affinity = self.model.geom_conaffinity.copy()
        self._original_rgba = self.model.geom_rgba.copy()
        self._original_gravcomp = self.model.body_gravcomp.copy()
        self._original_source_position = self.model.body_pos[self._source_bin_body].copy()
        self._spare_body = (
            self._id(mujoco.mjtObj.mjOBJ_BODY, "spare_target_bin")
            if scenario == "two-boxes"
            else None
        )
        self._active = np.ones(self.task_config.cookie_count, dtype=bool)
        self._prefilled: set[int] = set()
        self.initial_source_count = 0
        self._scenario_hold_count = 0
        self.scenario_metadata = {}

    def _restore_model(self):
        self.model.geom_contype[:] = self._original_contype
        self.model.geom_conaffinity[:] = self._original_affinity
        self.model.geom_rgba[:] = self._original_rgba
        self.model.body_gravcomp[:] = self._original_gravcomp
        self.model.body_pos[self._source_bin_body] = self._original_source_position

    def _remove_cookie(self, index: int):
        self._active[index] = False
        geoms = self._cookie_all_geoms[index]
        self.model.geom_contype[geoms] = 0
        self.model.geom_conaffinity[geoms] = 0
        self.model.geom_rgba[geoms, 3] = 0
        self.model.body_gravcomp[self._cookie_bodies[index]] = 1.0
        self.set_cookie_pose(index, (10.0 + index * 0.08, 10.0, 1.0))

    def _source_columns(self):
        positions = np.asarray(self.SOURCE_POSITIONS)
        return [
            sorted(
                np.flatnonzero(np.isclose(positions[:, 0], x)).tolist(),
                key=lambda index: positions[index, 1],
            )
            for x in np.unique(positions[:, 0])
        ]

    def _put_in_primary_slot(self, index: int, slot: int, *, offset_y: float = 0.0):
        position = self.privileged_target_slot_world(slot, self._target_cookie_center_z)
        rotation = self.data.xmat[self._target_bin_body].copy()
        position += rotation.reshape(3, 3) @ np.array([0.0, offset_y, 0.0])
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation)
        self.set_cookie_pose(index, tuple(position), tuple(quaternion))

    def reset(self, *, seed=None, options=None):
        self._restore_model()
        self._active[:] = True
        self._prefilled = set()
        _, info = super().reset(seed=seed, options=options)
        # Independent of the existing layout, appearance and column streams.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), 0xA3C001]))
        columns = self._source_columns()
        if len(columns) != 4 or any(len(column) != 20 for column in columns):
            raise ValueError("scenario benchmarks require the 4-column, 80-cookie 2x5 layout")
        removed = []
        setup = {}
        if self.scenario == "missing-columns":
            selected = sorted(rng.choice(4, size=int(rng.integers(1, 3)), replace=False).tolist())
            removed = [index for column in selected for index in columns[column]]
            setup["removed_columns"] = [column + 1 for column in selected]
        elif self.scenario == "partial-gaps":
            # Every column has both a missing front section and an internal gap.
            start = int(rng.integers(7, 11))
            removed = [
                index for column in columns for index in (column[:3] + column[start : start + 3])
            ]
            setup["gap_rows_zero_based"] = [0, 1, 2, start, start + 1, start + 2]
        elif self.scenario == "cross-column":
            starts = rng.integers(0, 13, size=4).tolist()
            removed = [
                index
                for column, start in zip(columns, starts, strict=True)
                for row, index in enumerate(column)
                if not start <= row < start + 8
            ]
            setup["remaining_row_starts_zero_based"] = starts
            setup["remaining_per_column"] = 8
        elif self.scenario == "prefilled":
            source_column = int(rng.integers(4))
            target_column = int(rng.integers(2))
            self._prefilled = set(columns[source_column][:5])
            slots = list(range(target_column, 10, 2))
            slot_rows = np.unique(np.asarray(self.TARGET_SLOTS_LOCAL)[:, 1])
            nominal_pitch = float(np.median(np.diff(slot_rows)))
            prefilled_pitch = min(0.0085, nominal_pitch + 0.45 * self.TARGET_SLOT_TOLERANCE[1])
            for index, slot in zip(sorted(self._prefilled), slots, strict=True):
                # Small gaps prevent initial collision meshes interlocking after
                # box yaw randomization; all five remain within slot tolerance.
                desired_y = (slot // 2 - 2) * prefilled_pitch
                offset = desired_y - self.TARGET_SLOTS_LOCAL[slot][1]
                self._put_in_primary_slot(index, slot, offset_y=offset)
            setup["prefilled_cookie_ids"] = sorted(self._prefilled)
            setup["prefilled_slot_ids"] = slots
            setup["prefilled_source_column"] = source_column + 1
            setup["prefilled_target_column"] = target_column + 1
            setup["prefilled_pitch_m"] = float(prefilled_pitch)
        elif self.scenario == "tilted":
            selected = sorted(rng.choice(4, size=2, replace=False).tolist())
            row = int(rng.integers(7, 12))
            indices = [columns[column][row] for column in selected]
            if self.tilt_level == "fallen":
                # Clear one adjacent cookie on either side so a real rigid
                # cookie can lie flat without interpenetrating its neighbours.
                removed = [
                    columns[column][row + offset] for column in selected for offset in (-1, 1)
                ]
            setup.update(tilted_cookie_ids=indices, tilt_level=self.tilt_level)
        for index in removed:
            self._remove_cookie(index)
        # Let contacts, including tilted and prefilled cookies, settle before
        # capturing both policy observations and the reference state.
        settle_steps = 0
        angles = (
            ((90.0,) if self.tilt_level == "fallen" else (16.0, 22.0, 28.0, 34.0))
            if self.scenario == "tilted"
            else (None,)
        )
        min_tilt = 60.0 if self.tilt_level == "fallen" else 5.0
        for requested_angle in angles:
            if requested_angle is not None:
                for index in indices:
                    position = self.cookie_positions[index].copy()
                    position[2] += 0.0006
                    current_q = self.data.xquat[self._cookie_bodies[index]].copy()
                    # Preserve yaw while replacing any previous roll.
                    rotation = self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)
                    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                    current_q[:] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
                    angle = np.deg2rad(requested_angle) / 2
                    tilt_q = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0])
                    quaternion = np.empty(4, dtype=np.float64)
                    mujoco.mju_mulQuat(quaternion, tilt_q, current_q)
                    if self.tilt_level == "fallen":
                        matrix = np.empty(9, dtype=np.float64)
                        mujoco.mju_quat2Mat(matrix, quaternion)
                        half_extent = np.abs(matrix.reshape(3, 3)) @ self.COOKIE_HALF_SIZE
                        position[2] = self.SOURCE_FLOOR_TOP_Z + half_extent[2] + 0.0002
                    self.set_cookie_pose(index, tuple(position), tuple(quaternion))
                setup["requested_tilt_deg"] = requested_angle
            mujoco.mj_forward(self.model, self.data)
            for _ in range(250):
                mujoco.mj_step(self.model, self.data)
            settle_steps += 250
            mujoco.mj_forward(self.model, self.data)
            if requested_angle is None or any(self._tilt_degrees()[i] >= min_tilt for i in indices):
                break
        # Reject unstable samples rather than presenting a still-falling scene
        # to the policy. Limits are the existing target-settling thresholds.
        for _ in range(9):
            speeds = [self._cookie_speed(i) for i in np.flatnonzero(self._active)]
            max_linear = max(speed[0] for speed in speeds)
            max_angular = max(speed[1] for speed in speeds)
            if max_linear <= self.task_config.max_linear_speed_m_s and (
                max_angular <= self.task_config.max_angular_speed_rad_s
            ):
                break
            for _ in range(125):
                mujoco.mj_step(self.model, self.data)
            settle_steps += 125
            mujoco.mj_forward(self.model, self.data)
        else:
            raise RuntimeError(f"invalid {self.scenario} initial state: cookies did not settle")
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise RuntimeError(f"invalid {self.scenario} initial state: non-finite physics state")
        self.data.time = 0.0
        self._success_hold_count = 0
        self._scenario_hold_count = 0
        source_mask = tuple(
            bool(self._active[i] and self._cookie_inside_source(i))
            for i in range(self.task_config.cookie_count)
        )
        self.initial_source_count = sum(source_mask)
        expected = int(np.count_nonzero(self._active)) - len(self._prefilled)
        if self.initial_source_count != expected:
            raise RuntimeError(
                f"invalid {self.scenario} initial state: {self.initial_source_count}/{expected} "
                "active source cookies remain inside the source box"
            )
        if self._prefilled and not all(self._cookie_inside_target(i) for i in self._prefilled):
            raise RuntimeError("prefilled cookies did not settle in the target box")
        if self._prefilled:
            initial_slots = self._target_slot_occupancy(
                tuple(i in self._prefilled for i in range(self.task_config.cookie_count))
            )
            if any(initial_slots[slot] < 0 for slot in setup["prefilled_slot_ids"]):
                raise RuntimeError("prefilled cookies do not match the configured target slots")
        tilts = self._tilt_degrees()
        if self.scenario == "tilted" and sum(tilts[i] >= min_tilt for i in indices) < 1:
            raise RuntimeError("tilted scenario settled without a tilted cookie")
        self._source_initially_filled = all(source_mask) and self._collection_touches_all_walls(
            source_mask, self.data.xpos[self._source_bin_body][:2], self.SOURCE_INNER_HALF_SIZE
        )
        self._initial_positions = self.cookie_positions.copy()
        self._initial_tilts = tilts
        target_bodies = [self._target_bin_body] + (
            [self._spare_body] if self._spare_body is not None else []
        )
        self.scenario_metadata = {
            "name": self.scenario,
            "version": 1,
            "active_cookie_ids": np.flatnonzero(self._active).tolist(),
            "removed_cookie_ids": removed,
            "initial_source_count": self.initial_source_count,
            "required_new_cookies": self.goal_count,
            "target_quotas": (
                [self.box_quota, self.box_quota] if self.box_quota is not None else [10]
            ),
            "initial_target_counts": [len(self._prefilled)]
            + ([0] if self._spare_body is not None else []),
            "initial_target_bin_poses": [
                {
                    "position": self.data.xpos[body].tolist(),
                    "quaternion": self.data.xquat[body].tolist(),
                }
                for body in target_bodies
            ],
            "initial_tilt_degrees": [round(float(value), 3) for value in tilts],
            "settle_physics_steps": settle_steps,
            "max_initial_cookie_linear_speed_m_s": max_linear,
            "max_initial_cookie_angular_speed_rad_s": max_angular,
            **setup,
        }
        self.randomization_metadata.update(
            scenario=self.scenario_metadata,
            source_bin_position=self.data.xpos[self._source_bin_body].tolist(),
            target_bin_position=self.data.xpos[self._target_bin_body].tolist(),
            target_bin_quaternion=self.data.xquat[self._target_bin_body].tolist(),
            cookie_positions=self.cookie_positions.tolist(),
            cookie_quaternions=self.data.xquat[list(self._cookie_bodies)].tolist(),
        )
        metrics = self._scenario_metrics()
        target_mask = tuple(
            bool(self._active[i] and self._cookie_inside_target(i))
            for i in range(self.task_config.cookie_count)
        )
        info.update(
            cookies_in_target=0,
            cookies_in_source=self.initial_source_count,
            required_cookies=self.goal_count,
            cookies_in_target_mask=target_mask,
            cookies_in_source_mask=source_mask,
            target_slot_occupancy=self._target_slot_occupancy(target_mask),
            source_initially_filled=self._source_initially_filled,
            scenario_metrics=metrics,
        )
        return self._observation(), info

    def _tilt_degrees(self):
        rotations = self.data.xmat[list(self._cookie_bodies)].reshape(-1, 3, 3)
        return np.rad2deg(np.arccos(np.clip(np.abs(rotations[:, 2, 2]), 0.0, 1.0)))

    def _inside_bin(self, index: int, body: int):
        original = self._target_bin_body
        try:
            self._target_bin_body = body
            return self._cookie_inside_target(index)
        finally:
            self._target_bin_body = original

    def _inside_bin_region(self, index: int, body: int):
        """Containment without the upright, velocity or release scoring rules."""
        rotation = self.data.xmat[body].reshape(3, 3)
        relative_position = rotation.T @ (
            self.data.xpos[self._cookie_bodies[index]] - self.data.xpos[body]
        )
        relative_rotation = rotation.T @ self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)
        half_extent = np.abs(relative_rotation) @ self.COOKIE_HALF_SIZE
        floor = (
            self.config.cookie_transfer.target_floor_z_m
            + self.config.cookie_transfer.bin_wall_thickness_m / 2
        )
        return bool(
            np.all(
                np.abs(relative_position[:2]) + half_extent[:2]
                <= self.TARGET_INNER_HALF_SIZE + self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
            and floor - 0.003
            <= relative_position[2] - half_extent[2]
            < self.TARGET_WALL_HEIGHT + self.CONTACT_CONTAINMENT_TOLERANCE_M
        )

    def _scenario_metrics(self):
        active = np.flatnonzero(self._active).tolist()
        bins = [self._target_bin_body] + (
            [self._spare_body] if self._spare_body is not None else []
        )
        in_bins = [[i for i in active if self._inside_bin(i, body)] for body in bins]
        in_bin_regions = [[i for i in active if self._inside_bin_region(i, body)] for body in bins]
        in_source = [i for i in active if self._cookie_inside_source(i)]
        present = set(in_source).union(*map(set, in_bin_regions))
        outside = [i for i in active if i not in present]
        transferred = [i for i in active if i not in self._prefilled]
        new_counts = [sum(i in transferred for i in contained) for contained in in_bins]
        score = (
            sum(min(n, self.box_quota) for n in new_counts)
            if self._spare_body is not None
            else min(new_counts[0], self.goal_count)
        )
        displacement = np.linalg.norm(self.cookie_positions - self._initial_positions, axis=1)
        tilt_delta = np.abs(self._tilt_degrees() - self._initial_tilts)
        disturbed = [i for i in in_source if displacement[i] > 0.01 or tilt_delta[i] > 15.0]
        return {
            "initial_source_count": self.initial_source_count,
            "expected_source_count": self.initial_source_count - self.goal_count,
            "target_counts": [len(items) for items in in_bins],
            "new_target_counts": new_counts,
            "new_correct_cookies": score,
            "prefilled_remaining": sum(i in in_bins[0] for i in self._prefilled),
            "cookies_in_source": len(in_source),
            "cookies_outside_containers": len(outside),
            "outside_cookie_ids": outside,
            "disturbed_source_cookie_count": len(disturbed),
            "disturbed_source_cookie_ids": disturbed,
            "disturbance_position_threshold_m": 0.01,
            "disturbance_tilt_threshold_deg": 15.0,
        }

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        metrics = self._scenario_metrics()
        count_ok = (
            metrics["target_counts"] == [self.box_quota, self.box_quota]
            if self._spare_body is not None
            else metrics["target_counts"] == [10]
        )
        if self._spare_body is None:
            count_ok = count_ok and all(int(i) >= 0 for i in info["target_slot_occupancy"])
            count_ok = count_ok and bool(info["target_touches_all_walls"])
        elif self.box_quota == 10 and count_ok:
            # Full boxes use the same one-to-one slot and boundary checks as
            # 2x5, separately in each box's own coordinate frame.
            original = self._target_bin_body
            try:
                for body in (original, self._spare_body):
                    self._target_bin_body = body
                    mask = tuple(
                        bool(self._active[i] and self._cookie_inside_target(i))
                        for i in range(self.task_config.cookie_count)
                    )
                    count_ok = count_ok and all(i >= 0 for i in self._target_slot_occupancy(mask))
                    count_ok = count_ok and self._collection_touches_all_walls(
                        mask, self.TARGET_CENTER, self.TARGET_INNER_HALF_SIZE, is_target=True
                    )
            finally:
                self._target_bin_body = original
        count_ok = count_ok and metrics["prefilled_remaining"] == len(self._prefilled)
        count_ok = count_ok and metrics["cookies_in_source"] == metrics["expected_source_count"]
        count_ok = count_ok and metrics["cookies_outside_containers"] == 0
        self._scenario_hold_count = self._scenario_hold_count + 1 if count_ok else 0
        success = self._scenario_hold_count >= self.task_config.success_hold_steps
        info.update(
            success=success,
            cookies_in_target=metrics["new_correct_cookies"],
            cookies_in_source=metrics["cookies_in_source"],
            required_cookies=self.goal_count,
            success_hold_count=self._scenario_hold_count,
            scenario_metrics=metrics,
        )
        return (
            observation,
            metrics["new_correct_cookies"] / info["required_cookies"],
            (terminated or success),
            truncated,
            info,
        )
