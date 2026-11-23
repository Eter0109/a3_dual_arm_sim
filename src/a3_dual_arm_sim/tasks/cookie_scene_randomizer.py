"""Episode initialization and contact damage for optional cookie scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from a3_dual_arm_sim.sim.cookie_scene_options import (
    COOKIES_PER_GROUP,
    LEAN_SUPPORT_HALF_HEIGHT_M,
    CookieSceneOptions,
)

if TYPE_CHECKING:
    from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv


class CookieSceneRandomizer:
    def __init__(self, env: A3CookieTransferEnv, options: CookieSceneOptions) -> None:
        self.env = env
        self.options = options
        self.model = env.model
        self.data = env.data
        self.gap_range = options.feasible_gap_range(env.config.cookie_transfer)
        self.boxes = options.target_boxes(env.config.cookie_transfer)
        self.target_bodies = tuple(
            env._id(mujoco.mjtObj.mjOBJ_BODY, name) for name, _, _ in self.boxes
        )
        self.target_joints = tuple(
            env._id(mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free") for name, _, _ in self.boxes
        )
        self.groups = self._source_groups()
        self.active = np.ones(env.task_config.cookie_count, dtype=bool)
        self.gaps: tuple[tuple[int, ...], ...] = ()
        self.tilted_group: tuple[int, ...] = ()
        self.tilt_deg = 0.0
        self.crushed_ids: list[int] = []
        self.peak_load_n = 0.0

        self._intact_geoms = tuple(
            np.flatnonzero(self.model.geom_bodyid == body) for body in env._cookie_bodies
        )
        self._crushed_geoms = np.empty(0, dtype=int)
        if options.cookie_crush_force_n is not None:
            self._crushed_geoms = np.array([
                env._id(mujoco.mjtObj.mjOBJ_GEOM, f"cookie_{i}_crushed")
                for i in range(env.task_config.cookie_count)
            ])
            self._intact_geoms = tuple(
                geoms[geoms != crushed]
                for geoms, crushed in zip(self._intact_geoms, self._crushed_geoms, strict=True)
            )
        self._contype = self.model.geom_contype.copy()
        self._conaffinity = self.model.geom_conaffinity.copy()
        self._gravcomp = self.model.body_gravcomp.copy()
        self._cookie_by_geom = {
            geom: i for i, geoms in enumerate(env._cookie_collision_geoms) for geom in geoms
        }
        self._finger_geoms = frozenset(env._left_finger_geoms)
        self._contact_force = np.empty(6)
        self._visual_geoms = tuple(
            env._id(mujoco.mjtObj.mjOBJ_GEOM, f"cookie_{i}_visual")
            for i in range(env.task_config.cookie_count)
        )
        self._lean_supports = ()
        if options.source_tilt_deg:
            self._lean_supports = tuple(
                self.model.body_mocapid[env._id(mujoco.mjtObj.mjOBJ_BODY, f"source_lean_support_{i}")]
                for i in range(COOKIES_PER_GROUP)
            )
        self.before_reset()

    def _source_groups(self) -> tuple[tuple[int, ...], ...]:
        columns: dict[float, list[int]] = {}
        for i, (x, _) in enumerate(self.env.SOURCE_POSITIONS):
            columns.setdefault(x, []).append(i)
        groups = []
        for indices in columns.values():
            indices.sort(key=lambda i: self.env.SOURCE_POSITIONS[i][1])
            for start in range(0, len(indices) - COOKIES_PER_GROUP + 1, COOKIES_PER_GROUP):
                groups.append(tuple(indices[start : start + COOKIES_PER_GROUP]))
        if self.gap_range[1] > len(groups) or self.options.source_tilt_deg and not groups:
            raise ValueError("scene variations need complete five-cookie groups in the source")
        return tuple(groups)

    def before_reset(self) -> None:
        for i in (*np.flatnonzero(~self.active), *self.crushed_ids):
            geoms = self._intact_geoms[i]
            self.model.geom_contype[geoms] = self._contype[geoms]
            self.model.geom_conaffinity[geoms] = self._conaffinity[geoms]
            self.model.body_gravcomp[self.env._cookie_bodies[i]] = self._gravcomp[
                self.env._cookie_bodies[i]
            ]
        self.model.geom_contype[self._crushed_geoms] = 0
        self.model.geom_conaffinity[self._crushed_geoms] = 0
        self.model.geom_rgba[self._crushed_geoms, 3] = 0
        self.active[:] = True
        self.crushed_ids.clear()
        self.peak_load_n = 0.0

    def move_target_boxes(self) -> None:
        if len(self.target_joints) == 1:
            return
        address = self.model.jnt_qposadr[self.target_joints[0]]
        pose = self.data.qpos[address : address + 7].copy()
        # Share the sampled translation and yaw. Each box yaws about its own
        # center, so a distant tray does not get an additional lever-arm shift.
        translation = pose[:3] - np.asarray(self.boxes[0][1])
        for joint, (_, nominal_position, _) in zip(
            self.target_joints[1:], self.boxes[1:], strict=True
        ):
            address = self.model.jnt_qposadr[joint]
            dof = self.model.jnt_dofadr[joint]
            self.data.qpos[address : address + 3] = np.asarray(nominal_position) + translation
            mujoco.mju_mulQuat(
                self.data.qpos[address + 3 : address + 7],
                pose[3:],
                self.model.qpos0[address + 3 : address + 7],
            )
            self.data.qvel[dof : dof + 6] = 0.0

    def arrange_source(self, seed: int) -> None:
        # Separate streams preserve the common scene choices when features are combined.
        low, high = self.gap_range
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7303]))
        gap_count = int(rng.integers(low, high + 1))
        selected = sorted(rng.choice(len(self.groups), size=gap_count, replace=False))
        self.gaps = tuple(self.groups[i] for i in selected)
        for group in self.gaps:
            for i in group:
                self.active[i] = False
                geoms = self._intact_geoms[i]
                self.model.geom_contype[geoms] = 0
                self.model.geom_conaffinity[geoms] = 0
                self.model.body_gravcomp[self.env._cookie_bodies[i]] = 1.0
                self.env.set_cookie_pose(i, (10.0 + 0.08 * i, 10.0, 1.0))

        if self._lean_supports:
            present = [group for group in self.groups if self.active[list(group)].all()]
            if not present:
                raise ValueError("source tilt needs at least one intact group of five")
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7304]))
            self.tilted_group = present[int(rng.integers(len(present)))]
            self.tilt_deg = float(self.options.source_tilt_deg * rng.choice((-1, 1)))
            self._lean_group()

    def _lean_group(self) -> None:
        # A thin sloped support under each flat base leaves the top and sides
        # free for pickup. The cookies are free bodies throughout the episode.
        theta = np.deg2rad(self.tilt_deg)
        tilt = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])
        half_height = self.env.COOKIE_HALF_SIZE[2]
        floor = self.env.SOURCE_FLOOR_TOP_Z
        for i, support in zip(self.tilted_group, self._lean_supports, strict=True):
            body = self.env._cookie_bodies[i]
            rotation = np.empty(4)
            mujoco.mju_mulQuat(rotation, self.data.xquat[body], tilt)
            normal = np.empty(3)
            mujoco.mju_rotVecQuat(normal, np.array([0.0, 0.0, 1.0]), rotation)
            base = self.data.xpos[body].copy()
            # Keep the projected cookie centers on the original lattice.
            base[:2] -= half_height * normal[:2]
            base[2] = floor + 0.0012
            self.data.mocap_pos[support] = base - LEAN_SUPPORT_HALF_HEIGHT_M * normal
            self.data.mocap_quat[support] = rotation
            self.env.set_cookie_pose(i, base + half_height * normal, rotation)

    def hide_inactive_geometry(self) -> None:
        for i in np.flatnonzero(~self.active):
            self.model.geom_rgba[self._intact_geoms[i], 3] = 0
        self.model.geom_rgba[self._crushed_geoms, 3] = 0

    def metadata(self) -> dict:
        tilts = {}
        for i in self.tilted_group:
            upright_z = self.data.xmat[self.env._cookie_bodies[i], 8]
            tilts[i] = float(np.rad2deg(np.arccos(np.clip(upright_z, -1, 1))))
        return {
            "gap_groups": [list(group) for group in self.gaps],
            "initial_source_count": int(self.active.sum()),
            "tilted_cookie_ids": list(self.tilted_group),
            "requested_tilt_deg": self.tilt_deg,
            "initial_tilt_deg": tilts,
            "target_boxes": [
                {
                    "name": name,
                    "position": self.data.xpos[body].tolist(),
                    "quaternion": self.data.xquat[body].tolist(),
                }
                for (name, _, _), body in zip(self.boxes, self.target_bodies, strict=True)
            ],
            "cookie_crush_force_n": self.options.cookie_crush_force_n,
        }

    def check_grip_load(self) -> None:
        threshold = self.options.cookie_crush_force_n
        if threshold is None:
            return
        pairs = self.data.contact.geom
        finger1, finger2 = self.env._left_finger_geoms
        finger_contacts = np.any((pairs == finger1) | (pairs == finger2), axis=1)
        loads: dict[tuple[int, int], float] = {}
        for index in np.flatnonzero(finger_contacts):
            contact = self.data.contact[index]
            first, second = contact.geom
            if second in self._finger_geoms:
                first, second = second, first
            if second not in self._cookie_by_geom:
                continue
            if contact.efc_address < 0:
                continue
            i = self._cookie_by_geom[second]
            mujoco.mj_contactForce(self.model, self.data, index, self._contact_force)
            key = (i, int(first))
            loads[key] = loads.get(key, 0.0) + max(0.0, float(self._contact_force[0]))
        for (i, _), load in loads.items():
            self.peak_load_n = max(self.peak_load_n, load)
            if load >= threshold and i not in self.crushed_ids:
                self._crush(i)

    def _crush(self, i: int) -> None:
        geoms = self._intact_geoms[i]
        crushed = self._crushed_geoms[i]
        rgba = self.model.geom_rgba[self._visual_geoms[i]].copy()
        self.model.geom_rgba[geoms, 3] = 0
        self.model.geom_contype[geoms] = 0
        self.model.geom_conaffinity[geoms] = 0
        self.model.geom_rgba[crushed] = rgba
        self.model.geom_contype[crushed] = self._contype[crushed]
        self.model.geom_conaffinity[crushed] = self._conaffinity[crushed]
        self.crushed_ids.append(i)
