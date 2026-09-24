from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .batch_plan import BatchPlan
from .config import SimConfig
from .contracts import ARM_JOINTS, ActionMode
from .env import A3DualArmEnv


def _yaw_quaternion(yaw: float) -> list[float]:
    """A rotation about z only, so a box never tips or lifts."""

    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _rotated_half_extent(half: np.ndarray, yaw: float) -> np.ndarray:
    """Half extent of a rectangle's axis-aligned bounding box after a yaw.

    A yawed box needs a wider berth than its unrotated half size, and the boxes
    are only tens of millimetres apart, so ignoring this would let the clearance
    check pass while the corners actually overlapped.
    """

    cos, sin = abs(math.cos(yaw)), abs(math.sin(yaw))
    return np.array(
        [cos * half[0] + sin * half[1], sin * half[0] + cos * half[1]],
        dtype=np.float64,
    )


def _rotate_xy(vector: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a 2-D offset about z, so a Cookie can follow its bin's yaw."""

    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array(
        [cos * vector[0] - sin * vector[1], sin * vector[0] + cos * vector[1]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CookieTransferTaskConfig:
    cookie_count: int = 80
    required_cookies: int = 10
    position_noise_m: float = 0.0
    yaw_noise_rad: float = 0.0
    success_hold_steps: int = 20
    max_linear_speed_m_s: float = 0.06
    max_angular_speed_rad_s: float = 0.30
    max_tilt_rad: float = np.deg2rad(15.0)
    terminate_on_success: bool = True
    require_exact_slots: bool = True
    require_released: bool = False
    #: Whether a Cookie has to still be standing to count as placed.
    #:
    #: The precision-fill scenes say yes: their contract is about putting ten
    #: Cookies into a 2x5 grid, and one lying on its side is not in its slot.  The
    #: relay scene says no, because its subject is the box exchange rather than the
    #: placement -- a Cookie shoved flat on its back by the push is still *in the
    #: box*, and rejecting the episode for it would grade the wrong thing.  A run
    #: that failed on exactly that reported "7/10 Cookies aboard" for a box holding
    #: all ten, three of them leaning about 24 degrees.
    require_upright: bool = True

    def __post_init__(self) -> None:
        if self.required_cookies != 10:
            raise ValueError("the 2x5 target contract requires exactly 10 cookies")


class A3CookieTransferEnv(A3DualArmEnv):
    """Video-inspired task: transfer packaged cookies from a large bin to a small bin."""

    @property
    def TARGET_SLOT_CENTERS(self) -> np.ndarray:
        return np.asarray(
            [
                self.privileged_target_slot_world(i, self._target_cookie_center_z)[:2]
                for i in range(len(self.TARGET_SLOTS_LOCAL))
            ]
        )

    CONTACT_CONTAINMENT_TOLERANCE_M = 0.004
    WALL_CONTACT_TOLERANCE_M = 0.026
    TARGET_FLOOR_TOP_Z = 0.775

    def __init__(
        self,
        config: SimConfig | str | Path | None = None,
        *,
        task_config: CookieTransferTaskConfig | None = None,
        action_mode: ActionMode = "joint_position",
        render_mode: str | None = None,
        render_cameras: bool = True,
    ) -> None:
        self.task_config = task_config or CookieTransferTaskConfig()
        if config is None:
            config = SimConfig(horizon=6000)
        super().__init__(
            config,
            action_mode=action_mode,
            render_mode=render_mode,
            render_cameras=render_cameras,
            scene="cookie_transfer",
        )
        scene_config = self.config.cookie_transfer
        if task_config is None:
            self.task_config = CookieTransferTaskConfig(
                cookie_count=len(scene_config.cookie_source_positions_m)
            )
        if self.task_config.cookie_count != len(scene_config.cookie_source_positions_m):
            raise ValueError("task cookie_count must match configured cookie source positions")
        self.SOURCE_POSITIONS = scene_config.cookie_source_positions_m
        #: Nominal source-bin centre, as configured.  The live centre is
        #: :attr:`SOURCE_CENTER`, which moves when the scene randomises the bin.
        self.SOURCE_NOMINAL_CENTER = np.asarray(
            scene_config.source_bin_center_m, dtype=np.float64
        )
        self.SOURCE_INNER_HALF_SIZE = (
            np.asarray(scene_config.source_bin_half_size_m, dtype=np.float64)
            - scene_config.bin_wall_thickness_m
        )
        self.TARGET_CENTER = np.zeros(2, dtype=np.float64)
        self.TARGET_INNER_HALF_SIZE = (
            np.asarray(scene_config.target_bin_half_size_m, dtype=np.float64)
            - scene_config.bin_wall_thickness_m
        )
        self.COOKIE_HALF_SIZE = np.asarray(scene_config.cookie_half_size_m, dtype=np.float64)
        # Row spacing, and the thickness of one jaw pad along the closing axis.
        # The expert needs both to place the pads in the groove between two
        # neighbours rather than on top of a Cookie's top face.
        _rows = sorted({float(p[1]) for p in scene_config.cookie_source_positions_m})
        self.COOKIE_PITCH_Y = (
            _rows[1] - _rows[0] if len(_rows) > 1 else 2.0 * self.COOKIE_HALF_SIZE[1]
        )
        self.TARGET_SLOTS_LOCAL = scene_config.target_slots_local_m
        self.TARGET_SLOT_TOLERANCE = np.asarray(
            scene_config.target_slot_tolerance_m, dtype=np.float64
        )
        #: How this box's slots are grouped into grasps; see :mod:`a3_dual_arm_sim.batch_plan`.
        #:
        #: The environment owns it because it owns the slot lattice, and both batch
        #: experts read it from here rather than each deriving their own -- a
        #: disagreement between the plan a batch is grasped by and the plan its
        #: slots are placed into is exactly the kind of mistake this makes
        #: impossible.  The column count is read off the lattice rather than
        #: configured, so it cannot drift from the slots it describes.
        self.batch_plan = BatchPlan.for_capacity(
            capacity=len(self.TARGET_SLOTS_LOCAL),
            per_grasp=self.config.batch_expert_per_grasp,
            columns=len({slot[0] for slot in self.TARGET_SLOTS_LOCAL}),
        )
        self.SOURCE_WALL_TOP_Z = (
            scene_config.source_wall_base_z_m + scene_config.source_bin_wall_height_m
        )
        self.TARGET_WALL_HEIGHT = scene_config.target_bin_wall_height_m
        self.SOURCE_FLOOR_TOP_Z = (
            scene_config.source_floor_z_m + scene_config.bin_wall_thickness_m / 2
        )
        self._target_cookie_center_z = (
            scene_config.target_floor_z_m
            + scene_config.bin_wall_thickness_m / 2
            + scene_config.cookie_half_size_m[2]
        )
        self.COOKIE_RESET_Z = scene_config.cookie_reset_z_m
        self.DEPLOYMENT_HOME = np.asarray(scene_config.deployment_home, dtype=np.float64)
        #: Home for the current episode; `reset` replaces it when the scene
        #: randomises the start pose.  Initialised so a caller that never resets
        #: still reads a sane pose.
        self._randomized_deployment_home = self.DEPLOYMENT_HOME.copy()
        self._scene_offset = np.zeros(2, dtype=np.float64)
        #: Source-bin yaw for this episode, so the Cookies can follow it.
        self._scene_yaw = 0.0
        self._scene_randomization_applied = False
        self._target_bin_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "target_bin")
        self._source_bin_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "source_bin")
        #: Every tabletop box, station first.  This is what the randomization and
        #: the clearance check iterate; ``_target_bin_body`` stays the station,
        #: because that is the one the environment's own success test grades.
        self._target_bin_bodies = tuple(
            self._id(mujoco.mjtObj.mjOBJ_BODY, name) for name in scene_config.target_bin_body_names
        )
        # -1 means "this scene has no second box", which is how the single-box
        # configs are built.  Kept as its own attribute because the relay reads it
        # to find box B, and because ``spare_target_bin`` is the name a config
        # states it by.
        self._spare_bin_body = self._id_or_none(
            mujoco.mjtObj.mjOBJ_BODY, "spare_target_bin"
        )
        self._cookie_bodies = tuple(
            self._id(mujoco.mjtObj.mjOBJ_BODY, f"cookie_{index}")
            for index in range(self.task_config.cookie_count)
        )
        self._cookie_joints = tuple(
            self._id(mujoco.mjtObj.mjOBJ_JOINT, f"cookie_{index}_free")
            for index in range(self.task_config.cookie_count)
        )
        # A Cookie's collision geom is several geoms rather than one -- the beveled
        # shape is built as a compound, and a mesh geom has to be face-split -- so
        # each Cookie maps to a set.  `_cookie_geoms` names the primary (collision)
        # geom per Cookie, which is what the batch expert labels a graph node with,
        # and `_cookie_collision_geoms` is the whole set that contact checks match
        # against.
        self._cookie_geoms = tuple(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"cookie_{index}_geom")
            for index in range(self.task_config.cookie_count)
        )
        self._cookie_collision_geoms = tuple(
            frozenset(
                geom for geom in range(self.model.ngeom)
                if self.model.geom_bodyid[geom] == body
                and self.model.geom_contype[geom] != 0
            )
            for body in self._cookie_bodies
        )
        self._left_finger_geoms = tuple(
            self._id(mujoco.mjtObj.mjOBJ_GEOM, f"L_finger_{finger}_geom")
            for finger in ("inner", "outer")
        )
        # Thickness of one jaw pad along the closing axis.  The pads have to sit in
        # the groove between two neighbours, so the expert sizes its descent
        # opening from this and COOKIE_PITCH_Y.
        self.PAD_THICKNESS_M = 2.0 * float(
            self.model.geom_size[self._left_finger_geoms[0]][1]
        )
        self._success_hold_count = 0
        self._source_initially_filled = False

    @property
    def cookie_positions(self) -> np.ndarray:
        return np.asarray([self.data.xpos[body_id].copy() for body_id in self._cookie_bodies])

    @property
    def target_bin_bodies(self) -> tuple[int, ...]:
        """Body ids of every tabletop box, station first.

        The station is ``target_bin_bodies[0]``, which is also
        ``_target_bin_body`` -- the box the environment's own success criterion
        grades.  A scene with one box has a one-element tuple.
        """

        return self._target_bin_bodies

    @property
    def success_hold_count(self) -> int:
        return self._success_hold_count

    def privileged_cookie_position(self, index: int) -> np.ndarray:
        """Return simulator-truth position for expert control, never policy input."""
        return self.data.xpos[self._cookie_bodies[index]].copy()

    def privileged_cookie_target_position(self, index: int) -> np.ndarray:
        """Return Cookie position in the moving target-bin frame."""
        target_position = self.data.xpos[self._target_bin_body]
        target_rotation = self.data.xmat[self._target_bin_body].reshape(3, 3)
        return target_rotation.T @ (self.data.xpos[self._cookie_bodies[index]] - target_position)

    def privileged_target_slot_world(self, slot_index: int, z: float) -> np.ndarray:
        """Convert a configured target slot into a simulator-truth world point."""
        target_position = self.data.xpos[self._target_bin_body]
        target_rotation = self.data.xmat[self._target_bin_body].reshape(3, 3)
        slot_xy = self.TARGET_SLOTS_LOCAL[slot_index]
        return target_position + target_rotation @ np.asarray(
            [slot_xy[0], slot_xy[1], z], dtype=np.float64
        )

    def _cookie_rotation(self, index: int) -> np.ndarray:
        """Orientation of the Cookie's body frame.

        Deliberately *not* ``geom_xmat``.  The Cookie's collision geom is a convex
        hull, and MuJoCo stores a mesh in its principal-axis frame, folding that
        rotation into the geom frame.  For the Cookie hull that is a 120 degree
        rotation about (1, 1, 1), so ``abs(geom_xmat) @ COOKIE_HALF_SIZE`` reports
        a y half-extent of 25 mm instead of 3.17 mm -- which made the containment
        checks read an upright Cookie as lying on its side.  The freejoint is on
        the body, so the body frame is the one that tracks the real orientation.
        """
        return self.data.xmat[self._cookie_bodies[index]].reshape(3, 3)

    def privileged_left_finger_contacts(self, index: int) -> tuple[bool, bool]:
        """Report target-Cookie contact for each left finger from MuJoCo contacts."""
        cookie_geoms = self._cookie_collision_geoms[index]
        contacts = [False, False]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            for finger_index, finger_geom in enumerate(self._left_finger_geoms):
                if finger_geom in pair and bool(pair & cookie_geoms):
                    contacts[finger_index] = True
        return contacts[0], contacts[1]

    def privileged_left_touch_values(self) -> tuple[float, float]:
        """Return left fingertip touch signals for expert-only grasp verification."""
        return (
            float(self._sensor("L_finger_inner_touch_sensor")[0]),
            float(self._sensor("L_finger_outer_touch_sensor")[0]),
        )

    @property
    def SOURCE_CENTER(self) -> np.ndarray:
        """Live source-bin centre in world XY.

        A property, not a constant, because the scene can move the bin between
        episodes (:class:`SceneRandomization`) and every containment check has to
        follow it -- otherwise a moved bin would report its Cookies as spilled.
        Falls back to the configured centre for a model without the mocap body.
        """

        body = getattr(self, "_source_bin_body", -1)
        if body is None or body < 0:
            return self.SOURCE_NOMINAL_CENTER.copy()
        return np.asarray(self.data.xpos[body][:2], dtype=np.float64)

    def privileged_cookie_in_source(self, index: int) -> bool:
        return self._cookie_inside_source(index)

    def privileged_cookie_in_target(self, index: int) -> bool:
        return self._cookie_inside_target(index)

    def privileged_cookie_in_target_region(self, index: int) -> bool:
        """Check target-bin geometry without requiring the Cookie to be settled."""
        local_position = self.privileged_cookie_target_position(index)
        return bool(
            abs(local_position[0])
            <= self.TARGET_INNER_HALF_SIZE[0] + self.CONTACT_CONTAINMENT_TOLERANCE_M
            and abs(local_position[1])
            <= self.TARGET_INNER_HALF_SIZE[1] + self.CONTACT_CONTAINMENT_TOLERANCE_M
            and -0.005
            <= local_position[2]
            <= self.TARGET_WALL_HEIGHT + self.CONTACT_CONTAINMENT_TOLERANCE_M + 0.035
        )

    def privileged_cookie_in_slot(self, index: int, slot_index: int) -> bool:
        if not self._cookie_inside_target(index):
            return False
        local_position = self.privileged_cookie_target_position(index)
        target_xy = np.asarray(self.TARGET_SLOTS_LOCAL[slot_index], dtype=np.float64)
        return bool(np.all(np.abs(local_position[:2] - target_xy) <= self.TARGET_SLOT_TOLERANCE))

    def _apply_scene_randomization(self) -> np.ndarray:
        """Move the boxes and jitter the start pose for this episode.

        Returns the source-bin XY offset, which the Cookie layout is then placed
        relative to.  Sizes come from :class:`SceneRandomization`; everything is
        drawn in an order that does not change the rest of the reset, so a config
        with all-zero ranges is byte-identical to one without this feature.

        All three boxes are drawn together and then checked as a set, because the
        constraint that matters is between them rather than on any one of them: the
        source and target boxes are only tens of millimetres apart, and a draw that
        puts them into each other is discarded and redrawn.  The source box is a
        mocap body with infinite mass, so such a collision would shove the target
        box out of the pose the expert planned for instead of being resolved
        between them.

        Only XY and yaw are drawn, so every box keeps its configured height and
        never starts floating or sunk.

        ``_randomized_deployment_home`` is stored rather than applied here, because
        ``reset`` writes the home after this returns.
        """

        scene = self.config.randomization
        self._randomized_deployment_home = self.DEPLOYMENT_HOME.copy()
        self._scene_randomization_applied = scene.enabled
        self._scene_yaw = 0.0
        if not scene.enabled:
            return np.zeros(2)

        rng = self.np_random
        cookie_scene = self.config.cookie_transfer
        nominal = [
            np.asarray(position[:2], dtype=np.float64)
            for position in cookie_scene.target_bin_world_positions_m
        ]
        # One range triple per box.  The station draws from the `target_bin_*`
        # ranges and every box behind it from the `spare_bin_*` ones, because those
        # are roles rather than indices: whatever ends up in the queue has to arrive
        # at the station inside the fill's window, which is exactly what bounds the
        # spare's ranges (its x to +/-4 mm, because the carry corrects y and not x).
        # A third box is in that position too, so it gets that bound as well.
        range_triples = [(scene.target_bin_x_m, scene.target_bin_y_m, scene.target_bin_yaw_rad)] + [
            (scene.spare_bin_x_m, scene.spare_bin_y_m, scene.spare_bin_yaw_rad)
        ] * (len(nominal) - 1)

        for attempt in range(1, scene.max_clearance_attempts + 1):
            source_offset = np.array(
                [scene.source_bin_x_m.draw(rng), scene.source_bin_y_m.draw(rng)]
            )
            source_yaw = scene.source_bin_yaw_rad.draw(rng)
            drawn = [
                (
                    centre + np.array([x_range.draw(rng), y_range.draw(rng)]),
                    yaw_range.draw(rng),
                )
                for centre, (x_range, y_range, yaw_range) in zip(
                    nominal, range_triples, strict=True
                )
            ]
            if self._boxes_are_clear(source_offset, source_yaw, drawn):
                break
        else:
            raise RuntimeError(
                f"no layout with {scene.min_box_clearance_m * 1000:.1f} mm of box "
                f"clearance in {scene.max_clearance_attempts} draws for "
                f"{len(nominal)} tabletop boxes; the randomization ranges ask for a "
                f"layout that cannot exist. Narrow the box ranges, lower "
                f"min_box_clearance_m, or lay the boxes further apart -- a lane's "
                f"queue gap has to leave room for the queue's own y range."
            )

        # 1. Source bin.  Cookies are placed relative to the live body, so this
        #    also carries them; `reset` adds the offset to every layout position.
        mocap_id = int(self.model.body_mocapid[self._source_bin_body])
        if mocap_id < 0:
            raise RuntimeError("the source_bin body is not a mocap body")
        nominal = self.SOURCE_NOMINAL_CENTER
        self.data.mocap_pos[mocap_id] = [
            nominal[0] + source_offset[0],
            nominal[1] + source_offset[1],
            0.0,
        ]
        self.data.mocap_quat[mocap_id] = _yaw_quaternion(source_yaw)
        self._scene_yaw = float(source_yaw)

        # 2. Every tabletop box.  Written from the same draw the clearance check
        #    approved, so what the check passed is what is placed.
        for body, (centre, yaw) in zip(self._target_bin_bodies, drawn, strict=True):
            self._place_free_box(body, centre, yaw)

        # 4. Arm start pose.  This is the variation that matters most for a policy:
        #    with it at zero every episode begins from a byte-identical pose, so a
        #    policy can score well by memorising one trajectory.
        if scene.arm_home_rad.movable:
            jitter = np.array(
                [scene.arm_home_rad.draw(rng) for _ in range(7)], dtype=np.float64
            )
            for side, base in (("left", 0), ("right", 8)):
                self._randomized_deployment_home[base : base + 7] += jitter
        return source_offset

    @property
    def _source_bin_half_xy(self) -> np.ndarray:
        """Source bin half size in XY, walls included, from the config.

        The config rather than the geoms: the clearance check runs before the body
        has been moved into place, so the live extents would describe the previous
        episode's pose.
        """

        scene = self.config.cookie_transfer
        return np.asarray(scene.source_bin_half_size_m, dtype=np.float64) + float(
            scene.bin_wall_thickness_m
        )

    @property
    def _target_bin_half_xy(self) -> np.ndarray:
        """Half size in XY of a tabletop box, walls included."""

        scene = self.config.cookie_transfer
        return np.asarray(scene.target_bin_half_size_m, dtype=np.float64) + float(
            scene.bin_wall_thickness_m
        )

    def _boxes_are_clear(
        self,
        source_offset: np.ndarray,
        source_yaw: float,
        boxes: list[tuple[np.ndarray, float]],
    ) -> bool:
        """Whether the drawn layout leaves every pair of boxes far enough apart.

        A yawed box needs a bigger berth than its half size suggests, so each pair
        is checked against the axis-aligned extent of the *rotated* rectangle.  Two
        boxes are apart if they are apart on either axis, which is the largest
        per-axis gap rather than the smallest -- boxes side by side in y are not
        overlapping merely because they share an x range.

        ``boxes`` is every tabletop box as ``(centre, yaw)``, station first, and the
        source bin is checked against all of them.  That is a quadratic loop over a
        handful of boxes, which is the honest shape of the constraint: it is a
        property of the *set*, and a lane of three has three pairs to keep apart
        rather than one.
        """

        clearance = self.config.randomization.min_box_clearance_m
        source_half = self._source_bin_half_xy
        target_half = self._target_bin_half_xy
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = [
            (self.SOURCE_NOMINAL_CENTER + source_offset, source_half, source_yaw),
            *((centre, target_half, yaw) for centre, yaw in boxes),
        ]
        for index, (centre, half, yaw) in enumerate(candidates):
            for other_centre, other_half, other_yaw in candidates[index + 1 :]:
                extent = _rotated_half_extent(half, yaw)
                other_extent = _rotated_half_extent(other_half, other_yaw)
                gap = np.maximum(
                    (centre - extent) - (other_centre + other_extent),
                    (other_centre - other_extent) - (centre + extent),
                )
                if float(np.max(gap)) < clearance:
                    return False
        return True

    def _place_free_box(self, body: int, xy: np.ndarray, yaw: float) -> None:
        """Write a free box's pose, keeping its configured height and roll/pitch."""

        joint = self.model.body_jntadr[body]
        address = self.model.jnt_qposadr[joint]
        nominal_z = float(self.data.qpos[address + 2])
        self.data.qpos[address : address + 3] = [xy[0], xy[1], nominal_z]
        self.data.qpos[address + 3 : address + 7] = _yaw_quaternion(yaw)
        self.data.qvel[
            self.model.jnt_dofadr[joint] : self.model.jnt_dofadr[joint] + 6
        ] = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = super().reset(
            seed=seed,
            options={"randomize_objects": False},
        )
        randomize = (options or {}).get("randomize_cookies", True)
        # Two independent switches, because they cost different things:
        #
        #   `randomize_scene`   moves the boxes and the arm start pose.  It shifts
        #                       the whole Cookie layout by one translation, so the
        #                       2.5 mm gaps the batch insertion depends on are
        #                       exactly preserved -- a batch scene can and should
        #                       have this on.
        #   `randomize_cookies` jitters each Cookie on its own, which does change
        #                       those gaps, so only the single-Cookie scene can
        #                       afford it.
        randomize_scene = (options or {}).get("randomize_scene", True)
        self._scene_offset = (
            self._apply_scene_randomization() if randomize_scene else np.zeros(2)
        )
        home = self._randomized_deployment_home if randomize_scene else self.DEPLOYMENT_HOME
        arm_values = np.r_[home[:7], home[8:15]]
        for name, value in zip(ARM_JOINTS, arm_values, strict=True):
            self.data.qpos[self._qpos_ids[name]] = value
            self.data.qvel[self._dof_ids[name]] = 0.0
        for side in ("L", "R"):
            for joint_id in self._finger_joints[side]:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = 0.0
                self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
        self._last_applied_action = home.copy()
        self._apply_controls(home)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        for index, (base_x, base_y) in enumerate(self.SOURCE_POSITIONS):
            # The layout is expressed in the source bin's own frame, so the bin's
            # drawn pose carries the Cookies: rotate about the bin centre first,
            # then translate.  Rotating the container without its contents would
            # leave the Cookies axis-aligned inside a skewed bin -- the rows the
            # expert enters would no longer be the rows the bin is holding.
            local = np.array([base_x, base_y]) - self.SOURCE_NOMINAL_CENTER
            turned = _rotate_xy(local, self._scene_yaw)
            base = self.SOURCE_NOMINAL_CENTER + turned + self._scene_offset
            base_x, base_y = float(base[0]), float(base[1])
            if randomize:
                dx, dy = self.np_random.uniform(
                    -self.task_config.position_noise_m,
                    self.task_config.position_noise_m,
                    size=2,
                )
                yaw = float(
                    self.np_random.uniform(
                        -self.task_config.yaw_noise_rad,
                        self.task_config.yaw_noise_rad,
                    )
                )
            else:
                dx = dy = yaw = 0.0
            # A Cookie inherits the bin's yaw, plus its own jitter.
            orientation = yaw + self._scene_yaw
            quaternion = (np.cos(orientation / 2), 0.0, 0.0, np.sin(orientation / 2))
            self.set_cookie_pose(index, (base_x + dx, base_y + dy, self.COOKIE_RESET_Z), quaternion)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self._success_hold_count = 0
        source_mask = tuple(
            self._cookie_inside_source(index) for index in range(self.task_config.cookie_count)
        )
        self._source_initially_filled = all(source_mask) and self._collection_touches_all_walls(
            source_mask,
            self.SOURCE_CENTER,
            self.SOURCE_INNER_HALF_SIZE,
        )
        observation = self._observation()
        info.update(
            success=False,
            cookies_in_target=0,
            cookies_in_source=sum(source_mask),
            required_cookies=self.task_config.required_cookies,
            cookies_in_target_mask=[False] * self.task_config.cookie_count,
            cookies_in_source_mask=source_mask,
            target_slot_occupancy=[-1] * self.task_config.required_cookies,
            source_initially_filled=self._source_initially_filled,
        )
        return observation, info

    def set_cookie_pose(
        self,
        index: int,
        position: tuple[float, float, float],
        quaternion: tuple[float, float, float, float] | None = None,
    ) -> None:
        joint_id = self._cookie_joints[index]
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        pos = np.asarray(position, dtype=np.float64)
        if hasattr(self, "_target_bin_body"):
            tb_pos = self.data.xpos[self._target_bin_body]
            tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
            p_rel = tb_mat.T @ (pos - tb_pos)
            if (
                abs(p_rel[0]) <= self.TARGET_INNER_HALF_SIZE[0] + 0.02
                and abs(p_rel[1]) <= self.TARGET_INNER_HALF_SIZE[1] + 0.02
            ):
                p_rel[2] = self._target_cookie_center_z
                pos = tb_pos + tb_mat @ p_rel
                if quaternion is None:
                    target_q = np.empty(4, dtype=np.float64)
                    mujoco.mju_mat2Quat(target_q, tb_mat.reshape(-1))
                    quaternion = tuple(target_q)
        if quaternion is None:
            quaternion = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[qpos_address : qpos_address + 3] = pos
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _cookie_speed(self, index: int, *, relative_to_target: bool = False) -> tuple[float, float]:
        dof_address = int(self.model.jnt_dofadr[self._cookie_joints[index]])
        velocity = self.data.qvel[dof_address : dof_address + 6].copy()
        if relative_to_target and hasattr(self, "_target_bin_body"):
            tb_cvel = self.data.cvel[self._target_bin_body]
            r = self.data.xpos[self._cookie_bodies[index]] - self.data.xpos[self._target_bin_body]
            v_expected = tb_cvel[3:] + np.cross(tb_cvel[:3], r)
            rel_v = velocity[:3] - v_expected
            rel_w = velocity[3:] - tb_cvel[:3]
            return float(np.linalg.norm(rel_v)), float(np.linalg.norm(rel_w))
        return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))

    def _cookie_region_status(
        self,
        index: int,
        center: np.ndarray,
        inner_half_size: np.ndarray,
        wall_top_z: float,
        *,
        floor_top_z: float = 0.753,
        require_upright: bool,
        require_settled: bool,
    ) -> bool:
        position = self.data.xpos[self._cookie_bodies[index]]
        rotation = self._cookie_rotation(index)
        world_half_extent = np.abs(rotation) @ self.COOKIE_HALF_SIZE
        lower_xy = center - inner_half_size
        upper_xy = center + inner_half_size
        footprint_inside = bool(
            np.all(
                position[:2] - world_half_extent[:2]
                >= lower_xy - self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
            and np.all(
                position[:2] + world_half_extent[:2]
                <= upper_xy + self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
        )
        vertically_inside = bool(
            position[2] - world_half_extent[2] >= floor_top_z - 0.003
            and position[2] - world_half_extent[2]
            <= wall_top_z + self.CONTACT_CONTAINMENT_TOLERANCE_M
        )
        upright = bool(abs(float(rotation[2, 2])) >= np.cos(self.task_config.max_tilt_rad))
        linear_speed, angular_speed = self._cookie_speed(index)
        settled = (
            linear_speed <= self.task_config.max_linear_speed_m_s
            and angular_speed <= self.task_config.max_angular_speed_rad_s
        )
        return (
            footprint_inside
            and vertically_inside
            and (upright or not require_upright)
            and (settled or not require_settled)
        )

    def _cookie_inside_target(self, index: int) -> bool:
        tb_pos = self.data.xpos[self._target_bin_body]
        tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
        c_pos = self.data.xpos[self._cookie_bodies[index]]
        c_mat = self._cookie_rotation(index)

        p_rel = tb_mat.T @ (c_pos - tb_pos)
        r_rel = tb_mat.T @ c_mat
        half_extent = np.abs(r_rel) @ self.COOKIE_HALF_SIZE

        footprint_inside = bool(
            np.all(
                np.abs(p_rel[:2]) + half_extent[:2]
                <= self.TARGET_INNER_HALF_SIZE + self.CONTACT_CONTAINMENT_TOLERANCE_M
            )
        )
        floor_top = (
            self.config.cookie_transfer.target_floor_z_m
            + self.config.cookie_transfer.bin_wall_thickness_m / 2
        )
        vertically_inside = bool(
            p_rel[2] - half_extent[2] >= floor_top - 0.003
            and p_rel[2] - half_extent[2] < self.TARGET_WALL_HEIGHT
        )
        if self.task_config.require_released:
            vertically_inside = vertically_inside and p_rel[2] - half_extent[2] <= floor_top + 0.004
        if not footprint_inside or not vertically_inside:
            return False
        upright = bool(
            abs(float(r_rel[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
            or abs(float(c_mat[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
        )
        linear_speed, angular_speed = self._cookie_speed(index, relative_to_target=True)
        settled = (
            linear_speed <= max(self.task_config.max_linear_speed_m_s, 0.09)
            and angular_speed <= self.task_config.max_angular_speed_rad_s
        )
        released = (
            not self.task_config.require_released
            or not any(self.privileged_left_finger_contacts(index))
        )
        return (
            footprint_inside
            and vertically_inside
            and (upright or not self.task_config.require_upright)
            and settled
            and released
        )

    def _cookie_inside_source(self, index: int) -> bool:
        return self._cookie_region_status(
            index,
            self.SOURCE_CENTER,
            self.SOURCE_INNER_HALF_SIZE,
            self.SOURCE_WALL_TOP_Z,
            floor_top_z=self.SOURCE_FLOOR_TOP_Z,
            require_upright=False,
            require_settled=False,
        )

    def _target_slot_occupancy(self, in_target: tuple[bool, ...]) -> tuple[int, ...]:
        tb_pos = self.data.xpos[self._target_bin_body]
        tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
        occupancy = [-1] * len(self.TARGET_SLOTS_LOCAL)
        for cookie_index, eligible in enumerate(in_target):
            if not eligible:
                continue
            c_pos = self.data.xpos[self._cookie_bodies[cookie_index]]
            p_rel = tb_mat.T @ (c_pos - tb_pos)
            distances = np.abs(np.asarray(self.TARGET_SLOTS_LOCAL) - p_rel[:2])
            matches = np.flatnonzero(np.all(distances <= self.TARGET_SLOT_TOLERANCE, axis=1))
            if len(matches) == 1 and occupancy[int(matches[0])] < 0:
                occupancy[int(matches[0])] = cookie_index
        return tuple(occupancy)

    def _collection_touches_all_walls(
        self,
        included: tuple[bool, ...],
        center: np.ndarray,
        inner_half_size: np.ndarray,
        *,
        is_target: bool = False,
    ) -> bool:
        edges: list[tuple[np.ndarray, np.ndarray]] = []
        if is_target:
            tb_pos = self.data.xpos[self._target_bin_body]
            tb_mat = self.data.xmat[self._target_bin_body].reshape(3, 3)
            for index, use_cookie in enumerate(included):
                if not use_cookie:
                    continue
                c_pos = self.data.xpos[self._cookie_bodies[index]]
                p_rel = tb_mat.T @ (c_pos - tb_pos)
                r_rel = tb_mat.T @ self._cookie_rotation(index)
                half_extent = (np.abs(r_rel) @ self.COOKIE_HALF_SIZE)[:2]
                edges.append((p_rel[:2] - half_extent, p_rel[:2] + half_extent))
            inner_lower = -inner_half_size
            inner_upper = inner_half_size
        else:
            for index, use_cookie in enumerate(included):
                if not use_cookie:
                    continue
                position = self.data.xpos[self._cookie_bodies[index]][:2]
                rotation = self._cookie_rotation(index)
                half_extent = (np.abs(rotation) @ self.COOKIE_HALF_SIZE)[:2]
                edges.append((position - half_extent, position + half_extent))
            inner_lower = center - inner_half_size
            inner_upper = center + inner_half_size

        if not edges:
            return False
        lower = np.min(np.asarray([edge[0] for edge in edges]), axis=0)
        upper = np.max(np.asarray([edge[1] for edge in edges]), axis=0)
        return bool(
            np.all(lower <= inner_lower + self.WALL_CONTACT_TOLERANCE_M)
            and np.all(upper >= inner_upper - self.WALL_CONTACT_TOLERANCE_M)
        )

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, _, safety_terminated, truncated, info = super().step(action)
        in_target = tuple(
            self._cookie_inside_target(index) for index in range(self.task_config.cookie_count)
        )
        in_source = tuple(
            self._cookie_inside_source(index) for index in range(self.task_config.cookie_count)
        )
        target_count = sum(in_target)
        source_count = sum(in_source)
        occupancy = self._target_slot_occupancy(in_target)
        target_touches_all_walls = self._collection_touches_all_walls(
            in_target,
            self.TARGET_CENTER,
            self.TARGET_INNER_HALF_SIZE,
            is_target=True,
        )
        exact_fill = (
            target_count == self.task_config.required_cookies
            and source_count == self.task_config.cookie_count - self.task_config.required_cookies
            and all(index >= 0 for index in occupancy)
            and target_touches_all_walls
        )
        count_fill = (
            target_count == self.task_config.required_cookies
            and source_count == self.task_config.cookie_count - self.task_config.required_cookies
        )
        task_filled = exact_fill if self.task_config.require_exact_slots else count_fill
        self._success_hold_count = self._success_hold_count + 1 if task_filled else 0
        success = self._success_hold_count >= self.task_config.success_hold_steps
        terminated = safety_terminated or (success and self.task_config.terminate_on_success)
        reward = min(target_count / self.task_config.required_cookies, 1.0)
        info.update(
            success=success,
            cookies_in_target=target_count,
            cookies_in_source=source_count,
            required_cookies=self.task_config.required_cookies,
            cookies_in_target_mask=in_target,
            cookies_in_source_mask=in_source,
            target_slot_occupancy=occupancy,
            target_touches_all_walls=target_touches_all_walls,
            source_initially_filled=self._source_initially_filled,
            exact_2x5_fill=exact_fill,
            count_fill=count_fill,
            success_criterion="exact_slots" if self.task_config.require_exact_slots else "released_count",
            success_hold_count=self._success_hold_count,
            required_success_hold_steps=self.task_config.success_hold_steps,
            cookie_positions=self.cookie_positions,
        )
        return observation, reward, terminated, truncated and not terminated, info
