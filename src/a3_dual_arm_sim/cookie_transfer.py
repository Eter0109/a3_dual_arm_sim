from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import SimConfig
from .contracts import ARM_JOINTS, LEFT_JOINTS, ActionMode
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


#: Jaw opening at the start of a LIBERO episode, in metres.
#:
#: **Not** 0.0387, which is where the jaws end up after the first "open" command and what an
#: earlier version of this file used.  It is 0.020833 -- half of the travel -- and it is the
#: same in all fifty of LIBERO's initial states.  Measured on the delivered evaluation, the
#: policy's first `observation.state` ends with `[0.020833, -0.020833]`: half open, and the
#: state it is told about, not a detail of the mechanism.
LIBERO_PANDA_FINGER_OPEN_M = 0.020833


def libero_init_arm(path: str, seed: int) -> tuple[np.ndarray, float]:
    """LIBERO's own start pose for episode `seed`: seven arm joints and the jaw opening.

    Read from the `.pruned_init` archive LIBERO ships -- 50 states per task, one per episode
    -- whose first row is laid out `[unused, seven arm joints, two jaws, objects...]`.  The
    arm is *not* the same from one episode to the next: measured, each joint varies with a
    standard deviation near a degree and a range of four to six, so a single fixed home is
    not any of the poses the policy was evaluated from.  The jaws are constant at 0.020833 in
    all fifty.

    `torch` is imported here rather than at module scope: the archive is a Torch blob and the
    rest of this package is importable without it.
    """

    import torch

    resolved = Path(path)
    if not resolved.is_absolute():
        from libero.libero import get_libero_path

        resolved = Path(get_libero_path("init_states")) / resolved
    path = str(resolved)
    states = torch.load(path, weights_only=False)
    row = np.asarray(states[int(seed) % len(states)], dtype=np.float64).ravel()
    if row.size < 10:
        raise ValueError(f"{path} holds {row.size} numbers; LIBERO's layout needs 10")
    return row[1:8], float(row[8])


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
        # The count is the scene's, not a constant.  The rule below is written in terms of
        # `required_cookies` already and needs no change to grade one piece instead of ten
        # -- what stopped that from being expressible was this check, which refused any
        # value but ten.
        if not 0 < self.required_cookies <= self.cookie_count:
            raise ValueError(
                f"required_cookies must be between 1 and cookie_count "
                f"({self.cookie_count}), got {self.required_cookies}"
            )
        # The ten belongs to the *exact-slot* contract rather than to the class: a 2x5 grid
        # has ten slots, and filling it means landing on all ten.  Tying it to
        # `require_exact_slots` keeps the constraint where it is actually true, so the
        # one-Cookie scene can opt out by opting out of the slot grid.
        if self.require_exact_slots and self.required_cookies != 10:
            raise ValueError("the exact-slot 2x5 contract requires exactly 10 cookies")


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
        #
        # The pitch is a property of the **bin**, so when the caller knows it -- a scene that
        # instantiates two Cookies out of a packed column does -- that value is used.  Deriving it
        # from the instantiated list is only right when the list *is* the packed column: a sparse
        # layout 70.7 mm apart made the pitch 70.7 mm, which put the descent opening at 0.51
        # instead of 0.08 and stopped the grasp verifying on every seed.
        configured_pitch = scene_config.cookie_source_row_pitch_m
        _rows = sorted({float(p[1]) for p in scene_config.cookie_source_positions_m})
        self.COOKIE_PITCH_Y = (
            float(configured_pitch)
            if configured_pitch is not None
            else (_rows[1] - _rows[0] if len(_rows) > 1 else 2.0 * self.COOKIE_HALF_SIZE[1])
        )
        self.TARGET_SLOTS_LOCAL = scene_config.target_slots_local_m
        # One sample per control step, for `_cookie_average_speed`.
        self._cookie_pose_history: dict[int, list[tuple[float, np.ndarray, np.ndarray]]] = {}
        self.TARGET_SLOT_TOLERANCE = np.asarray(
            scene_config.target_slot_tolerance_m, dtype=np.float64
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
        self._libero_init_states = str(getattr(scene_config, "libero_init_states", ""))
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
        # -1 means "this scene has no spare box", which is how the single-box
        # configs are built; `_apply_scene_randomization` skips it then.
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
        # **A jaw is a chain, not a body, and a chain is resolved by walking the tree rather than
        # by matching names.**  Each Panda finger is a finger body plus a tip body and *both* carry
        # collision meshes, so naming one geom per jaw -- which is what the query used to do -- can
        # only ever see the contacts that happen to land on that one mesh.  Measured through a real
        # grasp: nine contacts, every one of them on `gripper0_finger{1,2}_pad_collision`, while the
        # query looked at `L_finger_{inner,outer}_geom` on the finger bodies and reported no contact
        # at all.  Which of the two meshes touches depends on the pose, so no single name can be
        # right.
        #
        # The root is the finger joint the environment already resolves for its travel and sign, and
        # the descendants come from `body_parentid`; nothing here reads a name, because a name that
        # silently resolves to the wrong element is what this is repairing.
        self._left_jaw_geoms = tuple(self._jaw_geoms(finger) for finger in ("inner", "outer"))
        # Thickness of one jaw pad along the closing axis.  The pads have to sit in
        # the groove between two neighbours, so the expert sizes its descent
        # opening from this and COOKIE_PITCH_Y.
        self.PAD_THICKNESS_M = 2.0 * float(
            self.model.geom_size[self._left_finger_geoms[0]][1]
        )
        self._success_hold_count = 0
        self._source_initially_filled = False
        # The second-batch condition this *episode* runs under; drawn per reset from
        # `second_batch_modes` by the seed.  "full" until a reset says otherwise, so an
        # expert that reads it before any reset (a policy built ahead of the first
        # episode, a probe) sees the baseline rather than an unset attribute.
        self.second_batch_mode = "full"
        self._episode_seed: int | None = None

    @property
    def cookie_positions(self) -> np.ndarray:
        return np.asarray([self.data.xpos[body_id].copy() for body_id in self._cookie_bodies])

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

    def _jaw_geoms(self, finger: str) -> frozenset[int]:
        """Every geom that can collide, on one left jaw and the bodies below it.

        The tip body carries a small pad and the finger body carries the broad mesh, and which of
        them meets the object depends on how the jaws come in -- so the set has to be the whole
        chain.  Bodies are found by descending from the finger joint's body, not by name.
        """

        root = int(
            self.model.jnt_bodyid[
                self._id(mujoco.mjtObj.mjOBJ_JOINT, f"L_finger_{finger}_joint")
            ]
        )
        bodies = {root}
        for body in range(self.model.nbody):
            ancestor = int(self.model.body_parentid[body])
            while ancestor > 0:
                if ancestor in bodies:
                    bodies.add(body)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        return frozenset(
            geom
            for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) in bodies
            and (int(self.model.geom_contype[geom]) or int(self.model.geom_conaffinity[geom]))
        )

    def _jaw_contact_force(self, jaw_geoms: frozenset[int], cookie_geoms: frozenset[int]) -> float:
        """Normal force this jaw is applying to this object, summed over their contacts.

        The magnitude, not the sign: a contact pushes its two bodies apart, and which of the pair
        MuJoCo reports first is an accident of the geoms' order.
        """

        force = np.zeros(6, dtype=np.float64)
        total = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & cookie_geoms) or not (pair & jaw_geoms):
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            total += abs(float(force[0]))
        return total

    def privileged_left_finger_contacts(self, index: int) -> tuple[bool, bool]:
        """Report target-Cookie contact for each left jaw, over the jaw's whole chain."""
        cookie_geoms = self._cookie_collision_geoms[index]
        contacts = [False, False]
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & cookie_geoms):
                continue
            for jaw_index, jaw_geoms in enumerate(self._left_jaw_geoms):
                if pair & jaw_geoms:
                    contacts[jaw_index] = True
        return contacts[0], contacts[1]

    def privileged_left_touch_values(self, index: int | None = None) -> tuple[float, float]:
        """What each left jaw is doing to the object, in newtons.

        Called **with** an object index, this is the normal force summed over every contact between
        that jaw's chain and that object -- which is what "did we squeeze it" means, and what a
        per-body touch sensor cannot answer here.  A `touch` sensor reports the force on the body
        its site belongs to, and the jaw's contact moves between two bodies depending on the pose;
        measured, both channels read exactly 0.0000 through a grasp that had nine contacts.

        Called **without** one it is the raw fingertip sensors, unchanged, which is what
        `evaluation` and the older callers have always read -- and which is *not* what the
        observation vector uses either; that reads the sensors directly and is unaffected by any of
        this.
        """

        if index is None:
            return (
                float(self._sensor("L_finger_inner_touch_sensor")[0]),
                float(self._sensor("L_finger_outer_touch_sensor")[0]),
            )
        cookie_geoms = self._cookie_collision_geoms[index]
        return (
            self._jaw_contact_force(self._left_jaw_geoms[0], cookie_geoms),
            self._jaw_contact_force(self._left_jaw_geoms[1], cookie_geoms),
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
        spare_nominal = self.config.cookie_transfer.spare_target_bin_world_position_m
        has_spare = self._spare_bin_body >= 0 and spare_nominal is not None
        for attempt in range(1, scene.max_clearance_attempts + 1):
            source_offset = np.array(
                [scene.source_bin_x_m.draw(rng), scene.source_bin_y_m.draw(rng)]
            )
            source_yaw = scene.source_bin_yaw_rad.draw(rng)
            target = np.asarray(
                self.config.cookie_transfer.target_bin_world_position_m[:2],
                dtype=np.float64,
            ) + np.array([scene.target_bin_x_m.draw(rng), scene.target_bin_y_m.draw(rng)])
            target_yaw = scene.target_bin_yaw_rad.draw(rng)
            spare = None
            spare_yaw = 0.0
            if has_spare:
                spare = np.asarray(spare_nominal[:2], dtype=np.float64) + [
                    scene.spare_bin_x_m.draw(rng),
                    scene.spare_bin_y_m.draw(rng),
                ]
                spare_yaw = scene.spare_bin_yaw_rad.draw(rng)
            if self._boxes_are_clear(
                source_offset, source_yaw, target, target_yaw, spare, spare_yaw
            ):
                break
        else:
            raise RuntimeError(
                f"no layout with {scene.min_box_clearance_m * 1000:.1f} mm of box "
                f"clearance in {scene.max_clearance_attempts} draws; the "
                f"randomization ranges ask for a layout that cannot exist. Narrow "
                f"the box ranges or lower min_box_clearance_m."
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

        # 2. Working box, and 3. the spare when the scene has one.
        self._place_free_box(self._target_bin_body, target, target_yaw)
        if spare is not None:
            self._place_free_box(self._spare_bin_body, spare, spare_yaw)
        # The drawn poses, kept so `_assert_box_poses` can replay them.  The settle that
        # runs between this and the placement can explode, and MuJoCo's own safety net
        # then resets the data from inside `mj_step` -- wiping these very writes back to
        # the XML defaults.  The draws are Python state and survive; the mjData writes
        # do not, so what survives is what the replay re-writes.
        self._scene_draw = (
            source_offset,
            float(source_yaw),
            target,
            float(target_yaw),
            spare,
            float(spare_yaw),
        )

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
        target: np.ndarray,
        target_yaw: float,
        spare: np.ndarray | None,
        spare_yaw: float,
    ) -> bool:
        """Whether the drawn layout leaves every pair of boxes far enough apart.

        A yawed box needs a bigger berth than its half size suggests, so each pair
        is checked against the axis-aligned extent of the *rotated* rectangle.  Two
        boxes are apart if they are apart on either axis, which is the largest
        per-axis gap rather than the smallest -- boxes side by side in y are not
        overlapping merely because they share an x range.
        """

        clearance = self.config.randomization.min_box_clearance_m
        source_half = self._source_bin_half_xy
        target_half = self._target_bin_half_xy
        boxes = [
            (self.SOURCE_NOMINAL_CENTER + source_offset, source_half, source_yaw),
            (target, target_half, target_yaw),
        ]
        if spare is not None:
            boxes.append((spare, target_half, spare_yaw))
        for index, (centre, half, yaw) in enumerate(boxes):
            for other_centre, other_half, other_yaw in boxes[index + 1 :]:
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

    def _assert_box_poses(self) -> None:
        """Re-write the drawn box poses, because the settle before them can wipe them.

        **A reset has to end with the boxes where the draw put them, and the settle in
        between can silently take them away.**  Measured on
        `a3_libero_dense_batch_libero_mount` (seed 4, scene randomization on), one
        `mj_step` at a time through the settle that runs before the Cookies are placed:

            settle step 25   worst |qvel| 1.4e5 m/s, MuJoCo warns `QACC at DOF 165`
            settle step 26   `mocap_pos` back at the XML default, `time` 0, `qvel` 0

        The Cookies are still at their XML spawn poses while that settle runs -- they are
        placed after it -- and on a draw large enough the moved bin overlaps them and the
        settle explodes; MuJoCo's own safety net (`mj_checkPos` / `mj_checkVel`, *inside*
        `mj_step`, from C -- a Python-level tap on `mujoco.mj_resetData` never fires)
        then resets the data, which puts the source mocap and the free boxes back at the
        XML defaults.  The placement that follows writes the Cookies at the *drawn* offset
        into the *nominal* bin -- 4-6 mm inside a wall (contact table:
        `source_bin_front_wall <-> cookie_60..67_geom`, dist -6.0..-3.7 mm, 33-37 N) --
        and the first drop step fires them out at 10-23 m/s.  1 to 20 of 80 Cookies ended
        outside the source bin, on exactly the seeds whose draws were large enough to
        explode the settle: 3 of the first 10.

        The arm is already re-asserted at the end of the reset (`_assert_arm_at_home`, for
        the same class of wipe -- the arm's pose was the first thing this safety net took,
        and the fix for it is the pattern this follows).  This is the same bargain for the
        boxes: the draw is replayed rather than trusted to survive the settle.
        """

        draw = getattr(self, "_scene_draw", None)
        if draw is None:
            # No draw this episode: the boxes are at their XML poses, and the safety net
            # would restore exactly those.  Nothing to replay.
            return
        source_offset, source_yaw, target, target_yaw, spare, spare_yaw = draw
        mocap_id = int(self.model.body_mocapid[self._source_bin_body])
        nominal = self.SOURCE_NOMINAL_CENTER
        self.data.mocap_pos[mocap_id] = [
            nominal[0] + source_offset[0],
            nominal[1] + source_offset[1],
            0.0,
        ]
        self.data.mocap_quat[mocap_id] = _yaw_quaternion(source_yaw)
        self._place_free_box(self._target_bin_body, target, target_yaw)
        if spare is not None:
            self._place_free_box(self._spare_bin_body, spare, spare_yaw)

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
        # The second-batch condition for *this* episode, drawn before anything else so
        # the expert's `reset` -- which the runner calls right after this one -- reads
        # the condition the episode actually runs under.  Uniform over the configured
        # list by `seed % len(modes)`; the seed travels in the episode metadata, so the
        # condition is derivable from the dataset after the fact.  No seed (a probe
        # resetting without one) keeps the baseline.
        self._episode_seed = seed
        modes = self.config.cookie_transfer.second_batch_modes
        self.second_batch_mode = modes[int(seed) % len(modes)] if seed is not None else modes[0]
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
        home = np.asarray(
            self._randomized_deployment_home if randomize_scene else self.DEPLOYMENT_HOME,
            dtype=np.float64,
        ).copy()
        # The default, and the value the ablation switch falls back to: LIBERO's own half-open
        # start, which is what the state's last two numbers report.
        jaw_opening = LIBERO_PANDA_FINGER_OPEN_M
        if (self._libero_init_states and seed is not None
                and os.environ.get("A3_LIBERO_INIT", "1").strip() != "0"):
            # LIBERO's robot, LIBERO's start pose: one per episode, out of the file its own
            # evaluation reads.  Nothing here is invented, and nothing is jittered on top --
            # a randomised start would only move this arm away from where the policy expects
            # it, and the pose it expects already varies.
            arm7, jaw_opening = libero_init_arm(self._libero_init_states, int(seed))
            home[:7] = arm7
            # Both the pose **and the command that holds it**.  `settle` closes the loop on
            # the command, so setting only the state would be undone before the first
            # observation: the jaws would be driven back to whichever opening slot 7 held,
            # which is what the earlier version of this fix did -- measured, they ended at
            # full open against LIBERO's half.
            home[7] = jaw_opening / self._finger_travel["L"]
        arm_values = np.r_[home[:7], home[8:15]]
        for name, value in zip(ARM_JOINTS, arm_values, strict=True):
            self.data.qpos[self._qpos_ids[name]] = value
            self.data.qvel[self._dof_ids[name]] = 0.0
        # LIBERO's own episodes start with the jaws **half open**: its first
        # `robot0_gripper_qpos` is [0.020833, -0.020833].  This reset used to zero them, which
        # is closed for robosuite's gripper and open for the A3's -- the two grippers disagree
        # about which end of the travel zero is.  A policy told the wrong opening plans a
        # different first move; with the Panda that was a hand driven down onto the bowl, jaws
        # never closing on it.
        for side in ("L", "R"):
            for sign, joint_id in zip(
                self._finger_sign[side], self._finger_joints[side], strict=True
            ):
                travelled = self._finger_open_at_zero[side]
                value = 0.0 if travelled else sign * jaw_opening
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
                self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
        self._last_applied_action = home.copy()
        self._apply_controls(home)
        self.settle()
        # **The drawn box poses have to survive that settle, and on an exploding draw
        # they do not.**  The Cookies are still at their XML spawn poses while it runs
        # -- they are placed after it -- and on a draw large enough the moved bin
        # overlaps them and the settle explodes; MuJoCo's own safety net then resets
        # the data from inside `mj_step`, which puts the boxes back at the XML
        # defaults.  Replayed here, before the placement reads them, or the Cookies
        # are placed at the drawn offset into the *nominal* bin.  See
        # `_assert_box_poses` for the measured chain.
        self._assert_box_poses()
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
            # ...unless the scene declares a starting attitude, which is the **base** the yaw
            # turns on top of.  Without this the reset overwrote whatever the model was built
            # with -- and it overwrote it with a pure yaw, so a scene asking for the slab
            # rolled onto its broad face got it standing anyway, silently, with the same
            # pixels in the camera as before.  Empty is the identity and keeps every existing
            # scene byte-for-byte: a Cookie stands on its 6.4 mm edge, which is what A3's own
            # task means by upright.
            base_quat = self.config.cookie_transfer.cookie_source_quat_wxyz
            if base_quat:
                composed = np.zeros(4, dtype=np.float64)
                mujoco.mju_mulQuat(composed, np.asarray(quaternion), np.asarray(base_quat))
                quaternion = tuple(float(value) for value in composed)
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
        # **A reset has to end with the arm where the state convention says it is.**  Every
        # LIBERO manifest's `state_offset` and `state_rotation_reference` are calibrated *at the
        # arm's home* -- measured, A3's `(0.100, 0.250, 0.920)` plus `-0.3114,-0.2610,0.2541`
        # gives LIBERO's own `(-0.2112, -0.0111, 1.1742)` exactly -- so a reset that ends
        # anywhere else hands the policy a self-pose that is simply false, and the expert's IK,
        # which is seeded from the pose, a start it cannot reason from.
        #
        # Measured on the dense props scenes (`a3_libero_dense_batch`,
        # `a3_libero_panda_dense_batch_a3_placement`) the left arm ended `reset` at
        # `qpos = ctrl = 0` -- its *zero* pose, arm extended along +y with the tool frame at the
        # identity -- while the same code in `a3_cookie_batch` (`joint_position`) kept its home.
        # The eight-wide state the policy was then handed was **841.7 mm and 200.2 degrees** away
        # from LIBERO's own reset state, and the batch expert's reset IK -- seeded from the pose,
        # as a local solver is -- failed with `the closest pose is 7.73 mm from the target`,
        # which is what a bogus start pose looks like from outside.
        #
        # The *cause* is the `settle()` above, which runs before the Cookies are placed and so
        # settles a scene whose eighty Cookies are still at their spawn poses.  MuJoCo reports
        # `Nan, Inf or huge value in QACC at DOF 480. Time = 0.0780`, and 0.0780 s is exactly the
        # step at which the arm's pose is lost (traced: `mj_step` #139 of 250, with `time`
        # resetting to one step).  Settling the same scene *after* the placement, measured,
        # explodes nothing and keeps the arm at its home for all 100 steps -- so the pose is
        # re-asserted here rather than the settle being trusted.
        self._assert_arm_at_home(home)

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

    def _assert_arm_at_home(self, home: np.ndarray) -> None:
        """Write the left arm's home pose and the command that holds it, and check it stuck.

        Called at the end of every reset, for the measurement in the call site's comment.  Two
        array writes and one `mj_forward`, so it costs nothing, and it makes a scene whose
        settling destroys the arm's pose fail loudly instead of quietly handing a policy a
        self-pose that is 841 mm from where it really is.
        """

        for name, value in zip(ARM_JOINTS, np.r_[home[:7], home[8:15]], strict=True):
            self.data.qpos[self._qpos_ids[name]] = value
            self.data.qvel[self._dof_ids[name]] = 0.0
        self._apply_controls(home)
        self._last_applied_action = np.asarray(home, dtype=np.float64).copy()
        mujoco.mj_forward(self.model, self.data)
        actual = np.asarray([self.data.qpos[self._qpos_ids[name]] for name in LEFT_JOINTS])
        error = float(np.max(np.abs(actual - np.asarray(home[:7], dtype=np.float64))))
        if error > 1e-6:
            raise RuntimeError(
                f"reset left the left arm {error:.6f} rad from its home pose; every LIBERO "
                "manifest's state_offset is calibrated in that pose, so this cannot be reported "
                "as an observation"
            )

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
        # A teleport is not motion, and the window `_cookie_average_speed` measures must not see
        # one: a Cookie moved here by a test or by a reset would otherwise look like a Cookie that
        # had just travelled the whole distance in a single control step.
        self._cookie_pose_history.pop(index, None)
        mujoco.mj_forward(self.model, self.data)

    def place_cookie_in_slot(
        self,
        index: int,
        slot_index: int,
        *,
        lean_azimuth_rad: float,
        lean_rad: float,
        yaw_rad: float,
        max_jitter_m: float,
    ) -> None:
        """Teleport Cookie ``index`` standing into target slot ``slot_index``, imperfectly.

        The imperfections are the point.  A prefilled second-batch episode has to start
        from progress that looks like what a *policy's* first batch leaves behind, not
        like what the expert's does -- and the expert's is perfect, five upright at the
        slot centres, which is exactly the state a policy never produces.

        **The lean and the yaw are exact values, not maxima, and the caller draws them
        once per episode and passes the same ones to every Cookie of the group.**  That
        coherence is a correctness requirement, not a style: the five slots of a column
        are 6.33 mm apart in ``y`` and the Cookie is 6.33 mm thick, so the standing group
        has **zero** clearance cookie-to-cookie -- and a Cookie leaned *towards* its
        neighbour by an independently drawn angle puts its top edge 25*sin(lean) ~ 1.7 mm
        *inside* the neighbour's body, which `set_cookie_pose` writes without a collision
        check and the contact solver then resolves as an explosion.  Measured on the
        first version, which drew each Cookie's lean independently: the prefilled
        group's two *end* Cookies (indices 0 and 4, the ones with a single neighbour to
        be shoved into) ended the episode outside the box entirely.  A coherent lean --
        every Cookie of the group tipped about the same axis by the same angle -- is a
        parallelogram: no pair interpenetrates, the group leans the way a group that
        settled together actually leans, and the end Cookie's top overhangs the box
        floor, which has ~19 mm of margin to absorb it.

        The jitter is drawn here, per Cookie, because it is the one imperfection that is
        safe to draw independently: sub-millimetre overlaps resolve gently, and a
        per-Cookie jitter is the variety a uniform placement would otherwise lack.

        ``z`` is not an argument: the slot's world point is computed at the box's own
        cookie-centre height, and `set_cookie_pose`'s box branch re-snaps it there in
        the live box frame anyway -- which is also why this stays correct when the box
        has been randomised, because the slot is read through `privileged_target_slot_world`
        and the snap through the live body pose, not through the configured nominal.
        """

        world = self.privileged_target_slot_world(slot_index, self._target_cookie_center_z)
        jitter = self.np_random.uniform(-max_jitter_m, max_jitter_m, size=2)
        position = (world[0] + jitter[0], world[1] + jitter[1], world[2])
        axis = (np.cos(lean_azimuth_rad), np.sin(lean_azimuth_rad), 0.0)
        lean_q = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(lean_q, np.asarray(axis), lean_rad)
        yaw_q = np.array(
            [np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=np.float64
        )
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(quaternion, yaw_q, lean_q)
        self.set_cookie_pose(index, position, tuple(float(v) for v in quaternion))

    def perturb_settled_cookies(
        self,
        indices: tuple[int, ...],
        *,
        lean_azimuth_rad: float,
        lean_rad: float,
        yaw_rad: float,
        max_jitter_m: float,
    ) -> None:
        """Nudge already-placed Cookies: a coherent lean and yaw, a per-Cookie shift.

        For the inter-batch perturbation, applied to the five a batch just released so
        the next batch is inserted next to imperfect neighbours.  The lean and the yaw
        are exact values the caller drew once, and the coherence is the same correctness
        requirement as `place_cookie_in_slot`'s: the settled group has zero clearance
        cookie-to-cookie, so independently drawn leans interpenetrate the neighbours and
        the solver explodes the group apart (measured: the perturbed group's end Cookie
        was still re-settling when the expert's final check ran, and the episode was
        discarded for a state it had left two hundred steps later).  A coherent lean
        tips the group as a parallelogram -- no interpenetration, and the re-settle is
        the gentle relaxation of a group that leans together.

        Reads each Cookie's *live* pose and composes the perturbation onto it, so it is
        a nudge rather than a re-placement; `set_cookie_pose` zeroes the velocity and
        re-forwards, so the result is a Cookie at rest, leaning, that the contact
        solver then re-settles -- which is the realistic shape of "the previous batch
        did not land perfectly".
        """

        lean_q = np.empty(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(
            lean_q,
            np.array([np.cos(lean_azimuth_rad), np.sin(lean_azimuth_rad), 0.0]),
            lean_rad,
        )
        yaw_q = np.array(
            [np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=np.float64
        )
        perturb_q = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(perturb_q, yaw_q, lean_q)
        for index in indices:
            joint = int(self._cookie_joints[index])
            address = int(self.model.jnt_qposadr[joint])
            position = self.data.qpos[address : address + 3].copy()
            quaternion = self.data.qpos[address + 3 : address + 7].copy()
            jitter = self.np_random.uniform(-max_jitter_m, max_jitter_m, size=2)
            position[0] += jitter[0]
            position[1] += jitter[1]
            composed = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(composed, perturb_q, quaternion)
            self.set_cookie_pose(
                index, tuple(float(v) for v in position), tuple(float(v) for v in composed)
            )

    #: How many control steps of history the "is it still moving?" question is asked over.
    #:
    #: `qvel` cannot answer that question.  It is the velocity *inside* a physics substep, and the
    #: substeps within one control step oscillate: a Cookie lying still on the target box floor on
    #: the Panda is reported at 0.65 rad/s and 7.4 mm/s, while its pose moves 0.0145 mm and 0.027
    #: degrees per control step -- 0.0275 rad/s and 0.29 mm/s, smaller by a factor of 25, which is
    #: exactly the number of substeps in a control step (0.05 s / 0.002 s).  A predicate that reads
    #: the substep therefore never sees a placed Cookie settle, and no placement can be accepted.
    #: The window below measures the same thing `qvel` is meant to mean -- how fast the Cookie is
    #: moving -- but over the control step rather than inside it.
    SETTLE_WINDOW_SAMPLES = 10

    def _cookie_average_speed(self, index: int) -> tuple[float, float]:
        """The Cookie's average linear and angular speed over the last ``SETTLE_WINDOW_SAMPLES``.

        Measured from the Cookie's **pose**, as the net change across the window divided by the
        elapsed simulation time: a Cookie that is jittering in place has a large `qvel` and a small
        net change, which is the distinction ``settled`` is actually asking about.

        The samples are keyed by ``data.time`` so that calling this more than once in a control step
        does not push the same instant twice, and a run whose clock has gone backwards -- a fresh
        episode -- starts a new window rather than mixing two of them.  Until a window exists the
        instantaneous reading is returned, so the first few steps behave as they used to.
        """

        history = self._cookie_pose_history.setdefault(index, [])
        now = float(self.data.time)
        if history and history[-1][0] > now:
            history.clear()
        if not history or history[-1][0] < now:
            history.append(
                (
                    now,
                    self.data.xpos[self._cookie_bodies[index]].copy(),
                    self._cookie_rotation(index).copy(),
                )
            )
            del history[: -self.SETTLE_WINDOW_SAMPLES]
        if len(history) < 2:
            return self._cookie_speed(index)
        first_time, first_position, first_rotation = history[0]
        last_time, last_position, last_rotation = history[-1]
        elapsed = last_time - first_time
        if elapsed <= 0.0:
            return self._cookie_speed(index)
        linear = float(np.linalg.norm(last_position - first_position)) / elapsed
        relative = last_rotation @ first_rotation.T
        angular = float(
            np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        ) / elapsed
        return linear, angular

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
        linear_speed, angular_speed = self._cookie_average_speed(index)
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
        return self._containment_clauses(index)["in_target"]

    def privileged_target_containment(self, index: int) -> dict[str, Any]:
        """Every clause of the verdict, so a caller can report *why* rather than just *no*.

        `privileged_cookie_in_target` answers one bit, and one bit cannot separate a Cookie
        that never reached the box from one that is inside it a layer too high.  A recorded
        roll-out was misread exactly that way -- as a placement rejected for resting on
        another Cookie -- and the reading could not be checked against anything, because the
        verdict was all there was.  This is the same computation, exposed.

        Read-only: `_cookie_average_speed` keys its samples on `data.time`, so calling this
        as often as a renderer likes does not change what `settled` sees.  Verified rather
        than assumed, because a diagnostic that perturbs the thing it reports is worse than
        no diagnostic.
        """

        return dict(self._containment_clauses(index))

    def _containment_clauses(self, index: int) -> dict[str, Any]:
        """The clauses of `privileged_target_containment`, computed once for both callers."""

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
        # The Cookie's lowest face, in the box's own frame: the one number that says where in
        # the box it is, and the one a reader wants when a placement is refused.
        bottom = float(p_rel[2] - half_extent[2])
        #: Inside the box's interior volume: above the floor and below the rim.
        below_rim = bool(bottom < self.TARGET_WALL_HEIGHT)
        above_floor = bool(bottom >= floor_top - 0.003)
        #: Resting on the *floor* of the box, which is a stronger claim than being inside it,
        #: and the one that catches a Cookie perched on the rim.  See
        #: `tests/test_cookie_stacked_containment.py` for why it is not merely stacking.
        on_the_floor = bool(bottom <= floor_top + 0.004)
        vertically_inside = bool(below_rim and above_floor)
        if self.task_config.require_released:
            vertically_inside = vertically_inside and on_the_floor
        # No early return: every clause is computed even when an earlier one has already
        # decided the verdict, because the point of this function is to be read.
        upright = bool(
            abs(float(r_rel[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
            or abs(float(c_mat[2, 2])) >= np.cos(self.task_config.max_tilt_rad)
        )
        linear_speed, angular_speed = self._cookie_average_speed(index)
        settled = (
            linear_speed <= max(self.task_config.max_linear_speed_m_s, 0.09)
            and angular_speed <= self.task_config.max_angular_speed_rad_s
        )
        released = (
            not self.task_config.require_released
            or not any(self.privileged_left_finger_contacts(index))
        )
        return {
            "in_target": bool(
                footprint_inside
                and vertically_inside
                and (upright or not self.task_config.require_upright)
                and settled
                and released
            ),
            # The named clauses, in the order a failure is worth reading them.
            "footprint": footprint_inside,
            "above_floor": above_floor,
            "below_rim": below_rim,
            "on_the_floor": on_the_floor,
            "upright": upright,
            "settled": settled,
            "released": released,
            # Measurements, for a report that has to say how far off something was.
            "bottom_above_floor_m": bottom - floor_top,
            "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(float(r_rel[2, 2])), -1.0, 1.0)))),
            "box_xy_mm": (float(p_rel[0]) * 1000.0, float(p_rel[1]) * 1000.0),
        }

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
