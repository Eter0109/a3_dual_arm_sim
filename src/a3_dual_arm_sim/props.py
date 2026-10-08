"""Measuring LIBERO's props, which are not where they say they are.

The LIBERO objects dropped into ``a3_libero_props`` are wrapper bodies with a mesh body
nested 100 mm inside them (``pos="0 -0.1 0"``), a visual mesh, and forty-odd convex pieces
from a collision decomposition of the same surface.  Anything that reasons about "how close
the gripper got to the bowl" or "is the bowl over the plate" has to measure the geometry
rather than trust a number, and two of the available numbers are traps:

* ``xpos`` of the mesh body is the *body's* origin, not the bowl's centre -- the centre sits
  26.3 mm above it, so a distance to ``xpos`` is a distance to a point down near the bowl's
  underside.
* for a **mesh** geom, ``geom_aabb`` and ``geom_size`` hold the *unscaled* mesh extents.
  The XML's ``scale="0.7 0.7 0.7"`` is already baked into the compiled vertex array, so
  ``mesh_vert`` is the only honest source; scaling the extents again puts the bowl's rim at
  39 mm when it is at 56.
* LIBERO's objects all declare the same three sites -- ``bottom_site`` at z=-0.06,
  ``top_site`` at +0.04, ``horizontal_radius_site`` at (0.025, 0.025).  Those are a template
  copied into every object, not a measurement of it: they describe a bowl 100 mm tall with a
  35 mm radius, while the mesh shipping beside them is 53 mm tall and 56 mm in radius.

Both the trace probe and the video recorder need this, which is why it lives here rather
than in either of them.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: LIBERO's bowl, as the scene calls it: the mesh body, not the wrapper the prop was placed
#: at.  See the module docstring for why the wrapper is the wrong thing to measure from.
BOWL_MESH_BODY = "libero_bowl_object"

#: LIBERO's plate, the target of every ``libero_spatial`` task.
PLATE_MESH_BODY = "libero_plate_object"

#: The two sites that bracket the left gripper's opening.
FINGERTIPS = ("L_finger_inner_touch", "L_finger_outer_touch")

#: The site the IK solves for, so "where the arm went" is the point the policy controls.
EEF_SITE = "L_eef"

#: The body the A3's own arms hang off, used to find every arm body without naming them.
ARM_BASE_BODY = "base_link"

#: Every root a *robot* hangs off in these scenes.  There is more than one because
#: `a3_libero_panda` replaces the left arm with LIBERO's own Panda, which
#: `model._add_libero_panda` splices into the XML as a tree of its own rooted at
#: `robot0_base` -- a sibling of the table, **not** a descendant of `base_link`.  A walk
#: down from `base_link` alone therefore finds the A3's right arm and none of the arm that
#: does the work.
#:
#: That is not a cosmetic difference.  The contact check asks "is the object touching an
#: arm body", so a missing subtree turns every grasp into "the arm never touched it".
#: Measured on `a3_libero_panda`, before this was fixed: a rollout that displaced the bowl
#: 136.6 mm and lifted it 90.4 mm -- a grasp -- was reported as 0 steps of contact, and the
#: failure signature printed was the one for a rollout that never reached the object.
REFERENCE_ROBOT_BASE_BODIES = (ARM_BASE_BODY, "robot0_base")

#: The eight corners of a box, for turning a geom's axis-aligned extent into world points.
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _under(model, root_id: int, body: int) -> bool:
    """Is ``body`` the body ``root_id`` or one of its descendants?  Bodies form a tree."""

    node = body
    while True:
        if node == root_id:
            return True
        if node == 0:
            return root_id == 0
        node = int(model.body_parentid[node])


def prop_geoms(model, body_id: int) -> list[int]:
    """The geoms carried by a prop's body and everything under it."""

    return [index for index in range(model.ngeom)
            if _under(model, body_id, int(model.geom_bodyid[index]))]


def prop_points(model, data, body_id: int) -> np.ndarray:
    """Every corner of a prop's geometry, in world coordinates.

    Both the visual mesh and the collision pieces are walked, so the result describes the
    whole object rather than half of it.
    """

    points = []
    for index in prop_geoms(model, body_id):
        if model.geom_type[index] == int(mujoco.mjtGeom.mjGEOM_MESH):
            start = int(model.mesh_vertadr[model.geom_dataid[index]])
            count = int(model.mesh_vertnum[model.geom_dataid[index]])
            local = np.asarray(model.mesh_vert[start:start + count])
        else:
            local = (np.asarray(model.geom_aabb[index][:3])
                     + _CORNERS * np.asarray(model.geom_aabb[index][3:]))
        points.append(data.geom_xpos[index]
                      + local @ data.geom_xmat[index].reshape(3, 3).T)
    return np.vstack(points)


def prop_frame(model, data, body_id: int) -> tuple[np.ndarray, float, np.ndarray]:
    """A prop's centre, horizontal radius and overall size, measured off its geometry.

    Measured rather than written down, so that "the bowl is over the plate" is answered
    against this scene's plate instead of a number remembered from somewhere else.
    """

    points = prop_points(model, data, body_id)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(points[:, :2] - centre[:2], axis=1).max())
    return centre, radius, (high - low)


def robot_bodies(model, bases: tuple[str, ...] = REFERENCE_ROBOT_BASE_BODIES) -> set[int]:
    """Every body of every robot in this model, by walking down from each arm's base.

    Structure rather than names, so a rename cannot silently empty the set -- but the
    *roots* have to be named, because this model carries two robots and they are two
    separate trees.  See :data:`REFERENCE_ROBOT_BASE_BODIES` for the failure a single root
    caused.  Bases that are absent from this model are skipped rather than fatal, since no
    one scene has all of them; it is only an error if *none* of them is present.
    """

    roots = [
        root
        for root in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base) for base in bases
        )
        if root >= 0
    ]
    if not roots:
        raise ValueError(f"none of {bases!r} is a body in this model; the contact check "
                         f"needs the arms' bodies")
    return {
        body
        for body in range(model.nbody)
        if any(_under(model, root, body) for root in roots)
    }


def arm_qpos_addresses(model) -> np.ndarray:
    """Where the *working* arm's seven joints live in ``qpos``.

    Addressed by the names in ``contracts.LEFT_JOINTS`` rather than by counting, because
    ``model._add_libero_panda`` **renames** LIBERO's Panda joints to those names -- so the
    same seven strings work in every scene here, and a scene that wedged in a third arm
    could not silently shift the answer.  This exists because the obvious alternative,
    ``info["applied_action"][:7]``, is a joint target only for a joint-target arm: under an
    ``OperationalSpaceAdapter`` those entries are **torques**, which is how a probe came to
    report 17 rad of "motion" per step in a scene capped at 0.1 rad.
    """

    from a3_dual_arm_sim.contracts import LEFT_JOINTS

    addresses = []
    for name in LEFT_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"{name!r} is not a joint in this model; the measured "
                             f"joint motion needs the working arm's own joints")
        addresses.append(int(model.jnt_qposadr[joint]))
    return np.asarray(addresses)


def touching(model, data, prop_geoms: set[int] | list[int], bodies: set[int]) -> bool:
    """Is the prop in contact with any geom whose body is in ``bodies``, right now?

    Read from MuJoCo's own contact list rather than inferred from proximity.  A fingertip
    3 mm from a wall is a different thing from a fingertip pressing on it, and only one of
    those is evidence that a policy ever got hold of anything -- or ever let go.

    The first argument is named after :func:`prop_geoms` because that is what it is for;
    pass ``set(prop_geoms(model, bowl))`` once and reuse it, since this runs per step.
    """

    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        for near, far in ((first, second), (second, first)):
            if near in prop_geoms and int(model.geom_bodyid[far]) in bodies:
                return True
    return False
