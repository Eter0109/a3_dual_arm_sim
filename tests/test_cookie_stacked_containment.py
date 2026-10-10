"""What "contained" means for a 6.4 mm Cookie in a 20 mm deep box.

Both tests here exist because a reading of a rollout video said the opposite of what the
measurement says.  The claim was that a second Cookie landed on top of the first inside the
box and was rejected only by the clause requiring it to be *down on the box floor*, so that
clause should go.  Placing the Cookies by hand -- no policy, no sampling noise -- shows why
that reading cannot be right, and what the clause is actually holding up.

The box
-------
Measured on `a3_libero_panda_two_cookies`:

* floor top       6.0 mm above the box body's origin
* wall top       26.0 mm -- so **20 mm** of interior height
* inner footprint 57 x 29 mm (half sizes), plus 4 mm of containment tolerance
* a Cookie is a 50 x 6.4 x 25 mm slab, and `_target_cookie_center_z` is 18.5 mm = 6 + 25/2,
  so the placement the task is built around is the Cookie **standing on its 6.4 mm edge**,
  poking 5 mm above the rim.  `upright` is the clause that says so: it wants the Cookie's
  25 mm axis vertical.

Two standing Cookies are therefore 50 mm of Cookie in a 20 mm hole, and the upper one's
lowest face sits at 6 + 25 = 31 mm -- above the wall top.  It is not "in the box" by any
reading that keeps the word meaningful, and the clause that says so is the rim bound, not
the floor bound.  If the lower Cookie is lying flat instead, its own `upright` fails, so
there is no configuration in which a stack is contained and both Cookies are placed
acceptably.

What the floor clause is for
----------------------------
It is not decoration and it is not only about stacking.  A Cookie that comes to rest on the
**rim** has its lowest face at the wall top and can still be inside the footprint tolerance,
because the rim is where the footprint test stops: `below_rim` admits it by a tenth of a
millimetre, and the footprint test then decides on a millimetre of margin.  `on_the_floor`
is the clause that states the intent -- down on the floor of the box -- and the second test
here pins it, so that removing it fails loudly instead of quietly counting a Cookie balanced
on the box's own wall.

That case occurred rather than being hypothetical: in a four-seed probe of the 15k
checkpoint one roll-out left a Cookie standing on the rim at 25.87 mm, with only the
footprint test (by 3.5 mm) between it and a verdict of "contained".
"""

from __future__ import annotations

import numpy as np

from a3_dual_arm_sim.collection import scene_by_name

TWO_COOKIE_SCENE = "a3_libero_panda_two_cookies"


def _geometry(env, index: int) -> tuple[float, float]:
    """The Cookie's lowest face and its vertical half-extent, in the box frame, in metres."""

    tb_pos = env.data.xpos[env._target_bin_body]
    tb_mat = env.data.xmat[env._target_bin_body].reshape(3, 3)
    p_rel = tb_mat.T @ (env.data.xpos[env._cookie_bodies[index]] - tb_pos)
    half_extent = np.abs(tb_mat.T @ env._cookie_rotation(index)) @ env.COOKIE_HALF_SIZE
    return float(p_rel[2] - half_extent[2]), float(half_extent[2])


def _put_standing(env, index: int, xy: tuple[float, float], bottom: float) -> None:
    """Stand a Cookie at ``xy`` with its lowest face at ``bottom``, in the box frame.

    ``set_cookie_pose`` cannot express this: inside the box footprint it rewrites the height
    to `_target_cookie_center_z`, the one-layer placement, so a test that wants a Cookie at
    another height has to write the free joint itself.  The orientation is the box's own,
    which is the standing pose `upright` asks for.
    """

    import mujoco

    joint_id = env._cookie_joints[index]
    address = int(env.model.jnt_qposadr[joint_id])
    tb_pos = env.data.xpos[env._target_bin_body]
    tb_mat = env.data.xmat[env._target_bin_body].reshape(3, 3)
    half_height = float(env.COOKIE_HALF_SIZE[2])
    env.data.qpos[address : address + 3] = tb_pos + tb_mat @ np.array(
        [xy[0], xy[1], bottom + half_height], float
    )
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, tb_mat.reshape(-1))
    env.data.qpos[address + 3 : address + 7] = quaternion
    mujoco.mj_forward(env.model, env.data)


def _hold(env, steps: int) -> None:
    """Step while commanding no motion, at the scene's own action width."""

    from a3_dual_arm_sim.contracts import CARTESIAN_ACTION_DIM, JOINT_ACTION_DIM

    width = JOINT_ACTION_DIM if env.action_mode == "joint_position" else CARTESIAN_ACTION_DIM
    action = np.zeros(width)
    for _ in range(steps):
        env.step(action)


def test_two_standing_cookies_cannot_both_be_inside_the_box() -> None:
    """A stack of two is 50 mm of Cookie in a 20 mm box; the upper one is above the rim.

    The first Cookie is the placement the task asks for -- standing on the floor -- and it is
    contained.  The second is placed directly on top of it, and its lowest face lands above
    the wall top, which is what rejects it.
    """

    scene = scene_by_name(TWO_COOKIE_SCENE)
    env = scene.build_env(0.0, 0.0)
    try:
        env.reset(seed=0, options={})
        floor_top = float(
            env.config.cookie_transfer.target_floor_z_m
            + env.config.cookie_transfer.bin_wall_thickness_m / 2
        )
        thickness = 2 * float(env.COOKIE_HALF_SIZE[2])
        assert thickness > env.TARGET_WALL_HEIGHT - floor_top, (
            "this test is about a box shorter than one standing Cookie; the box has grown"
        )
        slot = np.asarray(env.TARGET_SLOTS_LOCAL[0], dtype=np.float64)
        xy = (float(slot[0]), float(slot[1]))

        _put_standing(env, 0, xy, floor_top)
        # `settled` needs a pose history, which only stepping produces.  A zero command is
        # "hold the pose you are in" at the scene's own action width.
        _hold(env, 40)
        assert env.privileged_cookie_in_target(0), "the task's own placement must count"

        # Directly on top, 1 mm clear of the Cookie below so `mj_forward` has no contact to
        # resolve and cannot move either of them.
        _put_standing(env, 1, xy, floor_top + thickness + 0.0011)

        bottom, half_extent = _geometry(env, 1)
        assert bottom >= env.TARGET_WALL_HEIGHT, (
            f"the upper Cookie's lowest face is {1000 * bottom:.2f} mm and the wall top is "
            f"{1000 * env.TARGET_WALL_HEIGHT:.2f} mm"
        )
        assert half_extent > 0.010, "and it is standing, not lying flat"
        assert not env.privileged_cookie_in_target(1), (
            "a Cookie whose lowest face is above the wall top is not inside the box"
        )
    finally:
        env.close()


def test_a_cookie_balanced_on_the_rim_is_not_contained() -> None:
    """`on_the_floor` is the clause that catches a Cookie resting on the box's own wall.

    Placed where the rim is, which the footprint tolerance reaches, with its lowest face a
    tenth of a millimetre below the wall top so the rim bound cannot be the reason.  If the
    floor clause is ever removed, this test turns green for the wrong answer and says so.
    """

    scene = scene_by_name(TWO_COOKIE_SCENE)
    env = scene.build_env(0.0, 0.0)
    try:
        env.reset(seed=0, options={})
        slot = np.asarray(env.TARGET_SLOTS_LOCAL[0], dtype=np.float64)
        half_extent_y = float(env.COOKIE_HALF_SIZE[1])
        y = float(env.TARGET_INNER_HALF_SIZE[1] + env.CONTACT_CONTAINMENT_TOLERANCE_M) - half_extent_y
        _put_standing(env, 0, (float(slot[0]), y), float(env.TARGET_WALL_HEIGHT) - 0.0001)

        bottom, _ = _geometry(env, 0)
        assert bottom < env.TARGET_WALL_HEIGHT, "the rim bound must not be what rejects it"
        assert not env._cookie_inside_target(0), (
            "a Cookie resting on the box's rim is not on the floor of the box"
        )
    finally:
        env.close()
