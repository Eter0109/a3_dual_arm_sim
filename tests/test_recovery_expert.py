"""Recovery-expert behaviour that does not need a full episode.

The plough itself only shows up after a batch has damaged the source bin, which
takes minutes of simulation, so what is pinned here is the part that can be
checked cheaply: the geometry the stroke is aimed with, the fact that an
undamaged scene behaves exactly like ``A3CookieBatchExpert``, and that the
back-most leaning Cookie is the one picked as the entry point.
"""

from pathlib import Path

import mujoco
import pytest
import numpy as np

from a3_dual_arm_sim.batch_expert import A3CookieBatchExpert
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.recovery_expert import A3CookieRecoveryExpert, RecoveryStage

CONFIG = Path("configs/cookie_batch.yaml")


def _env():
    env = A3CookieTransferEnv(load_config(CONFIG), render_cameras=False)
    env.reset(seed=0, options={"randomize_cookies": False})
    return env


def _lean(env, index: int, degrees: float, axis: int = 0) -> None:
    """Tilt a Cookie in place, the way a batch leaves its neighbours behind."""
    joint = env.model.body_jntadr[env._cookie_bodies[index]]
    address = env.model.jnt_qposadr[joint]
    half = np.deg2rad(degrees) / 2.0
    quat = np.zeros(4)
    quat[0] = np.cos(half)
    quat[axis + 1] = np.sin(half)
    env.data.qpos[address + 3 : address + 7] = quat
    mujoco.mj_forward(env.model, env.data)


def test_blade_and_core_geometry_come_from_the_model_and_config():
    env = _env()
    try:
        expert = A3CookieRecoveryExpert(env)
        expert.reset()
        bevel = env.config.cookie_transfer.cookie_edge_bevel_m
        # Closed jaws: the two 6.35 mm pads sit back to back (measured 6.362 mm
        # between their centres at zero opening), so the tool centre-line is
        # 12.71 mm from face to face and 6.36 mm from the leading face.
        pad_half = float(env.model.geom_size[env._left_finger_geoms[0]][1])
        assert expert._blade_half_y == pytest.approx(2.0 * pad_half, abs=1e-4)
        # The push has to land on the core's top corner, which is where the
        # chamfer cap starts and the flat back face ends.
        assert expert._core_half_z == float(env.COOKIE_HALF_SIZE[2]) - bevel
    finally:
        env.close()


def test_undamaged_scene_actions_match_the_batch_expert():
    """With nothing leaning, the recovery hook must not perturb the baseline."""
    env = _env()
    try:
        batch = A3CookieBatchExpert(env)
        batch.reset()
        recovery = A3CookieRecoveryExpert(env)
        recovery.reset()
        assert recovery._recovery_target() is None
        for step in range(24):
            batch_action = batch.act()
            recovery_action = recovery.act()
            assert recovery._stage is RecoveryStage.IDLE, f"recovery started at step {step}"
            np.testing.assert_array_equal(batch_action, recovery_action)
            env.step(batch_action)
        assert batch.phase is recovery.phase
    finally:
        env.close()


def test_back_most_leaner_is_the_recovery_entry_point():
    env = _env()
    try:
        expert = A3CookieRecoveryExpert(env)
        expert.reset()
        assert expert.leaning_cookies() == []
        for index, degrees in ((7, 30.0), (8, 40.0), (9, 20.0)):
            _lean(env, index, degrees)
        assert expert.leaning_cookies() == [7, 8, 9]
        # The blade descends behind its target, so every other leaner has a
        # neighbour in the way; only the back-most one is reachable.
        assert expert._recovery_target() == 9
    finally:
        env.close()


def test_stroke_aims_the_blade_at_the_chamfer_corner():
    """The stroke meets the 45 deg chamfer, never the flat face 4 mm below it.

    A push on the chamfer answers with a forward and downward reaction and the
    Cookie pivots upright; the same push on the flat back face answers with a
    forward and upward one and the Cookie leaves the floor.
    """
    env = _env()
    try:
        expert = A3CookieRecoveryExpert(env)
        expert.reset()
        _lean(env, 9, 30.0)
        expert._begin_recovery(9)
        # Invert the site offset to ask where the pad centre actually goes.
        pad_mid = np.asarray(expert._down) + np.asarray(expert._canonical) @ np.asarray(
            expert._pad_offset
        )
        body = env._cookie_bodies[9]
        corner = env.data.xpos[body] + env.data.xmat[body].reshape(3, 3) @ np.array(
            [0.0, float(env.COOKIE_HALF_SIZE[1]), expert._core_half_z]
        )
        blade_bottom = pad_mid[2] - expert._blade_half_z
        blade_back_face = pad_mid[1] - expert._blade_half_y
        assert abs(blade_bottom - (corner[2] + expert.BLADE_DROP_M)) < 1e-9
        assert abs(blade_back_face - (corner[1] + expert.BLADE_BACK_OFFSET_M)) < 1e-9
        # Descending further than the drop would miss the chamfer entirely.
        assert expert.BLADE_DROP_M <= 0.0
        assert expert._overhead[2] > expert._down[2]
    finally:
        env.close()


def test_plough_requires_a_sustained_upright_read():
    """One upright sample is not enough: a Cookie falling past vertical reads as
    upright for a step, which is how the first prototype declared victory early."""
    assert A3CookieRecoveryExpert.SETTLE_STEPS >= 5


def test_plough_step_actually_commands_motion():
    """The creep must move the blade, not just count the travel.

    ``_servo`` ignores its ``position`` argument at ``speed=0`` and advances its
    own ``_servo_pos`` instead, so a creep that only updates a local goal leaves
    the blade hovering while ``plough_advanced_m`` runs to the travel limit --
    a stroke that looks like it ran and moved nothing.  Pin the command itself.
    """
    env = _env()
    try:
        expert = A3CookieRecoveryExpert(env)
        expert.reset()
        _lean(env, 9, 30.0)
        expert._begin_recovery(9)
        # Jump straight to the creep, as if the descent had landed.
        expert._advance_stage(RecoveryStage.PLOUGH)
        start_site = env.data.site_xpos[expert._l_site].copy()
        for _ in range(4):
            expert._plough_step()
        assert expert._plough_advanced > 0.0
        # The commanded goal has to have moved, and by the creep distance.
        commanded = expert._servo_pos[1] - start_site[1]
        assert commanded < 0.0, "the creep drives the blade toward -y"
        assert abs(commanded + expert._plough_advanced) < 1e-9, (
            "the commanded position must follow the travel counter, not lag it"
        )
    finally:
        env.close()


def test_recovered_but_packed_column_is_not_offered_as_a_batch():
    """A stroke rights the row and closes its gaps; the insertion needs one.

    Measured pitch after a stroke is 6.336 mm against the layout's 8.833 mm, and
    the pads then press on the 1.35 mm-wide flat tops instead of entering a bevel
    (9.90 N jam with the layout pitch, 9.05 N aiming at the live pitch).  So the
    recovered column has to drop out of the candidate list rather than be tried.
    """
    env = _env()
    try:
        expert = A3CookieRecoveryExpert(env)
        expert.reset()
        pristine = [0, 1, 2, 3, 4]
        assert expert._insertion_gap_fits(pristine), "the laid-out row must stay graspable"
        # Close the row the way a stroke does: pack the centres to one Cookie
        # thickness apart, so the gap is zero.  Write qpos (`xpos` is derived).
        thickness = 2.0 * float(env.COOKIE_HALF_SIZE[1])
        first_y = float(env.SOURCE_POSITIONS[pristine[0]][1])
        for row, index in enumerate(pristine):
            joint = env.model.body_jntadr[env._cookie_bodies[index]]
            address = env.model.jnt_qposadr[joint]
            env.data.qpos[address + 1] = first_y + row * thickness
        mujoco.mj_forward(env.model, env.data)
        live = env.cookie_positions[pristine][:, 1]
        assert abs(float(np.median(np.diff(live))) - thickness) < 1e-6
        assert not expert._insertion_gap_fits(pristine), "a packed row must be rejected"
    finally:
        env.close()
