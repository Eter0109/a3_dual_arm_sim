"""The three facts that have to hold before a randomization range is trustworthy.

Scene randomization moves the boxes and jitters the start pose, and the scripted
expert depends on two consequences of that staying true:

  * the Cookies move *with* the source box, as one rigid translation.  The batch
    insertion is aimed at the layout's 2.5 mm gaps, so a range that moved them
    individually would silently invalidate the grasp it was tuned on.
  * every Cookie still registers as inside the source bin.  The containment check
    uses the live bin centre; if that were still the configured constant, a moved
    bin would report all eighty Cookies as spilled and no episode could ever
    succeed.
  * nothing moves further than it was asked to.  A wrong sign or an unclipped draw
    would look fine in one episode and break a shard in the next.

These run the simulator because the values come from the model, not from the
config: a range is only real once `reset` has applied it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from a3_dual_arm_sim.config import SceneRandomization, load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv

CONFIG = "configs/cookie_same_column.yaml"
MM = 1000.0
RANGES = dict(
    source_bin_xy_m=(0.006, 0.006),
    target_bin_xy_m=(0.005, 0.005),
    target_bin_yaw_rad=0.06,
    arm_home_rad=0.02,
)


class _Scene:
    """Open one environment with the ranges set, and close it at the end."""

    def __init__(self, ranges: dict | None = None):
        config = load_config(CONFIG)
        if ranges is not None:
            config = replace(config, randomization=SceneRandomization(**ranges))
        self.env = A3CookieTransferEnv(config, render_cameras=False)

    def __enter__(self) -> A3CookieTransferEnv:
        return self.env

    def __exit__(self, *exc) -> None:
        self.env.close()


OFF = {"randomize_cookies": False, "randomize_scene": False}


def _snapshot(env: A3CookieTransferEnv) -> dict[str, np.ndarray]:
    return {
        "source": np.asarray(env.SOURCE_CENTER).copy(),
        "target": np.asarray(env.data.xpos[env._target_bin_body]).copy(),
        "cookies": env.cookie_positions.copy(),
        "home": env.last_applied_action[:7].copy(),
    }


def test_both_switches_off_reproduces_the_previous_scene():
    """All-zero ranges, or both switches off, must be the layout from before.

    This is what lets the feature be added to a scene without re-validating an
    existing artifact: an episode collected with it off has to be byte-identical
    to one collected before it existed.
    """

    with _Scene(ranges=None) as env:
        env.reset(seed=0, options=OFF)
        first = _snapshot(env)
        env.reset(seed=1, options=OFF)
        second = _snapshot(env)
    for key in ("source", "target", "cookies", "home"):
        np.testing.assert_array_equal(first[key], second[key], err_msg=f"{key} changed")


def test_nothing_moves_further_than_asked():
    """Each drawn offset stays inside its configured half-range."""

    with _Scene(ranges=RANGES) as env:
        nominal_source = env.SOURCE_NOMINAL_CENTER.copy()
        nominal_target = np.asarray(env.config.cookie_transfer.target_bin_world_position_m[:2])
        source_moves, target_moves, home_moves, yaws = [], [], [], []
        for seed in range(8):
            env.reset(seed=seed)
            state = _snapshot(env)
            source_moves.append(state["source"] - nominal_source)
            target_moves.append(state["target"][:2] - nominal_target)
            home_moves.append(np.abs(state["home"] - env.DEPLOYMENT_HOME[:7]).max())
            rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
            yaws.append(abs(float(np.arctan2(rotation[1, 0], rotation[0, 0]))))
    source_moves = np.asarray(source_moves)
    target_moves = np.asarray(target_moves)
    assert np.abs(source_moves).max() <= RANGES["source_bin_xy_m"][0]
    assert np.abs(target_moves).max() <= RANGES["target_bin_xy_m"][0]
    assert max(home_moves) <= RANGES["arm_home_rad"]
    assert max(yaws) <= RANGES["target_bin_yaw_rad"]
    # And the sample was not all zeros, or the bounds above prove nothing.
    assert np.abs(source_moves).max() > 0.0
    assert max(home_moves) > 0.0
    assert max(yaws) > 0.0


def test_cookie_layout_is_a_rigid_translation_of_the_nominal_one():
    """The Cookies follow the source box exactly, gaps included.

    This is the fact that lets a batch scene have scene variation at all: the
    insertion is aimed at the 2.5 mm gaps of the measured layout, and a rigid
    translation preserves every one of them.
    """

    with _Scene(ranges=RANGES) as env:
        env.reset(seed=0, options=OFF)
        nominal = env.cookie_positions[:, :2].copy()
        for seed in range(4):
            env.reset(seed=seed)
            offset = np.asarray(env.SOURCE_CENTER) - env.SOURCE_NOMINAL_CENTER
            delta = env.cookie_positions[:, :2] - nominal
            # Every Cookie moved by the same vector, which is the bin's offset.
            assert delta.shape == (len(env.SOURCE_POSITIONS), 2)
            np.testing.assert_allclose(
                delta, np.broadcast_to(offset, delta.shape), atol=1e-12,
                err_msg=f"seed {seed}: the layout did not follow the source box",
            )
            # And the spread is at the float64 epsilon, not merely small: the
            # Cookies are placed from the same nominal numbers plus one offset, so
            # any deviation here is rounding, not a changed gap.  Measured
            # 2.8e-17 m = 2.8e-14 mm, which is why the README's earlier
            # "0.000000 mm" was a display rounding rather than an exact zero.
            spread = float(np.abs(delta - offset).max())
            assert spread < 1e-15, (
                f"seed {seed}: per-Cookie deviation {spread} m means the gaps changed"
            )


def test_every_cookie_is_inside_the_moved_bin():
    """Containment has to follow the live bin, or a moved bin spills everything."""

    with _Scene(ranges=RANGES) as env:
        for seed in range(6):
            env.reset(seed=seed)
            inside = [
                env.privileged_cookie_in_source(index)
                for index in range(len(env.SOURCE_POSITIONS))
            ]
            assert all(inside), f"seed {seed}: {inside.count(False)} Cookies outside the bin"


def test_source_box_yaw_stays_identity():
    """The source box is translated but never rotated.

    The batch experts find a row by grouping the configured layout on `x`
    (`np.isclose(source[:, 0], x)`), so a rotated layout would stop matching its
    own rows.  `SceneRandomization` has no field for it; this pins that the
    absence is deliberate and that nothing else rotates the body.
    """

    assert not hasattr(SceneRandomization(), "source_bin_yaw_rad")
    with _Scene(ranges=RANGES) as env:
        for seed in range(4):
            env.reset(seed=seed)
            rotation = np.asarray(env.data.xmat[env._source_bin_body]).reshape(3, 3)
            np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)


def test_randomization_rejects_a_negative_range():
    """A negative half-range would silently invert or zero the draw."""

    with pytest.raises(ValueError, match="non-negative"):
        SceneRandomization(source_bin_xy_m=(-0.001, 0.0))
    with pytest.raises(ValueError, match="two values"):
        SceneRandomization(target_bin_xy_m=(0.001,))
