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

from a3_dual_arm_sim.config import AxisRange, SceneRandomization, load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv

CONFIG = "configs/cookie_same_column.yaml"
MM = 1000.0
RANGES = dict(
    source_bin_x_m=AxisRange(-0.04, 0.04),
    source_bin_y_m=AxisRange(-0.04, 0.04),
    source_bin_yaw_rad=AxisRange(-0.04, 0.04),
    target_bin_x_m=AxisRange(-0.04, 0.02),
    target_bin_y_m=AxisRange(-0.03, 0.03),
    target_bin_yaw_rad=AxisRange(-0.06, 0.06),
    arm_home_rad=AxisRange(-0.02, 0.02),
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
    """Each drawn offset stays inside its configured range."""

    with _Scene(ranges=RANGES) as env:
        nominal_source = env.SOURCE_NOMINAL_CENTER.copy()
        nominal_target = np.asarray(env.config.cookie_transfer.target_bin_world_position_m[:2])
        source_moves, target_moves, home_moves, source_yaws, target_yaws = [], [], [], [], []
        for seed in range(8):
            env.reset(seed=seed)
            state = _snapshot(env)
            source_moves.append(state["source"] - nominal_source)
            target_moves.append(state["target"][:2] - nominal_target)
            home_moves.append(np.abs(state["home"] - env.DEPLOYMENT_HOME[:7]).max())
            source_rotation = env.data.xmat[env._source_bin_body].reshape(3, 3)
            source_yaws.append(
                float(np.arctan2(source_rotation[1, 0], source_rotation[0, 0]))
            )
            rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
            target_yaws.append(abs(float(np.arctan2(rotation[1, 0], rotation[0, 0]))))
    source_moves = np.asarray(source_moves)
    target_moves = np.asarray(target_moves)
    for index, name in enumerate(("x", "y")):
        assert source_moves[:, index].min() >= RANGES[f"source_bin_{name}_m"].low
        assert source_moves[:, index].max() <= RANGES[f"source_bin_{name}_m"].high
        assert target_moves[:, index].min() >= RANGES[f"target_bin_{name}_m"].low
        assert target_moves[:, index].max() <= RANGES[f"target_bin_{name}_m"].high
    assert min(source_yaws) >= RANGES["source_bin_yaw_rad"].low
    assert max(source_yaws) <= RANGES["source_bin_yaw_rad"].high
    assert max(home_moves) <= RANGES["arm_home_rad"].high
    assert max(target_yaws) <= RANGES["target_bin_yaw_rad"].high
    # And the sample was not all zeros, or the bounds above prove nothing.
    assert np.abs(source_moves).max() > 0.0
    assert np.abs(target_moves).max() > 0.0
    assert max(home_moves) > 0.0
    assert max(source_yaws) > 0.0
    assert max(target_yaws) > 0.0


def test_cookie_layout_is_a_rigid_transform_of_the_nominal_one():
    """The Cookies follow the source box exactly, gaps included.

    This is the fact that lets a batch scene have scene variation at all: the
    insertion is aimed at the 2.5 mm gaps of the measured layout, and a rigid
    transform -- a yaw about the bin centre, then a translation -- preserves every
    one of them.  A transform that resized or sheared the layout would change the
    gaps while still putting every Cookie somewhere plausible.
    """

    with _Scene(ranges=RANGES) as env:
        env.reset(seed=0, options=OFF)
        nominal = env.cookie_positions[:, :2].copy()
        centre = env.SOURCE_NOMINAL_CENTER
        for seed in range(4):
            env.reset(seed=seed)
            offset = np.asarray(env.SOURCE_CENTER) - centre
            rotation = env.data.xmat[env._source_bin_body].reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            cos, sin = np.cos(yaw), np.sin(yaw)
            spin = np.array([[cos, -sin], [sin, cos]])
            # Expected pose: rotate each nominal position about the bin centre,
            # then apply the drawn offset.
            expected = (nominal - centre) @ spin.T + centre + offset
            actual = env.cookie_positions[:, :2]
            np.testing.assert_allclose(
                actual, expected, atol=1e-12,
                err_msg=f"seed {seed}: the layout did not rigidly follow the source box",
            )
            # And the gaps are untouched: consecutive neighbours in a column stay
            # one pitch apart, which is what the insertion depends on.
            pitch = np.linalg.norm(np.diff(env.cookie_positions[:20, :2], axis=0), axis=1)
            nominal_pitch = np.linalg.norm(np.diff(nominal[:20], axis=0), axis=1)
            np.testing.assert_allclose(pitch, nominal_pitch, atol=1e-12)


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


def test_source_box_yaw_reaches_the_cookies():
    """A rotated bin carries its Cookies, rather than spinning around them.

    The Cookies are placed from the configured layout, so a yaw that only rotated
    the container would leave them axis-aligned inside a skewed bin -- the rows the
    expert enters would no longer be the rows the bin is holding.  Both the yaw and
    the containment check have to follow.
    """

    with _Scene(ranges=RANGES) as env:
        saw_yaw = False
        for seed in range(6):
            env.reset(seed=seed)
            rotation = env.data.xmat[env._source_bin_body].reshape(3, 3)
            bin_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            cookie_rotation = env.data.xmat[env._cookie_bodies[0]].reshape(3, 3)
            cookie_yaw = float(
                np.arctan2(cookie_rotation[1, 0], cookie_rotation[0, 0])
            )
            assert cookie_yaw == pytest.approx(bin_yaw, abs=1e-9), (
                f"seed {seed}: cookie yaw {cookie_yaw} does not follow the bin yaw {bin_yaw}"
            )
            saw_yaw = saw_yaw or abs(bin_yaw) > 1e-6
            assert all(
                env.privileged_cookie_in_source(index)
                for index in range(len(env.SOURCE_POSITIONS))
            ), f"seed {seed}: a rotated bin spilled its Cookies"
        assert saw_yaw, "no seed drew a yaw, so the check above proves nothing"


def test_a_range_that_cannot_leave_clearance_is_rejected():
    """Ranges asking for an impossible layout fail loudly, not silently.

    The source and target boxes are tens of millimetres apart, so a draw that puts
    them inside each other is discarded and redrawn.  If every draw is discarded,
    the ranges are asking for a layout that cannot exist, and falling back to a
    less varied episode would hide a config error behind a plausible-looking run.

    The source box is a mocap body with infinite mass, so such a collision would
    shove the target box out of the pose the expert planned for rather than being
    resolved between them.
    """

    with pytest.raises(RuntimeError, match="cannot exist"):
        with _Scene(
            ranges=dict(
                # Moves the source box *towards* the target box every time, so no
                # draw can leave the required gap.  The range has to be non-degenerate
                # to count as movement at all; `AxisRange(-0.30, -0.30)` is rejected
                # as ambiguous rather than being read as a 300 mm offset.
                source_bin_y_m=AxisRange(-0.30, -0.28),
                max_clearance_attempts=3,
            )
        ) as env:
            env.reset(seed=0)


def test_clearance_is_enforced_between_the_boxes():
    """Every accepted draw leaves the configured gap between box extents."""

    with _Scene(ranges=RANGES) as env:
        clearance = env.config.randomization.min_box_clearance_m
        for seed in range(8):
            env.reset(seed=seed)
            source = env.SOURCE_NOMINAL_CENTER + env._scene_offset
            target = env.data.xpos[env._target_bin_body][:2]
            source_half = env._source_bin_half_xy
            target_half = env._target_bin_half_xy
            source_rotation = env.data.xmat[env._source_bin_body].reshape(3, 3)
            gap = np.maximum(
                (source - source_half) - (target + target_half),
                (target - target_half) - (source + source_half),
            )
            assert float(np.max(gap)) >= clearance - 1e-9, (
                f"seed {seed}: boxes only {np.max(gap) * MM:.2f} mm apart "
                f"against a {clearance * MM:.2f} mm requirement"
            )
            del source_rotation


def test_randomization_rejects_an_inverted_range():
    """A high bound below the low bound would silently draw backwards."""

    with pytest.raises(ValueError, match="below low"):
        AxisRange(low=0.05, high=-0.05)
    with pytest.raises(ValueError, match="finite"):
        AxisRange(low=float("nan"), high=0.0)
    # A non-zero degenerate range is ambiguous, and reading it as "never move"
    # would quietly disable a range the config plainly meant to set.
    with pytest.raises(ValueError, match="degenerate"):
        AxisRange(low=0.05, high=0.05)
    # The default (0, 0) is the one degenerate range that means what it says.
    assert not AxisRange().movable
    with pytest.raises(ValueError, match="non-negative"):
        SceneRandomization(min_box_clearance_m=-0.01)
    with pytest.raises(ValueError, match="positive"):
        SceneRandomization(max_clearance_attempts=0)


def test_a_bare_number_is_read_as_a_symmetric_range():
    """The config accepts `0.05` as shorthand for `[-0.05, 0.05]`.

    Yaw and arm jitter are centred on the nominal pose, so a signed pair there is
    noise; the shorthand keeps those lines readable while the box ranges, which are
    genuinely asymmetric, spell both ends out.
    """

    import tempfile
    from pathlib import Path

    from a3_dual_arm_sim.config import load_config

    source = Path(CONFIG).read_text(encoding="utf-8")
    mirrored = source.replace("target_bin_yaw_rad: 0.10", "target_bin_yaw_rad: 0.10\n  spare_bin_yaw_rad: 0.03")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(mirrored, encoding="utf-8")
        config = load_config(path)
    assert config.randomization.spare_bin_yaw_rad.low == pytest.approx(-0.03)
    assert config.randomization.spare_bin_yaw_rad.high == pytest.approx(0.03)
