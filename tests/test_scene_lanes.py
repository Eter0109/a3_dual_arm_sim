"""A lane of N boxes: the config, the model and the reset all agree on it.

Phase 2 of `docs/configurable-scenes.md`.  Nothing here fills a box -- that is
Phase 3 and 4 -- so what is pinned is the part a fill would stand on: the model
builds every box the config asks for, the reset puts each one where its own
randomization range says, and no two boxes (or a box and the source bin) come
closer than the configured clearance.

The committed three-box config is checked against the spec that generated it, for
the same reason the two shipped configs are: a generated artifact that nothing
re-derives is a hand-written file wearing a "generated" header.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.config import load_config, target_bin_body_name
from a3_dual_arm_sim.cookie_transfer import (
    A3CookieTransferEnv,
    _rotated_half_extent,
)
from a3_dual_arm_sim.scene_spec import two_box_spec

ROOT = Path(__file__).resolve().parents[1]
THREE_BOX_CONFIG = ROOT / "configs" / "generated" / "cookie_three_box.yaml"
BASE_CONFIG = ROOT / "configs" / "cookie_two_box_batch.yaml"
sys.path.insert(0, str(ROOT / "scripts"))

from generate_scene_config import render_config

#: The spec the committed three-box config was generated from.  Written out rather
#: than imported so that changing the generator's defaults cannot silently change
#: what the committed file is supposed to be.
THREE_BOX_SPEC = replace(two_box_spec(), boxes=3, queue_gap_m=0.160)

#: Both randomization switches off, for the tests about the nominal layout.
FIXED_SCENE = {"randomize_cookies": False, "randomize_scene": False}


def _flatten(value: object) -> list[float]:
    """A nested tuple of numbers as a flat list, for comparing two configs."""

    if isinstance(value, (tuple, list)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]  # type: ignore[arg-type]


def _box_poses(env: A3CookieTransferEnv) -> list[tuple[np.ndarray, float]]:
    """Every tabletop box's live (centre, yaw)."""

    poses = []
    for body in env.target_bin_bodies:
        centre = env.data.xpos[body][:2].copy()
        rotation = env.data.xmat[body].reshape(3, 3)
        poses.append((centre, float(np.arctan2(rotation[1, 0], rotation[0, 0]))))
    return poses


def _worst_gap_m(env: A3CookieTransferEnv) -> float:
    """The largest per-axis gap over every pair, boxes and the source bin alike.

    The same rule the environment's own check uses, so a test that passes here
    cannot disagree with a reset that succeeded: two boxes are apart if they are
    apart on either axis.
    """

    scene = env.config.cookie_transfer
    half = np.asarray(scene.target_bin_half_size_m[:2], dtype=np.float64) + float(
        scene.bin_wall_thickness_m
    )
    source_half = env._source_bin_half_xy
    candidates: list[tuple[np.ndarray, np.ndarray, float]] = [
        (env.data.xpos[env._source_bin_body][:2].copy(), source_half, 0.0),
        *((centre, half, yaw) for centre, yaw in _box_poses(env)),
    ]
    worst = np.inf
    for index, (centre, own_half, yaw) in enumerate(candidates):
        for other_centre, other_half, other_yaw in candidates[index + 1 :]:
            extent = _rotated_half_extent(own_half, yaw)
            other_extent = _rotated_half_extent(other_half, other_yaw)
            gap = np.maximum(
                (centre - extent) - (other_centre + other_extent),
                (other_centre - other_extent) - (centre + extent),
            )
            worst = min(worst, float(np.max(gap)))
    return worst


# ------------------------------------------------------------------- the config


def test_the_committed_three_box_config_is_what_its_spec_generates():
    """A generated config has to stay regenerable, or it is a hand-written file.

    Byte-for-byte, because the generator is deterministic and the header states the
    spec's own numbers -- so a diff here means either the spec or the generator
    changed, and both are things a reader wants to be told about.
    """

    committed = THREE_BOX_CONFIG.read_text(encoding="utf-8")
    assert committed == render_config(BASE_CONFIG, THREE_BOX_SPEC)


def test_the_three_box_config_states_a_lane_and_a_moved_source_bin():
    """The keys a lane needs, and the source bin out of the lane's way.

    The offset is the derivation's, not a copied number: the parked line's front
    edge after two fills is at 0.030 + 2 * 0.087 + 0.031 = 0.235 m, and the source
    bin's front edge is 0.214 m, so the bin moves 46 mm to keep the clearance.
    """

    scene = load_config(THREE_BOX_CONFIG).cookie_transfer
    assert scene.target_bin_count == 3
    assert scene.target_bin_body_names == (
        "target_bin",
        "spare_target_bin",
        "queue_target_bin_2",
    )
    assert scene.source_bin_center_m == pytest.approx((0.175, 0.3610833333333334))
    assert scene.target_bin_world_positions_m[0] == pytest.approx((0.075, 0.030, 0.753))
    assert scene.target_bin_world_positions_m[2] == pytest.approx((0.075, -0.290, 0.753))
    # And the box geometry is unchanged from the two-box scene, because the lane is
    # more boxes of the same kind rather than a different box.  Compared through
    # `approx` because a generated config writes full precision while a shipped one
    # is written to ten significant digits.
    base = load_config(BASE_CONFIG).cookie_transfer
    assert list(_flatten(scene.target_bin_half_size_m)) == pytest.approx(
        list(_flatten(base.target_bin_half_size_m))
    )
    assert list(_flatten(scene.target_slots_local_m)) == pytest.approx(
        list(_flatten(base.target_slots_local_m))
    )


def test_the_lane_keys_follow_the_box_count():
    """One box states nothing, two state a spare, three add a queue.

    This is the schema's split, and it is what lets a two-box scene keep the keys
    its config, its tests and the relay's controllers all name.
    """

    one = replace(two_box_spec(), boxes=1).derive_config()
    assert "spare_target_bin_world_position_m" not in one
    assert "queue_target_bin_world_positions_m" not in one
    two = two_box_spec().derive_config()
    assert "spare_target_bin_world_position_m" in two
    assert "queue_target_bin_world_positions_m" not in two
    three = THREE_BOX_SPEC.derive_config()
    assert "spare_target_bin_world_position_m" in three
    assert three["queue_target_bin_world_positions_m"] == [[0.075, -0.29000000000000004, 0.753]]


def test_body_names_are_the_ones_everything_else_refers_to():
    assert target_bin_body_name(0) == "target_bin"
    assert target_bin_body_name(1) == "spare_target_bin"
    assert target_bin_body_name(2) == "queue_target_bin_2"
    assert target_bin_body_name(7) == "queue_target_bin_7"


def test_a_nominal_layout_that_is_too_tight_is_refused_at_load():
    """The check is at load, not at reset, because that is where the mistake is.

    A config whose nominal boxes already overlap would exhaust its redraws on the
    first episode and report a randomization problem -- which would send a reader
    to the ranges rather than to the box positions.
    """

    raw = THREE_BOX_CONFIG.read_text(encoding="utf-8")
    # Pull the queue box up until it is inside the spare's berth.  The literal is
    # the generated file's own text, so a change to how a position is written fails
    # here rather than silently testing nothing.
    broken = raw.replace("[0.075, -0.29000000000000004, 0.753]", "[0.075, -0.13, 0.753]")
    assert broken != raw, "the queue position literal moved; update this test"
    path = THREE_BOX_CONFIG.parent / "tmp_tight_lane.yaml"
    path.write_text(broken, encoding="utf-8")
    try:
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        message = str(excinfo.value)
        assert "the nominal box layout leaves only" in message
        assert "min_box_clearance_m" in message
    finally:
        path.unlink()


# ------------------------------------------------------------ the model and reset


def test_the_model_builds_every_box_the_config_asks_for():
    """Three free bodies with three free joints, and the names are the contract."""

    env = A3CookieTransferEnv(THREE_BOX_CONFIG, render_cameras=False)
    try:
        assert len(env.target_bin_bodies) == 3
        assert env._target_bin_body == env.target_bin_bodies[0]
        for name in ("target_bin", "spare_target_bin", "queue_target_bin_2"):
            body = env.model.body(name)
            assert body.id in env.target_bin_bodies
            joint = env.model.joint(f"{name}_free")
            assert joint.type == mujoco.mjtJoint.mjJNT_FREE
        # The boxes are a lane, so they are ordered in -y and share an x.
        ys = [env.data.xpos[body][1] for body in env.target_bin_bodies]
        assert ys == sorted(ys, reverse=True)
    finally:
        env.close()


def test_every_box_lands_inside_its_own_drawn_range():
    """A lane's boxes are drawn independently, and each stays in its own range.

    The station draws from `target_bin_*` and every box behind it from `spare_bin_*`,
    because those are roles rather than indices: whatever ends up in the queue has to
    arrive at the station inside the fill's window, which is what bounds the spare's
    ranges.  A third box is in that position too.
    """

    env = A3CookieTransferEnv(THREE_BOX_CONFIG, render_cameras=False)
    scene = env.config.cookie_transfer
    randomization = env.config.randomization
    ranges = [
        (randomization.target_bin_x_m, randomization.target_bin_y_m),
        (randomization.spare_bin_x_m, randomization.spare_bin_y_m),
        (randomization.spare_bin_x_m, randomization.spare_bin_y_m),
    ]
    try:
        for seed in range(8):
            env.reset(seed=seed, options={"randomize_cookies": False})
            for nominal, (x_range, y_range), (centre, _) in zip(
                scene.target_bin_world_positions_m,
                ranges,
                _box_poses(env),
                strict=True,
            ):
                dx = float(centre[0] - nominal[0])
                dy = float(centre[1] - nominal[1])
                assert x_range.low - 1e-9 <= dx <= x_range.high + 1e-9, (seed, dx)
                assert y_range.low - 1e-9 <= dy <= y_range.high + 1e-9, (seed, dy)
    finally:
        env.close()


def test_no_two_boxes_ever_come_closer_than_the_clearance():
    """Every pair, over many seeds, including the source bin.

    A lane makes this harder rather than easier: a three-box scene has three
    box-to-box pairs against the two-box scene's one.  The committed config keeps
    them apart by laying the queue out at the relay's 160 mm gap rather than the
    87 mm push pitch -- see
    `test_a_lane_at_the_push_pitch_is_refused_because_its_layout_is_too_tight` for
    why the pitch is not an option.
    """

    env = A3CookieTransferEnv(THREE_BOX_CONFIG, render_cameras=False)
    clearance = env.config.randomization.min_box_clearance_m
    try:
        gaps = []
        for seed in range(12):
            env.reset(seed=seed, options={"randomize_cookies": False})
            gap = _worst_gap_m(env)
            gaps.append(gap)
            assert gap >= clearance - 1e-9, (
                f"seed {seed}: worst gap {gap * 1000:.1f} mm against a "
                f"{clearance * 1000:.1f} mm clearance"
            )
    finally:
        env.close()
    # And the bound is not vacuous: the tightest draw really does come close to it.
    assert min(gaps) < 4.0 * clearance, (
        f"the tightest of {len(gaps)} draws was {min(gaps) * 1000:.1f} mm, which is "
        f"so far above the {clearance * 1000:.1f} mm bound that the check proves nothing"
    )


def test_a_lane_at_the_push_pitch_is_refused_because_its_layout_is_too_tight():
    """Why the committed config uses the relay's gap and not the derived pitch.

    The push pitch is `box_extent_y + min_box_clearance`, which is the spacing the
    line compacts to *after* a shove, and it is what the source bin's clearance is
    computed from.  Laying a lane out at it is a different question and the answer is
    no: two boxes 87 mm apart have 13 mm between their *outer* extents, against the
    25 mm the scene draws within.

    This pins the measurement rather than a message, because at the derived pitch the
    queue-budget check fires first and refuses the lane before anything gets as far as
    the layout check -- see
    `test_a_lane_at_the_push_pitch_has_no_room_for_its_own_yaw`.  Both refusals are
    true; the sharper one is the one a caller sees.
    """

    tight = replace(THREE_BOX_SPEC, queue_gap_m=None)
    assert tight.lane_pitch == pytest.approx(0.087)
    assert tight.queue_gap == tight.lane_pitch
    # The measurement: the layout spacing leaves 13 mm, not the 25 mm it draws within.
    outer_half_y = tight.target_bin_half_size_m[1] + tight.geometry.bin_wall_thickness_m
    nominal_gap = tight.queue_gap - 2.0 * outer_half_y
    assert nominal_gap == pytest.approx(0.013, abs=1e-9)
    assert nominal_gap < tight.min_box_clearance_m
    # And that is what `CookieSceneConfig.worst_nominal_box_gap_m` reports, so the
    # load-time check would catch it too if the lane got that far.
    config = load_config(THREE_BOX_CONFIG)
    assert config.cookie_transfer.worst_nominal_box_gap_m == pytest.approx(0.086, abs=1e-9)
    assert (
        config.cookie_transfer.worst_nominal_box_gap_m >= config.randomization.min_box_clearance_m
    )


def test_a_lane_at_the_push_pitch_has_no_room_for_its_own_yaw():
    """The same lane, refused one step earlier and by a sharper number.

    Rendering it calls `SceneSpec.validate`, which since Phase 5 also checks that the
    queue has room for the yaw its boxes arrive with -- and at the derived pitch it
    does not, because the pitch is exactly `box_extent_y + clearance` while a box
    turned by the 4 deg the fill tolerates is 70.6 mm wide.  So the message names the
    gap the lane would need (95.6 mm) rather than only saying the layout is too tight,
    which is the more useful of the two refusals.
    """

    tight = replace(THREE_BOX_SPEC, queue_gap_m=None)
    with pytest.raises(ValueError) as excinfo:
        tight.derive_config()
    message = str(excinfo.value)
    assert "leaves nothing for the queue to move" in message
    assert "the gap has to exceed 95.6 mm" in message


def test_the_source_bin_clears_the_line_the_fills_will_leave_behind():
    """The offset is measured against the parked line, not the laid-out one.

    After two fills the line has compacted to the push pitch and the first parked box
    has advanced 2 * 87 mm, so its front edge is at 0.235 m.  The source bin's front
    edge has to stay a clearance beyond that, and the config states the bin where the
    derivation put it.
    """

    spec = THREE_BOX_SPEC
    scene = load_config(THREE_BOX_CONFIG).cookie_transfer
    parked_front = spec.station_y_m + spec.lane_pitch * (spec.boxes - 1) + spec.box_extent_y_m / 2.0
    source_front = scene.source_bin_center_m[1] - scene.source_bin_half_size_m[1]
    assert source_front - parked_front >= spec.min_box_clearance_m - 1e-9
    assert spec.source_offset_m > 0.0, "this lane should need the bin moved"


def test_a_three_box_scene_resets_with_the_nominal_layout_when_randomization_is_off():
    """With both switches off, every box is exactly where the config says.

    The same property the two shipped scenes have, and what makes a fixed-layout run
    reproducible: an all-zero draw is the nominal layout rather than a nearly-nominal
    one.
    """

    env = A3CookieTransferEnv(THREE_BOX_CONFIG, render_cameras=False)
    scene = env.config.cookie_transfer
    try:
        for seed in (0, 3):
            env.reset(seed=seed, options=FIXED_SCENE)
            for body, nominal in zip(
                env.target_bin_bodies,
                scene.target_bin_world_positions_m,
                strict=True,
            ):
                np.testing.assert_allclose(
                    env.data.xpos[body][:2], np.asarray(nominal[:2]), atol=1e-9
                )
            np.testing.assert_allclose(
                env.data.xpos[env._source_bin_body][:2],
                np.asarray(scene.source_bin_center_m),
                atol=1e-9,
            )
    finally:
        env.close()


def test_the_boxes_are_all_served_by_one_task_config():
    """A lane is not a new task: the environment still grades its one station box.

    Worth pinning because it is why the three-box config needs no new task spec, and
    why the relay's own criterion lives in the expert rather than in the environment
    (see `TwoBoxCollectionPolicy`).  What changes with N is the workflow, not the
    contract.
    """

    env = A3CookieTransferEnv(THREE_BOX_CONFIG, render_cameras=False)
    try:
        assert env.task_config.required_cookies == 10
        assert env.task_config.cookie_count == 80
        assert len(env.TARGET_SLOTS_LOCAL) == 10
        # The environment grades `_target_bin_body`, which is the station.
        assert env._target_bin_body == env.target_bin_bodies[0]
    finally:
        env.close()
