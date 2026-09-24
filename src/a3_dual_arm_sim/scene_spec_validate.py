"""The Monte-Carlo gate: draw layouts and run the scene's own checks on each.

Why this exists rather than a formula.  The derived ranges in
:mod:`a3_dual_arm_sim.scene_spec` are per-axis bounds, and every one of them was
measured with the *other* axes at nominal.  The acceptance windows are not
axis-aligned -- the fill's is a diagonal band, and the relay's shipped ranges have a
corner outside their own window that no per-axis bound predicts.  So a range cannot be
proved by arithmetic; it has to be *sampled*.

What is checked, per drawn layout, and why each is not redundant with the others:

* **the clearance** -- ``reset`` itself, because the env redraws until it finds a
  layout that passes ``_boxes_are_clear`` and raises if it cannot.  So "the reset
  succeeded" is the clearance check, and it is the real one rather than a copy.
* **the fill's reachability** -- ``A3CookieBatchExpert(env).reset()``, which is the
  pre-check that decides whether the station's placement poses can be solved at all.
  This is the check that refused the relay's corner.
* **the right arm's reach for the queue** -- ``BoxSupportController.solve`` at each
  queue box's carry grip pose, because a box the carry cannot grip is a box the relay
  cannot move however reachable the station is.

The gate reports the *first* failure with the drawn offsets, the box and the numbers,
rather than a count: one reproducible layout is worth more than a failure rate, and a
rate would need a much larger sample to say anything.

Cost is dominated by ``reset`` at about 5.4 s, which is why the default draw count is
modest and why :func:`fit_randomization` shrinks rather than re-drawing from scratch.
The three *checks* are 65 ms together, so a caller that only wants to validate a layout
it has already drawn can afford many more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .batch_expert import A3CookieBatchExpert
from .box_support import BoxSupportController
from .config import AxisRange, SimConfig, load_config
from .cookie_transfer import A3CookieTransferEnv
from .scene_spec import SceneRandomizationSpec, SceneSpec

#: How many layouts to draw by default.  A compromise: enough to hit a corner that is
#: a few percent of the range, not enough to characterise a one-in-a-thousand one.
#: ``scripts/check_scene_randomization.py`` runs a larger sweep for the record.
DEFAULT_DRAWS = 40


@dataclass(frozen=True)
class DrawFailure:
    """One layout the scene could not use, with everything needed to reproduce it."""

    seed: int
    check: str
    box: str
    detail: str
    #: The drawn offsets, per box, in the order the env draws them.
    layout: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"seed {self.seed}: {self.check} failed for {self.box}: {self.detail}"


def _sweep(config: SimConfig, draws: int, *, quiet: bool = False) -> list[DrawFailure]:
    """Draw ``draws`` layouts and return every one the scene cannot use.

    One environment for the whole sweep: building it is the expensive part and nothing
    here depends on a fresh model.  Each draw is a ``reset``, which is what applies the
    ranges and the clearance check together.
    """

    failures: list[DrawFailure] = []
    env = A3CookieTransferEnv(config, render_cameras=False)
    try:
        for seed in range(draws):
            try:
                env.reset(seed=seed, options={"randomize_cookies": False})
            except RuntimeError as exc:
                # The env exhausted its redraws: the ranges ask for a layout it cannot
                # find.  That is the clearance check failing, and it is the honest
                # place to report it -- a config whose nominal layout is clear can
                # still have ranges that collide.
                failures.append(
                    DrawFailure(
                        seed=seed,
                        check="clearance",
                        box="the drawn layout",
                        detail=str(exc),
                    )
                )
                continue

            layout = _describe_layout(env)
            try:
                A3CookieBatchExpert(env).reset()
            except RuntimeError as exc:
                failures.append(
                    DrawFailure(
                        seed=seed,
                        check="fill reach",
                        box="the station box",
                        detail=str(exc),
                        layout=layout,
                    )
                )
                continue

            failure = _check_queue_reach(env, seed, layout)
            if failure is not None:
                failures.append(failure)
            if not quiet and not failures:
                continue
    finally:
        env.close()
    return failures


def _describe_layout(env: A3CookieTransferEnv) -> dict[str, Any]:
    """The drawn pose of every box and of the source bin, for a failure report."""

    scene = env.config.cookie_transfer
    described: dict[str, Any] = {}
    for name, body, nominal in zip(
        scene.target_bin_body_names,
        env.target_bin_bodies,
        scene.target_bin_world_positions_m,
        strict=True,
    ):
        rotation = env.data.xmat[body].reshape(3, 3)
        described[name] = {
            "dx_mm": float(env.data.xpos[body][0] - nominal[0]) * 1000.0,
            "dy_mm": float(env.data.xpos[body][1] - nominal[1]) * 1000.0,
            "dyaw_deg": math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])),
        }
    source = env.data.xpos[env._source_bin_body][:2]
    described["source_bin"] = {
        "dx_mm": float(source[0] - scene.source_bin_center_m[0]) * 1000.0,
        "dy_mm": float(source[1] - scene.source_bin_center_m[1]) * 1000.0,
    }
    return described


def _check_queue_reach(
    env: A3CookieTransferEnv, seed: int, layout: dict[str, Any]
) -> DrawFailure | None:
    """Whether the right arm can grip every box behind the station.

    A queue box is moved by pinching its rear wall, so the pose that has to be
    reachable is the grip's, at ``box_y - half_y``.  Only the boxes past the station
    are checked: the station box is never gripped, it is filled where it stands.
    """

    scene = env.config.cookie_transfer
    if scene.target_bin_count < 2:
        return None
    helper = BoxSupportController(env)
    half_y = scene.target_bin_half_size_m[1]
    for name, body in zip(scene.target_bin_body_names[1:], env.target_bin_bodies[1:], strict=True):
        position = env.data.xpos[body]
        grip = np.array([position[0], position[1] - half_y, 0.780])
        try:
            helper.solve(grip, helper.quat)
        except RuntimeError as exc:
            return DrawFailure(
                seed=seed,
                check="carry reach",
                box=name,
                detail=(
                    f"grip pose {grip[0] * 1000:.1f}, {grip[1] * 1000:.1f}, "
                    f"{grip[2] * 1000:.1f} mm: {exc}"
                ),
                layout=layout,
            )
    return None


def check_randomization(
    config: SimConfig | str | Path,
    *,
    randomization: SceneRandomizationSpec | None = None,
    draws: int = DEFAULT_DRAWS,
) -> list[DrawFailure]:
    """Every layout the scene cannot use: the range's corners, then ``draws`` draws.

    Takes a config rather than a spec because the ranges live in the config by the
    time anything can be drawn -- a spec derives them and a config states them, and
    the gate's job is to check what will actually be used.  ``randomization`` names
    the ranges for the corner sweep, which the config does not carry in a form the
    corners can be read off; without it only the draws run.
    """

    if isinstance(config, (str, Path)):
        config = load_config(config)
    failures = check_corners(config, randomization) if randomization else []
    failures.extend(_sweep(config, draws))
    return failures


def _check_one(
    env: A3CookieTransferEnv,
    *,
    box_index: int,
    offset_m: tuple[float, float],
    yaw_rad: float,
    seed: int,
) -> DrawFailure | None:
    """Move one box to an absolute drawn pose and run the scene's checks on it.

    Absolute rather than relative, because a caller checking many layouts reuses one
    environment and an incremental move would accumulate.  ``mj_forward`` after the
    write is what makes the checks see the new pose.
    """

    scene = env.config.cookie_transfer
    body = env.target_bin_bodies[box_index]
    joint = env.model.body_jntadr[body]
    address = env.model.jnt_qposadr[joint]
    nominal = scene.target_bin_world_positions_m[box_index]
    env.data.qpos[address : address + 3] = [
        nominal[0] + offset_m[0],
        nominal[1] + offset_m[1],
        nominal[2],
    ]
    env.data.qpos[address + 3 : address + 7] = [
        math.cos(yaw_rad / 2),
        0.0,
        0.0,
        math.sin(yaw_rad / 2),
    ]
    mujoco.mj_forward(env.model, env.data)
    layout = _describe_layout(env)
    name = scene.target_bin_body_names[box_index]

    if box_index == 0:
        # Only the station box is filled, so only it has a placement pose to solve.
        try:
            A3CookieBatchExpert(env).reset()
        except RuntimeError as exc:
            return DrawFailure(
                seed=seed,
                check="fill reach",
                box=name,
                detail=str(exc),
                layout=layout,
            )
    elif scene.target_bin_count >= 2:
        failure = _check_queue_reach(env, seed, layout)
        if failure is not None:
            return failure
    return None


def _check_one_in_fresh_env(
    config: SimConfig,
    *,
    box_index: int,
    offset_m: tuple[float, float],
    yaw_rad: float,
) -> list[DrawFailure]:
    env = A3CookieTransferEnv(config, render_cameras=False)
    try:
        env.reset(seed=0, options={"randomize_cookies": False, "randomize_scene": False})
        failure = _check_one(env, box_index=box_index, offset_m=offset_m, yaw_rad=yaw_rad, seed=-1)
        return [] if failure is None else [failure]
    finally:
        env.close()


def check_layout(
    config: SimConfig | str | Path,
    *,
    box_index: int = 0,
    offset_m: tuple[float, float] = (0.0, 0.0),
    yaw_rad: float = 0.0,
) -> list[DrawFailure]:
    """Run the scene's checks on one layout stated explicitly.

    The same checks the sweep runs, on a layout the caller names instead of one the
    ranges produced.  That is what makes a *corner* testable: a failure a range can
    only reach by chance is one a test cannot pin, and a bound is only meaningful if
    the point that violates it can be reproduced.
    """

    if isinstance(config, (str, Path)):
        config = load_config(config)
    return _check_one_in_fresh_env(config, box_index=box_index, offset_m=offset_m, yaw_rad=yaw_rad)


def check_corners(
    config: SimConfig | str | Path,
    randomization: SceneRandomizationSpec,
) -> list[DrawFailure]:
    """Every extreme of every range, checked explicitly.

    This is the part of the gate that earns its keep, and the reason is worth stating:
    **a range fails at its corners, and random draws do not reach them.**  Measured,
    six random draws of the relay's derived ranges found nothing while its
    ``(x = -140 mm, y = +37 mm)`` corner was refused -- so a shrink loop driven by
    random draws alone returned the failing range unchanged.  The corners are also
    deterministic, which is what lets :func:`fit_randomization` converge instead of
    hoping.

    Position and yaw are combined rather than varied one at a time, because the
    windows are coupled: the relay's own range fails at ``(+4 mm, -6 mm)`` and the
    derived one at ``(-140 mm, +37 mm)``, and neither is visible on a single axis.

    One environment for the whole sweep, with each box written to its absolute pose
    before the checks -- about 65 ms a corner, against 5.4 s for a fresh reset.
    """

    if isinstance(config, (str, Path)):
        config = load_config(config)
    env = A3CookieTransferEnv(config, render_cameras=False)
    failures: list[DrawFailure] = []
    try:
        env.reset(seed=0, options={"randomize_cookies": False, "randomize_scene": False})
        count = env.config.cookie_transfer.target_bin_count
        for index in range(count):
            ranges = randomization.station if index == 0 else randomization.queue
            for dx in (ranges.x_m.low, ranges.x_m.high):
                for dy in (ranges.y_m.low, ranges.y_m.high):
                    for yaw in (ranges.yaw_rad.low, ranges.yaw_rad.high):
                        failure = _check_one(
                            env,
                            box_index=index,
                            offset_m=(dx, dy),
                            yaw_rad=yaw,
                            seed=-1,
                        )
                        if failure is not None:
                            failures.append(failure)
    finally:
        env.close()
    return failures


def fit_randomization(
    spec: SceneSpec,
    *,
    base_config: SimConfig | str | Path,
    draws: int = DEFAULT_DRAWS,
    shrink: float = 0.75,
    attempts: int = 8,
) -> tuple[SceneRandomizationSpec, list[DrawFailure]]:
    """Shrink a spec's derived ranges until the gate passes, and say what it took.

    The derivation gives the formula's answer and the gate says whether it is usable;
    this is the loop between them.  It is driven by the **corners**, which are
    deterministic: a random-draw loop returns the failing range unchanged when no draw
    happens to reach the corner, which is what the relay's derived ranges did on the
    first attempt at this.

    Shrinking is uniform across every range rather than per-axis.  A failure names
    which box it was, but the honest response to "this corner is outside the window"
    is not always to move that one axis -- the windows are coupled, so a smaller draw
    in every direction is what makes the corner safe.  A per-axis shrink would keep
    more range; it would also need the failing axis to be the one that is wrong, which
    the relay's `(+4 mm, -6 mm)` corner shows it is not.

    Returns the fitted ranges and the failures of the *last* attempt, so a caller can
    see what the loop was fighting even when it succeeded.
    """

    if not 0.0 < shrink < 1.0:
        raise ValueError(f"shrink must be in (0, 1), got {shrink}")
    if isinstance(base_config, (str, Path)):
        base_config = load_config(base_config)

    current = spec.randomization
    failures: list[DrawFailure] = []
    for _ in range(attempts):
        config = _with_randomization(base_config, current)
        failures = check_corners(config, current)
        if not failures:
            failures = _sweep(config, draws)
        if not failures:
            return current, []
        current = _shrunk(current, shrink)
    return current, failures


def _shrunk(randomization: SceneRandomizationSpec, factor: float) -> SceneRandomizationSpec:
    """Every range scaled by ``factor``, keeping each one's sign."""

    def scale(axis: AxisRange) -> AxisRange:
        if not axis.movable:
            return axis
        return AxisRange(low=axis.low * factor, high=axis.high * factor)

    def box(box_range):
        return replace(
            box_range,
            x_m=scale(box_range.x_m),
            y_m=scale(box_range.y_m),
            yaw_rad=scale(box_range.yaw_rad),
        )

    return replace(
        randomization,
        station=box(randomization.station),
        queue=box(randomization.queue),
        source=box(randomization.source),
    )


def _with_randomization(base: SimConfig, randomization: SceneRandomizationSpec) -> SimConfig:
    """A config carrying ``randomization``, for the gate to draw from.

    The section is loaded through ``config.load_config``'s own parser rather than
    assigned, so the gate tests the ranges the way a run would receive them -- a
    ``max_clearance_attempts`` or a clearance that the parser would reject has to fail
    here too, not only in the file.
    """

    from .config import SceneRandomization

    section = randomization.as_config()
    parsed = SceneRandomization(
        source_bin_x_m=AxisRange(*section["source_bin_x_m"]),
        source_bin_y_m=AxisRange(*section["source_bin_y_m"]),
        source_bin_yaw_rad=AxisRange(*section["source_bin_yaw_rad"]),
        target_bin_x_m=AxisRange(*section["target_bin_x_m"]),
        target_bin_y_m=AxisRange(*section["target_bin_y_m"]),
        target_bin_yaw_rad=AxisRange(*section["target_bin_yaw_rad"]),
        spare_bin_x_m=AxisRange(*section["spare_bin_x_m"]),
        spare_bin_y_m=AxisRange(*section["spare_bin_y_m"]),
        spare_bin_yaw_rad=AxisRange(*section["spare_bin_yaw_rad"]),
        arm_home_rad=AxisRange(*section["arm_home_rad"]),
        min_box_clearance_m=section["min_box_clearance_m"],
        max_clearance_attempts=section["max_clearance_attempts"],
    )
    return replace(base, randomization=parsed)


def describe_failures(failures: list[DrawFailure]) -> str:
    """One line per failure, plus the layout of the first, for a log or a report."""

    if not failures:
        return "no failures"
    lines = [str(failure) for failure in failures[:5]]
    if len(failures) > 5:
        lines.append(f"... and {len(failures) - 5} more")
    lines.append(f"first failing layout: {failures[0].layout}")
    return "\n".join(lines)
