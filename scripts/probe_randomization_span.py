"""What a scene's randomization ranges actually draw, and whether they can be drawn.

`SceneRandomization` is a *request*.  Each box draws an offset from its configured
range, and the set is then checked: a draw that leaves two boxes closer than
`min_box_clearance_m` is discarded and redrawn, and `max_clearance_attempts` of those
in a row is a configuration error that **raises** rather than quietly falling back to
a narrower episode.  So a range that is too wide fails at `reset`, in the middle of a
collection, and the failure is a crash rather than a shortfall.

Two things a config file cannot show, and this reports:

* **Feasibility** -- how many seeds draw a legal layout at all.
* **Realisation** -- what offsets the live poses actually took, per axis.  The
  configured range is what was asked for; a sample of the realisation is what a
  dataset collected afterwards would contain.  A run whose axes never approach their
  configured bound is a run whose config did not take effect, which is otherwise
  invisible: `axes()` drops an all-zero axis from the summary, so a knob left at zero
  leaves no trace anywhere.

The clearance gap is recomputed here from the live poses rather than read from the
environment's own check, because the question is how much room is *left* rather than
whether a particular draw passed -- the guard's answer is already known to be yes for
every seed that returned.

Usage:

    python scripts/probe_randomization_span.py --seeds 0-199
    python scripts/probe_randomization_span.py --scene a3_libero_panda_two_cookies
"""

import argparse
import math
import os
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from a3_dual_arm_sim.collection import scene_by_name
from a3_dual_arm_sim.cookie_transfer import _rotate_xy

#: The axes of `SceneRandomization` that are drawn per episode, in the order the
#: summary records them.
AXES = (
    "source_bin_x_m",
    "source_bin_y_m",
    "source_bin_yaw_rad",
    "target_bin_x_m",
    "target_bin_y_m",
    "target_bin_yaw_rad",
    "arm_home_rad",
)


def _yaw_of(quaternion: np.ndarray) -> float:
    """Yaw of a Hamilton quaternion in MuJoCo's ``(w, x, y, z)`` order.

    The boxes are only ever yawed, so the general rotation matrix would be three
    lines of arithmetic for a quantity that is one ``atan2`` away.
    """

    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotated_half_extents(half_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Axis-aligned half extents of a rectangle with ``half_xy`` turned by ``yaw``."""

    cos, sin = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx, hy = (float(value) for value in half_xy)
    return np.array([hx * cos + hy * sin, hx * sin + hy * cos])


def _box_gap(env, source_xy, source_yaw, target_xy, target_yaw) -> float:
    """Largest per-axis gap between the two boxes' rotated extents.

    The largest rather than the smallest, which is `_boxes_are_clear`'s own rule:
    two boxes side by side in y are not overlapping merely because they share an x
    range.
    """

    source_half = _rotated_half_extents(env._source_bin_half_xy, source_yaw)
    target_half = _rotated_half_extents(env._target_bin_half_xy, target_yaw)
    return float(np.max(np.abs(np.asarray(source_xy) - np.asarray(target_xy)) - source_half - target_half))


def _parse_seeds(specification: str) -> list[int]:
    seeds: list[int] = []
    for piece in specification.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            low, high = piece.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(piece))
    return seeds


def _capture(env) -> dict[str, Any]:
    """Every randomised quantity, as this episode actually starts with it.

    Read after a reset rather than derived from the config, because the question is what
    the episode starts from -- and a value configured but never applied is exactly the
    failure this probe exists to catch.  `arm_home_rad` was 0.04 in the collection plan,
    0.0 in the config and absent from the summary, so 500 episodes were collected from a
    byte-identical starting pose and nothing anywhere compared the two.

    **Every value is copied.**  `np.asarray` over a slice of `data.xpos` or `data.qpos`
    returns a *view*, so a "nominal" read that way keeps tracking the live simulation and
    every difference against it comes out as exactly zero.  That is not a hypothetical: the
    first version of this probe did it, and reported a 0.000 mm source-bin translation
    beside a box-to-box gap that had visibly changed -- the disagreement between two
    figures from one run is the only reason it was caught.
    """

    target_joint = env.model.body_jntadr[env._target_bin_body]
    target_address = int(env.model.jnt_qposadr[target_joint])
    cookies = []
    for index in range(env.task_config.cookie_count):
        address = int(env.model.jnt_qposadr[env._cookie_joints[index]])
        cookies.append(
            (
                np.array(env.data.qpos[address : address + 3], dtype=np.float64),
                _yaw_of(env.data.qpos[address + 3 : address + 7]),
            )
        )
    return {
        "source_xy": np.array(env.SOURCE_CENTER, dtype=np.float64),
        "source_yaw": float(env._scene_yaw),
        "target_xy": np.array(
            env.data.qpos[target_address : target_address + 2], dtype=np.float64
        ),
        "target_yaw": _yaw_of(env.data.qpos[target_address + 3 : target_address + 7]),
        # The arm jitter is stored rather than applied here; `reset` writes the home after
        # this returns, so this is the pose the episode will begin from.
        "home": np.array(env._randomized_deployment_home[:7], dtype=np.float64),
        "cookies": cookies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_libero_panda_two_cookies")
    parser.add_argument("--seeds", default="0-199")
    parser.add_argument(
        "--position-noise",
        type=float,
        default=0.0,
        help="Per-Cookie XY jitter in metres, the same argument `collect` takes.  It is a "
        "build argument rather than a config field, so a probe that left it at zero "
        "would describe a collection nobody runs.",
    )
    parser.add_argument(
        "--yaw-noise",
        type=float,
        default=0.0,
        help="Per-Cookie yaw jitter in radians, likewise.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only the summary, not one line per seed that failed to draw",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    scene = scene_by_name(args.scene)
    env = scene.build_env(args.position_noise, args.yaw_noise)
    try:
        configured = env.config.randomization
        # The nominal is *measured*, on a pinned reset, rather than derived from the config.
        # Whatever frame the scene's own quantities live in, the difference against this is
        # the realisation -- and a derivation would be a second copy of the scene's
        # arithmetic, free to disagree with it.
        env.reset(
            seed=seeds[0], options={"randomize_scene": False, "randomize_cookies": False}
        )
        nominal = _capture(env)

        print(f"scene      : {args.scene}")
        print(f"seeds      : {len(seeds)}")
        print(f"clearance  : {configured.min_box_clearance_m * 1000:.1f} mm required, "
              f"{configured.max_clearance_attempts} draws before giving up")
        print()

        cookie: dict[str, list[float]] = {"x_m": [], "y_m": [], "yaw_rad": []}
        separation: list[float] = []
        offsets: dict[str, list[float]] = {name: [] for name in AXES}
        gaps: list[float] = []
        failed: list[tuple[int, str]] = []

        def delta(key: str, component: int | None = None) -> float:
            here, there = realised[key], nominal[key]
            if component is None:
                return float(here - there)
            return float(here[component] - there[component])

        for seed in seeds:
            try:
                env.reset(
                    seed=seed,
                    # `randomize_cookies` too, not just the scene: `collect` passes the
                    # scene's own flags, which for this scene are both True, and a probe
                    # that left the per-Cookie draw off would report a distribution no
                    # collection produces -- it did, as a four-line table of zeroes.
                    options={"randomize_scene": True, "randomize_cookies": True},
                )
            except RuntimeError as error:  # the clearance guard gave up
                failed.append((seed, str(error)))
                if not args.quiet:
                    print(f"  seed {seed}: {error}")
                continue

            realised = _capture(env)
            offsets["source_bin_x_m"].append(delta("source_xy", 0))
            offsets["source_bin_y_m"].append(delta("source_xy", 1))
            offsets["source_bin_yaw_rad"].append(delta("source_yaw"))
            offsets["target_bin_x_m"].append(delta("target_xy", 0))
            offsets["target_bin_y_m"].append(delta("target_xy", 1))
            offsets["target_bin_yaw_rad"].append(delta("target_yaw"))
            # Drawn once and added to both arms' seven joints, so either arm's seven
            # values are the same sample.
            offsets["arm_home_rad"].extend((realised["home"] - nominal["home"]).tolist())

            for index, (world, yaw) in enumerate(realised["cookies"]):
                nominal_world, nominal_yaw = nominal["cookies"][index]
                # Into the live bin's own frame: the Cookies ride the bin, so a world-frame
                # reading would mix the bin's movement with the Cookie's own placement, and
                # "where is it in the box" is the question the jitter answers.
                here = _rotate_xy(world[:2] - realised["source_xy"], -realised["source_yaw"])
                there = _rotate_xy(
                    nominal_world[:2] - nominal["source_xy"], -nominal["source_yaw"]
                )
                cookie["x_m"].append(float(here[0] - there[0]))
                cookie["y_m"].append(float(here[1] - there[1]))
                # A Cookie's world yaw is its own draw plus the bin's, so taking the bin's
                # back out leaves the per-Cookie rotation on its own.
                cookie["yaw_rad"].append(
                    (yaw - realised["source_yaw"]) - (nominal_yaw - nominal["source_yaw"])
                )
            if len(realised["cookies"]) >= 2:
                separation.append(
                    float(
                        np.linalg.norm(
                            realised["cookies"][0][0][:2] - realised["cookies"][1][0][:2]
                        )
                    )
                )
            gaps.append(
                _box_gap(
                    env,
                    realised["source_xy"],
                    realised["source_yaw"],
                    realised["target_xy"],
                    realised["target_yaw"],
                )
            )

        header = (f"{'axis':22s} {'configured':>22s} {'realised min':>13s} "
                  f"{'realised max':>13s} {'max |x|':>10s}")
        print("scene-level, per episode")
        print(header)
        for name in AXES:
            axis = getattr(configured, name)
            samples = np.asarray(offsets[name], dtype=np.float64)
            if samples.size == 0:
                print(f"{name:22s} {'(nothing drawn)':>22s}")
                continue
            unit = 1000.0 if name.endswith("_m") else 1.0
            suffix = "mm" if name.endswith("_m") else "rad"
            print(
                f"{name:22s} "
                f"{axis.low * unit:>10.3f} .. {axis.high * unit:<10.3f} "
                f"{samples.min() * unit:>12.3f} "
                f"{samples.max() * unit:>13.3f} "
                f"{np.abs(samples).max() * unit:>9.3f} {suffix}"
            )

        episodes = len(seeds) - len(failed)
        print()
        print(f"per-Cookie, in the source bin's own frame "
              f"({len(cookie['x_m'])} samples from {episodes} episodes)")
        print(header)
        for name, drawn, unit, suffix in (
            ("x_m", args.position_noise, 1000.0, "mm"),
            ("y_m", args.position_noise, 1000.0, "mm"),
            ("yaw_rad", args.yaw_noise, 1.0, "rad"),
        ):
            samples = np.asarray(cookie[name], dtype=np.float64)
            if samples.size == 0:
                continue
            print(
                f"{'cookie ' + name:22s} "
                f"{-drawn * unit:>10.3f} .. {drawn * unit:<10.3f} "
                f"{samples.min() * unit:>12.3f} "
                f"{samples.max() * unit:>13.3f} "
                f"{np.abs(samples).max() * unit:>9.3f} {suffix}"
            )
        if separation:
            nominal_separation = float(
                np.linalg.norm(nominal["cookies"][0][0][:2] - nominal["cookies"][1][0][:2])
            )
            pair = np.asarray(separation, dtype=np.float64)
            print(
                f"{'Cookie separation':22s} nominal {nominal_separation * 1000:8.1f} mm   "
                f"realised {pair.min() * 1000:8.1f} .. {pair.max() * 1000:.1f} mm   "
                f"(the relative geometry a policy cannot memorise)"
            )

        print()
        if gaps:
            print(f"box gap    : min {min(gaps) * 1000:.1f} mm, "
                  f"median {float(np.median(gaps)) * 1000:.1f} mm "
                  f"(required {configured.min_box_clearance_m * 1000:.1f} mm)")
        print(f"failures   : {len(failed)}/{len(seeds)} seeds could not draw a legal layout")

        # The realisation has to approach the request, or the config did not take
        # effect -- and an axis left at zero is dropped from the summary as well, so
        # nothing else would say so.
        slack = {}
        for name in AXES:
            axis = getattr(configured, name)
            if not axis.movable:
                continue
            biggest = max(abs(value) for value in offsets[name]) if offsets[name] else 0.0
            slack[name] = biggest / axis.maximum_magnitude()
        short = {name: share for name, share in slack.items() if share < 0.5}
        if short:
            print(
                "  warning: these axes never came within half their configured bound, "
                f"so they may not be in effect: {short}"
            )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
