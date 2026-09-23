"""Measure the three envelopes a cookie scene's layout is derived from.

A scene spec has to *derive* its box sizes, lane pitch and randomization ranges
from a few key parameters, and every derived bound is really a measurement of the
arm.  This script is where those measurements come from, so the numbers in
``docs/configurable-scenes.md`` can be re-taken rather than trusted.

Three sweeps, each answering one question:

1. **Where can a filling station be?**  The fill's own reachability pre-check
   (``A3CookieBatchExpert._validate_target_workspace``) is the authority, so the
   env is rebuilt at each pose and the check is run.  The usable set comes out as
   a diagonal band, which is why a station is one box rather than a region.

2. **Where can the right arm work?**  Both the push and the carry go through
   ``BoxSupportController.solve``, so its own acceptance rule is the envelope.
   The +y limit this prints is the reason a multi-box lane cannot push each parked
   box to its own slot, and the -y extent is what bounds the queue.

3. **How far can the source bin move out of the way?**  A lane pushes its filled
   boxes towards the source bin, so the bin has to move; this solves the first
   batch's pick pose at each offset to say how far it can.

Usage (from the repository root, with the headless MuJoCo setup):

    python scripts/measure_scene_envelopes.py [--config CONFIG] [--quick]

``--quick`` narrows each sweep to a couple of points, for checking that the
script still runs rather than for taking numbers.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from a3_dual_arm_sim.batch_expert import A3CookieBatchExpert
from a3_dual_arm_sim.box_support import BoxSupportController
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = ROOT / "configs" / "cookie_two_box_batch.yaml"
# Every sweep here is about the nominal layout, so both randomization switches are
# off: a drawn layout would make the numbers depend on the seed.
FIXED_SCENE = {"randomize_cookies": False, "randomize_scene": False}


def station_envelope(config_path: Path, quick: bool) -> None:
    xs = [0.075] if quick else [0.030, 0.060, 0.075, 0.090, 0.110]
    ys = [0.030, 0.090] if quick else [0.000, 0.030, 0.060, 0.090, 0.120, 0.150]
    base = load_config(config_path)
    print("=== station envelope: does the fill's own pre-check accept a box here? ===")
    print("  y\\x  " + "".join(f"{x * 1000:8.0f}" for x in xs))
    for y in ys:
        cells = []
        for x in xs:
            config = replace(
                base,
                cookie_transfer=replace(
                    base.cookie_transfer,
                    target_bin_world_position_m=(x, y, 0.753),
                ),
            )
            env = A3CookieTransferEnv(config, render_cameras=False)
            try:
                env.reset(seed=0, options=FIXED_SCENE)
                try:
                    A3CookieBatchExpert(env).reset()
                except RuntimeError:
                    cells.append("       .")
                else:
                    cells.append("       o")
            finally:
                env.close()
        print(f"  {y * 1000:5.0f} " + "".join(cells))
    print("  legend: o = both columns insertable, . = refused as unreachable")
    print()


def right_arm_envelope(config_path: Path, quick: bool) -> None:
    xs = [0.050, 0.075] if quick else [0.000, 0.025, 0.050, 0.075, 0.100, 0.125]
    ys = (
        [0.100, 0.125, 0.000, -0.350] if quick else [round(0.350 - 0.025 * i, 4) for i in range(29)]
    )
    # The push reaches for the box's rear wall at 0.755, the carry pinches it at
    # 0.780.  They are swept together because a lane has to be reachable for both.
    zs = [0.755, 0.780]
    env = A3CookieTransferEnv(config_path, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        helper = BoxSupportController(env)
        print("=== right-arm envelope: push (0.755) and carry (0.780) ===")
        print(f"  push quaternion = {np.round(helper.quat, 4).tolist()}")
        for z in zs:
            print(f"\n  --- z = {z:.3f} ---")
            print("   y\\x  " + "".join(f"{x * 1000:7.0f}" for x in xs))
            for y in ys:
                cells = []
                for x in xs:
                    try:
                        helper.solve(np.array([x, y, z]), helper.quat)
                    except RuntimeError:
                        cells.append("      .")
                    else:
                        cells.append("      o")
                print(f"  {y * 1000:5.0f} " + "".join(cells))
        print("  legend: o = IK accepted, . = refused (>12 mm or >0.04 rad)")
    finally:
        env.close()
    print()


def source_reach_map(config_path: Path, quick: bool) -> None:
    """Which (column, row) of the source layout can actually be grasped.

    The source grid is not uniformly reachable, and assuming it is was wrong: the
    far columns' first batch (x = 0.200, rows 0-4) is refused outright at 7.24 mm
    of IK error against a 4 mm tolerance, and the farther one at 19.12 mm -- so
    the shipped 4x20 layout is already only partly usable, and the expert's
    candidate rule quietly skips those cells.  That is the measurement that
    bounds how many Cookies a source bin can really offer, and the offset sweep
    exists to check that moving the bin +y (which a multi-box lane needs) does not
    shrink the usable set.
    """

    offsets_mm = [0, 148] if quick else [0, 58, 148]
    env = A3CookieTransferEnv(config_path, render_cameras=False)
    try:
        env.reset(seed=0, options=FIXED_SCENE)
        expert = A3CookieBatchExpert(env)
        expert.reset()
        source = np.asarray(env.SOURCE_POSITIONS)
        columns = sorted({float(x) for x in source[:, 0]})
        rows = sorted({float(y) for y in source[:, 1]})
        batch_size = 5
        print("=== source reach map: is a 5-Cookie batch starting here graspable? ===")
        target_q = np.empty(4)
        for offset_mm in offsets_mm:
            offset = offset_mm / 1000.0
            env._scene_offset = np.array([0.0, offset])
            mocap_id = int(env.model.body_mocapid[env._source_bin_body])
            nominal = env.SOURCE_NOMINAL_CENTER
            env.data.mocap_pos[mocap_id] = [nominal[0], nominal[1] + offset, 0.0]
            env.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(env.model, env.data)
            print(f"\n  --- source bin offset {offset_mm:+d} mm ---")
            print("  first row\\col  " + "".join(f"{x * 1000:7.0f}" for x in columns))
            for row in range(len(rows) - batch_size + 1):
                cells = []
                for column in columns:
                    indices = np.flatnonzero(np.isclose(source[:, 0], column))
                    batch = indices[row : row + batch_size].tolist()
                    positions = np.asarray([env.privileged_cookie_position(i) for i in batch])
                    pick_center = positions.mean(axis=0)
                    approach_eef = (
                        pick_center
                        + [0, 0, 0.010]
                        - expert._canonical @ expert._pad_offset
                        + [0, 0, expert.profile.grasp_clearance_m]
                    )
                    mujoco.mju_mat2Quat(target_q, expert._canonical.ravel())
                    try:
                        expert._solve_l(approach_eef, target_q, env.current_joint_action[:7])
                    except RuntimeError:
                        cells.append("      .")
                    else:
                        cells.append("      o")
                print(f"  {row:8d}       " + "".join(cells))
        print("  legend: o = a 5-Cookie batch from this row is graspable, . = refused")
        print("  (columns are the source layout's x; rows are counted from -y)")
    finally:
        env.close()
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--only",
        choices=("station", "right-arm", "source"),
        default=None,
        help="run one sweep instead of all three",
    )
    args = parser.parse_args()
    sweeps = {
        "station": station_envelope,
        "right-arm": right_arm_envelope,
        "source": source_reach_map,
    }
    selected = [args.only] if args.only else list(sweeps)
    started = time.time()
    for name in selected:
        sweeps[name](args.config, args.quick)
    print(f"elapsed {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
