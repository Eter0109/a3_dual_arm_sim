"""Derive a scene's randomization, gate it, and report what it took.

The Phase 5 record: for each shipped scene, what the formula derives, which corners the
gate refuses, and what the fitting loop converges to.  Run it after changing a measured
envelope or a spec parameter, because the derived ranges move with both.

    python scripts/check_scene_randomization.py                  # the three shipped scenes
    python scripts/check_scene_randomization.py --draws 40       # a larger random sweep

The default draw count is small on purpose: a reset is about 5.4 s, and the corners --
which is where a range fails -- are checked explicitly and deterministically, so the
draws are a backstop rather than the main check.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a3_dual_arm_sim.scene_spec import SceneSpec, same_column_spec, two_box_spec
from a3_dual_arm_sim.scene_spec_validate import (
    check_corners,
    check_randomization,
    describe_failures,
    fit_randomization,
)

#: Each shipped scene, the spec it decodes to, and the config it is built from.
SCENES: tuple[tuple[str, SceneSpec, str], ...] = (
    ("single box", same_column_spec(), "configs/cookie_same_column.yaml"),
    ("two boxes", two_box_spec(), "configs/cookie_two_box_batch.yaml"),
    (
        "three boxes",
        replace(two_box_spec(), boxes=3, queue_gap_m=0.160),
        "configs/generated/cookie_three_box.yaml",
    ),
)


def _describe(ranges) -> str:
    return (
        f"station x[{ranges.station.x_m.low * 1000:+.0f}, {ranges.station.x_m.high * 1000:+.0f}] "
        f"y[{ranges.station.y_m.low * 1000:+.0f}, {ranges.station.y_m.high * 1000:+.0f}] "
        f"yaw +-{ranges.station.yaw_rad.high:.3f}\n"
        f"      queue   x[{ranges.queue.x_m.low * 1000:+.0f}, {ranges.queue.x_m.high * 1000:+.0f}] "
        f"y[{ranges.queue.y_m.low * 1000:+.0f}, {ranges.queue.y_m.high * 1000:+.0f}] "
        f"yaw +-{ranges.queue.yaw_rad.high:.3f}\n"
        f"      source  y[{ranges.source.y_m.low * 1000:+.0f}, {ranges.source.y_m.high * 1000:+.0f}]"
        "   (mm, deg)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=6, help="random layouts per scene")
    parser.add_argument("--fit", action="store_true", help="also run the shrinking loop")
    args = parser.parse_args()

    for label, spec, config in SCENES:
        print(f"=== {label} ({config}) ===")
        try:
            spec.validate()
        except ValueError as exc:
            print(f"  spec refused:\n{exc}")
            continue
        ranges = spec.randomization
        print("  derived:")
        print(f"      {_describe(ranges)}")

        corners = check_corners(config, ranges)
        print(f"  corners: {len(corners)} refused")
        if corners:
            print("    " + describe_failures(corners).replace("\n", "\n    "))

        failures = check_randomization(config, randomization=ranges, draws=args.draws)
        print(f"  {args.draws} random draws: {len(failures) - len(corners)} refused")

        if args.fit:
            fitted, leftover = fit_randomization(spec, base_config=config, draws=args.draws)
            print("  fitted:")
            print(f"      {_describe(fitted)}")
            print(f"  leftover failures: {len(leftover)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
