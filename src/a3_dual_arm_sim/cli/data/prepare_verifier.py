"""Convert labeled temporal windows into seed-grouped verifier chat training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.data.verifier import prepare_verifier_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim data prepare-verifier", description=__doc__)
    parser.add_argument("--data-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-seeds", type=int, nargs="+")
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument(
        "--allow-missing-classes", action="store_true",
        help="diagnostic-only conversion; training/holdout coverage is incomplete",
    )
    args = parser.parse_args(argv)
    result = prepare_verifier_dataset(
        args.data_roots, args.output, holdout_seeds=args.holdout_seeds,
        holdout_fraction=args.holdout_fraction, split_seed=args.split_seed,
        max_samples_per_class=args.max_samples_per_class,
        require_class_coverage=not args.allow_missing_classes,
        window_frames=args.window_frames, frame_stride=args.frame_stride,
    )
    print(json.dumps({
        key: value for key, value in result.items() if key != "source_windows"
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
