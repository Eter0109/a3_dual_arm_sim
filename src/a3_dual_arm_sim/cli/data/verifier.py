"""Collect independently labeled temporal RGB windows without action recording."""

from __future__ import annotations

import argparse
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.data.verifier_collection import FAILURE_CASES, collect_verifier_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim data verifier", description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="new independent verifier dataset root"
    )
    parser.add_argument(
        "--config", type=Path, default=project_root() / "configs/envs/cookie_batch.yaml"
    )
    parser.add_argument(
        "--cases", nargs="+", choices=FAILURE_CASES, default=["miss_grasp", "drop", "stuck"]
    )
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seed", type=int, default=0, help="one seed; default 0")
    seeds_group.add_argument(
        "--seeds", type=int, nargs="+", help="explicit seed list; no implicit large collection"
    )
    parser.add_argument("--source-column", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--profile", choices=("fixed", "diverse"), default="fixed")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--post-fault-steps", type=int, default=80)
    parser.add_argument(
        "--skill-hold-steps",
        type=int,
        default=0,
        help="hold completed non-final skills before switching; stop if completion is lost",
    )
    parser.add_argument(
        "--terminal-hold-steps",
        type=int,
        default=0,
        help="after final verified skill, record additional safe-hold steps; default disabled",
    )
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--window-frames", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--render-recorded-frames-only",
        action="store_true",
        help="render RGB only at recorder frame-stride boundaries; physics still runs every step",
    )
    args = parser.parse_args(argv)
    if min(args.max_steps, args.post_fault_steps, args.record_every, args.frame_stride) < 1:
        parser.error("step limits and frame stride must be positive")
    if args.window_frames < 2:
        parser.error("temporal verification requires at least two frames")
    if args.terminal_hold_steps < 0:
        parser.error("--terminal-hold-steps must be non-negative")
    if args.skill_hold_steps < 0:
        parser.error("--skill-hold-steps must be non-negative")
    if len(set(args.cases)) != len(args.cases):
        parser.error("do not repeat a case in one dataset root")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("do not repeat a seed in one dataset root")
    return collect_verifier_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
