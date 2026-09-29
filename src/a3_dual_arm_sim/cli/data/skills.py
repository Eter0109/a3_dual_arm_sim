"""Collect single-box PICK_FIVE/PLACE_FIVE LeRobot v3 skill episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.data.skill_collection import collect_skill_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim data skills", description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path,
                        default=project_root() / "configs/envs/cookie_batch.yaml")
    parser.add_argument("--repo-id")
    parser.add_argument("--rollouts", type=int, default=20, help="successful full-box rollouts")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel simulation workers; one parent process writes LeRobot data")
    parser.add_argument("--profile", choices=("baseline", "diverse"), default="baseline")
    parser.add_argument("--source-column", choices=("first", "random", "1", "2", "3", "4"),
                        default="first")
    parser.add_argument("--source-bin-x-min", type=float, default=None,
                        help="only simulate layouts with source-bin center X at least this many metres")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first simulator seed for a new dataset; use a disjoint range for supplements")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.rollouts < 1 or args.max_steps < 1 or args.workers < 1:
        parser.error("--rollouts, --max-steps and --workers must be positive")
    if args.profile == "baseline" and args.source_column != "first":
        parser.error("--source-column requires --profile diverse")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")
    return collect_skill_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
