"""Collect successful randomized single-box episodes in LeRobot v3 format."""

from __future__ import annotations

import argparse
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.data.action_collection import collect_action_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim data actions", description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, default=project_root() / "configs/envs/cookie_batch.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--repo-id")
    parser.add_argument("--profile", choices=("baseline", "diverse"), default="baseline")
    parser.add_argument(
        "--source-column", choices=("first", "random", "1", "2", "3", "4"),
        default="first", help="source column for diverse collection (1-4 or balanced random)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    diverse = args.profile == "diverse"
    if not diverse and args.source_column != "first":
        parser.error("--source-column requires --profile diverse")
    return collect_action_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
