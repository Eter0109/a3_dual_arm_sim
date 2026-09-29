"""Prepare grouped skill-policy training; only --run starts training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.training.skills import prepare_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim train skills", description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--policy-type", choices=("smolvla", "pi05"), default="smolvla")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=5000,
                        help="save a checkpoint every N steps (default: 5000)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: audit and print only")
    mode.add_argument("--run", action="store_true", help="explicitly start training")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_training(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
