#!/usr/bin/env python3
"""Evaluate the closed-loop Cookie Expert and save per-seed JSON metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_dual_arm_sim.evaluation import evaluate_cookie_transfer


def parse_seeds(value: str) -> list[int]:
    if "-" in value and "," not in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate A3 Cookie transfer Expert")
    parser.add_argument("--seeds", default="0", help="Seed, comma list, or range")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument(
        "--fixed-layout",
        action="store_true",
        help="Disable configured Cookie pose randomization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cookie_transfer_evaluation.json"),
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    report = evaluate_cookie_transfer(
        seeds,
        max_steps=args.max_steps,
        randomize_cookies=not args.fixed_layout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    print(
        f"episodes={summary['episodes']} successes={summary['successes']} "
        f"success_rate={summary['success_rate']:.1%} "
        f"mean_completed={summary['mean_completed_cookies']:.2f}"
    )
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
