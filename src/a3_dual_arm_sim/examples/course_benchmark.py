"""Launch the complete course benchmark and save its report.

Run from the project root after installation::

    MUJOCO_GL=egl python -m a3_dual_arm_sim.examples.course_benchmark

All options are forwarded to the course benchmark CLI. Unless --output is
provided, a unique report path is chosen under outputs/course_benchmark/.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    from a3_dual_arm_sim.workflows.course_benchmark_cli import main as run_course

    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        return run_course()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    options, _ = parser.parse_known_args()
    output = options.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        output = Path("outputs/course_benchmark") / f"course_{stamp}_{os.getpid()}.json"
        sys.argv.extend(["--output", str(output)])

    if output.exists():
        print(f"Output already exists; choose a new --output path: {output}", file=sys.stderr)
        return 2

    print(f"Course report: {output}", file=sys.stderr, flush=True)
    return run_course()


if __name__ == "__main__":
    raise SystemExit(main())
