#!/usr/bin/env bash
# Run the two test suites. They cannot share one process.
#
#   bash scripts/recap/run_tests.sh          # both suites
#   bash scripts/recap/run_tests.sh --all    # only the fast one (recap), ~1 s
#
# Why two invocations
# -------------------
# Two different packages share the name a3_dual_arm_sim:
#
#   * this repo's, editable-installed at src/a3_dual_arm_sim/ -- flat layout, no data/
#     subpackage, holds the expert, the collector and the bridge client;
#   * main's, at /tmp/a3-main/src/a3_dual_arm_sim/ -- the environment, the recording
#     primitives and the workflows.
#
# Python keys modules by name, so only one can be imported per process. tests/ needs this
# repo's package; scripts/recap/tests needs main's. Running them together fails with
#
#   ModuleNotFoundError: No module named 'a3_dual_arm_sim.data'
#
# which reads as a missing file rather than as a name collision. See
# scripts/recap/tests/conftest.py for why no single-process arrangement fixes it.
#
# The split below is by *which package each suite needs*, and it is not negotiable.
# tests/test_history_encoder_patches.py looks like a history test but lives in tests/ and
# imports the launcher through this repo's package, so it belongs in the second group.
set -uo pipefail

cd "$(dirname "$0")/../.."
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
export A3_MAIN_ROOT="${A3_MAIN_ROOT:-/tmp/a3-main}"
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp

status=0

echo "=== scripts/recap/tests (needs main's package) ==="
"$PY" -m pytest scripts/recap/tests -q || status=1

if [ "${1:-}" != "--all" ]; then
    echo
    echo "=== tests/ (needs this repo's package) ==="
    "$PY" -m pytest tests -q || status=1
fi

echo
if [ "$status" -eq 0 ]; then
    echo "all requested suites passed"
else
    echo "at least one suite failed" >&2
fi
exit "$status"
