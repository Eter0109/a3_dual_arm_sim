"""Point `a3_dual_arm_sim` at main's tree for this directory's tests.

The collision
-------------
Two different packages share the name `a3_dual_arm_sim`:

* this repo's, editable-installed at `src/a3_dual_arm_sim/` -- flat layout, no `data/`
  subpackage, holds the expert, the collector and the bridge client;
* main's, at `/tmp/a3-main/src/a3_dual_arm_sim/` -- the environment, the recording
  primitives and the workflows.

Python keys modules by name, so **only one can be imported per process**.  The collector
needs main's (`a3_dual_arm_sim.data.recording`); `tests/` needs this repo's.  Combined in one
`pytest` process the result is

    ModuleNotFoundError: No module named 'a3_dual_arm_sim.data'

which reads as a missing file rather than as a name collision.

What this file does
-------------------
Puts main's `src` at the front of `sys.path`, exactly as `bench_pi05_on_main.py` does.  It is
safe because **a conftest only applies to its own directory**, so running `pytest tests` never
loads this file and `tests/` keeps using this repo's package undisturbed.

It does *not* purge `sys.modules`.  An earlier version did, and it broke all 29 tests in
`tests/`: pytest loads conftest files before importing test modules, so the purge ran before
`tests/` imported the package it needs.  Inserting a path is harmless; deleting cached modules
is not.

An even earlier version only *checked* that `a3_dual_arm_sim.data` was resolvable and refused
to start otherwise.  That was wrong in the opposite direction: the path insert that makes it
resolvable happens at import time in the test module, which is after conftest, so the check
failed on every standalone run.

The guard below catches the one case that cannot be repaired -- a combined invocation where
this repo's package is already imported -- and names the fix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MAIN_ROOT = os.environ.get("A3_MAIN_ROOT", "/tmp/a3-main")
MAIN_SRC = os.environ.get("A3_MAIN_SRC", os.path.join(MAIN_ROOT, "src"))


def _repo_package_loaded() -> bool:
    """True if `a3_dual_arm_sim` is imported and is not main's.

    Distinguishing by origin path rather than by presence, because main's package may
    legitimately already be loaded -- the collector tests import it themselves.
    """

    module = sys.modules.get("a3_dual_arm_sim")
    if module is None:
        return False
    origin = getattr(module, "__file__", "") or ""
    return Path(MAIN_SRC).as_posix() not in Path(origin).as_posix()


if not Path(MAIN_SRC).is_dir():
    raise RuntimeError(
        f"main's source tree is not at {MAIN_SRC}.  The collector imports the environment "
        "and the recording primitives from it; set A3_MAIN_ROOT to a checkout."
    )

if _repo_package_loaded():
    raise RuntimeError(
        "this process already imported this repo's `a3_dual_arm_sim`, which has no `data` "
        "subpackage, so the collector tests cannot run in it.  Two packages share that name "
        "(this repo's and main's) and only one can be imported per process.\n"
        "  Run the suites separately:\n"
        "    python -m pytest tests -q\n"
        "    A3_MAIN_ROOT=/tmp/a3-main python -m pytest scripts/recap/tests -q\n"
        "  or use scripts/recap/run_tests.sh, which does both."
    )

# Front of the path, so `import a3_dual_arm_sim` resolves to main's package rather than to
# the editable install that this repo's `pyproject.toml` put there.
while MAIN_SRC in sys.path:
    sys.path.remove(MAIN_SRC)
sys.path.insert(0, MAIN_SRC)
