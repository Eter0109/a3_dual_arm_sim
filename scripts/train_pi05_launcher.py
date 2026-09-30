"""Launch lerobot's trainer on this machine, with the shims the π0.5 checkpoint needs.

Why a launcher rather than `python -m lerobot.scripts.lerobot_train`
-------------------------------------------------------------------
``lerobot/pi05_base`` was saved by an older lerobot than the one installed here, and the
difference is not cosmetic: `policy_preprocessor.json` lists a step called
``relative_actions_processor``, while this version registers the same class under other
names.  ``PolicyProcessorPipeline.from_pretrained`` resolves step names by a plain dict
lookup and raises ``ImportError: Failed to load processor step from registry`` when it
misses -- an error that reads like a missing dependency, not a version skew.

Three ways to fix it, and why this is the one
--------------------------------------------
* Edit ``policy_preprocessor.json`` -- rejected: that is modifying a downloaded checkpoint,
  and the next `snapshot_download` would silently undo it.
* Patch the installed lerobot -- rejected: a venv-local edit that leaves no trace in this
  repository, so the next person to build an environment would hit the same wall with
  nothing to read.
* Register the alias here, before handing over -- chosen.  It is three lines, it lives
  next to the reason for it, and it is idempotent.

``setdefault`` rather than assignment, so if a future lerobot registers the old name
itself this becomes a no-op instead of an error.

The alias is safe because the step is ``enabled: false`` in this checkpoint: the class is
instantiated and then does nothing, so the *name* is the only thing that had to match.

Usage
-----
The same arguments `lerobot-train` takes::

    python scripts/train_pi05_launcher.py --policy.path=lerobot/pi05_base ...
"""

from __future__ import annotations

import sys


def _register_legacy_step_names() -> None:
    """Teach this lerobot the step name the checkpoint was saved with."""

    from lerobot.processor import RelativeActionsProcessorStep
    from lerobot.processor.pipeline import ProcessorStepRegistry

    ProcessorStepRegistry._registry.setdefault(
        "relative_actions_processor", RelativeActionsProcessorStep
    )


def main() -> int:
    _register_legacy_step_names()

    # Imported after the shim so nothing that resolves processors can run first.
    from lerobot.scripts.lerobot_train import main as lerobot_main

    # Not `raise SystemExit(lerobot_main())` -- lerobot's main returns None and signals
    # failure by exception, so the exit code has to come from the absence of one.
    lerobot_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
