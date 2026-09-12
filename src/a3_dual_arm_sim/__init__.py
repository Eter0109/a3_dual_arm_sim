"""A3 dual-arm MuJoCo simulation package.

The simulation modules import MuJoCo, whose wheels use CPU instructions that are
absent on some GPU-only workers (a virtualised node may expose no AVX at all,
turning the import into an ``Illegal instruction`` abort). Training, recording
and dataset inspection need none of that, so the public names below are resolved
on first attribute access rather than at import time. ``import
a3_dual_arm_sim.training`` therefore works on a host where ``import mujoco``
crashes, while ``from a3_dual_arm_sim import A3DualArmEnv`` still behaves as
before for callers that do want the simulator.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> module that defines it, relative to this package.
_EXPORTS: dict[str, str] = {
    "A3CookieTransferEnv": "cookie_transfer",
    "A3CookieTransferExpert": "expert",
    "A3DualArmEnv": "env",
    "A3GraspEnv": "grasp",
    "A3GraspExpert": "expert",
    "CookiePhase": "expert",
    "CookieTransferEpisodeResult": "evaluation",
    "evaluate_cookie_transfer": "evaluation",
    "run_cookie_transfer_episode": "evaluation",
}

# Kept literal so the exported surface stays readable; a test asserts that it
# matches _EXPORTS.
__all__ = [
    "A3CookieTransferEnv",
    "A3CookieTransferExpert",
    "A3DualArmEnv",
    "A3GraspEnv",
    "A3GraspExpert",
    "CookiePhase",
    "CookieTransferEpisodeResult",
    "evaluate_cookie_transfer",
    "run_cookie_transfer_episode",
]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Resolve a public name on first use (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    # Cache the resolved object so repeated access skips this hook.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
