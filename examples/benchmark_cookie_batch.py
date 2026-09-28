#!/usr/bin/env python3
"""Compatibility entrypoint; implementation: a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli", run_name="__main__")
else:
    import sys
    from importlib import import_module

    _implementation = import_module("a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli")
    globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith("__")})
    sys.modules[__name__] = _implementation
