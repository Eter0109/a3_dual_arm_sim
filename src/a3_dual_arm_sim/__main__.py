"""Unified module entry point, identical to the installed a3-sim command."""

from a3_dual_arm_sim.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
