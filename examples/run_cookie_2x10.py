"""Run the separate 2x10 expert, or export its XML with --write-model."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("a3_dual_arm_sim.workflows.run_cookie_2x10_cli", run_name="__main__")
