"""Collect the separate 2x10 scene using --first-grasp 0..9 or random."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("a3_dual_arm_sim.data.collect_cookie_2x10_cli", run_name="__main__")
