"""Where does a collection episode's wall time actually go?

Answers one question: is the ~117 s per episode spent on something several
processes can each do at once, or on something they would have to share?

Two candidates, and they parallelise very differently:

``physics``
    ``mj_step`` calls.  Single-threaded inside a process, but N processes each
    step their own environment with no shared state, so this scales.

``render``
    Camera reads.  These go through EGL on the one GPU, so N processes contend
    for the same device -- some speedup, but not linear.

The separation comes from building two environments that differ in exactly one
argument, ``render_cameras``, and differencing them.  Everything else -- scene,
seed, policy, step count -- is held fixed, so the subtraction means something in
a way that two independent whole-episode timings would not.

    MUJOCO_GL=egl /root/autodl-tmp/a3-collect/.venv/bin/python scripts/profile_collection.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Control steps to time.  Smoke episodes ran about 427, so a couple hundred is
#: representative of the mix without waiting for whole episodes.  Overridable so
#: the same script can drive a parallel-scaling run at a smaller step count.
STEPS = int(os.environ.get("PROFILE_STEPS", "200"))
REPEATS = int(os.environ.get("PROFILE_REPEATS", "2"))
SEED = int(os.environ.get("PROFILE_SEED", "4242"))
TASK = "transfer 10 cookies into target box"


def time_steps(benchmark, *, render_cameras: bool, steps: int = STEPS) -> tuple[float, float]:
    """Return (wall for ``steps`` control steps, wall for one reset)."""
    from a3_dual_arm_sim.workflows.policy_adapters import EpisodeContext, make_policy_adapter

    env = benchmark.create_env(render_cameras=render_cameras)
    runner = make_policy_adapter("same_column")

    reset_start = time.perf_counter()
    observation, _ = env.reset(
        seed=SEED,
        options={
            "randomize_cookies": True,
            "randomize_boxes": True,
            "randomize_source_bin": True,
            "randomize_target_bin": True,
            "appearance_seed": SEED,
        },
    )
    reset_wall = time.perf_counter() - reset_start

    if hasattr(runner, "bind_env"):
        runner.bind_env(env)
    if hasattr(runner, "reset"):
        try:
            runner.reset(EpisodeContext(seed=SEED, task=TASK, action_mode=env.action_mode))
        except TypeError:
            runner.reset(None)

    start = time.perf_counter()
    for _ in range(steps):
        action = runner.act(observation, TASK)
        observation, _, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    wall = time.perf_counter() - start
    env.close()
    return wall, reset_wall


def main() -> int:
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark

    benchmark = CookieBatchBenchmark(max_steps=1000)

    print(f"{STEPS} control steps, {REPEATS} repeats, seed {SEED}\n")

    timings: dict[str, list[float]] = {"off": [], "on": []}
    for label, render in (("off", False), ("on ", True)):
        for repeat in range(REPEATS):
            wall, reset_wall = time_steps(benchmark, render_cameras=render)
            timings[label.strip()].append(wall)
            print(f"  render {label}  repeat {repeat}: {wall:6.2f} s  "
                  f"({wall / STEPS * 1000:6.1f} ms/step)   reset {reset_wall:.2f} s")
        print()

    off = sum(timings["off"]) / len(timings["off"])
    on = sum(timings["on"]) / len(timings["on"])
    print(f"without cameras : {off:6.2f} s  ({off / STEPS * 1000:6.1f} ms/step)")
    print(f"with cameras    : {on:6.2f} s  ({on / STEPS * 1000:6.1f} ms/step)")
    print(f"camera cost     : {on - off:6.2f} s  "
          f"({(on - off) / on * 100:.0f}% of the camera-on total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
