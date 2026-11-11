#!/usr/bin/env python3
"""Where do the 61.6 ms of a control step go?

61.6 ms is wrong for MuJoCo.  This model is 121 bodies and 80 free bodies; `mj_step` should be a
couple of milliseconds.  If the cost is in Python bookkeeping rather than the solver, it can be
removed and the online loop's throughput goes up by the same factor -- which decides how many
episodes fit in an evening.

The profile is split into the pieces a control step actually contains:

    mj_step itself          | physics
    qpos / sensor reads     | cheap, just numpy views
    cookies_in_target       | a distance test per cookie, potentially a Python loop
    applied_action          | whatever normalisation the env applies
    reward / info assembly  | dict construction with tensors

`mj_step` is timed directly against the model so the comparison is unambiguous: if `mj_step` alone
is fast and `env.step` is slow, the time is in our code and we can take it back.

Usage
-----
    python scripts/recap/probe_step_cost.py [--steps 50]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def timed(label: str, fn, repeat: int) -> float:
    started = time.monotonic()
    for _ in range(repeat):
        fn()
    elapsed = (time.monotonic() - started) / repeat
    print(f"   {label:34s} {elapsed * 1000:8.2f} ms")
    return elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args(argv)

    import mujoco

    from bench_pi05_on_main import make_cookie_benchmark

    bench = make_cookie_benchmark(task=args.task, prompt_style="task", max_steps=1500)
    options = {
        "randomize_cookies": bench.randomize_cookies,
        "randomize_boxes": bench.randomize_boxes,
        "randomize_source_bin": bench.randomize_boxes,
        "randomize_target_bin": bench.randomize_boxes,
        "source_bin_noise_m": bench.source_bin_noise_m,
        "target_bin_noise_m": bench.target_bin_noise_m,
        "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
        "randomization_profile": bench.profile or "basic",
        "appearance_seed": args.seed,
        "randomization_settings": bench.randomization_settings,
    }

    env = bench.create_env(render_cameras=False)
    env.reset(seed=args.seed, options=options)
    action = np.zeros(16)

    print("=" * 70)
    print("model shape")
    print("=" * 70)
    model, data = env.model, env.data
    print(f"   nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom} nu={model.nu}")
    print(f"   solver={model.opt.solver} iterations={model.opt.iterations}"
          f" timestep={model.opt.timestep}")

    print()
    print("=" * 70)
    print("cost of each layer")
    print("=" * 70)

    timed("mj_step (physics only)", lambda: mujoco.mj_step(model, data), args.steps)
    timed("mj_forward (no integration)",
          lambda: mujoco.mj_forward(model, data), args.steps)

    def read_state():
        _ = np.asarray(env.data.qpos).copy()
        _ = np.asarray(env.data.qvel).copy()

    timed("qpos + qvel copy", read_state, args.steps)

    sub = getattr(env, "unwrapped", env)
    if hasattr(sub, "_count_cookies_in_target"):
        timed("_count_cookies_in_target", sub._count_cookies_in_target, args.steps)  # noqa: SLF001
    elif hasattr(sub, "cookies_in_target"):
        timed("cookies_in_target", sub.cookies_in_target, args.steps)

    timed("env.step (everything)", lambda: env.step(action), args.steps)

    print()
    print("=" * 70)
    print("cProfile of 30 env.step calls, cumulative, top 15")
    print("=" * 70)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(30):
        env.step(action)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    for line in stream.getvalue().splitlines():
        if line.strip() and ("{" in line or "ncalls" in line):
            print("   " + line.strip()[:110])

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
