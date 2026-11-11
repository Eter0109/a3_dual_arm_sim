#!/usr/bin/env python3
"""Why does the physics cost grow over an episode?

`train_dsrl_online.py` reports a per-round split.  The first round cost 1,164 ms of physics for
eight scenes -- almost exactly one scene's worth of 16 control steps, i.e. perfect parallelism --
and by round 8 the same 16 steps cost 5,748 ms.  Render, preprocess, prefix and sampling stayed
flat to the millisecond, so the growth is entirely inside the environment.

Two candidate causes, and they need opposite responses:

* **The simulation genuinely gets more expensive.**  `solver=2` (Newton) with `iterations=100`, and
  eighty free cookie bodies.  As the arm digs into the bin the active-constraint count rises and
  the solve cost rises with it.  If so nothing here is a bug; the loop's throughput simply has to
  be budgeted from a mid-episode measurement rather than a first-step one, and `--scenes` has to be
  chosen from the slow end.
* **Parallelism collapses.**  Eight stepping threads plus torch's OpenMP pool on sixteen cores can
  oversubscribe, and the GIL is still held during the Python bookkeeping around each `mj_step`.

The test separates them by time alone: step **one** scene through a whole episode and print the
cost every 100 steps.  Single-threaded, contention is impossible, so any growth is the simulation.
Then step eight scenes and compare against 8x the single-scene cost.

Usage
-----
    python scripts/recap/probe_physics_growth.py [--steps 1500] [--scenes 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")

import numpy as np

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STATE = "observation.state"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--bucket", type=int, default=100)
    args = parser.parse_args(argv)

    import mujoco

    from bench_pi05_on_main import make_cookie_benchmark

    bench = make_cookie_benchmark(task=args.task, prompt_style="short", max_steps=args.steps)

    def options(seed: int) -> dict:
        return {
            "randomize_cookies": bench.randomize_cookies,
            "randomize_boxes": bench.randomize_boxes,
            "randomize_source_bin": bench.randomize_boxes,
            "randomize_target_bin": bench.randomize_boxes,
            "source_bin_noise_m": bench.source_bin_noise_m,
            "target_bin_noise_m": bench.target_bin_noise_m,
            "target_bin_yaw_noise_rad": bench.target_bin_yaw_noise_rad,
            "randomization_profile": bench.profile or "basic",
            "appearance_seed": seed,
            "randomization_settings": bench.randomization_settings,
        }

    # =========================================================================================
    print("=" * 74)
    print(f"1. one scene, {args.steps} control steps, physics only")
    print("=" * 74)
    env = bench.create_env(render_cameras=False)
    env.reset(seed=args.seed, options=options(args.seed))
    rng = np.random.default_rng(0)
    # A slow, policy-shaped action: holding the start pose would never create the contacts the
    # real loop creates, and the question is precisely whether contacts are what costs.
    start = np.asarray(env._current_joint_action(), dtype=np.float64)  # noqa: SLF001
    sweep = rng.normal(0.0, 0.004, (args.steps, 16))
    print(f"   {'step':>6s} {'ms/step':>8s} {'nefc':>7s} {'ncon':>5s} {'nactive':>8s}")
    bucket_started = time.monotonic()
    bucket_steps = 0
    for step in range(args.steps):
        action = start + np.cumsum(sweep[max(0, step - 40):step + 1], axis=0).sum(axis=0) * 0.02
        env.step(np.clip(action, -3.0, 3.0))
        bucket_steps += 1
        if (step + 1) % args.bucket == 0:
            per = (time.monotonic() - bucket_started) / bucket_steps * 1000
            print(f"   {step + 1:6d} {per:8.2f} {int(env.data.nefc):7d} {int(env.data.ncon):5d}"
                  f" {int(env.data.ncon and env.data.nefc):8d}")
            bucket_started = time.monotonic()
            bucket_steps = 0

    # `nefc` is the number of scalar constraint functions the solver iterates over, so it is the
    # quantity that would explain a linear cost increase.  Compare it against the same measurement
    # on a scene that is just standing still.
    print()
    print("   reference: a fresh scene that only holds position")
    still = bench.create_env(render_cameras=False)
    still.reset(seed=args.seed, options=options(args.seed))
    pose = np.asarray(still._current_joint_action(), dtype=np.float64)  # noqa: SLF001
    still.step(pose)
    started = time.monotonic()
    for _ in range(60):
        still.step(pose)
    print(f"   hold-pose  {(time.monotonic() - started) / 60 * 1000:6.2f} ms/step"
          f"   nefc {int(still.data.nefc)}  ncon {int(still.data.ncon)}")
    still.close()
    print(f"   after the episode: nefc {int(env.data.nefc)}  ncon {int(env.data.ncon)}")
    env.close()

    # =========================================================================================
    print()
    print("=" * 74)
    print(f"2. {args.scenes} scenes in threads, {args.steps // 4} steps each")
    print("=" * 74)
    envs = []
    for position in range(args.scenes):
        e = bench.create_env(render_cameras=False)
        e.reset(seed=args.seed + position, options=options(args.seed + position))
        envs.append(e)
    poses = [np.asarray(e._current_joint_action(), dtype=np.float64) for e in envs]  # noqa: SLF001
    steps_each = args.steps // 4

    def play(position: int) -> list[float]:
        out = []
        e = envs[position]
        pose = poses[position]
        for step in range(steps_each):
            e.step(pose + sweep[step % len(sweep)])
            if (step + 1) % args.bucket == 0:
                out.append(float(e.data.nefc))
        return out

    started = time.monotonic()
    sequential_nefc = [play(position) for position in range(args.scenes)]
    seq = time.monotonic() - started
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.scenes) as pool:
        threaded_nefc = list(pool.map(play, range(args.scenes)))
    thr = time.monotonic() - started
    for e in envs:
        e.close()
    total = args.scenes * steps_each
    print(f"   sequential {seq:6.2f}s  {total / seq:6.1f} step/s")
    print(f"   threaded   {thr:6.2f}s  {total / thr:6.1f} step/s   speedup {seq / thr:.2f}x")
    print(f"   nefc over the run, first scene: {sequential_nefc[0][:6]}")
    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    print("  如果单场景的 ms/step 随 nefc 一起上升，那是求解器在更多接触上迭代，不是 bug；")
    print("  吞吐预算要用 episode 中段的数字，而不是第一步的数字。")
    print("  如果单场景平稳而 8 线程的相对加速比明显下降，那是线程超额订阅，需要降 --scenes 或限 OMP 线程。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
