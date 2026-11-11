#!/usr/bin/env python3
"""Two architectural questions the online loop cannot start without.

1. **Does a long-lived env drift across episodes?**
   `probe_env_determinism` found that `reset(seed=X)` on an *already used* environment produces a
   different layout than the first `reset(seed=X)` on a fresh one: the 16 joint values match, the
   free-body `qpos` does not.  The randomisation appears to be applied *relative to the current
   Cookie positions*, so resetting twice compounds it.  If so, a persistent env in an online loop
   gives episode 2 a scene that is not what seed 901 means, and every seed-based comparison drifts
   with episode count.

2. **Can rendering be kept out of the worker threads?**
   `probe_threaded_render` hit `EGL_BAD_ACCESS` from `eglMakeCurrent` inside a worker thread.  MuJoCo
   keeps one global EGL context, so two threads rendering at once collide.  Two ways out:
   (a) one process per scene, or (b) step physics in threads and render only on the main thread.
   Option (b) is far simpler, and it is viable only if the physics step without rendering is both
   cheap enough and genuinely thread-safe.

`create_env(render_cameras=False)` gives an observation with no images; the cameras can then be
rendered on demand.  This script measures whether that split works: physics throughput in threads
with rendering off, render cost on the main thread, and whether the two together reproduce the
single-threaded result.

Usage
-----
    python scripts/recap/probe_parallel_env.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CAMERAS = ("front", "left_wrist", "right_wrist")


def qpos_hash(env, n: int = 585) -> str:
    return hashlib.blake2b(np.ascontiguousarray(env.data.qpos[:n]).tobytes(),
                           digest_size=8).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args(argv)

    from bench_pi05_on_main import make_cookie_benchmark

    bench = make_cookie_benchmark(task=args.task, prompt_style="task", max_steps=1500)

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

    print("=" * 70)
    print("1. does a persistent env drift across resets?")
    print("=" * 70)
    env = bench.create_env(render_cameras=False)
    layouts = []
    for index in range(4):
        env.reset(seed=args.seed, options=options(args.seed))
        layouts.append(qpos_hash(env))
    fresh = []
    for index in range(4):
        other = bench.create_env(render_cameras=False)
        other.reset(seed=args.seed, options=options(args.seed))
        fresh.append(qpos_hash(other))
        other.close()
    env.close()
    print(f"   persistent env, four resets: {layouts}")
    print(f"   fresh env each time          : {fresh}")
    drift = len(set(layouts)) > 1
    print(f"   -> persistent env {'DRIFTS' if drift else 'is stable'}"
          f"; fresh env {'differs from it' if fresh != layouts else 'matches'}")

    print()
    print("=" * 70)
    print("2. physics step cost, rendering on vs off")
    print("=" * 70)
    for render in (False, True):
        e = bench.create_env(render_cameras=render)
        e.reset(seed=args.seed, options=options(args.seed))
        e.step(np.zeros(16))
        started = time.monotonic()
        for _ in range(args.steps):
            e.step(np.zeros(16))
        elapsed = (time.monotonic() - started) / args.steps
        print(f"   render={render!s:5s} {elapsed * 1000:6.1f} ms/step")
        e.close()

    print()
    print("=" * 70)
    print(f"3. {args.envs} scenes, physics in threads, rendering OFF")
    print("=" * 70)
    envs = [bench.create_env(render_cameras=False) for _ in range(args.envs)]
    for position, e in enumerate(envs):
        e.reset(seed=args.seed + position, options=options(args.seed + position))
    actions = np.random.default_rng(0).normal(0, 0.05, (args.steps, 16))

    def play(e) -> list[str]:
        out = []
        for action in actions:
            e.step(action)
            out.append(qpos_hash(e))
        return out

    started = time.monotonic()
    sequential = [play(e) for e in envs]
    seq_time = time.monotonic() - started
    for position, e in enumerate(envs):
        e.reset(seed=args.seed + position, options=options(args.seed + position))

    threaded: dict[int, list[str]] = {}
    errors: list[str] = []

    def worker(position: int) -> None:
        try:
            threaded[position] = play(envs[position])
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.envs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    thr_time = time.monotonic() - started
    for e in envs:
        e.close()

    print(f"   sequential {seq_time:.2f}s ({args.envs * args.steps / seq_time:.0f} step/s)")
    print(f"   threaded   {thr_time:.2f}s ({args.envs * args.steps / thr_time:.0f} step/s)"
          f"   speedup {seq_time / thr_time:.2f}x")
    if errors:
        print(f"   thread errors: {errors[:2]}")
    same = len(threaded) == args.envs and all(threaded[i] == sequential[i] for i in range(args.envs))
    print(f"   threaded physics == sequential: {'YES' if same else 'NO'}")

    print()
    print("=" * 70)
    print("4. render cost on the main thread, on demand")
    print("=" * 70)
    e = bench.create_env(render_cameras=False)
    e.reset(seed=args.seed, options=options(args.seed))
    e.step(np.zeros(16))
    renderer = getattr(e, "_renderer", None)
    print(f"   env has _renderer: {renderer is not None}")
    if renderer is not None:
        started = time.monotonic()
        for _ in range(8):
            for camera in CAMERAS:
                _ = np.asarray(e._render_camera(camera))  # noqa: SLF001
        per_call = (time.monotonic() - started) / (8 * len(CAMERAS))
        print(f"   one camera render {per_call * 1000:.1f} ms -> "
              f"{3 * per_call * 1000:.1f} ms for three")
    e.close()

    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    if drift:
        print("  env 持久化会漂移：每个 episode 必须新建 env（1.36 s/次，占 episode 的 ~1.3%）。")
    else:
        print("  env 可持久化，每 episode 只 reset。")
    if same and not errors:
        print("  无渲染的物理步进可在线程里并行，且逐位一致 —— 用「并行物理 + 主线程渲染」。")
    else:
        print("  线程化物理不可用，需要每场景一个进程。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
