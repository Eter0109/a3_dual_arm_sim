#!/usr/bin/env python3
"""Is MuJoCo stepping thread-safe?  A clean test, because the first one was confounded.

The confound
------------
`probe_parallel_env` reported "threaded physics != sequential" and it would have been wrong to
believe it: the sequential baseline ran on each environment's **first** reset, the threaded run on
its **second**, and `probe_parallel_env` itself established that a persistent environment returns a
different layout on reset #2 than on reset #1.  So the two runs started from different scenes and
the comparison measured the reset, not the threading.

The rule this file follows, and the reason it exists: **a fresh environment for every measurement
path.**  Fresh env, one reset, play, close.  Then the only difference between the sequential and
threaded runs is that the threaded one shares a process.

What is measured
----------------
* `physics`  -- rendering off, so no EGL context is involved.  Tests MuJoCo's own thread safety.
* `render`   -- rendering on.  Tests whether MuJoCo's single global EGL context tolerates threads,
  which `probe_threaded_render` suggested it does not (`EGL_BAD_ACCESS` from `eglMakeCurrent`).

Both are compared per scene against a same-seed sequential run of a *fresh* environment.

Usage
-----
    python scripts/recap/probe_env_threadsafe.py [--envs 4] [--steps 20]
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

CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def digest(env, step: int) -> str:
    """qpos plus, when present, the three camera frames."""
    payload = np.ascontiguousarray(env.data.qpos).tobytes()
    if hasattr(env, "_renderer") and env._renderer is not None:  # noqa: SLF001
        for key in CAMERA_KEYS:
            payload += np.ascontiguousarray(env._render_camera(key.split(".")[-1])).tobytes()  # noqa: SLF001
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--steps", type=int, default=20)
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

    actions = np.random.default_rng(0).normal(0, 0.05, (args.steps, 16))

    def make(seed: int, render: bool):
        """A fresh environment, exactly one reset.  Never reused."""
        env = bench.create_env(render_cameras=render)
        env.reset(seed=seed, options=options(seed))
        return env

    def play(env) -> list[str]:
        out = []
        for action in actions:
            env.step(action)
            out.append(digest(env, 0))
        return out

    for label, render in (("physics (render off)", False), ("render (three cameras)", True)):
        print("=" * 70)
        print(label)
        print("=" * 70)
        seeds = [args.seed + i for i in range(args.envs)]

        started = time.monotonic()
        sequential = [play(make(seed, render)) for seed in seeds]
        seq_time = time.monotonic() - started

        envs = [make(seed, render) for seed in seeds]
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
        for env in envs:
            env.close()

        total = args.envs * args.steps
        print(f"   sequential {seq_time:6.2f}s  {total / seq_time:6.1f} step/s")
        print(f"   threaded   {thr_time:6.2f}s  {total / thr_time:6.1f} step/s"
              f"   speedup {seq_time / thr_time:.2f}x")
        if errors:
            print(f"   thread errors ({len(errors)}): {errors[0][:100]}")
        complete = len(threaded) == args.envs
        same = complete and all(threaded[i] == sequential[i] for i in range(args.envs))
        print(f"   identical to sequential: {'YES' if same else 'NO'}"
              f"   (completed {len(threaded)}/{args.envs})")
        if complete and not same:
            for i in range(args.envs):
                if threaded[i] != sequential[i]:
                    for step, (a, b) in enumerate(zip(threaded[i], sequential[i])):
                        if a != b:
                            print(f"   scene {i} diverges at step {step}")
                            break
                    break
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
