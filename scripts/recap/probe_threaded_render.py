#!/usr/bin/env python3
"""Is threaded rendering deterministic?  Gate for the online loop.

Why this is a gate and not a footnote
-------------------------------------
MuJoCo releases the GIL, so four scenes stepping in four threads really do run at the same time
(measured 3.45x over sequential).  That is what makes parallel environments affordable here.  But
`A3CookieTransferEnv._observation` calls `self._render_camera(...)` three times per step, and EGL
contexts are process-global.  If two threads share one context, the images can interleave or come
back black -- and a black frame is not an error, it is an image the policy will happily encode.

The only acceptable outcome is: thread A's trajectory is bit-identical to thread A's trajectory run
alone.  Anything else and every observation the actor sees is suspect, which would be invisible in
the training curve and would look exactly like "online RL does not help".

What is compared
----------------
Three runs of the same scene, same seed, same action sequence:

* `sequential`  -- one scene, stepped in the main thread;
* `threaded`    -- four scenes stepped concurrently, this one among them;
* `repeat`      -- the first scene again, alone, after the concurrent run.

`threaded` must equal `repeat`.  If it does, concurrency is transparent.  Camera images are hashed,
not compared visually, because "looks the same" is how a 2-pixel offset survives.

Usage
-----
    python scripts/recap/probe_threaded_render.py [--threads 4] [--steps 6]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
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


def digest(env, steps: int, actions: np.ndarray | None = None) -> list[str]:
    """Hash of all three cameras plus the joint state, per step."""
    out = []
    for step in range(steps):
        action = np.zeros(16) if actions is None else actions[step]
        observation, *_ = env.step(action)
        payload = b"".join(np.ascontiguousarray(observation[key]).tobytes() for key in CAMERA_KEYS)
        payload += np.ascontiguousarray(observation["observation.state"]).tobytes()
        out.append(hashlib.blake2b(payload, digest_size=8).hexdigest())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    args = parser.parse_args(argv)

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
    # A fixed action sequence with real motion: zeros keep the arm still, and a still arm renders
    # the same frame every step, so image corruption would not show up.  Joint 0 and 8 sweep.
    rng = np.random.default_rng(0)
    actions = np.zeros((args.steps, 16))
    actions[:, 0] = 0.05 * np.sin(np.arange(args.steps))
    actions[:, 8] = 0.05 * np.cos(np.arange(args.steps))
    actions += rng.normal(0, 0.001, actions.shape)

    print(f"[render] {args.threads} scenes, {args.steps} steps, task {args.task}", flush=True)

    # --- 1. the reference trajectory, alone --------------------------------------------
    env = bench.create_env(render_cameras=True)
    env.reset(seed=args.seed, options=options)
    sequential = digest(env, args.steps, actions)
    env.close()
    print(f"[render] sequential  {sequential[0]} .. {sequential[-1]}", flush=True)

    # --- 2. the same scene, concurrently with others -----------------------------------
    envs = [bench.create_env(render_cameras=True) for _ in range(args.threads)]
    for position, e in enumerate(envs):
        e.reset(seed=args.seed + position, options=options)
    results: dict[int, list[str]] = {}

    def worker(position: int) -> None:
        results[position] = digest(envs[position], args.steps, actions)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for e in envs:
        e.close()
    threaded = results[0]
    print(f"[render] threaded    {threaded[0]} .. {threaded[-1]}", flush=True)

    # --- 3. the same scene again, alone ------------------------------------------------
    env = bench.create_env(render_cameras=True)
    env.reset(seed=args.seed, options=options)
    repeat = digest(env, args.steps, actions)
    env.close()
    print(f"[render] repeat      {repeat[0]} .. {repeat[-1]}", flush=True)

    print()
    same_threaded = threaded == repeat
    same_sequential = sequential == repeat
    print(f"  threaded == repeat      : {'YES' if same_threaded else 'NO'}")
    print(f"  sequential == repeat    : {'YES' if same_sequential else 'NO'}")
    print()
    if same_threaded and same_sequential:
        print("  判读：并发渲染是透明的。在线循环可以开 N 个场景。")
        verdict = 0
    elif same_sequential and not same_threaded:
        print("  判读：单线程可复现，但并发会改变图像 —— 并发不可用。")
        print("        要么每场景一个进程（各自独立的 EGL 上下文），要么串行。")
        verdict = 1
    else:
        print("  判读：连单线程都不可复现，说明 reset 本身带了未受控的状态。")
        print("        在线循环的每一步都无法归因，先修这个。")
        verdict = 2

    # Where the trajectories first diverge, when they do.
    if not same_threaded:
        for step, (a, b) in enumerate(zip(threaded, repeat)):
            if a != b:
                print(f"  first divergence at step {step}: threaded {a} vs repeat {b}")
                break
    return verdict


if __name__ == "__main__":
    sys.exit(main())
