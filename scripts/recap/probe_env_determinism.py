#!/usr/bin/env python3
"""Separate the three ways an "environment reset" can fail to be reproducible.

The online loop needs to know which of these is true before it can trust anything:

1. **reset is not reproducible** -- `reset(seed=X)` twice gives different scenes.  Then every
   episode is a different trial even at the same seed, and no A/B comparison on the same seed set
   means anything.
2. **stepping is not reproducible** -- same start state, same actions, different trajectory.  Then
   MuJoCo itself is non-deterministic here (thread count, SIMD paths, or the solver's unordered
   reduction), and the branch dataset's `gate_b_p95 = 0.0` would be a coincidence.
3. **rendering is not reproducible** -- the physics matches but the pixels do not.  Then the policy
   sees a slightly different image every run, which is survivable for training but fatal for
   measuring anything small.

The tests below isolate them: state-only digests for 1 and 2, image-only digests for 3.

Usage
-----
    python scripts/recap/probe_env_determinism.py
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


def state_hash(observation: dict) -> str:
    payload = np.ascontiguousarray(observation["observation.state"]).tobytes()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def image_hash(observation: dict) -> str:
    payload = b"".join(np.ascontiguousarray(observation[key]).tobytes() for key in CAMERA_KEYS)
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def qpos_hash(env) -> str:
    """The full physics state, not just the 16 commanded joints.

    Cookies are free bodies with their own qpos; a reset that places them with an unseeded RNG
    leaves the arm identical and every Cookie different.  Hashing only `observation.state` would
    call that reproducible.
    """
    data = np.ascontiguousarray(env.data.qpos)
    return hashlib.blake2b(data.tobytes(), digest_size=8).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
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
    print("1. reset reproducibility -- same seed, same env, twice")
    print("=" * 70)
    env = bench.create_env(render_cameras=False)
    env.reset(seed=args.seed, options=options(args.seed))
    first_qpos = qpos_hash(env)
    first_state = state_hash(env._observation())
    env.reset(seed=args.seed, options=options(args.seed))
    second_qpos = qpos_hash(env)
    second_state = state_hash(env._observation())
    print(f"   qpos  {first_qpos} vs {second_qpos}   {'SAME' if first_qpos == second_qpos else 'DIFFER'}")
    print(f"   state {first_state} vs {second_state}  {'SAME' if first_state == second_state else 'DIFFER'}")

    print()
    print("=" * 70)
    print("2. reset reproducibility -- same seed, two different env instances")
    print("=" * 70)
    other = bench.create_env(render_cameras=False)
    other.reset(seed=args.seed, options=options(args.seed))
    other_qpos = qpos_hash(other)
    print(f"   qpos {first_qpos} vs {other_qpos}   {'SAME' if first_qpos == other_qpos else 'DIFFER'}")
    other.close()

    print()
    print("=" * 70)
    print("3. stepping reproducibility -- same start, same actions")
    print("=" * 70)
    rng = np.random.default_rng(0)
    actions = rng.normal(0, 0.05, (args.steps, 16))

    def trajectory(target) -> tuple[list[str], list[str]]:
        states, images = [], []
        for action in actions:
            observation, *_ = target.step(action)
            states.append(qpos_hash(target))
            images.append(image_hash(observation))
        return states, images

    env.reset(seed=args.seed, options=options(args.seed))
    states_a, images_a = trajectory(env)
    env.reset(seed=args.seed, options=options(args.seed))
    states_b, images_b = trajectory(env)
    print(f"   physics {'SAME' if states_a == states_b else 'DIFFER'}"
          f"   images {'SAME' if images_a == images_b else 'DIFFER'}")
    if states_a != states_b:
        for step, (a, b) in enumerate(zip(states_a, states_b)):
            if a != b:
                print(f"   physics diverges at step {step}")
                break
    if images_a != images_b:
        for step, (a, b) in enumerate(zip(images_a, images_b)):
            if a != b:
                print(f"   images diverge at step {step}")
                break
    env.close()

    print()
    print("=" * 70)
    print(f"4. concurrency -- {args.threads} scenes in threads, same start states")
    print("=" * 70)
    # Sequential baseline, per scene, with rendering on (that is the mode the loop uses).
    sequential = []
    for position in range(args.threads):
        e = bench.create_env(render_cameras=True)
        e.reset(seed=args.seed + position, options=options(args.seed + position))
        sequential.append(trajectory(e))
        e.close()

    envs = [bench.create_env(render_cameras=True) for _ in range(args.threads)]
    for position, e in enumerate(envs):
        e.reset(seed=args.seed + position, options=options(args.seed + position))
    threaded: dict[int, tuple[list[str], list[str]]] = {}

    def worker(position: int) -> None:
        threaded[position] = trajectory(envs[position])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for e in envs:
        e.close()

    physics_ok = all(threaded[i][0] == sequential[i][0] for i in range(args.threads))
    images_ok = all(threaded[i][1] == sequential[i][1] for i in range(args.threads))
    print(f"   physics threaded == sequential: {'YES' if physics_ok else 'NO'}")
    print(f"   images  threaded == sequential: {'YES' if images_ok else 'NO'}")
    if not images_ok:
        for i in range(args.threads):
            if threaded[i][1] != sequential[i][1]:
                for step, (a, b) in enumerate(zip(threaded[i][1], sequential[i][1])):
                    if a != b:
                        print(f"   scene {i}: images diverge at step {step}")
                        break
                break

    print()
    print("=" * 70)
    print("判读")
    print("=" * 70)
    if first_qpos == second_qpos and first_qpos == other_qpos:
        print("  reset 可复现：同种子得到同一场景（含 80 个饼干的自由体状态）。")
    else:
        print("  reset 不可复现。在线 RL 的每个 episode 都是不同场景，这本身对训练无害，")
        print("  但任何同种子的 A/B 比较都不成立 —— 必须靠多种子平均，不能配对。")
    if states_a == states_b:
        print("  物理步进确定。")
    else:
        print("  物理步进不确定 —— MuJoCo 在这里是非确定的。任何逐步归因都无效。")
    if images_ok and physics_ok:
        print(f"  {args.threads} 线程并发与串行逐位一致，可以在线并行了。")
    else:
        print("  并发改变结果，必须每场景一个进程（各自独立 EGL 上下文）或串行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
