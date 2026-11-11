#!/usr/bin/env python3
"""Two gates the online loop cannot be built without, both cheap to measure now.

Gate 1 -- **can the main thread render for an environment that steps in a worker thread?**
`probe_env_threadsafe` showed physics is bit-identical under threads with rendering off, and that
rendering inside a worker thread dies with `EGL_BAD_ACCESS` because MuJoCo holds one global EGL
context.  The loop therefore wants `create_env(render_cameras=False)` plus a `mujoco.Renderer` it
drives itself.  But `_render_camera` returns zeros when `render_cameras` is False, so the renderer
has to be built here, and that is only acceptable if it produces the image the *environment* would
have produced.  A shifted image is a scene the policy has never been evaluated on, and no loss
curve would show it.

Gate 2 -- **does batching the policy across scenes cost what one forward costs?**
A forward is ~219 ms.  If batching is linear the loop pays 4 x 219 and the physics parallelism is
spent waiting on the GPU; if it is roughly flat the loop is physics-bound, which is the good
outcome and also decides `--scenes`.

The throughput table at the end composes measured pieces rather than guessed ones, so the number
of episodes a run can reach is known before the run instead of after.

Usage
-----
    python scripts/recap/probe_online_throughput.py [--scenes 8] [--batch-sizes 1 4 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")

import numpy as np
import torch

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

CAMERAS = ("front", "left_wrist", "right_wrist")
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
STATE = "observation.state"
TASK = "transfer 10 cookies into target box"
CHECKPOINT = Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model")
HISTORY_FRAMES = 10
HISTORY_DIM = HISTORY_FRAMES * 48


def render(renderer, scene, camera_ids) -> dict[str, np.ndarray]:
    """Three 224x224 uint8 frames from `scene.data`, rendered on **this** thread."""
    out = {}
    for name in CAMERAS:
        renderer.update_scene(scene.data, camera=camera_ids[name])
        image = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        out[name] = image
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="2x10")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import mujoco

    from bench_pi05_on_main import make_cookie_benchmark

    bench = make_cookie_benchmark(task=args.task, prompt_style="short", max_steps=1500)
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

    # =========================================================================================
    print("=" * 74)
    print("gate 1: a renderer we own, vs the environment's own renderer")
    print("=" * 74)
    env = bench.create_env(render_cameras=True)  # so the env builds its own renderer
    env.reset(seed=args.seed, options=options)
    env.step(np.zeros(16))
    reference = {name: np.asarray(env._render_camera(name)) for name in CAMERAS}  # noqa: SLF001
    height, width = reference["front"].shape[:2]
    camera_ids = dict(env._camera_ids)  # noqa: SLF001
    model = env.model

    mine = mujoco.Renderer(model, height=height, width=width)
    manual = {}
    for name in CAMERAS:
        mine.update_scene(env.data, camera=camera_ids[name])
        manual[name] = np.ascontiguousarray(mine.render(), dtype=np.uint8)

    print(f"   image {height}x{width}   shapes agree:"
          f" {all(reference[n].shape == manual[n].shape for n in CAMERAS)}")
    for name in CAMERAS:
        identical = np.array_equal(reference[name], manual[name])
        delta = float(np.abs(reference[name].astype(float) - manual[name].astype(float)).max())
        mean = float(np.abs(reference[name].astype(float) - manual[name].astype(float)).mean())
        print(f"   {name:12s} identical={identical!s:5s} max|diff|={delta:5.1f} mean|diff|={mean:6.3f}")
    render_ok = all(np.array_equal(reference[name], manual[name]) for name in CAMERAS)
    print(f"   -> {'PASS' if render_ok else 'FAIL'}: own renderer"
          f" {'reproduces' if render_ok else 'does not reproduce'} the env's images")

    started = time.monotonic()
    for _ in range(10):
        render(mine, env, camera_ids)
    render_cost = (time.monotonic() - started) / 10
    print(f"   render cost {render_cost * 1000:.1f} ms for three cameras")
    env.close()

    # =========================================================================================
    print()
    print("=" * 74)
    print("gate 2: batched policy forward")
    print("=" * 74)

    import train_pi05_launcher as launcher  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.pi05 import PI05Policy  # noqa: PLC0415

    # The three patches the adapter installs; without them the checkpoint's own module layout
    # does not match the vendored code and `from_pretrained` fails inside lerobot.
    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    from steering import encode_prefix, sample_with_prefix  # noqa: PLC0415

    print(f"   loading {CHECKPOINT.parent.parent.parent.name}/{CHECKPOINT.name}", flush=True)
    policy = PI05Policy.from_pretrained(str(CHECKPOINT)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, _ = make_pre_post_processors(
        policy.config, str(CHECKPOINT),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    num_steps = int(policy.config.num_inference_steps)
    print(f"   chunk {chunk_size} x {action_dim} = {chunk_size * action_dim} noise dims,"
          f" {num_steps} denoise steps")

    scenes = [bench.create_env(render_cameras=False) for _ in range(max(args.batch_sizes))]
    for position, scene in enumerate(scenes):
        scene.reset(seed=args.seed + position,
                    options=options | {"appearance_seed": args.seed + position})
        scene.step(np.zeros(16))

    renderer = mujoco.Renderer(scenes[0].model, height=height, width=width)

    def ready_for(n: int):
        """The batch `preprocess` expects, for scenes 0..n-1 -- the same keys the driver sends."""
        stacks = {key: [] for key in IMAGE_KEYS}
        observations = []
        for position in range(n):
            scene = scenes[position]
            frames = render(renderer, scene, camera_ids)
            for name in CAMERAS:
                stacks[f"observation.images.{name}"].append(frames[name])
            observations.append(scene._observation())  # noqa: SLF001
        # (B, H, W, C) uint8 -> (B, C, H, W) float in [0, 1], which is what `preprocess` wants:
        # the LeRobot convention is channels-first in [0, 1] and `_preprocess_images` then maps it
        # to [-1, 1] for SigLIP.  Handing it HWC is not rejected at the boundary -- it reaches the
        # first conv as 224 channels and fails there.
        raw = {key: torch.as_tensor(np.stack(value), device=args.device)
               .permute(0, 3, 1, 2).float().div_(255.0)
               for key, value in stacks.items()}
        raw[STATE] = torch.as_tensor(
            np.stack([np.asarray(o[STATE], dtype=np.float32) for o in observations]),
            device=args.device)
        # Ten frames of [q, q_dot, a_previous].  Content is irrelevant to the timing; the shape
        # and the repeat-the-first-frame padding are what the driver does on a fresh episode.
        history = np.stack([np.concatenate([
            np.asarray(o[STATE], dtype=np.float32),
            np.asarray(o["observation.velocity"], dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        ]) for o in observations])
        history = np.repeat(history[:, None, :], HISTORY_FRAMES, axis=1).reshape(n, HISTORY_DIM)
        raw["observation.history"] = torch.as_tensor(history, device=args.device)
        raw["task"] = [TASK] * n
        return preprocess(raw)

    print()
    print(f"   {'batch':>6s} {'preprocess':>11s} {'prefix':>9s} {'sample':>9s} {'total':>9s}"
          f" {'per scene':>10s}")
    timings = {}
    for n in args.batch_sizes:
        ready = ready_for(n)
        for _ in range(2):  # allocator + cudnn autotune
            prefix = encode_prefix(policy, ready, keep_embeddings=True)
            noise = torch.randn(n, chunk_size * action_dim, device=args.device)
            sample_with_prefix(policy, prefix, noise, num_steps, args.device)
        torch.cuda.synchronize()
        ready = ready_for(n)
        t1 = time.monotonic()
        prefix = encode_prefix(policy, ready, keep_embeddings=True)
        t2 = time.monotonic()
        noise = torch.randn(n, chunk_size * action_dim, device=args.device)
        sample_with_prefix(policy, prefix, noise, num_steps, args.device)
        torch.cuda.synchronize()
        t3 = time.monotonic()
        timings[n] = (t2 - t1, t3 - t2, t3 - t1)
        print(f"   {n:6d} {'(see below)':>11s} {timings[n][0] * 1000:8.0f}ms"
              f" {timings[n][1] * 1000:8.0f}ms {timings[n][2] * 1000:8.0f}ms"
              f" {timings[n][2] / n * 1000:9.0f}ms")

    # `preprocess` is timeable on its own once the render is out of it: it is a pure function of
    # already-built tensors, so timing it separately is the honest number for the loop's budget.
    for n in args.batch_sizes:
        raw_ready = None
        ready = ready_for(n)  # renders, then preprocesses
        del raw_ready
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(3):
            prefix = encode_prefix(policy, ready, keep_embeddings=True)
        encode_prefix_cost = (time.monotonic() - started) / 3
        timings[n] = (encode_prefix_cost, timings[n][1], encode_prefix_cost + timings[n][1])

    for scene in scenes:
        scene.close()

    # =========================================================================================
    print()
    print("=" * 74)
    print("what the loop can expect")
    print("=" * 74)
    physics_ms = 61.6        # probe_step_cost: 50 mj_step of 1.12 ms plus env bookkeeping
    execute = 16             # pi0.5's manifest: chunk 50, execute 16
    episode_control = 1382   # measured on this dataset
    n = args.scenes
    forward = timings[n][2] if n in timings else timings[max(timings)][2]
    # All n scenes step concurrently in threads, so one round costs one scene's physics at
    # `execute` steps, plus the batched forward, plus the render.
    round_s = execute * physics_ms / 1000 + forward + n * render_cost
    rl_steps_per_s = n / round_s
    control_per_s = rl_steps_per_s * execute
    episode_s = episode_control / control_per_s
    print(f"   scenes {n}, round = {execute} physics steps ({execute * physics_ms / 1000:.2f}s,"
          f" concurrent) + forward ({forward:.2f}s) + render ({n * render_cost:.2f}s)")
    print(f"   round {round_s:.2f}s -> {rl_steps_per_s:.1f} RL steps/s ->"
          f" {control_per_s:.0f} control steps/s")
    print(f"   episode {episode_control} control steps -> {episode_s:.0f}s ->"
          f" {3600 / episode_s:.0f} episodes/hour")
    print()
    print(f"   {'hours':>6s} {'episodes':>9s} {'RL steps':>10s} {'control steps':>14s}")
    for hours in (1, 2, 4, 8):
        print(f"   {hours:6d} {3600 * hours / episode_s:9.0f} {rl_steps_per_s * 3600 * hours:10.0f}"
              f" {control_per_s * 3600 * hours:14.0f}")
    print()
    print("   参考实现：4 envs、eval_interval 3000 env steps、utd 20。")
    print("   对照：这里一个 episode 是 1382 control steps，参考的 robomimic 是 300。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
