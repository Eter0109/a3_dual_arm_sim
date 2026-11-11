#!/usr/bin/env python3
"""Why does a control step cost 97 ms early in a rollout and 410 ms later, when a synthetic
action sequence costs 61 ms from start to finish?

Established already, and worth restating because each result rules out a candidate:

* One scene, 1,500 synthetic control steps: **61 ms every step**, `nefc` pinned at 1980.  So the
  simulator does not get slower with time, and resting contacts cost what resting contacts cost.
* One scene, a *real* policy rollout: 97.5 ms a step over the first 10 %, 410 ms over the last 10 %,
  with `nefc` rising only 1,998 -> 2,628 (+31 %).  A 31 % rise in constraint rows cannot produce a
  4.2x rise in cost, so `nefc` is not the driver.
* The same slowdown appears with eight scenes in threads and with one scene alone, so it is not
  thread contention.  Cgroup quota is 16 cores and nothing else of ours is running.

What is left is a question about *where inside `step`* the time goes, and that needs the state that
was actually slow.  So this probe records `(qpos, qvel, ctrl)` at every step of a real rollout,
notes how long each step took, then goes back to a slow step and re-times it in pieces:

    mj_step only            | the solver, on exactly the saved state
    mj_forward only         | constraint construction without integration
    obs + action + reward   | the rest of `A3CookieTransferEnv.step`

Re-timing the *saved* state is what makes this causal rather than correlational: the same numbers
are fed back to the solver, so if `mj_step` alone is fast there while the original step was slow,
the cost was never in the solver.

Usage
-----
    python scripts/recap/probe_step_slowdown.py [--rounds 30] [--execute 16]
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
os.environ.setdefault("A3_MAIN_ROOT", "/tmp/a3-main")

import numpy as np
import torch

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from steering import encode_prefix, sample_with_prefix  # noqa: E402
from train_dsrl_online import (  # noqa: E402
    CAMERAS,
    HISTORY_DIM,
    HISTORY_FRAMES,
    STATE,
    VELOCITY,
    Scene,
    _make_renderer,
    raw_batch,
    render_frames,
    reset_options,
)

TASK = ("grasp five yellow blocks together from the source bin, insert them upright into the "
        "marked slots of the 2 by 5 box, then return to the source bin, grasp five more, and "
        "fill all ten slots")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--execute", type=int, default=16)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import mujoco

    import train_pi05_launcher as launcher
    from bench_pi05_on_main import make_cookie_benchmark
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    launcher._patch_state_projector()  # noqa: SLF001
    launcher._patch_history_encoder()  # noqa: SLF001
    launcher._patch_obsolete_history_keys()  # noqa: SLF001

    checkpoint = Path("/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model")
    policy = PI05Policy.from_pretrained(str(checkpoint)).to(args.device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    chunk_size = int(policy.config.chunk_size)
    action_dim = int(policy.config.max_action_dim)
    num_steps = int(policy.config.num_inference_steps)
    original_dim = int(policy.config.output_features["action"].shape[0])

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=1500)
    env = bench.create_env(render_cameras=False)
    env.reset(seed=args.seed, options=reset_options(bench, args.seed))
    scene = Scene(index=0, env=env, renderer=_make_renderer(env.model),
                  camera_ids=dict(env._camera_ids), seed=args.seed)  # noqa: SLF001
    renderer = scene.renderer

    print("=" * 74)
    print(f"rolling out {args.rounds} rounds x {args.execute} control steps")
    print("=" * 74)
    print(f"   {'round':>6s} {'ms/step':>8s} {'nefc':>6s} {'ncon':>5s} {'|qvel|max':>10s}")

    records: list[dict] = []
    for round_index in range(args.rounds):
        frames = render_frames(renderer, scene)
        observation = env._observation()  # noqa: SLF001
        raw = raw_batch([scene], [frames], [observation], TASK, args.device)
        prepared = preprocess(raw)
        prefix = encode_prefix(policy, prepared, keep_embeddings=True)
        with torch.no_grad():
            noise = torch.randn(1, chunk_size * action_dim, device=args.device)
            chunk = sample_with_prefix(policy, prefix, noise, num_steps, args.device)
        executed = np.asarray(postprocess(chunk[:, :, :original_dim])[0].detach().float().cpu(),
                              dtype=np.float64)

        round_times = []
        for offset in range(min(args.execute, len(executed))):
            # Snapshot the state *before* the step, so the saved copy is what the solver saw.
            record = {
                "qpos": np.array(env.data.qpos, copy=True),
                "qvel": np.array(env.data.qvel, copy=True),
                "ctrl": np.array(env.data.ctrl, copy=True),
                "action": np.array(executed[offset], copy=True),
                "nefc": int(env.data.nefc),
                "ncon": int(env.data.ncon),
                "qvel_max": float(np.max(np.abs(env.data.qvel))),
            }
            started = time.monotonic()
            observation, reward, terminated, truncated, info = env.step(executed[offset])
            record["ms"] = (time.monotonic() - started) * 1000
            scene.history.append(scene.frame(observation))
            if len(scene.history) > HISTORY_FRAMES:
                del scene.history[0]
            scene.applied = np.asarray(info.get("applied_action"), dtype=np.float32)[:16].copy()
            round_times.append(record["ms"])
            records.append(record)
            if terminated or truncated:
                break
        print(f"   {round_index + 1:6d} {np.mean(round_times):8.1f} {records[-1]['nefc']:6d}"
              f" {records[-1]['ncon']:5d} {records[-1]['qvel_max']:10.1f}")
        if terminated or truncated:
            print("   episode ended")
            break

    # =========================================================================================
    print()
    print("=" * 74)
    print("re-timing saved states: is the cost in the solver or around it?")
    print("=" * 74)
    ms = np.asarray([r["ms"] for r in records])
    order = np.argsort(ms)
    slow = [records[i] for i in order[-6:]]
    fast = [records[i] for i in order[:6]]

    model, data = env.model, env.data
    substeps = int(env.config.substeps)
    print(f"   substeps per control step: {substeps}")

    def retime(sample: list[dict]) -> tuple[float, float, float]:
        """mj_step / mj_forward / the whole step, on recorded states."""
        step_ms, forward_ms, full_ms = 0.0, 0.0, 0.0
        for record in sample:
            data.qpos[:] = record["qpos"]
            data.qvel[:] = record["qvel"]
            data.ctrl[:] = record["ctrl"]
            mujoco.mj_forward(model, data)
            started = time.monotonic()
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            step_ms += (time.monotonic() - started) * 1000
            data.qpos[:] = record["qpos"]
            data.qvel[:] = record["qvel"]
            data.ctrl[:] = record["ctrl"]
            started = time.monotonic()
            mujoco.mj_forward(model, data)
            forward_ms += (time.monotonic() - started) * 1000
            data.qpos[:] = record["qpos"]
            data.qvel[:] = record["qvel"]
            data.ctrl[:] = record["ctrl"]
            mujoco.mj_forward(model, data)
            started = time.monotonic()
            env.step(record["action"])
            full_ms += (time.monotonic() - started) * 1000
        n = len(sample)
        return step_ms / n, forward_ms / n, full_ms / n

    for label, sample in (("fastest 6 steps", fast), ("slowest 6 steps", slow)):
        original = float(np.mean([r["ms"] for r in sample]))
        step_ms, forward_ms, full_ms = retime(sample)
        print(f"   {label:16s} originally {original:7.1f} ms"
              f"   mj_step x{substeps} {step_ms:7.1f}"
              f"   mj_forward {forward_ms:7.1f}"
              f"   env.step {full_ms:7.1f}")
        print(f"   {'':16s} nefc {int(np.mean([r['nefc'] for r in sample]))}"
              f"   |qvel|max {np.mean([r['qvel_max'] for r in sample]):.1f}")

    print()
    print("=" * 74)
    print("correlations over all steps")
    print("=" * 74)
    nefcs = np.asarray([r["nefc"] for r in records], dtype=float)
    qvels = np.asarray([r["qvel_max"] for r in records], dtype=float)
    for label, values in (("nefc", nefcs), ("ncon", np.asarray([r["ncon"] for r in records], float)),
                          ("|qvel|max", qvels), ("nefc x |qvel|", nefcs * qvels)):
        if values.std() > 0:
            print(f"   corr(ms, {label:12s}) = {np.corrcoef(ms, values)[0, 1]:+.3f}")
    print(f"   ms: min {ms.min():.1f} median {np.median(ms):.1f} max {ms.max():.1f}"
          f"  ratio {ms.max() / ms.min():.1f}x")

    print()
    print("=" * 74)
    print("结论的读法")
    print("=" * 74)
    print("  mj_step 在慢步上快 -> 开销不在求解器，而在 env.step 的 Python 侧或外部干扰。")
    print("  mj_step 在慢步上慢 -> 求解器确实在这些状态上工作更久，吞吐要按最坏情况预算。")
    print("  原始 ms 高但重放 ms 低 -> 当初的慢来自运行时环境（其他进程、限流），不是状态。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
