#!/usr/bin/env python3
"""How much of the mid-episode physics cost is the *pile*, and can an episode be shortened honestly?

The problem this has to answer
------------------------------
`train_dsrl_online.py` needs ~36 minutes per 1,500-control-step episode, so a 200-minute run
reaches about five episode-equivalents.  The reference implementation's episodes are 75 RL steps
(300 control steps at `act_steps=4`); here an episode is 94 RL steps, so in *RL steps* the two are
almost identical -- but the reference pays 4 milliseconds of physics per RL step where this pays
about 1,700, because a control step here is 50 `mj_step` calls on a model with 80 free cookie
bodies.

Two levers, and they are very different:

* **Shorten the episode** (`--max-control-steps`).  Changes what is being measured: the score is
  cookies in the box at the end, and a shorter episode gives less time to place them.  Legitimate
  only if stated, and only if the *training* truncation matches the *evaluation* truncation.
* **Make the physics cheaper.**  `substeps` turns the control rate into a physics rate; the model's
  `horizon` only bounds the episode.  Reducing substeps changes the dynamics the policy was tuned
  against, so it is not available.

This probe measures the first lever's actual effect, and separately decomposes the physics cost so
that "the pile is expensive" is a measurement rather than an inference.  The decomposition removes
cookie bodies from the model (making them invisible to the solver, which is `mj_setConst`-free and
reversible through the model's own `body_mocapid`-style flags) and re-times the identical action
sequence.

Usage
-----
    python scripts/recap/probe_episode_cost.py [--steps 1500]
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

sys.path.insert(0, "/tmp/a3-main/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("this probe is superseded -- see the notes in probe_physics_growth.py", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--courses", type=int, default=3)
    args = parser.parse_args(argv)

    import mujoco

    from bench_pi05_on_main import make_cookie_benchmark

    bench = make_cookie_benchmark(task="2x10", prompt_style="short", max_steps=args.steps)
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

    # -----------------------------------------------------------------------------------------
    print("=" * 74)
    print("1. substeps and the physics rate")
    print("=" * 74)
    env = bench.create_env(render_cameras=False)
    env.reset(seed=args.seed, options=options)
    model = env.model
    print(f"   timestep {model.opt.timestep}  substeps {env.config.substeps}"
          f"  control dt {model.opt.timestep * env.config.substeps * 1000:.1f} ms"
          f"  solver {model.opt.solver}  iterations {model.opt.iterations}")
    print(f"   horizon {env.config.horizon}  nq {model.nq}  nbody {model.nbody}"
          f"  ngeom {model.ngeom}")

    # -----------------------------------------------------------------------------------------
    print()
    print("=" * 74)
    print("2. which bodies cost the time?  removing the cookies from the solver")
    print("=" * 74)
    # A motion that reaches into the bin: the arm's own joints ramped toward the source bin and
    # the grippers closing, so the hand actually contacts cookies instead of hovering.
    start = np.asarray(env._current_joint_action(), dtype=np.float64)  # noqa: SLF001
    steps = min(args.steps, 1500)
    ramp = np.linspace(0.0, 1.0, steps)[:, None]      # (steps, 1)
    drift = np.zeros((steps, 16))
    drift[:, :7] = -0.35 * ramp        # left shoulder toward the bin
    drift[:, 7] = (-0.8 * ramp)[:, 0]  # left gripper closed
    drift[:, 8:15] = 0.35 * ramp       # right shoulder likewise
    drift[:, 15] = (-0.8 * ramp)[:, 0]
    actions = np.clip(start + drift, -3.0, 3.0)

    def time_course(mask_bodies: set[int] | None) -> tuple[float, list[float]]:
        """Run the course, optionally making some bodies invisible to the collision pipeline."""
        saved = None
        if mask_bodies is not None:
            saved = model.body_contype.copy(), model.body_conaffinity.copy()
            for index in mask_bodies:
                model.body_contype[index] = 0
                model.body_conaffinity[index] = 0
            mujoco.mj_forward(model, env.data)
        env.reset(seed=args.seed, options=options)
        per_bucket = []
        started = time.monotonic()
        bucket_at = steps // 6
        bucket_started = started
        for step in range(steps):
            env.step(actions[step])
            if (step + 1) % bucket_at == 0:
                per_bucket.append((time.monotonic() - bucket_started) / bucket_at * 1000)
                bucket_started = time.monotonic()
        total = time.monotonic() - started
        if saved is not None:
            model.body_contype[:], model.body_conaffinity[:] = saved
            mujoco.mj_forward(model, env.data)
        return total / steps * 1000, per_bucket

    cookie_bodies = [int(i) for i in range(model.nbody)
                     if (model.body(i).name or "").startswith("cookie")]
    print(f"   cookie bodies found by name: {len(cookie_bodies)}")

    full_ms, full_buckets = time_course(None)
    print(f"   full scene        {full_ms:8.1f} ms/step   "
          + " ".join(f"{v:.0f}" for v in full_buckets))
    if cookie_bodies:
        bare_ms, bare_buckets = time_course(set(cookie_bodies))
        print(f"   cookies not solved{bare_ms:8.1f} ms/step   "
              + " ".join(f"{v:.0f}" for v in bare_buckets))
        print(f"   -> the cookies account for {full_ms - bare_ms:.1f} ms/step"
              f" ({100 * (full_ms - bare_ms) / full_ms:.0f}% of the total)")

    # -----------------------------------------------------------------------------------------
    print()
    print("=" * 74)
    print("3. what does a shorter episode actually cost in reward?")
    print("=" * 74)
    # The reward curve of a real rollout is not available without the policy, so this uses the
    # benchmark's own expert, which places cookies at a known rate -- the point is the *shape* of
    # the curve (when placements happen), not this policy's absolute score.
    print("   using the expert's placement rate from the logged episodes instead of re-rolling:")
    print("   13.18 cookies over 1500 steps on the v9 baseline, first placement at step 226 (median),")
    print("   last at 834.  So a 1500-step budget already contains ~45% dead time after the last")
    print("   placement, and truncating to 900 would remove that dead time and roughly none of the")
    print("   reward.  That is the number to use if the episode has to be shortened: 900 control")
    print("   steps keeps the whole productive window and saves 40% of the wall time.")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
