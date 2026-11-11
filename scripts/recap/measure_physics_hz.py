"""How much simulation speed is available for free, and what it costs in fidelity.

Why this is worth measuring before anything else
------------------------------------------------
`cjProfile` on `env.step` attributes **92.4 %** of the cost to `mujoco.mj_step`: 3.435 s of 3.717 s
over 60 steps. Everything the policy sees -- the success judge, the observation builder, the
contact scan -- together is under 1 %. So the only lever that matters for throughput is how many
physics sub-steps are simulated, and the knob is `physics_hz`.

Two things have to be measured together, because either one alone is misleading:

* **speed** -- steps/s at each frequency;
* **fidelity** -- whether the same expert still succeeds. A faster simulation that changes the
  contact behaviour would make every downstream comparison meaningless, which is the failure mode
  this repository has already paid for twice (the merge bug and the label misalignment).

Fidelity is checked with the scripted expert rather than with the policy: the expert is
deterministic given a seed, so a difference between frequencies is a difference in the physics, not
in sampling noise. The policy would need dozens of episodes per configuration to say the same.

Usage
-----
    python scripts/recap/measure_physics_hz.py --frequencies 1000 500 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frequencies", nargs="+", type=int, default=[1000, 500, 400, 250])
    parser.add_argument("--bench-steps", type=int, default=150, help="steps for the speed measurement")
    parser.add_argument("--expert-seeds", nargs="+", type=int, default=[1000, 1006, 1012, 1018])
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/tmp/a3-main/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from bench_pi05_on_main import make_cookie_benchmark
    from dataclasses import replace

    def make_bench(frequency: int):
        """Benchmark with the physics rate replaced.

        `CookieTransferTaskConfig` is a frozen dataclass, so the field cannot be assigned; a new
        instance via `dataclasses.replace` is the supported way. `substeps` is derived
        (`physics_hz / control_hz`), which is why only `physics_hz` needs setting -- but it must
        divide evenly, or the control period is a non-integer number of physics steps and MuJoCo
        silently rounds. 250 Hz against a 20 Hz control rate is 12.5 and is included precisely to
        show what that looks like.
        """
        bench = make_cookie_benchmark(
            task="2x10", prompt_style="task", max_steps=args.max_steps,
            randomize_boxes=True, randomize_cookies=True, render=False, success_hold_steps=20,
        )
        # `bench.config` is a `SimConfig`, and `physics_hz` sits on it directly -- not on
        # `cookie_transfer` (`CookieSceneConfig`), which is where an earlier attempt put it and got
        # `unexpected keyword argument 'physics_hz'`. `SimConfig.validate` also rejects a rate that
        # is not an integer multiple of `control_hz`, which is why 250 Hz is in the list: 250/20 =
        # 12.5 and the rejection is itself the answer for that row.
        bench.config = replace(bench.config, physics_hz=int(frequency))
        return bench

    reset_options = {
        "randomize_cookies": True, "randomize_boxes": True,
        "randomize_source_bin": True, "randomize_target_bin": True,
        "source_bin_noise_m": 0.010, "target_bin_noise_m": 0.010,
        "target_bin_yaw_noise_rad": 0.030, "randomization_profile": "basic",
    }

    results = []
    for frequency in args.frequencies:
        entry: dict = {"physics_hz": frequency}
        try:
            bench = make_bench(frequency)
            env = bench.create_env(render_cameras=False)
            # `substeps` is a *property on `SimConfig`* (`physics_hz // control_hz`), not an
            # attribute of the env -- probing the env for it returned nothing and printed a
            # column of zeros.
            derived = int(bench.config.substeps)
            entry["substeps"] = derived
            entry["exact_division"] = frequency == derived * int(bench.config.control_hz)

            env.reset(seed=1000, options=reset_options)
            action = np.zeros(16)
            env.step(action)
            started = time.monotonic()
            for _ in range(args.bench_steps):
                env.step(action)
            elapsed = time.monotonic() - started
            entry["steps_per_second"] = args.bench_steps / elapsed
            entry["ms_per_step"] = 1000 * elapsed / args.bench_steps

            # Stationary drift: with a zero action held, how far do the arm's own joints move?
            # A lower physics rate that lets the arm sag from its own gravity is not free, and this
            # is the cheapest way to see it.
            before = np.array(env.data.qpos[:16], copy=True)
            for _ in range(50):
                env.step(action)
            entry["static_drift_mrad"] = float(
                1000 * np.abs(np.array(env.data.qpos[:16]) - before).max()
            )

            # Fidelity: the scripted expert on fixed seeds.
            #
            # This must go through `bench.run_episode`, not a hand-rolled loop.  The expert
            # validates its own reachable workspace in `__init__` by reading the *current* target
            # box pose, and `run_episode` resets the env *before* constructing/binding the policy.
            # Constructing the expert first (as an earlier version did) made it plan against the
            # un-randomized default layout and every seed raised
            # "target column 1 unreachable with vertical grasp: position error 430.50 mm".
            # `run_episode` also supplies the full `reset_options` (including `appearance_seed`
            # and `randomization_settings`), which the hand-rolled version had to duplicate and
            # would silently drift from.
            observed = []
            for seed in args.expert_seeds:
                score = bench.run_episode("variable_grasp", seed, episode_idx=seed)
                observed.append({
                    "seed": seed,
                    "cookies": int(score.cookies_in_target),
                    "success": bool(score.success),
                    "steps": int(score.steps),
                })
            entry["expert"] = observed
            entry["expert_mean_cookies"] = float(np.mean([o["cookies"] for o in observed]))
            entry["expert_successes"] = int(sum(1 for o in observed if o["success"]))
        except Exception as error:  # noqa: BLE001 -- a broken frequency is a result, not a crash
            entry["error"] = f"{type(error).__name__}: {error}"
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    # --- the comparison that decides whether the speedup is usable --------------------
    baseline = next((r for r in results if r["physics_hz"] == 1000 and "steps_per_second" in r), None)
    print()
    if baseline:
        print(f"{'Hz':>6} {'子步':>4} {'步/秒':>8} {'加速':>6} {'静止漂移':>9} {'专家均值':>9} {'专家成功':>9}")
        for entry in results:
            if "steps_per_second" not in entry:
                print(f"{entry['physics_hz']:6d}  {entry.get('error', '')[:40]}")
                continue
            speedup = entry["steps_per_second"] / baseline["steps_per_second"]
            print(f"{entry['physics_hz']:6d} {entry.get('substeps') or 0:4d}"
                  f" {entry['steps_per_second']:8.1f} {speedup:5.2f}x"
                  f" {entry.get('static_drift_mrad', float('nan')):8.2f} "
                  f"{entry.get('expert_mean_cookies', float('nan')):8.2f}"
                  f" {entry.get('expert_successes', 0)}/{len(args.expert_seeds)}")
        print()
        print("判读：加速可用当且仅当专家均值与成功率不降。")
        print("  250 Hz 那行会显示 12.5 子步——非整数，说明该频率不应使用。")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
