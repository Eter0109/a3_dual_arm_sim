"""Where a placement's steps actually go, phase by phase.

Stage 1's gate asks for 200 steps or fewer per Cookie.  The same expert, the same
arm and the same Cookie count take 155 steps with the Cookie 40 mm clear of the
source bin's wall and 431 with it where A3 puts it, and both solve, so the
difference is inside the phases rather than in the outcome.  A bare total cannot
say which phase pays it, which is what this prints.

Nothing here is a conclusion about the *expert*: the corner placement is the case
`a3_cookie_position_m` exists to exercise, and the dataset's own draw insets the
usable region so that a Cookie never stands there.

Usage:

    python scripts/probe_panda_phase_budget.py --seeds 0,1,2
"""

import argparse
import collections
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from a3_dual_arm_sim.collection import libero_props_scene

# Kept in step with `run_expert_gate.py`, which is where these are explained.
LAYOUTS = {
    "collect": {
        "matched_cameras": True,
        "a3_single_cookie": True,
        "a3_layout_offset_m": (0.020, -0.200),
    },
    "collect clear of wall": {
        "matched_cameras": True,
        "a3_single_cookie": True,
        "a3_layout_offset_m": (0.020, -0.200),
        "a3_cookie_position_m": (0.1194, 0.0711),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=900)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]

    for name, layout in LAYOUTS.items():
        scene = libero_props_scene(
            task_name="a3_libero_panda_single_cookie",
            hold_steps=40,
            panda=True,
            cartesian=True,
            **layout,
        )
        env = scene.build_env(0.0, 0.0)
        policy = scene.build_policy(env)
        expert = policy.expert
        print(f"\n{name}")
        totals: collections.Counter[str] = collections.Counter()
        runs = 0
        for seed in seeds:
            observation, _ = env.reset(seed=seed, options=None)
            policy.reset()
            observed: collections.Counter[str] = collections.Counter()
            order: list[str] = []
            started = time.monotonic()
            steps = 0
            for step in range(1, args.steps + 1):
                observation, _, terminated, truncated, info = env.step(policy.act(observation))
                steps = step
                phase = expert.phase.name
                observed[phase] += 1
                if not order or order[-1] != phase:
                    order.append(phase)
                if terminated or truncated:
                    break
            runs += 1
            totals.update(observed)
            placed = info.get("cookies_in_target")
            print(
                f"  seed {seed}: {steps} steps, placed={placed}"
                f"  {time.monotonic() - started:.1f}s"
            )
            print(f"    {' -> '.join(order)}")
            for phase, count in observed.most_common():
                print(f"      {phase:18s} {count:4d}")
        if runs:
            print("  per-phase mean across seeds")
            for phase, count in totals.most_common():
                print(f"    {phase:18s} {count / runs:7.1f}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
