"""Where in the source bin the Panda can actually grasp a Cookie, and what the rest costs.

This is the inset stage 4 needs before it can randomise the draw.  The plan said the usable region
is inset by the **jaw footprint** rather than by the slab's; the sweep below is what says by how
much, and the answer is not what the plan assumed.

Measured on A3's own layout, moving the Cookie away from the near wall along +y in 1 mm steps:

| offset from A3's placement | steps | DESCEND | CLOSE | retries |
|---|---|---|---|---|
| 0 - 8 mm | 431 | 120 | 180 | none |
| 9 mm | 254 | 110 | 12 | none |
| 10 mm and out | 157 | 15 | 12 | none |

The causes the plan guessed at are both wrong, and it matters which one is true:

* There are **no retries**.  `expert.retry_reasons` is empty at every offset and the phase path is
  identical, so the expert is not re-aiming at a reachable grasp.  Nothing is abandoned.
* It is not the jaw hitting the wall.  The bin's inner half-extent is 101 mm in y and the Cookie
  stands nowhere near it at these offsets.

What happens is that the arm cannot **converge** on the grasp pose.  DESCEND and CLOSE both end on
`_motion_done`, which in Cartesian space requires the site to be within 4 mm and 0.02 rad of the
aim *and* the jaw ramp to have finished; both phases carry an escape timeout (120 and 180 steps) so
that an unreachable aim does not hang.  Inside the band those timeouts are exactly what the phase
counts show -- 120 and 180 -- and outside it the same phases take 15 and 12.  The grasp still
succeeds either way, so the cost is the timeout, not the outcome.

Read this as a lower bound on the draw: a Cookie nearer than about 10 mm beyond A3's placement
costs 275 extra steps per episode, which a collection cannot pay.

Usage:

    python scripts/probe_panda_wall_clearance.py --seeds 0,1,2
"""

import argparse
import collections
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from a3_dual_arm_sim.collection import libero_props_scene

#: A3's own Cookie placement, which is what every offset below is measured from.  It stands in the
#: source bin's near-left corner, and the sweep moves it along +y, away from the near wall.
A3_COOKIE_POSITION = (0.1194, 0.0311)

LAYOUT = {
    "matched_cameras": True,
    "a3_single_cookie": True,
    "a3_layout_offset_m": (0.020, -0.200),
}


def run(scene, seeds, steps):
    """One configuration: what it solved and where its steps went."""

    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    expert = policy.expert
    phases: collections.Counter[str] = collections.Counter()
    solved = 0
    totals = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed, options=None)
            policy.reset()
            for step in range(1, steps + 1):
                observation, _, terminated, truncated, info = env.step(policy.act(observation))
                phases[expert.phase.name] += 1
                if terminated or truncated:
                    totals.append(step)
                    break
            else:
                totals.append(steps)
            if info.get("cookies_in_target"):
                solved += int(info["cookies_in_target"])
    finally:
        env.close()
    return solved, totals, phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--offsets", default="0,5,10,15,20,25,30,35,40")
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    offsets = [float(value) for value in args.offsets.split(",") if value.strip()]

    print(
        f"seeds {seeds}   offsets are measured from A3's own placement {A3_COOKIE_POSITION},"
        " which is what the collect config uses; a Cookie moves +y, away from the near wall"
    )
    print(
        f"  {'offset':>7s} {'y (m)':>8s} {'solved':>7s} {'steps':>13s}"
        f"  {'CLOSE':>6s} {'DESCEND':>8s}  other"
    )
    for offset in offsets:
        y = A3_COOKIE_POSITION[1] + offset / 1000.0
        scene = libero_props_scene(
            task_name="a3_libero_panda_single_cookie",
            hold_steps=40,
            panda=True,
            cartesian=True,
            a3_cookie_position_m=(A3_COOKIE_POSITION[0], y),
            **LAYOUT,
        )
        try:
            solved, totals, phases = run(scene, seeds, args.steps)
        except Exception as exc:  # noqa: BLE001 - a sweep reports; it does not stop
            print(f"  {offset:6.0f}mm {y:8.4f}   failed to run: {type(exc).__name__}: {exc}")
            continue
        # Shares of the whole sweep, not of one seed: the counter accumulates every seed's steps,
        # and these are per-seed averages, so the denominator has to be the same kind of total.
        every_step = max(sum(phases.values()), 1)
        share = {name: count / every_step for name, count in phases.items()}
        print(
            f"  {offset:6.0f}mm {y:8.4f} "
            f"{solved:4d}/{len(seeds):<2d} "
            f"{min(totals):5d}-{max(totals):<4d}  "
            f"{share.get('CLOSE', 0.0) * 100:5.0f}% "
            f"{share.get('DESCEND', 0.0) * 100:7.0f}%  "
            f"{100 * (1 - share.get('CLOSE', 0.0) - share.get('DESCEND', 0.0)):4.0f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
