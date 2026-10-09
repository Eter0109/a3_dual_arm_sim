"""Stage 1's gate, on the arm this repository collects with: LIBERO's Panda.

The collection runs on the Panda and its parameters and low-level control are fixed, so A3's arm is
not a configuration here.  What it is kept for is the *expert*: A3's scene is where the path
planning, the phase machine and the recording were developed, and that is the reference for how a
collection is driven rather than a thing to run.

The two configurations differ in one axis, the layout:

  * **a3 layout**    -- A3's arrangement where A3 puts it.  The layout the expert was tuned
                       against, so a failure here is about the expert or the arm.
  * **collect**      -- the same arrangement slid 200 mm into LIBERO's camera frame, which is the
                       configuration the dataset needs.  A failure that appears here and not above
                       is about the offset.

Usage:

    python scripts/run_expert_gate.py --seeds 0,1,2
"""

import argparse
import os
import statistics
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from a3_dual_arm_sim.collection import libero_props_scene
from a3_dual_arm_sim.runner import EpisodeRunner

#: A3's arrangement where A3 puts it, with the Cookie standing in the source bin's corner.  The
#: layout the expert was written and tuned against, so it is the reference for anything about the
#: *expert* rather than about the scene.
A3_LAYOUT = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.0, 0.0),
)

#: The same arrangement slid into LIBERO's camera frame, with the Cookie still in the bin.  This is
#: the collection configuration: LIBERO's agentview is aimed at where LIBERO's bowl used to be and
#: A3's props are 200 mm away from that, so the offset is the price of leaving the camera alone.
#:
#: The Cookie's own placement is A3's (`a3_cookie_position_m` left unset) -- the registered scene
#: moves and quarter-turns it, which is a visibility experiment and not a candidate configuration.
COLLECT_LAYOUT = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
)

#: The same, with the Cookie slid 40 mm further into the bin.  This configuration exists because
#: the cost of a Cookie that stands near A3's own placement is large, and measuring it is how that
#: number was found: 431 steps against 157, with DESCEND and CLOSE running to their 120- and
#: 180-step escape timeouts because the arm does not converge on the grasp pose there.  It is a
#: **reachability** inset and not a collision one -- see `probe_panda_wall_clearance.py`, which
#: sweeps it, and the plan's Stage 4, which is where the number is used.
COLLECT_CLEAR_OF_WALL = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
    a3_cookie_position_m=(0.1194, 0.0711),
)

#: Stage 2: two Cookies, in the same column and ten rows apart (about 71 mm), so the second is not
#: the first one's neighbour when the first is lifted out.  Row 2 rather than row 0 because row 0 is
#: A3's own placement, which the sweep above shows sits inside the band the arm cannot converge in.
TWO_COOKIES_LAYOUT = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
    a3_cookie_positions_m=((0.1194, 0.04875), (0.1194, 0.11941667)),
)

#: (label, task name, layout, overrides).  The task name is per configuration rather than fixed
#: because the two-Cookie scene declares its own task -- it asks for two pieces and its success
#: rule counts two, so a summary that borrowed the single-piece contract would be wrong.
CONFIGURATIONS = (
    ("a3 layout", "a3_libero_panda_single_cookie", A3_LAYOUT, dict(panda=True, cartesian=True)),
    (
        "collect",
        "a3_libero_panda_single_cookie",
        COLLECT_LAYOUT,
        dict(panda=True, cartesian=True),
    ),
    (
        "collect clear of wall",
        "a3_libero_panda_single_cookie",
        COLLECT_CLEAR_OF_WALL,
        dict(panda=True, cartesian=True),
    ),
    (
        "two cookies",
        "a3_libero_panda_two_cookies",
        TWO_COOKIES_LAYOUT,
        dict(panda=True, cartesian=True),
    ),
)


def run_one(scene, seeds, steps, task, quiet):
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    runner = EpisodeRunner(
        env,
        policy,
        task=task,
        recorded_action="requested",
        save_failed_episodes=True,
    )
    rows = []
    try:
        for seed in seeds:
            started = time.monotonic()
            result = runner.run(seed=seed, max_steps=steps)
            expert = policy.expert
            # The environment's own count, not the expert's bookkeeping: `terminate_on_success`
            # ends the episode the moment the Cookie is in and released, which can be before the
            # expert's own RETRACT accounting has run, so `completed_cookie_indices` reads 0 on a
            # solved episode and would report the successes as failures.
            in_target = result.final_info.get("cookies_in_target")
            rows.append(
                {
                    "seed": seed,
                    "success": result.success,
                    "steps": result.steps,
                    "seconds": time.monotonic() - started,
                    "reason": expert.failure_reason,
                    "retries": len(expert.retry_reasons),
                    "cookies": -1 if in_target is None else int(in_target),
                }
            )
            if not quiet:
                row = rows[-1]
                print(
                    f"    seed {row['seed']}: success={row['success']!s:5s} "
                    f"steps={row['steps']:4d} placed={row['cookies']} "
                    f"retries={row['retries']} {row['seconds']:5.1f}s"
                    + (f"  reason={row['reason']}" if row["reason"] else "")
                )
                # Where it ended and how it got there.  A rollout that runs to its cap with no
                # failure reason and no retries is *stuck*, not failing, and the phase history is
                # the only thing that says where.
                history = [phase.name for phase in expert.transition_history]
                print(f"      ended in {expert.phase.name}; path {' -> '.join(history[-8:])}")
    finally:
        runner.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--task", default="")
    parser.add_argument("--only", default="", help="run only the named configuration")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    print(f"seeds {seeds}   cap {args.steps} steps per episode")
    summary = []
    for label, task_name, layout, overrides in CONFIGURATIONS:
        if args.only and args.only != label:
            continue
        scene = libero_props_scene(
            task_name=task_name,
            hold_steps=40,
            **layout,
            **overrides,
        )
        task = args.task or scene.task.prompt
        print(f"\n  {label}")
        started = time.monotonic()
        try:
            rows = run_one(scene, seeds, args.steps, task, args.quiet)
        except Exception as exc:  # noqa: BLE001 - a gate reports; it does not kill the sweep
            print(f"    FAILED to run: {type(exc).__name__}: {exc}")
            summary.append((label, None, None, time.monotonic() - started))
            continue
        solved = sum(1 for row in rows if row["success"])
        steps = [row["steps"] for row in rows]
        print(
            f"    solved {solved}/{len(rows)}   "
            f"median steps {int(statistics.median(steps))}   "
            f"{time.monotonic() - started:.0f}s total"
        )
        summary.append((label, solved, statistics.median(steps), time.monotonic() - started))

    print("\n  summary")
    for label, solved, steps, seconds in summary:
        outcome = "did not run" if solved is None else f"{solved} solved, median {int(steps)} steps"
        print(f"    {label:18s} {outcome:34s} {seconds:5.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
