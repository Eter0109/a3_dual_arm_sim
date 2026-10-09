"""Stage 1's gate: can one cookie expert drive either arm, and in which command space?

Four runs, chosen so that each one moves a single axis away from a reference that is already
known to work:

  * **a3 joint**        -- A3's arm, A3's own layout, joint commands.  The recorded baseline:
                           3 of 3 solved, 143 steps.  A regression check on the refactor: if
                           moving the command stage behind `command_space` changed the plan, it
                           shows up here.
  * **a3 cartesian**    -- same arm, same layout, cartesian commands.  Isolates the *command
                           space* with the arm held fixed.
  * **panda cartesian** -- same commands, same layout, LIBERO's arm.  Isolates the *arm*, with
                           the command space held fixed.  This is the pair the plan turns on.
  * **panda collect**   -- LIBERO's arm, cartesian, and the layout the scene actually needs: A3's
                           arrangement slid 200 mm into LIBERO's camera frame.  This is the
                           collection configuration, and it is the one that has to work.

A3's arm is deliberately absent from the last two.  It cannot reach the offset layout -- recorded
on the scene itself, and the reason the scene carries a note about fixing the expert "only when it
is actually needed -- for collecting data" -- so asking it to try would measure its workspace again
rather than this port.  The two behaviour probes measured the *Panda* reaching every object in that
layout, which is what makes it the right arm for the job.

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

CONFIGURATIONS = (
    ("a3 joint", A3_LAYOUT, dict(panda=False, cartesian=False)),
    ("a3 cartesian", A3_LAYOUT, dict(panda=False, cartesian=True)),
    ("panda cartesian", A3_LAYOUT, dict(panda=True, cartesian=True)),
    ("panda collect", COLLECT_LAYOUT, dict(panda=True, cartesian=True)),
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
    for label, layout, overrides in CONFIGURATIONS:
        if args.only and args.only != label:
            continue
        scene = libero_props_scene(
            task_name="a3_libero_panda_single_cookie",
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
