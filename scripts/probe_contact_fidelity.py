"""Does LIBERO's contact solver still hold our gripper?  Ask the expert that can grasp.

Reproducing LIBERO's environment means adopting its solver, and two of those settings are
about *contact*: `noslip_iterations` 0 against this harness's 5, and `impratio` 20 against the
cookie scene's 10.  The cookie scene was tuned with its own values, the no-slip pass among
them, and that pass is what stops a held object sliding out of the fingers between solver
steps.  LIBERO's Panda holds the same bowl with none of it -- which says something about
robosuite's gripper and nothing about ours.

So this does not measure a policy.  It runs the scripted expert, which is known to grasp, and
counts how often it completes a transfer with LIBERO's contact settings on and off.  If the
expert's success rate collapses, the setting is not transferable and reproducing LIBERO's
number costs this arm a grasp; if it does not move, contact is not where the difference is
and the rest of the restore can be judged on the policy's own behaviour.

Runs on the CPU and needs no checkpoint.

    python scripts/probe_contact_fidelity.py --episodes 12
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

#: See the module docstring: these are the settings `libero_fidelity` changes that can only
#: matter through contact.
CONTACT_KEYS = ("noslip_iterations", "impratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="a3_cookie_single_piece",
                        help="A scene whose scripted expert completes the task, so that a "
                             "failure is a statement about the physics.")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=900)
    return parser.parse_args()


def run_episodes(scene, episodes: int, max_steps: int) -> dict:
    """One scripted expert episode per seed, and what became of it."""

    env = scene.build_env(0.0, 0.0)
    try:
        expert = scene.build_policy(env)
        solver = {
            "noslip_iterations": int(env.model.opt.noslip_iterations),
            "impratio": int(round(float(env.model.opt.impratio))),
            "integrator": int(env.model.opt.integrator),
            "timestep": float(env.model.opt.timestep),
        }
        completed = 0
        lengths = []
        for seed in range(episodes):
            observation, _ = env.reset(seed=seed, options=None)
            expert.reset()
            step = 0
            placed = 0
            # Success is read from the environment rather than from the policy's own
            # attributes: `build_policy` hands back a recording wrapper, and the number the
            # task is scored on is the one that matters anyway.
            while step < max_steps and placed < 1:
                observation, _, terminated, truncated, info = env.step(expert.act(observation))
                placed = max(placed, int(info.get("cookies_in_target", 0)))
                step += 1
                if terminated or truncated:
                    break
            completed += placed >= 1
            lengths.append(step)
        return {"solver": solver, "completed": completed, "episodes": episodes,
                "median_steps": float(np.median(lengths))}
    finally:
        env.close()


def main() -> int:
    args = parse_args()

    # The scene's own builder loads the config by path, so the flag has to go in through the
    # loader the builder actually calls rather than through a config object handed in.
    import a3_dual_arm_sim.collection as collection
    from a3_dual_arm_sim.collection import scene_by_name
    from a3_dual_arm_sim.config import load_config

    original = collection.load_config
    print(f"scene {args.scene}   episodes {args.episodes}   max steps {args.max_steps}")
    print(f"(LIBERO's own values are {dict(noslip_iterations=0, impratio=20)})\n")

    results = {}
    for fidelity in (False, True):
        collection.load_config = lambda path=None, _f=fidelity: replace(
            original(path), libero_fidelity=_f
        )
        scene = scene_by_name(args.scene)
        outcome = run_episodes(scene, args.episodes, args.max_steps)
        results[fidelity] = outcome
        label = "LIBERO's contact settings" if fidelity else "this scene's own"
        print(f"  {label:<28} completed {outcome['completed']}/{outcome['episodes']}  "
              f"median {outcome['median_steps']:.0f} steps  {outcome['solver']}")
    collection.load_config = original

    before = results[False]["completed"]
    after = results[True]["completed"]
    print()
    if after < before:
        print(f"  => LIBERO's contact settings cost this gripper {before - after} of "
              f"{args.episodes} grasps.  Matching LIBERO's solver is not free here: the")
        print(f"     no-slip pass this scene was tuned with is doing work the Panda's gripper")
        print(f"     does not need, so reproducing LIBERO's number means losing ours.")
    else:
        print("  => the expert is unaffected, so contact is not where this restore bites;")
        print("     whatever changes the policy's behaviour is elsewhere in the bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
