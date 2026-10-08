"""One yardstick for any object on the table -- the ladder's instrument.

Axis 4 of the ladder changes the object and nothing else, so the two ends have to be
measured by the *same code in the same units* or the comparison is between two probes
rather than between two scenes.  This runs a policy in a scene, watches one named body,
and reports the quantities Gate 3 of `docs/scene-transition-and-bug-triage.md` asks for:

  * the nearest fingertip's distance to that body's centre;
  * how far the body ever moved, and whether anything ever touched it;
  * whether the gripper closed *while it was on the body* or after it had left it.

Why a second probe rather than a flag on the first
--------------------------------------------------
`probe_libero_props_trace.py` is about a *bowl*, and not incidentally: its central
quantities are the rim's radius and the fingertip's height against the rim, because the
bowl is a hollow open-topped shell and "25 mm from the centre" describes a fingertip above
the opening and one reaching into the cavity equally well.  A 6.3 mm-thick Cookie standing
on its edge has no mouth, so those two numbers are undefined rather than small.

Retrofitting it would also change the probe the guard's LIBERO canary runs on.  That
canary's numbers are a recorded baseline (`configs/invariants_baseline.json`) and the
format of its output is part of what is compared, so a probe that reports different things
after an edit would move the baseline instead of catching a regression.

The two ends
------------
    # the working end: the policy places LIBERO's bowl on LIBERO's plate, 35/40
    python scripts/probe_ladder_object.py --scene a3_libero_panda \\
        --object libero_bowl_object --target libero_plate_object \\
        --manifest configs/policies/pi05_libero_panda.yaml \\
        --task "pick up the black bowl between the plate and the ramekin and place it on the plate"

    # axis 4: the same scene, camera, robot and physics, with A3's Cookie instead
    python scripts/probe_ladder_object.py --scene a3_libero_panda_cookie \\
        --object cookie_0 --target libero_plate_object \\
        --manifest configs/policies/pi05_libero_panda.yaml \\
        --task "pick up the yellow block and place it on the plate"

A **capability** failure degrades monotonically along such an axis; a **bug** is a cliff at
a wiring boundary.  The numbers this prints are what tells the two apart, so they are
printed rather than summarised into one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.props import (
    EEF_SITE,
    FINGERTIPS,
    arm_qpos_addresses,
    prop_frame,
    prop_geoms,
    robot_bodies,
    touching,
)

PROJECT = Path(__file__).resolve().parents[1]

#: One unit of action, under the scene's `cartesian_translation_scale_m`.  Same constant as
#: the bowl probe, and for the same reason: the model's command is in normalised OSC units
#: and a number in those units is not readable.
ACTION_UNIT_M = 0.05

#: `observation.state[7]` -- the left gripper's opening, in the scene's own convention.
GRIPPER_STATE_INDEX = 7

#: A close is a *fraction of this run's own open value*, not an absolute opening.
#:
#: The absolute version of this existed and was wrong.  0.05 was calibrated on the A3's
#: finger servo, which reaches 0.027 over an **empty** jaw travel.  In `a3_libero_panda`
#: the jaws stop at **0.0823** because they are holding a 112 mm bowl, and a gripper
#: gripping something cannot reach the value it reaches gripping nothing.  Against 0.05 the
#: probe reported `gripper closed on 0/150 steps` and
#: `moved it without ever closing the gripper on it` for a rollout that closed on step 47,
#: held the hold, and carried the bowl 114.7 mm of the 117.6 mm to the plate.  The error is
#: not merely conservative: it asserts "the policy cannot grasp", which is the hypothesis
#: this ladder exists to test, so a threshold that can only fail is worse than none.
#:
#: The reference is the **maximum opening seen in the run** -- the open value in both
#: scenes (A3 1.0; the Panda's `pruned_init` poses 0.6755, rising to 0.994) -- and the rule
#: is "below half of it".  Both real holds (0.027 and 0.0823) are far below that and both
#: idles are above, so no scene needs a constant of its own.
CLOSED_FRACTION_OF_OPEN = 0.5

#: How far the object has to move before it counts as moved at all.  The props settle to
#: within 0.63 mm of their start height in this scene (measured, `libero_props_baseline12`),
#: so this is a settled-noise allowance rather than a physical tolerance.
MOVED_MM = 2.0


def _closed_steps(trace: dict[str, np.ndarray]) -> np.ndarray:
    """The steps on which the gripper counts as closed, against its own open value.

    Returned rather than reduced to a boolean because both callers need to know *which*
    step, and comparing a closure time against a contact time is the only way to tell "it
    picked the object up" from "it put the object down and then closed".
    """

    reference = float(np.max(trace["gripper"]))
    return np.flatnonzero(trace["gripper"] < CLOSED_FRACTION_OF_OPEN * reference)


def _held_steps(trace: dict[str, np.ndarray]) -> np.ndarray:
    """The steps on which the gripper was **closed and on the object at the same time**.

    This is the whole grasp test, and it used to be a one-step lookback instead: `was the
    object in contact on the step before the first closed step`.  On `a3_libero_panda` seed
    2 the gripper crossed its close threshold on the same step it first touched the bowl --
    it had been shutting over the approach, 0.99 -> 0.81 -> 0.57 
    -> 0.35 across steps 34-37 -- so the step before the close was `no contact`, and a
    rollout with **84** steps of closed-and-touching was reported as `closed the gripper
    after leaving it`.  Reading a *duration* off a step boundary is the error; a state that
    holds for 84 steps cannot be inferred from the one step before its first frame.
    """

    closed = np.zeros(len(trace["gripper"]), dtype=bool)
    closed[_closed_steps(trace)] = True
    return np.flatnonzero(closed & (trace["touch_arm"] > 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="a3_libero_panda")
    parser.add_argument("--object", default="libero_bowl_object",
                        help="the body to watch; `cookie_0` in the axis-4 scene")
    parser.add_argument("--target", default="libero_plate_object",
                        help="the body a placement is against; empty to skip")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT / "configs/policies/pi05_libero_panda.yaml")
    parser.add_argument("--task", default="",
                        help="the instruction; empty uses the scene's own prompt")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--trace", type=Path, default=None)
    return parser.parse_args()


def _signature(trace: dict[str, np.ndarray]) -> str:
    """Which of Gate 3's failure signatures this rollout is.

    Read off the three observations, not off a threshold that decides pass or fail:
    whether anything touched the object, whether it moved, and whether the gripper closed
    while it was still on it.  The point of naming the shape is that the shapes mean
    different things -- "never touched it" and "touched it but never moved it" are both
    0% and are not the same finding.
    """

    if float(np.abs(trace["joint_motion_mrad"]).max()) <= 0.0:
        return "the arm never moved (execution bug, not capability)"
    touched = int(trace["touch_arm"].sum())
    moved = float(trace["object_mm"].max())
    closed = _closed_steps(trace)
    held = _held_steps(trace)
    if touched == 0:
        # `touch_arm` is MuJoCo's contact list read **once per control step**, so it is a
        # lower bound, and the bound is loosest for the smallest object.  A 112 mm bowl
        # presents a contact that lasts tens of steps and cannot be missed; a 6.3 mm Cookie
        # brushed for a few physics substeps can be, and on seed 0 it was: the Cookie slid
        # 15.9 mm over 75 steps with no recorded contact at any sample.  Saying "never
        # touched the object" there is not conservative, it is false, and it is false in the
        # direction that flatters the working end of the ladder -- the bias is against
        # exactly the object the axis is about.  Displacement is the channel that cannot miss
        # it, so it decides which of the two sentences this is.
        if moved > MOVED_MM:
            return (f"never recorded in contact, yet it moved {moved:.1f} mm -- the contact "
                    f"was between samples")
        return "never touched the object"
    if moved <= MOVED_MM:
        return "touched it, never moved it"
    if closed.size == 0:
        return "moved it without ever closing the gripper on it"
    if held.size == 0:
        return "closed the gripper, but never while it was on the object"
    return "grasped it and carried it"


def _grasp_timing(trace: dict[str, np.ndarray]) -> str:
    """Did the gripper close while it was on the object, or after it had left it?

    Reported as a *timing* rather than as a rate because a grip rate cannot separate those
    two, and they are different failures with the same summary number.
    """

    closed = _closed_steps(trace)
    held = _held_steps(trace)
    contacts = np.flatnonzero(trace["touch_arm"] > 0)
    if closed.size == 0:
        return "the gripper never closed"
    if contacts.size == 0:
        return (f"the gripper first closed on step {int(closed[0])}, on an object it had "
                f"never touched")
    if held.size:
        return (f"the gripper was closed on the object for {held.size} steps, first on step "
                f"{int(held[0])} (first contact {int(contacts[0])}, first close "
                f"{int(closed[0])})")
    return (f"the gripper first closed on step {int(closed[0])} and never while it was on "
            f"the object, whose last contact was step {int(contacts[-1])}")


def main() -> int:
    args = parse_args()

    from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
    from a3_dual_arm_sim.bridge.manifest import load_manifest
    from a3_dual_arm_sim.collection import scene_by_name
    from a3_dual_arm_sim.contracts import STATE, EpisodeContext

    scene = scene_by_name(args.scene)
    task = args.task or scene.task.prompt
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gateway = None
    runs: list[dict[str, np.ndarray]] = []

    for seed in seeds:
        env = scene.build_env(0.0, 0.0)
        env.use_fast_render()
        observation, _ = env.reset(seed=seed, options=None)

        model = env.model
        watched = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object)
        target = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target)
                  if args.target else -1)
        tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in FINGERTIPS]
        eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        if watched < 0 or min(eef, *tips) < 0:
            raise SystemExit(
                f"{args.scene} has no body {args.object!r}, or is missing the arm's sites"
            )

        if gateway is None:
            gateway = ExternalPolicyGateway(load_manifest(args.manifest), env=env)
        else:
            gateway.rebind(env)
        gateway.reset(EpisodeContext(seed=seed, task=task, action_mode="joint_position"))

        # `geom_xpos` is derived, and a fresh reset has not necessarily been forwarded.
        mujoco.mj_forward(model, env.data)
        start, radius_m, size_m = prop_frame(model, env.data, watched)
        target_centre = None if target < 0 else prop_frame(model, env.data, target)[0]
        watched_geoms = set(prop_geoms(model, watched))
        arm_bodies = robot_bodies(model)
        arm_qpos = arm_qpos_addresses(model)

        if seed == seeds[0]:
            print(f"scene    {args.scene}")
            print(f"watched  {args.object}   centre {np.round(start, 4)}  "
                  f"radius {radius_m * 1000:.1f} mm  size "
                  f"{np.round(size_m * 1000, 1)} mm")
            if target_centre is not None:
                print(f"target   {args.target}   centre {np.round(target_centre, 4)}  "
                      f"radius {prop_frame(model, env.data, target)[1] * 1000:.1f} mm   "
                      f"start distance "
                      f"{np.linalg.norm(start[:2] - target_centre[:2]) * 1000:.1f} mm")
            print(f"task     {task!r}")
            print()

        rows = []
        previous_joints: np.ndarray | None = None
        for step in range(args.steps):
            action = gateway.act(observation, task)
            observation, _, terminated, truncated, info = env.step(action)

            centre = prop_frame(model, env.data, watched)[0]
            fingertips = [np.asarray(env.data.site_xpos[t]) for t in tips]
            gap = min(float(np.linalg.norm(tip - centre)) for tip in fingertips)
            eef_position = np.asarray(env.data.site_xpos[eef], dtype=np.float64)
            joints = np.asarray(env.data.qpos[arm_qpos], dtype=np.float64)

            rows.append(
                {
                    "step": step,
                    "gap_mm": gap * 1000.0,
                    "object_mm": float(np.linalg.norm(centre - start)) * 1000.0,
                    "object_height_mm": float(centre[2] - start[2]) * 1000.0,
                    "object_target_mm": (float("nan") if target_centre is None else
                                         float(np.linalg.norm(
                                             centre[:2] - target_centre[:2])) * 1000.0),
                    # MuJoCo's own contact list, not a proximity threshold.
                    "touch_arm": float(touching(model, env.data, watched_geoms, arm_bodies)),
                    "touch_table": float(touching(model, env.data, watched_geoms, {0})),
                    "gripper": float(np.asarray(observation[STATE], dtype=np.float64)[
                        GRIPPER_STATE_INDEX]),
                    "eef_x_mm": float(eef_position[0] - start[0]) * 1000.0,
                    "eef_y_mm": float(eef_position[1] - start[1]) * 1000.0,
                    "eef_z_mm": float(eef_position[2] - start[2]) * 1000.0,
                    # The arm-moved signature: **measured** joint motion, in mrad.  It used
                    # to be read off `info["applied_action"]`, which is wrong in the Panda
                    # scene in two separate ways -- `env._limit_joint_action` returns those
                    # seven entries *unclipped* when the arm is driven by an
                    # `OperationalSpaceAdapter`, and under that adapter they are **torques
                    # in N.m, not joint angles**.  The difference is not subtle: it reported
                    # a peak of 17 129 "mrad" per step, which is 17 rad, in a scene whose
                    # per-step cap is 0.1 rad.  A quantity recorded under a unit it does not
                    # have is how a rollout that never moved gets mistaken for one that did,
                    # and the reverse.
                    "joint_motion_mrad": 0.0 if previous_joints is None
                    else float(np.abs(joints - previous_joints).max()) * 1000.0,
                    # Kept, but named for what it is: the raw entry the servo received,
                    # in whatever unit the scene's action mode puts there.  Compare it with
                    # nothing across scenes -- `joint_motion_mrad` is that quantity.
                    "command_max": float(np.abs(
                        np.asarray(info["applied_action"], dtype=np.float64)[:7]).max()),
                }
            )
            previous_joints = joints
            if terminated or truncated:
                print(f"seed {seed}: episode ended at step {step}", flush=True)
                break
        env.close()

        trace = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        runs.append(trace)
        closest = int(np.argmin(trace["gap_mm"]))
        open_reference = float(np.max(trace["gripper"]))
        print(
            f"seed {seed}: nearest fingertip {trace['gap_mm'][closest]:.1f} mm from the "
            f"object centre at step {closest}  |  object moved "
            f"{trace['object_mm'].max():.2f} mm, peak height "
            f"{trace['object_height_mm'].max():+.1f} mm  |  gripper closed on "
            f"{_closed_steps(trace).size}/{len(rows)} steps and closed *on the object* for "
            f"{_held_steps(trace).size}, arm on "
            f"the object for {int(trace['touch_arm'].sum())}, on the table for "
            f"{int(trace['touch_table'].sum())}  |  {len(rows)} of {args.steps} steps run\n"
            f"        gripper: open value this run {open_reference:.4f}, so closed means "
            f"below {CLOSED_FRACTION_OF_OPEN * open_reference:.4f}, and its minimum was "
            f"{float(trace['gripper'].min()):.4f} on step "
            f"{int(trace['gripper'].argmin())}\n"
            f"        {_grasp_timing(trace)}   ->  {_signature(trace)}",
            flush=True,
        )

    if gateway is not None:
        gateway.close()

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.trace, **{f"{key}_s{index}": trace[key]
                                for index, trace in enumerate(runs) for key in trace})

    # The three Gate-3 quantities, per seed, in the units the decision is made in.  Printed
    # as a list rather than a mean because this policy samples fresh flow-matching noise per
    # chunk: the spread across seeds is itself a finding (`probe_libero_props_trace.py` says
    # the same thing at more length).
    closest = [float(t["gap_mm"].min()) for t in runs]
    print()
    print(f"nearest fingertip to {args.object}, per seed: "
          + ", ".join(f"{value:.1f}" for value in closest)
          + f" mm   (min {min(closest):.1f}, spread {max(closest) - min(closest):.1f})")
    print("object displaced, per seed: "
          + ", ".join(f"{float(t['object_mm'].max()):.2f}" for t in runs) + " mm")
    print("peak height above its start, per seed: "
          + ", ".join(f"{float(t['object_height_mm'].max()):+.1f}" for t in runs) + " mm")
    print("steps with the arm touching the object, per seed: "
          + ", ".join(f"{int(t['touch_arm'].sum())}/{len(t['touch_arm'])}" for t in runs))
    if runs and not np.isnan(runs[0]["object_target_mm"]).all():
        print(f"object to {args.target} at the last step, per seed: "
              + ", ".join(f"{float(t['object_target_mm'][-1]):.1f}" for t in runs)
              + " mm;  best "
              + ", ".join(f"{float(t['object_target_mm'].min()):.1f}" for t in runs))
    print("peak arm motion per step, per seed: "
          + ", ".join(f"{float(np.abs(t['joint_motion_mrad']).max()):.2f}" for t in runs)
          + " mrad")
    print("signature, per seed: "
          + " | ".join(_signature(trace) for trace in runs))

    if args.trace is not None:
        print(f"trace   {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
