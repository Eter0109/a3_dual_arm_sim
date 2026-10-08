"""Where, step by step, the LIBERO-interface policy stops short of the bowl.

The rollout's own summary says it failed and cannot say where.  `corrected 0.062`
and `motion 0.025` describe the *commands*; they are the same numbers a policy that
stood still would produce, because a command the environment accepts is not a command
that reaches anything.  What separates "it is doing the right thing badly" from "it is
doing something else" is the geometry: how far the fingertips get from the object, and
when they turn back.

So this records, every step:

* how the nearest fingertip sits against the bowl's **mouth** -- its distance from the
  bowl's vertical axis and its height against the rim.  A bare distance to the centre
  cannot tell "reaching into the bowl" from "hovering above it", and the two are opposite
  findings;
* whether the bowl is touching the arm and whether it is touching the table, which are the
  two facts that decide whether anything was ever picked up or put down;
* what the bowl did about the plate: its horizontal distance to the plate's centre, and its
  height above where it started, because a displacement magnitude cannot distinguish
  "carried to the plate" from "knocked off the table in the other direction";
* the gripper's opening, because this policy cycles it open and shut several times and
  a cycle that never closes on the object is the signature;
* the commanded Cartesian step in millimetres, against the 50 mm one unit of action
  means under this scene's OSC scale.

Usage::

    MUJOCO_GL=egl python scripts/probe_libero_props_trace.py --steps 600
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from a3_dual_arm_sim.props import (  # noqa: E402
    BOWL_MESH_BODY,
    EEF_SITE,
    FINGERTIPS,
    PLATE_MESH_BODY,
    arm_qpos_addresses,
    prop_frame,
    prop_geoms,
    prop_points,
    robot_bodies,
    touching,
)

PROJECT = Path(__file__).resolve().parents[1]

#: One unit of action, under the scene's `cartesian_translation_scale_m`.
ACTION_UNIT_M = 0.05

#: How near its starting height the bowl has to be to count as *resting on the table*
#: rather than held or airborne.  The table is rigid and the bowl does not deform, so this
#: is a noise allowance rather than a physical tolerance.
RESTING_TOL_M = 0.005

#: How many control steps the bowl has to hold still, at the end of an episode, before it
#: counts as put down rather than still being moved.  50 steps is 2.5 s.
SETTLED_STEPS = 50


#: LIBERO's own instruction for `libero_spatial` task 0, word for word.  It names three
#: objects and which one to take, and the scene now holds all three -- see the `libero_props`
#: block in `configs/libero_props.yaml`.  The simplified wording this used to carry
#: ("pick up the black bowl and place it on the plate") referred to a scene that had one bowl
#: and one plate in it, which is not the scene the policy was trained on.
LIBERO_TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_libero_props")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT / "configs/policies/pi05_libero_a3_cartesian.yaml")
    parser.add_argument("--task", default=LIBERO_TASK)
    parser.add_argument("--seeds", default="0,1,2",
                        help="The policy samples fresh flow-matching noise per chunk, so two "
                             "rollouts of the same seed differ by more than the effect being "
                             "measured -- one run is one sample.  Several seeds share one "
                             "model process.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--trace", type=Path,
                        default=Path("/root/autodl-tmp/logs/libero_props_trace.npz"))
    parser.add_argument("--figure", type=Path,
                        default=Path("/root/autodl-tmp/logs/videos/libero_props_trace.png"))
    return parser.parse_args()


def _grasp_timing(trace: dict[str, np.ndarray]) -> str:
    """Did the gripper close while it was on the bowl, or after it had left it?

    The order matters more than either number on its own.  A policy that shuts on the
    object and one that shuts a hundred steps after walking away from it produce the same
    grip-rate and the same "it touched the bowl" summary, while being different failures
    with different causes -- so the sequence is reported as a sentence rather than left to
    be reconstructed from two column maxima.
    """

    contact = np.flatnonzero(trace["touch_arm"] > 0)
    closed = np.flatnonzero(trace["gripper"] < 0.05)
    if not len(contact):
        return "the bowl was never touched"
    first_touch, last_touch = int(contact.min()), int(contact.max())
    if not len(closed):
        return f"touched on steps {first_touch}-{last_touch}, and the gripper never closed"
    first_close = int(closed[0])
    if first_close < first_touch:
        return "the gripper was already shut when it arrived"
    if first_close <= last_touch:
        return (f"the gripper closed on step {first_close}, while it was still on the bowl"
                f" (touched {first_touch}-{last_touch})")
    return (f"touched on steps {first_touch}-{last_touch}, but the gripper did not close "
            f"until step {first_close} -- {first_close - last_touch} steps after it had "
            f"left")


def _verdict(trace: dict[str, np.ndarray], plate_radius_m: float) -> tuple[bool, str]:
    """Was the bowl put on the plate, and if not, which part of it missed?

    A placement is three conditions at the last recorded step: the bowl's centre is within
    the plate's own measured radius, the arm is not touching it, and it is **settled** --
    it has not moved in the last ``SETTLED_STEPS`` steps.

    Settled is what does the work, and it replaced a test that was simply wrong.  The
    first version of this function asked for contact with the *table*, which is the one
    place a correctly placed bowl is not: it is standing on the plate, 15 mm up and
    touching nothing else.  That version reported a rollout which picked the bowl up,
    carried it to the plate and left it there as a failure, and it took a rendered video to
    notice -- the numbers were all consistent, they were just being read against a rule
    that could not be satisfied.  "Has not moved recently" catches resting on the plate,
    resting on the table, and being held or falling, without having to say which surface.
    """

    over = float(trace["bowl_plate_mm"][-1]) <= plate_radius_m * 1000.0
    held = bool(trace["touch_arm"][-1])
    moved = trace["bowl_mm"]
    settled = (len(moved) > SETTLED_STEPS
               and abs(float(moved[-1]) - float(moved[-1 - SETTLED_STEPS])) < 2.0)
    if over and not held and settled:
        return True, "PLACED"
    if over and held:
        return False, "not placed -- over the plate but still in the gripper"
    if over and not settled:
        return False, "not placed -- over the plate but still moving"
    return False, "not placed -- settled on the table, off the plate"


def main() -> int:
    args = parse_args()

    from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
    from a3_dual_arm_sim.bridge.manifest import load_manifest
    from a3_dual_arm_sim.collection import scene_by_name
    from a3_dual_arm_sim.contracts import STATE, EpisodeContext

    scene = scene_by_name(args.scene)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gateway = None
    runs: list[dict[str, np.ndarray]] = []
    for seed in seeds:
        env = scene.build_env(0.0, 0.0)
        env.use_fast_render()
        observation, _ = env.reset(seed=seed, options=None)

        model = env.model
        bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOWL_MESH_BODY)
        plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PLATE_MESH_BODY)
        tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FINGERTIPS]
        eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        if min(bowl, plate, eef, *tips) < 0:
            raise SystemExit(
                "the scene does not carry the LIBERO props; "
                "a3_libero_props is the one built for them"
            )

        # One model process for every seed.  `rebind` is what makes that legal: the
        # gateway caches values derived from the environment, and the environment is new
        # per episode.
        if gateway is None:
            gateway = ExternalPolicyGateway(load_manifest(args.manifest), env=env)
        else:
            gateway.rebind(env)
        gateway.reset(EpisodeContext(seed=seed, task=args.task, action_mode="joint_position"))

        # `geom_xpos` is derived, and a fresh reset has not necessarily been forwarded yet.
        mujoco.mj_forward(model, env.data)
        bowl_start, bowl_radius_m, bowl_size_m = prop_frame(model, env.data, bowl)
        plate_centre, plate_radius_m, _ = prop_frame(model, env.data, plate)
        bowl_geoms = set(prop_geoms(model, bowl))
        arm_bodies = robot_bodies(model)
        arm_qpos = arm_qpos_addresses(model)
        bowl_top_z = float(bowl_start[2] + bowl_size_m[2] / 2.0)
        # The task direction, as a unit vector on the table.  Every "the bowl moved N mm"
        # figure this probe has printed is a magnitude, and a magnitude whose *sign along
        # this axis* is unrecorded cannot say whether the policy is doing the task badly or
        # doing something else in the opposite direction.
        goal_xy = plate_centre[:2] - bowl_start[:2]
        goal_xy = goal_xy / np.linalg.norm(goal_xy)
        # Printed, not assumed.  Every number the verdict rests on is a measurement that
        # has to be checkable against the mesh, because the numbers this probe used before
        # -- a 39 mm rim and a centre 26 mm below the real one -- were both wrong.
        if seed == seeds[0]:
            print(
                f"bowl  centre {np.round(bowl_start, 4)}  radius {bowl_radius_m * 1000:.1f} mm  "
                f"mouth at z {bowl_top_z:.4f}, top of the mouth {bowl_size_m[2] * 1000:.1f} mm "
                f"above its centre\n"
                f"plate centre {np.round(plate_centre, 4)}  radius {plate_radius_m * 1000:.1f} mm\n"
                f"      bowl starts {np.linalg.norm(bowl_start[:2] - plate_centre[:2]) * 1000:.1f} mm "
                f"from the plate centre, so a placement is ~"
                f"{np.linalg.norm(bowl_start[:2] - plate_centre[:2]) * 1000:.0f} mm of travel\n"
                f"      a placement needs the bowl within {plate_radius_m * 1000:.0f} mm of the "
                f"plate centre, on the table and not in the gripper",
                flush=True,
            )

        rows = []
        previous_joints: np.ndarray | None = None
        for step in range(args.steps):
            action = gateway.act(observation, args.task)
            observation, _, terminated, truncated, _unused = env.step(action)
            joints = np.asarray(env.data.qpos[arm_qpos], dtype=np.float64)
            # The model's own first row, before clipping.  Read from the gateway rather
            # than from `action`, which is the post-IK sixteen-wide joint target -- taking
            # its first three entries as metres is a factor of about fifty out.
            command = gateway.metrics.get("libero_command") or [0.0] * 7

            points = prop_points(model, env.data, bowl)
            bowl_centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
            fingertips = [np.asarray(env.data.site_xpos[t]) for t in tips]
            gap = min(float(np.linalg.norm(tip - bowl_centre)) for tip in fingertips)
            # Against the *mouth*, not the centre.  The bowl is a hollow open-topped shell,
            # so "25 mm from the centre" describes a fingertip above the opening and one
            # reaching into the cavity equally well, and those are opposite findings.
            radial = min(
                float(np.linalg.norm(tip[:2] - bowl_centre[:2])) for tip in fingertips
            )
            above_rim = min(float(tip[2] - bowl_top_z) for tip in fingertips)
            eef_position = np.asarray(env.data.site_xpos[eef], dtype=np.float64)
            # The command is the model's own first row in normalised OSC units; the
            # translation half is world-frame, so its projection on the goal axis is the
            # travel the policy is *asking* for towards the plate, in metres.
            asked_xy = np.asarray(command[:2], dtype=np.float64) * ACTION_UNIT_M
            asked_norm = float(np.linalg.norm(asked_xy))
            toward_plate = float(np.dot(asked_xy, goal_xy))
            align_deg = (float(np.degrees(np.arccos(np.clip(
                np.dot(asked_xy / asked_norm, goal_xy), -1.0, 1.0))))
                if asked_norm > 1e-9 else float("nan"))
            rows.append(
                {
                    "step": step,
                    "eef_x_mm": float(eef_position[0] - bowl_start[0]) * 1000.0,
                    "eef_y_mm": float(eef_position[1] - bowl_start[1]) * 1000.0,
                    "eef_z_mm": float(eef_position[2] - bowl_start[2]) * 1000.0,
                    "toward_plate_mm": toward_plate * 1000.0,
                    "align_deg": align_deg,
                    "gap_mm": gap * 1000.0,
                    "radial_mm": radial * 1000.0,
                    "above_rim_mm": above_rim * 1000.0,
                    # MuJoCo's own contact list, not a proximity threshold, so "the arm
                    # touched the bowl" and "the bowl was on the table" are observations.
                    "touch_arm": float(touching(model, env.data, bowl_geoms, arm_bodies)),
                    "touch_table": float(touching(model, env.data, bowl_geoms, {0})),
                    "bowl_mm": float(np.linalg.norm(bowl_centre - bowl_start)) * 1000.0,
                    # The two numbers that decide whether the task happened at all, and
                    # which displacement alone cannot: where the bowl is against the plate,
                    # and whether it is on the table or in the air.
                    "bowl_plate_mm": float(
                        np.linalg.norm(bowl_centre[:2] - plate_centre[:2])
                    ) * 1000.0,
                    "bowl_height_mm": float(bowl_centre[2] - bowl_start[2]) * 1000.0,
                    "gripper": float(np.asarray(observation[STATE], dtype=np.float64)[7]),
                    "command_mm": float(np.abs(command[:3]).max()) * ACTION_UNIT_M * 1000.0,
                    "command_grip": float(command[6]),
                    # The working arm's **measured** joint motion per control step, in mrad.
                    # This field used to be `applied[:7]`, which is a joint target only for
                    # a joint-target arm: `env._limit_joint_action` returns those seven
                    # entries unclipped when an `OperationalSpaceAdapter` drives the arm,
                    # and under that adapter they are **torques in N.m**.  In the Panda
                    # scenes this field therefore held the step-to-step change of a torque
                    # vector and reported peaks of 17 rad per step against a 0.1 rad cap.
                    # Nothing read it, but a mislabelled number in a stored trace is a trap
                    # for whoever does read it next -- see F6 in `docs/bug-catalogue.md`.
                    "joint_motion_mrad": 0.0 if previous_joints is None
                    else float(np.abs(joints - previous_joints).max()) * 1000.0,
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
        reason = _verdict(trace, plate_radius_m)[1]
        print(
            f"seed {seed}: nearest fingertip {trace['gap_mm'][closest]:.1f} mm from the bowl "
            f"centre at step {closest}; {trace['radial_mm'][closest]:.1f} mm off its axis and "
            f"{trace['above_rim_mm'][closest]:+.1f} mm against the mouth "
            f"(mouth radius {bowl_radius_m * 1000:.0f} mm)  bowl moved "
            f"{trace['bowl_mm'].max():.2f} mm  gripper closed on "
            f"{int((trace['gripper'] < 0.05).sum())}/{len(rows)} steps  "
            f"arm on the bowl for {int(trace['touch_arm'].sum())}, on the table for "
            f"{int(trace['touch_table'].sum())}\n"
            f"        bowl to plate centre {trace['bowl_plate_mm'][0]:.1f} -> "
            f"{trace['bowl_plate_mm'][-1]:.1f} mm (best "
            f"{trace['bowl_plate_mm'].min():.1f}), height "
            f"{trace['bowl_height_mm'].max():+.1f} peak / "
            f"{trace['bowl_height_mm'][-1]:+.1f} at the end  ->  {reason}\n"
            f"        {_grasp_timing(trace)}",
            flush=True,
        )

    if gateway is not None:
        gateway.close()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.trace, **{f"{key}_s{index}": trace[key]
                            for index, trace in enumerate(runs) for key in trace})

    closest = [float(trace["gap_mm"].min()) for trace in runs]
    print()
    print(f"nearest fingertip to the bowl centre, over {len(runs)} seeds: "
          + ", ".join(f"{value:.1f}" for value in closest)
          + f" mm   (min {min(closest):.1f}, spread {max(closest) - min(closest):.1f})")
    print(f"nearest fingertip against the mouth (axis distance / height vs the rim), "
          f"per seed: "
          + ", ".join(f"{float(t['radial_mm'].min()):.1f}/{float(t['above_rim_mm'].min()):+.1f}"
                      for t in runs)
          + f" mm   (mouth radius {bowl_radius_m * 1000:.0f} mm)")
    print(f"bowl displaced, per seed: "
          + ", ".join(f"{float(t['bowl_mm'].max()):.2f}" for t in runs) + " mm")
    print("steps with the arm touching the bowl, per seed: "
          + ", ".join(f"{int(t['touch_arm'].sum())}/{len(t['touch_arm'])}" for t in runs))
    # Displacement is a magnitude, and a magnitude cannot tell "carried to the plate" from
    # "knocked off the table in the other direction".  These two can.
    print(f"bowl to plate centre at the last step, per seed: "
          + ", ".join(f"{float(t['bowl_plate_mm'][-1]):.1f}" for t in runs)
          + f" mm (plate radius {plate_radius_m * 1000:.0f} mm);  best "
          + ", ".join(f"{float(t['bowl_plate_mm'].min()):.1f}" for t in runs))
    print("verdict, per seed: "
          + ", ".join(_verdict(trace, plate_radius_m)[1].split(" --")[0] for trace in runs))
    # The direction the policy *asks* for, which no displacement magnitude can show.  A
    # sum near zero means it never intends to cross the table at all, whatever the bowl
    # ended up doing; a negative one means it is asking to go the other way.
    print("commanded travel towards the plate, summed, per seed: "
          + ", ".join(f"{float(t['toward_plate_mm'].sum()):+.0f}" for t in runs)
          + f" mm   (the task needs about +"
            f"{float(np.linalg.norm(plate_centre[:2] - bowl_start[:2])) * 1000:.0f} mm)")
    print("mean angle between the commanded translation and the bowl->plate direction: "
          + ", ".join(f"{float(np.nanmean(t['align_deg'])):.0f}" for t in runs)
          + " deg   (0 = straight at the plate, 90 = across it, 180 = straight away)")
    print("left eef at the last step, relative to the bowl (x, y, z in mm): "
          + ", ".join(f"({float(t['eef_x_mm'][-1]):+.0f}, {float(t['eef_y_mm'][-1]):+.0f}, "
                      f"{float(t['eef_z_mm'][-1]):+.0f})" for t in runs))

    _plot(runs, args.figure, plate_radius_m, bowl_radius_m)
    print(f"trace   {args.trace}")
    print(f"figure  {args.figure}")
    return 0


def _cycles(gripper: np.ndarray) -> int:
    """How many times the gripper went from open to closed and back."""

    closed = gripper < 0.05
    return int(np.count_nonzero(np.diff(closed.astype(int)) == 1))


def _plot(
    runs: list[dict[str, np.ndarray]],
    destination: Path,
    plate_radius_m: float,
    bowl_radius_m: float,
) -> None:
    """Eight panels, one line colour per seed.

    Overlaid rather than averaged, because the spread *is* a finding: this policy samples
    fresh noise per chunk, and two rollouts at the same seed differ by more than the
    effect any of these numbers is being used to argue.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    figure, axes = plt.subplots(8, 1, figsize=(11, 22), sharex=True)

    for index, trace in enumerate(runs):
        colour = colours[index % len(colours)]
        step = trace["step"]
        axes[0].plot(step, trace["radial_mm"], color=colour, label=f"seed {index}")
        axes[1].plot(step, trace["above_rim_mm"], color=colour)
        axes[2].plot(step, trace["bowl_mm"], color=colour)
        axes[2].plot(step, trace["bowl_plate_mm"], color=colour, ls=":", alpha=0.85)
        axes[3].plot(step, trace["bowl_height_mm"], color=colour)
        axes[4].plot(step, trace["gripper"], color=colour)
        axes[4].fill_between(step, 0.0, 1.0, where=trace["touch_arm"] > 0,
                             color=colour, alpha=0.13, linewidth=0)
        axes[5].plot(step, trace["eef_y_mm"], color=colour)
        axes[5].plot(step, trace["eef_x_mm"], color=colour, ls=":", alpha=0.8)
        axes[6].plot(step, np.cumsum(trace["toward_plate_mm"]), color=colour)
        axes[7].plot(step, trace["command_mm"], color=colour)

    axes[0].axhline(bowl_radius_m * 1000, color="k", ls="--",
                    label=f"mouth radius ({bowl_radius_m * 1000:.0f} mm)")
    axes[0].set_ylabel("distance from the"
                       "\nbowl's axis (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("the nearest fingertip, measured across the bowl's mouth rather "
                      "than to its centre", loc="left", fontsize=10)

    axes[1].axhline(0.0, color="k", ls="--", label="the rim")
    axes[1].set_ylabel("fingertip against\nthe rim (mm)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("the same fingertip's height against the rim -- below zero is at the "
                      "height of the bowl's opening, or under it", loc="left", fontsize=10)

    axes[2].axhline(plate_radius_m * 1000, color="k", ls="--", lw=1,
                    label=f"plate radius ({plate_radius_m * 1000:.0f} mm)")
    axes[2].set_ylabel("bowl (mm)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("how far the bowl has moved (solid), and how far it still is from "
                      "the plate centre (dotted)", loc="left", fontsize=10)

    axes[3].axhline(0.0, color="k", ls="--", lw=1)
    axes[3].axhspan(-RESTING_TOL_M * 1000, RESTING_TOL_M * 1000, color="k", alpha=0.12)
    axes[3].set_ylabel("bowl height (mm)")
    axes[3].set_title(f"the bowl's height above where it started; the shaded band is the "
                      f"+-{RESTING_TOL_M * 1000:.0f} mm of resting on the table",
                      loc="left", fontsize=10)

    axes[4].axhline(0.05, color="k", ls="--", label="closed")
    axes[4].set_ylabel("gripper opening")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].set_title("the gripper's opening, with the steps the bowl is in contact with "
                      "the arm shaded", loc="left", fontsize=10)

    axes[5].axhline(0.0, color="k", ls="--", lw=1, label="the bowl's axis")
    axes[5].set_ylabel("eef relative to\nthe bowl (mm)")
    axes[5].legend(loc="upper right", fontsize=8)
    axes[5].set_title("where the tool actually goes: along the bowl-to-plate axis (solid), "
                      "and across it (dotted)", loc="left", fontsize=10)

    axes[6].axhline(0.0, color="k", ls="--", lw=1)
    axes[6].set_ylabel("asked for, towards\nthe plate (mm)")
    axes[6].set_title("the travel the policy asks for, accumulated along the bowl-to-plate "
                      "direction", loc="left", fontsize=10)

    axes[7].axhline(ACTION_UNIT_M * 1000, color="k", ls="--",
                    label=f"one action unit ({ACTION_UNIT_M * 1000:.0f} mm)")
    axes[7].set_ylabel("commanded Cartesian step (mm)")
    axes[7].set_xlabel("control step")
    axes[7].legend(loc="upper right", fontsize=8)
    axes[7].set_title("what the model asks for, before the clip", loc="left", fontsize=10)

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
