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

PROJECT = Path(__file__).resolve().parents[1]

#: The bowl's mesh body, not the wrapper the prop was placed at.  LIBERO authors every
#: object as a wrapper body with a mesh body nested 100 mm inside it (`pos="0 -0.1 0"`),
#: so the geometry -- and therefore the bowl a camera sees and a gripper can touch -- is
#: nowhere near the number in the scene config.
BOWL_MESH_BODY = "libero_bowl_object"
PLATE_MESH_BODY = "libero_plate_object"
FINGERTIPS = ("L_finger_inner_touch", "L_finger_outer_touch")

#: One unit of action, under the scene's `cartesian_translation_scale_m`.
ACTION_UNIT_M = 0.05

#: How near its starting height the bowl has to be to count as *resting on the table*
#: rather than held or airborne.  The table is rigid and the bowl does not deform, so this
#: is a noise allowance rather than a physical tolerance.
RESTING_TOL_M = 0.005

#: The eight corners of a box, for turning a geom's axis-aligned extent into world points.
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _under(model, root_id: int, body: int) -> bool:
    """Is ``body`` the body ``root_id`` or one of its descendants?  Bodies form a tree."""

    node = body
    while True:
        if node == root_id:
            return True
        if node == 0:
            return root_id == 0
        node = int(model.body_parentid[node])


def _prop_geoms(model, body_id: int) -> list[int]:
    """The geoms carried by a prop's body and everything under it."""

    return [index for index in range(model.ngeom)
            if _under(model, body_id, int(model.geom_bodyid[index]))]


def _prop_points(model, data, body_id: int) -> np.ndarray:
    """Every corner of a prop's geometry, in world coordinates.

    LIBERO props are a visual mesh *plus* forty-odd convex pieces from a collision
    decomposition of the same surface, so both have to be walked or the footprint is half
    an object.  Two traps are worth naming, because this probe fell into both of them and
    each one made the policy look worse than it was:

    * For a **mesh** geom, ``geom_aabb`` and ``geom_size`` hold the *unscaled* mesh
      extents.  The XML's ``scale="0.7 0.7 0.7"`` is already baked into the compiled
      vertex array, so ``mesh_vert`` is the only honest source -- reading the extents and
      scaling them again puts the bowl's rim at 39 mm when it is at 56.
    * LIBERO's objects all declare the same three sites: ``bottom_site`` at z=-0.06,
      ``top_site`` at +0.04, ``horizontal_radius_site`` at (0.025, 0.025).  Those are a
      template copied into every object, not a measurement of it: they describe a bowl
      100 mm tall with a 35 mm radius, while the mesh shipping beside them is 53 mm tall
      and 56 mm in radius.
    """

    points = []
    for index in _prop_geoms(model, body_id):
        if model.geom_type[index] == int(mujoco.mjtGeom.mjGEOM_MESH):
            start = int(model.mesh_vertadr[model.geom_dataid[index]])
            count = int(model.mesh_vertnum[model.geom_dataid[index]])
            local = np.asarray(model.mesh_vert[start:start + count])
        else:
            local = (np.asarray(model.geom_aabb[index][:3])
                     + _CORNERS * np.asarray(model.geom_aabb[index][3:]))
        points.append(data.geom_xpos[index]
                      + local @ data.geom_xmat[index].reshape(3, 3).T)
    return np.vstack(points)


def _prop_frame(model, data, body_id: int) -> tuple[np.ndarray, float, np.ndarray]:
    """A prop's centre, horizontal radius and overall size, all measured off its geometry.

    Measured rather than written down, so that "the bowl is over the plate" is answered
    against this scene's plate instead of a number remembered from somewhere else.
    """

    points = _prop_points(model, data, body_id)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(points[:, :2] - centre[:2], axis=1).max())
    return centre, radius, (high - low)


def _robot_bodies(model) -> set[int]:
    """Every body of the arm, found by walking down from its base rather than by name."""

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base < 0:
        raise SystemExit("no base_link; the contact check needs the arm's bodies")
    return {body for body in range(model.nbody) if _under(model, base, body)}


def _touching(model, data, prop_geoms: set[int], bodies: set[int]) -> bool:
    """Is the prop in contact with any geom whose body is in ``bodies``, right now?

    Read from MuJoCo's own contact list rather than inferred from proximity.  A fingertip
    3 mm from a wall is a different thing from a fingertip pressing on it, and only one of
    those is evidence that the policy ever got hold of anything -- or ever let go.
    """

    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        for near, far in ((first, second), (second, first)):
            if near in prop_geoms and int(model.geom_bodyid[far]) in bodies:
                return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_libero_props")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT / "configs/policies/pi05_libero_a3_cartesian.yaml")
    parser.add_argument("--task", default="pick up the black bowl and place it on the plate")
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
    the plate's own measured radius, it is touching the table, and the arm is not touching
    it.  The last two come from MuJoCo's contact list rather than from a height threshold,
    so "resting" and "still held" are observed instead of inferred.
    """

    over = float(trace["bowl_plate_mm"][-1]) <= plate_radius_m * 1000.0
    on_table = bool(trace["touch_table"][-1])
    held = bool(trace["touch_arm"][-1])
    if over and on_table and not held:
        return True, "PLACED"
    if held:
        return False, "not placed -- the arm is still on the bowl at the last step"
    if over and not on_table:
        return False, "not placed -- over the plate but off the table"
    return False, "not placed -- on the table, off the plate"


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
        if min(bowl, plate, *tips) < 0:
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
        bowl_start, bowl_radius_m, bowl_size_m = _prop_frame(model, env.data, bowl)
        plate_centre, plate_radius_m, _ = _prop_frame(model, env.data, plate)
        bowl_geoms = set(_prop_geoms(model, bowl))
        arm_bodies = _robot_bodies(model)
        bowl_top_z = float(bowl_start[2] + bowl_size_m[2] / 2.0)
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
        previous: np.ndarray | None = None
        for step in range(args.steps):
            action = gateway.act(observation, args.task)
            observation, _, terminated, truncated, info = env.step(action)
            applied = np.asarray(info["applied_action"], dtype=np.float64)
            # The model's own first row, before clipping.  Read from the gateway rather
            # than from `action`, which is the post-IK sixteen-wide joint target -- taking
            # its first three entries as metres is a factor of about fifty out.
            command = gateway.metrics.get("libero_command") or [0.0] * 7

            points = _prop_points(model, env.data, bowl)
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
            rows.append(
                {
                    "step": step,
                    "gap_mm": gap * 1000.0,
                    "radial_mm": radial * 1000.0,
                    "above_rim_mm": above_rim * 1000.0,
                    # MuJoCo's own contact list, not a proximity threshold, so "the arm
                    # touched the bowl" and "the bowl was on the table" are observations.
                    "touch_arm": float(_touching(model, env.data, bowl_geoms, arm_bodies)),
                    "touch_table": float(_touching(model, env.data, bowl_geoms, {0})),
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
                    "motion_mrad": 0.0 if previous is None
                    else float(np.abs(applied[:7] - previous[:7]).max()) * 1000.0,
                }
            )
            previous = applied
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
    """Six panels, one line colour per seed.

    Overlaid rather than averaged, because the spread *is* a finding: this policy samples
    fresh noise per chunk, and two rollouts at the same seed differ by more than the
    effect any of these numbers is being used to argue.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    figure, axes = plt.subplots(6, 1, figsize=(11, 17), sharex=True)

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
        axes[5].plot(step, trace["command_mm"], color=colour)

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

    axes[5].axhline(ACTION_UNIT_M * 1000, color="k", ls="--",
                    label=f"one action unit ({ACTION_UNIT_M * 1000:.0f} mm)")
    axes[5].set_ylabel("commanded Cartesian step (mm)")
    axes[5].set_xlabel("control step")
    axes[5].legend(loc="upper right", fontsize=8)
    axes[5].set_title("what the model asks for, before the clip", loc="left", fontsize=10)

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
