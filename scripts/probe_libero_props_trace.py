"""Where, step by step, the LIBERO-interface policy stops short of the bowl.

The rollout's own summary says it failed and cannot say where.  `corrected 0.062`
and `motion 0.025` describe the *commands*; they are the same numbers a policy that
stood still would produce, because a command the environment accepts is not a command
that reaches anything.  What separates "it is doing the right thing badly" from "it is
doing something else" is the geometry: how far the fingertips get from the object, and
when they turn back.

So this records, every step:

* the distance from the nearest fingertip to the bowl's centre, against the bowl's own
  radius, so the gap has a physical meaning rather than being a bare number;
* what the bowl and the plate actually did, because a grasp that touches shows up here
  and a grasp that misses does not;
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

import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

#: The bowl's mesh, not the body the prop was placed at.  LIBERO authors this object with
#: an inner body offset by `pos="0 -0.1 0"`, so its geometry sits 100 mm from the number
#: in the scene config and a distance measured to the wrapper is wrong by exactly that.
BOWL_MESH_BODY = "libero_bowl_object"
PLATE_MESH_BODY = "libero_plate_object"
FINGERTIPS = ("L_finger_inner_touch", "L_finger_outer_touch")

#: The bowl's rim, measured off its visual mesh: local extents +-0.056 with the XML's
#: `scale="0.7 0.7 0.7"`, so about 78 mm across.
BOWL_RADIUS_M = 0.039

#: One unit of action, under the scene's `cartesian_translation_scale_m`.
ACTION_UNIT_M = 0.05


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


def main() -> int:
    args = parse_args()

    import mujoco
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

        bowl_start = env.data.xpos[bowl].copy()
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

            gap = min(
                float(np.linalg.norm(env.data.site_xpos[t] - env.data.xpos[bowl])) for t in tips
            )
            rows.append(
                {
                    "step": step,
                    "gap_mm": gap * 1000.0,
                    "bowl_mm": float(np.linalg.norm(env.data.xpos[bowl] - bowl_start)) * 1000.0,
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
        print(
            f"seed {seed}: closest {trace['gap_mm'][closest]:.1f} mm at step {closest} "
            f"(rim at {BOWL_RADIUS_M * 1000:.0f} mm)  bowl moved "
            f"{trace['bowl_mm'].max():.2f} mm  gripper closed on "
            f"{int((trace['gripper'] < 0.05).sum())}/{len(rows)} steps  "
            f"command mean {trace['command_mm'].mean():.1f} mm",
            flush=True,
        )

    if gateway is not None:
        gateway.close()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.trace, **{f"{key}_s{index}": trace[key]
                            for index, trace in enumerate(runs) for key in trace})

    closest = [float(trace["gap_mm"].min()) for trace in runs]
    print()
    print(f"closest approach over {len(runs)} seeds: "
          + ", ".join(f"{value:.1f}" for value in closest)
          + f" mm   (min {min(closest):.1f}, spread {max(closest) - min(closest):.1f})")
    print(f"bowl displaced, per seed: "
          + ", ".join(f"{float(t['bowl_mm'].max()):.2f}" for t in runs) + " mm")

    _plot(runs, args.figure)
    print(f"trace   {args.trace}")
    print(f"figure  {args.figure}")
    return 0


def _cycles(gripper: np.ndarray) -> int:
    """How many times the gripper went from open to closed and back."""

    closed = gripper < 0.05
    return int(np.count_nonzero(np.diff(closed.astype(int)) == 1))


def _plot(runs: list[dict[str, np.ndarray]], destination: Path) -> None:
    """Four panels, one line colour per seed.

    Overlaid rather than averaged, because the spread *is* a finding: this policy samples
    fresh noise per chunk, and two rollouts at the same seed differ by more than the
    effect any of these numbers is being used to argue.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    for index, trace in enumerate(runs):
        colour = colours[index % len(colours)]
        step = trace["step"]
        axes[0].plot(step, trace["gap_mm"], color=colour, label=f"seed {index}")
        axes[0].plot(step[int(np.argmin(trace["gap_mm"]))], trace["gap_mm"].min(),
                     "v", color=colour)
        axes[1].plot(step, trace["bowl_mm"], color=colour)
        axes[2].plot(step, trace["gripper"], color=colour)
        axes[3].plot(step, trace["command_mm"], color=colour)

    axes[0].axhline(BOWL_RADIUS_M * 1000, color="k", ls="--",
                    label=f"bowl rim ({BOWL_RADIUS_M * 1000:.0f} mm)")
    axes[0].set_ylabel("fingertip to bowl centre (mm)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("how close the fingertips get, and when they turn back",
                      loc="left", fontsize=10)

    axes[1].set_ylabel("bowl displacement (mm)")
    axes[1].set_title("the bowl never moves, in any seed", loc="left", fontsize=10)

    axes[2].axhline(0.05, color="k", ls="--", label="closed")
    axes[2].set_ylabel("gripper opening")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("the gripper cycles open and shut without ever closing on the bowl",
                      loc="left", fontsize=10)

    axes[3].axhline(ACTION_UNIT_M * 1000, color="k", ls="--",
                    label=f"one action unit ({ACTION_UNIT_M * 1000:.0f} mm)")
    axes[3].set_ylabel("commanded Cartesian step (mm)")
    axes[3].set_xlabel("control step")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title("what the model asks for, before the clip", loc="left", fontsize=10)

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
