"""A three-block stacking task, as a control for the cookie failure.

The cookie scene asks for something unusual: five 50x25x6.3 mm pieces inserted
upright into 2.5 mm gaps, ten times, over 540 steps.  Every measurement so far says
the fine-tuned policies fail it because their per-step error exceeds the expert's own
step size, so the question this exists to answer is whether that is the *task's*
fault or the *system's*.  Three blocks of 70x70x80 mm, stacked, is the opposite end
of that scale: a task that appears in essentially every robot-learning dataset, with
no millimetre clearances and about a third of the steps.

The scene ships with three differently coloured blocks already -- object_0 red,
object_1 blue, object_2 green -- but only the red one is placed in the task config's
workspace; the other two sit at y = -0.48 m, in the right arm's half of the table and
outside what the left arm reaches.  So this repositions all three into one row the
left arm can reach, which is what makes them stackable by a single arm at all.

Nothing here changes the scene or the datasets.  It is a probe: it builds the
environment, moves three blocks, drives the model, and reports what happened.

Success is measured, not declared.  The scene's own contract is "lift the red cube
80 mm", which a stack does not satisfy, so the outcome is read off the simulator:
for each ordered pair of blocks, is one resting above the other with their footprints
overlapping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway  # noqa: E402
from a3_dual_arm_sim.bridge.manifest import load_manifest  # noqa: E402
from a3_dual_arm_sim.contracts import EpisodeContext  # noqa: E402
from a3_dual_arm_sim.grasp import A3GraspEnv, GraspTaskConfig  # noqa: E402

MANIFESTS = {
    "base": Path("/root/a3_dual_arm_sim/configs/policies/pi05_base.yaml"),
    "b1": Path("/root/a3_dual_arm_sim/configs/policies/pi05_a3_finetuned.yaml"),
    "adv": Path("/root/a3_dual_arm_sim/configs/policies/pi05_a3_adv40k.yaml"),
}
STACK_PROMPTS = {
    "stack": "stack the three blocks",
    "on": "put the red block on the green block",
    "pile": "pick up the blocks and stack them on top of each other",
}

#: Block half-sizes, read from the model rather than assumed.  The three are identical.
BLOCK_HALF = np.array([0.035, 0.035, 0.040])
TABLE_Z = 0.750
BLOCK_Z = TABLE_Z + BLOCK_HALF[2]  # resting centre height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--which", choices=sorted(MANIFESTS), default="base")
    parser.add_argument("--manifest", type=Path, default=None, help="overrides --which")
    parser.add_argument("--task", default=STACK_PROMPTS["stack"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layout",
        choices=("row", "original"),
        default="row",
        help="'row' puts all three blocks in front of the left arm; 'original' leaves them",
    )
    parser.add_argument(
        "--grasp-hold",
        action="store_true",
        help="Use the scene's own benchmark: lift the red cube.  Contrast, not the probe.",
    )
    return parser.parse_args()


def place_blocks(env: A3GraspEnv, layout: str) -> list[np.ndarray]:
    """Put the three blocks where the left arm can reach them, and return their homes.

    The row is along y because that is the direction the left arm sweeps: its reach
    was measured at x in [-0.93, 0.23], y in [-0.13, 0.78] at table height, and the
    red block's original spot (0.127, 0.468) -- which the scene's own expert uses --
    sits in the dense part of it.  90 mm of pitch clears the 70 mm blocks with 20 mm
    between them, so a block can be picked without disturbing its neighbour.
    """

    if layout == "original":
        return [env.data.xpos[env._object_body].copy()]

    home = np.array([0.127, 0.468, BLOCK_Z])
    placements = [
        home + [0.0, -0.090, 0.0],  # red, nearest the arm's own working line
        home,
        home + [0.0, +0.090, 0.0],  # green, furthest
    ]
    for index, position in enumerate(placements):
        env._set_free_object(index, tuple(position))
    for _ in range(120):
        import mujoco

        mujoco.mj_step(env.model, env.data)
    env.data.time = 0.0
    return placements


def block_centres(env: A3GraspEnv) -> np.ndarray:
    return np.array([env.data.xpos[b].copy() for b in env._block_bodies])


def stacked_pairs(centres: np.ndarray) -> list[tuple[int, int]]:
    """(lower, upper) pairs where ``upper`` rests on ``lower``.

    Two conditions, both read from the simulator: the upper block's centre is about
    one block-height above the lower one's, and their footprints overlap.  A block
    that merely passed overhead on its way somewhere else does not count, which is
    why the height window is narrow rather than "higher than".
    """

    found: list[tuple[int, int]] = []
    for low in range(3):
        for high in range(3):
            if low == high:
                continue
            rise = centres[high][2] - centres[low][2]
            if not (0.5 * BLOCK_HALF[2] < rise < 1.6 * BLOCK_HALF[2]):
                continue
            overlap = np.abs(centres[high][:2] - centres[low][:2]) < BLOCK_HALF[:2]
            if bool(overlap.all()):
                found.append((low, high))
    return found


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest or MANIFESTS[args.which])

    env = A3GraspEnv(
        config=None,
        task_config=GraspTaskConfig(position_noise_m=0.0),
        action_mode="joint_position",
        render_cameras=True,
    )
    env._block_bodies = tuple(
        env._id(__import__("mujoco").mjtObj.mjOBJ_BODY, f"object_{i}") for i in range(3)
    )
    tool = env._id(__import__("mujoco").mjtObj.mjOBJ_BODY, "L_robotiq_2f85")

    gateway = ExternalPolicyGateway(manifest, env=env, keep_transport=True)
    print(f"model : {manifest.name}")
    print(f"task  : {args.task}")
    print(f"blocks: {'a reachable row' if args.layout == 'row' else 'their default spots'}")
    print()

    try:
        gateway.rebind(env)
        observation, _ = env.reset(seed=args.seed, options=None)
        homes = place_blocks(env, args.layout)
        observation = env._observation()

        start = block_centres(env)
        print("start centres:")
        for name, centre in zip(("red", "blue", "green"), start):
            print(f"  {name:6s} {np.round(centre, 4)}")
        print()

        gateway.reset(
            EpisodeContext(seed=args.seed, task=args.task, action_mode="joint_position")
        )

        ee_path: list[np.ndarray] = []
        nearest: list[float] = []
        highest: list[float] = []
        rows: list[dict[str, Any]] = []

        for step in range(1, args.steps + 1):
            action = gateway.act(observation, args.task)
            observation, _, terminated, truncated, info = env.step(action)
            centres = block_centres(env)
            ee = env.data.xpos[tool].copy()
            ee_path.append(ee)
            gap = float(np.min(np.linalg.norm(centres - ee, axis=1)))
            nearest.append(gap)
            highest.append(float((centres[:, 2] - start[:, 2]).max()))
            if step % 40 == 0 or step == args.steps:
                rows.append({"step": step, "nearest_mm": gap * 1000, "lift_mm": highest[-1] * 1000})
            if args.grasp_hold and info.get("success"):
                break
            if terminated or truncated:
                break

        final = block_centres(env)
        stacked = stacked_pairs(final)
        ee_path = np.array(ee_path)
        travel = float(np.abs(np.diff(ee_path, axis=0)).sum()) if len(ee_path) > 1 else 0.0

        print(f"ran {len(ee_path)} steps")
        print()
        print("  step   nearest block   highest lift")
        for row in rows:
            print(f"  {row['step']:5d}   {row['nearest_mm']:8.1f} mm   {row['lift_mm']:8.1f} mm")
        print()
        print(f"  end-effector travel        : {travel * 1000:.0f} mm")
        print(f"  closest approach to a block: {min(nearest) * 1000:.1f} mm  "
              f"(at step {int(np.argmin(nearest)) + 1})")
        print(f"  closest approach at the end: {nearest[-1] * 1000:.1f} mm")
        print(f"  greatest lift of any block : {max(highest) * 1000:.1f} mm")
        print()
        print("end centres:")
        for name, centre in zip(("red", "blue", "green"), final):
            moved = np.linalg.norm(centre - start[["red", "blue", "green"].index(name)])
            print(f"  {name:6s} {np.round(centre, 4)}   moved {moved * 1000:.1f} mm")
        print()
        if stacked:
            print(f"  STACKED: {[(('red','blue','green')[a], ('red','blue','green')[b]) for a, b in stacked]}")
        else:
            print("  no block is resting on another")
        print()
        print(f"  blocks reachable from the arm's own line: {'yes' if args.layout == 'row' else 'no'}")
        print(f"  first block home was {np.round(homes[0], 4)}")
    finally:
        gateway.shutdown()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
