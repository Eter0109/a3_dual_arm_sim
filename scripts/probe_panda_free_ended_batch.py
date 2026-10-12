"""How large a batch can LIBERO's Panda take, and what stops it?

`a3_libero_panda_dense_batch` asks for five Cookies at a time out of the 4 x 20 packing, and the
answer there is **none at all** -- not five, not one.  The reason is a length rather than a
controller: the jaw's contact face is 12.45 mm from the finger geom origin the expert measures
separation between, the finger body's mesh reaches ~27 mm past that, and the packing leaves
**2.50 mm** in the groove between two Cookies and **3.3 / 15.1 mm** between a column's two ends
and the bin wall.  Measured on step 18 of `DESCEND`:

    L_finger_outer_geom <-> cookie_5  dist -0.36 mm  64.3 N
    L_finger_outer_geom <-> cookie_6  dist -0.38 mm  83.0 N

-- one-sided, on the *neighbours* of the batch, with the other jaw at zero.

That is a statement about the hand and the packing together, so it is testable by moving only the
packing: this runs the same scene, arm, cameras, physics block, expert, batch size and target box,
with ten Cookies laid out as **two columns of five** and the columns placed where a five-run's
**ends are clear**.  It is the same experiment `a3_libero_panda_two_cookies` already passes for a
single piece (2/2, this same jaw), one batch size up.

    python scripts/probe_panda_free_ended_batch.py
    python scripts/probe_panda_free_ended_batch.py --low-end-only   # reproduce the first failure

What the two ends say
---------------------
* columns left where the dense layout puts them (low end 3.3 mm from the wall) -- `DESCEND` jams
  one-sided at **64.9 N on the low pad alone**, and the contact names the obstruction rather than
  the batch: `source_bin_right_wall <-> L_finger_inner_geom`, `dist -0.86 mm`, maximum penetration
  1.52 mm.  The jaw's *body* is on the wall, and nothing about the Cookies is involved;
* columns moved to the middle of the bin (low end 73.0 mm) -- the pick **succeeds**: the five are
  grasped, lifted and carried (`in source: 5`), and the run then stops at `DESCEND_TO_PLACE`
  because the target box is 45.3 mm wide across the jaws while the batch plus two 27 mm jaws needs
  ~96 mm.  A3 places into that same box because its fingertip is 1.00 mm thick:
  41.67 + 2 x 1.0 = 43.7 mm.

So the price of five-at-a-time on this hand is stated in two numbers: a five-run with clear ends
to pick from, and a box wider than the jaws to place into.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.collection import dense_cookie_layout, libero_props_scene
from a3_dual_arm_sim.runner import EpisodeRunner

#: The row pitch of the dense packing, which is the pitch this probe keeps.
PITCH_M = 0.008833333333333333
#: The source bin's own interior along y, from `source_bin_center_m` and `source_bin_half_size_m`
#: of `configs/libero_props.yaml`.  A five-run's ends are measured against this.
BIN_Y_M = (0.0199, 0.2101)
#: Column 0's first and last row, read off the dense layout so the x's are the packing's own.
COLUMNS = (0, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2600)
    parser.add_argument(
        "--low-end-only",
        action="store_true",
        help="leave the columns where the dense layout puts them, to reproduce the 64.9 N jam",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = dense_cookie_layout((0.020, -0.200))
    # The columns' x from the dense layout; their y either the layout's own first five rows or the
    # same five rows moved to the middle of the bin.
    if args.low_end_only:
        rows = [layout[index][1] for index in range(5)]
    else:
        centre_y = 0.5 * (BIN_Y_M[0] + BIN_Y_M[1])
        rows = [centre_y + (index - 2) * PITCH_M for index in range(5)]
    columns = [layout[column * 20][0] for column in COLUMNS]
    positions = tuple((x, y) for x in columns for y in rows)

    print(f"ten Cookies, {len(columns)} columns of five:")
    for x, y in positions:
        print(f"  x {x:.4f}  y {y:.4f}")
    print(
        "  a five-run's low end: "
        f"{(min(rows) - PITCH_M / 2 - BIN_Y_M[0]) * 1000:.1f} mm of clear y to the bin wall"
    )
    print(flush=True)

    scene = libero_props_scene(
        task_name="a3_libero_panda_dense_batch",
        matched_cameras=True,
        panda=True,
        a3_single_cookie=True,
        a3_layout_offset_m=(0.020, -0.200),
        a3_cookie_positions_m=positions,
        batch_expert=True,
        libero_physics=False,
    )
    env = scene.build_env(0.0, 0.0)
    # No part of the physics or the expert reads a pixel, and the probe is run headless when the
    # GPU is held by another job.
    env.render_cameras = False
    policy = scene.build_policy(env)
    runner = EpisodeRunner(
        env, policy, task=scene.task.prompt, recorded_action="requested", save_failed_episodes=False
    )
    result = runner.run(seed=args.seed, max_steps=args.max_steps)
    expert = policy.expert

    print()
    print("success    :", result.success, " (the dense contract's 70-remain cannot hold on ten)")
    print("steps      :", result.steps, " safety:", result.safety_reason)
    print("ended in   :", expert.phase.name, " failure:", expert.failure_reason)
    print(
        "in target  :",
        result.final_info.get("cookies_in_target"),
        " in source:",
        result.final_info.get("cookies_in_source"),
    )
    print("phase path :", " -> ".join(item.name for item in expert.transition_history))
    print(f"max pad force: {expert.max_pad_force:.2f} N")
    print(f"max contact penetration: {expert.max_contact_penetration_m * 1000:.4f} mm")

    model, data = env.model, env.data
    mujoco.mj_forward(model, data)
    jaw = {int(geom) for jaw_set in env._left_jaw_geoms for geom in jaw_set}
    print(f"\ncontacts on a jaw at the end ({data.ncon} in the model):")
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in jaw and second not in jaw:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)

        def name(geom: int) -> str:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"

        print(
            f"  {name(first):<30} <-> {name(second):<30}"
            f"  dist {float(contact.dist) * 1000:+6.2f} mm"
            f"  |fn| {float(np.linalg.norm(force[:3])):7.2f} N"
            f"  y {float(contact.pos[1]):+0.4f} z {float(contact.pos[2]):0.4f}"
        )


if __name__ == "__main__":
    main()
