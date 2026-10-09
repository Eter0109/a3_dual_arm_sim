"""What the transit leg does for the placement, by taking it away and watching.

`MOVE_TO_SLOT` moves in two stages: to `q_transit` first, and then to the place pose.
The obvious question is why it does not go straight to the place pose, since that is
where it is going anyway, and the first stage is what carried the Cookie away from
the box on the Panda scene (measured: a 250 mm excursion against a 150 mm trip, see
`scripts/probe_carry_path.py`).

The answer is not "nothing" -- removing the stage makes the three configurations
that can be placed comfortably *faster* (clear of wall 155 -> 119 steps, two
cookies 356 -> 273) and breaks `a3 layout`, which is the one configuration where the
Cookie stands at A3's own bin position, the position the usable-region measurement
puts outside the region the dataset may draw from.  "It breaks one config" is not an
answer, so this prints what actually differs: the tool's error against the place pose
through the descent, and the joint configuration the descent starts from.

Usage::

    python scripts/probe_transit_leg.py --config a3 --skip
    python scripts/probe_transit_leg.py --config a3
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

#: Kept in step with `scripts/run_expert_gate.py`.
LAYOUTS = {
    "a3": dict(matched_cameras=True, a3_single_cookie=True, a3_layout_offset_m=(0.0, 0.0)),
    "clear": dict(
        matched_cameras=True,
        a3_single_cookie=True,
        a3_layout_offset_m=(0.020, -0.200),
        a3_cookie_position_m=(0.1194, 0.0711),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(LAYOUTS), default="a3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--skip", action="store_true", help="Never take the transit leg")
    parser.add_argument("--every", type=int, default=8)
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name="a3_libero_panda_single_cookie",
        hold_steps=40,
        panda=True,
        cartesian=True,
        **LAYOUTS[args.config],
    )
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    expert = policy.expert
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()

    if args.skip:
        # Reproduce the variant that was measured and rejected: the leg is never taken, so the arm
        # drives from the lift pose straight to the place pose.
        original = expert._advance

        def advance(phase, _original=original, _expert=expert):
            _original(phase)
            if phase.name == "MOVE_TO_SLOT":
                _expert._move_stage = "preplace"

        expert._advance = advance

    print(f"config {args.config}  skip_transit={args.skip}  seed {args.seed}")
    print("  step  phase            stage     pos err (mm)  rot err (deg)   joints 1-3 (rad)")

    warned = False
    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

        phase = expert.phase.name
        if phase == "DESCEND_TO_PLACE" and step % args.every == 0:
            # The three clauses of `aligned`, which is what actually ends this phase.  They are all
            # about the *Cookie*, not the tool, so a descent can be perfect in tool terms and still
            # never satisfy them.
            index = expert._current_cookie()
            live = expert.env.privileged_cookie_target_position(index)
            slot = np.asarray(expert.env.TARGET_SLOTS_LOCAL[expert._current_slot()])
            box_rotation = expert.data.xmat[expert._tb_id].reshape(3, 3)
            relative = box_rotation.T @ expert.data.xmat[expert.env._cookie_bodies[index]].reshape(3, 3)
            print(
                f"        cookie xy off slot {np.linalg.norm(live[:2] - slot) * 1000:7.3f} mm"
                f"   height off plan {abs(live[2] - expert._placement_height) * 1000:7.3f} mm"
                f"   tilt {np.degrees(np.arccos(np.clip(abs(relative[2, 2]), -1.0, 1.0))):6.2f} deg"
                f"   settle {expert._settle_count}"
            )
        if phase in ("MOVE_TO_SLOT", "DESCEND_TO_PLACE"):
            place = expert._waypoints.get("place")
            if place is not None:
                position_error, rotation_error = expert._goal_error(place)
                if step % args.every == 0 or not warned:
                    stage = expert._move_stage if phase == "MOVE_TO_SLOT" else ""
                    print(
                        f"  {step:4d}  {phase:16s} {stage:9s}"
                        f"  {np.linalg.norm(position_error) * 1000:11.3f}"
                        f"  {np.degrees(np.linalg.norm(rotation_error)):13.3f}"
                        f"   {np.round(expert.data.qpos[expert._l_qpos][:3], 3)}"
                    )
                    warned = True
        if terminated or truncated:
            break

    print(f"\n  final: phase={expert.phase.name} reason={expert.failure_reason}")
    print(f"  success={info.get('success')} steps={step}")
    for index in range(len(env._cookie_bodies)):
        body = env._cookie_bodies[index]
        print(
            f"  Cookie {index} at {np.round(env.data.xpos[body] * 1000, 2)} mm,"
            f" in target {env.privileged_cookie_in_target(index)}"
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
