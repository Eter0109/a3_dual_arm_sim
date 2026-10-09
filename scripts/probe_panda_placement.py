"""Which clause of "the Cookie is in the target box" fails after the job looks finished?

The Panda now walks the whole state machine -- grasp, lift, transfer, place, release all verify --
and dies at the last check with "released Cookie did not remain upright and stable in target".  That
sentence is one message over five clauses, and they call for different work:

    footprint_inside   the Cookie is over the box's opening
    vertically_inside  it is between the floor and the wall top, and (with `require_released`) down
                       on the floor rather than perched
    upright            within `max_tilt_rad` of vertical, in the box's frame or the world's
    settled            moving slower than the thresholds, **relative to the box**
    released           the jaws are not touching it any more

`released` is worth watching closely here, because it is the one clause this session changed:
`privileged_left_finger_contacts` used to look at a single named geom per jaw and now covers the
jaw's whole chain, so it can report a contact the old query could not see.  A jaw resting against a
Cookie it has just let go of would therefore fail a check that used to pass -- which would make this
failure mine rather than the scene's, and the two need telling apart.

The clauses are recomputed here rather than reached for, because the environment's predicate answers
with one bool.  To keep that from drifting, the recomputation is compared against the real predicate
every step and any disagreement is printed as such.
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from a3_dual_arm_sim.collection import libero_props_scene

BASE = dict(
    matched_cameras=True,
    a3_single_cookie=True,
    a3_layout_offset_m=(0.020, -0.200),
    cartesian=True,
    panda=True,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear-of-wall", action="store_true")
    args = parser.parse_args()

    layout = dict(BASE)
    if args.clear_of_wall:
        layout["a3_cookie_position_m"] = (0.1194, 0.0711)
    scene = libero_props_scene(task_name="a3_libero_panda_single_cookie", hold_steps=40, **layout)
    env = scene.build_env(0.0, 0.0)
    policy = scene.build_policy(env)
    policy.reset()
    expert = policy.expert
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()

    print(
        f"clear_of_wall={args.clear_of_wall}  require_released={env.task_config.require_released}"
        f"  require_upright={env.task_config.require_upright}"
        f"  max_tilt {np.rad2deg(env.task_config.max_tilt_rad):.0f} deg"
    )
    print(
        f"  box inner half {np.round(env.TARGET_INNER_HALF_SIZE * 1000, 2)} mm"
        f"   wall height {env.TARGET_WALL_HEIGHT * 1000:.1f} mm"
    )
    reported = False
    for step in range(1, args.steps + 1):
        action = expert.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if expert.phase.name != "RETRACT":
            if terminated or truncated:
                print(f"  ended at step {step} terminated={terminated} truncated={truncated}")
                break
            continue
        index = expert._current_cookie()
        if not reported:
            reported = True
            print(f"\ndynamic: the Cookie has been released, phase RETRACT (from step {step})")
        elif step % args.every:
            continue

        box_position = np.asarray(env.data.xpos[env._target_bin_body], dtype=np.float64)
        box_rotation = np.asarray(env.data.xmat[env._target_bin_body]).reshape(3, 3)
        cookie_position = np.asarray(env.data.xpos[env._cookie_bodies[index]], dtype=np.float64)
        cookie_rotation = env._cookie_rotation(index)
        relative_position = box_rotation.T @ (cookie_position - box_position)
        relative_rotation = box_rotation.T @ cookie_rotation
        half_extent = np.abs(relative_rotation) @ env.COOKIE_HALF_SIZE

        floor_top = (
            env.config.cookie_transfer.target_floor_z_m
            + env.config.cookie_transfer.bin_wall_thickness_m / 2
        )
        footprint = bool(
            np.all(
                np.abs(relative_position[:2]) + half_extent[:2]
                <= env.TARGET_INNER_HALF_SIZE + env.CONTACT_CONTAINMENT_TOLERANCE_M
            )
        )
        bottom = float(relative_position[2] - half_extent[2])
        vertical = bool(bottom >= floor_top - 0.003 and bottom < env.TARGET_WALL_HEIGHT)
        if env.task_config.require_released:
            vertical = bool(vertical and bottom <= floor_top + 0.004)
        upright = bool(
            abs(float(relative_rotation[2, 2])) >= np.cos(env.task_config.max_tilt_rad)
            or abs(float(cookie_rotation[2, 2])) >= np.cos(env.task_config.max_tilt_rad)
        )
        linear, angular = env._cookie_speed(index, relative_to_target=True)
        settled = bool(
            linear <= max(env.task_config.max_linear_speed_m_s, 0.09)
            and angular <= env.task_config.max_angular_speed_rad_s
        )
        contacts = env.privileged_left_finger_contacts(index)
        released = not any(contacts)
        mine = footprint and vertical and upright and settled and released
        theirs = env.privileged_cookie_in_target(index)
        print(
            f"  {step:4d} verdict {theirs!s:5s} (recomputed {mine!s:5s}"
            f"{'' if mine == theirs else '  <<< DISAGREES'})  "
            f"footprint {footprint!s:5s} vertical {vertical!s:5s} upright {upright!s:5s} "
            f"settled {settled!s:5s} released {released!s:5s}"
        )
        print(
            f"        Cookie in box frame {np.round(relative_position * 1000, 2)} mm"
            f"  half-extent {np.round(half_extent * 1000, 2)}"
            f"  bottom {bottom * 1000:7.2f} mm vs floor {floor_top * 1000:.2f}"
            f"  tilt {np.rad2deg(np.arccos(np.clip(abs(relative_rotation[2, 2]), -1, 1))):5.1f} deg"
            f"  speed {linear:.4f}/{angular:.4f}"
            f"  contacts {contacts}"
        )
        if terminated or truncated:
            print(f"  ended at step {step} terminated={terminated} truncated={truncated}")
            break
    print(f"  final: phase={expert.phase.name} reason={expert.failure_reason}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
