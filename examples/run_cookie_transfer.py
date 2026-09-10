#!/usr/bin/env python3
"""Run dual-arm A3 cookie transfer task with privileged expert.

Right arm holds and tilts the target box (10 deg), left arm grasps 10 cookies
from the source bin and packs them into the target box in a 2x5 configuration.
"""

from __future__ import annotations

import argparse
import sys
from a3_dual_arm_sim import A3CookieTransferEnv, A3CookieTransferExpert


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A3 Cookie Transfer Demo")
    parser.add_argument(
        "--render", action="store_true", help="Render simulation interactively"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    print("=" * 60)
    print("Dual-Arm A3 Cookie Transfer Task (10 Blocks Pick & Place)")
    print(f"Render mode: {render_mode or 'headless (fast)'}")
    print("=" * 60)

    env = A3CookieTransferEnv(
        render_mode=render_mode,
        render_cameras=False,
    )
    try:
        expert = A3CookieTransferExpert(env)
        obs, info = env.reset(seed=args.seed, options={"randomize_cookies": False})
        expert.reset()
        print(f"Expert planned {len(expert.actions)} actions.")

        terminated = False
        step = 0
        max_steps = len(expert.actions) + 100

        while not terminated and step < max_steps:
            action = expert.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if step % 150 == 0 or terminated:
                in_target = info.get("cookies_in_target", 0)
                in_source = info.get("cookies_in_source", 0)
                hold_cnt = info.get("success_hold_count", 0)
                print(
                    f"Step {step:4d} | in_target={in_target:2d}/10 | in_source={in_source:2d}/20 | hold={hold_cnt:2d}"
                )

        print("\n" + "=" * 60)
        print("Task Finished!")
        print(f"Total steps: {step}")
        print(f"Success: {info.get('success')}")
        print(f"Exact 2x5 Fill: {info.get('exact_2x5_fill')}")
        print(f"Cookies in Target Bin: {info.get('cookies_in_target')}/10")
        print(f"Cookies in Source Bin: {info.get('cookies_in_source')}/20")
        print(f"Touches All 4 Walls: {info.get('target_touches_all_walls')}")
        print(f"Target Slot Occupancy: {info.get('target_slot_occupancy')}")
        print("=" * 60)

        return 0 if info.get("success") else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
