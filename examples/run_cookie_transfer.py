#!/usr/bin/env python3
"""Run dual-arm A3 cookie transfer task with privileged expert.

Experimental contact-based expert: right arm holds and tilts the target box,
left arm attempts to pack 10 cookies. A completed run is not necessarily successful.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from a3_dual_arm_sim import A3CookieTransferEnv, A3CookieTransferExpert


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A3 Cookie Transfer Demo")
    parser.add_argument("--render", action="store_true", help="Render simulation interactively")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=6000, help="Maximum control steps")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    print("=" * 60)
    print("Dual-Arm A3 Cookie Transfer Task (10 Blocks Pick & Place)")
    print(f"Render mode: {render_mode or 'headless (fast)'}")
    print("=" * 60)

    env = A3CookieTransferEnv(
        args.config,
        render_mode=render_mode,
        render_cameras=False,
    )
    try:
        expert = A3CookieTransferExpert(env)
        obs, info = env.reset(seed=args.seed, options={"randomize_cookies": False})
        expert.reset()
        print("Closed-loop expert ready (actions are decided one step at a time).")

        terminated = False
        step = 0
        previous_phase = expert.phase

        while not terminated and not expert.failed and step < args.max_steps:
            started = time.monotonic()
            action = expert.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            step += 1

            if expert.phase is not previous_phase:
                print(f"Step {step:4d} | {expert.status}")
                previous_phase = expert.phase
            if step % 150 == 0 or terminated or expert.failed:
                in_target = info.get("cookies_in_target", 0)
                in_source = info.get("cookies_in_source", 0)
                hold_cnt = info.get("success_hold_count", 0)
                print(
                    f"Step {step:4d} | target={in_target:2d}/10 | "
                    f"source remaining={in_source:2d}/30 | hold={hold_cnt:2d}"
                )
            if truncated:
                break
            if args.render:
                time.sleep(max(0.0, 1 / env.config.control_hz - (time.monotonic() - started)))

        print("\n" + "=" * 60)
        print("Task Finished!")
        print(f"Total steps: {step}")
        print(f"Success: {info.get('success')}")
        print(f"Exact 2x5 Fill: {info.get('exact_2x5_fill')}")
        print(f"Cookies in Target Bin: {info.get('cookies_in_target')}/10")
        print(f"Cookies Remaining in Source Bin: {info.get('cookies_in_source')}/30")
        print(f"Touches All 4 Walls: {info.get('target_touches_all_walls')}")
        print(f"Target Slot Occupancy: {info.get('target_slot_occupancy')}")
        print(f"Expert State: {expert.status}")
        print(f"Retries Per Slot: {expert.retry_counts}")
        if expert.failure_reason:
            print(f"Failure Reason: {expert.failure_reason}")
        print("=" * 60)

        return 0 if info.get("success") else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
