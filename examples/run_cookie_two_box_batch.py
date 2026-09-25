#!/usr/bin/env python3
"""Fill A, push it out with the right arm, carry B in, and fill B.

This is an independent example. It does not record training demonstrations.
The original single-box example remains examples/run_cookie_batch.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.relay_batch_expert import RelayBatchExpert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cookie_two_box_batch.yaml")
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--physics-hz", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.physics_hz is not None:
        if args.physics_hz <= 0 or args.physics_hz % config.control_hz:
            parser.error("--physics-hz must be a positive multiple of control_hz")
        config = replace(config, physics_hz=args.physics_hz)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if config.cookie_transfer.spare_target_bin_world_position_m is None:
        parser.error("the dual-box workflow requires a spare_target_bin_world_position_m")

    env = A3CookieTransferEnv(
        config,
        render_mode="human" if args.render else None,
        render_cameras=False,
        task_config=CookieTransferTaskConfig(
            require_exact_slots=False,
            require_released=True,
            terminate_on_success=False,
        ),
    )
    start = time.monotonic()
    steps = 0
    terminated = truncated = False
    info = {}
    expert = None
    try:
        env.reset(seed=args.seed, options={"randomize_cookies": False})
        expert = RelayBatchExpert(env)
        expert.reset()
        last_status = None
        for steps in range(1, args.max_steps + 1):
            tick = time.monotonic()
            action = expert.act()
            _, _, terminated, truncated, info = env.step(action)
            status = (
                expert.stage,
                expert.fill.phase if expert.fill else None,
                expert.pusher.phase if expert.stage in ("PUSH", "CARRY") else None,
            )
            if status != last_status or (args.debug and steps % 100 == 0):
                print(f"step={steps} {expert.status}", flush=True)
                last_status = status
            if expert.failed or expert.done or terminated or truncated:
                break
            if args.render:
                time.sleep(max(0.0, 1.0 / config.control_hz - (time.monotonic() - tick)))

        a_count, b_count = expert.counts()
        source_count = sum(
            env.privileged_cookie_in_source(i)
            for i in range(len(env._cookie_bodies))
        )
        failure = expert.failed
        if not expert.done and not failure:
            if terminated:
                failure = f"environment safety termination: {info.get('safety_reason')}"
            elif truncated:
                failure = "environment horizon reached"
            else:
                failure = "maximum steps reached"
        result = {
            "seed": args.seed,
            "success": bool(expert.done and not failure),
            "steps": steps,
            "wall_seconds": round(time.monotonic() - start, 2),
            "stage": expert.stage,
            "failure_reason": failure,
            "safety_reason": info.get("safety_reason"),
            "box_a_cookie_count": a_count,
            "box_b_cookie_count": b_count,
            "source_cookie_count": source_count,
            "box_a_position_m": env.data.xpos[expert.box_a].tolist(),
            "box_b_position_m": env.data.xpos[expert.box_b].tolist(),
            "box_b_station_error_m": float(
                ((env.data.xpos[expert.box_b, :2] - expert.station) ** 2).sum() ** 0.5
            ),
            "fills": expert.fill_reports,
            "pushes": expert.push_reports,
            "success_criterion": "10 released upright cookies in each box; 60 remain in source; A pushed out and B grasped into station within 8 mm",
            "training_data_recorded": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0 if result["success"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
