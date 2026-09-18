#!/usr/bin/env python3
"""Run/evaluate the bevel-insertion five-Cookie batch expert without recording training data."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from a3_dual_arm_sim.batch_expert import A3CookieBatchExpert
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cookie_batch.yaml"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--snapshots", type=Path, help="Optional diagnostic PNGs, not a dataset")
    parser.add_argument("--physics-hz", type=int, help="Override simulation rate for comparison")
    parser.add_argument("--sliding-friction", type=float, help="Constant Cookie sliding friction")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    config = load_config(args.config)
    if args.physics_hz is not None:
        if args.physics_hz <= 0 or args.physics_hz % config.control_hz != 0:
            parser.error("--physics-hz must be a positive multiple of control_hz")
        config = replace(config, physics_hz=args.physics_hz)
    if args.sliding_friction is not None:
        if not math.isfinite(args.sliding_friction) or args.sliding_friction <= 0:
            parser.error("--sliding-friction must be finite and positive")
        friction = (args.sliding_friction, *config.cookie_transfer.cookie_friction[1:])
        config = replace(
            config, cookie_transfer=replace(config.cookie_transfer, cookie_friction=friction)
        )
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
    renderer = None
    try:
        obs, info = env.reset(seed=args.seed, options={"randomize_cookies": False})
        expert = A3CookieBatchExpert(env)
        expert.reset()
        if args.snapshots:
            import mujoco
            from PIL import Image

            args.snapshots.mkdir(parents=True, exist_ok=True)
            env.model.vis.global_.offwidth = 960
            env.model.vis.global_.offheight = 720
            renderer = mujoco.Renderer(env.model, height=720, width=960)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.15, 0.19, 0.78]
            camera.distance, camera.azimuth, camera.elevation = 0.65, 135, -45
        previous = None
        step = 0
        for step in range(1, args.max_steps + 1):
            tick = time.monotonic()
            action = expert.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if renderer is not None and (expert.phase != previous or step % 100 == 0):
                renderer.update_scene(env.data, camera=camera)
                Image.fromarray(renderer.render()).save(
                    args.snapshots / f"{step:04d}_{expert.phase.name}.png"
                )
            if expert.phase != previous:
                print(f"step={step} {expert.status}", flush=True)
                if args.debug and expert.batch_indices:
                    print(
                        f"pick_eef={expert._pick_eef} pad_offset={expert._pad_offset}", flush=True
                    )
                previous = expert.phase
            if args.debug and (
                step % 50 == 0 or (expert.phase.name == "LIFT" and expert.phase_steps <= 12)
            ):
                print(
                    f"step={step} eef={env.data.site_xpos[expert._l_site].round(4)} "
                    f"opening={action[7]:.4f} target={info['cookies_in_target']} "
                    f"actual_gap_mm={env.current_joint_action[7] * 85:.3f} "
                    f"pad_forces={expert._contact_chain()[1].round(3)} "
                    f"positions={expert._positions().round(4).tolist()}",
                    flush=True,
                )
                contacts = [
                    c
                    for c in env.data.contact
                    if c.geom1 in env._left_finger_geoms or c.geom2 in env._left_finger_geoms
                ]
                print(
                    f"finger_penetration_mm={-1000 * min([c.dist for c in contacts] + [0]):.3f} "
                    f"ncon={env.data.ncon}",
                    flush=True,
                )
            if terminated or truncated or expert.failed or expert.finished:
                break
            if args.render:
                time.sleep(max(0, 1 / env.config.control_hz - (time.monotonic() - tick)))
        success = bool(info.get("success", False)) and expert.phase.name == "DONE"
        failure_reason = expert.failure_reason
        if not success and failure_reason is None:
            if terminated:
                failure_reason = "environment terminated before batch completion"
            elif truncated:
                failure_reason = "environment horizon reached"
            elif not expert.finished:
                failure_reason = "maximum steps reached"
            else:
                failure_reason = "final released-count/source-count criterion not met"
        result = {
            "seed": args.seed,
            "success": success,
            "steps": step,
            "wall_seconds": round(time.monotonic() - start, 2),
            "cookies_in_target": info.get("cookies_in_target"),
            "cookies_in_source": info.get("cookies_in_source"),
            "phase": expert.phase.name,
            "failure_phase": expert.failure_phase,
            "failure_reason": failure_reason,
            "completed_cookies": list(expert.completed_cookie_indices),
            "batches": expert.batch_reports,
            "max_pad_force_n": expert.max_pad_force,
            "max_contact_penetration_mm": 1000 * expert.max_contact_penetration_m,
            "physics_hz": env.config.physics_hz,
            "sliding_friction": env.config.cookie_transfer.cookie_friction[0],
            "cookie_collision_mode": env.config.cookie_transfer.cookie_collision_mode,
            "target_support": "tabletop; right arm parked",
            "success_criterion": "exactly 10 released, upright, settled, contained; 70 remain in source",
            "training_data_recorded": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0 if result["success"] else 1
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
