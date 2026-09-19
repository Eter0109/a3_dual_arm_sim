#!/usr/bin/env python3
"""Run/evaluate the bevel-insertion five-Cookie batch expert without recording training data."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from a3_dual_arm_sim.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import A3CookieTransferEnv, CookieTransferTaskConfig
from a3_dual_arm_sim.expert import CookiePhase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cookie_same_column.yaml"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-second-batch-state", type=Path, help="Save a reproducible debug checkpoint after the first release")
    parser.add_argument("--resume-second-batch-state", type=Path, help="Resume from a checkpoint made with the same config")
    parser.add_argument("--save-second-aligned-state", type=Path, help="Save a debug checkpoint after straightening the second batch")
    parser.add_argument("--resume-second-aligned-state", type=Path, help="Resume after second-batch straightening")
    parser.add_argument("--save-second-lift-state", type=Path, help="Save a debug checkpoint once the second batch is lifted")
    parser.add_argument("--resume-second-lift-state", type=Path, help="Resume a second-lift checkpoint with the same config")
    parser.add_argument("--save-second-open-state", type=Path, help="Save a debug checkpoint at second-batch release")
    parser.add_argument("--resume-second-open-state", type=Path, help="Resume a second-batch release checkpoint")
    parser.add_argument("--snapshots", type=Path, help="Optional diagnostic PNGs, not a dataset")
    parser.add_argument("--physics-hz", type=int, help="Override simulation rate for comparison")
    parser.add_argument(
        "--release-physics-hz", type=int,
        help="Use finer integration only while releasing the second batch",
    )
    parser.add_argument("--sliding-friction", type=float, help="Constant Cookie sliding friction")
    parser.add_argument(
        "--cookie-contact-time-constant", type=float,
        help="Override Cookie contact softness for numerical stability comparisons",
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    config = load_config(args.config)
    if args.physics_hz is not None:
        if args.physics_hz <= 0 or args.physics_hz % config.control_hz != 0:
            parser.error("--physics-hz must be a positive multiple of control_hz")
        config = replace(config, physics_hz=args.physics_hz)
    if args.release_physics_hz is not None and (
        args.release_physics_hz <= 0
        or args.release_physics_hz % config.control_hz != 0
    ):
        parser.error("--release-physics-hz must be a positive multiple of control_hz")
    if args.sliding_friction is not None:
        if not math.isfinite(args.sliding_friction) or args.sliding_friction <= 0:
            parser.error("--sliding-friction must be finite and positive")
        friction = (args.sliding_friction, *config.cookie_transfer.cookie_friction[1:])
        config = replace(
            config, cookie_transfer=replace(config.cookie_transfer, cookie_friction=friction)
        )
    if args.cookie_contact_time_constant is not None:
        value = args.cookie_contact_time_constant
        if not math.isfinite(value) or value <= 0:
            parser.error("--cookie-contact-time-constant must be finite and positive")
        config = replace(
            config,
            cookie_transfer=replace(
                config.cookie_transfer,
                cookie_solref=(value, config.cookie_transfer.cookie_solref[1]),
            ),
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
        expert = A3SameColumnBatchExpert(env)
        expert.reset()
        start_step = 1
        if args.resume_second_batch_state or args.resume_second_aligned_state or args.resume_second_lift_state or args.resume_second_open_state:
            resume_path = (
                args.resume_second_batch_state
                or args.resume_second_aligned_state
                or args.resume_second_lift_state
                or args.resume_second_open_state
            )
            with np.load(resume_path, allow_pickle=False) as checkpoint:
                state = checkpoint["state"]
                if state.size != mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS):
                    raise ValueError("checkpoint model does not match the current config")
                mujoco.mj_setState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                env.data.ctrl[:] = checkpoint["ctrl"]
                env._last_applied_action = checkpoint["last_action"].copy()
                env._step_count = int(checkpoint["step_count"])
                env._source_initially_filled = True
                env._success_hold_count = 0
                expert.completed_cookie_indices = list(range(5))
                expert.batch_reports = json.loads(str(checkpoint["batch_reports_json"]))
                expert.batch_index = 1
                if args.resume_second_aligned_state:
                    expert._base_push_attempts = 2
                if args.resume_second_lift_state or args.resume_second_open_state:
                    expert.batch_indices = list(range(5, 10))
                    expert.current_cookie_index = 5
                    expert.target_slot_index = 1
                    expert._group_offset = checkpoint["group_offset"].copy()
                    expert._held_offsets = checkpoint["held_offsets"].copy()
                    expert._opening = float(checkpoint["opening"])
                    expert._advance(
                        CookiePhase.OPEN if args.resume_second_open_state else CookiePhase.MOVE_TO_SLOT
                    )
                mujoco.mj_forward(env.model, env.data)
                obs = env._observation()
                start_step = env._step_count + 1
        if args.snapshots:
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
        for step in range(start_step, args.max_steps + 1):
            if (
                args.release_physics_hz is not None
                and expert.batch_index == 1
                and expert.phase in (
                    CookiePhase.OPEN, CookiePhase.RETRACT, CookiePhase.VERIFY_RELEASE,
                )
                and env.config.physics_hz != args.release_physics_hz
            ):
                env.config = replace(env.config, physics_hz=args.release_physics_hz)
                env.model.opt.timestep = 1.0 / args.release_physics_hz
                print(f"step={step} release_physics_hz={args.release_physics_hz}", flush=True)
            tick = time.monotonic()
            action = expert.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if (
                args.save_second_batch_state
                and expert.phase.name == "SELECT_COOKIE"
                and expert.batch_index == 1
                and step == env._step_count
            ):
                state = np.empty(
                    mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                )
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                args.save_second_batch_state.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.save_second_batch_state,
                    state=state,
                    ctrl=env.data.ctrl,
                    last_action=env.last_applied_action,
                    step_count=step,
                    batch_reports_json=json.dumps(expert.batch_reports),
                )
                args.save_second_batch_state = None
            if (
                args.save_second_aligned_state
                and expert.phase is CookiePhase.SELECT_COOKIE
                and expert.batch_index == 1
                and expert._base_push_attempts == 2
            ):
                state = np.empty(
                    mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                )
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                args.save_second_aligned_state.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.save_second_aligned_state,
                    state=state,
                    ctrl=env.data.ctrl,
                    last_action=env.last_applied_action,
                    step_count=step,
                    batch_reports_json=json.dumps(expert.batch_reports),
                )
                args.save_second_aligned_state = None
            if (
                args.save_second_lift_state
                and expert.phase is CookiePhase.MOVE_TO_SLOT
                and expert.batch_index == 1
            ):
                state = np.empty(
                    mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                )
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                args.save_second_lift_state.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.save_second_lift_state,
                    state=state,
                    ctrl=env.data.ctrl,
                    last_action=env.last_applied_action,
                    step_count=step,
                    batch_reports_json=json.dumps(expert.batch_reports),
                    group_offset=expert._group_offset,
                    held_offsets=expert._held_offsets,
                    opening=expert._opening,
                )
                args.save_second_lift_state = None
            if (
                args.save_second_open_state
                and expert.phase is CookiePhase.OPEN
                and expert.batch_index == 1
            ):
                state = np.empty(
                    mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                )
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                args.save_second_open_state.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.save_second_open_state,
                    state=state,
                    ctrl=env.data.ctrl,
                    last_action=env.last_applied_action,
                    step_count=step,
                    batch_reports_json=json.dumps(expert.batch_reports),
                    group_offset=expert._group_offset,
                    held_offsets=expert._held_offsets,
                    opening=expert._opening,
                )
                args.save_second_open_state = None
            if renderer is not None and (expert.phase != previous or step % 100 == 0):
                renderer.update_scene(env.data, camera=camera)
                Image.fromarray(renderer.render()).save(
                    args.snapshots / f"{step:04d}_{expert.phase.name}.png"
                )
            if expert.phase != previous:
                print(f"step={step} {expert.status}", flush=True)
                if args.debug and expert.batch_index == 1 and expert.phase.name in ("SELECT_COOKIE", "APPROACH"):
                    for index in range(5, 11):
                        body = env._cookie_bodies[index]
                        up = env.data.xmat[body].reshape(3, 3)[:, 2]
                        tilt_deg = math.degrees(math.acos(min(1.0, abs(float(up[2])))))
                        print(
                            f"source_cookie={index} pos={env.data.xpos[body].round(5).tolist()} "
                            f"tilt_deg={tilt_deg:.2f} up={up.round(3).tolist()}",
                            flush=True,
                        )
                if args.debug and expert.batch_indices:
                    print(
                        f"pick_eef={getattr(expert, '_pick_eef', None)} "
                        f"pad_offset={expert._pad_offset} "
                        f"aligned_grasp={getattr(expert, '_aligned_grasp', False)} "
                        f"rear_clearance_mm={1000 * getattr(expert, '_rear_clearance_m', float('inf')):.2f}",
                        flush=True,
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
                if expert.batch_index == 1 and expert.phase in (CookiePhase.DESCEND, CookiePhase.CLOSE):
                    print(
                        "pad_centers="
                        f"{env.data.geom_xpos[list(env._left_finger_geoms)].round(5).tolist()} "
                        "pad_long_axes="
                        f"{env.data.geom_xmat[list(env._left_finger_geoms)].reshape(2, 3, 3)[:, :, 2].round(3).tolist()} "
                        f"target={expert._pick_eef.round(5).tolist()}",
                        flush=True,
                    )
            if args.debug and expert.failed:
                print("blocked_contact_details:", flush=True)
                for cid, contact in enumerate(env.data.contact):
                    if not ({int(contact.geom1), int(contact.geom2)} & set(env._left_finger_geoms)):
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, cid, force)
                    if force[0] < 0.5:
                        continue
                    bodies = [
                        mujoco.mj_id2name(
                            env.model, mujoco.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[g])
                        )
                        for g in (contact.geom1, contact.geom2)
                    ]
                    print(
                        f"bodies={bodies} force_n={force[0]:.2f} "
                        f"penetration_mm={max(0.0, -contact.dist) * 1000:.2f}",
                        flush=True,
                    )
            if terminated or truncated or expert.failed or expert.finished:
                if args.debug and terminated:
                    dof = int(np.argmax(np.abs(env.data.qvel)))
                    joint = int(env.model.dof_jntid[dof])
                    print(
                        f"safety_reason={info.get('safety_reason')} "
                        f"max_qvel={env.data.qvel[dof]:.2f} "
                        f"joint={mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)}",
                        flush=True,
                    )
                    ranked = []
                    for cid, contact in enumerate(env.data.contact):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, cid, force)
                        ranked.append((force[0], contact))
                    for force, contact in sorted(ranked, key=lambda x: x[0], reverse=True)[:8]:
                        names = [
                            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                            for g in (contact.geom1, contact.geom2)
                        ]
                        print(
                            f"contact={names} force_n={force:.1f} "
                            f"penetration_mm={max(0.0, -contact.dist) * 1000:.2f}",
                            flush=True,
                        )
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
            "safety_reason": info.get("safety_reason"),
            "completed_cookies": list(expert.completed_cookie_indices),
            "batches": expert.batch_reports,
            "max_pad_force_n": expert.max_pad_force,
            "max_contact_penetration_mm": 1000 * expert.max_contact_penetration_m,
            "physics_hz": config.physics_hz,
            "release_physics_hz": args.release_physics_hz,
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
