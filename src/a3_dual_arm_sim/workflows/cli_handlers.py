from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.controllers.teleop import KeyboardTeleopPolicy
from a3_dual_arm_sim.controllers.teleop_panel import run_teleop_control_panel
from a3_dual_arm_sim.data.collection import collect_grasp_dataset
from a3_dual_arm_sim.data.recording import LeRobotV3Recorder
from a3_dual_arm_sim.learning.training import train_smolvla
from a3_dual_arm_sim.policies.base import HoldPolicy, load_policy
from a3_dual_arm_sim.sim.env import A3DualArmEnv
from a3_dual_arm_sim.sim.model import SceneName, build_model, write_generated_xml
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.tasks.grasp import A3GraspEnv
from a3_dual_arm_sim.workflows.runner import EpisodeRunner


def _recorder(args: argparse.Namespace, env: A3DualArmEnv) -> LeRobotV3Recorder | None:
    if args.record is None:
        return None
    return LeRobotV3Recorder(
        args.record,
        repo_id=args.repo_id,
        fps=env.config.control_hz,
        image_height=env.config.image_height,
        image_width=env.config.image_width,
        use_videos=args.videos,
    )


def _validate_camera_recording(args: argparse.Namespace) -> None:
    if getattr(args, "record", None) is not None and not getattr(args, "camera_render", True):
        raise ValueError(
            "--no-camera-render cannot be combined with --record because it would "
            "save black policy-camera frames"
        )


def _inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    bundle = build_model(config, scene=args.scene)
    if args.write_xml is not None:
        write_generated_xml(args.write_xml, config, scene=args.scene)
    summary = {
        "model": f"A3 dual-arm {args.scene}",
        "arm_joint_count": len(bundle.source_joints),
        "joint_names": [joint.name for joint in bundle.source_joints],
        "joint_ranges": {joint.name: joint.limits for joint in bundle.source_joints},
        "mujoco": {
            "nq": bundle.model.nq,
            "nv": bundle.model.nv,
            "nu": bundle.model.nu,
            "nsensor": bundle.model.nsensor,
            "nsensordata": bundle.model.nsensordata,
        },
        "invalid_source_meshes_excluded": ["L_LAST_S.STL", "R_LAST_S.STL"],
        "generated_xml": str(args.write_xml) if args.write_xml is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    env = _make_env(
        args.scene,
        args.config,
        action_mode="joint_position",
        render_cameras=False,
    )
    task = (
        "transfer exactly ten upright square cookie blocks into the 2x5 box"
        if args.scene == "cookie_transfer"
        else "hold the A3 home pose"
    )
    runner = EpisodeRunner(env, HoldPolicy(), task=task)
    try:
        result = runner.run(seed=args.seed, max_steps=args.steps)
        state = env.current_joint_action
        summary = {
            **result.__dict__,
            "finite_state": bool(np.all(np.isfinite(state))),
            "max_abs_joint_velocity": float(np.max(np.abs(env.data.qvel))),
            "steps_requested": args.steps,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not result.terminated and summary["finite_state"] else 1
    finally:
        runner.close()


def _run(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    render_mode = "human" if args.render else None
    env = _make_env(
        args.scene,
        args.config,
        action_mode=policy.action_mode,
        render_mode=render_mode,
        render_cameras=args.camera_render,
    )
    recorder = _recorder(args, env)
    runner = EpisodeRunner(
        env,
        policy,
        task=args.task or _default_task(args.scene),
        recorder=recorder,
        realtime=args.render,
    )
    try:
        result = runner.run(seed=args.seed, max_steps=args.steps)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 0 if not result.terminated else 1
    finally:
        runner.close()


def _teleop(args: argparse.Namespace) -> int:
    recording_available = args.record is not None
    policy = KeyboardTeleopPolicy(
        recording=recording_available,
        recording_available=recording_available,
    )
    env = _make_env(
        args.scene,
        args.config,
        action_mode="cartesian_delta",
        render_mode="human",
        render_cameras=args.camera_render,
    )
    env.set_key_callback(policy.handle_key)
    recorder = _recorder(args, env)
    runner = EpisodeRunner(
        env,
        policy,
        task=args.task or _default_task(args.scene),
        recorder=recorder,
        realtime=True,
    )
    mode = (
        "interactive debug: Human Viewer on, Policy RGB off, recording off"
        if not args.camera_render
        else "teleoperation capture: Human Viewer and Policy RGB on"
    )
    print(f"Mode: {mode}")
    print(
        "Use the separate A3 control panel for keyboard/buttons. Keep focus on that "
        "panel; the MuJoCo Viewer is display-only."
    )
    try:
        result = run_teleop_control_panel(
            policy,
            lambda: runner.run(seed=args.seed, max_steps=args.steps),
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 0 if not result.terminated else 1
    finally:
        runner.close()


def _make_env(
    scene: SceneName,
    config: Path | None,
    **kwargs: Any,
) -> A3DualArmEnv:
    if scene == "cookie_transfer":
        return A3CookieTransferEnv(config, **kwargs)
    return A3DualArmEnv(config, scene=scene, **kwargs)


def _default_task(scene: SceneName) -> str:
    if scene == "cookie_transfer":
        return "transfer exactly ten upright square cookie blocks into the 2x5 box"
    return "move the A3 dual-arm robot"


def _read_episode_context(root: Path, episode: int) -> tuple[int, str]:
    metadata = root / "a3_episode_metadata.jsonl"
    if not metadata.exists():
        return 0, ""
    for line in metadata.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if int(item["episode_index"]) == episode:
            return int(item["seed"]), str(item.get("task", ""))
    return 0, ""


def _read_episode_actions(root: Path, episode: int) -> list[np.ndarray]:
    """Read replay actions without constructing LeRobot's image/video decoder."""
    try:
        import pyarrow.dataset as pa_dataset
    except ImportError as exc:
        raise RuntimeError("Install recording support with: pip install -e '.[dataset]'") from exc
    dataset = pa_dataset.dataset(root / "data", format="parquet")
    table = dataset.to_table(
        columns=["frame_index", "action"],
        filter=pa_dataset.field("episode_index") == episode,
    )
    rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError(f"Episode {episode} has no actions in {root}")
    return [np.asarray(row["action"], dtype=np.float64) for row in rows]


def _replay(args: argparse.Namespace) -> int:
    actions = _read_episode_actions(args.root, args.episode)
    summary_path = args.root / "collection_summary.json"
    is_grasp = False
    if summary_path.is_file():
        is_grasp = json.loads(summary_path.read_text(encoding="utf-8")).get("task") == "a3_grasp"
    seed, recorded_task = _read_episode_context(args.root, args.episode)
    is_cookie = "cookie" in recorded_task.lower()
    if is_grasp:
        env = A3GraspEnv(
            args.config,
            action_mode="joint_position",
            render_mode="human" if args.render else None,
            render_cameras=False,
        )
    else:
        env = _make_env(
            "cookie_transfer" if is_cookie else "sandbox",
            args.config,
            action_mode="joint_position",
            render_mode="human" if args.render else None,
            render_cameras=False,
        )
    env.reset(seed=seed)
    steps = 0
    final_info: dict[str, Any] = {}
    try:
        for action in actions:
            _, _, terminated, truncated, final_info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        result = {
            "episode": args.episode,
            "steps": steps,
            "seed": seed,
            "task": "a3_grasp" if is_grasp else recorded_task or "sandbox",
            "success": bool(final_info.get("success", False)),
            "safety_reason": final_info.get("safety_reason"),
        }
        if is_grasp:
            result["grasp_validation"] = {
                "both_fingers_contact": all(
                    final_info.get("finger_object_contacts", (False, False))
                ),
                "grasped": bool(final_info.get("grasped", False)),
                "object_touches_table": bool(final_info.get("object_touches_table", True)),
                "lift_m": float(final_info.get("lift_m", 0.0)),
                "linear_speed_m_s": float(final_info.get("object_linear_speed_m_s", 0.0)),
                "angular_speed_rad_s": float(final_info.get("object_angular_speed_rad_s", 0.0)),
                "grasp_center_error_m": float(final_info.get("grasp_center_error_m", 0.0)),
                "stable_hold_steps": int(final_info.get("success_hold_count", 0)),
                "required_stable_hold_steps": env.task_config.success_hold_steps,
            }
        elif is_cookie:
            result["cookie_validation"] = {
                "cookies_in_target": int(final_info.get("cookies_in_target", 0)),
                "cookies_in_source": int(final_info.get("cookies_in_source", 0)),
                "required_cookies": int(final_info.get("required_cookies", 0)),
                "exact_2x5_fill": bool(final_info.get("exact_2x5_fill", False)),
                "target_touches_all_walls": bool(final_info.get("target_touches_all_walls", False)),
                "target_slot_occupancy": list(final_info.get("target_slot_occupancy", ())),
                "stable_hold_steps": int(final_info.get("success_hold_count", 0)),
                "required_stable_hold_steps": int(final_info.get("required_success_hold_steps", 0)),
            }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        env.close()


def _collect_grasp(args: argparse.Namespace) -> int:
    summary = collect_grasp_dataset(
        args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        start_seed=args.seed,
        max_attempts=args.max_attempts,
        config=args.config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _train_smolvla(args: argparse.Namespace) -> int:
    summary = train_smolvla(
        dataset_root=args.root,
        repo_id=args.repo_id,
        base_model=args.model,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        lr=getattr(args, "lr", 5e-5),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
