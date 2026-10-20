from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .collection import collect_cookie_dataset, collect_grasp_dataset
from .config import VISUAL_STYLES, load_config
from .cookie_transfer import A3CookieTransferEnv
from .env import A3DualArmEnv
from .grasp import A3GraspEnv
from .model import build_model, write_generated_xml
from .policy import HoldPolicy, load_policy, policy_requires_env
from .recording import LeRobotV3Recorder
from .runner import EpisodeRunner
from .tasks import COOKIE_PROMPT
from .teleop import KeyboardTeleopPolicy
from .teleop_panel import run_teleop_control_panel
from .training import default_base_model, train_smolvla


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
    if getattr(args, "record", None) is not None and not getattr(
        args, "camera_render", True
    ):
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
    task = COOKIE_PROMPT if args.scene == "cookie_transfer" else "hold the A3 home pose"
    runner = EpisodeRunner(env, HoldPolicy(), task=task)
    try:
        result = runner.run(seed=args.seed, max_steps=args.steps)
        state = env.current_joint_action
        summary = {
            **result.to_dict(),
            "finite_state": bool(np.all(np.isfinite(state))),
            "max_abs_joint_velocity": float(np.max(np.abs(env.data.qvel))),
            "steps_requested": args.steps,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not result.terminated and summary["finite_state"] else 1
    finally:
        runner.close()


def _run(args: argparse.Namespace) -> int:
    if args.policy_manifest is not None:
        os.environ["A3_POLICY_MANIFEST"] = str(args.policy_manifest)
    if args.policy is None:
        if args.policy_manifest is None:
            raise ValueError("either --policy or --policy-manifest is required")
        args.policy = "a3_dual_arm_sim.bridge.gateway:make_policy"
    render_mode = "human" if args.render else None
    if policy_requires_env(args.policy):
        # Privileged experts plan against live simulator state, so the
        # environment has to exist before the policy is constructed.
        env = _make_env(
            args.scene,
            args.config,
            action_mode="joint_position",
            render_mode=render_mode,
            render_cameras=args.camera_render,
            fast_render=args.fast_render,
            visual_style=args.visual_style,
        )
        policy = load_policy(args.policy, env=env)
    else:
        policy = load_policy(args.policy)
        env = _make_env(
            args.scene,
            args.config,
            action_mode=policy.action_mode,
            render_mode=render_mode,
            render_cameras=args.camera_render,
            fast_render=args.fast_render,
            visual_style=args.visual_style,
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
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
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
        fast_render=args.fast_render,
        visual_style=args.visual_style,
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
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if not result.terminated else 1
    finally:
        runner.close()


def _policy_check(args: argparse.Namespace) -> int:
    """Audit one or more manifests without loading any model.

    Every failure this reports would otherwise surface mid-rollout, after the
    weights were downloaded and the model had loaded.
    """

    from .bridge.manifest import describe_manifest, load_manifest

    reports: list[dict[str, Any]] = []
    failed = False
    for path in args.manifests:
        try:
            manifest = load_manifest(path)
        except ValueError as exc:
            reports.append({"path": str(path), "usable": False, "problems": [str(exc)]})
            failed = True
            continue
        report = describe_manifest(manifest)
        problems = list(report["derived"]["problems"])
        reports.append({"path": str(path), "usable": not problems, **report})
        failed = failed or bool(problems)
    print(json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2))
    return 1 if failed else 0


#: Reset options that pin a scene to its canonical layout.  Spelled out rather than
#: defaulted because the three names belong to different scenes -- the plain sandbox
#: reads ``randomize_objects``, the cookie scenes read ``randomize_scene`` and
#: ``randomize_cookies`` -- and each environment ignores the ones it does not use.
DETERMINISTIC_RESET_OPTIONS = {
    "randomize_objects": False,
    "randomize_scene": False,
    "randomize_cookies": False,
}


def _policy_compare(args: argparse.Namespace) -> int:
    """Run several manifests on one scene and report them side by side.

    Every manifest is driven through the same environment, the same seeds and the
    same task string, because the point is to make the models comparable rather
    than to give each its best case.
    """

    from .bridge.compare import compare, format_table
    from .bridge.gateway import ExternalPolicyGateway, GatewayCache
    from .bridge.manifest import ManifestError, load_manifest, validate_manifest

    manifests = []
    for path in args.manifests:
        manifest = load_manifest(path)
        problems = validate_manifest(manifest)
        if problems:
            raise ManifestError(f"{path} is not usable:\n  - " + "\n  - ".join(problems))
        manifests.append(manifest)

    def env_factory() -> A3DualArmEnv:
        # No `action_mode` here, deliberately.  This used to request `joint_position`
        # unconditionally, and `_make_env` correctly refuses that for any registered scene
        # that drives `cartesian_delta` -- which is the whole LIBERO-Panda family, so the
        # scenes every `pi05_libero_*.yaml` manifest exists to evaluate could not be
        # compared at all.  The error names the scene's action space and reads as a
        # problem with the scene rather than with this line.
        #
        # The guard is for catching a *caller* that asks for a space the scene does not
        # have, and `compare` is not such a caller: each manifest declares the space the
        # **model** emits, and the bridge converts between that and the environment's own
        # (see `docs/external-policy-bridge.md`).  A scene that pins its own mode is then
        # the only author of the environment side, which is what it already is for its
        # task config.
        #
        # Non-registered scenes are unaffected: `A3DualArmEnv` and `A3CookieTransferEnv`
        # both default to `joint_position`, so dropping the argument leaves `sandbox`,
        # `cookie_transfer` and `--config` runs byte-identical.
        return _make_env(
            args.scene,
            args.config,
            render_cameras=True,
            fast_render=args.fast_render,
            visual_style=args.visual_style,
        )

    # One gateway per manifest, reused across episodes; see :class:`GatewayCache` for
    # why only one is held at a time and what that costs if it is not.
    gateways = GatewayCache(
        lambda manifest, env: ExternalPolicyGateway(manifest, env=env, keep_transport=True)
    )

    seeds = _parse_seeds(args.seeds)
    try:
        report = compare(
            manifests,
            env_factory=env_factory,
            policy_factory=gateways.gateway_for,
            seeds=seeds,
            steps=args.steps,
            # The scene's own prompt unless overridden.  ``compare`` falls back to a
            # hardcoded sandbox string otherwise, which on a cookie scene would ask every
            # model to "move the A3 dual-arm robot" while the frames fill with cookies --
            # a silent mismatch between the task the scene implements and the task the
            # model was told.
            task=args.task or _default_task(args.scene),
            scene=args.scene,
            # The scene's own layout defaults, which randomise, unless pinned.  This used
            # to be the other way round, and the cost was a silent one: with every seed on
            # the canonical layout, a three-seed table was one configuration measured
            # three times, and nothing in the output said so.  A table that reads like a
            # sample average and is not one is worse than a table that says what it did.
            reset_options=None if args.randomize else DETERMINISTIC_RESET_OPTIONS,
            randomized=args.randomize,
        )
    finally:
        # The gateways own the model processes now that they are shared across
        # episodes, and ``runner.close`` deliberately leaves a shared transport alone.
        # Nothing else will close them, so the process would outlive the command.
        gateways.shutdown()
    table = format_table(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(table)
    if args.output is not None:
        print(f"\nJSON report: {args.output}")
    return 0


def _parse_seeds(raw: str) -> list[int]:
    """Read ``--seeds`` as ``0,1,2`` or ``0-4``."""

    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, stop = part.partition("-")
            seeds.extend(range(int(start), int(stop) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds produced no seeds")
    return seeds


def _scene_choices() -> tuple[str, ...]:
    """Scene names a policy-driving command accepts.

    Two different things share the word "scene" here, and conflating them produces
    confusing errors.  ``SceneName`` in :mod:`a3_dual_arm_sim.model` names a MuJoCo
    *layout* (``sandbox`` or the cookie table).  A registered collection scene names
    a *task contract* plus the config file that lays it out, and several of them
    share the cookie layout.  Only the collection registry knows which config goes
    with which task, so the choices come from it rather than from a hand-kept list.
    """

    from .collection import registered_scenes

    return ("sandbox", "cookie_transfer", *sorted(registered_scenes()))


def _styled_config(config: Path | None, visual_style: str) -> Any:
    """A ``SimConfig`` carrying the requested appearance.

    The appearance is compiled into the model XML, so it cannot be applied to a live
    scene the way the render flags can -- it has to reach the constructor.  All three
    environments accept a ``SimConfig`` in place of a path, so the config is loaded
    here, restyled, and handed over.
    """

    from dataclasses import replace

    loaded = load_config(config)
    return loaded if visual_style == loaded.visual_style else replace(
        loaded, visual_style=visual_style
    )


def _make_env(
    scene: str,
    config: Path | None,
    *,
    fast_render: bool = False,
    visual_style: str = "plain",
    **kwargs: Any,
) -> A3DualArmEnv:
    from .collection import registered_scenes, scene_by_name

    if scene in registered_scenes():
        if config is not None:
            raise ValueError(
                f"--config cannot be combined with the registered scene {scene!r}: "
                "that scene owns its config file, and honouring both would silently "
                "pick one.  Point --config at a copy of the scene's own YAML instead."
            )
        # Built through the scene's own builder so the task config -- which decides
        # what counts as success (``require_exact_slots``, ``require_released``,
        # ``terminate_on_success``) -- is the one the datasets were collected with.
        # Repeating it here would drift.
        #
        # Zero noise for the canonical layout: a comparison table and a rendered
        # video of "the same" configuration have to be the same configuration.
        env: A3DualArmEnv = scene_by_name(scene, visual_style=visual_style).build_env(0.0, 0.0)
        # The builder has no parameters for the constructor flags, but the
        # environment reads all three at step time, so setting them afterwards is
        # equivalent to passing them in.
        env.render_cameras = bool(kwargs.get("render_cameras", True))
        if kwargs.get("render_mode") is not None:
            env.render_mode = kwargs["render_mode"]
        action_mode = kwargs.get("action_mode")
        if action_mode is not None and action_mode != env.action_mode:
            raise ValueError(
                f"the registered scene {scene!r} drives the robot in "
                f"{env.action_mode!r}, but {action_mode!r} was requested.  The scene's "
                "task config is written for its own action space."
            )
    elif scene == "cookie_transfer":
        env = A3CookieTransferEnv(_styled_config(config, visual_style), **kwargs)
    else:
        env = A3DualArmEnv(_styled_config(config, visual_style), scene=scene, **kwargs)  # type: ignore[arg-type]
    if fast_render:
        # Applied after construction so every scene keeps its own defaults;
        # cookie_transfer, for example, supplies its own horizon.
        env.use_fast_render()
    return env

def _default_task(scene: str) -> str:
    from .collection import registered_scenes, scene_by_name

    if scene in registered_scenes():
        # The scene's own prompt, so the string a model is conditioned on is the one
        # its datasets were recorded with.
        return scene_by_name(scene).task.prompt
    if scene == "cookie_transfer":
        return COOKIE_PROMPT
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
                "angular_speed_rad_s": float(
                    final_info.get("object_angular_speed_rad_s", 0.0)
                ),
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
                "target_touches_all_walls": bool(
                    final_info.get("target_touches_all_walls", False)
                ),
                "target_slot_occupancy": list(
                    final_info.get("target_slot_occupancy", ())
                ),
                "stable_hold_steps": int(final_info.get("success_hold_count", 0)),
                "required_stable_hold_steps": int(
                    final_info.get("required_success_hold_steps", 0)
                ),
            }
        print(json.dumps(result, indent=2))
        return 0
    finally:
        env.close()


def _collect_scene(args: argparse.Namespace) -> int:
    """Collect a registered scene.  One command for every scene, by design."""

    from .collection import collect_dataset, scene_by_name

    scene = scene_by_name(args.scene, visual_style=args.visual_style)
    post_reset = None
    post_reset_description = None
    if getattr(args, "main_aligned", None) is not None:
        from .main_alignment import post_reset_hook

        # The aligned arrangement *is* the variation policy here: the reset still draws whatever
        # the scene's own randomisation asks for, and then it is overwritten onto main's bench
        # scene with these sigmas.  Zero sigmas reproduce main's evaluation scene exactly.
        post_reset = post_reset_hook(
            args.main_aligned,
            bin_noise_m=args.align_bin_noise,
            bin_noise_dist=args.align_noise_dist,
            bin_noise_horizontal_only=not args.align_noise_3d,
            cookie_noise_m=args.position_noise,
            cookie_noise_dist=args.align_noise_dist,
            cookie_noise_horizontal_only=not args.align_noise_3d,
            cookie_yaw_noise_rad=args.align_cookie_yaw_noise,
            bin_yaw_noise_rad=args.align_bin_yaw_noise,
            pose_noise_rad=args.align_pose_noise,
        )
        # The hook runs after `env.reset` and rewrites the whole arrangement, so the summary's
        # own noise fields describe a perturbation that is no longer in the frames.  This is
        # the field that says what the dataset actually contains.
        post_reset_description = {
            "source_scene": str(args.main_aligned),
            "bin_noise_m": args.align_bin_noise,
            "bin_noise_dist": args.align_noise_dist,
            "bin_noise_3d": bool(args.align_noise_3d),
            "bin_yaw_noise_rad": args.align_bin_yaw_noise,
            "cookie_position_noise_m": args.position_noise,
            "cookie_yaw_noise_rad": args.align_cookie_yaw_noise,
            "arm_pose_noise_rad": args.align_pose_noise,
        }
    summary = collect_dataset(
        args.root,
        scene,
        repo_id=args.repo_id or f"local/{args.scene}",
        episodes=args.episodes,
        start_seed=args.seed,
        max_attempts=args.max_attempts,
        position_noise_m=args.position_noise,
        yaw_noise_rad=args.yaw_noise,
        hold_steps=args.hold_steps,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fast_render=args.fast_render,
        save_failed_episodes=args.keep_failures,
        rollover_mb=getattr(args, "rollover_mb", None),
        post_reset=post_reset,
        post_reset_description=post_reset_description,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _collect_grasp(args: argparse.Namespace) -> int:
    summary = collect_grasp_dataset(
        args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        start_seed=args.seed,
        max_attempts=args.max_attempts,
        config=args.config,
        fast_render=args.fast_render,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _collect_cookie(args: argparse.Namespace) -> int:
    summary = collect_cookie_dataset(
        args.root,
        repo_id=args.repo_id,
        episodes=args.episodes,
        start_seed=args.seed,
        max_attempts=args.max_attempts,
        position_noise_m=args.position_noise,
        yaw_noise_rad=args.yaw_noise,
        hold_steps=args.hold_steps,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        config=args.config,
        fast_render=args.fast_render,
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
        dry_run=args.dry_run,
        job_name=args.job_name,
        save_freq=args.save_freq,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)


def _add_recording(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record", type=Path, default=None, help="New LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/a3-dual-arm")
    parser.add_argument("--videos", action="store_true", help="Encode camera streams as video")


def _add_camera_rendering(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera-render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render the three Policy RGB observations; use --no-camera-render "
            "for responsive viewer-only debugging"
        ),
    )


def _add_fast_render(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fast-render",
        action="store_true",
        help="Skip shadow and reflection passes; much faster without a GPU",
    )


def _add_visual_style(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visual-style",
        choices=VISUAL_STYLES,
        default="plain",
        help=(
            "Renderer appearance. 'plain' is the original look and stays the "
            "default because the rendered pixels are a dataset contract; "
            "'detailed' gives the Cookies their own materials so packed blades "
            "read as separate objects. It changes no physical parameter."
        ),
    )


def _add_rollover(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rollover-mb",
        type=float,
        default=None,
        help=(
            "Seal each parquet and video file once it reaches this many megabytes "
            "(default: LeRobot's 100 MB). A dataset is only readable once its "
            "writer is closed, so with the default a long collection stays sealed "
            "until the process exits; a smaller value makes completed files "
            "readable while the run continues. Use it when something must train on "
            "episodes collected so far."
        ),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="a3-sim", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Compile and describe the A3 model")
    inspect_parser.add_argument("--config", type=Path, default=None)
    inspect_parser.add_argument(
        "--scene", choices=("sandbox", "cookie_transfer"), default="sandbox"
    )
    inspect_parser.add_argument("--write-xml", type=Path, default=None)
    inspect_parser.set_defaults(function=_inspect)

    smoke = commands.add_parser("smoke", help="Run a deterministic headless hold episode")
    _add_common(smoke)
    smoke.add_argument(
        "--scene", choices=_scene_choices(), default="sandbox"
    )
    smoke.add_argument("--steps", type=int, default=1000)
    smoke.set_defaults(function=_smoke)

    run = commands.add_parser("run", help="Run a replaceable Python policy plugin")
    _add_common(run)
    _add_recording(run)
    _add_camera_rendering(run)
    _add_fast_render(run)
    _add_visual_style(run)
    run.add_argument(
        "--scene", choices=_scene_choices(), default="sandbox"
    )
    run.add_argument(
        "--policy",
        default=None,
        help="Python module:factory; defaults to the bridge gateway when "
        "--policy-manifest is given",
    )
    run.add_argument(
        "--policy-manifest",
        type=Path,
        default=None,
        help="External-policy manifest; see docs/external-policy-bridge.md",
    )
    run.add_argument("--task", default=None)
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--render", action="store_true")
    run.set_defaults(function=_run)

    policy = commands.add_parser(
        "policy", help="Inspect external-policy manifests"
    )
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    policy_check = policy_commands.add_parser(
        "check", help="Audit manifests without loading a model"
    )
    policy_check.add_argument("manifests", type=Path, nargs="+")
    policy_check.set_defaults(function=_policy_check)

    compare = commands.add_parser(
        "compare",
        help="Run several external policies on one scene and report them together",
    )
    _add_common(compare)
    _add_fast_render(compare)
    _add_visual_style(compare)
    compare.add_argument(
        "--scene", choices=_scene_choices(), default="sandbox"
    )
    compare.add_argument(
        "--manifest",
        dest="manifests",
        type=Path,
        action="append",
        required=True,
        help="A policy manifest; repeatable",
    )
    compare.add_argument(
        "--seeds",
        default="0",
        help="Seeds to run each model on, e.g. '0,1,2' or '0-4'",
    )
    compare.add_argument("--steps", type=int, default=None)
    compare.add_argument("--task", default=None)
    compare.add_argument(
        "--randomize",
        dest="randomize",
        action="store_true",
        default=True,
        help=(
            "Sample each episode's layout from its seed.  On by default: with it off "
            "every seed runs the identical scene, so a multi-seed table is one "
            "configuration repeated while reading like a sample average."
        ),
    )
    compare.add_argument(
        "--no-randomize",
        dest="randomize",
        action="store_false",
        help=(
            "Pin every episode to the scene's canonical layout.  Reproducible, and it "
            "agrees with a rendered video of the same configuration; use it when that "
            "matters more than whether the numbers generalise."
        ),
    )
    compare.add_argument(
        "--output", type=Path, default=None, help="Write the full report as JSON"
    )
    compare.set_defaults(function=_policy_compare)

    teleop = commands.add_parser("teleop", help="Control both arms with the keyboard")
    _add_common(teleop)
    _add_recording(teleop)
    _add_camera_rendering(teleop)
    _add_fast_render(teleop)
    _add_visual_style(teleop)
    teleop.add_argument(
        "--scene", choices=_scene_choices(), default="sandbox"
    )
    teleop.add_argument("--task", default=None)
    teleop.add_argument("--steps", type=int, default=None)
    teleop.set_defaults(function=_teleop)

    replay = commands.add_parser("replay", help="Replay canonical actions from a dataset episode")
    replay.add_argument("--config", type=Path, default=None)
    replay.add_argument("--root", type=Path, required=True)
    replay.add_argument("--repo-id", default="local/a3-dual-arm")
    replay.add_argument("--episode", type=int, default=0)
    replay.add_argument("--render", action="store_true")
    replay.set_defaults(function=_replay)

    collect = commands.add_parser(
        "collect-grasp", help="Collect successful A3 grasp expert episodes"
    )
    _add_common(collect)
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--repo-id", default="local/a3-grasp")
    collect.add_argument("--episodes", type=int, default=10)
    collect.add_argument("--max-attempts", type=int, default=None)
    _add_fast_render(collect)
    collect.set_defaults(function=_collect_grasp)

    collect_cookie = commands.add_parser(
        "collect-cookie",
        help="Collect successful A3 cookie-transfer expert episodes",
    )
    _add_common(collect_cookie)
    collect_cookie.add_argument("--root", type=Path, required=True)
    collect_cookie.add_argument("--repo-id", default="local/a3-cookie")
    collect_cookie.add_argument("--episodes", type=int, default=10)
    collect_cookie.add_argument("--max-attempts", type=int, default=None)
    collect_cookie.add_argument(
        "--position-noise",
        type=float,
        default=0.002,
        help="Per-cookie XY jitter in metres; the only source of scene variation",
    )
    collect_cookie.add_argument(
        "--yaw-noise",
        type=float,
        default=0.05,
        help="Per-cookie yaw jitter in radians",
    )
    collect_cookie.add_argument(
        "--hold-steps",
        type=int,
        default=40,
        help="Steps to keep recording after the expert finishes its last placement",
    )
    collect_cookie.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard when several processes collect in parallel",
    )
    collect_cookie.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of shards; seeds are interleaved across them",
    )
    _add_fast_render(collect_cookie)
    collect_cookie.set_defaults(function=_collect_cookie)

    collect = commands.add_parser(
        "collect",
        help="Collect any registered scene; the scene owns env, expert and contract",
    )
    _add_common(collect)
    collect.add_argument("--root", type=Path, required=True, help="Dataset output root")
    collect.add_argument(
        "--scene",
        required=True,
        help="Registered scene name (see a3_dual_arm_sim.collection.registered_scenes)",
    )
    collect.add_argument("--repo-id", default=None, help="Defaults to local/<scene>")
    collect.add_argument("--episodes", type=int, default=10)
    collect.add_argument("--max-attempts", type=int, default=None)
    collect.add_argument(
        "--position-noise",
        type=float,
        default=0.002,
        help="Per-cookie XY jitter in metres; ignored by scenes with a fixed layout",
    )
    collect.add_argument(
        "--yaw-noise",
        type=float,
        default=0.05,
        help="Per-cookie yaw jitter in radians",
    )
    collect.add_argument(
        "--hold-steps",
        type=int,
        default=40,
        help="Steps to keep recording after the expert finishes its last placement",
    )
    collect.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Index of this shard when several processes collect in parallel",
    )
    collect.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of shards; seeds are interleaved across them",
    )
    collect.add_argument(
        "--keep-failures",
        action="store_true",
        help="Also write episodes the environment did not accept; for diagnosis, "
        "not for training",
    )
    # Reproduce main's benchmark arrangement inside this branch, with a small perturbation so the
    # dataset covers more than one exact pose.  See `main_alignment.apply_scene`; the gate is
    # `scripts/verify_main_alignment.py`, and a millimetre matters (a 23 mm bin offset turned a
    # 5/10 policy into 0/20 on main's benchmark).
    collect.add_argument(
        "--main-aligned",
        type=Path,
        default=None,
        help="Path to a scene dump from dump_main_scene.py; every reset is rewritten onto it",
    )
    collect.add_argument(
        "--align-bin-noise",
        type=float,
        default=0.0,
        help="Noise on the position of both bins, on top of the aligned arrangement",
    )
    collect.add_argument(
        "--align-noise-dist",
        choices=("uniform", "normal"),
        default="uniform",
        help=(
            "Shape of the aligned-path noises.  `uniform` is main's: it draws "
            "uniform(-s, s) per axis, so s is a bound.  `normal` treats s as a sigma, "
            "which is not comparable to main's profile numbers."
        ),
    )
    collect.add_argument(
        "--align-noise-3d",
        action="store_true",
        help=(
            "Also perturb the vertical axis.  Off by default because main never does, "
            "and because this branch's source bin is a mocap body whose z offset cannot "
            "settle -- measured: no attempt with |z| > 4 mm was ever accepted."
        ),
    )
    collect.add_argument(
        "--align-cookie-yaw-noise",
        type=float,
        default=0.0,
        help=(
            "Sigma/bound in radians for a per-Cookie yaw re-draw around upright.  "
            "main's `basic` uses 0.015.  Without this the aligned hook keeps the dumped "
            "orientations verbatim, so `--yaw-noise` has no effect on the aligned path."
        ),
    )
    collect.add_argument(
        "--align-pose-noise",
        type=float,
        default=0.0,
        help="Gaussian sigma in radians applied to the arm's home joints",
    )
    collect.add_argument(
        "--align-bin-yaw-noise",
        type=float,
        default=0.0,
        help="Gaussian sigma in radians applied to the target box yaw",
    )
    _add_fast_render(collect)
    # Collection was the one command without this, on the grounds that pixels are a
    # dataset contract and the default should not drift.  That reasoning still holds
    # for the *default*, which stays plain; what it missed is that a dataset is
    # allowed to choose its own appearance deliberately, and for a fine-tune it must
    # -- a model conditioned on "yellow blocks" and "marked slots" has to be trained
    # on frames that show them.
    _add_visual_style(collect)
    _add_rollover(collect)
    collect.set_defaults(function=_collect_scene)

    train = commands.add_parser(
        "train-smolvla", help="Fine-tune SmolVLA on a validated A3 grasp dataset"
    )
    train.add_argument("--root", type=Path, required=True, help="LeRobot v3 dataset root")
    train.add_argument("--repo-id", default="local/a3-grasp")
    train.add_argument("--model", type=Path, default=default_base_model())
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seed", type=int, default=1000)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--job-name", default="a3_smolvla")
    train.add_argument(
        "--save-freq",
        type=int,
        default=2000,
        help="Steps between checkpoints; 0 saves only at the end",
    )
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(function=_train_smolvla)
    return root


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        _validate_camera_recording(args)
    except ValueError as exc:
        argument_parser.error(str(exc))
    exit_code = int(args.function(args))
    uses_human_viewer = args.command == "teleop" or bool(
        getattr(args, "render", False)
    )
    if uses_human_viewer:
        # On Python 3.13, combining mujoco.viewer with mujoco.Renderer can finish
        # explicit cleanup successfully and then SIGSEGV during native GLFW module
        # finalization.  At this point runner.close() has already flushed recordings
        # and closed both contexts, so bypass only the faulty second teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
