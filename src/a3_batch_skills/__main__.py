"""Run the new batch controller against the current, unchanged A3 scene."""
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .contracts import BatchConfig, Phase
from .cameras import apply_camera_reference
from .controller import BatchSkillController
from .executors import ReferenceBatchExecutor, PolicyBatchExecutor
from .grounding import GroundedBatchExecutor
from .motion import LeftArmServo
from .perception import SimulatorSceneProvider
from .verification import inspect_placement


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class Video:
    def __init__(self, path, fps, width, height):
        import av
        self.av = av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"

    def add(self, rgb):
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("2x5", "2x10"), default="2x5")
    parser.add_argument("--seed", type=int, default=26124)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor", choices=("grounded", "expert", "pi05"), default="grounded")
    parser.add_argument("--profile", choices=("basic", "medium", "diverse"), default="basic")
    parser.add_argument("--placement-mode", choices=("strict", "contained_unstacked"), default="contained_unstacked")
    parser.add_argument("--source-column", default="1", choices=("1", "2", "3", "4", "auto"))
    parser.add_argument("--no-adjust", action="store_true")
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument("--prealign", action="store_true", help="Experimental physical pre-grasp normalization")
    alignment.add_argument("--no-prealign", action="store_true", help="Keep the policy's current entry pose (default)")
    parser.add_argument("--max-adjustments", type=int, default=1)
    parser.add_argument("--exercise-adjustment", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-steps", type=int, default=750)
    parser.add_argument("--camera-reference", type=Path)
    parser.add_argument("--render-protocol", choices=("native", "single_sample"), default="native")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-module-dir", type=Path)
    parser.add_argument("--policy-scope", choices=("pick", "transfer"), default="transfer")
    parser.add_argument("--policy-prompt-file", type=Path,
                        help="Explicit prompt; preserves training wording for clean policy comparisons")
    parser.add_argument("--n-action-steps", type=int,
                        help="Omit to retain the checkpoint's own execution horizon")
    parser.add_argument("--goal-conditioned", action="store_true",
                        help="Requires goal_conditioning.json from actual skill adaptation in the checkpoint")
    parser.add_argument("--record-skills", action="store_true",
                        help="Record each grounded batch as a separate lossless RGB/left8 training clip")
    parser.add_argument("--unmarked-skills", action="store_true",
                        help="Record raw views instead of explicit projected goal cues")
    parser.add_argument("--rgbd-goals",action="store_true",
        help="Record independently measured, latched RGB-D goal cues; teacher still uses truth for actions")
    parser.add_argument("--rgbd-elevation",type=float,default=-65.)
    parser.add_argument("--rgbd-azimuth",type=float,default=180.)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.n_action_steps is not None and args.n_action_steps < 1:
        parser.error("--n-action-steps must be positive")
    if args.output.exists():
        parser.error("Output already exists; use a distinct experiment directory")
    if args.record_skills and args.executor != "grounded":
        parser.error("Skill demonstrations currently require the goal-bound grounded executor")
    if args.rgbd_goals and (not args.record_skills or args.unmarked_skills):
        parser.error("--rgbd-goals requires --record-skills and marked images")
    if args.goal_conditioned and args.executor != "pi05":
        parser.error("--goal-conditioned applies to an adapted pi05 checkpoint")
    if args.executor == "pi05" and not all((args.job, args.checkpoint, args.camera_reference,
                                            args.policy_module_dir)):
        parser.error("pi05 requires --job, --checkpoint, --policy-module-dir and --camera-reference")
    if args.executor == "pi05" and args.job:
        os.environ.setdefault("HF_HOME", str(args.job / "hf"))
        os.environ["HF_DATASETS_CACHE"] = str(args.job / "cache/batch_skills")
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    options = {"first_grasp": 5} if args.task == "2x10" else {}
    max_steps = args.max_steps or {"2x5": 1500, "2x10": 3000}[args.task]
    column = "1" if args.source_column == "auto" else args.source_column
    benchmark = make_cookie_benchmark(args.task, profile=args.profile, source_column=column,
                                      max_steps=max_steps, **options)
    if args.camera_reference:
        apply_camera_reference(benchmark, args.camera_reference)
    env = benchmark.create_env(render_cameras=args.executor == "pi05" or args.record_skills)
    policy, video, recorder, visual_sensor = None, None, None, None
    selected_success = False
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        if args.render_protocol == "single_sample":
            env.model.vis.quality.offsamples = 0
        observation, info = env.reset(seed=args.seed, options={
            "randomize_cookies": benchmark.randomize_cookies,
            "randomize_boxes": benchmark.randomize_boxes,
            "randomize_source_bin": benchmark.randomize_boxes,
            "randomize_target_bin": benchmark.randomize_boxes,
            "source_bin_noise_m": benchmark.source_bin_noise_m,
            "target_bin_noise_m": benchmark.target_bin_noise_m,
            "target_bin_yaw_noise_rad": benchmark.target_bin_yaw_noise_rad,
            "randomization_profile": args.profile, "appearance_seed": args.seed,
            "randomization_settings": benchmark.randomization_settings})
        config = BatchConfig(placement_mode=args.placement_mode, adjust_placement=not args.no_adjust,
            align_policy_entry=args.prealign,
            max_adjustments=args.max_adjustments, max_batch_steps=args.batch_steps,
            source_column="auto" if args.source_column == "auto" else int(args.source_column),
            policy_scope=args.policy_scope,retreat_clearance_m=.160 if args.rgbd_goals else .060)
        servo = LeftArmServo(env)
        provider = SimulatorSceneProvider(env)
        if args.executor == "grounded":
            executor = GroundedBatchExecutor(env, servo, config, args.seed)
        elif args.executor == "expert":
            executor = ReferenceBatchExecutor(env, servo, config, args.seed)
        else:
            sys.path.insert(0, str(args.policy_module_dir))
            import torch
            torch.set_num_threads(2)
            torch.manual_seed(args.seed)
            torch.use_deterministic_algorithms(True)
            if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 14 * 1024**3:
                raise RuntimeError("At least 14 GiB free GPU memory is required before loading pi05")
            write_json(args.output / "status.json", dict(status="loading_policy", interaction_steps=0))
            policy = importlib.import_module("pi05_adapter").Pi05BenchmarkPolicy(args.checkpoint, args.job)
            if args.goal_conditioned:
                from .markers import GoalMarkedPolicy
                metadata = json.loads((args.checkpoint/"goal_conditioning.json").read_text())
                policy = GoalMarkedPolicy(policy, env, provider, metadata)
            if args.n_action_steps is not None:
                if args.n_action_steps > policy.config.chunk_size:
                    raise ValueError("n_action_steps exceeds the model action chunk")
                policy.config.n_action_steps = args.n_action_steps
            prompt = args.policy_prompt_file.read_text(encoding="utf-8").strip() if args.policy_prompt_file else None
            if args.policy_prompt_file and not prompt:
                raise ValueError("The policy prompt must not be empty")
            executor = PolicyBatchExecutor(policy, servo, config, args.seed, instruction_override=prompt)
        controller = BatchSkillController(env, provider, servo, executor,
            config=config, exercise_adjustment=args.exercise_adjustment)
        if args.record_skills:
            from .recording import SkillClipRecorder
            visual_spec=None
            if args.rgbd_goals:
                from .rgbd import MujocoRGBDViews,geometry_spec_from_env
                visual_sensor=MujocoRGBDViews(env,views=((args.rgbd_elevation,args.rgbd_azimuth),(-90.,90.),(-65.,90.)))
                visual_spec=geometry_spec_from_env(env)
            recorder = SkillClipRecorder(args.output/"skill_clips", env, seed=args.seed,
                marked=not args.unmarked_skills,visual_sensor=visual_sensor,visual_spec=visual_spec,
                source_column=config.source_column,provenance=dict(task=args.task,profile=args.profile,
                    source_column=args.source_column,executor=executor.name,
                    camera_sha256=hashlib.sha256(args.camera_reference.read_bytes()).hexdigest() if args.camera_reference else None,
                    render_protocol=args.render_protocol,placement_mode=args.placement_mode))
        if args.video:
            video = Video(args.output / "rollout.mp4", env.config.control_hz // 2,
                          env.config.image_width, env.config.image_height)
        states, actions, phases = [], [], []
        native_ever, right_error = False, 0.
        for step in range(max_steps):
            previous_observation = observation
            previous_phase = controller.phase.value
            states.append(np.asarray(observation["observation.state"][:8]).copy())
            phases.append(controller.phase.value)
            action = controller.act(observation, info)
            prepared = recorder is not None and hasattr(controller, "goal")
            if prepared:
                recorder.prepare(previous_observation, controller.goal, provider.observe(previous_observation),
                                 step, previous_phase)
            observation, _, terminated, truncated, info = env.step(action)
            if prepared:
                recorder.commit(env.last_applied_action[:8],
                    batch_completed=controller.batch_index > controller.goal.index,
                    failed=controller.phase == Phase.FAILED, reason=controller.failure_reason)
            actions.append(env.last_applied_action[:8].copy())
            right_error = max(right_error, float(np.max(np.abs(env.last_applied_action[8:] - servo.right_hold))))
            native_ever |= bool(info.get("success", False))
            if video and step % 2 == 0:
                if env.render_cameras:
                    rgb = observation["observation.images.front"]
                else:
                    env.render_cameras = True
                    try:
                        rgb = env._render_camera("front")
                    finally:
                        env.render_cameras = False
                video.add(rgb)
            if (step + 1) % 50 == 0:
                status = dict(step=step + 1, phase=controller.phase.value,
                    completed_batches=controller.batch_index,
                    native_count=int(info.get("cookies_in_target", 0)),
                    placement_stage=getattr(getattr(controller, "placer", None), "stage", None),
                    pick_alignment_stage=getattr(getattr(controller, "pick_aligner", None), "stage", None),
                    adjustment_stage=getattr(getattr(controller, "adjuster", None), "stage", None))
                status["executor_diagnostics"] = getattr(executor, "diagnostics", None)
                write_json(args.output / "status.json", status)
                print(json.dumps(status), flush=True)
            if controller.finished or terminated or truncated:
                break
        result = controller.summary()
        native_final = bool(info.get("success", False) and not env._safety_stop)
        selected_success = controller.phase == Phase.DONE and not env._safety_stop
        if controller.report is not None:
            final_report = inspect_placement(controller.provider.observe(observation), controller.goal,
                controller.completed, config)
            selected_success &= (final_report.complete and
                                 len(final_report.accepted_ids) == env.task_config.required_cookies)
            result["final_accepted_ids"] = sorted(final_report.accepted_ids)
        if config.placement_mode == "strict":
            selected_success &= native_final
        result.update(task=args.task, seed=args.seed, selected_success=selected_success,
            profile=args.profile,
            native_success=native_final, native_success_ever=native_ever,
            native_count=int(info.get("cookies_in_target", 0)),
            native_source_count=int(info.get("cookies_in_source", 0)),
            interaction_steps=step + 1, wall_seconds=time.monotonic() - started,
            budget_exhausted=step + 1 >= max_steps and not controller.finished,
            executor=executor.name, localization="simulator_truth",
            camera_reference=str(args.camera_reference) if args.camera_reference else "main_defaults",
            camera_sha256=hashlib.sha256(args.camera_reference.read_bytes()).hexdigest() if args.camera_reference else None,
            render_protocol=args.render_protocol, right_target_max_error=right_error,
            learned_action_dim=8, policy_training_steps=0, physics_resets_between_batches=0,
            native_scoring_modified=False, placement_protocol="contained-unstacked-floor-clearance-v2")
        result["policy_prompt"] = executor.instruction if args.executor == "pi05" else None
        result["n_action_steps"] = policy.config.n_action_steps if policy is not None else None
        result["goal_binding"] = getattr(executor, "goal_binding", "unconditioned_or_reference")
        result["skill_clips"] = len(recorder.clips) if recorder else 0
        if result["budget_exhausted"] and not result["failure_reason"]:
            result["failure_reason"] = "Total control-step budget exhausted"
        np.savez_compressed(args.output / "left8_trace.npz", state=np.asarray(states),
                            action=np.asarray(actions), phase=np.asarray(phases))
        write_json(args.output / "result.json", result)
        write_json(args.output / "status.json", dict(status="complete", success=selected_success,
            phase=controller.phase.value, interaction_steps=step + 1))
        print(json.dumps({k: result[k] for k in ("selected_success", "phase", "failure_reason",
            "completed_batches", "interaction_steps", "native_count")}), flush=True)
        return 0 if selected_success else 2
    except Exception as exc:
        write_json(args.output / "error.json", dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if recorder:
            recorder.close(task_success=selected_success)
        if video:
            video.close()
        if policy:
            policy.close()
        if visual_sensor:
            visual_sensor.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
