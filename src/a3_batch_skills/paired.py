"""Clean pi05 comparison: unchanged full-task execution vs boundary queue clearing.

One model is loaded and reused serially. No geometric placement, changed prompt,
physical prealignment, simulator restore, or per-boundary RNG reseed is added.
"""
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
from .__main__ import Video, write_json
from .cameras import apply_camera_reference
from .contracts import BatchConfig
from .perception import SimulatorSceneProvider
from .verification import accepted_ids


class BoundaryMonitor:
    def __init__(self, required, stable_steps=20):
        self.required, self.stable_steps = required, stable_steps
        self.previous, self.stable = frozenset(), 0
        self.completed, self.events = set(), []
        self.peak_count = 0
        self.completed_once = set()

    def observe(self, scene, config, pad_position, step):
        accepted = accepted_ids(scene, config)
        self.peak_count = max(self.peak_count, len(accepted))
        self.stable = self.stable+1 if accepted == self.previous else 1
        self.previous = accepted
        self.completed_once.update(accepted)
        new = sorted(accepted-self.completed)
        boundary = None
        if self.stable >= self.stable_steps and len(new) >= 5:
            group = new[:5]
            center = np.asarray([scene.cookies[i].position for i in group]).mean(0)
            clear = np.linalg.norm(np.asarray(pad_position)-center) >= .040
            if clear and not scene.held_ids:
                self.completed.update(group)
                boundary = dict(event="verified_released_boundary", step=step,
                                batch=len(self.completed)//5, cookie_ids=group)
                self.events.append(boundary)
        done = len(accepted) >= self.required and self.stable >= self.stable_steps and not scene.held_ids
        return boundary, done, accepted


def run_case(policy, args, task, seed, mode):
    from a3_dual_arm_sim.contracts import EpisodeContext
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    budget = 1500 if task == "2x5" else 3000
    benchmark = make_cookie_benchmark(task, first_grasp=5 if task == "2x10" else None,
        profile=args.profile, source_column=args.source_column, max_steps=budget)
    apply_camera_reference(benchmark, args.camera_reference)
    env = benchmark.create_env(render_cameras=True)
    root = args.output/f"{task}_s{seed}_{mode}"
    root.mkdir(exist_ok=False)
    video = None
    started = time.monotonic()
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        env.model.vis.quality.offsamples = 0
        observation, info = env.reset(seed=seed, options=dict(
            randomize_cookies=benchmark.randomize_cookies, randomize_boxes=benchmark.randomize_boxes,
            randomize_source_bin=benchmark.randomize_boxes, randomize_target_bin=benchmark.randomize_boxes,
            source_bin_noise_m=benchmark.source_bin_noise_m, target_bin_noise_m=benchmark.target_bin_noise_m,
            target_bin_yaw_noise_rad=benchmark.target_bin_yaw_noise_rad,
            randomization_profile=args.profile, appearance_seed=seed,
            randomization_settings=benchmark.randomization_settings))
        digest = hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()).hexdigest()
        rgb_digest = hashlib.sha256(b"".join(observation["observation.images."+k].tobytes()
            for k in ("front","left_wrist","right_wrist"))).hexdigest()
        right_hold = env.last_applied_action[8:].copy()
        policy.reset(EpisodeContext(seed, args.prompt, policy.action_mode))
        provider, config = SimulatorSceneProvider(env), BatchConfig()
        monitor = BoundaryMonitor(env.task_config.required_cookies)
        if args.video:
            video = Video(root/"rollout.mp4",env.config.control_hz//2,env.config.image_width,env.config.image_height)
        states, actions, clear_steps = [], [], []
        right_error, native_ever, done = 0., False, False
        for step in range(budget):
            scene = provider.observe(observation)
            pads = env.data.geom_xpos[list(env._left_finger_geoms)].mean(0)
            boundary, done, accepted = monitor.observe(scene, config, pads, step)
            if done or scene.safety_stop:
                break
            if boundary and mode == "queue_reset":
                # Call the neural queue reset, not the adapter's reseeding reset.
                policy.policy.reset()
                clear_steps.append(step)
            states.append(np.asarray(observation["observation.state"][:8]).copy())
            action = np.asarray(policy.act(observation,args.prompt)).copy()
            action[8:] = right_hold
            observation, _, terminated, truncated, info = env.step(action)
            actions.append(env.last_applied_action[:8].copy())
            right_error = max(right_error,float(np.max(abs(env.last_applied_action[8:]-right_hold))))
            native_ever |= bool(info.get("success"))
            if video and step % 2 == 0:
                video.add(observation["observation.images.front"])
            if (step+1) % 100 == 0:
                write_json(root/"status.json",dict(step=step+1,mode=mode,accepted=len(accepted),queue_clears=clear_steps))
                print(json.dumps(dict(task=task,seed=seed,mode=mode,step=step+1,accepted=len(accepted))),flush=True)
            if terminated or truncated:
                break
        final_scene = provider.observe(observation)
        _, done, accepted = monitor.observe(final_scene,config,
            env.data.geom_xpos[list(env._left_finger_geoms)].mean(0),len(actions))
        source = np.asarray(env.SOURCE_POSITIONS)
        x = np.unique(source[:,0])[int(args.source_column)-1]
        allowed = set(np.flatnonzero(np.isclose(source[:,0],x)).tolist())
        wrong_source = sorted(set(accepted)-allowed)
        result = dict(task=task,seed=seed,mode=mode,profile=args.profile,
            max_steps=budget,interaction_steps=len(actions),wall_seconds=time.monotonic()-started,
            selected_success=bool(done and not final_scene.safety_stop and not wrong_source),
            accepted_ids=sorted(accepted),peak_accepted_count=monitor.peak_count,
            native_count=int(info.get("cookies_in_target",0)),native_success=bool(info.get("success",False)),
            native_success_ever=native_ever,previously_accepted_finally_lost=sorted(monitor.completed_once-set(accepted)),
            completed_batch_ids=sorted(monitor.completed),events=monitor.events,queue_clear_steps=clear_steps,
            prompt=args.prompt,prompt_sha256=hashlib.sha256(args.prompt.encode()).hexdigest(),
            camera_sha256=hashlib.sha256(args.camera_reference.read_bytes()).hexdigest(),
            initial_physics_sha256=digest,initial_rgb_sha256=rgb_digest,
            n_action_steps=policy.config.n_action_steps,chunk_size=policy.config.chunk_size,
            num_inference_steps=policy.config.num_inference_steps,
            source_column=args.source_column,wrong_source_ids=wrong_source,
            boundary_rng_reseed=False,geometric_takeover=False,physical_prealignment=False,
            physics_resets_between_batches=0,right_target_max_error=right_error,
            localization="simulator_truth_for_boundary_control_and_scoring" if mode == "queue_reset" else "simulator_truth_for_scoring_only",
            render_protocol="single_sample",
            selection_metric="contained_unstacked_stable20_released",policy_training_steps=0)
        np.savez_compressed(root/"left8_trace.npz",state=np.asarray(states),action=np.asarray(actions))
        write_json(root/"result.json",result)
        return result
    finally:
        if video:
            video.close()
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--policy-module-dir",type=Path,required=True)
    p.add_argument("--camera-reference",type=Path,required=True)
    p.add_argument("--prompt-file",type=Path,required=True)
    p.add_argument("--tasks",nargs="+",choices=("2x5","2x10"),default=["2x5","2x10"])
    p.add_argument("--seeds",nargs="+",type=int,default=[26220])
    p.add_argument("--profile",choices=("basic","medium","diverse"),default="basic")
    p.add_argument("--source-column",choices=("1","2","3","4"),default="1")
    p.add_argument("--n-action-steps",type=int)
    p.add_argument("--video",action="store_true")
    a = p.parse_args()
    a.prompt = a.prompt_file.read_text(encoding="utf-8").strip()
    if not a.prompt:
        p.error("Empty original prompt")
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("HF_HOME",str(a.job/"hf"))
    sys.path.insert(0,str(a.policy_module_dir))
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 14*1024**3:
        raise RuntimeError("At least 14 GiB free before loading the single shared model")
    policy = importlib.import_module("pi05_adapter").Pi05BenchmarkPolicy(a.checkpoint,a.job)
    rows = []
    try:
        if a.n_action_steps is not None:
            if not 1 <= a.n_action_steps <= policy.config.chunk_size:
                p.error("Invalid action execution horizon")
            policy.config.n_action_steps = a.n_action_steps
        write_json(a.output/"protocol.json",dict(prompt=a.prompt,
            prompt_source=str(a.prompt_file),camera_sha256=hashlib.sha256(a.camera_reference.read_bytes()).hexdigest(),
            n_action_steps=policy.config.n_action_steps,chunk_size=policy.config.chunk_size,
            tasks=a.tasks,seeds=a.seeds,modes=["continuous","queue_reset"],models_loaded=1,
            phase="development",note="2x10 uses the same supplied training prompt as its paired control; report this transfer protocol explicitly."))
        for task in a.tasks:
            for seed in a.seeds:
                pair = []
                for mode in ("continuous","queue_reset"):
                    row = run_case(policy,a,task,seed,mode)
                    pair.append(row)
                    rows.append(row)
                    write_json(a.output/"results.json",rows)
                invariant_keys = ("initial_physics_sha256","initial_rgb_sha256","prompt_sha256","camera_sha256",
                                  "max_steps","n_action_steps","chunk_size","num_inference_steps")
                check = {key:pair[0][key] == pair[1][key] for key in invariant_keys}
                write_json(a.output/f"{task}_s{seed}_pair_check.json",check)
                if not all(check.values()):
                    raise RuntimeError("Paired protocol mismatch: "+repr(check))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
