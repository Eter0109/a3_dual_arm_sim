"""Deploy an adapted left8 VLA using RGB-D goals and visual batch boundaries.

The benchmark's privileged score is logged separately after actions. It is never
passed to the policy or boundary supervisor. There is no expert takeover.
"""
import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from .cameras import apply_camera_reference
from .motion import LeftArmServo
from .rgbd import MujocoRGBDViews,geometry_spec_from_env
from .visual_goals import VisualGoalPolicy,camera_calibrations_from_fk,VISUAL_MARKER_VERSION
from .visual_controller import VisualBatchController


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--policy-module-dir",type=Path,required=True)
    p.add_argument("--camera-reference",type=Path,required=True)
    p.add_argument("--task",choices=("2x5","2x10"),default="2x5")
    p.add_argument("--profile",choices=("basic","medium","diverse"),default="basic")
    p.add_argument("--seed",type=int,default=26360)
    p.add_argument("--source-column",choices=("1","2","3","4","auto"),default="1")
    p.add_argument("--vision-interval",type=int,default=5)
    p.add_argument("--elevation",type=float,default=-65.)
    p.add_argument("--azimuth",type=float,default=180.)
    p.add_argument("--video",action="store_true")
    a=p.parse_args()
    if a.output.exists():p.error("Never overwrite an evaluation")
    if a.vision_interval<1:p.error("Invalid vision interval")
    descriptor=json.loads((a.checkpoint/"goal_conditioning.json").read_text())
    if descriptor.get("goal_conditioning")!=VISUAL_MARKER_VERSION or not descriptor.get("training_completed"):
        p.error("This mode requires a checkpoint actually adapted on measured RGB-D batch cues")
    cfg=json.loads((a.checkpoint/"config.json").read_text())
    if any(cfg[part][key]["shape"]!=[8] for part,key in
           (("input_features","observation.state"),("output_features","action"))):
        p.error("Expected a left8-only continuation checkpoint")
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<14*1024**3:
        p.error("At least 14 GiB free GPU memory required")
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    sys.path.insert(0,str(a.policy_module_dir))
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from a3_dual_arm_sim.contracts import EpisodeContext
    budget={"2x5":1500,"2x10":3000}[a.task]
    benchmark=make_cookie_benchmark(a.task,first_grasp=5 if a.task=="2x10" else None,
        profile=a.profile,source_column="1" if a.source_column=="auto" else a.source_column,max_steps=budget)
    apply_camera_reference(benchmark,a.camera_reference)
    env=benchmark.create_env(render_cameras=True)
    a.output.mkdir(parents=True)
    sensor=policy=video=None
    started=time.monotonic()
    try:
        env.task_config=replace(env.task_config,terminate_on_success=False)
        obs,info=env.reset(seed=a.seed,options=dict(randomize_cookies=benchmark.randomize_cookies,
            randomize_boxes=benchmark.randomize_boxes,randomize_source_bin=benchmark.randomize_boxes,
            randomize_target_bin=benchmark.randomize_boxes,source_bin_noise_m=benchmark.source_bin_noise_m,
            target_bin_noise_m=benchmark.target_bin_noise_m,target_bin_yaw_noise_rad=benchmark.target_bin_yaw_noise_rad,
            randomization_profile=a.profile,appearance_seed=a.seed,randomization_settings=benchmark.randomization_settings))
        servo=LeftArmServo(env)
        sensor=MujocoRGBDViews(env,views=((a.elevation,a.azimuth),(-90.,90.),(-65.,90.)))
        spec=geometry_spec_from_env(env)
        policy=importlib.import_module("pi05_adapter").Pi05BenchmarkPolicy(a.checkpoint,a.job)
        torch.manual_seed(a.seed)
        visual=VisualGoalPolicy(policy,descriptor)
        column="auto" if a.source_column=="auto" else int(a.source_column)
        context=EpisodeContext(a.seed,"Transfer the marked five cookies, release, and retreat.",policy.action_mode)
        controller=VisualBatchController(visual,spec,column=column,context=context)
        if a.video:
            from .__main__ import Video
            video=Video(a.output/"rollout.mp4",env.config.control_hz//2,env.config.image_width,env.config.image_height)
        right_error=0.
        for step in range(budget):
            frame=sensor.capture() if step%a.vision_interval==0 else None
            pads=env.data.geom_xpos[list(env._left_finger_geoms)].copy()  # robot FK only
            gap=np.linalg.norm(pads[1]-pads[0])-2*env.config.cookie_transfer.left_finger_pad_half_thickness_m
            action=controller.act(obs,camera_calibrations_from_fk(env),frame=frame,
                pad_midpoint_world=pads.mean(0),jaw_gap_m=float(gap))
            obs,_,terminated,truncated,info=env.step(servo.hold() if action is None else servo.canonical(action))
            right_error=max(right_error,float(np.max(abs(env.last_applied_action[8:]-servo.right_hold))))
            if video and step%2==0:video.add(obs["observation.images.front"])
            if (step+1)%50==0:
                status=dict(step=step+1,phase=controller.phase,completed_batches=len(controller.completed),
                    native_count_evaluation_only=int(info.get("cookies_in_target",0)))
                (a.output/"status.json").write_text(json.dumps(status));print(json.dumps(status),flush=True)
            if controller.finished or terminated or truncated:break
        result=dict(task=a.task,seed=a.seed,phase=controller.phase,completed_batches=len(controller.completed),
            failure_reason=controller.failure_reason,events=controller.events,interaction_steps=step+1,
            native_success_evaluation_only=bool(info.get("success",False)),
            native_count_evaluation_only=int(info.get("cookies_in_target",0)),
            localization=descriptor["localization"],goal_conditioning=VISUAL_MARKER_VERSION,
            action_source="single_adapted_left8_vla",geometric_takeover=False,
            object_truth_used_for_decisions=False,physics_resets_between_batches=0,
            right_target_max_error=right_error,sensor=sensor.protocol,n_action_steps=policy.config.n_action_steps,
            checkpoint=str(a.checkpoint),camera_reference=str(a.camera_reference),wall_seconds=time.monotonic()-started)
        (a.output/"result.json").write_text(json.dumps(result,indent=2))
        return 0 if controller.phase=="DONE" else 2
    finally:
        if video:video.close()
        if policy:policy.close()
        if sensor:sensor.close()
        env.close()


if __name__=="__main__":raise SystemExit(main())
