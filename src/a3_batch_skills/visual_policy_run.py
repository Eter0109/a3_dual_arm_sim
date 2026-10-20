"""Deploy an adapted left8 VLA using RGB-D goals and visual batch boundaries.

The benchmark's privileged score is logged separately after actions. It is never
passed to control. Optional RGB-D placement is explicitly a hybrid result.
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
from .visual_controller import VisualBatchController,SUPERVISOR_VERSION
from .rgbd_localization import TARGET_DETECTOR_VERSION


def main(argv=None,policy_factory=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--policy-module-dir",type=Path,required=True)
    p.add_argument("--camera-reference",type=Path,required=True)
    p.add_argument("--task",choices=("2x5","2x10"),default="2x5")
    p.add_argument("--profile",choices=("basic","medium","diverse"),default="basic")
    p.add_argument("--seed",type=int,default=26360)
    p.add_argument("--max-batches",type=int,help="Development scope only; never labels partial work as full task success")
    p.add_argument("--source-column",choices=("1","2","3","4","auto"),default="1")
    p.add_argument("--vision-interval",type=int)
    p.add_argument("--placement-control",choices=("vla","rgbd-servo"),default="vla")
    p.add_argument("--elevation",type=float,default=-65.)
    p.add_argument("--azimuth",type=float,default=180.)
    p.add_argument("--extra-rgbd-view",type=float,nargs=2,action="append",default=[],
                   metavar=("ELEVATION","AZIMUTH"),help="Additional calibrated depth view; policy RGB unchanged")
    p.add_argument("--video",action="store_true")
    p.add_argument("--record-placement-probes",action="store_true",
                   help="Save at most 32 RGB-D/proprioception packets; no physics snapshots")
    p.add_argument("--audit-placement-truth",action="store_true",
                   help="Log separate privileged labels AFTER decisions; never available to control")
    a=p.parse_args(argv)
    if a.vision_interval is None:a.vision_interval=5
    if a.output.exists():p.error("Never overwrite an evaluation")
    if a.vision_interval<1:p.error("Invalid vision interval")
    if a.max_batches is not None and not 1<=a.max_batches<={"2x5":2,"2x10":4}[a.task]:
        p.error('Invalid development batch scope')
    if a.audit_placement_truth and a.placement_control!='rgbd-servo':
        p.error("Placement truth audit is only for the optional hybrid diagnostic")
    if a.placement_control=="rgbd-servo" and not a.extra_rgbd_view:
        a.extra_rgbd_view=[[-25.,0.],[-25.,180.]]
    descriptor=json.loads((a.checkpoint/"goal_conditioning.json").read_text())
    if descriptor.get("goal_conditioning")!=VISUAL_MARKER_VERSION or not descriptor.get("training_completed"):
        p.error("This mode requires a checkpoint actually adapted on measured RGB-D batch cues")
    cfg=json.loads((a.checkpoint/"config.json").read_text())
    if any(cfg[part][key]["shape"]!=[8] for part,key in
           (("input_features","observation.state"),("output_features","action"))):
        p.error("Expected a left8-only continuation checkpoint")
    import torch
    if policy_factory is None and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<14*1024**3):
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
    sensor=policy=video=audit=None
    started=time.monotonic()
    try:
        env.task_config=replace(env.task_config,terminate_on_success=False)
        obs,info=env.reset(seed=a.seed,options=dict(randomize_cookies=benchmark.randomize_cookies,
            randomize_boxes=benchmark.randomize_boxes,randomize_source_bin=benchmark.randomize_boxes,
            randomize_target_bin=benchmark.randomize_boxes,source_bin_noise_m=benchmark.source_bin_noise_m,
            target_bin_noise_m=benchmark.target_bin_noise_m,target_bin_yaw_noise_rad=benchmark.target_bin_yaw_noise_rad,
            randomization_profile=a.profile,appearance_seed=a.seed,randomization_settings=benchmark.randomization_settings))
        servo=LeftArmServo(env)
        sensor=MujocoRGBDViews(env,views=((a.elevation,a.azimuth),(-90.,90.),(-65.,90.),
                                        *[tuple(v) for v in a.extra_rgbd_view]))
        spec=geometry_spec_from_env(env)
        policy=(policy_factory or importlib.import_module("pi05_adapter").Pi05BenchmarkPolicy)(a.checkpoint,a.job)
        torch.manual_seed(a.seed)
        visual=VisualGoalPolicy(policy,descriptor)
        column="auto" if a.source_column=="auto" else int(a.source_column)
        context=EpisodeContext(a.seed,"Transfer the marked five cookies, release, and retreat.",policy.action_mode)
        placement=None
        if a.placement_control=="rgbd-servo":
            from .rgbd_placement import RGBDPlacementPolicy
            placement=RGBDPlacementPolicy(visual,servo,spec)
        controller=VisualBatchController(placement or visual,spec,column=column,context=context)
        if placement:placement.scene_provider=lambda:controller.last_scene
        if placement:placement.budget_provider=lambda:max(1,controller.batch_steps-(controller.step-controller.batch_started)-30)
        if a.audit_placement_truth:
            from .placement_audit import PlacementTruthAudit
            audit=PlacementTruthAudit(a.output/'placement_truth_audit.jsonl')
        recorder=None
        if a.record_placement_probes:
            from .placement_observation import PlacementProbeRecorder
            recorder=PlacementProbeRecorder(a.output/"placement_probes",spec)
        if a.video:
            from .__main__ import Video
            video=Video(a.output/"rollout.mp4",env.config.control_hz//2,env.config.image_width,env.config.image_height)
        right_error=0.
        for step in range(budget):
            placing=placement and placement.phase in ("WAIT_VIEW","VISUAL_SERVO")
            frame=sensor.capture() if step%a.vision_interval==0 or placing else None
            pads=env.data.geom_xpos[list(env._left_finger_geoms)].copy()  # robot FK only
            gap=np.linalg.norm(pads[1]-pads[0])-2*env.config.cookie_transfer.left_finger_pad_half_thickness_m
            trace_count=len(placement.trace) if placement else 0
            if placement:
                from .rgbd_placement import RobotPlacementState
                placement.update(frame,RobotPlacementState(env.data.site_xpos[servo.site].copy(),
                    env.data.site_xmat[servo.site].reshape(3,3).copy(),pads.mean(0),float(gap),
                    float(env.last_applied_action[7]),float(env.data.time)))
            calibrations=camera_calibrations_from_fk(env)
            action=controller.act(obs,calibrations,frame=frame,
                pad_midpoint_world=pads.mean(0),jaw_gap_m=float(gap))
            if placement and placement.failure and not controller.finished:
                controller.require_recovery(placement.failure);action=None
            if recorder:
                recorder.add(step+1,frame,obs,controller.goal,
                    tool_position=env.data.site_xpos[servo.site].copy(),
                    tool_rotation=env.data.site_xmat[servo.site].reshape(3,3).copy(),
                    pad_midpoint=pads.mean(0),jaw_gap=float(gap),opening=float(env.last_applied_action[7]),
                    calibrations=calibrations)
            # Control has already returned. Privileged labels are written only
            # to a separate file, aligned with the same pre-action observation.
            if audit:
                pose=placement.last_pose
                if pose is not None and abs(pose.timestamp-float(env.data.time))>1e-6:pose=None
                audit.write_after_decision(env,placement.goal,pose,float(env.data.time))
            obs,_,terminated,truncated,info=env.step(servo.hold() if action is None else servo.canonical(action))
            if placement and len(placement.trace)>trace_count:
                placement.trace[-1].update(global_step=step+1,applied_left8=env.last_applied_action[:8].tolist())
            right_error=max(right_error,float(np.max(abs(env.last_applied_action[8:]-servo.right_hold))))
            if video and step%2==0:video.add(obs["observation.images.front"])
            if (step+1)%50==0:
                status=dict(step=step+1,phase=controller.phase,completed_batches=len(controller.completed),
                    native_count_evaluation_only=int(info.get("cookies_in_target",0)),
                    visual_diagnostics=controller.diagnostics)
                if placement:status.update(placement_phase=placement.phase,
                    placement_stage=placement.placement.stage if placement.placement else None,
                    placement_perception_error=placement.last_error,action_counts=placement.counts)
                (a.output/"status.json").write_text(json.dumps(status));print(json.dumps(status),flush=True)
            if controller.finished or terminated or truncated:break
            if a.max_batches is not None and len(controller.completed)>=a.max_batches:break
        result=dict(task=a.task,seed=a.seed,phase=controller.phase,completed_batches=len(controller.completed),
            failure_reason=controller.failure_reason,events=controller.events,interaction_steps=step+1,
            native_success_evaluation_only=bool(info.get("success",False)),
            native_count_evaluation_only=int(info.get("cookies_in_target",0)),
            localization=descriptor["localization"],goal_conditioning=VISUAL_MARKER_VERSION,
            target_detector=TARGET_DETECTOR_VERSION,supervisor=SUPERVISOR_VERSION,
            visual_diagnostics=controller.diagnostics,
            action_source="vla_transport_rgbd_servo_placement" if placement else "single_adapted_left8_vla",
            geometric_takeover=bool(placement and placement.counts['visual_servo']),
            object_truth_used_for_decisions=False,physics_resets_between_batches=0,
            right_target_max_error=right_error,sensor=sensor.protocol,n_action_steps=policy.config.n_action_steps,
            checkpoint=str(a.checkpoint),camera_reference=str(a.camera_reference),wall_seconds=time.monotonic()-started)
        result["placement_probe_packets"]=recorder.count if recorder else 0
        result['placement_control']=a.placement_control
        result['placement_truth_audit_after_decisions']=bool(audit)
        result['evaluation_scope_batches']=a.max_batches or {'2x5':2,'2x10':4}[a.task]
        result['requested_scope_success']=len(controller.completed)>=result['evaluation_scope_batches']
        if placement:
            from .rgbd_placement import PLACEMENT_VERSION
            from .rgbd_held import HELD_POSE_VERSION
            result.update(placement_version=PLACEMENT_VERSION,held_pose_version=HELD_POSE_VERSION,
                placement_events=placement.events,action_counts=placement.counts,
                placement_stage=placement.placement.stage if placement.placement else None,
                placement_perception_error=placement.last_error,
                last_batch_placement_transitions=placement.placement.events if placement.placement else [])
            with (a.output/'placement_trace.jsonl').open('w') as stream:
                for row in placement.trace:stream.write(json.dumps(row)+'\n')
        (a.output/"result.json").write_text(json.dumps(result,indent=2))
        return 0 if result['requested_scope_success'] else 2
    finally:
        if audit:audit.close()
        if video:video.close()
        if policy:policy.close()
        if sensor:sensor.close()
        env.close()


if __name__=="__main__":raise SystemExit(main())
