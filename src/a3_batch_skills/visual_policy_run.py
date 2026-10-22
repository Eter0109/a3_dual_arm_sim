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
    p.add_argument('--record-training-clips',action='store_true',help='Full sensor/action clips; visual/native acceptance required')
    p.add_argument('--n-action-steps',type=int,help='Explicit deployment execution horizon')
    p.add_argument("--source-column",choices=("1","2","3","4","auto"),default="1")
    p.add_argument("--vision-interval",type=int)
    p.add_argument("--placement-control",choices=("vla","rgbd-servo"),default="vla")
    p.add_argument('--observation-mode',choices=('passive','active-left-wrist'),default='passive')
    p.add_argument('--placement-mode',choices=('strict','contained_unstacked'),default='strict')
    p.add_argument('--sensor-mode',choices=('legacy-fixed','native-three'),default='legacy-fixed')
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
    modern=a.sensor_mode=='native-three' or a.observation_mode!='passive' or a.placement_mode!='strict'
    if a.observation_mode=='active-left-wrist' and a.sensor_mode!='native-three':p.error('Active wrist observation requires native-three depth')
    if modern and a.placement_control!='vla':p.error('New observation/acceptance modes keep all manipulation actions in the VLA')
    geometry_weights=(a.checkpoint/'geometry_conditioning.json').exists()
    if a.vision_interval is None:a.vision_interval=1 if geometry_weights or a.record_training_clips else 5
    if a.output.exists():p.error("Never overwrite an evaluation")
    if a.vision_interval<1:p.error("Invalid vision interval")
    if a.max_batches is not None and not 1<=a.max_batches<={"2x5":2,"2x10":4}[a.task]:
        p.error('Invalid development batch scope')
    if a.audit_placement_truth and a.placement_control!='rgbd-servo':
        p.error("Placement truth audit is only for the optional hybrid diagnostic")
    if (a.placement_control=="rgbd-servo" or geometry_weights or a.record_training_clips) and not a.extra_rgbd_view:
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
    sensor=policy=video=audit=training_recorder=control_log=None
    started=time.monotonic()
    try:
        env.task_config=replace(env.task_config,terminate_on_success=False)
        if a.sensor_mode=='native-three':
            from .native_sensor import NativeThreeRGBD
            sensor=NativeThreeRGBD(env)
        obs,info=env.reset(seed=a.seed,options=dict(randomize_cookies=benchmark.randomize_cookies,
            randomize_boxes=benchmark.randomize_boxes,randomize_source_bin=benchmark.randomize_boxes,
            randomize_target_bin=benchmark.randomize_boxes,source_bin_noise_m=benchmark.source_bin_noise_m,
            target_bin_noise_m=benchmark.target_bin_noise_m,target_bin_yaw_noise_rad=benchmark.target_bin_yaw_noise_rad,
            randomization_profile=a.profile,appearance_seed=a.seed,randomization_settings=benchmark.randomization_settings))
        servo=LeftArmServo(env)
        if sensor is None:sensor=MujocoRGBDViews(env,views=((a.elevation,a.azimuth),(-90.,90.),(-65.,90.),
                                        *[tuple(v) for v in a.extra_rgbd_view]))
        spec=geometry_spec_from_env(env)
        if policy_factory is None:
            if geometry_weights:
                from .geometry_adapter import GeometryPi05Adapter
                factory=GeometryPi05Adapter
            else:factory=importlib.import_module("pi05_adapter").Pi05BenchmarkPolicy
        else:factory=policy_factory
        policy=factory(a.checkpoint,a.job)
        if a.n_action_steps is not None:
            if not 1<=a.n_action_steps<=policy.config.chunk_size:p.error('Invalid execution horizon')
            policy.config.n_action_steps=a.n_action_steps
        torch.manual_seed(a.seed)
        geometry=None
        if geometry_weights or a.record_training_clips:
            from .geometry_context import GeometryContext
            geometry=GeometryContext(spec)
        visual=VisualGoalPolicy(policy,descriptor,geometry_context=geometry)
        column="auto" if a.source_column=="auto" else int(a.source_column)
        context=EpisodeContext(a.seed,"Transfer the marked five cookies, release, and retreat.",policy.action_mode)
        placement=None
        if a.placement_control=="rgbd-servo":
            from .rgbd_placement import RGBDPlacementPolicy
            placement=RGBDPlacementPolicy(visual,servo,spec)
        localizer=None
        if modern:
            from .visual_fusion import FusedRGBDLocalizer
            localizer=FusedRGBDLocalizer(spec,allow_fallen=a.placement_mode=='contained_unstacked')
        controller=VisualBatchController(placement or visual,spec,column=column,context=context,
            localizer=localizer,placement_mode=a.placement_mode,observation_mode=a.observation_mode,modern=modern)
        observer=None
        if a.observation_mode=='active-left-wrist':
            from .active_observation import LeftWristObservation
            observer=LeftWristObservation(sensor,env)
        control_log=(a.output/'control.jsonl').open('w',buffering=1)
        action_sources=dict(model=0,observation_move=0,hold=0);snapshot_count=0;last_phase=None
        if geometry:geometry.scene_provider=lambda:controller.last_scene
        if placement:placement.scene_provider=lambda:controller.last_scene
        if placement and geometry:geometry.observed_provider=lambda:placement.last_pose
        if placement:placement.budget_provider=lambda:max(1,controller.batch_steps-(controller.step-controller.batch_started)-30)
        if a.audit_placement_truth:
            from .placement_audit import PlacementTruthAudit
            audit=PlacementTruthAudit(a.output/'placement_truth_audit.jsonl')
        recorder=None
        if a.record_placement_probes:
            from .placement_observation import PlacementProbeRecorder
            recorder=PlacementProbeRecorder(a.output/"placement_probes",spec)
        if a.record_training_clips:
            from .sensor_recording import SensorPolicyClipRecorder
            training_recorder=SensorPolicyClipRecorder(a.output/'skill_clips',env.config.control_hz,
                provenance=dict(task=a.task,profile=a.profile,seed=a.seed,checkpoint=str(a.checkpoint),
                    placement_control=a.placement_control,sensor=sensor.protocol,full_sensor_trajectory=True))
        if a.video:
            from .__main__ import Video
            video=Video(a.output/"rollout.mp4",env.config.control_hz//2,env.config.image_width,env.config.image_height)
        right_error=0.
        for step in range(budget):
            placing=placement and placement.phase in ("WAIT_VIEW","VISUAL_SERVO")
            need_frame=modern or step%a.vision_interval==0 or placing
            frame=(sensor.capture(obs) if a.sensor_mode=='native-three' else sensor.capture()) if need_frame else None
            pads=env.data.geom_xpos[list(env._left_finger_geoms)].copy()  # robot FK only
            gap=np.linalg.norm(pads[1]-pads[0])-2*env.config.cookie_transfer.left_finger_pad_half_thickness_m
            trace_count=len(placement.trace) if placement else 0
            counts_before=dict(placement.counts) if placement else {}
            if placement:
                from .rgbd_placement import RobotPlacementState
                placement.update(frame,RobotPlacementState(env.data.site_xpos[servo.site].copy(),
                    env.data.site_xmat[servo.site].reshape(3,3).copy(),pads.mean(0),float(gap),
                    float(env.last_applied_action[7]),float(env.data.time)))
            if geometry:
                from .geometry_context import robot_state_from_env
                geometry.update(frame,robot_state_from_env(env))
            calibrations=camera_calibrations_from_fk(env)
            action=controller.act(obs,calibrations,frame=frame,
                pad_midpoint_world=pads.mean(0),jaw_gap_m=float(gap),
                observation_settled=bool(observer and observer.active and observer._phase=='CAPTURE'))
            source='model' if action is not None else 'hold'
            if observer and controller.phase=='OBSERVE':
                from .rgbd_localization import LocalizationFailure
                if not observer.active:
                    try:observer.begin(obs['observation.state'],controller.last_scene,frame,
                        controller.observation_reason,attempt=controller.observation_attempts-1,
                        remaining=min(200-controller.observation_steps,
                            controller.batch_steps-(controller.step-controller.batch_started)),
                        verify=controller._after_observation=='VERIFY_RELEASE')
                    except LocalizationFailure as exc:controller.observation_finished(str(exc),0)
                if observer.active:
                    action=observer.act(obs['observation.state'],frame.timestamp);source='observation_move'
                    if observer.done:
                        controller.observation_finished(observer.error,observer.used_steps)
                        action=None;source='hold'
            action_sources[source]+=1
            stage=controller.phase+('_'+observer._phase if observer and observer.active else '')
            changed=stage!=last_phase
            if modern and frame is not None and changed and (snapshot_count<19 or controller.finished and snapshot_count<20):
                from .rgbd import save_rgbd_packet
                destination=a.output/'sensor_probes';destination.mkdir(exist_ok=True)
                save_rgbd_packet(destination/f'{snapshot_count:02d}_{stage.lower()}_{step+1:04d}.npz',frame)
                if controller.phase=='EXECUTE' and controller.goal is not None:
                    from PIL import Image
                    from .visual_goals import annotate_visual_goal
                    images=annotate_visual_goal(obs,calibrations,controller.goal)
                    for name in ('front','left_wrist','right_wrist'):
                        Image.fromarray(images['observation.images.'+name]).save(destination/f'{snapshot_count:02d}_policy_{name}.png')
                snapshot_count+=1
            last_phase=stage
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
            prepared=False
            if training_recorder:
                goal=controller.goal or training_recorder.goal
                if goal is not None:
                    from .visual_goals import annotate_visual_goal
                    clip_source='vla' if action is not None else 'boundary_hold'
                    if placement:
                        clip_source=next((key for key in placement.counts if placement.counts[key]>counts_before.get(key,0)),'boundary_hold')
                    training_recorder.prepare(annotate_visual_goal(obs,calibrations,goal),geometry.build(),
                        goal,step+1,clip_source);prepared=True
            applied=servo.hold() if action is None else servo.canonical(action)
            control_log.write(json.dumps(dict(step=step+1,timestamp=float(env.data.time),source=source,
                phase=controller.phase,left_state=np.asarray(obs['observation.state'])[:8].tolist(),
                left_command=applied[:8].tolist(),completed_batches=len(controller.completed)))+'\n')
            obs,_,terminated,truncated,info=env.step(applied)
            if prepared:
                training_recorder.commit(env.last_applied_action[:8],
                    visual_verified=len(controller.completed)>training_recorder.goal.index,
                    native_count=int(info.get('cookies_in_target',0)),failed=controller.finished,
                    reason=controller.failure_reason)
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
            action_source=('vla_transport_rgbd_servo_placement' if placement else
                'single_left8_vla_with_empty_hand_observation_moves' if observer else 'single_adapted_left8_vla'),
            geometric_takeover=bool(placement and placement.counts['visual_servo']),
            object_truth_used_for_decisions=False,physics_resets_between_batches=0,
            right_target_max_error=right_error,sensor=sensor.protocol,n_action_steps=policy.config.n_action_steps,
            checkpoint=str(a.checkpoint),camera_reference=str(a.camera_reference),wall_seconds=time.monotonic()-started)
        result["placement_probe_packets"]=recorder.count if recorder else 0
        result['placement_control']=a.placement_control
        result.update(observation_mode=a.observation_mode,placement_mode=a.placement_mode,sensor_mode=a.sensor_mode,
            action_sources=action_sources,observation_steps=controller.total_observation_steps,
            observation_events=observer.events if observer else [],model_weights_updated=False,
            manipulation_action_takeover=False if modern else bool(placement),
            supervisor_protocol='native_observe_containment_v1' if modern else SUPERVISOR_VERSION,
            strict_benchmark_success=bool(info.get('success',False)))
        result['placement_truth_audit_after_decisions']=bool(audit)
        result['learned_geometry']=geometry_weights
        result['training_clips']=len(training_recorder.clips) if training_recorder else 0
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
        if control_log:control_log.close()
        if training_recorder:training_recorder.close()
        if audit:audit.close()
        if video:video.close()
        if policy:policy.close()
        if sensor:sensor.close()
        env.close()


if __name__=="__main__":raise SystemExit(main())
