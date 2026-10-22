"""CPU-only native-camera/empty-hand observation checks; no learned task score."""
import argparse
import json
from pathlib import Path
import numpy as np
from .cameras import apply_camera_reference
from .native_sensor import NativeThreeRGBD
from .rgbd import geometry_spec_from_env,save_rgbd_packet
from .visual_fusion import FusedRGBDLocalizer
from .active_observation import LeftWristObservation
from .visual_evidence import PlanningEvidence
from .rgbd_localization import LocalizationFailure
from .motion import LeftArmServo


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera-reference',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite evidence')
    a.output.mkdir(parents=True);rows=[]
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    for task in ('2x5','2x10'):
        for seed in (26380,26490,26491):
            case=a.output/f'{task}_{seed}';case.mkdir()
            benchmark=make_cookie_benchmark(task,first_grasp=5 if task=='2x10' else None,
                profile='basic',source_column='1',max_steps=1500 if task=='2x5' else 3000)
            apply_camera_reference(benchmark,a.camera_reference)
            env=benchmark.create_env(render_cameras=True);sensor=NativeThreeRGBD(env)
            record=dict(task=task,seed=seed,mode='observation_only_no_pi05_manipulation')
            try:
                obs,_=env.reset(seed=seed,options=dict(randomize_cookies=benchmark.randomize_cookies,
                    randomize_boxes=benchmark.randomize_boxes,randomize_source_bin=benchmark.randomize_boxes,
                    randomize_target_bin=benchmark.randomize_boxes,source_bin_noise_m=benchmark.source_bin_noise_m,
                    target_bin_noise_m=benchmark.target_bin_noise_m,target_bin_yaw_noise_rad=benchmark.target_bin_yaw_noise_rad,
                    randomization_profile='basic',appearance_seed=seed,randomization_settings=benchmark.randomization_settings))
                before=env.data.qpos.copy();mounts=env.model.cam_quat.copy()
                frame=sensor.capture(obs);save_rgbd_packet(case/'entry.npz',frame)
                assert np.array_equal(before,env.data.qpos) and np.array_equal(mounts,env.model.cam_quat)
                spec=geometry_spec_from_env(env);localizer=FusedRGBDLocalizer(spec)
                scene=localizer.observe(frame)
                record.update(sensor=sensor.protocol,source_detected=len(scene.source_cookies),target_detected=len(scene.target_cookies),
                    camera_and_physics_unchanged_by_capture=True)
                evidence=PlanningEvidence();evidence.update(frame,scene,spec,0,'auto',safe_static=True)
                try:
                    evidence.goal(scene,spec,0,frame.timestamp);record['initial_goal_available']=True
                except LocalizationFailure as exc:record.update(initial_goal_available=False,initial_error=str(exc))
                observer=LeftWristObservation(sensor,env);servo=LeftArmServo(env)
                try:
                    observer.begin(obs['observation.state'],scene,frame,record.get('initial_error','Source entry diagnostic'))
                    while observer.active:
                        command=observer.act(obs['observation.state'],float(env.data.time))
                        obs,_,_,_,_=env.step(servo.canonical(command))
                        frame=sensor.capture(obs)
                        try:
                            scene=localizer.observe(frame);evidence.update(frame,scene,spec,0,'auto',safe_static=True)
                        except LocalizationFailure:pass
                    record.update(observation_steps=observer.used_steps,observation_error=observer.error,events=observer.events,
                        returned_max_joint_error=float(np.max(abs(obs['observation.state'][:7]-observer.home))))
                    save_rgbd_packet(case/'returned.npz',frame)
                    try:
                        evidence.goal(scene,spec,0,frame.timestamp);record['returned_goal_available']=True
                    except LocalizationFailure as exc:record.update(returned_goal_available=False,returned_error=str(exc))
                except LocalizationFailure as exc:record['observation_planning_error']=str(exc)
            except LocalizationFailure as exc:record['localization_error']=str(exc)
            finally:sensor.close();env.close()
            rows.append(record);(case/'result.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(record),flush=True)
    (a.output/'result.json').write_text(json.dumps(dict(cases=rows,model_weights_loaded=False),indent=2))


if __name__=='__main__':main()
