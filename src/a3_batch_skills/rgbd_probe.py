"""Capture RGB-D and optional truth-only audit labels, without moving the robot."""
import argparse
import json
from pathlib import Path
import numpy as np
from .rgbd import MujocoRGBDSensor,unproject,geometry_spec_from_env


def main():
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from PIL import Image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--task",choices=("2x5","2x10"),default="2x5")
    p.add_argument("--seed",type=int,default=26300)
    p.add_argument("--profile",choices=("basic","medium","diverse"),default="basic")
    p.add_argument("--resolution",type=int,default=768)
    p.add_argument("--depth-noise-m",type=float,default=0.)
    p.add_argument("--elevation",type=float,default=-90.)
    p.add_argument("--azimuth",type=float,default=90.)
    p.add_argument("--audit-truth",action="store_true",help="Separate evaluation labels, never localizer inputs")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    benchmark=make_cookie_benchmark(a.task,first_grasp=5 if a.task=="2x10" else None,
                                    profile=a.profile,source_column="1",max_steps=1500)
    env=benchmark.create_env(render_cameras=False)
    sensor=None
    try:
        env.reset(seed=a.seed,options=dict(randomize_cookies=True,randomize_boxes=True,
            randomize_source_bin=True,randomize_target_bin=True,source_bin_noise_m=benchmark.source_bin_noise_m,
            target_bin_noise_m=benchmark.target_bin_noise_m,target_bin_yaw_noise_rad=benchmark.target_bin_yaw_noise_rad,
            randomization_profile=a.profile,appearance_seed=a.seed,randomization_settings=benchmark.randomization_settings))
        sensor=MujocoRGBDSensor(env,resolution=a.resolution,depth_noise_m=a.depth_noise_m,seed=a.seed,
                               elevation=a.elevation,azimuth=a.azimuth)
        frame=sensor.capture()
        Image.fromarray(frame.rgb).save(a.output/"rgb.png")
        np.savez_compressed(a.output/"frame.npz",rgb=frame.rgb,depth_m=frame.depth_m,
            intrinsic=frame.intrinsic,world_from_optical=frame.world_from_optical,timestamp=frame.timestamp)
        points=unproject(frame)
        spec=geometry_spec_from_env(env)
        (a.output/"spec.json").write_text(json.dumps(spec,indent=2))
        report=dict(sensor.protocol,seed=a.seed,task=a.task,profile=a.profile,
            point_z_quantiles=np.nanquantile(points[:,:,2],[.01,.1,.5,.9,.99]).tolist(),
            calibration_source=frame.calibration_source,simulation_steps=0)
        (a.output/"capture.json").write_text(json.dumps(report,indent=2))
        if a.audit_truth:
            # This file is written after capture and is read only by audit tools.
            labels=dict(cookie_positions=env.cookie_positions.tolist(),
                source_bin_position=env.data.xpos[env._source_bin_body].tolist(),
                target_bin_position=env.data.xpos[env._target_bin_body].tolist(),
                target_bin_rotation=env.data.xmat[env._target_bin_body].reshape(3,3).tolist(),
                source_lattice=np.asarray(env.SOURCE_POSITIONS).tolist(),usage="evaluation_only_not_perception_input")
            (a.output/"truth_audit_only.json").write_text(json.dumps(labels,indent=2))
        print(json.dumps(report))
    finally:
        if sensor:sensor.close()
        env.close()


if __name__=="__main__":
    main()
