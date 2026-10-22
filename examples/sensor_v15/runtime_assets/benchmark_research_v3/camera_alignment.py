"""Explicit camera profiles on the unmodified main benchmark."""
from dataclasses import asdict,fields,replace
import json
from pathlib import Path


def apply_camera_profile(benchmark, profile, reference_path):
    if profile=='main':return
    if profile!='dataset':raise ValueError(profile)
    import mujoco
    from a3_dual_arm_sim.config import CameraConfig
    from a3_dual_arm_sim.sim.randomization import AppearanceRandomizer
    ref=json.loads(Path(reference_path).read_text())
    camera=ref['cameras']
    if camera.get('reference_mirror') or camera.get('front_quat_wxyz'):
        raise ValueError('This profile expects unmirrored look-at cameras')
    before=asdict(benchmark.config)
    kwargs={f.name:camera[f.name] for f in fields(CameraConfig)}
    benchmark.config=replace(benchmark.config,cameras=CameraConfig(**kwargs),
        image_height=ref['image_height'],image_width=ref['image_width'])
    after=asdict(benchmark.config)
    assert all(before[k]==after[k] for k in before if k not in ('cameras','image_height','image_width'))
    original=benchmark.create_env
    def create_env(*,render_cameras=False):
        env=original(render_cameras=render_cameras)
        # Restore the data-generating camera semantics before the randomizer
        # snapshots its nominal state. Poses alone do not restore targetbody mode.
        for side,prefix in [('left','L'),('right','R')]:
            cid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_CAMERA,side+'_wrist')
            bid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,prefix+'_wrist_camera_target')
            assert cid>=0 and bid>=0
            env.model.cam_mode[cid]=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
            env.model.cam_targetbodyid[cid]=bid
        mujoco.mj_forward(env.model,env.data)
        env._appearance_randomizer=AppearanceRandomizer(env.model)
        return env
    benchmark.create_env=create_env


def validate_alignment(benchmark,reference_path):
    import mujoco,numpy as np
    ref=json.loads(Path(reference_path).read_text())
    env=benchmark.create_env(render_cameras=False)
    errors=[]
    try:
        env.reset(seed=0,options={'randomize_cookies':False,'randomize_boxes':False})
        addresses=[int(env.model.jnt_qposadr[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in ref['joint_names']]
        for pose in ref['reference_poses']:
            env.data.qpos[addresses]=pose['joint_angles'];mujoco.mj_forward(env.model,env.data)
            for name,expected in pose['cameras'].items():
                cid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_CAMERA,name)
                errors.append({'camera':name,'position_error_m':float(np.max(np.abs(env.data.cam_xpos[cid]-expected['position']))),
                    'rotation_angle_deg':float(np.rad2deg(np.arccos(np.clip((np.trace(env.data.cam_xmat[cid].reshape(3,3) @ np.asarray(expected['rotation']).reshape(3,3).T)-1)/2,-1,1)))),
                    'rotation_matrix_error':float(np.max(np.abs(env.data.cam_xmat[cid]-expected['rotation']))),
                    'fovy_error_deg':abs(float(env.model.cam_fovy[cid])-expected['fovy'])})
    finally:env.close()
    return {'poses':len(ref['reference_poses']),'max_position_error_m':max(x['position_error_m'] for x in errors),
        'max_rotation_matrix_error':max(x['rotation_matrix_error'] for x in errors),
        'max_fovy_error_deg':max(x['fovy_error_deg'] for x in errors),'details':errors}
