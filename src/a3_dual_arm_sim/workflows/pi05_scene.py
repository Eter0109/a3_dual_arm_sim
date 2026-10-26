"""Explicit PI05 training-camera/scene reproduction, separate from native scoring."""
from dataclasses import fields, replace
from pathlib import Path
import hashlib
import json
import numpy as np


def configure(benchmark, path, mode):
    if mode not in ('cameras', 'full'):
        raise ValueError('PI05 scene mode must be cameras or full')
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    allowed = {f.name for f in fields(benchmark.config.cameras)}
    camera = {k: tuple(v) if isinstance(v, list) else v
              for k, v in data['camera_config'].items() if k in allowed}
    benchmark.config = replace(benchmark.config, cameras=replace(benchmark.config.cameras, **camera))
    if mode == 'full':
        positions = np.asarray(data['cookie_positions'])
        quats = np.asarray(data['cookie_quaternions'])
        if positions.shape != (80, 3) or quats.shape != (80, 4):
            raise ValueError('Expected 80 cookie positions and quaternions')
        if not np.isfinite(positions).all() or not np.isfinite(quats).all():
            raise ValueError('Non-finite fingerprint')
        benchmark.randomize_boxes = False
        benchmark.randomize_cookies = False
    benchmark.pi05_scene = data
    benchmark.pi05_scene_protocol = {'mode': mode, 'path': str(path),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'wrist_camera_mode': 'fixed_main_reference',
        'evaluation_scope': 'training_scene_reconstruction' if mode == 'full' else 'camera_override'}


def apply_after_reset(env, data, mode):
    import mujoco
    model, state = env.model, env.data
    def named(kind, name):
        i = mujoco.mj_name2id(model, kind, name)
        if i < 0:
            raise ValueError(f'Missing scene object {name}')
        return i
    if mode == 'full':
        for name, value in data['joints'].items():
            j = named(mujoco.mjtObj.mjOBJ_JOINT, name)
            state.qpos[model.jnt_qposadr[j]] = value
            state.qvel[model.jnt_dofadr[j]] = 0
        for i, (pos, quat) in enumerate(zip(data['cookie_positions'], data['cookie_quaternions'], strict=True)):
            j = named(mujoco.mjtObj.mjOBJ_JOINT, f'cookie_{i}_free')
            q, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
            state.qpos[q:q+7] = [*pos, *quat]
            state.qvel[v:v+6] = 0
        for name, pose in data['bodies'].items():
            b = named(mujoco.mjtObj.mjOBJ_BODY, name)
            if model.body_parentid[b] != 0:
                raise ValueError(f'Expected world-attached body {name}')
            j = model.body_jntadr[b]
            if model.body_jntnum[b] and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                q, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
                state.qpos[q:q+7] = [*pose['xpos'], *pose['xquat']]
                state.qvel[v:v+6] = 0
            else:
                model.body_pos[b] = pose['xpos']
                model.body_quat[b] = pose['xquat']
        env._last_applied_action = env.current_joint_action.copy()
        env._apply_controls(env._last_applied_action)
    # Keep robot_builder's fixed wrist xyaxes. The reference PNGs match this
    # orientation; forcing TARGETBODY (as the provider helper does) flips them.
    mujoco.mj_forward(model, state)
    # Return a fresh image/state pair, not reset's pre-override observation.
    return env._observation()
