"""Explicit numerical protocol and repeated-input checks for paired ablations."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import numpy as np


def configure_determinism():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return {'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8',
            'tf32': False, 'reduced_precision_reduction': False, 'sdpa': 'math',
            'warning': 'Scores use this numerical protocol; do not pool with earlier pilot runs.'}


def observation_signature(obs):
    digest = hashlib.sha256()
    for name in sorted(k for k in obs if k.startswith('observation.images.') or k=='observation.state'):
        digest.update(name.encode())
        digest.update(np.asarray(obs[name]).tobytes())
    return digest.hexdigest()


def physical_signature(env):
    digest = hashlib.sha256()
    for value in (env.data.qpos, env.data.qvel, env.data.act, env.data.ctrl,
                  np.asarray([env.data.time]), env.model.body_pos,
                  env.model.cam_pos, env.model.cam_quat, env.model.cam_fovy):
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def match_prefix_observation(observation, reference):
    """Use only same-time images from an identical physical state before intervention.

    This controls the observed +/-1 RGB rasterization jitter. It never uses
    future images or replaces observations after action intervention.
    """
    matched = dict(observation)
    max_error = 0
    changed = 0
    for key, image in reference['_images'].items():
        delta = np.abs(observation[key].astype(np.int16)-image.astype(np.int16))
        max_error = max(max_error, int(delta.max()))
        changed += int(np.count_nonzero(delta))
        if delta.max() > 1 or np.count_nonzero(delta) > .001*delta.size:
            raise RuntimeError(f'Image difference exceeds observed rasterization jitter: {key}')
        matched[key] = image
    return matched, {'max_rgb_error': max_error, 'different_channel_values': changed}


def repeatability_check(inner, observation, task, context):
    actions = []
    for _ in range(3):
        inner.reset(context)
        actions.append(inner.act(observation, task).copy())
    error = max(float(np.max(np.abs(a-actions[0]))) for a in actions)
    inner.reset(context)
    report = {'same_input_repeats': 3, 'max_action_error': error,
              'observation_sha256': observation_signature(observation), 'passed': error==0.}
    if not report['passed']:
        raise RuntimeError(f'Repeated input is not deterministic: {report}')
    return report
