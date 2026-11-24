"""Capture each course scene after reset (seed 0) from two presentation views.

Run from the repository root with MUJOCO_GL=egl. The free cameras here are for
illustration; they do not change the cameras, actions, or physics of a policy.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
from a3_dual_arm_sim.workflows.course_protocol import COURSE_TASKS

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / 'images'
SEED = 0
WIDTH, HEIGHT = 960, 600
TARGET_DETAIL = {'prefilled', 'two-boxes', 'rotated-target', 'variable-prefill'}
TILT_DETAIL = {'tilted', '2x10-tilted', 'whole-row-lean', '2x10-whole-row-lean'}


def reset_options(benchmark, profile):
    return {
        'randomize_cookies': True, 'randomize_boxes': True,
        'randomize_source_bin': True, 'randomize_target_bin': True,
        'source_bin_noise_m': benchmark.source_bin_noise_m,
        'target_bin_noise_m': benchmark.target_bin_noise_m,
        'target_bin_yaw_noise_rad': benchmark.target_bin_yaw_noise_rad,
        'randomization_profile': profile, 'appearance_seed': SEED,
        'randomization_settings': benchmark.randomization_settings,
    }


def mark_cookies(image, segmentation, env, indices):
    """Mark visible selected cookies using MuJoCo's geometry segmentation."""
    draw = ImageDraw.Draw(image)
    marked = []
    for index in indices:
        geoms = env._cookie_all_geoms[index]
        mask = np.isin(segmentation[:, :, 0], geoms)
        mask &= segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        y, x = np.where(mask)
        if not len(x):
            continue
        # A visible object pixel rather than a geometric centroid may be safer
        # when a tilted cookie is partially occluded by its neighbours.
        middle = len(x) // 2
        px, py = int(x[middle]), int(y[middle])
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill='#151b26')
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill='#ffffff')
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill='#1762ba')
        marked.append(int(index))
    return marked


def capture(task):
    profile = 'advanced' if task in ('2x5', '2x10') else 'basic'
    bench = make_cookie_benchmark(task, profile=profile, protocol='course-v1')
    # Larger frame buffer for the report only; dynamics are unchanged.
    bench.config = replace(bench.config, image_width=WIDTH, image_height=HEIGHT)
    env = bench.create_env(render_cameras=False)
    renderer = None
    try:
        _, info = env.reset(seed=SEED, options=reset_options(bench, profile))
        renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
        overview = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(overview)
        env._configure_viewer_camera(overview)
        renderer.update_scene(env.data, camera=overview)
        Image.fromarray(renderer.render()).save(IMAGES / f'{task}_overview.png')

        detail = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(detail)
        detail.type = mujoco.mjtCamera.mjCAMERA_FREE
        scenario = getattr(env, 'scenario_metadata', {})
        markers = []
        if task in TILT_DETAIL:
            if 'tilted_cookie_ids' in scenario:
                indices = scenario['tilted_cookie_ids']
                index = indices[0]
                point = env.data.xpos[env._cookie_bodies[index]].copy()
                distance = 0.16
                markers = indices
            else:
                indices = sorted(env._eligible)
                index = indices[len(indices) // 2]
                point = env.data.xpos[env._cookie_bodies[index]].copy()
                distance = 0.25
                markers = [index]
            detail.lookat[:] = point
            detail.azimuth, detail.elevation = 140, -24
        else:
            body = env._target_bin_body if task in TARGET_DETAIL else env._source_bin_body
            point = env.data.xpos[body].copy()
            if task == 'two-boxes':
                point = (point + env.data.xpos[env._spare_body]) / 2
            detail.lookat[:] = [point[0], point[1], 0.79]
            distance = 0.30 if task in TARGET_DETAIL else 0.39
            detail.azimuth, detail.elevation = 135, -62
        detail.distance = distance
        renderer.update_scene(env.data, camera=detail)
        frame = Image.fromarray(renderer.render())
        marked = []
        if markers:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(env.data, camera=detail)
            segmented = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            marked = mark_cookies(frame, segmented, env, markers)
            if not marked:
                raise RuntimeError(f'{task}: could not mark a tilted cookie in the detail image')
        frame.save(IMAGES / f'{task}_detail.png')
        if scenario:
            target_counts = info['scenario_metrics']['target_counts']
        else:
            target_counts = [int(info.get('cookies_in_target', 0))]
        fields = ('removed_columns', 'gap_rows_zero_based', 'remaining_row_starts_zero_based',
                  'remaining_per_column', 'initial_column_counts', 'tilted_cookie_ids',
                  'lean_column', 'lean_sign', 'prefilled_slot_ids',
                  'requested_yaw_deg', 'scenario_attempt')
        scene = {key: scenario[key] for key in fields if key in scenario}
        # Keep only the two relevant tilt angles, not an 80-element dump.
        if 'tilted_cookie_ids' in scene:
            angles = scenario['initial_tilt_degrees']
            scene['initial_tilt_degrees'] = {str(i): angles[i] for i in scene['tilted_cookie_ids']}
        return {'task': task, 'seed': SEED, 'profile': profile,
                'source_count': int(info['cookies_in_source']), 'target_counts': target_counts,
                'source_position_m': env.data.xpos[env._source_bin_body].tolist(),
                'target_position_m': env.data.xpos[env._target_bin_body].tolist(),
                'overview': f'images/{task}_overview.png', 'detail': f'images/{task}_detail.png',
                'marked_cookie_ids': marked, 'scenario': scene}
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    scenes = []
    for task in COURSE_TASKS:
        scenes.append(capture(task))
        print(f'[{len(scenes)}/{len(COURSE_TASKS)}] {task}', flush=True)
    package = Path(__file__).resolve().parents[1]
    sources = ('workflows/course_protocol.py', 'workflows/benchmark.py', 'config.py',
               'tasks/cookie_scenarios.py', 'tasks/cookie_transfer.py', 'tasks/cookie_2x10.py',
               'tasks/cookie_2x10_settings.py', 'tasks/expanded_ik.py', 'tasks/source_clearance.py',
               'sim/env.py', 'sim/model.py', 'sim/object_builder.py', 'sim/robot_builder.py',
               'sim/randomization.py')
    hashes = {p: hashlib.sha256((package / p).read_bytes()).hexdigest() for p in sources}
    for name in ('cookie_batch.yaml', 'cookie_2x10.yaml', 'randomization.yaml', 'randomization_2x10.yaml'):
        hashes['configs/' + name] = hashlib.sha256((resource_root() / 'configs' / name).read_bytes()).hexdigest()
    manifest = {'purpose': 'Post-reset physical scene illustrations; not a policy observation or success result',
                'seed': SEED, 'image_size': [WIDTH, HEIGHT], 'protocol': 'course-v1',
                'render_backend': 'egl', 'mujoco_version': mujoco.__version__,
                'python_version': platform.python_version(), 'sha256': hashes, 'scenes': scenes}
    (ROOT / 'screenshots_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
