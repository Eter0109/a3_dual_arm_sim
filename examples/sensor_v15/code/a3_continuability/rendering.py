"""Explicit camera protocol for paired simulation research.

Single-sample rendering avoids observed EGL/MSAA +/-1 RGB jitter on the
experiment host. It changes edge antialiasing, so it is an opt-in protocol,
must be shared by all comparisons, and does not silently rewrite benchmarks.
No image cache, reference-frame substitution, or pixel rounding is used.
"""
MODES = ('native', 'single_sample')


def configure_rendering(env, mode='native'):
    if mode not in MODES:
        raise ValueError(f'Unknown rendering protocol: {mode}')
    quality = env.model.vis.quality
    if not hasattr(env, '_a3_native_offsamples'):
        env._a3_native_offsamples = int(quality.offsamples)
    wanted = 0 if mode == 'single_sample' else env._a3_native_offsamples
    # MuJoCo allocates multisample buffers at Renderer construction time.
    # Updating just model.vis.quality after that does not recreate them.
    if int(quality.offsamples) != wanted and env._renderer is not None:
        env._renderer.close()
        env._renderer = None
    quality.offsamples = wanted
    env._a3_render_protocol = mode
    return rendering_metadata(env)


def rendering_metadata(env):
    return dict(protocol=getattr(env, '_a3_render_protocol', 'native'),
                offsamples=int(env.model.vis.quality.offsamples),
                native_offsamples=int(getattr(env, '_a3_native_offsamples', env.model.vis.quality.offsamples)),
                image_width=int(env.config.image_width), image_height=int(env.config.image_height),
                cameras='unchanged', image_cache=False)


def restore_rendering(env, metadata):
    if metadata is None:
        # Legacy snapshots did not record scalar rendering quality. The caller
        # must register its selected protocol and regenerate the observation.
        return
    if (metadata['image_width'], metadata['image_height']) != (env.config.image_width, env.config.image_height):
        raise ValueError('Snapshot camera resolution differs from the environment')
    env._a3_native_offsamples = int(metadata['native_offsamples'])
    actual = configure_rendering(env, metadata['protocol'])
    if actual['offsamples'] != metadata['offsamples']:
        raise ValueError('Snapshot rendering configuration is inconsistent')
