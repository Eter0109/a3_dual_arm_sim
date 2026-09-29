from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from a3_dual_arm_sim.core.contracts import FRONT_IMAGE, validate_observation
from a3_dual_arm_sim.envs.base import A3DualArmEnv


class FakeRenderer:
    def __init__(self, model, *, height, width):
        self.model = model
        self.height = height
        self.width = width
        self.depth_enabled = False
        self.scene = SimpleNamespace(flags=np.array([False, True, False]))
        self.updates = []
        self.failure = None
        self.disable_calls = 0
        # Noncontiguous double precision input verifies both ownership and cast.
        self.depth_buffer = np.arange(height * width * 2, dtype=np.float64).reshape(
            height, width * 2
        )[:, ::2] + 0.5
        self.depth_buffer = self.depth_buffer[:, ::-1]

    def enable_depth_rendering(self):
        self.depth_enabled = True
        if self.failure == "enable":
            self.scene.flags[:] = True
            raise RuntimeError("fake enable failure")

    def disable_depth_rendering(self):
        self.depth_enabled = False
        self.disable_calls += 1

    def update_scene(self, data, *, camera):
        self.updates.append((data.time, camera))
        if self.failure == "update":
            self.scene.flags[:] = True
            raise RuntimeError("fake update failure")

    def render(self):
        if self.depth_enabled:
            self.scene.flags[:] = True
            if self.failure == "render":
                raise RuntimeError("fake render failure")
            if self.failure == "shape":
                return np.zeros((self.height + 1, self.width))
            return self.depth_buffer
        return np.full((self.height, self.width, 3), 127, dtype=np.uint8)


def bare_env(*, render_cameras=True):
    env = A3DualArmEnv.__new__(A3DualArmEnv)
    env.render_cameras = render_cameras
    env.config = SimpleNamespace(image_height=2, image_width=3)
    env.model = object()
    env.data = SimpleNamespace(time=1.25)
    env._camera_ids = {"front": 0, "left_wrist": 1, "right_wrist": 2}
    env._renderer = None
    env._step_count = 17
    return env


def test_optional_depth_reuses_lazy_renderer_and_returns_owned_contiguous_metres(monkeypatch):
    created = []

    def make_renderer(model, *, height, width):
        renderer = FakeRenderer(model, height=height, width=width)
        created.append(renderer)
        return renderer

    monkeypatch.setattr("a3_dual_arm_sim.envs.base.mujoco.Renderer", make_renderer)
    env = bare_env()
    depth = env.render_depth("left_wrist")
    assert len(created) == 1
    renderer = created[0]
    assert depth.shape == (2, 3) and depth.dtype == np.float32
    assert depth.flags.c_contiguous and depth.flags.owndata
    assert not np.shares_memory(depth, renderer.depth_buffer)
    np.testing.assert_allclose(depth, renderer.depth_buffer)
    np.testing.assert_array_equal(renderer.scene.flags, [False, True, False])
    assert not renderer.depth_enabled
    assert renderer.updates == [(1.25, 1)]
    assert env.data.time == 1.25 and env._step_count == 17
    rgb = env._render_camera("left_wrist")
    assert len(created) == 1
    assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, np.full((2, 3, 3), 127, dtype=np.uint8))
    renderer.depth_buffer[:] = -1
    assert np.all(depth > 0)


@pytest.mark.parametrize("failure", ["enable", "update", "render", "shape"])
def test_depth_failure_restores_rgb_mode_and_scene_flags(failure):
    env = bare_env()
    renderer = FakeRenderer(env.model, height=2, width=3)
    renderer.failure = failure
    env._renderer = renderer
    with pytest.raises(RuntimeError):
        env.render_depth("front")
    assert renderer.disable_calls == 1 and not renderer.depth_enabled
    np.testing.assert_array_equal(renderer.scene.flags, [False, True, False])
    renderer.failure = None
    rgb = env._render_camera("front")
    assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8
    assert env.data.time == 1.25 and env._step_count == 17


@pytest.mark.parametrize("camera", ["missing", "", None, 1, ["front"]])
def test_unknown_depth_camera_rejected_before_disabled_flag_or_gl_creation(camera, monkeypatch):
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError("must not create an OpenGL context")

    monkeypatch.setattr("a3_dual_arm_sim.envs.base.mujoco.Renderer", forbidden_renderer)
    env = bare_env(render_cameras=False)
    with pytest.raises(ValueError, match="unknown depth camera"):
        env.render_depth(camera)
    assert env._renderer is None


def test_disabled_camera_depth_is_explicit_error_not_fake_zeros(monkeypatch):
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError("must not create an OpenGL context")

    monkeypatch.setattr("a3_dual_arm_sim.envs.base.mujoco.Renderer", forbidden_renderer)
    env = bare_env(render_cameras=False)
    with pytest.raises(RuntimeError, match="render_cameras=True"):
        env.render_depth("front")
    assert env._renderer is None
    rgb = env._render_camera("front")
    assert rgb.dtype == np.uint8 and not np.any(rgb)


def test_real_rgb_depth_rgb_is_aligned_and_does_not_step_or_change_observation_contract():
    """Run on the server with EGL; deliberately uses the actual MuJoCo renderer."""
    env = A3DualArmEnv(render_cameras=True)
    try:
        observation, _ = env.reset(seed=0)
        validate_observation(observation)
        before_rgb = env._render_camera("front").copy()
        before_time = float(env.data.time)
        before_step = env._step_count
        before_qpos = env.data.qpos.copy()
        for camera in ("front", "left_wrist", "right_wrist"):
            depth = env.render_depth(camera)
            assert depth.shape == (env.config.image_height, env.config.image_width)
            assert depth.dtype == np.float32 and depth.flags.c_contiguous
            assert np.all(np.isfinite(depth)) and np.all(depth > 0)
            assert float(env.data.time) == before_time and env._step_count == before_step
            np.testing.assert_array_equal(env.data.qpos, before_qpos)
        after_rgb = env._render_camera("front")
        assert after_rgb.shape == before_rgb.shape and after_rgb.dtype == np.uint8
        np.testing.assert_allclose(after_rgb, before_rgb, atol=1)
        assert np.ptp(env.render_depth("front")) > 0
        assert observation[FRONT_IMAGE].dtype == np.uint8
        assert not any("depth" in key for key in observation)
        assert not any("depth" in key for key in env.observation_space.spaces)
        validate_observation(observation)
    finally:
        env.close()
