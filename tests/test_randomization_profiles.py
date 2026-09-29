import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.sim.randomization import COOKIE_PALETTE, choose_column
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark


def test_column_selection_is_balanced_and_independent():
    for start in (0, 100, 7124):
        assert sorted(choose_column("random", seed) for seed in range(start, start + 4)) == [
            1,
            2,
            3,
            4,
        ]
    for column in range(1, 5):
        assert choose_column(str(column), 19) == column
    with pytest.raises(ValueError):
        choose_column("5", 0)


def test_profile_reset_determinism_restore_and_palette():
    benchmark = CookieBatchBenchmark(profile="advanced", source_column="random")
    env = benchmark.create_env(render_cameras=False)
    try:
        base = {key: getattr(env.model, key).copy() for key in env._appearance_randomizer.FIELDS}
        colors = set()
        for seed in range(16):
            env.reset(seed=seed, options={"randomization_profile": "advanced"})
            colors.add(env.randomization_metadata["cookie_color"])
        assert colors == set(COOKIE_PALETTE)
        env.reset(seed=12, options={"randomization_profile": "advanced"})
        first = env.randomization_metadata
        qpos = env.data.qpos.copy()
        front = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        assert env.model.cam_mode[front] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        rotated = env.data.cam_xmat[front].copy()
        # Removing just the sampled rotation must change the rendered pose.
        env.model.cam_mode[front] = base["cam_mode"][front]
        mujoco.mj_forward(env.model, env.data)
        assert not np.allclose(rotated, env.data.cam_xmat[front])
        env.reset(seed=12, options={"randomization_profile": "advanced"})
        assert env.randomization_metadata == first
        np.testing.assert_array_equal(qpos, env.data.qpos)
        env.reset(seed=12, options={"randomization_profile": "basic"})
        for key, value in base.items():
            np.testing.assert_allclose(getattr(env.model, key), value, atol=1e-12)
        # Appearance doesn't consume physical reset RNG or alter physics.
        np.testing.assert_array_equal(qpos, env.data.qpos)
    finally:
        env.close()


def test_layout_profile_and_column_are_orthogonal():
    for name in ("basic", "medium", "advanced"):
        for column in ("1", "2", "3", "4", "random"):
            b = CookieBatchBenchmark(profile=name, source_column=column)
            assert b.profile == name
            assert b.source_column == column
    assert CookieBatchBenchmark(profile="basic").source_bin_noise_m == 0.010
    assert CookieBatchBenchmark(profile="medium").source_bin_noise_m == 0.015
    assert CookieBatchBenchmark(profile="advanced").source_bin_noise_m == 0.020
