import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.sim.randomization import choose_column, load_randomization_config
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
        assert colors == set(load_randomization_config()["colors"])
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


def test_custom_config_snapshot_and_environment(tmp_path):
    import yaml

    config = load_randomization_config()
    config["profile"] = "advanced"
    config["source_column"] = "4"
    config["profiles"]["advanced"]["source_bin_noise_m"] = 0.007
    config["colors"] = {"test_blue": [0.1, 0.2, 0.8, 1.0]}
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    benchmark = CookieBatchBenchmark(randomization_config=path)
    assert benchmark.profile == "advanced"
    assert benchmark.source_column == "4"
    assert benchmark.source_bin_noise_m == 0.007
    # Editing the file after initialization must not alter a running experiment.
    config["colors"] = {"red": [1.0, 0.0, 0.0, 1.0]}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = benchmark.create_env(render_cameras=False)
    try:
        env.reset(
            seed=12,
            options={
                "randomization_profile": benchmark.profile,
                "randomization_settings": benchmark.randomization_settings,
            },
        )
        assert env.randomization_metadata["cookie_color"] == "test_blue"
        assert env.randomization_metadata["configuration"] == benchmark.randomization_settings
        np.testing.assert_allclose(
            env.model.geom_rgba[env._appearance_randomizer.cookie_geoms],
            np.tile([0.1, 0.2, 0.8, 1], (len(env._appearance_randomizer.cookie_geoms), 1)),
        )
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["negative", "nan", "unknown", "color", "profile", "column"])
def test_invalid_config_is_rejected(mutation):
    config = load_randomization_config()
    if mutation == "negative":
        config["profiles"]["advanced"]["source_bin_noise_m"] = -1
    elif mutation == "nan":
        config["profiles"]["advanced"]["light_fraction"] = float("nan")
    elif mutation == "unknown":
        config["profiles"]["advanced"]["typo"] = 1
    elif mutation == "color":
        config["colors"]["golden"] = [2, 0, 0, 1]
    elif mutation == "profile":
        config["profile"] = "missing"
    else:
        config["source_column"] = 5
    with pytest.raises(ValueError):
        load_randomization_config(config=config)
