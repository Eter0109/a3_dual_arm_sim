"""Profile restrictions for all expanded cookie benchmark environments."""

import os
import subprocess
import sys

import pytest
import yaml

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10 import SOURCE_TASKS
from a3_dual_arm_sim.tasks.cookie_scenarios import SCENARIO_TASKS
from a3_dual_arm_sim.workflows.benchmark import CookieScenarioBenchmark, make_cookie_benchmark

EXPANDED_TASKS = (*SCENARIO_TASKS, *SOURCE_TASKS)


@pytest.mark.parametrize("task", EXPANDED_TASKS)
def test_expanded_tasks_default_to_basic_and_reject_other_profiles(task):
    benchmark = make_cookie_benchmark(task)
    assert benchmark.profile == "basic"
    assert benchmark.randomization_settings["profile"] == "basic"
    assert benchmark.source_bin_noise_m == pytest.approx(0.01)
    assert benchmark.cookie_noise_m == pytest.approx(0.0003)
    for profile in ("medium", "advanced"):
        with pytest.raises(ValueError, match="only supports the basic"):
            make_cookie_benchmark(task, profile=profile)


@pytest.mark.parametrize("task", ("2x5", "2x10"))
@pytest.mark.parametrize("profile", ("medium", "advanced"))
def test_original_tasks_keep_their_profiles(task, profile):
    assert make_cookie_benchmark(task, profile=profile).profile == profile


def test_direct_scenario_constructor_rejects_nonbasic_profile():
    with pytest.raises(ValueError, match="only supports the basic"):
        CookieScenarioBenchmark("partial-gaps", profile="advanced")


@pytest.mark.parametrize("task", ("sparse-exact-ten", "2x10-four-source"))
def test_reset_rejects_nonbasic_profile_and_above_basic_noise(task):
    env = make_cookie_benchmark(task).create_env(render_cameras=False)
    try:
        with pytest.raises(ValueError, match="only supports the basic"):
            env.reset(seed=1, options={"randomization_profile": "advanced"})
        with pytest.raises(ValueError, match="source_bin_noise_m.*exceeds"):
            env.reset(seed=1, options={
                "randomization_profile": "basic", "randomize_boxes": True,
                "source_bin_noise_m": 0.02,
            })
        env.reset(seed=1, options={"randomization_profile": "basic"})
        assert env.randomization_metadata["profile"] == "basic"
        assert env.randomization_metadata["configuration"]["profile"] == "basic"
    finally:
        env.close()


@pytest.mark.parametrize("task", ("partial-gaps", "2x10-paired-columns"))
def test_config_snapshot_defaults_to_basic_without_mutating_input(task):
    settings = load_randomization_config()
    settings["profile"] = "advanced"
    benchmark = make_cookie_benchmark(task, randomization_settings=settings)
    assert settings["profile"] == "advanced"
    assert benchmark.profile == "basic"
    assert benchmark.randomization_settings["profile"] == "basic"
    assert benchmark.source_bin_noise_m == pytest.approx(settings["profiles"]["basic"]["source_bin_noise_m"])


def test_scenario_config_file_default_is_selected_as_basic(tmp_path):
    settings = load_randomization_config()
    settings["profile"] = "advanced"
    settings["source_column"] = "3"
    path = tmp_path / "randomization.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    benchmark = make_cookie_benchmark("partial-gaps", randomization_config=path)
    assert benchmark.profile == "basic"
    assert benchmark.source_column == "3"
    assert benchmark.randomization_settings["profile"] == "basic"


@pytest.mark.parametrize("task", ("variable-prefill", "2x10-four-source"))
def test_cli_rejects_an_explicit_nonbasic_profile_before_running(task, tmp_path):
    output = tmp_path / "results.json"
    result = subprocess.run(
        [sys.executable, str(project_root() / "examples/benchmark_cookie_batch.py"),
         "--task", task, "--profile", "medium", "--episodes", "1", "--max-steps", "1",
         "--output", str(output)],
        capture_output=True, text=True, env=dict(os.environ, MUJOCO_GL="egl"), check=False,
    )
    assert result.returncode == 2
    assert "only --profile basic" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("task", ("two-boxes", "2x10-four-source"))
def test_parallel_workers_record_the_basic_profile(task):
    benchmark = make_cookie_benchmark(task, max_steps=1)
    result = benchmark.evaluate("expert", num_episodes=2, seed_start=4, workers=2, verbose=False)
    for episode in result.episodes:
        assert episode.randomization["profile"] == "basic"
        assert episode.randomization["configuration"]["profile"] == "basic"
        assert episode.randomization["reset_options"]["randomization_profile"] == "basic"
