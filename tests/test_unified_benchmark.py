import json
import os
import subprocess
import sys

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark, run_cookie_batch_benchmark

ROOT = project_root()


def test_task_prompts_and_thresholds():
    old = make_cookie_benchmark("2x5", prompt_style="training", source_column="1")
    new = make_cookie_benchmark(
        "2x10",
        prompt_style="training",
        first_grasp=3,
        success_hold_steps=7,
        wall_contact_tolerance_m=0.02,
    )
    assert old._episode_task(1, 0) == "transfer 10 cookies into target box"
    assert "20 cookies" in new._episode_task(1, 0)
    assert "Grasp 3 cookies first" in new._episode_task(1, 0)
    env = new.create_env()
    try:
        assert env.task_config.success_hold_steps == 7
        assert env.task_config.wall_contact_tolerance_m == 0.02
    finally:
        env.close()


def test_unified_convenience_twenty_cookie_capacity():
    result = run_cookie_batch_benchmark(
        "expert", task="2x10", first_grasp=3, num_episodes=1, max_steps=1, verbose=False
    )
    assert result.max_possible_score == 20
    assert result.episodes[0].max_score == 20
    assert "/ 20.00" in result.summary_table()


def test_parallel_retains_task_plan_and_metadata(tmp_path):
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/benchmark_cookie_batch.py"),
            "--task",
            "2x10",
            "--policy",
            "expert",
            "--first-grasp",
            "3",
            "--profile",
            "medium",
            "--source-column",
            "2",
            "--episodes",
            "2",
            "--max-steps",
            "1",
            "--workers",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        env=dict(os.environ, MUJOCO_GL="egl"),
        capture_output=True,
    )
    result = json.loads(output.read_text())
    assert result["summary"]["max_possible_score"] == 40
    assert result["run"]["success_conditions"]["required_cookies_in_source"] == 60
    for episode in result["episodes"]:
        assert episode["max_score"] == 20 and episode["source_column"] == 2
        assert episode["randomization"]["profile"] == "medium"
        assert "Grasp 3 cookies first" in episode["randomization"]["task_instruction"]


def test_diagnostic_task_dispatch(tmp_path):
    output = tmp_path / "report.json"
    traces = tmp_path / "traces"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/benchmark_cookie_batch.py"),
            "--task",
            "2x10",
            "--policy",
            "variable_grasp",
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--diagnostic-dir",
            str(traces),
            "--output",
            str(output),
        ],
        check=True,
        env=dict(os.environ, MUJOCO_GL="egl"),
        capture_output=True,
    )
    assert json.loads((traces / "seed_0/result.json").read_text())["max_score"] == 20
    assert (traces / "seed_0/front.mp4").stat().st_size > 0
