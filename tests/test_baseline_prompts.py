"""Task-specific baseline prompts must survive CLI, worker and protocol dispatch."""

import json
import os
import subprocess
import sys

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows import evaluate_smolvla_grid_cli as grid
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


def test_short_twenty_prompt_preserves_native_default_and_plan_metadata():
    native = make_cookie_benchmark("2x10", prompt_style="training", first_grasp=3)
    short = make_cookie_benchmark("2x10", prompt_style="short", first_grasp=3)
    assert "Grasp 3 cookies first" in native._episode_task(1, 0)
    assert short._episode_task(1, 0) == "transfer 20 cookies into target box"
    metadata = short._episode_metadata(1, 0)
    assert metadata["task_instruction"] == short._episode_task(1, 0)
    assert metadata["prompt_style"] == "short"
    assert metadata["grasp_counts"] == [3, 7, 3, 7]
    old = make_cookie_benchmark("2x5", prompt_style="training")
    assert old._episode_metadata(1, 0)["task_instruction"] == "transfer 10 cookies into target box"


def test_short_prompt_survives_parallel_cli(tmp_path):
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(project_root() / "examples/benchmark_cookie_batch.py"),
            "--task",
            "2x10",
            "--policy",
            "expert",
            "--prompt-style",
            "short",
            "--first-grasp",
            "3",
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
        capture_output=True,
        env=dict(os.environ, MUJOCO_GL="egl"),
    )
    report = json.loads(output.read_text())
    assert report["run"]["prompt_style"] == "short"
    assert report["summary"]["max_possible_score"] == 40
    for episode in report["episodes"]:
        assert episode["randomization"]["task_instruction"] == "transfer 20 cookies into target box"
        assert episode["randomization"]["prompt_style"] == "short"
        assert episode["randomization"]["grasp_counts"] == [3, 7, 3, 7]


def test_grid_records_requested_baseline_protocol(tmp_path, monkeypatch, capsys):
    for name in ("config.json", "policy_preprocessor.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grid",
            "--checkpoint",
            str(tmp_path),
            "--dry-run",
            "--prompt-style-2x5",
            "training",
            "--prompt-style-2x10",
            "short",
            "--profiles",
            "basic",
        ],
    )
    assert grid.main() == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["prompt_styles"] == {"2x5": "training", "2x10": "short"}
    assert protocol["prompt_2x5"] == "transfer 10 cookies into target box"
    assert protocol["prompt_2x10"] == "transfer 20 cookies into target box"
    assert protocol["profiles"] == ["basic"]
    assert protocol["max_steps"] == 500


def test_grid_summary_only_includes_selected_profiles():
    rows = [{"benchmark": "2x10", "profile": "basic", "cookies_in_target": 5}]
    results = grid.summarize(rows, 20, ["basic"])
    assert len(results) == 2
    assert all(r["profile"] == "basic" for r in results)
    assert results[1]["mean_final_count"] == 5
    assert results[1]["total_capacity"] == 20
