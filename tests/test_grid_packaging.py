"""Grid configuration must work outside a source checkout."""

import json

from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.workflows import evaluate_smolvla_grid_cli as grid


def test_grid_dry_run_outside_checkout(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "policy_preprocessor.json"):
        (checkpoint / name).write_text("{}")
    resources = resource_root()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(grid, "project_root", lambda: tmp_path)
    monkeypatch.setattr(grid, "resource_root", lambda: resources)
    monkeypatch.setattr("sys.argv", ["grid", "--checkpoint", str(checkpoint), "--dry-run"])
    assert grid.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["max_steps"] == 500
    assert report["randomization_2x5"]["source_column"] == "1"
    assert report["randomization_2x10"]["source_column"] == "1"
    assert not (tmp_path / "outputs").exists()


def test_wheel_does_not_report_callers_git_revision(tmp_path, monkeypatch):
    package = tmp_path / "site-packages/a3_dual_arm_sim/workflows"
    package.mkdir(parents=True)
    monkeypatch.setattr(grid, "__file__", str(package / "evaluate_smolvla_grid_cli.py"))

    def forbidden(*args, **kwargs):
        raise AssertionError("wheel must not query an unrelated checkout")

    monkeypatch.setattr(grid.subprocess, "run", forbidden)
    assert grid.source_revision() is None
