"""Exercise independent YAML behavior through the public task entrypoints."""

import json
import sys

import pytest
import yaml

from a3_dual_arm_sim.sim.randomization import choose_column, load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings


@pytest.mark.parametrize("mode", [0, 9, "random"])
def test_grasp_mode_is_independent_of_strength_and_column(tmp_path, mode):
    config = load_randomization_config()
    config["first_grasp"] = mode
    path = tmp_path / "settings.yaml"
    expected = [Cookie2x10Plan.from_seed(mode, seed).batch_sizes for seed in range(12)]
    for profile in ("basic", "medium", "advanced"):
        for column in ("1", "2", "3", "4", "random"):
            config.update(profile=profile, source_column=column)
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            settings, selected = load_cookie_2x10_settings(path)
            assert selected == mode and settings["profile"] == profile
            assert settings["source_column"] == column
            assert [
                Cookie2x10Plan.from_seed(selected, seed).batch_sizes for seed in range(12)
            ] == expected
            assert [choose_column(settings["source_column"], seed) for seed in range(12)] == [
                choose_column(column, seed) for seed in range(12)
            ]


def test_task_configs_cannot_be_mixed_by_accident(tmp_path):
    config = load_randomization_config()
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="randomization_2x10.yaml"):
        load_cookie_2x10_settings(path)
    config["first_grasp"] = 0
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="randomization config requires"):
        load_randomization_config(path)
    settings, _ = load_cookie_2x10_settings(path)
    assert "first_grasp" not in settings
    assert load_cookie_2x10_settings(path)[1] == 0


@pytest.mark.parametrize("invalid", [-1, 10, True, "RANDOM", 1.5])
def test_bad_yaml_count_fails_before_collection(tmp_path, invalid):
    config = {**load_randomization_config(), "first_grasp": invalid}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_cookie_2x10_settings(path)


@pytest.mark.parametrize("mode", [0, 9, "random"])
def test_demo_and_collection_use_same_config_prompt(tmp_path, monkeypatch, capsys, mode):
    from a3_dual_arm_sim.data import collect_cookie_2x10_cli
    from a3_dual_arm_sim.workflows import run_cookie_2x10_cli

    config = {
        **load_randomization_config(),
        "profile": "medium",
        "source_column": "random",
        "first_grasp": mode,
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect", "--dry-run", "--randomization-config", str(path), "--root", str(root)],
    )
    assert collect_cookie_2x10_cli.main() == 0
    collection = json.loads(capsys.readouterr().out)
    assert collection["collection"]["first_grasp_mode"] == mode
    assert not root.exists()
    monkeypatch.setattr(sys, "argv", ["demo", "--dry-run", "--randomization-config", str(path)])
    assert run_cookie_2x10_cli.main() == 0
    demo = json.loads(capsys.readouterr().out)
    assert demo["profile"] == "medium"
    assert demo["prompt"] == collection["preview"][0]["prompt"]
    assert demo["grasp_counts"] == collection["preview"][0]["grasp_counts"]


def test_cli_zero_override_does_not_fall_back_to_yaml(tmp_path, monkeypatch, capsys):
    from a3_dual_arm_sim.workflows import run_cookie_2x10_cli

    path = tmp_path / "settings.yaml"
    path.write_text(
        yaml.safe_dump({**load_randomization_config(), "first_grasp": 9}), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["demo", "--dry-run", "--randomization-config", str(path), "--first-grasp", "0"],
    )
    assert run_cookie_2x10_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["grasp_counts"] == [0, 10, 0, 10]


def test_default_demo_and_collection_ignore_legacy_yaml(tmp_path, monkeypatch, capsys):
    from a3_dual_arm_sim.data import collect_cookie_2x10_cli
    from a3_dual_arm_sim.sim import randomization
    from a3_dual_arm_sim.tasks import cookie_2x10_settings
    from a3_dual_arm_sim.workflows import run_cookie_2x10_cli
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark
    from a3_dual_arm_sim.workflows.cookie_2x10_execution import Cookie2x10EpisodeExecution

    old_config = load_randomization_config()
    old_config.update(profile="medium", source_column="4")
    old_path = tmp_path / "randomization.yaml"
    old_path.write_text(yaml.safe_dump(old_config), encoding="utf-8")
    new_config = {
        **load_randomization_config(),
        "profile": "advanced",
        "source_column": "2",
        "first_grasp": 9,
    }
    new_path = tmp_path / "randomization_2x10.yaml"
    new_path.write_text(yaml.safe_dump(new_config), encoding="utf-8")
    monkeypatch.setattr(randomization, "DEFAULT_RANDOMIZATION_CONFIG", old_path)
    monkeypatch.setattr(cookie_2x10_settings, "DEFAULT_COOKIE_2X10_SETTINGS", new_path)
    for entry in (run_cookie_2x10_cli, collect_cookie_2x10_cli):
        monkeypatch.setattr(sys, "argv", ["task", "--dry-run"])
        assert entry.main() == 0
        result = json.loads(capsys.readouterr().out)
        payload = result.get("collection", result)
        assert payload["first_grasp_mode"] == 9
        assert payload.get("profile", payload.get("randomization_profile")) == "advanced"
        assert str(payload["source_column"]) == "2"
    legacy = CookieBatchBenchmark(randomization_config=old_path)
    assert legacy.required_cookies == 10 and legacy.profile == "medium"
    assert legacy.source_column == "4"
    new = Cookie2x10EpisodeExecution()
    assert new.required_cookies == 20 and new.first_grasp == 9
    assert new.profile == "advanced" and new.source_column == "2"
    old_config.update(profile="basic", source_column="random")
    old_path.write_text(yaml.safe_dump(old_config), encoding="utf-8")
    again = Cookie2x10EpisodeExecution()
    assert again.randomization_settings == new.randomization_settings
    assert again.first_grasp == new.first_grasp


def test_legacy_expert_keeps_two_batches_of_five():
    from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert

    legacy = object.__new__(A3VariedColumnBatchExpert)
    assert legacy.batch_size == 5 and legacy.num_batches == 2
    new = object.__new__(A3Cookie2x10Expert)
    new.plan = Cookie2x10Plan.from_seed(3, 0)
    new.batch_index = 0
    assert new.batch_size == 3 and new.num_batches == 4
