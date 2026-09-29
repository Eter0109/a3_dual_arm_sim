import pytest

from a3_dual_arm_sim.core.paths import asset_root, default_config_path, project_root


def test_resource_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("A3_PROJECT_ROOT", raising=False)
    expected = project_root()
    monkeypatch.chdir(tmp_path)
    assert project_root() == expected
    assert default_config_path() == expected / "configs" / "envs" / "default.yaml"
    assert default_config_path().is_file()
    assert asset_root() == expected / "assets" / "a3"


def test_resource_root_can_be_explicit(tmp_path, monkeypatch):
    (tmp_path / "configs" / "envs").mkdir(parents=True)
    monkeypatch.setenv("A3_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_invalid_resource_root_is_not_silently_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("A3_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="configs/envs"):
        project_root()
