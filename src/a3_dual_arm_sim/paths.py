from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Source checkout root, or the caller's workspace for an installed wheel."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else Path.cwd()


def resource_root() -> Path:
    """Read-only configs/assets bundled in wheels; source resources in editable installs."""
    bundled = Path(__file__).resolve().parent / "_resources"
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return resource_root() / "configs" / "default.yaml"


def asset_root() -> Path:
    return resource_root() / "assets" / "a3"
