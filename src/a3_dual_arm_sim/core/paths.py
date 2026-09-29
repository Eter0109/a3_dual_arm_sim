from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Find resources without depending on package depth or working directory."""
    override = os.environ.get("A3_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "configs" / "envs").is_dir():
            raise FileNotFoundError(f"A3_PROJECT_ROOT has no configs/envs directory: {root}")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / "configs" / "envs").is_dir():
            return root
    raise FileNotFoundError("A3 resources not found; set A3_PROJECT_ROOT to the checkout")


def default_config_path() -> Path:
    return project_root() / "configs" / "envs" / "default.yaml"


def asset_root() -> Path:
    return project_root() / "assets" / "a3"


def configure_hf_cache() -> None:
    """Set shared local caches on use, never as a package-import side effect."""
    runtime = project_root() / ".runtime"
    os.environ.setdefault("HF_HOME", str(runtime / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(runtime / "datasets"))
