import subprocess
import sys

import pytest

LEGACY_TOP_LEVEL_MODULES = [
    "actions",
    "batch_expert",
    "benchmark",
    "box_support",
    "collection",
    "cookie_transfer",
    "diagnostics",
    "env",
    "evaluation",
    "expert",
    "grasp",
    "model",
    "policy",
    "recording",
    "recovery",
    "runner",
    "same_column_batch_expert",
    "smolvla_policy",
    "teleop",
    "teleop_panel",
    "training",
]


@pytest.mark.parametrize("name", LEGACY_TOP_LEVEL_MODULES)
def test_legacy_shim_modules_are_removed(name):
    with pytest.raises(ModuleNotFoundError):
        __import__("a3_dual_arm_sim." + name)


def test_root_import_keeps_training_optional():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import a3_dual_arm_sim; assert 'torch' not in sys.modules; assert 'lerobot' not in sys.modules",
        ],
        check=True,
    )


def test_paths_after_move():
    from a3_dual_arm_sim.learning.training import default_base_model
    from a3_dual_arm_sim.paths import resource_root
    from a3_dual_arm_sim.workflows.benchmark import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH == resource_root() / "configs" / "cookie_batch.yaml"
    assert DEFAULT_CONFIG_PATH.is_file()
    assert default_base_model() == "lerobot/smolvla_base"


@pytest.mark.parametrize(
    "entry",
    [
        "benchmark_cookie_batch.py",
        "collect_cookie_benchmark.py",
        "run_cookie_batch.py",
        "train_after_collection.py",
    ],
)
def test_example_help(entry):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / entry), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
