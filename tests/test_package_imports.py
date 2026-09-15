from __future__ import annotations

import subprocess
import sys

import pytest

# Importing the simulation stack costs MuJoCo, whose wheels require CPU
# instructions that a GPU-only worker may not have. Training and recording must
# stay usable there, so these checks run in a fresh interpreter where the real
# import graph is observable.


def _run(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_import_does_not_load_mujoco() -> None:
    result = _run(
        "import sys, a3_dual_arm_sim;"
        "print('mujoco' in sys.modules, bool(a3_dual_arm_sim.__version__))"
    )
    assert result.returncode == 0, result.stderr
    # Only the two facts under test: MuJoCo is absent from the import graph, and
    # the package still exposes a version. Asserting the version string itself
    # would fail on every release for no reason.
    assert result.stdout.strip() == "False True"


def test_training_modules_import_without_mujoco() -> None:
    result = _run(
        "import sys;"
        "import a3_dual_arm_sim.training;"
        "import a3_dual_arm_sim.recording;"
        "from a3_dual_arm_sim import policy, contracts, config, paths;"
        "print('mujoco' in sys.modules)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_public_names_resolve_lazily_and_unknown_names_raise() -> None:
    result = _run(
        "import a3_dual_arm_sim as a3;"
        "print(a3.__all__ == sorted(a3.__all__));"
        "print(sorted(a3.__all__) == sorted(a3._EXPORTS));"
        "print(all(name in dir(a3) for name in a3.__all__));"
        "\ntry:\n    a3.NotAnExport\nexcept AttributeError:\n    print('raised')"
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines == ["True", "True", "True", "raised"], lines


@pytest.mark.parametrize("name", ["A3DualArmEnv", "A3CookieTransferEnv"])
def test_simulation_names_still_pull_in_mujoco(name: str) -> None:
    result = _run(
        "import sys, a3_dual_arm_sim as a3;"
        f"value = a3.{name};"
        "print(value.__name__, 'mujoco' in sys.modules)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{name} True"
