"""The CLI must accept a π₀.₅ checkpoint on 2x10, and still refuse the two
combinations that are genuinely invalid.

Why this file exists
--------------------
`benchmark_cookie_batch_cli` rejects `--policy pi05:` on any task that is not
2x5-with-`pi05_aligned` or one of the six scenario benchmarks.  2x10 was in
neither branch, so the two benchmarks the checkpoint actually serves could not
both be run through the shipped entry point.

The three cases below pin the intended behaviour:

* 2x5 still needs `--prompt-style pi05_aligned`, because that benchmark can emit
  more than one sentence and the checkpoint was fine-tuned on exactly one of
  them.  Relaxing this would silently move 2x5 onto an out-of-distribution
  instruction.
* 2x10 needs no alignment flag: its instruction is the per-seed plan sentence
  from `Cookie2x10Plan`, and `--prompt-style` only changes it for `"short"`.
* 2x10 must still reject `--pi05-scene`.  The fingerprint describes the 2x5
  arrangement (ten slots, two rows) and would restore a layout the twenty-slot
  box does not have.

The assertions read the parser's own diagnostics, so they fail loudly if the
messages drift apart from the accepted combinations.
"""

import subprocess
import sys

import pytest

from a3_dual_arm_sim.paths import project_root

ROOT = project_root()
SCRIPT = ROOT / "examples" / "benchmark_cookie_batch.py"
CHECKPOINT = "pi05:/nonexistent/checkpoint"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def error_text(result):
    return f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(("--task", "2x10"), id="2x10"),
        pytest.param(("--task", "2x10", "--first-grasp", "3"), id="2x10-first-grasp"),
        pytest.param(("--task", "missing-columns"), id="scenario"),
    ],
)
def test_pi05_reaches_the_launcher_for_supported_tasks(extra):
    """No parser error for the tasks the checkpoint serves.

    The checkpoint path does not exist, so the run must fail later -- the point
    is only that the *argument validation* let it through.
    """
    result = run_cli("--policy", CHECKPOINT, "--episodes", "1", "--max-steps", "1", *extra)
    text = error_text(result)
    assert "pi05: requires" not in text, text
    assert "not a valid choice" not in text, text


def test_pi05_on_2x5_still_requires_the_aligned_prompt():
    result = run_cli("--policy", CHECKPOINT, "--task", "2x5", "--episodes", "1", "--max-steps", "1")
    assert result.returncode != 0
    assert "pi05_aligned" in error_text(result)


def test_pi05_on_2x10_still_rejects_the_2x5_fingerprint(tmp_path):
    result = run_cli(
        "--policy",
        CHECKPOINT,
        "--task",
        "2x10",
        "--pi05-scene",
        str(tmp_path / "fingerprint.json"),
        "--episodes",
        "1",
        "--max-steps",
        "1",
    )
    assert result.returncode != 0
    assert "--pi05-scene requires pi05: and --task 2x5" in error_text(result)
