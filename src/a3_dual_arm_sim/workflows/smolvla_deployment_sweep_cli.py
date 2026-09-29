"""Run the development-only matrix sequentially and write the stage-three gate."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from a3_dual_arm_sim.paths import project_root


def rank(rows, settings):
    successes = [r for r in rows if r["success"]]
    rules = (
        int(settings["anchor_right_arm"])
        + int(settings["gripper_sharpening"])
        + int(settings["ema_alpha"] > 0)
    )
    return (
        -len(successes),
        -sum(r["cookies_in_target"] for r in rows) / len(rows),
        sum(r["failure_reason"] == "environment_safety_terminated" for r in rows),
        sum(r["steps"] for r in successes) / len(successes) if successes else float("inf"),
        rules,
    )


def best(directory):
    variants = json.loads((directory / "manifest.json").read_text())["variants"]
    ranked = []
    for name, settings in variants.items():
        rows = [json.loads(p.read_text()) for p in sorted((directory / name).glob("seed_*.json"))]
        ranked.append((rank(rows, settings), name, settings))
    return min(ranked, key=lambda x: (x[0], x[1]))[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = project_root()
    output = args.root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL="egl")

    def run(stage, candidate=None):
        command = [
            sys.executable,
            "-u",
            str(root / "examples/diagnose_smolvla.py"),
            "--root",
            str(output / stage),
            "--stage",
            stage,
        ]
        if candidate is not None:
            path = output / f"{stage}_input.json"
            text = json.dumps(candidate, indent=2)
            if path.exists() and path.read_text() != text:
                raise RuntimeError("Candidate changed on resume")
            path.write_text(text)
            command += ["--candidate", str(path)]
        subprocess.run(command, cwd=root, env=env, check=True)

    run("matrix")
    candidate = best(output / "matrix")
    run("horizon", candidate)
    candidate = best(output / "horizon")
    if (
        not candidate["anchor_right_arm"]
        and not candidate["gripper_sharpening"]
        and candidate["ema_alpha"] == 0
    ):
        run("rules", candidate)
        candidate = best(output / "rules")
    run("verify", candidate)
    summary = json.loads((output / "verify/summary.json").read_text())["candidate"]
    (output / "selected_candidate.json").write_text(json.dumps(candidate, indent=2))
    (output / "gate.json").write_text(
        json.dumps(
            {
                "development": summary,
                "stage_three_required": summary["successes"] < 16,
                "acceptance_started": False,
                "candidate": candidate,
            },
            indent=2,
        )
    )
    print("DEVELOPMENT GATE", summary, flush=True)


if __name__ == "__main__":
    main()
