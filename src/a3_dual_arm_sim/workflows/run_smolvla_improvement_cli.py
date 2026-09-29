"""Continue a completed diagnostic baseline through the authorized development gates."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from a3_dual_arm_sim.paths import project_root
from a3_dual_arm_sim.workflows.smolvla_deployment_sweep_cli import rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/smolvla_front_close_left_20k"))
    args = parser.parse_args()
    project = project_root()
    output = (project / args.root).resolve()
    env = dict(os.environ, MUJOCO_GL="egl")

    def run(script, *args):
        subprocess.run(
            [sys.executable, "-u", str(project / "examples" / script), *map(str, args)],
            cwd=project,
            env=env,
            check=True,
        )

    baseline = json.loads((output / "baseline/summary.json").read_text())
    if baseline["expert"]["successes"] != 3:
        raise RuntimeError("Expert reference failed: inspect diagnostics before proceeding")
    audit = json.loads((output / "data_audit.json").read_text())
    if not audit["stats_match"] or not all(audit["stats_match"].values()):
        raise RuntimeError("Normalization audit failed")
    run("summarize_smolvla_trace.py", "--root", output / "baseline")
    checkpoint = project / "outputs/smolvla_front_close_left_20k/model/checkpoints/020000"
    if not (output / "offline_error.json").exists():
        run(
            "smolvla_offline_error.py",
            "--root",
            project / "datasets/a3_front_close_left_100",
            "--checkpoint",
            checkpoint / "pretrained_model",
            "--output",
            output / "offline_error.json",
        )
    run("smolvla_deployment_sweep.py", "--root", output / "deployment")
    gate = json.loads((output / "deployment/gate.json").read_text())
    candidate = gate["candidate"]
    if gate["stage_three_required"]:
        additional = project / "datasets/a3_single_box_additional_100"
        command = [
            "--root",
            additional,
            "--repo-id",
            "local/a3-single-box-additional-100",
            "--seed-start",
            "2000",
            "--max-attempts",
            "500",
            "--episodes",
            "100",
        ]
        if additional.exists():
            command += ["--resume"]
        run("collect_cookie_benchmark.py", *command)
        retrain = output / "retraining"
        if not (retrain / "train_command.json").exists():
            run(
                "prepare_smolvla_retraining.py",
                "--original",
                project / "datasets/a3_front_close_left_100",
                "--additional",
                additional,
                "--gate",
                output / "deployment/gate.json",
                "--output",
                retrain,
            )
        train = json.loads((retrain / "train_command.json").read_text())
        final = retrain / "model/checkpoints/020000/pretrained_model/config.json"
        if not final.exists():
            if (retrain / "model").exists():
                raise RuntimeError(
                    "Interrupted training found; explicit checkpoint resume required"
                )
            with (retrain / "train.log").open("w") as stream:
                subprocess.run(
                    train, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
        selection = []
        for step in ("010000", "020000"):
            ckpt = retrain / "model/checkpoints" / step
            candidate_path = output / f"retrain_{step}_input.json"
            candidate_path.write_text(json.dumps(candidate, indent=2))
            directory = output / f"retrain_{step}"
            run(
                "diagnose_smolvla.py",
                "--stage",
                "verify",
                "--candidate",
                candidate_path,
                "--checkpoint-root",
                ckpt,
                "--root",
                directory,
            )
            rows = [
                json.loads(p.read_text()) for p in (directory / "candidate").glob("seed_*.json")
            ]
            selection.append((rank(rows, candidate), step, ckpt, rows))
        selected = min(selection, key=lambda x: (x[0], x[1]))
        checkpoint = selected[2]
        successes = sum(r["success"] for r in selected[3])
        if successes < 16:
            (output / "final_status.json").write_text(
                json.dumps(
                    {
                        "status": "development_target_not_met",
                        "successes": successes,
                        "episodes": 20,
                        "acceptance_started": False,
                        "checkpoint": str(checkpoint),
                    },
                    indent=2,
                )
            )
            return
    frozen = dict(candidate)
    frozen_path = output / "frozen_candidate.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise RuntimeError("Frozen candidate changed; do not reuse acceptance seeds")
    frozen_path.write_text(json.dumps(frozen, indent=2))
    run(
        "diagnose_smolvla.py",
        "--stage",
        "acceptance",
        "--candidate",
        frozen_path,
        "--checkpoint-root",
        checkpoint,
        "--root",
        output / "acceptance",
    )
    summary = json.loads((output / "acceptance/summary.json").read_text())
    (output / "final_status.json").write_text(
        json.dumps(
            {
                "status": "acceptance_complete",
                "target_met": summary["candidate"]["successes"] >= 16,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
