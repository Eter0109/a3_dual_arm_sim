"""Resume deployment experiments after the existing four-weight evaluation exits."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from a3_dual_arm_sim.paths import project_root


def rank(rows):
    successes = [r for r in rows if r["success"]]
    return (
        -len(successes),
        -sum(r["cookies_in_target"] for r in rows) / len(rows),
        sum(r.get("failure_reason") == "environment_safety_terminated" for r in rows),
        sum(r["steps"] for r in successes) / len(successes) if successes else math.inf,
    )


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [center - radius, center + radius]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    project = project_root()
    previous, root = (project / args.previous).resolve(), (project / args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        path = root / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)

    def run(command, log):
        with (root / log).open("a") as stream:
            subprocess.run(
                [sys.executable, "-u", *command],
                cwd=project,
                env=dict(os.environ, MUJOCO_GL="egl", HF_HUB_OFFLINE="1"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def rows(folder, count):
        result = [json.loads(p.read_text()) for p in sorted(folder.glob("seed_*.json"))]
        if len(result) != count or len({r["seed"] for r in result}) != count:
            raise RuntimeError(f"Incomplete or duplicate episodes: {folder}")
        return result

    try:
        save("status.json", {"stage": "waiting_for_previous"})
        if args.wait_pid:
            while True:
                try:
                    os.kill(args.wait_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(30)
        candidates = []
        for step in ("020000", "040000"):
            for weight in ("pretrained_model", "pretrained_model_ema"):
                name = f"{step}_{weight}"
                result = rows(previous / name / "candidate", 20)
                candidates.append((rank(result), name, step, weight))
        _, name, step, weight = min(candidates)
        checkpoint = previous / "model/checkpoints" / step
        dataset = previous / "dataset"
        settings = {
            "weights": weight,
            "ema_alpha": 0,
            "anchor_right_arm": False,
            "gripper_sharpening": False,
            "align_vertical": False,
            "clamp_z": False,
            "n_action_steps": 8,
            "num_steps": 25,
        }
        save("selected_weights.json", settings)
        save("selection.json", {"selected": name, "ranking": candidates})
        save("status.json", {"stage": "diagnosis"})
        for label, policy in [
            ("model", f"smolvla:{previous}/model/checkpoints/020000/pretrained_model"),
            ("expert", "same_column"),
        ]:
            trace = root / f"diagnosis_{label}"
            if (trace / "seed_1001/result.json").exists():
                continue
            if trace.exists():
                raise RuntimeError(
                    f"Incomplete diagnostic preserved at {trace}; select a fresh root"
                )
            command = [
                "examples/benchmark_cookie_batch.py",
                "--policy",
                policy,
                "--episodes",
                "1",
                "--seed-start",
                "1001",
                "--workers",
                "1",
                "--max-steps",
                "1000",
                "--diagnostic-dir",
                str(trace),
            ]
            if label == "model":
                command += [
                    "--inference-seed",
                    "1001",
                    "--n-action-steps",
                    "8",
                    "--dataset-root",
                    str(dataset),
                ]
            run(command, label + ".log")

        def evaluate(stage, settings_file, destination):
            run(
                [
                    "examples/diagnose_smolvla.py",
                    "--stage",
                    stage,
                    "--checkpoint-root",
                    str(checkpoint),
                    "--dataset-root",
                    str(dataset),
                    "--candidate",
                    str(settings_file),
                    "--root",
                    str(destination),
                ],
                stage + ".log",
            )

        save("status.json", {"stage": "horizon_comparison"})
        evaluate("horizon", root / "selected_weights.json", root / "horizon")
        horizon = min(
            (1, 4, 8), key=lambda n: (rank(rows(root / "horizon" / f"horizon_{n}", 5)), n != 8, n)
        )
        settings["n_action_steps"] = horizon
        save("candidate.json", settings)
        save("status.json", {"stage": "development", "n_action_steps": horizon})
        evaluate("verify", root / "candidate.json", root / "verify")
        development = rows(root / "verify/candidate", 20)
        successes = sum(r["success"] for r in development)
        if successes < 12:
            save(
                "status.json",
                {
                    "stage": "needs_reviewed_recovery_data",
                    "development_successes": successes,
                    "episodes": 20,
                    "acceptance_started": False,
                    "reason": "Review diagnostic videos; collect and approve 10 recovery demonstrations before expansion. No automatic retraining.",
                },
            )
            return
        # Never consume the reserved set after another pipeline has already used it.
        used = list(previous.glob("acceptance/**/seed_*.json"))
        if used:
            save(
                "status.json",
                {"stage": "needs_fresh_acceptance_seeds", "existing_results": len(used)},
            )
            return
        settings["weights"] = str(checkpoint / weight)
        save("frozen_candidate.json", settings)
        save("status.json", {"stage": "acceptance"})
        evaluate("acceptance", root / "frozen_candidate.json", root / "acceptance")
        final = rows(root / "acceptance/candidate", 20)
        successes = sum(r["success"] for r in final)
        save(
            "status.json",
            {
                "stage": "complete",
                "successes": successes,
                "episodes": 20,
                "target_met": successes >= 10,
                "wilson_95": wilson(successes, 20),
            },
        )
    except BaseException as error:
        save("status.json", {"stage": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
