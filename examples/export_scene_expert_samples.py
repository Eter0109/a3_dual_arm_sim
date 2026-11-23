"""Render accepted expert trajectories and verify a left8 LeRobot v3 dataset.

Replays exactly the applied joint commands in the same seeded physical scene.
The numerical trajectory must agree frame-by-frame with the accepted run; no
old observation is paired with a new or divergent physical rollout. Failed
attempts are never exported as demonstrations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from audit_scene_experts import prepare, save_json
from a3_dual_arm_sim.contracts import EpisodeContext, FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE, STATE
from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.data.recording import LeRobotV3Recorder


def export_group(rows, run_dir, root, *, use_videos):
    case = rows[0]["case"]
    dest = root / f"{case['task']}_{case['scenario']}"
    bench, env = prepare(case, cameras=True)
    env.close()
    task = rows[0]["randomization"]["task_instruction"]
    repo_id = "local/" + dest.name.replace("_", "-")
    recorder = LeRobotV3Recorder(
        dest, repo_id=repo_id, fps=bench.config.control_hz,
        image_height=bench.config.image_height, image_width=bench.config.image_width,
        use_videos=use_videos, left_arm_only=True,
    )
    summary = {
        "schema_version": 2, "task": "cookie_transfer", "task_instruction": task,
        "target_layout": case["task"], "required_cookies": rows[0]["max_score"],
        "training_arm_mode": "left", "stored_action_mode": "joint_position",
        "complete_demonstrations": True, "use_videos": use_videos,
        "collection_method": "verified physical replay of an accepted expert trajectory",
        "source_run": str(run_dir), "repo_id": repo_id,
        "first_grasp_mode": 5 if case["task"] == "2x10" else None,
        "case_ids": [r["case"]["id"] for r in rows],
    }
    if case["scene"]:
        summary["scene_expert_version"] = 1
    save_json(dest / "collection_summary.json", summary)
    evidence = []
    try:
        for row in rows:
            case = row["case"]
            t0 = time.monotonic()
            bench, env = prepare(case, cameras=True)
            max_state_error = 0.0
            n = 0
            try:
                with np.load(run_dir / "cases" / case["id"] / "audit_replay_only.npz") as replay:
                    actions, states = replay["applied_action16"], replay["state16"]
                observation, info = env.reset(seed=case["seed"], options=bench.reset_options(case["seed"]))
                for key in (FRONT_IMAGE, LEFT_WRIST_IMAGE, RIGHT_WRIST_IMAGE):
                    assert observation[key].std() > 1.0, f"blank {key}"
                context = EpisodeContext(seed=case["seed"], task=task, action_mode="cartesian_delta")
                recorder.start_episode(context, "verified_scene_expert_replay")
                for n, (action, wanted) in enumerate(zip(actions, states, strict=True), start=1):
                    max_state_error = max(max_state_error, float(np.max(np.abs(observation[STATE] - wanted))))
                    if max_state_error > 1e-5:
                        raise RuntimeError(f"physical state disagrees at step {n}: {max_state_error}")
                    previous = observation
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated or not np.allclose(info["applied_action"], action, rtol=0, atol=1e-9):
                        raise RuntimeError(f"physical replay disagrees at step {n}: {info.get('safety_reason')}")
                    recorder.add_frame(previous, info["applied_action"])
                assert n == row["steps"] and info["success"]
                assert info["cookies_in_target"] == row["max_score"]
                assert info["cookies_in_source"] == row["cookies_in_source"]
                metadata = dict(row["randomization"])
                if case["scene"]:
                    # Independently re-evaluate the complete scene at the end
                    # of the camera replay; keep original grasp reports.
                    metadata["scenario_metrics"] = info["scenario_metrics"]
                metadata.update(physical_replay_verified=True, replay_max_state_error=max_state_error)
                recorder.set_episode_metadata(metadata)
                recorder.finish_episode(success=True)
                record = {"case": case["id"], "export_ok": True, "frames": n,
                          "max_state_error": max_state_error, "wall_seconds": round(time.monotonic() - t0, 2)}
            except Exception as exc:
                recorder.discard_episode()
                record = {"case": case["id"], "export_ok": False, "frames": n,
                          "max_state_error": max_state_error, "reason": str(exc)}
            finally:
                env.close()
            evidence.append(record)
            save_json(dest / "replay_evidence.json", evidence)
            print(record, flush=True)
    finally:
        recorder.close()
    audit = audit_training_dataset(dest, repo_id=repo_id)
    save_json(dest / "dataset_audit.json", audit)
    return {"dataset": str(dest), "audit": audit, "episodes": evidence}


def export_group_safe(rows, run_dir, root, use_videos):
    try:
        return export_group(rows, run_dir, root, use_videos=use_videos)
    except Exception as exc:
        return {"case_ids": [r["case"]["id"] for r in rows], "error": str(exc), "audit": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--inline-images", action="store_true")
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2),
                        help="At most two renderer processes on the same bound physical GPU")
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    rows = [r for r in result["results"] if r["success"] and r["saved_numeric_demo"]]
    if args.case:
        rows = [r for r in rows if r["case"]["id"] in args.case]
        assert len(rows) == len(set(args.case)), "requested case was not accepted"
    if not rows:
        raise RuntimeError("there are no accepted expert demonstrations to render")
    args.output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"]["task"], row["case"]["scenario"])].append(row)
    exports = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(export_group_safe, group, args.run, args.output,
                                                not args.inline_images) for group in groups.values()]):
            exported = future.result()
            exports.append(exported)
            save_json(args.output / "exports.json", exports)
            print("Dataset result", exported.get("dataset", exported.get("case_ids")),
                  "audit", exported.get("audit"), "error", exported.get("error"), flush=True)
    print("Exported", len(exports), "datasets", flush=True)


if __name__ == "__main__":
    main()
