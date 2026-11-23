"""Repeatable expert/collection smoke matrix for opt-in Cookie scenes.

Runs real continuous physical episodes using the public benchmark runner.
Accepts data only after its full release, retreat and stable success contract.
No object-state manipulation is used by the rollout or recorder.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import mujoco
import numpy as np

from a3_dual_arm_sim.contracts import EEF_POSE, FORCE, STATE, VELOCITY
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

PROFILES = ("basic", "medium", "advanced")
SCENARIOS = (
    "native", "missing_five_groups", "missing_other_columns", "tilted_pick",
    "fragile", "two_boxes", "three_boxes", "four_boxes", "combined",
)


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value), encoding="utf-8")
    tmp.replace(path)


def cases():
    for scenario, task, profile in product(SCENARIOS, ("2x5", "2x10"), PROFILES):
        scene, seed = {}, 0
        if scenario == "missing_five_groups":
            scene, seed = {"source_gap_groups": [2, 5]}, 4
        elif scenario == "missing_other_columns":
            scene, seed = {"source_gap_groups": [2, 5]}, 3
        elif scenario == "tilted_pick":
            scene = {"source_tilt_deg": 8}
        elif scenario == "fragile":
            scene = {"cookie_crush_force_n": 12}
        elif scenario in ("two_boxes", "three_boxes", "four_boxes"):
            scene = {"target_box_count": {"two_boxes": 2, "three_boxes": 3, "four_boxes": 4}[scenario]}
        elif scenario == "combined":
            scene, seed = {
                "source_gap_groups": [2, 5], "target_box_count": 4 if task == "2x5" else 3,
                "source_tilt_deg": 8, "cookie_crush_force_n": 12,
            }, 4
        yield {
            "id": f"{task}_{profile}_{scenario}_s{seed}", "task": task,
            "profile": profile, "scene": scene, "scenario": scenario, "seed": seed,
            "source_column": "1", "first_grasp": 5 if task == "2x10" else None,
            "max_steps": (1600 if task == "2x5" else 3200) * scene.get("target_box_count", 1),
        }


def prepare(case, *, cameras=False):
    settings = load_randomization_config()
    settings["scene"] = case["scene"]
    settings["profile"] = case["profile"]
    settings["source_column"] = case["source_column"]
    bench = make_cookie_benchmark(
        case["task"], profile=case["profile"], source_column=case["source_column"],
        randomization_settings=settings, first_grasp=case["first_grasp"],
        max_steps=case["max_steps"], complete_demonstrations=True,
        success_hold_steps=20, placement_order="canonical",
    )
    return bench, bench.create_env(render_cameras=cameras)


class AcceptedRecorder:
    def __init__(self, destination):
        self.destination = destination
        self.frames = []
        self.accepted = False
        self.count = 0
        self.metadata = {}

    def start_episode(self, context, controller_type):
        self.context, self.controller_type = context, controller_type

    def set_episode_metadata(self, metadata):
        self.metadata = dict(metadata)

    def add_frame(self, observation, action):
        self.frames.append((observation[STATE].copy(), np.asarray(action).copy(),
                            observation[VELOCITY][:8].copy(), observation[EEF_POSE][:7].copy(),
                            observation[FORCE][[0, 1, 2, 3, 4, 5, 12, 13, 16]].copy(),
                            float(observation["time"])))
        self.count += 1

    def finish_episode(self, success=None):
        assert success is True and self.frames
        s, a, v, p, f, t = map(np.asarray, zip(*self.frames))
        assert s.shape == a.shape == (self.count, 16)
        assert np.allclose(np.diff(t), .05, atol=1e-8)
        assert np.ptp(a[:, 8:], axis=0).max() < 1e-8
        assert all(np.isfinite(x).all() for x in (s, a, v, p, f, t))
        np.savez_compressed(self.destination / "demonstration_left8.npz", observation_state=s[:, :8],
                            action=a[:, :8].astype(np.float32), observation_velocity=v,
                            observation_eef_pose=p, observation_force=f, timestamp=t)
        np.savez_compressed(self.destination / "audit_replay_only.npz", state16=s, applied_action16=a)
        save_json(self.destination / "episode_metadata.json", {
            "format": "a3_raw_left8", "fps": 20, "frames": self.count,
            "task": self.context.task, "seed": self.context.seed,
            "controller_type": self.controller_type, "accepted_by_native_runner": True,
            "episode": self.metadata,
        })
        self.accepted = True
        self.frames.clear()

    def discard_episode(self):
        if self.frames:
            # Failed actions are a separate diagnostic replay artifact, never
            # training data. This makes future mid-episode diagnosis cheaper
            # without claiming a resumed suffix as a new complete success.
            states, actions, _, _, _, timestamps = map(np.asarray, zip(*self.frames))
            np.savez_compressed(self.destination / "failed_attempt_replay_only.npz",
                                state16=states, applied_action16=actions, timestamp=timestamps)
        self.frames.clear()

    def close(self):
        self.frames.clear()


class Trace:
    def __init__(self, dest):
        self.dest = dest
        self.steps = 0
        self.samples = []
        self.last_phase = None

    def add_frame(self, observation, action, info, *, phase, env, **kwargs):
        self.steps += 1
        if phase == self.last_phase and self.steps % 25:
            return
        transition = phase != self.last_phase
        self.last_phase = phase
        finger = set(env._left_finger_geoms)
        mapping = {g: i for i, geoms in enumerate(env._cookie_collision_geoms) for g in geoms}
        loads = {}
        for j in range(env.data.ncon):
            c = env.data.contact[j]
            a, b = int(c.geom1), int(c.geom2)
            if b in finger:
                a, b = b, a
            if a not in finger:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(env.model, env.data, j, wrench)
            key = f"{a}:{mapping.get(b, env.model.geom(b).name)}"
            loads[key] = loads.get(key, 0) + max(0.0, float(wrench[0]))
        self.samples.append({
            "step": self.steps, "phase": phase, "in_target": info["cookies_in_target"],
            "in_source": info["cookies_in_source"], "metrics": info.get("scenario_metrics"),
            "left_eef": env.data.site_xpos[env._eef_sites[0]].copy(),
            "left_rotation": env.data.site_xmat[env._eef_sites[0]].copy(),
            "gripper": env.current_joint_action[7], "gripper_command": info["applied_action"][7],
            "contact_loads": loads,
        })
        if transition and phase in ("OPEN", "RETRACT", "VERIFY_RELEASE", "FAILED") and env.scene_options.enabled:
            self.samples[-1]["cookie_poses"] = [{
                "id": i, "p": env.privileged_cookie_position(i),
                "up": env.data.xmat[env._cookie_bodies[i]].reshape(3, 3)[:, 2],
            } for i in np.flatnonzero(env._scene_randomizer.active)]
        if self.steps % 100 == 0:
            save_json(self.dest / "progress.json", self.samples[-1])


def run_case(case, root):
    dest = root / "cases" / case["id"]
    dest.mkdir(parents=True, exist_ok=False)
    recorder, diagnostic = AcceptedRecorder(dest), Trace(dest)
    env = None
    t0 = time.monotonic()
    try:
        bench, env = prepare(case)
        result = bench.run_episode("expert", seed=case["seed"], env=env,
                                   recorder=recorder, diagnostic=diagnostic).to_dict()
        result.update(evaluation_valid=True, saved_numeric_demo=recorder.accepted)
        tilt = set(result["randomization"].get("scene", {}).get("tilted_cookie_ids", []))
        transferred = {i for r in result["randomization"].get("grasp_reports", [])
                       if r.get("released") for i in r.get("cookies", [])}
        result["tilted_ids_transferred"] = sorted(tilt & transferred)
        if case["scene"] and not result["success"]:
            # Independent diagnosis of misplaced groups, never fed to the
            # policy or recorded in the training features.
            relevant = {i for report in result["randomization"].get("grasp_reports", [])
                        for i in report.get("cookies", [])}
            result["diagnostic_final_cookies"] = [{
                "id": i, "world_position": env.privileged_cookie_position(i),
                "world_up": env.data.xmat[env._cookie_bodies[i]].reshape(3, 3)[:, 2],
                "target_positions": [env.privileged_cookie_target_position(i, target_body=body)
                                     for body in env.target_box_bodies],
                "contained": [env.privileged_cookie_in_target(i, target_body=body)
                              for body in env.target_box_bodies],
            } for i in sorted(relevant)]
    except Exception as exc:
        result = {"success": False, "score": None, "max_score": (10 if case["task"] == "2x5" else 20)
                  * case["scene"].get("target_box_count", 1), "steps": diagnostic.steps,
                  "phase": "EXCEPTION", "failure_reason": str(exc), "traceback": traceback.format_exc(),
                  "evaluation_valid": False, "saved_numeric_demo": False}
    finally:
        recorder.close()
        if env is not None:
            env.close()
    result.update(case=case, total_wall_seconds=round(time.monotonic() - t0, 2))
    save_json(dest / "phase_trace.json", diagnostic.samples)
    save_json(dest / "result.json", result)
    return result


def report(rows, planned):
    return {
        "planned": planned, "completed": len(rows), "successful": sum(r["success"] for r in rows),
        "saved_numeric_demos": sum(r["saved_numeric_demo"] for r in rows),
        "errors": sum(not r["evaluation_valid"] for r in rows),
        "results": sorted(rows, key=lambda x: x["case"]["id"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    planned = list(cases())
    if args.case:
        by_id = {c["id"]: c for c in planned}
        assert len(set(args.case)) == len(args.case) and all(i in by_id for i in args.case), "invalid or duplicate case identifier"
        planned = [by_id[i] for i in args.case]
    elif args.pilot:
        desired = {"native", "missing_five_groups", "tilted_pick", "two_boxes"}
        planned = [c for c in planned if c["scenario"] in desired and c["profile"] == "basic"]
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "planned_cases.json", planned)
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_case, c, args.output) for c in planned]):
            row = future.result()
            rows.append(row)
            save_json(args.output / "results.json", report(rows, len(planned)))
            print(f"[{len(rows)}/{len(planned)}]", row["case"]["id"], row["score"], row["success"],
                  row["phase"], row["steps"], row.get("failure_reason"), row["total_wall_seconds"], flush=True)
    final = report(rows, len(planned))
    final["wall_seconds"] = round(time.monotonic() - started, 2)
    save_json(args.output / "results.json", final)
    print("DONE", final["completed"], final["successful"], final["errors"], flush=True)


if __name__ == "__main__":
    main()
