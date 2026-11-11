#!/usr/bin/env python3
"""Does `adopt` plan and place correctly in the twenty-slot box?

Rebuilds the occupancy a v9 2x10 episode ended on -- read from the recorded
`success_diagnostics.json` -- by teleporting that many Cookies into the same slot
indices, then hands the box to `A3Cookie2x10RecoveryExpert` and lets it run.

What this measures and what it does not:

* it **does** exercise the part that was missing: reading an occupancy that is
  scattered across both ten-row columns, cutting it into contiguous free runs,
  turning those into groups of five, aiming each group at its own rows, and placing
  it there;
* it does **not** reproduce the policy's exact world geometry (the benchmark's box
  pose is randomised per seed and this scene's is its own), nor the arm pose the
  policy stopped in.  So the numbers here are about the *plan* and the *placement*,
  not about the takeover's physical difficulty -- that needs a live policy phase,
  i.e. the collector.

Usage:

    python scripts/probe_2x10_recovery.py --seeds 0 2 3 4 7 --steps 900
    python scripts/probe_2x10_recovery.py --dry-run          # plan only, no physics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_dual_arm_sim.batch_expert_2x10 import A3Cookie2x10RecoveryExpert  # noqa: E402
from a3_dual_arm_sim.collection import scene_by_name  # noqa: E402

#: Where the recorded v9 2x10 episodes live.
DIAG = Path("/root/autodl-tmp/diag/v9_run/2x10")


def recorded_occupancy(seed: int) -> tuple[list[int], int]:
    """The slot indices a policy left filled, and how many Cookies it had in the box."""

    path = DIAG / f"seed_{seed}" / "success_diagnostics.json"
    data = json.loads(path.read_text())
    conditions = data["final_conditions"]
    occupancy = conditions["target_slot_occupancy"]
    filled = [index for index, cookie in enumerate(occupancy) if cookie >= 0]
    return filled, int(conditions["cookies_in_target"])


def build_env(seed: int):
    scene = scene_by_name("a3_cookie_2x10_recovery")
    env = scene.build_env(0.0, 0.0)
    observation, _ = env.reset(
        seed=seed,
        options={"randomize_scene": False, "randomize_cookies": False},
    )
    return env, observation


def prefill(env, slots: list[int], rng: np.random.Generator) -> list[int]:
    """Put one Cookie standing in each of ``slots``, drawn from the source box.

    The imperfections are the batch expert's own `_prefill_first_batch` recipe, drawn
    once for the whole group: a coherent lean is a parallelogram, while independently
    drawn leans interpenetrate (the slots are one Cookie thick apart).
    """

    lean_azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    lean = float(rng.uniform(0.0, np.deg2rad(3.0)))
    yaw = float(rng.uniform(-np.deg2rad(0.8), np.deg2rad(0.8)))
    # Take from the far end of the last source column, so the rows the expert is
    # about to use are left pristine -- the same choice `_prefill_first_batch` makes.
    order = sorted(
        range(len(env.SOURCE_POSITIONS)),
        key=lambda index: (-float(np.asarray(env.SOURCE_POSITIONS)[index, 0]),
                           -float(np.asarray(env.SOURCE_POSITIONS)[index, 1])),
    )
    used = []
    for slot, cookie in zip(slots, order):
        env.place_cookie_in_slot(
            cookie,
            slot,
            lean_azimuth_rad=lean_azimuth,
            lean_rad=lean,
            yaw_rad=yaw,
            max_jitter_m=0.00015,
        )
        used.append(cookie)
    return used


def slots_filled(env) -> int:
    inside = tuple(
        env._cookie_inside_target(index) for index in range(len(env.cookie_positions))
    )
    return sum(1 for index in env._target_slot_occupancy(inside) if index >= 0)


def in_target(env) -> int:
    """How many Cookies are inside the target box -- **the number main's 2x10 scores**.

    Slot occupancy is a stricter measure and the wrong one for this comparison: the batch
    expert's group of five lands 3-5 on slot centres even when all five are inside the box,
    which is exactly why the 2x5 contract uses ``require_exact_slots=False``.  A probe that
    reported slots would understate the recovery by the same amount the 2x5 data does.
    """

    return sum(
        1 for index in range(len(env.cookie_positions)) if env._cookie_inside_target(index)
    )


def run_one(seed: int, steps: int, dry_run: bool) -> dict:
    filled, recorded_in_target = recorded_occupancy(seed)
    env, _observation = build_env(seed)
    rng = np.random.default_rng(seed)
    prefill(env, filled, rng)
    before = slots_filled(env)

    env.second_batch_mode = "adopt"
    expert = A3Cookie2x10RecoveryExpert(env)
    expert.reset()

    report = {
        "seed": seed,
        "recorded_in_target": recorded_in_target,
        "recorded_slots": len(filled),
        "slots_before": before,
        "in_target_before": in_target(env),
        "planned": [
            {"column": column, "rows": list(rows)} for column, rows in expert._targets
        ],
        "skipped_runs": expert._skipped_runs,
        "infeasible": expert.adopt_infeasible,
        "adopted_grasp": list(getattr(expert, "adopted_grasp", []) or []),
    }
    if dry_run or expert.adopt_infeasible is not None:
        env.close()
        report.update(
            steps=0,
            slots_after=before,
            in_target_after=report["in_target_before"],
            failed=expert.adopt_infeasible is not None,
            failure_phase="ADOPT" if expert.adopt_infeasible else None,
            failure_reason=expert.adopt_infeasible,
        )
        return report

    observation = env._observation()
    task = "probe"
    taken = 0
    for taken in range(1, steps + 1):
        action = expert.act(observation, task)
        observation, _reward, terminated, truncated, _info = env.step(action)
        if expert.failed or expert.finished or terminated or truncated:
            break
    report.update(
        steps=taken,
        slots_after=slots_filled(env),
        in_target_after=in_target(env),
        phase=expert.phase.name,
        failed=bool(expert.failed),
        finished=bool(expert.finished),
        failure_phase=getattr(expert, "failure_phase", None),
        failure_reason=getattr(expert, "failure_reason", None),
        batches=len(expert.batch_reports),
        status=expert.status,
    )
    env.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 3, 4, 7])
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="plan only, no physics")
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        report = run_one(seed, args.steps, args.dry_run)
        results.append(report)
        gained = report["slots_after"] - report["slots_before"]
        delivered = report["in_target_after"] - report["in_target_before"]
        print(
            f"seed {seed:>2} | recorded {report['recorded_in_target']:>2} "
            f"| in-target {report['in_target_before']:>2} -> {report['in_target_after']:>2} "
            f"(+{delivered}) "
            f"| slots {report['slots_before']:>2} -> {report['slots_after']:>2} (+{gained}) "
            f"| plan {[len(t['rows']) for t in report['planned']]} "
            f"| cols {[t['column'] for t in report['planned']]} "
            f"| {'FAILED ' + str(report.get('failure_phase')) if report.get('failed') else 'ok'} "
            f"| {report.get('steps', 0)} steps",
            flush=True,
        )
        if report["infeasible"]:
            print(f"           infeasible: {report['infeasible']}", flush=True)
        if report.get("failure_reason"):
            print(f"           reason: {report['failure_reason']}", flush=True)

    out = Path("/root/autodl-tmp/diag/2x10_recovery_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
