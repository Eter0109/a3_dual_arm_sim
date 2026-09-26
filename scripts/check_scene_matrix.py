"""Phase 7: the representative-and-boundary matrix, as a success-rate sweep.

Its own sizing note says about twelve scenes at three seeds each, which is a couple of
hours -- and that is only true if the runs skip rendering, because the renderer is
measured at 71% of a step (307 of 428 ms) and none of these questions are about pixels.
So each cell *derives* its config (through the same `render_config` the pinned-config
test runs, so a cell cannot disagree with a shipped file), builds the environment with
rendering off, and drives the scripted expert for a bounded number of steps.

The point is not to run the cross product: it is to know that every mechanism and every
boundary in the configurable-scene framework works, and to see the ones where success
collapses rather than merely dips.

    python scripts/check_scene_matrix.py [--seeds 0,1,2] [--shard i/n] [--only name]

Each episode prints one line when it finishes, so a sweep can be followed with `tail -f`
and stopped without losing what it already measured.  Cells are independent, which is
what `--shard` splits across processes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from generate_scene_config import render_config

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.cookie_transfer import (
    A3CookieTransferEnv,
    CookieTransferTaskConfig,
)
from a3_dual_arm_sim.expert import CookiePhase
from a3_dual_arm_sim.relay_batch_expert import RelayBatchExpert
from a3_dual_arm_sim.same_column_batch_expert import A3SameColumnBatchExpert
from a3_dual_arm_sim.scene_spec import SceneSpec, same_column_spec, two_box_spec

TWO_BOX_BASE = PROJECT / "configs" / "cookie_two_box_batch.yaml"
SINGLE_BASE = PROJECT / "configs" / "cookie_same_column.yaml"

#: One entry per matrix cell.  A boundary is a representative run with a single
#: parameter moved, which is why this is a dozen cells rather than a cross product.
#:
#: One entry per matrix cell.  A boundary is a representative run with a single
#: parameter moved, which is why this is a dozen cells rather than a cross product.
#:
#: Six cells are *expected* to be refused, and the matrix records the refusal rather
#: than hiding it -- each one is a boundary the plan's table named but the derivation
#: disagreed with, and the derivation is what the simulator enforces.  Between them
#: they show that the two knobs are tightly coupled at this station:
#:
#: * the capacity boundary is **9 or 10**, and each bound is a different measured
#:   limit: below it a column has fewer rows than a grasp needs, 11 to 14 derive a
#:   shorter remainder batch (below the measured grasp floor of 5, and a capacity of 14
#:   was run to confirm it -- "batch 2 CLOSE timed out, pad forces=[0.0, 0.0] N"), 15
#:   and up put a batch deeper than the fill can place, and 20 is past the push pads'
#:   reach as well;
#: * `per_grasp` is **5 only**: 1 and 2 are refused by the placement depth, 3 and 4 by
#:   the remainder batch;
#: * `per_grasp_10` needs ten rows, which the jaw travel cannot reach at this station
#:   at all, so it is refused before the depth bound is even consulted;
#: * `source_exact` cannot be expressed at all.  The source layout supplies 40
#:   graspable Cookies and `source_cookies=20` asks for a layout that supplies 20, so
#:   two boxes needing 20 of them are refused -- the axis needs a *layout* knob (usable
#:   columns) rather than a count to have a boundary here.
VARIANTS: list[dict] = [
    {"name": "base", "single": False, "spec": {}},
    {"name": "randomized", "single": False, "spec": {}, "randomize": True},
    {"name": "single_randomized", "single": True, "spec": {}, "randomize": True},
    {"name": "per_grasp_1", "single": False, "spec": {"per_grasp": 1}},
    {"name": "per_grasp_2", "single": False, "spec": {"per_grasp": 2}},
    {"name": "per_grasp_3", "single": False, "spec": {"per_grasp": 3}},
    {"name": "per_grasp_4", "single": False, "spec": {"per_grasp": 4}},
    {"name": "per_grasp_10", "single": False, "spec": {"per_grasp": 10, "box_capacity": 20}},
    {"name": "boxes_1", "single": False, "spec": {"boxes": 1}},
    {"name": "boxes_3", "single": False, "spec": {"boxes": 3}},
    {"name": "boxes_4", "single": False, "spec": {"boxes": 4}},
    {"name": "capacity_9", "single": False, "spec": {"box_capacity": 9}},
    {"name": "capacity_14", "single": False, "spec": {"box_capacity": 14}},
    {"name": "capacity_18", "single": False, "spec": {"box_capacity": 18}},
    {"name": "source_exact", "single": False, "spec": {"source_cookies": 20}},
    {"name": "queue_gap_wide", "single": False, "spec": {"queue_gap_m": 0.260}},
]


def spec_for(variant: dict) -> SceneSpec:
    """The spec for one cell: a shipped spec with this cell's parameters moved."""

    spec = same_column_spec() if variant["single"] else two_box_spec()
    if variant["spec"]:
        spec = replace(spec, **variant["spec"])
    return spec


def config_for(variant: dict):
    """Derive this cell's config through the generator, so it cannot drift from a file.

    ``render_config`` is the code the test that pins a committed config against its
    spec runs, which is the point: a cell built any other way could disagree with what
    the repository actually ships.
    """

    base = SINGLE_BASE if variant["single"] else TWO_BOX_BASE
    text = render_config(base, spec_for(variant))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        config = load_config(path)
    finally:
        path.unlink(missing_ok=True)
    # `per_grasp` is tuning rather than geometry -- the spec derives the rows and the
    # config states the grasp size -- so both ends of the check have to move together.
    per_grasp = variant["spec"].get("per_grasp")
    if per_grasp is not None:
        config = replace(
            config,
            batch_expert_per_grasp=per_grasp,
            min_verified_batch_expert_per_grasp=per_grasp,
        )
    return config


def run_one(variant: dict, seed: int, max_steps: int) -> dict:
    """One episode: derive, reset, drive the expert, grade it the way the scene does.

    A refusal from the framework is an *outcome*, not a harness fault.  Both the spec's
    bounds check and the expert's reachability pre-check raise rather than return, and
    they raise for exactly the layouts this matrix exists to probe -- so they are caught
    here and reported as the episode's failure, with the framework's own sentence as the
    reason.
    """

    started = time.time()
    config = None
    env = None
    try:
        config = config_for(variant)
        env = A3CookieTransferEnv(
            config,
            render_cameras=False,
            task_config=CookieTransferTaskConfig(
                require_exact_slots=False,
                # The batch scenes' own contract: a Cookie shoved flat by a push is
                # still in the box, and grading posture would reject a run for
                # succeeding at the harder half.
                require_released=True,
                require_upright=False,
                terminate_on_success=False,
            ),
        )
        env.reset(
            seed=seed,
            options={
                "randomize_scene": bool(variant.get("randomize")),
                "randomize_cookies": False,
            },
        )
        expert = A3SameColumnBatchExpert(env) if variant["single"] else RelayBatchExpert(env)
        expert.reset()
        # The two experts announce completion differently: the relay has a ``done``
        # flag, while the single-box expert ends in ``CookiePhase.DONE`` and expects a
        # hold window after it (see ``SingleBoxCollectionPolicy``).  Both are read here
        # so one cell's interface cannot decide another's grading.
        hold_steps = 40
        completed_for = 0
        step = 0
        for step in range(1, max_steps + 1):
            # No inner guard: a refusal here propagates to the `except` below, which
            # reports it with the framework's own sentence as the episode's reason.
            action = expert.act()
            _, _, terminated, truncated, _ = env.step(action)
            finished = bool(getattr(expert, "done", False)) or (
                "phase" in dir(expert) and getattr(expert, "phase", None) is CookiePhase.DONE
            )
            completed_for = completed_for + 1 if finished else 0
            if expert.failed or terminated or truncated:
                break
            if completed_for >= hold_steps:
                break

        boxes = getattr(env, "target_bin_bodies", ())
        # Geometric rather than `privileged_cookie_in_target`: that helper reads the
        # *current* target body, which for a lane is whichever box the relay last
        # switched to, so counting a lane through it would report the station box
        # twice and the queue never.  A Cookie is in a box when its centre is inside
        # the box's own half extents.
        half = np.asarray(env.config.cookie_transfer.target_bin_half_size_m)
        placed = sum(
            1
            for body in boxes
            for index in range(len(env._cookie_bodies))
            if np.all(
                np.abs(env.data.xpos[env._cookie_bodies[index]][:2] - env.data.xpos[body][:2])
                <= half
            )
        )
        source = sum(
            env.privileged_cookie_in_source(index) for index in range(len(env._cookie_bodies))
        )
        return {
            "variant": variant["name"],
            "seed": seed,
            "step": step,
            "seconds": round(time.time() - started, 1),
            "stage": "DONE" if completed_for >= hold_steps else expert.stage,
            "done": completed_for >= hold_steps,
            "failed": expert.failed,
            "counts": expert.all_counts() if hasattr(expert, "all_counts") else None,
            "placed": placed,
            # The *capacity*, not the slot count: a lattice is `ceil(capacity /
            # columns)` rows tall, so a capacity that is not a multiple of the column
            # count leaves its last slots empty by design (`BatchPlan` documents it).
            # The config stores only the slots -- capacity is a spec-level quantity --
            # so it comes from the spec.  Counting slots made a capacity-9 cell report
            # "15/20" when 18 is the most it could ever place.
            "expected": min(
                spec_for(variant).box_capacity, len(env.config.cookie_transfer.target_slots_local_m)
            )
            * len(boxes),
            "source": source,
            "boxes": len(boxes),
        }
    except (RuntimeError, ValueError) as error:
        return {
            "variant": variant["name"],
            "seed": seed,
            "step": 0,
            "seconds": round(time.time() - started, 1),
            "stage": "REFUSED",
            "done": False,
            "failed": f"{type(error).__name__}: {error}",
            "counts": None,
            "placed": 0,
            "expected": 0,
            "source": None,
            "boxes": 0,
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--shard", default="0/1", help="i/n, for parallel cells")
    parser.add_argument("--only", default=None, help="one cell by name, or a prefix")
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    seeds = [int(part) for part in args.seeds.split(",") if part != ""]
    index_text, _, count_text = args.shard.partition("/")
    index, count = int(index_text), int(count_text)
    cells = [
        (position, variant)
        for position, variant in enumerate(VARIANTS)
        if position % count == index
        and (
            args.only is None
            or variant["name"] == args.only
            or variant["name"].startswith(args.only)
        )
    ]

    print(
        f"matrix: {len(cells)} cells x {len(seeds)} seeds on shard {index}/{count}, seeds {seeds}",
        flush=True,
    )
    results: list[dict] = []
    for position, variant in cells:
        for seed in seeds:
            try:
                result = run_one(variant, seed, args.max_steps)
            except Exception as error:  # noqa: BLE001 - one cell must not stop a sweep
                result = {
                    "variant": variant["name"],
                    "seed": seed,
                    "error": f"{type(error).__name__}: {error}",
                    "done": False,
                }
            results.append(result)
            if "error" in result:
                verdict, detail = "ERROR", result["error"]
            elif result["done"]:
                verdict = "OK"
                detail = (
                    f"step {result['step']} ({result['seconds']}s) "
                    f"{result['placed']}/{result['expected']} placed, "
                    f"source {result['source']}"
                )
            else:
                verdict = "FAIL"
                detail = (
                    f"step {result['step']} ({result['seconds']}s) stage "
                    f"{result['stage']} {result['failed']}; "
                    f"{result['placed']}/{result['expected']} placed"
                )
            print(
                f"[{position:2d}] {variant['name']:18s} seed {seed}: {verdict}  {detail}",
                flush=True,
            )
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(results, indent=2))

    successes = sum(1 for result in results if result.get("done"))
    print(f"\n{successes}/{len(results)} episodes completed", flush=True)
    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
