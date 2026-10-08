#!/usr/bin/env python
"""The guard set, as one command: what must still be true after an edit.

`docs/scene-transition-and-bug-triage.md` §5 asks for two invariants after every edit --
`a3_libero_panda` on seeds 0-9 expecting at least 9 placements, and the Cookie fingerprint --
and §4 step 0 asks for the numbers to be *recorded* so that drift is a diff.  This is both:

    python scripts/guard.py --record      # once, on a state that is known good
    python scripts/guard.py               # after every edit
    python scripts/guard.py --cheap       # the no-GPU half, in seconds

**Why the baseline is a tracked file and not a log.**  An expectation that lives under
`logs/` is gitignored, so it cannot be diffed, and a baseline that cannot be diffed is a
number somebody read once.  `configs/invariants_baseline.json` is a statement about this
scene, and statements belong in the repository next to the thing they describe.

**Why two halves.**  The cheap half loads a scene and renders one frame; the expensive half
starts a policy process and runs ten rollouts.  Both run by default because the guard's whole
job is the cross-scene case -- the mirror-scope bug (A5) was introduced by a change made for
one scene and was invisible from it -- and a guard you skip is not a guard.  `--cheap` exists
for the inner loop, and says so in its output.

**Every check reports three things**: what it measured, what it expected, and by how much they
differ.  A guard that prints `FAIL` without the numbers cannot be argued with, and the two
times this repository has had a wrong invariant the numbers were the only way to see it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BASELINE = PROJECT / "configs/invariants_baseline.json"
WORKDIR = Path("/root/autodl-tmp/logs/invariants")

COOKIE_SCENE = "a3_cookie_single_piece"
COOKIE_DATASET = Path("/root/autodl-tmp/datasets/a3_cookie_one")
BATCH_SCENE = "a3_cookie_batch"
BATCH_DATASET = Path("/root/autodl-tmp/datasets/a3_cookie_adv_1000_merged")

LIBERO_SCENE = "a3_libero_panda"
LIBERO_MANIFEST = PROJECT / "configs/policies/pi05_libero_panda.yaml"
LIBERO_STEPS = 300

#: The environment every child process needs.  Set here rather than in the caller's shell
#: because the guard is the thing that must be reproducible; an `MUJOCO_GL` left unset does
#: not fail, it renders nothing and compares the fingerprint against a blank frame.
CHILD_ENV = {
    "MUJOCO_GL": "egl",
    "HF_HOME": "/root/autodl-tmp/hf",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

#: How far a value may drift before it is drift and not noise.  Each one is justified by what
#: the quantity is, not by a round number:
#:
#: `state` is exact arithmetic -- the same fifteen joint angles written in and read back --
#: so the only tolerance it needs is float noise.  Anything above it is a convention error,
#: which is exactly how the quaternion-order bug was found.
STATE_TOLERANCE = 1e-6
#: The render is stochastic: the scene randomises prop placement, and the fingerprint is
#: compared against the *within-episode* noise floor.  A factor of two on the achieved MSE is
#: generous for that and still catches the failure it exists for, which is a mirrored or
#: rotated image -- those score 55x worse, not 2x.
IMAGE_MSE_FACTOR = 2.0
#: The gripper travels 0 -> 1; the open/close separation is ~0.97 and the check below is a
#: fifth of it.  A convention error puts it at ~0.
GRIPPER_OPEN_TOLERANCE = 0.20
#: The delivery ratio is a servo property with a measured value of 0.43 (D1/D2 in the
#: catalogue).  A quarter either way covers a retune; the failure it guards against moved it
#: by half.
TRACKING_RELATIVE_TOLERANCE = 0.25
#: Drift is a **second claim** about a replay, reported in its own row: the row above it asks
#: "is this within the tolerance the tool documents", this one asks "has it moved since the
#: baseline".  A single row cannot answer both -- a residual 70x its baseline can sit happily
#: under an absolute tolerance -- and merging them would produce a verdict that is neither.
#: The floor is what keeps a near-zero baseline (single piece: 3.5e-05) from turning float
#: noise into an infinite ratio.
REPLAY_DRIFT_FACTOR = 5.0
REPLAY_DRIFT_FLOOR = 0.05


class Report:
    """One row per check: measured, expected, and the difference between them."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, bool]] = []

    def add(self, name: str, measured: str, expected: str, difference: str,
            ok: bool) -> None:
        self.rows.append((name, measured, expected, difference, ok))

    @property
    def ok(self) -> bool:
        return all(row[4] for row in self.rows)

    def print(self) -> None:
        width = max((len(row[0]) for row in self.rows), default=0)
        print(f"\n{'check':<{width}}  {'measured':>16}  {'expected':>16}  {'difference':>14}")
        print("-" * (width + 2 + 18 + 18 + 16))
        for name, measured, expected, difference, ok in self.rows:
            mark = " " if ok else "!"
            print(f"{name:<{width}}  {measured:>16}  {expected:>16}  {difference:>14} {mark}")


#: Lines worth relaying to the terminal while a child runs.  The guard's slow half is ten
#: rollouts and several minutes of it is the checkpoint loading; a runner that prints nothing
#: until it is finished is indistinguishable from one that has hung, and the first thing a
#: person does with a silent long command is kill it.  Everything is still written to the log
#: in full -- this is a filter on the *console*, not on the record.
RELAY_PREFIXES = (
    "seed ", "verdict", "nearest", "bowl ", "steps with", "mean angle", "left eef",
    "trace", "figure", "episode ", "scene ", "state ", "image ", "gripper ", "tracking ",
    "one step", "open loop", "json ", "INVARIANTS", "INVARIANT", "DYNAMICS", "STOPPED",
)


def run(command: list[str], log: Path) -> tuple[int, str]:
    """Run one child, tee it to `log`, relay the summary lines, return `(code, stdout)`."""

    environment = {**os.environ, **{k: v for k, v in CHILD_ENV.items() if k not in os.environ}}
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(command)}", flush=True)
    collected: list[str] = []
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=PROJECT, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in process.stdout:
            collected.append(line)
            handle.write(line)
            if line.lstrip().startswith(RELAY_PREFIXES):
                print("    " + line.rstrip(), flush=True)
        process.wait()
    return process.returncode, "".join(collected)


def python(script: str) -> list[str]:
    """The venv that has mujoco and lerobot, not whichever interpreter ran this file."""

    return [PYTHON, str(PROJECT / "scripts" / script)]


#: Resolved in `main` from `--python`, defaulting to the interpreter running this file.  The
#: guard itself is stdlib-only, so it *can* be started by any python -- and starting it with
#: one that lacks mujoco would put an import traceback in a log file and report `exited 1`,
#: which says nothing about the scene.  The check below turns that into one sentence.
PYTHON = sys.executable


def parse_placements(output: str) -> tuple[int, int]:
    """`(placements, rollouts)` from the probe's `verdict, per seed:` summary line."""

    match = re.search(r"verdict, per seed:\s*(.*)", output)
    if match is None:
        raise SystemExit("the rollout probe printed no per-seed verdict line; its output is "
                         "in the log and the run cannot be judged without it")
    verdicts = [part.strip() for part in match.group(1).split(",") if part.strip()]
    return sum(verdict == "PLACED" for verdict in verdicts), len(verdicts)


# --------------------------------------------------------------------------------------
# The four measurements
# --------------------------------------------------------------------------------------


def measure_invariants(log: Path) -> dict:
    """Channels 1, 2, 4, 5: the fingerprint, the state convention, tracking, the gripper."""

    output = WORKDIR / "cookie_invariants.json"
    code, text = run(python("check_invariants.py")
                     + ["--scene", COOKIE_SCENE, "--dataset", str(COOKIE_DATASET),
                        "--output", str(output)], log)
    # The tool's own exit code is authoritative when it fails: it holds an absolute invariant
    # (`|ours - recorded| ~ 0`), and that is a stronger statement than any comparison to a
    # baseline can make.
    if code != 0:
        print(text[-4000:])
        raise SystemExit(f"check_invariants.py exited {code}; see {log}")
    return json.loads(output.read_text())


def measure_replay(scene: str, dataset: Path, log: Path) -> dict:
    """Channels 3 and 6: the teacher-forced step, and the open loop that follows it."""

    output = WORKDIR / f"replay_{scene}.json"
    code, text = run(python("replay_vs_dataset.py")
                     + ["--scene", scene, "--dataset", str(dataset),
                        "--output", str(output)], log)
    if code != 0 or not output.exists():
        print(text[-4000:])
        raise SystemExit(f"replay_vs_dataset.py exited {code} on {scene}; see {log}")
    return json.loads(output.read_text())


def measure_rollouts(seeds: str, log: Path) -> tuple[int, int, str]:
    """The policy half: `(placements, rollouts, the probe's own output)`."""

    code, text = run(python("probe_libero_props_trace.py")
                     + ["--scene", LIBERO_SCENE, "--manifest", str(LIBERO_MANIFEST),
                        "--seeds", seeds, "--steps", str(LIBERO_STEPS),
                        "--trace", str(WORKDIR / "libero_guard.npz"),
                        "--figure", str(WORKDIR / "libero_guard.png")], log)
    if code != 0:
        print(text[-4000:])
        raise SystemExit(f"probe_libero_props_trace.py exited {code}; see {log}")
    placements, rollouts = parse_placements(text)
    return placements, rollouts, text


# --------------------------------------------------------------------------------------
# The comparisons
# --------------------------------------------------------------------------------------


def compare(baseline: dict, invariants: dict, replay: dict, rollouts: dict | None,
            expected_placements: int, report: Report) -> None:
    """Every check, in the order the ledger lists them."""

    # ---- channel 2: the state convention, absolute ----
    measured = invariants["state"]["max_abs_diff"]
    report.add("state |ours - recorded|", f"{measured:.2e}",
               f"< {STATE_TOLERANCE:.0e}", f"{measured - 0:.2e}",
               measured < STATE_TOLERANCE)

    # ---- channel 1: the fingerprint, per camera ----
    for camera, entry in invariants["images"].items():
        was = baseline["invariants"]["images"].get(camera)
        if was is None:
            report.add(f"image {camera}", entry["best"], "as_is", "not in baseline", False)
            continue
        achieved = entry["scores"][entry["best"]]
        same = entry["best"] == was["best"]
        allowed = was["mse"] * IMAGE_MSE_FACTOR
        # The difference column has to name the reason the row failed, and for a wrong
        # orientation the MSE ratio is not the reason *or* informative: flipping the image
        # costs nothing in error while it disagrees with the dataset completely.
        report.add(f"image {camera} ({entry['best']})",
                   f"{achieved:.0f}", f"<= {allowed:.0f}",
                   f"{achieved / max(was['mse'], 1e-9):.2f}x" if same
                   else f"{entry['best']} != {was['best']}",
                   bool(entry["passes"]) and same and achieved <= allowed)

    # ---- channel 5: the gripper round trip ----
    was = baseline["invariants"]["gripper"]
    for state in ("open", "close"):
        measured = invariants["gripper"][state]
        report.add(f"gripper {state}", f"{measured:.4f}", f"{was[state]:.4f}",
                   f"{abs(measured - was[state]):.4f}",
                   abs(measured - was[state]) <= GRIPPER_OPEN_TOLERANCE)
    report.add("gripper ordered", str(invariants["gripper"]["ordered"]), "True", " ",
               bool(invariants["gripper"]["ordered"]))

    # ---- channel 4: the delivery ratio ----
    measured = invariants["tracking"]["fraction"]
    was = baseline["invariants"]["tracking"]
    allowed = was * TRACKING_RELATIVE_TOLERANCE
    report.add("tracking fraction", f"{measured:.3f}", f"{was:.3f} +/- {allowed:.3f}",
               f"{measured - was:+.3f}", abs(measured - was) <= allowed)

    # ---- channels 3 and 6: the replay, per scene ----
    for scene, result in replay.items():
        teacher = result["teacher_forced"]
        was = baseline["replay"][scene]
        ratio = teacher["free_space_residual_over_step"]
        verdict = result["verdict"]
        report.add(f"replay {scene} verdict", verdict, "DYNAMICS MATCH", " ",
                   verdict == "DYNAMICS MATCH")
        report.add(f"replay {scene} residual", f"{ratio:.3f} of a step",
                   f"<= {result['tolerance_of_step']:.2f}",
                   f"within {ratio / result['tolerance_of_step']:.0%} of it",
                   ratio <= result["tolerance_of_step"])
        allowance = max(REPLAY_DRIFT_FLOOR, REPLAY_DRIFT_FACTOR * was["residual_over_step"])
        report.add(f"replay {scene} drift", f"{ratio:.4f}",
                   f"{was['residual_over_step']:.4f} +/- {allowance:.4f}",
                   f"{ratio - was['residual_over_step']:+.4f}",
                   abs(ratio - was["residual_over_step"]) <= allowance)
        report.add(f"replay {scene} open loop", f"{result['open_loop']['arm_peak_rad']:.4f}",
                   f"{was['open_loop_peak_rad']:.4f} rad", " ",
                   result["open_loop"]["arm_peak_rad"] <= max(
                       was["open_loop_peak_rad"] * 3.0, 0.05))

    # ---- the policy ----
    if rollouts is None:
        return
    placements, total = rollouts["placements"], rollouts["rollouts"]
    report.add("placements", f"{placements}/{total}", f">= {expected_placements}/{total}",
               f"{placements - expected_placements:+d}", placements >= expected_placements)


def record(seeds: str, expected_placements: int, cheap: bool) -> dict:
    """Run everything once and write the baseline.

    `--cheap` here records a baseline with no rollout row, which is enough to exercise every
    other comparison and is how the guard itself is smoke-tested.  It is not enough to guard
    the policy, and the file says so rather than leaving the absence to be inferred.
    """

    print("recording the baseline -- this must be run on a state that is known good")
    invariants = measure_invariants(WORKDIR / "record_cookie_invariants.log")
    replay = {
        COOKIE_SCENE: measure_replay(COOKIE_SCENE, COOKIE_DATASET,
                                     WORKDIR / "record_replay_cookie.log"),
        BATCH_SCENE: measure_replay(BATCH_SCENE, BATCH_DATASET,
                                    WORKDIR / "record_replay_batch.log"),
    }
    rollouts: dict | None = None
    if not cheap:
        placements, total, _text = measure_rollouts(seeds, WORKDIR / "record_rollouts.log")
        rollouts = {"placements": placements, "total": total}

    baseline = {
        "_comment": "Written by scripts/guard.py --record.  Edit by re-recording, not by hand.",
        "seeds": seeds,
        "expected_placements": expected_placements,
        "invariants": {
            "scene": COOKIE_SCENE,
            "dataset": str(COOKIE_DATASET),
            "images": {camera: {"best": entry["best"], "mse": entry["scores"][entry["best"]],
                                "noise": entry["noise"], "margin": entry["margin"]}
                       for camera, entry in invariants["images"].items()},
            "state_max_abs_diff": invariants["state"]["max_abs_diff"],
            "gripper": {state: invariants["gripper"][state] for state in ("open", "close")},
            "tracking": invariants["tracking"]["fraction"],
        },
        "replay": {scene: {"residual_over_step":
                           result["teacher_forced"]["free_space_residual_over_step"],
                           "open_loop_peak_rad": result["open_loop"]["arm_peak_rad"],
                           "verdict": result["verdict"]}
                   for scene, result in replay.items()},
        "rollouts": rollouts,
    }
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"\nbaseline {BASELINE}")
    for scene, result in replay.items():
        print(f"  replay {scene}: {result['verdict']} at "
              f"{result['teacher_forced']['free_space_residual_over_step']:.2f} of a step")
    if rollouts is None:
        print("  rollouts NOT recorded (--cheap): the policy half of this baseline is missing")
    else:
        print(f"  rollouts {rollouts['placements']}/{rollouts['total']}")
    return baseline


def check_interpreter() -> None:
    """Fail loudly if the interpreter handed to the children has no mujoco in it."""

    probe = subprocess.run([PYTHON, "-c", "import mujoco"], text=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if probe.returncode != 0:
        raise SystemExit(f"{PYTHON} cannot import mujoco, so every check would fail for a "
                         f"reason that has nothing to do with the scene.\n"
                         f"        run it as: /root/autodl-tmp/a3-collect/.venv/bin/python "
                         f"scripts/guard.py   (or pass --python)\n"
                         f"        {probe.stderr.strip().splitlines()[-1] if probe.stderr else ''}")


def main() -> int:
    global PYTHON
    args = parse_args()
    PYTHON = args.python
    check_interpreter()
    if args.record:
        record(args.seeds, args.expect_placements or 9, args.cheap)
        return 0
    if not args.baseline.exists():
        raise SystemExit(f"no baseline at {args.baseline}; run `guard.py --record` on a state "
                         f"that is known good, and read it before trusting it")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    print(f"guard set for {PROJECT.name}"
          + ("   (--cheap: the policy half is not run)" if args.cheap else ""))

    print("\ninvariants (no GPU)")
    invariants = measure_invariants(WORKDIR / "cookie_invariants.log")
    print("\nreplay (no GPU)")
    replay = {
        COOKIE_SCENE: measure_replay(COOKIE_SCENE, COOKIE_DATASET,
                                     WORKDIR / "replay_cookie.log"),
        BATCH_SCENE: measure_replay(BATCH_SCENE, BATCH_DATASET,
                                    WORKDIR / "replay_batch.log"),
    }

    rollouts = None
    if not args.cheap:
        print(f"\nrollouts ({LIBERO_SCENE}, seeds {args.seeds}) -- this is the slow one")
        placements, total, _text = measure_rollouts(args.seeds, WORKDIR / "rollouts.log")
        rollouts = {"placements": placements, "rollouts": total}

    report = Report()
    expected = (baseline["expected_placements"] if args.expect_placements is None
                else args.expect_placements)
    compare(baseline, invariants, replay, rollouts, expected, report)
    report.print()

    if report.ok:
        print("\nGUARD HOLDS")
        if args.cheap:
            print("(--cheap: the policy half was not run, so this is half the guard)")
        return 0
    failed = sum(1 for row in report.rows if not row[4])
    print(f"\nGUARD FAILED -- {failed} of {len(report.rows)} checks; the numbers above are the "
          f"claim, and `git diff` on {BASELINE.relative_to(PROJECT)} is what changed")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true",
                        help="run everything and write the baseline; do this only on a state "
                             "that is known good")
    parser.add_argument("--cheap", action="store_true",
                        help="skip the policy rollouts: the half that needs no GPU and no "
                             "checkpoint, in seconds")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--expect-placements", type=int, default=None,
                        help="override the baseline's expectation for how many of the "
                             "rollouts place the bowl; --cheap skips the rollouts entirely")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter for the child scripts; it must be the venv "
                             "that has mujoco and lerobot")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
