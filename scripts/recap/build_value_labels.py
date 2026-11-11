#!/usr/bin/env python3
"""Build RECAP value targets and per-frame rewards from a collection run's traces.

Why the labels come from `trace.jsonl` and not from a return computation
------------------------------------------------------------------------
The plan (§2.3) replaced RLinf's normalised-return target with the final Cookie count,
because a return normalised over `[-T_max, 0]` spends most of its 201 C51 bins separating
"lost by a lot" from "lost by a little" while the success boundary -- the only part that
matters -- gets about 26 bins.  The count target has a fixed range `[0, required_cookies]`,
needs no magic `failure_reward`, and is exactly the number the judge scores.

Two consequences worth stating:

* **No `compute_returns` stage.**  `cookies_in_target` per frame is already in the trace,
  written by main's own `info` in the same loop iteration as the dataset frame.
* **`is_success` needs no backfill.**  It is the final frame's `after.success`, again from
  main's judge.  The 400-episode training set has no such column (§2.5); this run does.

What it writes
--------------
One parquet with one row per episode:

| column | meaning |
|---|---|
| `part`, `episode_index`, `seed` | provenance, so a trace can be found again |
| `frames` | `T`, the number of control steps |
| `success` | strict contract, main's verdict |
| `value_target` | the layered potential reached at frame `T-1` -- what the value model regresses |
| `peak_count_target` | the old count-only target, kept so the two can be compared |
| `counts` | `cookies_in_target` at every frame, `T` long |
| `slotted`, `walls`, `source_kept` | the other three layers, as per-frame running maxima |
| `rewards` | change in the layered potential, length `T` (the first entry is the opening potential) |
| `instruction` | the sentence this episode was driven with |

Plus a `--check` mode that enforces the four invariants the plan lists, because a value
target that silently disagrees with the trace is the kind of bug that trains a model to
predict the wrong thing perfectly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layered_reward import DEFAULT_WEIGHTS as WEIGHTS, episode_rewards  # noqa: E402


def load_labels(part: Path) -> list[dict[str, Any]]:
    """The part's episode labels, **in file order**.

    File order is the dataset's episode order: `merge_parts.py` appends each part's episodes in
    exactly the order `episode_labels.jsonl` lists them, and `train_value.py` maps label rows to
    episodes *positionally* because the episodes table carries no seed.

    This returned a `dict` keyed by seed and the caller iterated `sorted(labels)`, which silently
    reordered any part whose labels are not already seed-ascending.  Measured on the 400-episode
    set: the rollout part happened to be ascending, so those 300 rows were fine, while the expert
    part is ordered 1000, 1006, 1012, ... and **98 of its 100 rows ended up attached to the wrong
    episode**.  The value head was trained on that corruption and still reached Spearman 0.699,
    which is why nobody noticed -- a bug that costs accuracy without failing is the expensive
    kind.  Sorting is now the caller's decision and there is only one correct one: don't.
    """
    path = part / "episode_labels.jsonl"
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_trace(part: Path, seed: int) -> list[dict[str, Any]]:
    path = part / "traces" / f"seed_{seed}" / "trace.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_row(part: Path, seed: int, label: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    """One episode: counts, rewards, and the value target they must agree on.

    **The reward is the change in the running maximum, not the change in the count.**

    Measured on the first 12 collected episodes, `cookies_in_target` is not monotone:
    100 % of episodes show at least one frame where the count falls, 18.5 times per episode
    on average.  Almost all of it is flicker -- 80.9 % of the excursions below the running
    maximum last exactly one frame, 94.9 % last five or fewer -- because a Cookie being
    regrasped or nudged inside the box leaves the containment region for a frame and comes
    straight back.

    Taking the raw difference would therefore produce **222 negative rewards against 294
    positive ones over 12 episodes**, i.e. 43 % of the non-zero signal would be noise
    telling the value function that a good state is bad.  `maximum.accumulate` removes it:
    73 positive, **0 negative**.

    The cost is stated rather than hidden: the running maximum ends at the episode's *peak*,
    not its final count, so on the 3 of 12 episodes that lost a Cookie for good it
    overestimates by 7 Cookies in total (166 against 159 over 12 episodes, 4.4 %).  Real
    losses are rare and small next to the flicker they would otherwise be buried in, and
    `value_target` is therefore also the peak so that the reward and the target agree --
    a value head trained on the final count while the reward spoke about peaks would be
    fitting two different functions.
    """

    counts = [int(r["after"]["cookies_in_target"]) for r in trace]
    running = np.maximum.accumulate(np.asarray(counts, dtype=np.int16)) if counts else np.zeros(0, np.int16)

    # The reward is the change in a *layered* potential, not the change in the count.  See
    # `layered_reward.py` for the measurements that motivate it: the count alone cannot separate
    # 31.2 % of episodes that put all 20 Cookies in the box from the 19.5 % that also seat them
    # in valid slots, and that gap (`in_box - slotted`, mean +1.04, non-zero in 62.5 %) is the
    # graded signal the old reward discarded.  `counts`/`running` are still recorded verbatim so
    # the two definitions can be compared on the same rows rather than across runs.
    layered = episode_rewards(trace, WEIGHTS)
    rewards = layered["rewards"]
    # The value target is the potential the episode actually reached, on the same scale as the
    # reward -- a value head trained on a count while the reward spoke about potentials would be
    # fitting two different functions, which is the mistake the old comment warned about.
    value_target = float(layered["potential"][-1]) if layered["potential"] else 0.0
    return {
        # The full path, not `part.name`: two collection roots both use `part_00..part_N`,
        # so a bare name collides between the rollout set and the expert set -- and the trace
        # lookup then reads the wrong root's `traces/seed_N/trace.jsonl`, silently pairing a
        # rollout's counts with an expert's episode that happens to share a seed.  Measured on
        # the first 400-episode run: 12 `(seed, episode_index)` pairs collided this way.
        #
        # `"{root}/{part}"` rather than `str(part)` keeps the label short while staying
        # unique; the full path is what `load_trace` needs, so it is stored separately.
        "part": f"{part.parent.name}/{part.name}",
        "part_path": str(part),
        "episode_index": int(label["episode_index"]),
        "seed": seed,
        "frames": len(counts),
        "success": bool(label["success"]),
        "value_target": value_target,
        "final_count": counts[-1] if counts else 0,
        "peak_count": int(running[-1]) if counts else 0,
        # The count-only target, kept beside the layered one so the two can be compared without
        # re-deriving it.  It is what every earlier run was trained against.
        "peak_count_target": float(running[-1]) if counts else 0.0,
        "required_cookies": int(label.get("required_cookies", 20)),
        "counts": counts,
        "running_max": [int(v) for v in running],
        "rewards": rewards,
        "potential": layered["potential"],
        "slotted": layered["slotted"],
        "walls": layered["walls"],
        "source_kept": layered["source"],
        "instruction": label.get("instruction", ""),
        "label_cookies_in_target": int(label.get("cookies_in_target", -1)),
        "label_success": bool(label.get("success", False)),
    }


def check(rows: list[dict[str, Any]]) -> list[str]:
    """The four invariants the plan lists, as error strings (empty means healthy)."""

    problems: list[str] = []
    for row in rows:
        where = f"{row['part']}/seed{row['seed']}"
        if row["frames"] != row["label_steps"]:
            problems.append(
                f"{where}: trace has {row['frames']} frames but the label claims "
                f"{row['label_steps']} -- the dataset and the trace would drift"
            )
        # The nesting I first assumed -- `slotted <= in_box` -- is **false**, and measurably so:
        # 355 of 400 episodes violate it.  `_target_slot_occupancy` tests the Cookie's *centre*
        # against `TARGET_SLOT_TOLERANCE`, while `cookies_in_target` requires the Cookie's whole
        # oriented bounding box to stay inside the target's inner volume.  A Cookie lying tilted
        # can therefore be within 10 mm x 2.5 mm of a slot centre while a corner pokes out over
        # the wall, giving `slotted = in_box + 1`.  That is a real state, not a bug in the trace.
        #
        # What does hold is the slot table's size, which is what the assignment loop can occupy.
        if row["counts"]:
            over = [i for i, s in enumerate(row["slotted"])
                    if s > row["required_cookies"]]
            if over:
                problems.append(
                    f"{where}: slotted > {row['required_cookies']} at {len(over)} frame(s), "
                    f"first {over[0]}"
                )
            # Both layers describe the same 20 Cookies, so a count larger than the episode's
            # Cookie budget means the schema changed under us.
            bad_count = [i for i, c in enumerate(row["counts"])
                         if c < 0 or c > row["required_cookies"]]
            if bad_count:
                problems.append(
                    f"{where}: in_box out of [0, {row['required_cookies']}] at "
                    f"{len(bad_count)} frame(s), first {bad_count[0]}"
                )
        # The count-only target was the peak of the running maximum, so it must be at least the
        # final count and at most the required number of Cookies.
        if row["counts"]:
            if row["peak_count_target"] < row["final_count"]:
                problems.append(
                    f"{where}: peak {row['peak_count_target']} < final count "
                    f"{row['final_count']} -- the accumulator ran backwards"
                )
            if row["peak_count_target"] > row["required_cookies"]:
                problems.append(
                    f"{where}: peak {row['peak_count_target']} exceeds required "
                    f"{row['required_cookies']}"
                )
        # Negative rewards must be impossible once the running maximum is used: a Cookie
        # that flickers out for a frame is not a setback, and telling the value function
        # otherwise was worth 222 spurious negatives in the first 12 episodes.
        negatives = [(i, x) for i, x in enumerate(row["rewards"]) if x < 0]
        if negatives:
            problems.append(f"{where}: {len(negatives)} negative reward(s) at {negatives[:3]}")
        if row["frames"] == 0:
            problems.append(f"{where}: zero frames")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", required=True, help="part_* directories")
    parser.add_argument("--output", required=True, help="output parquet path")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "the merged dataset these parts were assembled into.  Optional, but when given the "
            "label rows are checked against its episode frame counts -- the only way to catch a "
            "silently shuffled positional mapping."
        ),
    )
    parser.add_argument("--check", action="store_true", help="only validate, do not write")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any invariant violation (default: report and continue)",
    )
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for part_arg in args.parts:
        part = Path(part_arg)
        if not part.is_dir():
            missing.append(f"{part}: not a directory")
            continue
        labels = load_labels(part)
        if not labels:
            missing.append(f"{part}: no episode_labels.jsonl")
            continue
        # File order, not `sorted(...)`: see `load_labels` for the 98-misaligned-episodes bug that
        # sorting caused.
        for label in labels:
            seed = int(label["seed"])
            trace = load_trace(part, seed)
            row = build_row(part, seed, label, trace)
            row["label_steps"] = int(label.get("steps", -1))
            rows.append(row)

    if missing:
        for message in missing:
            print(f"[labels] {message}")
        if not rows:
            return 1

    # The alignment check the whole positional mapping depends on.  The episodes table carries no
    # seed, so `train_value.py` pairs label row `i` with episode `i`; if that pairing is wrong the
    # head learns a shuffled target and simply reports a worse number, with nothing failing.  An
    # episode's frame count is a fingerprint of it, so compare them when the dataset is reachable.
    episodes_path = (
        Path(args.dataset) / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if args.dataset
        else None
    )
    if episodes_path is not None and episodes_path.is_file():
        import pyarrow.parquet as pq

        lengths = np.asarray(pq.read_table(episodes_path).column("length").to_pylist())
        if len(lengths) == len(rows):
            agree = sum(1 for row, length in zip(rows, lengths) if row["frames"] == int(length))
            if agree != len(rows):
                print(
                    f"[labels] ALIGNMENT BROKEN: {agree}/{len(rows)} label rows match their "
                    f"episode's frame count.  `--parts` must be given in the same order the "
                    f"dataset was merged in, and each part's `episode_labels.jsonl` must be in "
                    f"episode order."
                )
                return 1
            print(f"[labels] alignment verified against {episodes_path.name}: {agree}/{len(rows)}")
        else:
            print(
                f"[labels] {len(rows)} label rows vs {len(lengths)} dataset episodes; skipping the "
                "alignment check (they describe different sets)"
            )

    total_frames = sum(r["frames"] for r in rows)
    successes = sum(1 for r in rows if r["success"])
    print(f"[labels] {len(rows)} episodes from {len(set(r['part'] for r in rows))} parts")
    print(f"[labels] {total_frames} frames, {successes} strict successes "
          f"({100.0 * successes / max(len(rows), 1):.1f} %)")
    by_target: dict[int, int] = {}
    for row in rows:
        by_target[row["value_target"]] = by_target.get(row["value_target"], 0) + 1
    print(f"[labels] value_target (peak) distribution: {dict(sorted(by_target.items()))}")
    losses = [(r["seed"], r["peak_count"], r["final_count"])
              for r in rows if r["final_count"] < r["peak_count"]]
    if losses:
        print(f"[labels] {len(losses)} episode(s) ended below their peak "
              f"(the running-max reward overestimates those): {losses[:5]}")
    print(f"[labels] reward: non-zero frames "
          f"{sum(1 for r in rows for x in r['rewards'] if x)}, "
          f"positive {sum(1 for r in rows for x in r['rewards'] if x > 0)}, "
          f"negative {sum(1 for r in rows for x in r['rewards'] if x < 0)}")

    problems = check(rows)
    if problems:
        print(f"[labels] {len(problems)} invariant violation(s):")
        for problem in problems[:20]:
            print(f"    {problem}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        if args.strict:
            return 1
    else:
        print("[labels] all invariants hold")

    if args.check:
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            key: [row[key] for row in rows]
            for key in ("part", "episode_index", "seed", "frames", "success",
                        "value_target", "final_count", "peak_count", "peak_count_target",
                        "required_cookies", "instruction")
        }
        | {
            "counts": [np.asarray(row["counts"], dtype=np.int16) for row in rows],
            "running_max": [np.asarray(row["running_max"], dtype=np.int16) for row in rows],
            # float32, not int16: the layered potential is on a 0.5-granularity scale (a seated
            # Cookie is worth 1.5, a dropped one 1.0), and casting to int would silently round
            # every seating bonus away -- the reward would look like the old count-only one
            # while claiming to be layered.
            "rewards": [np.asarray(row["rewards"], dtype=np.float32) for row in rows],
            "potential": [np.asarray(row["potential"], dtype=np.float32) for row in rows],
            "slotted": [np.asarray(row["slotted"], dtype=np.int16) for row in rows],
            "walls": [np.asarray(row["walls"], dtype=np.int16) for row in rows],
            "source_kept": [np.asarray(row["source_kept"], dtype=np.int16) for row in rows],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)
    print(f"[labels] wrote {output} ({output.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
