#!/usr/bin/env python3
"""Build one training set that carries two tasks and two instructions.

The brief is a single checkpoint that scores on main's **2x5** and **2x10**
benchmarks.  There are two obstacles, and neither is about frames:

1. **Instruction mismatch.**  A pi0.5 fine-tune answers the sentence it was trained
   on and does nothing for any other -- measured repeatedly on this line (main's
   harness prompt makes these checkpoints move no Cookie at all).  main's benchmarks
   send their *own* strings, and they are not the strings in our datasets:

       2x5  "Transfer 10 cookies from source column 1 into the target box in two
             batches of five."                      (main, prompt_style="task")
       2x5  "grasp five yellow blocks together ... fill all ten slots"
                                                    (ours, in the 2x5 dataset)
       2x10 the plan prompt, identical in main and in the collected 2x10 data

   The 2x5 string is not a coincidence of one seed: `choose_column("1", seed)`
   returns 1 for every seed, so main's 2x5 instruction is a single fixed sentence.
   This script therefore **relabels** the 2x5 episodes to that sentence.  No frame
   changes -- pi0.5 is not history-conditioned (`n_obs_steps = 1`), so the label is
   the whole of the conditioning, and relabelling it is exactly the correction.

2. **`task_index` is per-dataset.**  Each source numbers its own tasks from zero.
   `aggregate_datasets` unions the task tables but copies the data rows verbatim,
   so merging two datasets whose index 0 means different sentences silently
   retrains one task on the other's instruction.  This script rewrites every source
   to one global table *before* aggregating, then verifies by resolving each row's
   index back to a string.

    python scripts/build_multitask_dataset.py \\
        --out /root/autodl-tmp/datasets/a3_multitask_v1 \\
        --repo-id local/a3-multitask-v1 \\
        --two-by-five /root/autodl-tmp/datasets/a3_graft_v1_mix \\
        --two-by-ten '/root/autodl-tmp/datasets/a3_2x10_v1/shard_*'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

#: main's 2x5 instruction, verbatim (`EpisodeExecution._episode_task`, column 1 because
#: `choose_column("1", seed)` is constant).  Relabelling to anything else keeps the
#: acceptance harness from ever matching a sentence the model knows.
TWO_BY_FIVE_PROMPT = (
    "Transfer 10 cookies from source column 1 into the target box in two batches of five."
)

#: main's `prompt_style="short"` pair: one constant sentence per task, no plan and no
#: column.  The 2x10 plan prompt has eleven variants (n=1..9, a skip variant, and the short
#: one), picked per episode by `first_grasp: random`, and covering all eleven to a useful
#: standard is not realistic: measured on the collected shards, n=5 has 30 episodes and
#: n=0 has 21, so a 15000-step arm would see a handful of each.  Two constant sentences
#: make the instruction a task switch rather than a plan the model has to reproduce.
SHORT_PROMPTS = {
    "2x5": "transfer 10 cookies into target box",
    "2x10": "transfer 20 cookies into target box",
}


def episode_tasks(root: Path, *, plan_prompts: bool = False) -> dict[int, str]:
    """``episode_index -> task string`` for one dataset, from its own metadata.

    With ``plan_prompts`` the 2x10 plan sentence is read from
    ``randomization.task_instruction`` instead of the ``task`` column.  That distinction
    matters because a previous build **rewrote the source's own ``task`` column in
    place** to a single short sentence, so the plan that each episode was actually
    collected under survives only in the metadata.  Nothing else can recover it: the
    plan is ``[n, 10-n, n, 10-n]`` and ``n`` was drawn per episode from the seed, so the
    sentence is a property of the episode rather than of the task.
    """

    path = root / "a3_episode_metadata.jsonl"
    if not path.is_file():
        raise SystemExit(f"{root} has no a3_episode_metadata.jsonl; cannot know its tasks")
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        text = str(record["task"])
        if plan_prompts:
            recorded = (record.get("randomization") or {}).get("task_instruction")
            if recorded:
                text = str(recorded)
        out[int(record["episode_index"])] = text
    return out


def data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"{root} has no data parquet files")
    return files


def rewrite_tasks(root: Path, global_tasks: list[str], by_episode: dict[int, int]) -> int:
    """Point one dataset at ``global_tasks`` and relabel its rows.

    ``by_episode`` maps the dataset's own episode index to an index in
    ``global_tasks``.  Both the task table and the ``task_index`` column are written,
    because a reader uses the column and a trainer uses the table -- they have to
    agree, and only rewriting both makes that true.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # The rewrite is in place, so the source's own numbering is preserved the first time
    # it is touched.  Without this, a relabel would destroy the only record of which
    # instruction an episode was actually collected under; with it, the change is
    # reversible by restoring one file.
    original = root / "meta" / "tasks.parquet"
    backup = root / "meta" / "tasks.original.parquet"
    if original.is_file() and not backup.is_file():
        backup.write_bytes(original.read_bytes())

    table = pd.DataFrame(
        {"task_index": range(len(global_tasks))},
        index=pd.Index(global_tasks, name="task"),
    )
    table.to_parquet(original)

    rewritten = 0
    for path in data_files(root):
        data = pq.read_table(path)
        episodes = np.asarray(data["episode_index"].to_numpy())
        missing = sorted(set(int(e) for e in episodes) - set(by_episode))
        if missing:
            raise SystemExit(f"{root.name}/{path.name}: {len(missing)} episodes have no task")
        column = np.array([by_episode[int(e)] for e in episodes], dtype=np.int64)
        index = data.schema.get_field_index("task_index")
        data = data.set_column(
            index, pa.field("task_index", pa.int64()), pa.array(column, type=pa.int64())
        )
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(data, temporary)
        temporary.replace(path)
        rewritten += 1
    return rewritten


def verify(root: Path, expected: dict[int, str], name: str) -> None:
    """Read the rewritten rows back and resolve them to strings.

    The check is deliberately done through the *column*, not through the plan, because
    the failure mode being guarded against is precisely that the two disagree.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    lookup = list(tasks.index)
    total = mismatched = 0
    for path in data_files(root):
        data = pq.read_table(path, columns=["episode_index", "task_index"])
        episodes = np.asarray(data["episode_index"].to_numpy())
        indices = np.asarray(data["task_index"].to_numpy())
        for episode, index in zip(episodes, indices):
            total += 1
            if lookup[int(index)] != expected[int(episode)]:
                mismatched += 1
    if mismatched:
        raise SystemExit(f"{name}: {mismatched}/{total} rows resolve to the wrong instruction")
    print(f"  {name}: {total} rows, {len(lookup)} instruction(s), 0 mismatched", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--two-by-five",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more 2x5 datasets.  More than one is normal: the canonical set carries "
            "the first-batch behaviour and the graft collection carries the second-batch "
            "recovery, and both are 2x5.  Their episodes are relabelled to main's sentence."
        ),
    )
    parser.add_argument(
        "--two-by-ten",
        nargs="+",
        default=None,
        help=(
            "One or more globs for the 20-cookie shards, e.g. "
            "'/path/a3_2x10_v1/shard_*' '/path/a3_2x10_n5/shard_*'.  More than one is normal "
            "when the 2x10 data comes from two collections, e.g. one that drew the grasp "
            "plan per episode and a later one that forced a single plan."
        ),
    )
    parser.add_argument(
        "--two-by-ten-repeat",
        type=int,
        default=1,
        help=(
            "How many times each 2x10 shard is concatenated.  This is the sampling weight, "
            "and it is the cheap knob: a 15000-step arm sees 90000 of the dataset's 853113 "
            "frames -- 10.5%% -- so collecting more data cannot fix a composition problem.  "
            "At repeat=1 the 2x10 frames are 33.7%% of the set, so two thirds of every batch "
            "teaches the 2x5 behaviour, including its 'stop after two batches'; at repeat=2 "
            "they are 50.4%%.  Duplicating is not free of consequence -- the same 280 "
            "episodes are revisited rather than new ones added -- but the run never "
            "completes one pass of the set, so this trades variety the model cannot reach "
            "anyway for the balance it can actually learn from."
        ),
    )
    parser.add_argument(
        "--two-by-five-prompt",
        default=TWO_BY_FIVE_PROMPT,
        help="Instruction to stamp on the 2x5 episodes; defaults to main's own.",
    )
    parser.add_argument(
        "--keep-two-by-five-prompt",
        action="store_true",
        help="Do not relabel the 2x5 episodes (use the sentence already in the data).",
    )
    parser.add_argument(
        "--plan-prompts",
        action="store_true",
        help=(
            "For 2x10, take each episode's instruction from the plan recorded in its own "
            "metadata (`randomization.task_instruction`) instead of its `task` column.  The "
            "plan sentence names the batch schedule -- 'Grasp 5 cookies first, then 5 to "
            "fill target column 1 with 10. Repeat: grasp 5, then 5 to fill target column 2 "
            "with 10.' -- so it tells the model how many batches remain, which the short "
            "sentence does not.  This is the sentence main's `--prompt-style task` produces "
            "for the same seed, so training and evaluation agree by construction."
        ),
    )
    parser.add_argument(
        "--short-prompts",
        action="store_true",
        help=(
            "Relabel every episode to main's `prompt_style=\"short\"` sentence for its task: "
            "'transfer 10 cookies into target box' for 2x5 and 'transfer 20 cookies into "
            "target box' for 2x10.  Two instructions in total, constant across seeds.  The "
            "evaluation must then use --prompt-style short as well, or it asks a sentence "
            "the model never saw.  This replaces the per-seed plan prompt, which has eleven "
            "variants for 2x10 and cannot be covered usefully at this data size."
        ),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help=(
            "Where to hard-link the sources before rewriting their task columns.  "
            "Defaults to <first source's parent>/<out name>_stage, which is on the same "
            "filesystem as the sources by construction: hard links cannot cross one, and "
            "a stage on /tmp fails with 'Invalid cross-device link'.  Every source is "
            "linked (free) and the rewrite replaces files by rename, so the original "
            "datasets are never modified."
        ),
    )
    args = parser.parse_args()

    # The aggregator re-encodes video through a temporary file in TMPDIR, and on this site
    # `/tmp` lives on the 30 GB system disk that is already ~85% full: the merge dies with
    # 'No space left on device' part-way through.  Both spellings are set because the
    # encoders resolve the directory differently -- `tempfile` reads the module attribute,
    # and the C libraries behind PyAV read the environment variable.
    scratch = args.out.parent / ".tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)
    print(f"scratch directory: {scratch}")

    sources: list[tuple[str, str, Path]] = []
    for path in args.two_by_five:
        sources.append(("2x5", path.name, path))
    if args.two_by_ten:
        for pattern in args.two_by_ten:
            # The glob is expanded per repeat so a re-run cannot see a different file set
            # between the copies, which would silently change the sampling weight.
            matched = sorted(glob.glob(pattern))
            if not matched:
                raise SystemExit(f"--two-by-ten matched nothing: {pattern!r}")
            # Two globs routinely match the same leaf name -- every shard of every
            # collection is called `shard_NN` -- and the staging directory is keyed by
            # name, so the pattern's parent has to be part of the key.
            origin = Path(pattern).parent.name or "root"
            for repeat in range(max(1, args.two_by_ten_repeat)):
                for path in matched:
                    suffix = "" if repeat == 0 else f"r{repeat}"
                    sources.append(("2x10", f"{origin}_{Path(path).name}{suffix}", Path(path)))
    print(f"{len(sources)} source dataset(s), 2x10 repeat={max(1, args.two_by_ten_repeat)}")

    # Hard-link the sources so the task rewrite (which replaces files by rename) never
    # touches the datasets it read.  Videos are the bulk of every source and are linked,
    # not copied; only the parquet files are rewritten and charged against the disk.
    stage = args.stage_dir or Path(sources[0][2]).parent / f"{args.out.name}_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    staged: list[tuple[str, str, Path]] = []
    for kind, name, root in sources:
        target = stage / f"{kind}_{name}"
        shutil.copytree(root, target, copy_function=os.link)
        staged.append((kind, name, target))
    sources = staged
    print(f"staged {len(sources)} dataset(s) with hard links under {stage}")

    # The global table is the union in first-appearance order, so a run is reproducible
    # from the same inputs in the same order.
    global_tasks: list[str] = []

    def intern(text: str) -> int:
        if text not in global_tasks:
            global_tasks.append(text)
        return global_tasks.index(text)

    plans: list[tuple[str, Path, dict[int, str]]] = []
    for kind, name, root in sources:
        local = episode_tasks(root, plan_prompts=(args.plan_prompts and kind == "2x10"))
        if args.short_prompts:
            target = SHORT_PROMPTS[kind]
            expected = {episode: target for episode in local}
            note = f" (relabelled to the short prompt {target!r})"
        elif kind == "2x5" and not args.keep_two_by_five_prompt:
            expected = {episode: args.two_by_five_prompt for episode in local}
            note = " (relabelled to main's 2x5 string)"
        else:
            expected = dict(local)
            note = ""
        plans.append((name, root, expected))
        print(
            f"  {kind}/{name}: {len(local)} episodes, "
            f"{len(set(expected.values()))} distinct instruction(s){note}"
        )

    # The table is complete once every source has been interned, so the rewrite happens
    # once per source against the final indices.  Writing during the interning loop would
    # rewrite the multi-gigabyte parquet files twice.
    for name, root, expected in plans:
        rewrite_tasks(
            root,
            global_tasks,
            {episode: intern(text) for episode, text in expected.items()},
        )
        verify(root, expected, name)

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; remove it first")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"aggregating {len(plans)} dataset(s) -> {args.out}")

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[args.repo_id] * len(plans),
        aggr_repo_id=args.repo_id,
        roots=[root for _name, root, _expected in plans],
        aggr_root=args.out,
        data_files_size_in_mb=500.0,
        video_files_size_in_mb=500.0,
    )

    info = json.loads((args.out / "meta" / "info.json").read_text())
    print(
        f"\nwrote {args.out}\n"
        f"  episodes {info['total_episodes']}, frames {info['total_frames']}\n"
        f"  instructions {len(global_tasks)}:"
    )
    for text in global_tasks:
        print(f"    - {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
