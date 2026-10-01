"""Report a sharded collection: success rate per column, colour balance, seed hygiene.

The randomization docs are explicit that a collection is only described
honestly by per-column successes over per-column *attempts* -- failed attempts
stay in the denominator -- and that colour balance is not guaranteed by
construction and has to be inspected from the saved sidecars.  Both checks live
here because both are easy to skip and expensive to discover after training.

The duplicate check is the one that matters most for a sharded run.  Randomised
scenes are a pure function of the seed, so two shards that touched the same seed
produced byte-identical episodes: the dataset would look big while containing
less.  `collect_sharded.sh` prevents this by construction, and this script
verifies it actually held rather than trusting the argument.

A short per-column attempt count is reported as a warning rather than a failure
because the docs recommend at least 25 attempts per column before a run is
considered measured at all.

    python scripts/shard_report.py --out-root datasets/codex_advanced_calib
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

#: The docs' minimum for treating a per-column success rate as measured.
MIN_ATTEMPTS_PER_COLUMN = 25


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


#: Failure messages embed measured values -- "position error 2.43 mm", a cookie
#: index list, a force reading -- so grouping them verbatim yields one line per
#: failure and hides which failure actually dominates. Each pattern collapses a
#: family onto a stable label.
_FAILURE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "target column N" here is a placement column *inside the target box*, not a
    # source column: the expert rejects randomised layouts whose second slot it
    # cannot reach with a vertical grasp. The cross-tab prints source columns, so
    # the label spells this out to keep the two from being read as one thing.
    (re.compile(r"column (\d+) unreachable with vertical grasp"), "target-box slot {0} unreachable (layout rejected)"),
    (re.compile(r"only \d+/10 cookies in target box"), "fewer than 10 cookies placed in two batches"),
    (re.compile(r"Cookie slipped out of batch during transport"), "a cookie slipped out of the batch in transport"),
    (re.compile(r"batch phase timeout"), "batch phase timeout"),
    (re.compile(r"released Cookies not upright"), "a released cookie was not upright or contained"),
    (re.compile(r"Cookie dropped during lift"), "a cookie was dropped during lift"),
    (re.compile(r"insertion blocked"), "insertion blocked"),
    (re.compile(r"environment_safety_terminated"), "environment safety stop"),
    (re.compile(r"max_steps_exceeded"), "max steps exceeded"),
)


def summarise_failure(message: str) -> str:
    """Collapse a free-text failure into a stable category."""
    for pattern, label in _FAILURE_PATTERNS:
        match = pattern.search(message)
        if match:
            return label.format(*match.groups()) if match.groups() else label
    return message[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--min-attempts",
        type=int,
        default=MIN_ATTEMPTS_PER_COLUMN,
        help=f"Warn below this many attempts per column (default {MIN_ATTEMPTS_PER_COLUMN})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.out_root.expanduser().resolve()
    shard_dirs = sorted(p for p in root.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        print(f"no shard_* directories under {root}")
        return 1

    all_attempts: list[dict] = []
    saved: list[tuple[str, dict]] = []
    per_shard: list[tuple[str, int, int]] = []

    for shard in shard_dirs:
        attempts = read_jsonl(shard / "attempts.jsonl")
        episodes = read_jsonl(shard / "a3_episode_metadata.jsonl")
        for row in attempts:
            all_attempts.append({**row, "_shard": shard.name})
        for row in episodes:
            saved.append((shard.name, row))
        per_shard.append((shard.name, sum(bool(r.get("success")) for r in attempts), len(attempts)))

    successes = sum(bool(r.get("success")) for r in all_attempts)
    print(f"run root       : {root}")
    print(f"shards         : {len(shard_dirs)}")
    print(f"attempts       : {len(all_attempts)}")
    print(f"saved episodes : {len(saved)}")
    rate = successes / len(all_attempts) * 100 if all_attempts else 0.0
    print(f"success rate   : {successes}/{len(all_attempts)} = {rate:.1f}%")
    print()

    print("per shard (saved/attempts):")
    for name, ok, total in per_shard:
        bar = "#" * round(20 * ok / total) if total else ""
        print(f"  {name}  {ok:>4}/{total:<4}  {bar}")
    print()

    # Attempts per column, not just successes: a column that fails often still
    # has to be counted, otherwise the hard columns look as good as the easy ones.
    attempts_by_column: Counter = Counter()
    successes_by_column: Counter = Counter()
    for row in all_attempts:
        column = row.get("source_column")
        attempts_by_column[column] += 1
        if row.get("success"):
            successes_by_column[column] += 1

    print("per column (successes/attempts):")
    shortfall = False
    for column in sorted(attempts_by_column, key=lambda c: (c is None, c)):
        total = attempts_by_column[column]
        ok = successes_by_column[column]
        flag = ""
        if total < args.min_attempts:
            flag = f"  <-- fewer than {args.min_attempts} attempts"
            shortfall = True
        print(f"  column {column}: {ok:>4}/{total:<4} = {ok / total * 100:5.1f}%{flag}")
    print()

    reasons = Counter(
        summarise_failure(str(row.get("failure_reason") or row.get("phase") or "unknown"))
        for row in all_attempts
        if not row.get("success")
    )
    if reasons:
        failures = sum(reasons.values())
        print(f"failure categories ({failures} failed attempts):")
        columns = sorted(attempts_by_column, key=lambda c: (c is None, c))
        for reason, count in reasons.most_common():
            per_column = Counter(
                row.get("source_column")
                for row in all_attempts
                if not row.get("success")
                and summarise_failure(str(row.get("failure_reason") or row.get("phase") or "unknown"))
                == reason
            )
            spread = " ".join(f"c{c}={per_column.get(c, 0)}" for c in columns)
            print(f"  {count:>5}  {count / failures * 100:5.1f}%  {reason:<44} {spread}")
        print("  (cN = source column N; a failure concentrated in one column points at" \
              " the expert rather than the harness)")
        print()

    colours = Counter((row.get("randomization") or {}).get("cookie_color") for _, row in saved)
    colours.pop(None, None)
    if colours:
        print("saved-episode colour balance (appearance palette):")
        total = sum(colours.values())
        for colour, count in colours.most_common():
            print(f"  {colour:<10} {count:>5}  {count / total * 100:5.1f}%")
        print()

    seed_counts = Counter(row.get("seed") for _, row in saved)
    duplicated = {seed: n for seed, n in seed_counts.items() if n > 1}
    if duplicated:
        print(f"DUPLICATE SEEDS in saved episodes: {len(duplicated)}")
        for seed, count in sorted(duplicated.items())[:20]:
            print(f"  seed {seed} saved {count} times")
        print("  These episodes are identical; the shard seed blocks overlapped.")
    else:
        print(f"seed hygiene   : no duplicate seeds across {len(saved)} saved episodes")

    all_seeds = [row.get("seed") for row in all_attempts]
    if len(set(all_seeds)) != len(all_seeds):
        print(f"note: {len(all_seeds) - len(set(all_seeds))} duplicate seeds among all attempts "
              "(expected only if shards overlapped)")
    print()

    if shortfall:
        print(f"WARNING: some columns have fewer than {args.min_attempts} attempts; "
              "the per-column rates above are not yet measured, only indicative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
