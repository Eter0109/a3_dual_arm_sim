#!/usr/bin/env bash
# Collect cookie episodes with several independent processes at once.
#
# A single collection process uses about one core: the cost is almost entirely
# `mj_step` (measured 97%), and camera rendering -- the only part that has to
# share the GPU -- is about 3%.  So the work shards cleanly and N processes give
# close to N times the throughput.  Measured per-process slowdown was 9% at
# N=16, i.e. about 14.7x aggregate.
#
# Each shard writes its own LeRobot dataset because the writer needs a private
# directory.  `scripts/merge_codex_shards.py` concatenates them afterwards.
#
# WHY DISJOINT SEEDS MATTER
#   Two shards that ran the same seed would produce identical episodes: the
#   scene, the randomisation and the appearance are all pure functions of the
#   seed.  Duplicates add nothing and silently shrink the effective dataset.
#   The collector checks `len(attempts) >= max_attempts` *before* running a
#   seed, so a shard starting at S and capped at B attempts can only ever touch
#   seeds S .. S+B-1.  Setting the seed block size equal to the attempt budget
#   therefore makes overlap impossible by construction.
#
# WHY BLOCKS ARE MULTIPLES OF FOUR
#   `choose_column("random", seed)` is `divmod(seed, 4)` into a per-group
#   permutation of the four columns, so every four consecutive seeds cover all
#   four columns exactly once.  Block sizes that are multiples of four make each
#   shard -- and therefore the whole run -- exactly balanced across columns.
#
# Usage:
#   scripts/collect_sharded.sh --shards 16 --attempts-per-shard 8 \
#       --out-root datasets/codex_advanced_calib --seed-base 10000
#
# Resume a stopped run (per shard, skipping ones already finished):
#   scripts/collect_sharded.sh ... --resume

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${A3_PYTHON:-/root/autodl-tmp/a3-collect/.venv/bin/python}"

SHARDS=16
ATTEMPTS_PER_SHARD=8
EPISODES_PER_SHARD=0        # 0 => use ATTEMPTS_PER_SHARD (attempt cap binds)
SEED_BASE=10000
OUT_ROOT=""
REPO_PREFIX=""
MAX_CONSECUTIVE_FAILURES=25
RESUME=0
DRY_RUN=0

die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards)                  SHARDS="$2"; shift 2 ;;
    --attempts-per-shard)      ATTEMPTS_PER_SHARD="$2"; shift 2 ;;
    --episodes-per-shard)      EPISODES_PER_SHARD="$2"; shift 2 ;;
    --seed-base)               SEED_BASE="$2"; shift 2 ;;
    --out-root)                OUT_ROOT="$2"; shift 2 ;;
    --repo-prefix)             REPO_PREFIX="$2"; shift 2 ;;
    --max-consecutive-failures) MAX_CONSECUTIVE_FAILURES="$2"; shift 2 ;;
    --resume)                  RESUME=1; shift ;;
    --dry-run)                 DRY_RUN=1; shift ;;
    -h|--help)                 sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$OUT_ROOT" ]]   || die "--out-root is required"
[[ -n "$REPO_PREFIX" ]] || die "--repo-prefix is required (must be unique per out-root)"
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || die "--shards must be a positive integer"
[[ "$ATTEMPTS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] || die "--attempts-per-shard must be a positive integer"
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || die "--seed-base must be a non-negative integer"
(( ATTEMPTS_PER_SHARD % 4 == 0 )) || \
  echo "warning: attempts-per-shard=$ATTEMPTS_PER_SHARD is not a multiple of 4; column counts will be uneven" >&2
(( EPISODES_PER_SHARD == 0 )) && EPISODES_PER_SHARD="$ATTEMPTS_PER_SHARD"

OUT_ROOT="$(cd "$REPO_ROOT" && mkdir -p "$OUT_ROOT" && cd "$OUT_ROOT" && pwd)"
LOG_DIR="$OUT_ROOT/logs"

total_attempts=$(( SHARDS * ATTEMPTS_PER_SHARD ))
seed_end=$(( SEED_BASE + total_attempts - 1 ))

echo "sharded collection"
echo "  shards              : $SHARDS"
echo "  attempts per shard  : $ATTEMPTS_PER_SHARD"
echo "  successes per shard : $EPISODES_PER_SHARD (cap; attempts are the real budget)"
echo "  seed range          : $SEED_BASE .. $seed_end  (disjoint by construction)"
echo "  attempts per column : $(( total_attempts / 4 )) if all seeds are used"
echo "  output              : $OUT_ROOT"
echo "  python              : $PYTHON"
echo

if (( DRY_RUN )); then
  for (( i = 0; i < SHARDS; i++ )); do
    printf '  shard %02d  seed %d..%d  %s\n' \
      "$i" "$(( SEED_BASE + i * ATTEMPTS_PER_SHARD ))" \
      "$(( SEED_BASE + (i + 1) * ATTEMPTS_PER_SHARD - 1 ))" \
      "$OUT_ROOT/shard_$(printf '%02d' "$i")"
  done
  exit 0
fi

[[ -x "$PYTHON" ]] || die "python not found: $PYTHON"
mkdir -p "$LOG_DIR"

export MUJOCO_GL=egl
export PYTHONPATH="$REPO_ROOT/src"

# Refuse to start unless every shard directory is either absent or a resumable
# dataset.  Catching this up front beats discovering it after two hours.
for (( i = 0; i < SHARDS; i++ )); do
  shard_root="$OUT_ROOT/shard_$(printf '%02d' "$i")"
  if [[ -d "$shard_root" ]] && [[ -n "$(ls -A "$shard_root" 2>/dev/null)" ]]; then
    if (( ! RESUME )); then
      die "shard dir is not empty: $shard_root (pass --resume to continue it)"
    fi
    [[ -f "$shard_root/collection_summary.json" ]] || \
      die "cannot resume $shard_root: no collection_summary.json"
  fi
done

pids=()
for (( i = 0; i < SHARDS; i++ )); do
  idx="$(printf '%02d' "$i")"
  shard_root="$OUT_ROOT/shard_$idx"
  seed_start=$(( SEED_BASE + i * ATTEMPTS_PER_SHARD ))
  log="$LOG_DIR/shard_$idx.log"
  resume_flag=()
  (( RESUME )) && resume_flag=(--resume)

  "$PYTHON" "$REPO_ROOT/examples/collect_cookie_benchmark.py" \
      --root "$shard_root" \
      --repo-id "$REPO_PREFIX-shard-$idx" \
      --episodes "$EPISODES_PER_SHARD" \
      --max-attempts "$ATTEMPTS_PER_SHARD" \
      --seed-start "$seed_start" \
      "${resume_flag[@]}" \
      > "$log" 2>&1 &
  pids+=($!)
  echo "  started shard $idx  pid ${pids[-1]}  seeds $seed_start..$(( seed_start + ATTEMPTS_PER_SHARD - 1 ))  log $log"
done

echo
echo "waiting for $SHARDS shards. Tail a log for progress, e.g.:"
echo "  tail -f $LOG_DIR/shard_00.log"
echo

failed=0
for (( i = 0; i < SHARDS; i++ )); do
  idx="$(printf '%02d' "$i")"
  if wait "${pids[$i]}"; then
    echo "  shard $idx finished ok"
  else
    code=$?
    echo "  shard $idx FAILED (exit $code) -- see $LOG_DIR/shard_$idx.log" >&2
    failed=$(( failed + 1 ))
  fi
done

echo
"$PYTHON" "$REPO_ROOT/scripts/shard_report.py" --out-root "$OUT_ROOT" || true

echo
if (( failed > 0 )); then
  echo "$failed shard(s) failed; re-run with --resume after inspecting the logs" >&2
  exit 1
fi
echo "all $SHARDS shards finished. Merge with:"
echo "  $PYTHON scripts/merge_codex_shards.py --out-root $OUT_ROOT --repo-id <merged-repo-id>"
