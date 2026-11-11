#!/usr/bin/env bash
# Launch N shards of `branch_rollouts_new.py` and wait for all of them.
#
# Why shards: one replay is single-core bound (MuJoCo's solver has no useful thread
# parallelism at this model size), so N processes give close to N times the throughput
# until the 16 cores are saturated.  Measured single-process rate is 0.15-0.32 s per
# simulated step depending on how much contact the horizon contains, so the wall time is
# not predictable from the step count alone -- the script prints its own per-shard
# estimate before it starts.
#
# Usage:
#   bash scripts/recap/run_branch_shards.sh                 # the default full run
#   bash scripts/recap/run_branch_shards.sh 4               # only 4 shards
#   HORIZON=100 ACTIONS=4 bash scripts/recap/run_branch_shards.sh
#
# Everything after `--` is passed through to the Python script unchanged.

set -uo pipefail

SHARDS="${1:-8}"
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
DATASET=/root/autodl-tmp/datasets/a3_recap_all
OUTROOT="$DATASET/branches"
LOGDIR=/root/autodl-tmp/logs

HORIZON="${HORIZON:-80}"
ACTIONS="${ACTIONS:-4}"
SIGMAS="${SIGMAS:-0.0 0.10 0.25}"
# Episodes **per shard**, not in total.  The whole set is 400, so 8 x 4 = 32 distinct
# episodes and roughly 2.2 branch points each (measured: `--min-gap` and
# `--max-branch-step` cap points per episode well below `--per-episode`).  Setting this
# to the total by mistake makes each shard replay the whole set and turns a 1 h run into
# an 8 h one.
EPISODES="${EPISODES:-4}"
PER_EPISODE="${PER_EPISODE:-5}"
LEAD="${LEAD:-15}"
MAX_BRANCH_STEP="${MAX_BRANCH_STEP:-600}"

mkdir -p "$OUTROOT" "$LOGDIR"
TAG="run_$(date +%m%d_%H%M)"

# Two threads per shard on 16 cores: MuJoCo is single-core bound but the parquet load and
# the numpy reduction benefit from a second one, and 8 x 2 leaves nothing oversubscribed.
export OMP_NUM_THREADS=2 MUJOCO_GL=egl TMPDIR=/root/autodl-tmp/tmp

echo "=== $TAG: $SHARDS shards, $EPISODES episodes, $PER_EPISODE points/episode," \
     "horizon $HORIZON, lead $LEAD, actions $ACTIONS, sigmas [$SIGMAS] ==="

pids=()
for (( shard = 0; shard < SHARDS; shard++ )); do
    "$PY" -u "$PROJECT/scripts/recap/branch_rollouts_new.py" \
        --shard "$shard" --shards "$SHARDS" \
        --episodes "$EPISODES" --per-episode "$PER_EPISODE" \
        --horizon "$HORIZON" --lead "$LEAD" \
        --actions "$ACTIONS" --sigmas $SIGMAS \
        --max-branch-step "$MAX_BRANCH_STEP" \
        --progress-every 200 \
        --output "$OUTROOT/${TAG}_s$(printf '%02d' "$shard").npz" \
        > "$LOGDIR/${TAG}_s$(printf '%02d' "$shard").log" 2>&1 &
    pids+=($!)
    # Stagger the starts by a second so eight processes do not all page in the parquet at
    # once; on 90 GB this is not fatal, it just makes the first progress line readable.
    sleep 1
done

echo "launched: ${pids[*]}"
echo "logs:     $LOGDIR/${TAG}_s*.log"
echo "waiting..."

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

echo
echo "=== $TAG finished, $failed of $SHARDS shards exited non-zero ==="
for log in "$LOGDIR/${TAG}"_s*.log; do
    echo "--- $log"
    grep -E "^\[branch\] [0-9]+/|GATE|  OK|  FAIL|^  n=|Opportunity|ratio|sigma|->" "$log" \
        | tail -22
done

echo
echo "=== aggregate ==="
"$PY" - "$OUTROOT" "$TAG" <<'PY'
import sys, json, glob
import numpy as np

root, tag = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{root}/{tag}_s*.json"))
if not files:
    print("no shard json found")
    raise SystemExit(1)

totals = {}
for path in files:
    d = json.load(open(path))
    for key in ("branches", "points_probed", "skipped_points"):
        totals[key] = totals.get(key, 0) + (d.get(key) or 0)
    for key in ("gate_a0_max", "gate_a1_max", "gate_b_p95"):
        if d.get(key) is not None:
            totals.setdefault(f"max_{key}", []).append(d[key])
    print(f"{path.rsplit('/',1)[-1]:28s} branches {d.get('branches',0):5d}  "
          f"ratio {d.get('ratio')}  opp {d.get('opportunity_rate')}  "
          f"gate_failed {d.get('gate_failed')}")

print()
print(f"branches {totals.get('branches',0)}  points {totals.get('points_probed',0)}  "
      f"skipped {totals.get('skipped_points',0)}")
for key, values in totals.items():
    if key.startswith("max_"):
        print(f"{key}: {max(values):.3e}")

# Pool the branch records across shards and recompute the primary number, so the headline
# is not an average of ratios (which would weight a 2-point shard the same as a 40-point one).
point_ids, sigmas, cstart, cend = [], [], [], []
for path in glob.glob(f"{root}/{tag}_s*.npz"):
    data = np.load(path)
    offset = max(point_ids) + 1 if point_ids else 0
    point_ids.extend((data["branch_point"] + offset).tolist())
    sigmas.extend(data["branch_sigma"].tolist())
    cstart.extend(data["branch_count_start"].tolist())
    cend.extend(data["branch_count_end"].tolist())
if not point_ids:
    raise SystemExit(0)

point_ids = np.array(point_ids); sigmas = np.array(sigmas)
cend = np.array(cend); cstart = np.array(cstart)
within, means, changed, opp = [], [], [], []
for pid in np.unique(point_ids):
    mask = point_ids == pid
    pert = mask & (sigmas > 0)
    ctrl = mask & (sigmas == 0)
    if pert.sum() < 2 or not ctrl.any():
        continue
    within.append(float(np.var(cend[pert])))
    means.append(float(np.mean(cend[pert])))
    changed.append(float(np.mean(cend[pert] != cend[ctrl][0])))
    opp.append(float(np.max(cend[ctrl]) != cstart[ctrl][0]))

print()
print(f"pooled states: {len(within)}")
print(f"within-state variance {np.mean(within):.4f}")
print(f"across-state variance {np.var(means):.4f}")
print(f"ratio                 {np.mean(within) / max(np.var(means), 1e-12):.3f}")
print(f"changed fraction      {np.mean(changed):.3f}")
print(f"opportunity rate      {np.mean(opp):.3f}")
if len(within) < 10:
    print("-> PRELIMINARY: fewer than 10 states; readable direction, not a number.")
elif np.mean(opp) < 0.5:
    print("-> INCONCLUSIVE: too few states had the opportunity to change.")
else:
    ratio = np.mean(within) / max(np.var(means), 1e-12)
    print("-> " + ("action-sensitive" if ratio > 0.30
                   else "weak" if ratio > 0.05
                   else "state-only"))
PY
