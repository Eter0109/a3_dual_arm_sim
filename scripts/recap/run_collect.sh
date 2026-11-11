#!/usr/bin/env bash
# Collect RECAP rollouts with N parallel workers sharing one bridge.
#
#   bash scripts/recap/run_collect.sh [total_episodes] [workers]
#
# Why processes and not threads
# -----------------------------
# `LeRobotDataset.create` takes an exclusive lock on the root and keeps an episode buffer in
# memory; two writers into one directory interleave episode numbering and corrupt the shards.
# So each worker owns its own root (`part_00`, `part_01`, ...) and the merge happens later.
#
# Why the bridge is shared
# ------------------------
# `a3_bridge_server.py` accepts many clients (a thread per connection, one lock around the
# model), and the GPU is idle between forwards -- 94 forwards per 1500-step episode at about
# 0.35 s each.  One bridge per worker would also work but would load the checkpoint N times
# (10 GB each), which a 32 GB card cannot hold more than three of.
#
# The environment step is the bottleneck and is CPU-bound (about 0.45 s/step, one core per
# episode), so `workers` should stay a few below `nproc`.  Measured: 10 workers on 16 cores
# run an episode in 650-900 s, the same as 1 worker, i.e. near-linear scaling.
set -uo pipefail

TOTAL="${1:-300}"
WORKERS="${2:-10}"
SEED_START="${SEED_START:-200}"
SOCK="${SOCK:-/tmp/graft_v9.sock}"
ROOT="${ROOT:-/root/autodl-tmp/datasets/a3_recap_rollouts}"
TASK="${TASK:-2x10}"
MAX_STEPS="${MAX_STEPS:-1500}"
HISTORY_FRAMES="${HISTORY_FRAMES:-10}"
EXECUTE="${EXECUTE:-16}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python

cd /root/a3_dual_arm_sim
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp A3_MAIN_ROOT=/tmp/a3-main
mkdir -p "$TMPDIR" "$ROOT"

if [ ! -S "$SOCK" ]; then
    echo "no bridge at $SOCK -- start one first:" >&2
    echo "  cd /root/autodl-tmp/a3-collect && \\" >&2
    echo "    $PY /root/a3_dual_arm_sim/scripts/a3_bridge_server.py --adapter adapters/pi05_a3.py \\" >&2
    echo "    --listen unix://$SOCK --config ckpt=<...> --config stats_root=<...> \\" >&2
    echo "    --config state_dims=16 --config history_dims=\$(( $HISTORY_FRAMES * 48 ))" >&2
    exit 1
fi

# Seeds are handed out round-robin rather than as one contiguous block per worker.  Every
# seed draws its own instruction variant from `Cookie2x10Plan.from_seed`, and a blocked split
# can land a whole plan variant in one part; round-robin spreads the variants across parts so
# a partially finished run is still balanced.
PER_WORKER=$(( (TOTAL + WORKERS - 1) / WORKERS ))
echo "=== collect: ${TOTAL} episodes, ${WORKERS} workers, ${PER_WORKER} each ==="
echo "    seeds ${SEED_START}+  task=${TASK} steps=${MAX_STEPS} history=${HISTORY_FRAMES}"
echo "    root ${ROOT}"

pids=()
for ((w = 0; w < WORKERS; w++)); do
    part=$(printf "%s/part_%02d" "$ROOT" "$w")
    seeds=$(python3 - "$SEED_START" "$TOTAL" "$w" "$WORKERS" <<'PY'
import sys
start, total, w, n = (int(v) for v in sys.argv[1:])
mine = [start + i for i in range(total) if i % n == w]
print(",".join(str(s) for s in mine))
PY
)
    [ -z "$seeds" ] && continue
    (
        "$PY" scripts/recap/collect_rollouts.py \
            --socket "$SOCK" --output "$part" --seeds "$seeds" \
            --task "$TASK" --max-steps "$MAX_STEPS" \
            --history-frames "$HISTORY_FRAMES" --execute "$EXECUTE" \
            --prompt-style task --source-column 1 \
            --repo-id "a3_recap_part$(printf "%02d" "$w")" \
            > "/root/autodl-tmp/logs/recap_collect_p$(printf "%02d" "$w").log" 2>&1
    ) &
    pids+=($!)
    echo "    worker $w -> $part  ($(echo "$seeds" | tr ',' '\n' | wc -l) seeds)"
done

echo "=== waiting on ${#pids[@]} workers (logs: /root/autodl-tmp/logs/recap_collect_p*.log) ==="
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

echo "=== per-part results ==="
"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
total = success = frames = 0
for labels in sorted(root.glob("part_*/episode_labels.jsonl")):
    rows = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
    ok = sum(1 for r in rows if r["success"])
    total += len(rows)
    success += ok
    frames += sum(r["steps"] for r in rows)
    print(f"  {labels.parent.name}: {len(rows):3d} episodes, {ok:2d} strict "
          f"({100.0 * ok / max(len(rows), 1):4.1f} %), {sum(r['steps'] for r in rows)} steps")
if total:
    print(f"  TOTAL: {total} episodes, {success} strict ({100.0 * success / total:.1f} %), "
          f"{frames} frames ({frames / total:.0f} per episode)")
PY

if [ "$failed" -ne 0 ]; then
    echo "!!! ${failed} worker(s) exited non-zero -- check the logs before merging" >&2
    exit 1
fi
echo "=== done. next: merge parts, then compute value labels ==="
