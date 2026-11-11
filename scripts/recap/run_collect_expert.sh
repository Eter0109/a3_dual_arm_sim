#!/usr/bin/env bash
# Collect expert demonstrations in parallel. Expert-only, CPU-only.
#
#   bash scripts/recap/run_collect_expert.sh [total_episodes] [workers]
#
# Why a separate script from `run_collect.sh`
# ------------------------------------------
# `run_collect.sh` drives `--policy bridge` and therefore needs the GPU resident bridge.
# The expert is a deterministic controller with no torch import at all -- it reads MuJoCo
# state and computes joint targets -- so this path uses **no GPU** and can run at the same
# time as a rollout collection without contending for the card.  That is what makes a single
# overnight pass possible instead of two sequential ones.
#
# The division of labour measured on this box: 8 rollout workers plus 6 expert workers on 16
# cores, with the rollout workers waiting on the bridge every 16 steps and the expert never
# waiting at all.  Both sets take the same per-episode wall time as they do alone.
#
# Each worker still owns its own root (see `run_collect.sh` for why), and the same
# `merge_parts.py` joins them afterwards -- including the expert and the rollout set, since
# the collector writes both with one schema.
set -uo pipefail

TOTAL="${1:-100}"
WORKERS="${2:-6}"
SEED_START="${SEED_START:-1000}"
ROOT="${ROOT:-/root/autodl-tmp/datasets/a3_recap_expert}"
TASK="${TASK:-2x10}"
MAX_STEPS="${MAX_STEPS:-1500}"
HISTORY_FRAMES="${HISTORY_FRAMES:-10}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python

cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp A3_MAIN_ROOT=/tmp/a3-main
mkdir -p "$TMPDIR" "$ROOT"

# The collector's own guard already refuses a non-empty output root, per part, so this only
# needs to catch the case where a previous run left a partial tree behind: the per-part guard
# would then fail inside each worker and the summary at the end would look like a data problem
# rather than a stale directory.
if [ -d "$ROOT" ] && [ -n "$(ls -A "$ROOT" 2>/dev/null)" ]; then
    echo "refusing to overwrite non-empty $ROOT (remove it, or set ROOT elsewhere)" >&2
    exit 1
fi

echo "=== expert: ${TOTAL} episodes, ${WORKERS} workers ==="
echo "    seeds ${SEED_START}+  task=${TASK} steps=${MAX_STEPS}  (CPU only, no GPU)"

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
            --policy expert --output "$part" --seeds "$seeds" \
            --task "$TASK" --max-steps "$MAX_STEPS" \
            --history-frames "$HISTORY_FRAMES" \
            --prompt-style task --source-column 1 \
            --repo-id "a3_recap_expert$(printf "%02d" "$w")" \
            > "/root/autodl-tmp/logs/recap_expert_p$(printf "%02d" "$w").log" 2>&1
    ) &
    pids+=($!)
    echo "    worker $w -> $part  ($(echo "$seeds" | tr ',' '\n' | wc -l) seeds)"
done

echo "=== waiting on ${#pids[@]} workers (logs: /root/autodl-tmp/logs/recap_expert_p*.log) ==="
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
echo "=== done. next: merge_parts.py --parts $ROOT/part_* ===="
