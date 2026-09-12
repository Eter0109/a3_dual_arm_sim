#!/bin/bash
# Launch one batch of parallel cookie-collection shards.
#
# Kept separate from collect_cookie_nightly.sh so a second batch can start while
# the first is still running: bash reads a script incrementally, so editing a file
# that a running job is executing is unsafe.
#
# Usage:  bash scripts/collect_cookie_batch.sh <output_name> <shards> <episodes> <seed_base>

set -uo pipefail

OUTPUT_NAME="${1:?output name required}"
SHARDS="${2:-6}"
EPISODES="${3:-6}"
SEED_BASE="${4:-2000}"

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROJECT}/outputs/datasets/${OUTPUT_NAME}"
LOG_DIR="${PROJECT}/logs/${OUTPUT_NAME}"

source /lab/haoq_lab/cse12311731/miniconda3/etc/profile.d/conda.sh
conda activate a3_sim

mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

echo "batch=${OUTPUT_NAME} shards=${SHARDS} episodes=${EPISODES} seed_base=${SEED_BASE}"
echo "output=${OUTPUT_ROOT}"
echo

pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
    root="${OUTPUT_ROOT}/shard_$(printf '%02d' "${shard}")"
    log="${LOG_DIR}/shard_$(printf '%02d' "${shard}").log"

    if [ -d "${root}" ] && [ -n "$(ls -A "${root}" 2>/dev/null)" ]; then
        echo "  shard ${shard}: content already present, skipping"
        continue
    fi

    setsid a3-sim collect-cookie \
        --root "${root}" \
        --repo-id "local/${OUTPUT_NAME}-shard-$(printf '%02d' "${shard}")" \
        --episodes "${EPISODES}" \
        --seed "${SEED_BASE}" \
        --shard-index "${shard}" \
        --shard-count "${SHARDS}" \
        --fast-render \
        > "${log}" 2>&1 &
    pids+=($!)
    echo "  shard ${shard}: pid=${!}"
done

echo
echo "waiting for ${#pids[@]} shards..."
for pid in "${pids[@]}"; do
    wait "${pid}" && echo "  pid=${pid} done" || echo "  pid=${pid} FAILED"
done

echo
echo "=== ${OUTPUT_NAME} summary ==="
total=0
for ((shard = 0; shard < SHARDS; shard++)); do
    summary="${OUTPUT_ROOT}/shard_$(printf '%02d' "${shard}")/collection_summary.json"
    if [ -f "${summary}" ]; then
        accepted=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['accepted_episodes'])" "${summary}")
        attempts=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['attempts'])" "${summary}")
        echo "  shard $(printf '%02d' "${shard}"): ${accepted} accepted / ${attempts} attempts"
        total=$((total + accepted))
    else
        echo "  shard $(printf '%02d' "${shard}"): no summary"
    fi
done
echo "  total accepted: ${total}"
