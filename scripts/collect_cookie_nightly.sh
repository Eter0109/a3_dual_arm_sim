#!/bin/bash
# Collect cookie-transfer demonstrations in parallel shards.
#
# Each shard writes its own LeRobot dataset: the writer emits one parquet file
# and one video set per dataset, so concurrent writers cannot share a root. Seeds
# are interleaved across shards, and the shards are merged afterwards.
#
# Usage:  bash scripts/collect_cookie_nightly.sh [shards] [episodes_per_shard]
#
# The scripted expert reaches a valid pack roughly half the time with the default
# placement noise, so episodes_per_shard should be sized against that: asking for
# 6 from a shard typically costs 12-15 attempts.

set -uo pipefail

SHARDS="${1:-8}"
EPISODES="${2:-6}"
SEED_BASE="${3:-1000}"

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROJECT}/outputs/datasets/cookie_overnight"
LOG_DIR="${PROJECT}/logs/cookie_collection"

source /lab/haoq_lab/cse12311731/miniconda3/etc/profile.d/conda.sh
conda activate a3_sim

mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

echo "shards=${SHARDS} episodes_per_shard=${EPISODES} seed_base=${SEED_BASE}"
echo "output=${OUTPUT_ROOT}"
echo

pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
    root="${OUTPUT_ROOT}/shard_$(printf '%02d' "${shard}")"
    log="${LOG_DIR}/shard_$(printf '%02d' "${shard}").log"

    if [ -d "${root}" ] && [ -n "$(ls -A "${root}" 2>/dev/null)" ]; then
        echo "  shard ${shard}: ${root} already has content, skipping"
        continue
    fi

    # Keep the recorded episode short: the wrapper stops the rollout shortly
    # after the expert's final placement instead of running to the horizon.
    setsid a3-sim collect-cookie \
        --root "${root}" \
        --repo-id "local/a3-cookie-shard-$(printf '%02d' "${shard}")" \
        --episodes "${EPISODES}" \
        --seed "${SEED_BASE}" \
        --shard-index "${shard}" \
        --shard-count "${SHARDS}" \
        --fast-render \
        > "${log}" 2>&1 &
    pids+=($!)
    echo "  shard ${shard}: pid=${!} -> ${log}"
done

echo
echo "waiting for ${#pids[@]} shards..."
failed=0
for pid in "${pids[@]}"; do
    if wait "${pid}"; then
        echo "  pid=${pid} done"
    else
        echo "  pid=${pid} FAILED"
        failed=$((failed + 1))
    fi
done

echo
echo "=== summary ==="
total=0
for ((shard = 0; shard < SHARDS; shard++)); do
    summary="${OUTPUT_ROOT}/shard_$(printf '%02d' "${shard}")/collection_summary.json"
    if [ -f "${summary}" ]; then
        accepted=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['accepted_episodes'])" "${summary}")
        attempts=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['attempts'])" "${summary}")
        echo "  shard $(printf '%02d' "${shard}"): ${accepted} accepted in ${attempts} attempts"
        total=$((total + accepted))
    else
        echo "  shard $(printf '%02d' "${shard}"): no summary"
    fi
done
echo "  total accepted episodes: ${total}"
echo "  failed shards: ${failed}"
