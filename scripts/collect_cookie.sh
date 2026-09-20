#!/bin/bash
# Collect demonstrations in parallel shards, for any registered scene.
#
# Each shard writes its own LeRobot dataset: the writer emits one parquet file and
# one video set per dataset, so concurrent writers cannot share a root.  Seeds are
# interleaved across shards (`shard_index + attempt * shard_count`), so the union of
# the shards is still one clean seed sweep.  Merge them afterwards with
# `scripts/merge_cookie_shards.py`.
#
# Usage:
#   bash scripts/collect_cookie.sh <output_name> <scene> [shards] [episodes] [seed_base]
#
#   # A night's collection: 8 shards x 6 episodes of the same-column scene.
#   bash scripts/collect_cookie.sh a3_cookie_overnight a3_cookie_same_column
#
#   # A second run while the first is still going needs its own output name and a
#   # disjoint seed base, otherwise the two sweeps overlap.
#   bash scripts/collect_cookie.sh a3_cookie_second a3_cookie_same_column 8 6 480
#
#   # Inspect what can be collected and what each scene needs.
#   a3-sim collect --help      # and see a3_dual_arm_sim.collection.registered_scenes
#
# Running a scene is not a matter of choosing one that exists: the batch scenes
# collect on the current dense layout, while `a3_cookie_transfer` no longer does
# (see its builder docstring) and would spend the whole run accepting nothing.
#
# Sizing: the scripted expert does not succeed on every attempt, so ask for fewer
# episodes than you want attempts.  A shard given `episodes` will run until it has
# that many successes or hits its own attempt cap, which defaults to five times
# `episodes`.
#
# Site facts, all overridable:
#   CONDA_SH    path to conda's profile script   (default: the dev machine's)
#   A3_ENV      environment name to activate     (default: a3_sim)
#   A3_EXTRA    extra `a3-sim collect` arguments (default: --fast-render)

set -uo pipefail

OUTPUT_NAME="${1:?output name required, e.g. a3_cookie_overnight}"
SCENE="${2:?scene name required, e.g. a3_cookie_same_column}"
SHARDS="${3:-8}"
EPISODES="${4:-6}"
SEED_BASE="${5:-1000}"

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROJECT}/outputs/datasets/${OUTPUT_NAME}"
LOG_DIR="${PROJECT}/logs/${OUTPUT_NAME}"

# Both the interpreter path and the environment name are site facts.  The defaults
# are the machine this was developed on; set CONDA_SH and A3_ENV elsewhere.
source "${CONDA_SH:-/lab/haoq_lab/cse12311731/miniconda3/etc/profile.d/conda.sh}"
conda activate "${A3_ENV:-a3_sim}"

# `--fast-render` halves frame time.  It must match however the dataset is later
# evaluated, or a checkpoint sees pixels it was not trained on; every other script
# in this directory sets it unconditionally for the same reason.
EXTRA_ARGS=(${A3_EXTRA:---fast-render})

mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

echo "scene    =${SCENE}"
echo "shards   =${SHARDS}  episodes_per_shard=${EPISODES}  seed_base=${SEED_BASE}"
echo "output   =${OUTPUT_ROOT}"
echo "extra    =${EXTRA_ARGS[*]}"
echo

pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
    root="${OUTPUT_ROOT}/shard_$(printf '%02d' "${shard}")"
    log="${LOG_DIR}/shard_$(printf '%02d' "${shard}").log"

    if [ -d "${root}" ] && [ -n "$(ls -A "${root}" 2>/dev/null)" ]; then
        echo "  shard ${shard}: ${root} already has content, skipping"
        continue
    fi

    setsid a3-sim collect \
        --scene "${SCENE}" \
        --root "${root}" \
        --repo-id "local/${OUTPUT_NAME}-shard-$(printf '%02d' "${shard}")" \
        --episodes "${EPISODES}" \
        --seed "${SEED_BASE}" \
        --shard-index "${shard}" \
        --shard-count "${SHARDS}" \
        "${EXTRA_ARGS[@]}" \
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
        echo "  shard $(printf '%02d' "${shard}"): no summary (it did not finish)"
    fi
done
echo "  total accepted episodes: ${total}"
echo "  failed shards: ${failed}"
echo
echo "next: merge the shards into one training dataset"
echo "  python scripts/merge_cookie_shards.py \\"
echo "    --shards ${OUTPUT_ROOT}/shard_* \\"
echo "    --repo-id local/${OUTPUT_NAME}-merged \\"
echo "    --output ${PROJECT}/outputs/datasets/${OUTPUT_NAME}_merged"
