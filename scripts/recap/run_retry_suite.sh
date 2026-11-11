#!/usr/bin/env bash
# Run N independent attempts over the SAME scenes, then report what best-of-N would score.
#
#   bash scripts/recap/run_retry_suite.sh <checkpoint> <tag> <scenes> <attempts>
#
# Why independent attempts are enough
# ------------------------------------
# The bridge seeds torch only when `A3_PI05_SEED` is set (see `pi05_a3.py`'s `infer`), and it is
# not set here, so every forward pass draws fresh flow-matching noise.  That means two runs over
# the same `--seeds` are genuinely independent attempts at the same scene -- there is nothing to
# change and no new seed to invent.  The prefix needs no special handling.
#
# What this is for
# ----------------
# v9 measured on 8 scenes, three attempts each: within-scene standard deviation 3.34 Cookies, no
# scene repeated identically, best-of-3 mean 16.00 against a single-attempt mean of 12.62.  That
# is a large gain from arithmetic alone, so it is worth confirming on a wider scene set before
# anyone builds policy on it -- 8 scenes is a small base for a +3.4 claim.
set -euo pipefail

CKPT="${1:?usage: run_retry_suite.sh <checkpoint> <tag> <scenes> <attempts>}"
TAG="${2:?usage: run_retry_suite.sh <checkpoint> <tag> <scenes> <attempts>}"
SCENES="${3:-20}"
ATTEMPTS="${4:-3}"
SHARDS="${5:-12}"

cd "$(dirname "$0")/../.."
ROOTS=()
for (( attempt = 1; attempt <= ATTEMPTS; attempt++ )); do
    ROOTS+=("${TAG}_a${attempt}")
    A3_POLICY_CKPT="$CKPT" \
        bash scripts/recap/eval_conditioned.sh "${TAG}_a${attempt}" "" "$SCENES" "$SHARDS"
done

echo
/root/autodl-tmp/a3-collect/.venv/bin/python scripts/recap/best_of_n.py \
    --roots "${ROOTS[@]}" \
    --output "/root/autodl-tmp/datasets/retry_${TAG}.json"
