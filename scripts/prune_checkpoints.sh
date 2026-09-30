#!/usr/bin/env bash
# Keep only the newest N checkpoints of a lerobot training run.
#
#   bash scripts/prune_checkpoints.sh <run_dir> [keep] [interval_s]
#
# Why this is needed
# ------------------
# `lerobot_train.py` has `save_freq` but no `save_total_limit`, so a 20,000-step run at
# `save_freq=2000` above writes ten checkpoints.  Each one holds the model plus the
# optimizer state for the trainable parameters, which for π0.5 is the better part of
# 15 GB -- 150 GB in total, against 103 GB of free space.  Nothing warns; the run simply
# dies at whatever hour the disk fills, having already spent the time.
#
# Why a pruner rather than a larger `save_freq`
# --------------------------------------------
# Because the checkpoints are the point.  A long fine-tune is judged by evaluating
# intermediates -- the loss curve of a flow-matching action expert says little about
# whether the policy works, and the cheap way to find out is to run the first few
# checkpoints on the task.  Widening `save_freq` to fit the disk would keep the *last*
# one and throw away the ability to see when it started working.
#
# The two-minute guard
# --------------------
# A checkpoint directory appears before its files finish being written, so a pruner keyed
# only on "newest N" can delete the one currently being saved if two are close together.
# Anything modified within `GUARD_S` seconds is therefore left alone regardless of rank,
# and the next pass picks it up.
#
# Runs in the foreground by default and prints each pass, so it is usable from a terminal;
# `--daemon` backgrounds it with the same output behaviour.

set -uo pipefail

RUN_DIR="${1:?usage: prune_checkpoints.sh <run_dir> [keep] [interval_s]}"
KEEP="${2:-3}"
INTERVAL="${3:-300}"
GUARD_S="${GUARD_S:-120}"

CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "no checkpoints directory yet at ${CHECKPOINT_DIR}; waiting for it"
fi

prune_once() {
    # The checkpoint directories are named by step with six digits -- `002000`, not
    # `checkpoint-002000`.  That is easy to get wrong, and getting it wrong is silent: a
    # glob that matches nothing removes nothing, the pruner reports success on every pass,
    # and the disk fills at whatever hour the tenth checkpoint lands.  The pattern here
    # accepts both spellings so a lerobot version that changes to the prefixed form keeps
    # working, and the `-regex` is the real filter (`find -name` cannot express
    # "digits only").
    local now kept=0 removed=0
    now=$(date +%s)

    while read -r mtime path; do
        [ -n "${path:-}" ] || continue
        # Never touch a directory that may still be being written.
        if [ $((now - mtime)) -lt "${GUARD_S}" ]; then
            echo "  keep   $(basename "${path}") ($((now - mtime))s old, inside the guard)"
            continue
        fi
        if [ "${kept}" -lt "${KEEP}" ]; then
            kept=$((kept + 1))
            echo "  keep   $(basename "${path}")"
            continue
        fi
        echo "  remove $(basename "${path}")"
        rm -rf "${path}"
        removed=$((removed + 1))
    done < <(
        find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d \
            \( -regex '.*/[0-9]+' -o -name 'checkpoint-*' \) \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk '{print int($1), $2}'
    )

    if [ "${removed}" -eq 0 ] && [ "${kept}" -eq 0 ]; then
        echo "  (no checkpoints yet)"
    elif [ "${removed}" -gt 0 ]; then
        echo "  removed ${removed}, kept ${kept}"
    fi
}

if [ "${DAEMON:-0}" = "1" ] || [ "${4:-}" = "--daemon" ]; then
    echo "pruning ${CHECKPOINT_DIR}, keeping ${KEEP}, every ${INTERVAL}s"
    while true; do
        echo "[$(date '+%H:%M:%S')] pass"
        prune_once
        sleep "${INTERVAL}"
    done
else
    echo "pruning ${CHECKPOINT_DIR}, keeping ${KEEP}"
    prune_once
fi
