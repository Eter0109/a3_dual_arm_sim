#!/usr/bin/env bash
# Wait for `build_clean_v9.py` to finish, add the derived columns, then start v9.
#
# The three stages are separate scripts with separate logs because each one is worth
# re-running alone: a build that fails halfway should not force the training step to be
# re-planned, and `add_history_window.py` is cheap enough to redo.  This exists only so the
# chain does not need a human at 3 a.m., and it refuses to continue on a build that did not
# reach the expected episode count -- a short dataset would silently change what "30000
# steps" means.
#
#   bash scripts/chain_v9.sh [expected_episodes] [steps]
set -u
cd /root/a3_dual_arm_sim

EXPECTED="${1:-400}"
STEPS="${2:-30000}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BUILD=/root/autodl-tmp/datasets/a3_clean_v9
WITH_HISTORY=/root/autodl-tmp/datasets/a3_clean_v9h
LOG=/root/autodl-tmp/logs/chain_v9.log

export PYTHONPATH=/root/a3_dual_arm_sim/src
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }

say "=== chain start: waiting for the build to reach ${EXPECTED} episodes ==="
for _ in $(seq 1 480); do   # 480 x 30 s = 4 h ceiling
    done_count=$(wc -l < "$BUILD/a3_episode_metadata.jsonl" 2>/dev/null || echo 0)
    if ! pgrep -f "build_clean_v9.py" > /dev/null; then
        if [ "$done_count" -ge "$EXPECTED" ]; then break; fi
        say "build stopped at ${done_count}/${EXPECTED} episodes -- refusing to continue"
        exit 1
    fi
    sleep 30
done
done_count=$(wc -l < "$BUILD/a3_episode_metadata.jsonl" 2>/dev/null || echo 0)
say "build finished: ${done_count} episodes"
[ "$done_count" -ge "$EXPECTED" ] || { say "short build, stopping"; exit 1; }

if [ -e "$WITH_HISTORY/meta/info.json" ]; then
    say "2/3 history: ${WITH_HISTORY} already exists, skipping"
else
    say "2/3 history: deriving observation.history and observation.progress"
    $PY scripts/add_history_window.py \
        --source "$BUILD" --out "$WITH_HISTORY" --repo-id local/a3-clean-v9h \
        --steps 10 --max-batches 4 --cookies-per-batch 5 \
        > /root/autodl-tmp/logs/add_history_v9.log 2>&1
    STATUS=$?
    say "2/3 history exit=$STATUS $(grep -a 'history window' /root/autodl-tmp/logs/add_history_v9.log | tail -1)"
    [ "$STATUS" -eq 0 ] || exit 1
fi

say "3/3 starting the v9 pipeline (${STEPS} steps)"
setsid nohup bash scripts/multitask_v9_pipeline.sh "$STEPS" 8 \
    >> /root/autodl-tmp/logs/nohup_v9.log 2>&1 < /dev/null &
say "3/3 v9 launched; follow /root/autodl-tmp/logs/pipeline_multitask_v9.log"
