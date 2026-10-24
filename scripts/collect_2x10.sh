#!/usr/bin/env bash
# Collect the 20-cookie task, eight shards at a time.
#
# Why this exists separately from the 2x5 collection: main's 2x10 collector is a
# single-process CLI with no shard argument.  Its seed advances by one per attempt
# (`seed = max(saved, attempts) + 1`), so shards are made disjoint by starting each
# one a thousand seeds apart rather than by interleaving -- the seed only decides the
# instruction variant (`[n, 10-n, n, 10-n]`) and the layout draw, so spaced blocks cover
# the same space as an interleaved sweep.
#
# Cost, measured on this machine (`probe_2x10_t`, seed 0): **386.9 s and 1071 steps for
# a 20/20 episode**.  That is 6.4 minutes per accepted episode at eight shards, so 320
# episodes is about 4.3 hours.
#
#   bash /root/autodl-tmp/logs/collect_2x10.sh [episodes_per_shard] [shards]
set -u

EPISODES="${1:-40}"
SHARDS="${2:-8}"
BASE_SEED=0
SEED_STRIDE=1000
OUT_ROOT=/root/autodl-tmp/datasets/a3_2x10_v1
LOG_DIR=/root/autodl-tmp/logs/collect_2x10
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
MAIN=/tmp/a3-main

export MUJOCO_GL=egl
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export PYTHONPATH="$MAIN/src"

mkdir -p "$LOG_DIR" "$OUT_ROOT"
cd "$MAIN"

echo "2x10 collection: $SHARDS shards x $EPISODES episodes, seeds ${BASE_SEED}+${SEED_STRIDE}k"
pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
    root="$OUT_ROOT/shard_$(printf '%02d' "$shard")"
    log="$LOG_DIR/shard_$(printf '%02d' "$shard").log"
    if [ -d "$root" ] && [ -n "$(ls -A "$root" 2>/dev/null)" ]; then
        echo "  shard $shard: already has content, skipping"
        continue
    fi
    seed_start=$(( BASE_SEED + shard * SEED_STRIDE ))
    setsid nohup "$PY" -m a3_dual_arm_sim.data.collect_cookie_2x10_cli \
        --root "$root" \
        --repo-id "local/a3-2x10-v1-shard-$(printf '%02d' "$shard")" \
        --episodes "$EPISODES" \
        --seed-start "$seed_start" \
        --max-attempts $(( EPISODES * 3 )) \
        --max-steps 3200 \
        > "$log" 2>&1 < /dev/null &
    pids+=($!)
    echo "  shard $shard: seeds from $seed_start, pid=$!"
done

echo
echo "waiting for ${#pids[@]} shards..."
for pid in "${pids[@]}"; do
    wait "$pid" || echo "  pid=$pid exited non-zero"
done

echo
echo "=== summary ==="
total=0
for ((shard = 0; shard < SHARDS; shard++)); do
    root="$OUT_ROOT/shard_$(printf '%02d' "$shard")"
    if [ -f "$root/meta/info.json" ]; then
        n=$("$PY" -c "import json;print(json.load(open('$root/meta/info.json'))['total_episodes'])")
        f=$("$PY" -c "import json;print(json.load(open('$root/meta/info.json'))['total_frames'])")
        echo "  shard $(printf '%02d' "$shard"): $n episodes, $f frames"
        total=$(( total + n ))
    else
        echo "  shard $(printf '%02d' "$shard"): no dataset"
    fi
done
echo "  total: $total episodes"
