#!/usr/bin/env bash
# Six hours overnight: one long constrained-DSRL run, plus data collection running beside it.
#
#   bash scripts/recap/run_dsrl_night.sh [hours]
#
# Why these are two jobs and not one loop
# ---------------------------------------
# The obvious design is collect -> retrain -> collect, and it does not work here: `train_dsrl.py`
# observes through the frozen SigLIP summary cached in `features_v9` plus the pi0 VLM's final-token
# hidden state, and a newly collected episode has neither.  The VLM feature costs one batched pass
# (~45 ms per frame decode + 16 ms VLM) and is cached under `<dataset>/dsrl_cache`, but the newly
# collected episodes are not in `a3_recap_all` at all, so a nightly loop would still spend its time
# featurising and never close the loop.  Rather than build an expensive fake iteration, the night
# does the two things that are each useful on their own:
#
#   1. **A long training run** (the real question).  Offline DSRL diverged by `|mean| 0.27 -> 5.02`
#      in 1000 epochs; with the soft clip it went 0.267 -> 0.107 in 300.  Whether that holds for
#      tens of thousands of epochs is unknown and is what tonight answers.  Checks are written
#      every 500 epochs so the morning has a curve, not an endpoint.
#   2. **Fresh episodes** (for a later run).  Collected with the bridge, stored with traces, ready
#      to be featurised once rather than in a loop.  `--no-record` keeps this cheap: traces only,
#      no video, so 300 episodes is a few GB instead of a few hundred.
#
# Memory: the bridge needs 9.7 GB, and the trainer holds the 4 B-parameter policy (~10 GB) only
# while it builds or loads the feature cache -- the training loop itself is three MLPs on cached
# features and never touches the VLA.  Measured together at 20 GB of 32.  While the cache is being
# built the two share the GPU and both run slower; after that the trainer is CPU-bound and the
# bridge has the card to itself, which is a materially different profile from the one this script
# was first written for.
set -uo pipefail

HOURS="${1:-6}"
SHARDS=8            # 8 collectors + 1 trainer on 16 cores; 12 collectors starved the trainer

cd "$(dirname "$0")/../.."
PROJECT="$PWD"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
ROOT=/root/autodl-tmp/datasets/dsrl_night
LOG=/root/autodl-tmp/logs/dsrl_night.log
CKPT=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
SOCK=/tmp/dsrl_night.sock

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp A3_MAIN_ROOT=/tmp/a3-main
mkdir -p "$ROOT" /root/autodl-tmp/logs "$TMPDIR"

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }

say "=== overnight: ${HOURS}h -- constrained DSRL training + fresh collection ==="

pkill -f "a3_bridge_server.py --adapter" 2>/dev/null || true
sleep 5
cd /root/autodl-tmp/a3-collect
setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
    --adapter adapters/pi05_a3.py --listen "unix://$SOCK" \
    --config "ckpt=$CKPT" --config state_dims=16 --config "history_dims=480" \
    > /root/autodl-tmp/logs/bridge_dsrl_night.log 2>&1 < /dev/null &
cd "$PROJECT"
for _ in $(seq 1 120); do
    grep -q "listening on" /root/autodl-tmp/logs/bridge_dsrl_night.log 2>/dev/null && break
    sleep 5
done
grep -q "listening on" /root/autodl-tmp/logs/bridge_dsrl_night.log || { say "bridge failed"; exit 1; }
say "bridge ready"

# --- job 1: collection, in the background ------------------------------------------------
# Sampled episode length is ~12 min, so `HOURS` hours x 8 shards is roughly
# HOURS * 60 / 12 * 8 episodes; the seed range is set generously and the process is killed at the
# deadline rather than being made to stop cleanly -- a partial part is still usable via merge.
collect_seed_end=$(( 2000 + HOURS * 60 ))
mkdir -p "$ROOT/collect"
for (( shard = 0; shard < SHARDS; shard++ )); do
    from=$(( 2000 + shard * 20 ))
    PYTHONPATH=/root/a3_dual_arm_sim/src "$PY" scripts/recap/collect_rollouts.py \
        --policy bridge --socket "$SOCK" \
        --output "$ROOT/collect/part_$(printf '%02d' "$shard")" \
        --seeds "${from}-${collect_seed_end}" --task 2x10 --max-steps 1500 \
        --no-record --prompt-style task --history-frames 10 --execute 16 \
        > "/root/autodl-tmp/logs/dsrl_collect_$(printf '%02d' "$shard").log" 2>&1 &
done
say "collection started on ${SHARDS} shards"

# --- job 2: long training, in the foreground ---------------------------------------------
# 400 epochs took 130 s with the GPU to itself under the old per-update VLA forward.  The loop is
# now three MLPs (~4 ms per update, CPU-bound), so 60000 epochs x 20 updates is roughly an hour of
# gradient work plus a few minutes of feature caching; `--check-every 500` still leaves a curve.
started=$(date +%s)
PYTHONPATH=/root/a3_dual_arm_sim/src "$PY" scripts/recap/train_dsrl.py \
    --epochs 60000 --batch 64 --utd 20 --check-every 500 \
    --transitions 2048 --distill-per-frame 8 \
    --output "$ROOT/actor_long/checkpoint.pt" \
    2>&1 | tee -a "$LOG"
say "training finished after $(( ($(date +%s) - started) / 60 )) min"

# --- stop the collectors -----------------------------------------------------------------
pkill -f "collect_rollouts.py --policy bridge --socket $SOCK" 2>/dev/null || true
sleep 5
pkill -f "a3_bridge_server.py --adapter adapters/pi05_a3.py --listen unix://$SOCK" 2>/dev/null || true
collected=$(cat "$ROOT"/collect/part_*/episode_labels.jsonl 2>/dev/null | wc -l)
say "=== done: ${collected} episodes collected -> $ROOT/collect ==="
say "next: featurise them once, then rerun train_dsrl.py with --dataset pointing at the merged set"
