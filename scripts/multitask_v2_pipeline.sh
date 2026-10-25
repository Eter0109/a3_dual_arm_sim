#!/usr/bin/env bash
# Balanced 2x5/2x10 arm: train on the re-weighted set, then evaluate both benchmarks
# with main's own instructions.
#
# Why this replaces the 30000-step arm
# ------------------------------------
# Three checkpoints of the first multitask arm, on the same six seeds, 2x10 with main's
# instruction:
#
#     15k   4.50  (3 of 6 seeds zero)
#     22.5k 5.00  (1 of 6 zero)      <- peak
#     30k   4.33  (2 of 6 zero)      <- and the previous, single-seed reading was 9
#
# So more steps past ~20k do not help, and the earlier "under-trained" reading was wrong.
# The composition is the binding constraint: at repeat=1 the 2x10 frames are 33.7% of the
# set, so two thirds of every batch teaches 2x5 -- including the property that stops the
# arm at two batches, which is exactly what it does on 2x10 (most seeds move out only six
# or seven of the twenty).
#
# More data cannot fix that, and the arithmetic says so: a 15000-step arm sees 90000 of
# the set's 853113 frames, 10.5%, i.e. roughly 156 of 1480 episodes once.  The data is
# 9.5x the budget.  What is affordable is the *sampling weight*, so the 2x10 shards are
# concatenated twice (`--two-by-ten-repeat 2`), which takes them from 33.7% to 50.4% of
# the frames.  Training also stops inside the measured peak (15000) instead of running
# past it.
#
#   bash /root/autodl-tmp/logs/multitask_v2_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-15000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=/root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
DATA=/root/autodl-tmp/datasets/a3_multitask_v2
OUT=/root/autodl-tmp/pi05-runs/a3_multitask_v2_${STEPS}
SOCK=/tmp/mtv2_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v2.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v2_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== balanced pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing"; exit 1; }
FRAMES=$("$PY" -c "import json;print(json.load(open('$DATA/meta/info.json'))['total_frames'])")
EPISODES=$("$PY" -c "import json;print(json.load(open('$DATA/meta/info.json'))['total_episodes'])")
say "0/3 dataset: ${EPISODES} episodes, ${FRAMES} frames; ${STEPS}x6 = $(python3 -c "print(round(${STEPS}*6/${FRAMES},3))") epoch"

if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from 4.05, checkpoints every $(( STEPS / 2 ))"
    pkill -f a3_bridge_server.py 2>/dev/null
    sleep 5
    env A3_PI05_BASE="$BASE" A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_multitask_v2_${STEPS}" \
          --policy.optimizer_lr=3e-06 \
          --policy.scheduler_decay_lr=2e-07 \
          --policy.scheduler_warmup_steps=$(( STEPS / 30 )) \
          --policy.scheduler_decay_steps="$STEPS" \
          --save_freq=$(( STEPS / 2 )) \
          --dataset.video_backend=pyav \
          > "$TRAIN_LOG" 2>&1
    STATUS=$?
    say "1/3 train exit=$STATUS $(grep -ao 'Training: 100%[^|]*' "$TRAIN_LOG" | tail -1)"
    [ "$STATUS" -eq 0 ] || exit 1
fi

CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
say "2/3 serving $(basename "$OUT")"
pkill -f a3_bridge_server.py 2>/dev/null
sleep 5
cd /root/autodl-tmp/a3-collect
setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
    --adapter adapters/pi05_a3.py --listen "unix://$SOCK" --config "ckpt=$CKPT" \
    > /root/autodl-tmp/logs/bridge_mtv2${STEPS}.log 2>&1 < /dev/null &
cd /root/a3_dual_arm_sim
for _ in $(seq 1 180); do
    grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv2${STEPS}.log 2>/dev/null && break
    sleep 5
done
grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv2${STEPS}.log || { say "2/3 bridge never came up"; exit 1; }

# One episode per process: a 2x10 episode at the 3200-step budget measured 5262 s.
say "3/3 evaluating 2x5 (${EVAL_SEEDS}, 900 steps) + 2x10 (${EVAL_SEEDS}, 3200 steps)"
for start in $(seq 0 $(( EVAL_SEEDS - 1 ))); do
    setsid nohup "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x5 --episodes 1 --seed-start "$start" --max-steps 900 \
        --output "/root/autodl-tmp/logs/v2_2x5_${start}.json" \
        > "/root/autodl-tmp/logs/v2_2x5_${start}.log" 2>&1 < /dev/null &
    setsid nohup "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x10 --episodes 1 --seed-start "$start" --max-steps 3200 \
        --output "/root/autodl-tmp/logs/v2_2x10_${start}.json" \
        > "/root/autodl-tmp/logs/v2_2x10_${start}.log" 2>&1 < /dev/null &
done
wait
say "3/3 evaluation finished"

"$PY" - >> "$LOG" 2>&1 <<'PY'
import glob, json
for row, cap in (("2x5", 10), ("2x10", 20)):
    rows = {}
    for path in sorted(glob.glob(f"/root/autodl-tmp/logs/v2_{row}_*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        for episode in data.get("episodes", []):
            rows[episode["seed"]] = episode
    if not rows:
        print(f"{row}: no results")
        continue
    keys = sorted(rows)
    scores = [rows[k]["score"] for k in keys]
    moved = [80 - rows[k]["cookies_in_source"] for k in keys]
    print(f"{row}: mean={sum(scores)/len(scores):.2f}/{cap} per-seed={scores} "
          f"zeros={scores.count(0)} moved-out={moved}")
print("reference (main's instruction): 2x5 30000-step arm 3.38/10 | "
      "2x10 15000/22500/30000-step arms 4.50/5.00/4.33")
print("targets: 2x5 3-4/10, 2x10 7-8/20")
PY
say "=== balanced pipeline done ==="
