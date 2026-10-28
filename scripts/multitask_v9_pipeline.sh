#!/usr/bin/env bash
# v9: a clean base, trained the way openpi says to, from the untouched pi0.5 weights.
#
# Why start over
# --------------
# Nine arms were fine-tuned from `a3_b2_aligned_v3_4000` ("4.05"), and 4.05 itself came from
# 4000 steps over a set it had seen 5.6 % of -- roughly 56 episodes' worth.  Each arm then
# had 15000 steps (10 % of its own set) to learn two tasks on top of that, and every one of
# them equilibrated to a trade: v7 kept 2x5 at 6.25 and lost 2x10 to 2.50, v8 did the
# reverse.  None of them has been a base anyone could add to, because each one is already
# bent toward a task.
#
# The other reason is data hygiene.  Every arm inherited the whole of
# `a3_b2_main_aligned_v2_merged`, whose 1200 2x5 episodes include 500 that deliver one
# batch under an instruction asking for two (9.48), and every arm up to v7 trained on 190
# unique 2x10 episodes repeated three times.  `a3_clean_v9` is free of both.
#
# `a3_clean_v9` (built by `build_clean_v9.py`), 400 episodes:
#
#   * 200 2x5 -- whole two-batch demonstrations only, instruction = main's own sentence
#   * 200 2x10 -- 20 per plan prompt, all ten plans equally represented
#   * expert only -- no `a3_graft_v1_merged`, whose first half is policy rollouts
#
# The two tasks are 50 : 50 by episode.  By frame they are 32 : 68, because a 2x10 episode
# is twice as long; that is unavoidable if the counts are to be equal, and it is the
# counts that were asked for.
#
# The recipe is openpi's, unmodified
# ---------------------------------
# `configuration_pi05.py` defaults are peak 2.5e-05, floor 2.5e-06, warmup 1000, decay
# 30000 -- and 30000 is also the step budget here, so `--policy.scheduler_decay_steps`
# needs no auto-scaling and the run is the shape openpi intends.  Nothing is tuned: the
# point of this arm is to be a *reference*, so the only departures from stock are the two
# modules this line adds (state projector, history encoder) and the batch size the card
# allows.
#
# Budget arithmetic, so the reading is unambiguous:
#
#   30000 steps x 6 = 180000 frames seen
#   dataset frames (after `--drop-n-last-frames 49`): ~280000
#   -> ~64 % of one epoch, ~230 episodes' worth of gradients
#
#   bash scripts/multitask_v9_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-30000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
# The untouched release.  Not a local checkpoint: this arm exists to find out what the
# stacked fine-tunes cost, which needs the weights nobody has touched.
BASE=lerobot/pi05_base
DATA=/root/autodl-tmp/datasets/a3_clean_v9h
OUT=/root/autodl-tmp/pi05-runs/a3_clean_v9_${STEPS}
SOCK=/tmp/mtv9_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v9.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v9_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v9 clean-base pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing: $DATA"; exit 1; }
"$PY" -c "
import json, pyarrow.parquet as pq
info = json.load(open('$DATA/meta/info.json'))
assert 'observation.history' in info['features'], 'v9 dataset has no history feature'
tasks = pq.read_table('$DATA/meta/tasks.parquet').to_pylist()
print('dataset ok:', info['total_episodes'], 'episodes,', info['total_frames'],
      'frames,', len(tasks), 'instructions')
" || { say "dataset check failed"; exit 1; }

if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from pi05_base, openpi recipe, clean data"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    sleep 8
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    # A3_PI05_BASE is deliberately not set: `train_pi05_a3.sh` falls back to
    # `lerobot/pi05_base`, which is the whole point of this arm.
    env A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS=10 A3_HISTORY_PROGRESS_WEIGHT=0.1 \
        A3_HISTORY_KL_WEIGHT=0.05 A3_HISTORY_DROPOUT=0.1 \
        A3_NEW_MODULE_LR=1e-04 \
        A3_DROP_N_LAST_FRAMES=49 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_clean_v9_${STEPS}" \
          --policy.optimizer_lr=2.5e-05 \
          --policy.scheduler_decay_lr=2.5e-06 \
          --policy.scheduler_warmup_steps=1000 \
          --policy.scheduler_decay_steps="$STEPS" \
          --save_freq=3000 \
          --dataset.video_backend=pyav \
          > "$TRAIN_LOG" 2>&1
    STATUS=$?
    say "1/3 train exit=$STATUS"
    [ "$STATUS" -eq 0 ] || exit 1
    say "1/3 base:    $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a '  base ' | tail -1)"
    say "1/3 modules: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -aE 'new-module lr' | tail -1 | cut -c1-150)"
    say "1/3 history: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a '^\[history\]' | tail -1 | cut -c1-150)"
fi

CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
say "2/3 latent check on $(basename "$OUT")"
"$PY" scripts/check_history_latent.py --checkpoint "$CKPT" --data "$DATA" --limit 40000 2>&1 \
    | tee -a "$LOG" | grep -aE "R\^2|corr|latent:|VERDICT"

# `--prompt-style task`: 2x5 gets main's two-batches-of-five sentence, 2x10 gets the
# per-seed plan, which is exactly what this dataset was labelled with.
say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS}), main's prompts"
bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" --history --prompt-style task --tag v9 2>&1 \
    | tee -a "$LOG"

say "=== v9 pipeline done ==="
