#!/usr/bin/env bash
# v7: the same data and the same hyper-parameters as v6, minus the episodes that lie about
# their own instruction.
#
# v6 scored 4.50 on 2x5 and 4.50 on 2x10.  Its traces (9.46) show five of eight 2x5 seeds
# either stopping after one batch or freezing with the second batch in the jaw -- and the
# data says why: 41.7% of the 2x5 episodes in `a3_multitask_v6` are 302 frames long and end
# with one batch delivered, under an instruction that asks for ten cookies on all 1200.
#
# The 570 2x10 episodes are all complete (1043-1095 frames, four batches), so the same
# contamination is a plausible explanation for 2x10's "five cookies then stop" as well: a
# model taught by 500 episodes that "five in the box ends the task" carries that prior into
# the other task.
#
# Nothing about the dataset changes on disk.  `A3_MIN_EPISODE_FRAMES=400` installs a filter
# inside `EpisodeAwareSampler`, and episode length separates the two populations exactly
# (302-303 against 506-1176), so this is a sampler setting rather than a 12 GB re-encode.
#
#   bash scripts/multitask_v7_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-15000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=/root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
DATA=/root/autodl-tmp/datasets/a3_multitask_v6
OUT=/root/autodl-tmp/pi05-runs/a3_multitask_v7_${STEPS}
SOCK=/tmp/mtv7_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v7.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v7_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v7 pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing"; exit 1; }

if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from 4.05, history encoder on, short episodes excluded"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    sleep 8
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    env A3_PI05_BASE="$BASE" A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS=10 A3_HISTORY_PROGRESS_WEIGHT=0.1 \
        A3_HISTORY_KL_WEIGHT=0.05 A3_HISTORY_DROPOUT=0.1 \
        A3_NEW_MODULE_LR=1e-04 \
        A3_MIN_EPISODE_FRAMES=400 \
        A3_DROP_N_LAST_FRAMES=49 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_multitask_v7_${STEPS}" \
          --policy.optimizer_lr=3e-06 \
          --policy.scheduler_decay_lr=2e-07 \
          --policy.scheduler_warmup_steps=$(( STEPS / 30 )) \
          --policy.scheduler_decay_steps="$STEPS" \
          --save_freq=1500 \
          --dataset.video_backend=pyav \
          > "$TRAIN_LOG" 2>&1
    STATUS=$?
    say "1/3 train exit=$STATUS $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a 'Training: 100%' | tail -1 | cut -c1-40)"
    [ "$STATUS" -eq 0 ] || exit 1
    say "1/3 filter:    $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a 'episode filter' | tail -1)"
    say "1/3 history:   $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a '^\[history\]' | tail -1 | cut -c1-150)"
fi

CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
say "2/3 latent check on $(basename "$OUT")"
"$PY" scripts/check_history_latent.py --checkpoint "$CKPT" --data "$DATA" --limit 40000 2>&1 \
    | tee -a "$LOG" | grep -aE "R\^2|corr|latent:|VERDICT"

say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS})"
bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" --history --tag v7 2>&1 | tee -a "$LOG"

say "=== v7 pipeline done ==="
