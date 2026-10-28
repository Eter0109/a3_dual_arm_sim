#!/usr/bin/env bash
# v8: teach the model *how much* to move, by restoring the 2x10 plan prompt.
#
# Where the line stands entering this run
# --------------------------------------
# v7 (500 contradictory 2x5 episodes dropped at the sampler) reached 6.25/10 on 2x5 --
# past the 3-4 acceptance target -- but fell to 2.50/20 on 2x10, and the traces say why
# it stops: it delivers exactly one batch, or two.  Measured directly on v7, same scene,
# 400 steps, one token changed:
#
#     2x10 + "transfer 20 cookies"    moved out 10, delivered 5
#     2x10 + "transfer 10 cookies"    moved out  5, delivered 5
#
# so the instruction *is* read and it *is* worth a factor of two in effort.  What it
# cannot express is the rest of the schedule.  The two tasks share almost every visual
# state -- ten Cookies in the box ends a 2x5 task and is halfway through a 2x10 one --
# and the only thing separating them was `10` versus `20` in a sentence with no plan.
# At batch 6 and 15000 steps (8.8% of the set) that is too fine a condition to learn, so
# the model defaulted to the far more common stopping pattern.
#
# What changes, and what does not
# -------------------------------
# The 2x10 episodes go back to the sentence they were collected under:
#
#     Transfer 20 cookies from source column 1 into the 2x10 target box.
#     Grasp 5 cookies first, then 5 to fill target column 1 with 10.
#     Repeat: grasp 5, then 5 to fill target column 2 with 10.
#     Release all 20 cookies upright in the target box and leave 60 in the source box.
#
# which names the batch schedule, the two columns, and the 60 Cookies that must stay
# behind.  Ten variants exist (n = 1..9 plus a skip-grasp plan) and all ten are kept,
# because `--prompt-style task` reproduces whichever one the seed draws -- so training
# and evaluation agree by construction rather than by our choosing a sentence.
#
# The 2x5 side uses main's own one-liner, which is structurally unlike any 2x10 plan:
#
#     Transfer 10 cookies from source column 1 into the target box in two batches of five.
#
# so the two tasks differ in length, vocabulary and structure rather than by a digit.
#
# The 2x10 data also changes: `a3_2x10_v1` (280 episodes, ten plans) was **never used**
# by v4-v7, which trained on 190 unique episodes repeated three times.  This arm uses all
# 440 unique episodes once.  That is a second variable and it points the same way -- the
# memorisation this arm is trying to break is a function of seeing 190 layouts 3x each.
#
#   bash scripts/multitask_v8_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-15000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=/root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
DATA=/root/autodl-tmp/datasets/a3_multitask_v8h
OUT=/root/autodl-tmp/pi05-runs/a3_multitask_v8_${STEPS}
SOCK=/tmp/mtv8_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v8.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v8_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v8 pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing"; exit 1; }
"$PY" -c "
import json
info = json.load(open('$DATA/meta/info.json'))
assert 'observation.history' in info['features'], 'v8 dataset has no history feature'
print('dataset ok:', info['total_episodes'], 'episodes,', info['total_frames'], 'frames')
" || { say "dataset check failed"; exit 1; }

if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from 4.05, plan prompts, short episodes excluded"
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
          "$DATA" "$OUT" "$STEPS" 6 "a3_multitask_v8_${STEPS}" \
          --policy.optimizer_lr=3e-06 \
          --policy.scheduler_decay_lr=2e-07 \
          --policy.scheduler_warmup_steps=$(( STEPS / 30 )) \
          --policy.scheduler_decay_steps="$STEPS" \
          --save_freq=1500 \
          --dataset.video_backend=pyav \
          > "$TRAIN_LOG" 2>&1
    STATUS=$?
    say "1/3 train exit=$STATUS"
    [ "$STATUS" -eq 0 ] || exit 1
    say "1/3 filter:  $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a 'episode filter' | tail -1)"
    say "1/3 history: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a '^\[history\]' | tail -1 | cut -c1-150)"
fi

CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
say "2/3 latent check on $(basename "$OUT")"
"$PY" scripts/check_history_latent.py --checkpoint "$CKPT" --data "$DATA" --limit 40000 2>&1 \
    | tee -a "$LOG" | grep -aE "R\^2|corr|latent:|VERDICT"

# `--prompt-style task`: 2x5 gets main's two-batches-of-five sentence and 2x10 gets the
# per-seed plan, which is what the checkpoint was trained on.  `short` here would measure
# a sentence the model never saw.
say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS}), plan prompts"
bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" --history --prompt-style task --tag v8 2>&1 | tee -a "$LOG"

say "=== v8 pipeline done ==="
