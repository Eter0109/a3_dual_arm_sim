#!/usr/bin/env bash
# Ablation on v9: same recipe, same data, same budget -- history latent removed.
#
# The question
# ------------
# v9 (clean base, 30000 steps, openpi's own schedule) is the first arm to clear both
# targets: 2x5 5.12 / 2x10 13.62 on the aligned protocol, with a history latent that
# reaches R^2 0.853 on task progress.  A latent that fits progress is not the same claim as
# a latent that *helps control* -- it could be a well-trained passenger.  The only way to
# tell is to remove it and change nothing else.
#
# What is held constant, deliberately
# ----------------------------------
#   * base        : `lerobot/pi05_base`, untouched release
#   * dataset     : `a3_clean_v9h`, same 400 episodes, same order, same video files
#                   (it is hard-linked to `a3_clean_v9`, so nothing is duplicated)
#   * steps       : 30000, batch 6  -> 180000 frames, the same gradient budget
#   * schedule    : openpi defaults -- peak 2.5e-05, floor 2.5e-06, warmup 1000,
#                   decay 30000, so the LR curve is point-for-point the same
#   * unfreeze    : 18 LLM layers + vision
#   * state proj  : ON.  This ablation removes the *history latent* only; the state
#                   projector is a different module and stays, otherwise the comparison
#                   would move two things at once
#   * lr for new  : 1e-04, the same rate the fresh parameters got in v9
#   * drop tails  : 49 frames
#
# What is different
# -----------------
#   A3_HISTORY_ENCODER=0.  `_patch_history_encoder` prints "history encoder    : off" and
#   returns before injecting anything, so the model has no `history_encoder`, no
#   `history_proj`, no extra prefix token and no auxiliary progress/KL loss.  The
#   `observation.history` column in the dataset is simply never read -- using the same
#   files rather than rebuilding a no-history copy is what keeps the episode order and the
#   frame sampling identical.
#
# Reading the result
# ------------------
# Evaluation uses the protocol that was verified against `randomization.task_instruction`
# in the recording run: `--prompt-style task --source-column 1` on 2x5 (the flag that
# actually selects main's two-batches sentence is `source_column`, not `prompt_style`), and
# no `--history` on either task.  Compare against those same 8 seeds:
#
#   2x5   v9 with latent  5.12 +- 0.64      cap 10
#   2x10  v9 with latent 13.62 +- 1.78      cap 20
#
#   bash scripts/multitask_v9_nohist_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-30000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=lerobot/pi05_base
DATA=/root/autodl-tmp/datasets/a3_clean_v9h
OUT=/root/autodl-tmp/pi05-runs/a3_clean_v9nohist_${STEPS}
TAG="v9nh"
LOG=/root/autodl-tmp/logs/pipeline_v9_nohist.log
TRAIN_LOG=/root/autodl-tmp/logs/train_v9_nohist_${STEPS}.log
CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v9 no-history ablation start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing: $DATA"; exit 1; }
"$PY" -c "
import json
info = json.load(open('$DATA/meta/info.json'))
print('dataset ok:', info['total_episodes'], 'episodes,', info['total_frames'], 'frames')
" || { say "dataset check failed"; exit 1; }

if [ -e "$CKPT/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from pi05_base, history latent OFF"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    sleep 8
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    # A3_HISTORY_* weights/steps are omitted on purpose: with the encoder off they are
    # never read, and leaving them out of the command makes the omitted module visible in
    # the log instead of buried among settings that do nothing.
    env A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=0 \
        A3_NEW_MODULE_LR=1e-04 \
        A3_DROP_N_LAST_FRAMES=49 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_clean_v9nohist_${STEPS}" \
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
    say "1/3 off:     $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a 'history encoder' | head -1)"
    say "1/3 base:    $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a '  base ' | tail -1)"
    say "1/3 modules: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -aE 'new-module lr' | tail -1 | cut -c1-150)"
    # The negative control: if these keys are present, the switch did not take and the arm
    # is a second copy of v9 rather than an ablation.
    "$PY" - "$CKPT" <<'PY' | tee -a "$LOG"
import sys
from safetensors import safe_open
with safe_open(sys.argv[1] + "/model.safetensors", framework="pt") as f:
    keys = list(f.keys())
bad = [k for k in keys if "history_encoder" in k or "history_proj" in k]
print(f"1/3 control: {len(keys)} tensors, history keys present = {len(bad)} "
      f"({'OK, ablation' if not bad else 'FAILED, switch ignored'})")
PY
fi

say "2/3 latent check: skipped (no history encoder in this arm)"
say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS}), aligned protocol"
# No --history: the adapter would otherwise build a 480-float window and try to feed a
# checkpoint that has no encoder to consume it.
bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" \
    --prompt-style task --source-column 1 --tag "$TAG" 2>&1 | tee -a "$LOG"

say "=== v9 no-history ablation done ==="
