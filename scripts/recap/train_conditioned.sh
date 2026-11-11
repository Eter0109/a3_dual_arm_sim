#!/usr/bin/env bash
# Stage 4A: advantage-conditioned fine-tuning by prompt suffix.
#
#   bash scripts/recap/train_conditioned.sh [steps] [lr]
#
# What this run is
# ----------------
# v9, fine-tuned further on the 400-episode RECAP set, where each frame's instruction carries
# `" Advantage: positive"` or `" Advantage: negative"` according to the value model's advantage.
# Inference then asks only the positive form, so the policy is pushed toward the part of its own
# distribution the value function scored as good.
#
# Three settings differ from the v9 recipe, and each has a reason
# --------------------------------------------------------------
# * **lr 1e-5, not 2.5e-5.**  v9 already fits this scene; the point of 4A is to reweight what it
#   already does, not to relearn it.  A rate that re-learns would overwrite the behaviour the
#   advantage labels describe.
# * **10k steps, not 30k.**  The dataset is 552k frames against v9's 312k, so 10k steps already
#   sees more samples per parameter than v9's 30k did.  Longer runs on this data mostly memorise
#   (measured on the value head: train loss kept falling while holdout Spearman peaked at epoch 8
#   of 40).
# * **`--policy.train_expert_only=true`** is inherited unchanged: it is what makes pi0.5 fit a
#   32 GB card, and `train_pi05_a3.sh` already documents that.
#
# The history module is kept, not reset.  The condition is carried by the *sentence*, so nothing
# about the state or history channels changes, and `A3_HISTORY_*` must match what the checkpoint
# was built with or the encoder is silently re-shaped.
set -euo pipefail

STEPS="${1:-10000}"
LR="${2:-1e-5}"

cd "$(dirname "$0")/../.."
PROJECT="$PWD"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
# Overridable because the same recipe now has to serve two different questions: the conditioned
# set (a3_recap_cond) for the advantage experiments, and the filtered plain-BC sets built by
# `subset_dataset.py` (a3_recap_succ / a3_recap_good).  A hard-coded path here means the second
# kind of run silently trains on the first kind's data.
DATA="${A3_RECAP_DATA:-/root/autodl-tmp/datasets/a3_recap_cond}"
BASE=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
# A caller-supplied tag, because two runs of this script with the same step count are *different
# experiments* whenever the labels changed underneath them -- and sharing `OUT` makes the second
# one exit with "checkpoint already exists" after printing a reassuring log, or worse, resume into
# the first one's optimiser state.  `A3_RECAP_TAG=4b` keeps them apart.
TAG="${A3_RECAP_TAG:-4a}"
OUT=/root/autodl-tmp/pi05-runs/a3_recap_${TAG}_${STEPS}
LOG=/root/autodl-tmp/logs/pipeline_recap_${TAG}_${STEPS}.log
TRAIN_LOG=/root/autodl-tmp/logs/train_recap_${TAG}_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }

say "=== RECAP ${TAG} start (${STEPS} steps, lr ${LR}) ==="
say "data   : $DATA"
say "base   : $BASE"
say "output : $OUT"

[ -e "$DATA/meta/info.json" ] || { say "dataset missing: $DATA"; exit 1; }
[ -e "$BASE/model.safetensors" ] || { say "base checkpoint missing: $BASE"; exit 1; }

# A previous run that wrote a checkpoint means the work is done; anything else (a partial
# directory, a crash) should start over rather than resume into an unknown state.
if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "checkpoint already exists, nothing to do"
    exit 0
fi

# Resume rather than restart when a complete checkpoint is on disk.
#
# A checkpoint counts as resumable only if its `training_state/optimizer_state.safetensors` is
# there *and* non-empty: a run killed while saving (which happened -- the disk filled at 7500 and
# `save_file` raised `os error 28`) leaves the model weights behind but not the optimizer state,
# and resuming from that silently restarts the optimizer at step 0 while `training_step.json`
# claims otherwise.
#
# The LR schedule is a cosine over `scheduler_decay_steps`, so a restart from 5000 with the
# original `--scheduler_decay_steps=10000` continues the same curve.  Recomputing it for the
# remaining 5000 steps instead would restart the cosine at its peak and spend the run at a rate
# the first half never saw.
RESUME_DIR=""
for candidate in $(ls -r "$OUT/checkpoints" 2>/dev/null | grep -E '^[0-9]{6}$' || true); do
    state="$OUT/checkpoints/$candidate/training_state"
    if [ -s "$state/optimizer_state.safetensors" ] && [ -s "$state/training_step.json" ]; then
        RESUME_DIR="$OUT/checkpoints/$candidate"
        break
    fi
done

RESUME_ARGS=()
START_STEP=0
if [ -n "$RESUME_DIR" ]; then
    START_STEP=$(python3 -c "import json;print(json.load(open('$RESUME_DIR/training_state/training_step.json'))['step'])")
    say "resuming from $RESUME_DIR (step $START_STEP)"
    # `--policy.path` must be absent here: `TrainPipelineConfig.validate` looks for it first and
    # returns early when it is present, so `checkpoint_path` is never derived from
    # `--config_path` and `load_training_state(None, ...)` restarts the optimizer at step 0 while
    # the run looks like a continuation.  Setting `A3_PI05_BASE` to the empty string is what makes
    # `train_pi05_a3.sh` omit it.
    RESUME_ARGS=(
        --resume=true
        --config_path="$RESUME_DIR/pretrained_model/train_config.json"
    )
    BASE=""
elif [ -d "$OUT" ]; then
    say "removing incomplete output $OUT"
    rm -rf "$OUT"
fi

# Anything that holds the card must go first: the trainer needs ~26 GB of the 32, and a resident
# bridge is 10 GB.  `pkill` on the pattern rather than on a PID because the bridge is started by
# whichever script ran last.
say "freeing the GPU"
pkill -f a3_bridge_server.py 2>/dev/null || true
pkill -f "extract_value_features.py" 2>/dev/null || true
sleep 8

# Disk guard.  A checkpoint is 21 GB, not the 8.8 GB its `model.safetensors` suggests: the
# optimizer state is another 12 GB, and lerobot 0.5.1 has no `save_total_limit` to prune them
# (see `train_pi05_a3.sh`).  That mistake cost a 2.5-hour run at step 7500 to `os error 28`, so
# the old `training_state` directories are now removed from a background loop as soon as a newer
# complete one exists.  Model weights are never touched -- only the state needed to resume, and
# only once it has been superseded.
disk_guard() {
    while true; do
        sleep 120
        local available
        available=$(df --output=avail -BG /root/autodl-tmp | tail -1 | tr -dc '0-9')
        [ -n "$available" ] || continue
        if [ "$available" -lt 30 ]; then
            # Keep the newest resumable state, drop every older one.
            local stale
            stale=$(ls -r "$OUT/checkpoints" 2>/dev/null | grep -E '^[0-9]{6}$' | tail -n +2)
            for step_dir in $stale; do
                if [ -d "$OUT/checkpoints/$step_dir/training_state" ]; then
                    rm -rf "$OUT/checkpoints/$step_dir/training_state"
                    echo "$(date '+%m-%d %H:%M') disk guard: dropped training_state of $step_dir" \
                        >> "$LOG"
                fi
            done
        fi
    done
}
disk_guard &
GUARD=$!
say "disk guard started (pid $GUARD, threshold 30 GB)"

say "training"
# `A3_HISTORY_STEPS=10` and the per-step width are properties of the checkpoint, not knobs: the
# encoder does `view(steps, 48)` and a mismatch is either a shape error or, worse, a silently
# different input.
env A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
    A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS=10 A3_HISTORY_PROGRESS_WEIGHT=0.1 \
    A3_HISTORY_KL_WEIGHT=0.05 A3_HISTORY_DROPOUT=0.1 \
    A3_NEW_MODULE_LR=1e-04 \
    A3_DROP_N_LAST_FRAMES=49 \
    A3_PI05_BASE="$BASE" \
    bash scripts/train_pi05_a3.sh \
      "$DATA" "$OUT" "$STEPS" 6 "a3_recap_${TAG}_${STEPS}" \
      --policy.optimizer_lr="$LR" \
      --policy.scheduler_decay_lr=1e-06 \
      --policy.scheduler_warmup_steps=500 \
      --policy.scheduler_decay_steps="$STEPS" \
      --save_freq=2500 \
      --dataset.video_backend=pyav \
      "${RESUME_ARGS[@]}" \
      > "$TRAIN_LOG" 2>&1
STATUS=$?
kill "$GUARD" 2>/dev/null || true
say "train exit=$STATUS"
[ "$STATUS" -eq 0 ] || { tail -20 "$TRAIN_LOG" | tee -a "$LOG"; exit 1; }

CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
[ -e "$CKPT/model.safetensors" ] || { say "no checkpoint written"; exit 1; }
say "checkpoint: $CKPT"
say "history: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a 'history encoder' | head -1)"
say "=== RECAP ${TAG} done ==="
say ""
say "next: evaluate against v9 on the SAME harness, seeds 300-399"
say "  A3_PI05_BASE=$CKPT bash scripts/recap/eval_conditioned.sh"
