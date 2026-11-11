#!/usr/bin/env bash
# v9 with a 200-frame history window instead of 10.  Everything else held fixed.
#
# The question, and the measurement behind it
# -------------------------------------------
# v9's encoder sees the last 10 frames of `[q, q_dot, a_previous]` -- 0.5 s at 20 Hz.  Measured
# off the v9 evaluation traces, one placement cycle (five cookies picked, carried, released)
# takes a median of **264 frames = 13.2 s**.  A 10-frame window therefore covers 3.8 % of the
# cycle it is supposed to describe: the arm may be reaching down or carrying back, and the
# window cannot tell those apart.
#
# A ridge regression on the progress label, split by episode so consecutive frames cannot leak,
# puts numbers on what is recoverable from the window:
#
#   frames    1     10     20     50    100    200
#   R^2    0.603  0.651  0.691  0.818  0.907  0.966
#
# One frame already carries 0.60 -- the source box is emptied in a consistent order, so the
# pose alone says roughly how far along the arm is.  The recurrence buys the rest, and it is
# still climbing at 200 frames.  200 is the choice here because:
#
#   * 2x5 episodes are 548-550 frames, so 200 is 36 % of an episode; a longer window would make
#     most of a short episode padding;
#   * it covers 0.76 of one placement cycle, i.e. enough to straddle "before" and "after";
#   * R^2 0.966 leaves little of the label unexplained.
#
# What is held constant
# ---------------------
# Base (`lerobot/pi05_base`), episodes (the same 400), step budget (30000 x batch 6), the openpi
# schedule (2.5e-05 peak, 2.5e-06 floor, 1000 warmup, 30000 decay), unfreeze (18 LLM layers +
# vision), state projector, `A3_DROP_N_LAST_FRAMES=49`, and every auxiliary weight
# (progress 0.1, KL 0.05, dropout 0.1).  The parameter count is unchanged at 630018: the GRU
# width does not depend on the window length, only the number of recurrence steps.
#
# `a3_clean_v9h200` was built from `a3_clean_v9` by `add_history_window.py --steps 200`, and
# every column except `observation.history` was verified byte-identical to `a3_clean_v9h`
# (state, velocity, action, progress, eef_pose, indices, timestamps).  That is what makes v9 a
# valid control rather than merely a similar arm: the two checkpoints differ in the window
# length and nothing else.
#
# The dataset had to be written with multiple parquet row groups.  At 200 steps the history
# column is 312470 x 9600 = 3.0e9 values, and HF `datasets` sizes its read batch from the first
# row group, so a single row group makes it read the whole column at once and die with
# `OSError: List index overflow` inside a worker -- surfaced as the far less useful
# `DatasetGenerationError`.  See the guard in `add_history_window.py`.
#
#   bash scripts/multitask_v9w200_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-30000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=lerobot/pi05_base
DATA=/root/autodl-tmp/datasets/a3_clean_v9h200
OUT=/root/autodl-tmp/pi05-runs/a3_clean_v9w200_${STEPS}
LOG=/root/autodl-tmp/logs/pipeline_v9w200.log
TRAIN_LOG=/root/autodl-tmp/logs/train_v9w200_${STEPS}.log
CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
FRAMES=200

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v9 200-frame window pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing: $DATA"; exit 1; }
"$PY" -c "
import json
info = json.load(open('$DATA/meta/info.json'))
width = info['features']['observation.history']['shape'][0]
assert width == $FRAMES * 48, f'history is {width} wide, expected $((FRAMES*48))'
print('dataset ok:', info['total_episodes'], 'episodes,', info['total_frames'],
      'frames, history', width)
" || { say "dataset check failed"; exit 1; }

if [ -e "$CKPT/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from pi05_base, history ${FRAMES} frames"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    sleep 8
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    # Only `A3_HISTORY_STEPS` differs from the v9 command.  A3_PI05_BASE stays unset so
    # `train_pi05_a3.sh` falls back to `lerobot/pi05_base`.
    env A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS="$FRAMES" A3_HISTORY_PROGRESS_WEIGHT=0.1 \
        A3_HISTORY_KL_WEIGHT=0.05 A3_HISTORY_DROPOUT=0.1 \
        A3_NEW_MODULE_LR=1e-04 \
        A3_DROP_N_LAST_FRAMES=49 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_clean_v9w200_${STEPS}" \
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
    # The negative control for *this* arm: the window the encoder is built with has to be the
    # one the dataset carries.  A 10-step encoder against 9600-wide data raises inside the GRU
    # on the first forward, but a silently truncated window would not, so the width is stated.
    "$PY" - "$CKPT" "$FRAMES" <<'PY' | tee -a "$LOG"
import sys
from safetensors import safe_open
with safe_open(sys.argv[1] + "/model.safetensors", framework="pt") as f:
    keys = list(f.keys())
need = [k for k in keys if k.startswith("model.history_encoder.to_mu")]
print(f"1/3 control: {len(keys)} tensors, history keys present = {len([k for k in keys if 'history' in k])}, "
      f"to_mu weight {list(f.get_tensor(need[0]).shape) if need else 'MISSING'} (window {sys.argv[2]} frames)")
PY
fi

say "2/3 latent check on $(basename "$OUT") (window ${FRAMES})"
A3_HISTORY_STEPS="$FRAMES" "$PY" scripts/check_history_latent.py \
    --checkpoint "$CKPT" --data "$DATA" --limit 40000 2>&1 \
    | tee -a "$LOG" | grep -aE "R\^2|corr|latent:|VERDICT"

# The window length is a property of the dataset, so the adapter needs 9600 and the driver
# needs 200 frames.  Both are derived from FRAMES rather than written twice.
say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS}), window ${FRAMES}"
HISTORY_FRAMES="$FRAMES" bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" \
    --history --prompt-style task --source-column 1 --tag v9w200 2>&1 | tee -a "$LOG"

say "=== v9 200-frame pipeline done ==="
