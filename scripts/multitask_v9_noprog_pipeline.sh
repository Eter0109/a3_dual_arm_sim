#!/usr/bin/env bash
# v9 with the progress head switched off: the latent stays, its supervision does not.
#
# The question
# ------------
# 9.53 left a contradiction: a 200-frame window raised the latent's R^2 on task progress from
# 0.853 to 0.942 and made 2x10 *worse* (13.22 -> 11.05, paired p = 0.0079).  One explanation is
# that the auxiliary progress loss is steering the latent toward information only the auxiliary
# head reads -- "how many batches are done" -- and away from the phase information the action
# expert needs.  If that is true, removing the supervision while keeping the channel should
# recover, or beat, the 10-frame arm's 13.22.
#
# `A3_HISTORY_PROGRESS_WEIGHT=0` does exactly one thing, and it is worth being precise about
# what it does *not* do:
#
#   loss = flow + progress_weight * progress_mse + kl_weight * kl
#
#   * flow loss, the KL term (0.05), the GRU, `to_mu`, `to_logvar` and `history_proj` are all
#     unchanged and all still receive gradients -- verified by hand, not assumed:
#     `progress.weight` grad becomes exactly 0, `to_mu` 3.3e-05, `gru.to_gates` 5.7e-05.
#   * the `progress` head (258 params) keeps running and keeps printing its MSE, but its
#     gradient is zero, so it stays at its `std=0.02` initialisation forever.  Its logged loss
#     is therefore *not* evidence of anything in this arm.
#   * the history token is still built and still concatenated into the prefix.
#
# So this is "latent without a stated job", not "no latent".
#
# What this arm cannot measure
# ----------------------------
# `check_history_latent.py` computes R^2 *through that head*.  With the head untrained its
# output is a fixed random projection and R^2 will sit near 0 whether or not the latent is
# informative.  The latent diagnostic is therefore void for this arm and the step is skipped
# rather than reported as a failure -- reporting "VERDICT: the latent does not carry progress"
# here would be measuring the untrained head, which is exactly the class of mistake 9.52 and
# 9.53 were about.
#
# The risk this arm *does* carry: with the progress term gone, only KL(0.05) constrains `mu`,
# and nothing pushes it away from 0.  If the latent collapses the arm degenerates into the
# no-history baseline wearing a dead token, and a low score would mean "collapsed", not
# "the latent is useless".  The training log prints `mu_std` every 250 steps
# (`[history] step N ... mu_std=x`), and v9 held 0.0576 with the head on, so anything an order
# of magnitude below that is a collapse and the run should be read as such.
#
# Everything else is v9: same base, same 400 episodes, same 30000 x 6 budget, same openpi
# schedule, same unfreeze, same state projector, same dropout, same KL weight.
#
#   bash scripts/multitask_v9_noprog_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-30000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=lerobot/pi05_base
DATA=/root/autodl-tmp/datasets/a3_clean_v9h
OUT=/root/autodl-tmp/pi05-runs/a3_clean_v9noprog_${STEPS}
LOG=/root/autodl-tmp/logs/pipeline_v9noprog.log
TRAIN_LOG=/root/autodl-tmp/logs/train_v9noprog_${STEPS}.log
MU_LOG=/root/autodl-tmp/logs/mu_std_v9noprog_${STEPS}.log
CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v9 no-progress-supervision pipeline start (${STEPS} steps) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing: $DATA"; exit 1; }

if [ -e "$CKPT/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from pi05_base, progress_weight=0, kl_weight=0.05"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    sleep 8
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    # Only A3_HISTORY_PROGRESS_WEIGHT differs from the v9 command.
    env A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS=10 A3_HISTORY_PROGRESS_WEIGHT=0 \
        A3_HISTORY_KL_WEIGHT=0.05 A3_HISTORY_DROPOUT=0.1 \
        A3_NEW_MODULE_LR=1e-04 \
        A3_DROP_N_LAST_FRAMES=49 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_clean_v9noprog_${STEPS}" \
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
    say "1/3 history: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -a 'history encoder' | head -1)"
    say "1/3 modules: $(tr '\r' '\n' < "$TRAIN_LOG" | grep -aE 'new-module lr' | tail -1 | cut -c1-150)"
fi

# The collapse check.  v9 held mu_std = 0.0576 with the head on; this arm has only KL holding
# the latent, so the per-250-step trace is what separates "collapsed" from "informative".
#
# No `^` anchor: the trainer writes its progress bar with `\r` and no newline, so these lines
# end up appended *after* `Training: ...` rather than at the start of one.  Anchoring to the
# line start silently returns only the first record -- which is exactly the record that
# predates any training, so a collapse over the next 30000 steps would look like a flat line.
say "2/3 mu_std trace (collapse check; v9 reference 0.0576)"
tr '\r' '\n' < "$TRAIN_LOG" | grep -a '\[history\] step' \
    | sed -E 's/.*\[history\] step +([0-9]+).*mu_std=([0-9.]+).*/\1 \2/' > "$MU_LOG"
"$PY" - "$MU_LOG" <<'PY' | tee -a "$LOG"
import sys
rows = []
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) == 2:
        try:
            rows.append((int(parts[0]), float(parts[1])))
        except ValueError:
            pass
if not rows:
    print("2/3 mu_std: no rows (the log format changed?)")
    raise SystemExit(0)
first, last = rows[0], rows[-1]
tail = [v for _, v in rows[-6:]]
print(f"2/3 mu_std: {len(rows)} samples, first {first[1]:.4f} @step {first[0]}, "
      f"last {last[1]:.4f} @step {last[0]}")
print(f"2/3 mu_std tail (last 6): {[round(v, 4) for v in tail]}")
print(f"2/3 VERDICT: {'COLLAPSED' if last[1] < 0.005 else 'alive'}"
      f" (v9 reference 0.0576, collapse threshold 0.005)")
PY

say "2/3 latent check: SKIPPED -- its R^2 is computed through the progress head, which this"
say "    arm never trains.  A number from it would measure the head, not the latent."
say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS})"
bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" \
    --history --prompt-style task --source-column 1 --tag v9np 2>&1 | tee -a "$LOG"

say "=== v9 no-progress pipeline done ==="
