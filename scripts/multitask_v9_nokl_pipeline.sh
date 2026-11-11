#!/usr/bin/env bash
# v9 with the KL term removed: the latent keeps its progress supervision but loses its
# capacity constraint.
#
# The question
# ------------
# 9.52 measured v9 (10 frames + progress supervision, KL 0.05) against no encoder at all and
# found +2.46 on 2x10.  9.53 found that widening the window to 200 frames *lowered* 2x10 by
# 2.17 while raising the latent's R^2 from 0.853 to 0.942.  9.54 found that deleting the
# progress term collapses the latent outright (mu_std 0.0576 -> 0.0017, KL -> 0) and drops
# 2x10 to the no-encoder level.
#
# Across those three, one term has never been varied: the KL.  In 200-frame it compresses a
# 20x larger input into the same 128 dimensions, which is the natural suspect for why more
# information upstream produced less useful information downstream.  Removing KL while keeping
# the progress term tests that directly:
#
#   * progress stays at 0.1, so the latent has a stated job and cannot collapse the way
#     9.54's arm did -- that is why this experiment is possible at all.
#   * nothing pushes mu toward 0, so the latent is free to widen.
#
# The risk, and what to watch
# ---------------------------
# `mu` reaches the network only as `history_proj(mu) * sqrt(2048)`, and with no KL there is
# nothing bounding it.  The flow loss has a gradient incentive to make a prefix token large
# (attention reads magnitude), so mu_std can run away rather than collapse.  Both directions
# are failures and they look nothing alike in the score: a collapsed latent degenerates to
# "no encoder" (~10.76), a runaway token swamps the image and language tokens and would show up
# as a policy that ignores instruction and vision -- i.e. near-zero on both tasks.
#
# The training log prints `mu_std` and the prefix token std every 250 steps.  Reference values:
#
#   v9            mu_std 0.0576   token std 3.4335   (image tokens ~4.2, language ~9.6)
#   noprog        mu_std 0.0017   (collapsed)
#
# so the healthy band is roughly 0.02-0.3 for mu_std and 2-12 for the token.  A token std far
# above the language tokens is the runaway case.
#
# `to_logvar` in this arm
# ----------------------
# With KL weighted 0, nothing differentiates `to_logvar`; it stays at its init.  That is
# harmless because `latent()` returns `mu` unless sampling is on (`A3_HISTORY_SAMPLE` defaults
# to "0"), so logvar never reaches the forward pass.  It does mean this arm trains 16512 fewer
# parameters than v9 -- noted so the parameter counts are not compared naively.
#
# Everything else is v9: same base, same 400 episodes, same 30000 x 6 budget, same openpi
# schedule, same unfreeze, same state projector, same dropout, same progress weight.
#
#   bash scripts/multitask_v9_nokl_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-30000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=lerobot/pi05_base
DATA=/root/autodl-tmp/datasets/a3_clean_v9h
OUT=/root/autodl-tmp/pi05-runs/a3_clean_v9nokl_${STEPS}
LOG=/root/autodl-tmp/logs/pipeline_v9nokl.log
TRAIN_LOG=/root/autodl-tmp/logs/train_v9nokl_${STEPS}.log
LATENT_LOG=/root/autodl-tmp/logs/latent_scale_v9nokl_${STEPS}.log
CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== v9 no-KL pipeline start (${STEPS} steps) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing: $DATA"; exit 1; }

if [ -e "$CKPT/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from pi05_base, progress=0.1, kl=0"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    sleep 8
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    # Only A3_HISTORY_KL_WEIGHT differs from the v9 command.
    env A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS=10 A3_HISTORY_PROGRESS_WEIGHT=0.1 \
        A3_HISTORY_KL_WEIGHT=0 A3_HISTORY_DROPOUT=0.1 \
        A3_NEW_MODULE_LR=1e-04 \
        A3_DROP_N_LAST_FRAMES=49 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_clean_v9nokl_${STEPS}" \
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

# Latent-scale check.  Both failure directions live here: mu_std collapsing (9.54's failure) or
# the prefix token outgrowing the language tokens and drowning them out.
#
# No `^` anchor: the trainer writes its progress bar with `\r` and no newline, so `[history]`
# records end up appended after `Training: ...` rather than starting a line.  Anchoring to the
# line start returns only the first record, which is the one from before any training -- a
# flattening latent then looks like a flat line.  Same bug as 9.55.
say "2/3 latent scale trace (mu_std healthy 0.02-0.3; token std healthy 2-12)"
tr '\r' '\n' < "$TRAIN_LOG" | grep -a '\[history\] step' \
    | sed -E 's/.*\[history\] step +([0-9]+).*mu_std=([0-9.]+).*/\1 \2/' > "$LATENT_LOG"
"$PY" - "$TRAIN_LOG" "$LATENT_LOG" <<'PY' | tee -a "$LOG"
import re, sys

train_log, mu_log = sys.argv[1], sys.argv[2]
text = open(train_log, errors="ignore").read().replace("\r", "\n")
rows = []
for line in text.splitlines():
    m = re.search(r"\[history\] step +(\d+).*mu_std=([0-9.]+)", line)
    if not m:
        continue
    tok = re.search(r"prefix token.*?: std ([0-9.]+)", line) or re.search(r"std ([0-9.]+)\s*\(image", line)
    rows.append((int(m.group(1)), float(m.group(2))))

if not rows:
    print("2/3 latent scale: no rows parsed")
    raise SystemExit(0)

first, last = rows[0], rows[-1]
tail = [v for _, v in rows[-5:]]
mu_last = last[1]
mu_max = max(v for _, v in rows)
verdict = "COLLAPSED" if mu_last < 0.005 else ("RUNAWAY" if mu_last > 1.0 else "alive")
print(f"2/3 mu_std: {len(rows)} samples")
print(f"    first {first[1]:.4f} @step {first[0]}   max {mu_max:.4f}   last {mu_last:.4f} @step {last[0]}")
print(f"    tail (last 5): {[round(v, 4) for v in tail]}")
print(f"    v9 reference 0.0576, noprog 0.0017")
print(f"2/3 VERDICT: {verdict}")
PY

say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS})"
bash scripts/bench_arm.sh "$CKPT" "$EVAL_SEEDS" \
    --history --prompt-style task --source-column 1 --tag v9nokl 2>&1 | tee -a "$LOG"

say "=== v9 no-KL pipeline done ==="
