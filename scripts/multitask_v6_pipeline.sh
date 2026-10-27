#!/usr/bin/env bash
# The history-encoder arm: a latent of the robot's own recent trajectory.
#
# v4 places everything it picks up and stops after one batch of five on a task that needs
# four (9.43).  v5's two progress scalars moved 15 Cookies in one seed -- proof the
# information exists -- but they were privileged and their attribution was muddy.
#
# This arm replaces them with a genuine hidden state, addressing both problems at once:
#
#   * the encoder sees NO IMAGES.  Its input is the last 10 frames of [q, q_dot, a_prev],
#     so the bottleneck is the input modality rather than a loss term: it cannot represent
#     "the Cookies look messy", only "where the arm has been and what it did", which is
#     where task progress actually lives;
#   * progress is supervised (auxiliary head, discarded at inference) and the latent is
#     compressed toward N(0, I) by a KL term, so it cannot spend capacity on anything the
#     flow loss does not need.
#
# No DAgger pass: a history *window* reads the actual past at train and test time, so there
# is no predicted-past to drift; and the training mix already contains the model's own
# rollouts (the grafted episodes' first phase, 9.40).  `A3_HISTORY_DROPOUT=0.1` truncates
# windows at random, which is the cheap remaining piece.
#
#   bash /root/autodl-tmp/logs/multitask_v6_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-15000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=/root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
DATA=/root/autodl-tmp/datasets/a3_multitask_v6
OUT=/root/autodl-tmp/pi05-runs/a3_multitask_v6_${STEPS}
SOCK=/tmp/mtv6_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v6.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v6_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== history-encoder pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing"; exit 1; }
"$PY" -c "
import json
i = json.load(open('$DATA/meta/info.json'))
assert 'observation.history' in i['features'], 'dataset has no history feature'
assert i['features']['observation.history']['shape'][0] == 480, 'unexpected window width'
print('dataset ok:', i['total_episodes'], 'episodes,', i['total_frames'], 'frames,',
      i['features']['observation.history']['shape'][0], '-wide window')
" || { say "dataset check failed"; exit 1; }

if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from 4.05, history encoder on"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    pkill -f "prune_checkpoints.py $OUT" 2>/dev/null
    # 21 GB per checkpoint and no `save_total_limit` in this lerobot: at a 1500-step
    # interval the run wants 210 GB out of a ~130 GB gap, so a watcher keeps the earliest
    # checkpoint (the anchor the auxiliary-loss curve starts from) and the newest two.
    setsid nohup "$PY" scripts/prune_checkpoints.py "$OUT" \
        > /root/autodl-tmp/logs/prune_$(basename "$OUT").log 2>&1 < /dev/null &
    sleep 8
    env A3_PI05_BASE="$BASE" A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        A3_HISTORY_ENCODER=1 A3_HISTORY_STEPS=10 A3_HISTORY_PROGRESS_WEIGHT=0.1 \
        A3_HISTORY_KL_WEIGHT=0.05 A3_HISTORY_DROPOUT=0.1 \
        A3_NEW_MODULE_LR=1e-04 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_multitask_v6_${STEPS}" \
          --policy.optimizer_lr=3e-06 \
          --policy.scheduler_decay_lr=2e-07 \
          --policy.scheduler_warmup_steps=$(( STEPS / 30 )) \
          --policy.scheduler_decay_steps="$STEPS" \
          --save_freq="${SAVE_FREQ:-1500}" \
          --dataset.video_backend=pyav \
          > "$TRAIN_LOG" 2>&1
    STATUS=$?
    say "1/3 train exit=$STATUS $(grep -ao 'Training: 100%[^|]*' "$TRAIN_LOG" | tail -1)"
    [ "$STATUS" -eq 0 ] || exit 1
    say "1/3 history terms: $(grep -a '^\[history\]' "$TRAIN_LOG" | tail -1 | cut -c1-150)"
fi

CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
say "2/3 serving $(basename "$OUT")"
pkill -f a3_bridge_server.py 2>/dev/null
sleep 5
cd /root/autodl-tmp/a3-collect
setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
    --adapter adapters/pi05_a3.py --listen "unix://$SOCK" --config "ckpt=$CKPT" \
    --config stats_root="$DATA" --config state_dims=16 --config history_dims=480 \
    > /root/autodl-tmp/logs/bridge_mtv6${STEPS}.log 2>&1 < /dev/null &
cd /root/a3_dual_arm_sim
for _ in $(seq 1 180); do
    grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv6${STEPS}.log 2>/dev/null && break
    sleep 5
done
grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv6${STEPS}.log || { say "2/3 bridge never came up"; exit 1; }

say "3/3 evaluating 2x5 (${EVAL_SEEDS}) + 2x10 (${EVAL_SEEDS}), history channels on"
for start in $(seq 0 $(( EVAL_SEEDS - 1 ))); do
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x5 --prompt-style short --history-window \
        --episodes 1 --seed-start "$start" --max-steps 900 \
        --output "/root/autodl-tmp/logs/v6_2x5_${start}.json" \
        > "/root/autodl-tmp/logs/v6_2x5_${start}.log" 2>&1 &
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x10 --prompt-style short --history-window \
        --episodes 1 --seed-start "$start" --max-steps 3200 \
        --output "/root/autodl-tmp/logs/v6_2x10_${start}.json" \
        > "/root/autodl-tmp/logs/v6_2x10_${start}.log" 2>&1 &
done
wait
say "3/3 evaluation finished"

"$PY" - >> "$LOG" 2>&1 <<'PY'
import glob, json
for row, cap in (("2x5", 10), ("2x10", 20)):
    rows = {}
    for path in sorted(glob.glob(f"/root/autodl-tmp/logs/v6_{row}_*.json")):
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
print("reference (no history): 2x5 v4 3.50/10, 2x10 v4 3.75/20 (one batch then stop)")
print("v5 probe (privileged scalars): 2x5 3.88, 2x10 3.12, moved out 15 once")
print("targets: 2x5 3-4/10, 2x10 7-8/20")
PY
say "=== history-encoder pipeline done ==="
