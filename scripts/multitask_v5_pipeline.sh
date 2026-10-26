#!/usr/bin/env bash
# The progress-channel arm: teach the model where it is in the task.
#
# The diagnosis this answers
# -------------------------
# v4 places everything it picks up -- moved-out equals scored in all eight 2x10 episodes --
# but it stops after one batch of five on a task that needs four.  Nothing in its input says
# how far along it is: pi0.5 sees one frame and one instruction per sample, so "the box is
# empty" and "two batches are already in" look identical apart from the picture.
#
# What changes
# ------------
# Two entries are appended to `observation.state`: batches delivered / 4 and cookies placed
# / 20, both mapped into [-1, 1] because pi0.5 discretises the state into 256 bins over that
# range and writes it into the prompt text.  Nothing is projected by a learned layer, so the
# 4.05 checkpoint still loads and the change costs a few tokens.
#
# The count is derived from the effector's own y coordinate rather than the gripper, and
# that choice is measured: the jaw channel is unusable (it sits at 0.31 and only reaches
# 0.49-0.58 on release, so 0.62 missed one of four deliveries and 0.45 counted a median of
# six), while the two boxes are 215 mm apart in y and the hand crosses all of it.  On the v4
# data that rule labels 570 of 570 20-cookie episodes as exactly four deliveries.
#
# The same rule is applied at evaluation time, from the observation the policy already
# receives, so the channel means the same thing on both sides and no privileged state is
# needed.
#
#   bash /root/autodl-tmp/logs/multitask_v5_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-15000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=/root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
SOURCE=/root/autodl-tmp/datasets/a3_multitask_v4
DATA=/root/autodl-tmp/datasets/a3_multitask_v5
OUT=/root/autodl-tmp/pi05-runs/a3_multitask_v5_${STEPS}
SOCK=/tmp/mtv5_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v5.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v5_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== progress-channel pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

[ -e "$DATA/meta/info.json" ] || { say "dataset missing"; exit 1; }
ENTRIES=$("$PY" -c "import json;print(json.load(open('$DATA/meta/info.json'))['features']['observation.state']['shape'][0])")
[ "$ENTRIES" = "18" ] || { say "state is ${ENTRIES}-wide, expected 18"; exit 1; }
say "0/3 dataset: $("$PY" -c "import json;i=json.load(open('$DATA/meta/info.json'));print(i['total_episodes'],'episodes,',i['total_frames'],'frames')"), state ${ENTRIES}-wide"

if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "1/3 train: checkpoint exists, skipping"
else
    say "1/3 train: ${STEPS} steps from 4.05"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    sleep 8
    env A3_PI05_BASE="$BASE" A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_multitask_v5_${STEPS}" \
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
    > /root/autodl-tmp/logs/bridge_mtv5${STEPS}.log 2>&1 < /dev/null &
cd /root/a3_dual_arm_sim
for _ in $(seq 1 180); do
    grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv5${STEPS}.log 2>/dev/null && break
    sleep 5
done
grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv5${STEPS}.log || { say "2/3 bridge never came up"; exit 1; }

# One episode per process: a 2x10 episode at the 3200-step budget costs ~90 minutes under
# load.  No `setsid` here -- the case that made the v4 summary run before its results existed
# was `setsid` detaching the children from the process group that `wait` was watching.
say "3/3 evaluating 2x5 (${EVAL_SEEDS}, 900 steps) + 2x10 (${EVAL_SEEDS}, 3200 steps), progress channels on"
for start in $(seq 0 $(( EVAL_SEEDS - 1 ))); do
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x5 --prompt-style short --progress-channels \
        --episodes 1 --seed-start "$start" --max-steps 900 \
        --output "/root/autodl-tmp/logs/v5_2x5_${start}.json" \
        > "/root/autodl-tmp/logs/v5_2x5_${start}.log" 2>&1 &
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x10 --prompt-style short --progress-channels \
        --episodes 1 --seed-start "$start" --max-steps 3200 \
        --output "/root/autodl-tmp/logs/v5_2x10_${start}.json" \
        > "/root/autodl-tmp/logs/v5_2x10_${start}.log" 2>&1 &
done
wait
say "3/3 evaluation finished"

"$PY" - >> "$LOG" 2>&1 <<'PY'
import glob, json
for row, cap in (("2x5", 10), ("2x10", 20)):
    rows = {}
    for path in sorted(glob.glob(f"/root/autodl-tmp/logs/v5_{row}_*.json")):
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
print("reference (no progress channels): 2x5 v4 3.50/10, 2x10 v4 3.75/20 "
      "(moved-out == scored, i.e. one batch then stop)")
print("targets: 2x5 3-4/10, 2x10 7-8/20")
PY
say "=== progress-channel pipeline done ==="
