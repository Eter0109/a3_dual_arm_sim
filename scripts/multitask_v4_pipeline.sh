#!/usr/bin/env bash
# The single-plan arm: one instruction, one action pattern, four batches of five.
#
# Why this shape
# --------------
# Two failures decided it.  First, main's 2x10 plan prompt has ten usable variants
# ("Grasp n cookies first, then 10-n", n=1..9 plus a skip variant) sampled per episode by
# `first_grasp: random`, and a missing variant is not a partial loss: under a sentence the
# checkpoint has never read it moves 2 Cookies in 3200 steps and scores 0/20 (9.42).
# Second, putting all ten under one short instruction does not remove the problem, it
# moves it inside the model: the same observation and the same sentence then have to
# produce ten different grasp plans, which is a plan the data never teaches.
#
# n=5 is the one plan that makes the instruction unambiguous and needs no new skill.  The
# 20-cookie task becomes four batches of five, which is the 2x5 task run twice more -- so
# the 1200 2x5 episodes are direct demonstrations of the same motion, and the only thing
# the model has to learn is that this task does not stop after two of them.
#
# The data
# --------
#   2x5        1200 episodes   straight from the existing collections, relabelled
#   2x10 n=5    ~190 episodes  = 30 recovered from the mixed-plan collection (the subset
#                              whose prompt already said "Grasp 5 cookies first") plus
#                              ~160 collected with `--first-grasp 5`
# and the 2x10 shards are concatenated three times, which puts them at ~50% of the frames
# -- the same balance the v2 arm had, so the only variable against v2 is that the plan is
# single instead of ten-way.
#
#   bash /root/autodl-tmp/logs/multitask_v4_pipeline.sh [steps] [eval_seeds]
set -u
cd /root/a3_dual_arm_sim

STEPS="${1:-15000}"
EVAL_SEEDS="${2:-8}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
BASE=/root/autodl-tmp/pi05-runs/a3_b2_aligned_v3_4000/checkpoints/004000/pretrained_model
TWO_BY_FIVE_1=/root/autodl-tmp/datasets/a3_b2_main_aligned_v2_merged
TWO_BY_FIVE_2=/root/autodl-tmp/datasets/a3_graft_v1_merged
TWO_BY_TEN_1='/root/autodl-tmp/datasets/old_n5/shard_*'
TWO_BY_TEN_2='/root/autodl-tmp/datasets/a3_2x10_n5/shard_*'
TWO_BY_TEN_REPEAT=3
DATA=/root/autodl-tmp/datasets/a3_multitask_v4
OUT=/root/autodl-tmp/pi05-runs/a3_multitask_v4_${STEPS}
SOCK=/tmp/mtv4_${STEPS}.sock
LOG=/root/autodl-tmp/logs/pipeline_multitask_v4.log
TRAIN_LOG=/root/autodl-tmp/logs/train_multitask_v4_${STEPS}.log

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }
say "=== single-plan pipeline start (${STEPS} steps, ${EVAL_SEEDS} eval seeds) ==="

# ---- 0. wait for the n=5 collection --------------------------------------------
for _ in $(seq 1 600); do
    pgrep -f collect_cookie_2x10 > /dev/null || break
    sleep 60
done
if pgrep -f collect_cookie_2x10 > /dev/null; then
    say "0/4 n=5 collection still running after 10h; stopping rather than training on a partial set"
    exit 1
fi
N5_NEW=$("$PY" - <<'PY'
import glob, json
episodes = 0
for root in sorted(glob.glob("/root/autodl-tmp/datasets/a3_2x10_n5/shard_*")):
    try:
        episodes += json.load(open(f"{root}/meta/info.json"))["total_episodes"]
    except Exception:
        pass
print(episodes)
PY
)
say "0/4 n=5 collection done: ${N5_NEW} newly collected episodes (+30 recovered from the mixed set)"
[ "${N5_NEW:-0}" -ge 100 ] || { say "0/4 only ${N5_NEW} new n=5 episodes; need >=100"; exit 1; }

# ---- 1. build -------------------------------------------------------------------
if [ -e "$DATA/meta/info.json" ]; then
    say "1/4 dataset: $DATA exists, reusing"
else
    say "1/4 building: 2x5 relabelled + 2x10 n=5 x${TWO_BY_TEN_REPEAT}, short prompts only"
    rm -rf "$DATA" "${DATA}_stage"
    "$PY" scripts/build_multitask_dataset.py \
        --out "$DATA" --repo-id local/a3-multitask-v4 \
        --two-by-five "$TWO_BY_FIVE_1" "$TWO_BY_FIVE_2" \
        --two-by-ten "$TWO_BY_TEN_1" "$TWO_BY_TEN_2" \
        --two-by-ten-repeat "$TWO_BY_TEN_REPEAT" \
        --short-prompts \
        > /root/autodl-tmp/logs/build_mt_v4.log 2>&1
    STATUS=$?
    say "1/4 build exit=$STATUS $(grep -a '^wrote' -A 4 /root/autodl-tmp/logs/build_mt_v4.log | tr '\n' ' ' | cut -c1-140)"
    [ "$STATUS" -eq 0 ] || exit 1
fi
FRAMES=$("$PY" -c "import json;print(json.load(open('$DATA/meta/info.json'))['total_frames'])")
EPISODES=$("$PY" -c "import json;print(json.load(open('$DATA/meta/info.json'))['total_episodes'])")
SHARE=$("$PY" - <<PY
import json, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
d = Path("$DATA")
idx = pd.read_parquet(d / "meta/tasks.parquet").index.tolist()
i10 = {k for k, x in enumerate(idx) if x.startswith("transfer 20")}
frames = total = 0
for p in sorted((d / "data").glob("chunk-*/*.parquet")):
    t = pq.read_table(p, columns=["task_index"])["task_index"].to_numpy()
    total += len(t)
    frames += sum(int((t == k).sum()) for k in i10)
print(f"{frames} of {total} ({frames/total*100:.1f}%)")
PY
)
say "1/4 dataset: ${EPISODES} episodes, ${FRAMES} frames, 2x10 frames ${SHARE}; ${STEPS}x6 = $(python3 -c "print(round(${STEPS}*6/${FRAMES},3))") epoch"

# ---- 2. train -------------------------------------------------------------------
if [ -e "$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors" ]; then
    say "2/4 train: checkpoint exists, skipping"
else
    say "2/4 train: ${STEPS} steps from 4.05, checkpoints every $(( STEPS / 2 ))"
    pkill -f a3_bridge_server.py 2>/dev/null
    pkill -f train_pi05_launcher 2>/dev/null
    sleep 8
    env A3_PI05_BASE="$BASE" A3_UNFREEZE_LLM_LAYERS=18 A3_UNFREEZE_VISION=1 A3_STATE_PROJECTOR=1 \
        bash scripts/train_pi05_a3.sh \
          "$DATA" "$OUT" "$STEPS" 6 "a3_multitask_v4_${STEPS}" \
          --policy.optimizer_lr=3e-06 \
          --policy.scheduler_decay_lr=2e-07 \
          --policy.scheduler_warmup_steps=$(( STEPS / 30 )) \
          --policy.scheduler_decay_steps="$STEPS" \
          --save_freq=$(( STEPS / 2 )) \
          --dataset.video_backend=pyav \
          > "$TRAIN_LOG" 2>&1
    STATUS=$?
    say "2/4 train exit=$STATUS $(grep -ao 'Training: 100%[^|]*' "$TRAIN_LOG" | tail -1)"
    [ "$STATUS" -eq 0 ] || exit 1
fi

# ---- 3. serve -------------------------------------------------------------------
CKPT="$OUT/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
say "3/4 serving $(basename "$OUT")"
pkill -f a3_bridge_server.py 2>/dev/null
sleep 5
cd /root/autodl-tmp/a3-collect
setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
    --adapter adapters/pi05_a3.py --listen "unix://$SOCK" --config "ckpt=$CKPT" \
    > /root/autodl-tmp/logs/bridge_mtv4${STEPS}.log 2>&1 < /dev/null &
cd /root/a3_dual_arm_sim
for _ in $(seq 1 180); do
    grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv4${STEPS}.log 2>/dev/null && break
    sleep 5
done
grep -q "listening on" /root/autodl-tmp/logs/bridge_mtv4${STEPS}.log || { say "3/4 bridge never came up"; exit 1; }

# ---- 4. evaluate ----------------------------------------------------------------
# `--prompt-style short` on both rows: the checkpoint was trained on main's short
# sentences, so asking anything else measures a task it never saw.  One episode per
# process because a 2x10 episode at the 3200-step budget costs ~90 minutes.
say "4/4 evaluating 2x5 (${EVAL_SEEDS}, 900 steps) + 2x10 (${EVAL_SEEDS}, 3200 steps), short prompts"
for start in $(seq 0 $(( EVAL_SEEDS - 1 ))); do
    setsid nohup "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x5 --prompt-style short --episodes 1 --seed-start "$start" --max-steps 900 \
        --output "/root/autodl-tmp/logs/v4_2x5_${start}.json" \
        > "/root/autodl-tmp/logs/v4_2x5_${start}.log" 2>&1 < /dev/null &
    setsid nohup "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x10 --prompt-style short --episodes 1 --seed-start "$start" --max-steps 3200 \
        --output "/root/autodl-tmp/logs/v4_2x10_${start}.json" \
        > "/root/autodl-tmp/logs/v4_2x10_${start}.log" 2>&1 < /dev/null &
done
wait
say "4/4 evaluation finished"

"$PY" - >> "$LOG" 2>&1 <<'PY'
import glob, json
for row, cap in (("2x5", 10), ("2x10", 20)):
    rows = {}
    for path in sorted(glob.glob(f"/root/autodl-tmp/logs/v4_{row}_*.json")):
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
print("reference: 2x5 v2 arm 5.00/10 (short not tested) | 2x10 best so far 5.00/20")
print("targets: 2x5 3-4/10, 2x10 7-8/20")
PY
say "=== single-plan pipeline done ==="
