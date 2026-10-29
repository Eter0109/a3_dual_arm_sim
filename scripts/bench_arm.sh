#!/usr/bin/env bash
# Benchmark one checkpoint on main's 2x5 and 2x10, optionally with the history channel.
#
#   bash scripts/bench_arm.sh <checkpoint_dir> [seeds] [--history] [--prompt-style P] [--tag NAME]
#
# `--prompt-style` and `--source-column` must both match what the checkpoint was trained
# on.  A pi0.5 fine-tune answers the sentence it saw and nothing else, so the wrong pair
# measures the sentence rather than the policy:
#
#   --prompt-style short                        -> 'transfer 10 cookies into target box'
#   --prompt-style task --source-column 1       -> 'Transfer 10 cookies from source column 1
#                                                   into the target box in two batches of five.'
#
# Defaults are main's own: 1000 steps for 2x5, 3200 for 2x10.  Override with STEPS_2X5 /
# STEPS_2X10 in the environment when comparing against older rows that used 900.
#
# Exists because the pipeline's evaluation stage is welded to a training run: it serves the
# checkpoint the run just produced and nothing else.  Two things need more than that:
#
#   * the mid-run checkpoint, when the final one is still an hour away and the question "is
#     this arm worth finishing" has to be answered now;
#   * a *previous* arm's weights served with the launcher's patches installed, which is the
#     only way to isolate what the state projector contributes -- every arm before 9.45 was
#     benchmarked with it silently absent.
#
# `--history` sets `history_dims=480`; without it the adapter serves the checkpoint as a
# plain pi0.5 with only the modules the checkpoint actually contains, which is what makes the
# two numbers comparable.
set -u
cd /root/a3_dual_arm_sim

CKPT="${1:?usage: bench_arm.sh <checkpoint_dir> [seeds] [--history] [--tag NAME]}"
SEEDS="${2:-8}"
shift 2 2>/dev/null || shift 1
HISTORY=0
PROMPT_STYLE="short"
SOURCE_COLUMN=""
TAG="arm"
while [ $# -gt 0 ]; do
    case "$1" in
        --history) HISTORY=1 ;;
        --prompt-style) PROMPT_STYLE="${2:?--prompt-style needs a value}"; shift ;;
        --source-column) SOURCE_COLUMN="${2:?--source-column needs a value}"; shift ;;
        --tag) TAG="${2:?--tag needs a name}"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

PY=/root/autodl-tmp/a3-collect/.venv/bin/python
DATA=/root/autodl-tmp/datasets/a3_multitask_v6
SOCK=/tmp/bench_${TAG}.sock
LOGS=/root/autodl-tmp/logs
mkdir -p "$LOGS"

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp

[ -e "$CKPT/model.safetensors" ] || { echo "no model.safetensors in $CKPT" >&2; exit 1; }

HISTORY_ARGS=()
if [ "$HISTORY" -eq 1 ]; then
    HISTORY_ARGS=(--config history_dims=480)
fi

pkill -f "a3_bridge_server.py --adapter adapters/pi05_a3.py --listen unix://$SOCK" 2>/dev/null
sleep 3

cd /root/autodl-tmp/a3-collect
setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
    --adapter adapters/pi05_a3.py --listen "unix://$SOCK" \
    --config "ckpt=$CKPT" --config stats_root="$DATA" --config state_dims=16 \
    "${HISTORY_ARGS[@]}" \
    > "$LOGS/bridge_${TAG}.log" 2>&1 < /dev/null &
cd /root/a3_dual_arm_sim

for _ in $(seq 1 240); do
    grep -q "listening on" "$LOGS/bridge_${TAG}.log" 2>/dev/null && break
    grep -qi "error\|Traceback" "$LOGS/bridge_${TAG}.log" 2>/dev/null && { echo "bridge failed:"; tail -5 "$LOGS/bridge_${TAG}.log"; exit 1; }
    sleep 5
done
grep -q "listening on" "$LOGS/bridge_${TAG}.log" || { echo "bridge never came up" >&2; exit 1; }
# The patches are what make this a measurement of the arm rather than of a stripped model.
grep -a "launcher patches" "$LOGS/bridge_${TAG}.log" | head -1

WINDOW_ARGS=()
if [ "$HISTORY" -eq 1 ]; then
    WINDOW_ARGS=(--history-window)
fi

for start in $(seq 0 $(( SEEDS - 1 ))); do
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x5 --prompt-style "$PROMPT_STYLE" "${WINDOW_ARGS[@]}" ${SOURCE_COLUMN} \
        --episodes 1 --seed-start "$start" --max-steps "${STEPS_2X5:-1000}" \
        --output "$LOGS/${TAG}_2x5_${start}.json" \
        > "$LOGS/${TAG}_2x5_${start}.log" 2>&1 &
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x10 --prompt-style "$PROMPT_STYLE" "${WINDOW_ARGS[@]}" \
        --episodes 1 --seed-start "$start" --max-steps "${STEPS_2X10:-3200}" \
        --output "$LOGS/${TAG}_2x10_${start}.json" \
        > "$LOGS/${TAG}_2x10_${start}.log" 2>&1 &
done
wait

"$PY" - "$TAG" <<'PY'
import glob, json, sys

tag = sys.argv[1]
for row, cap in (("2x5", 10), ("2x10", 20)):
    scores, moved = [], []
    for path in sorted(glob.glob(f"/root/autodl-tmp/logs/{tag}_{row}_*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        # `evaluate` returns BenchmarkResult, whose to_dict is {summary, episodes}.
        for episode in data.get("episodes", []):
            scores.append(episode.get("cookies_in_target", episode.get("score", 0)))
            moved.append(episode.get("cookies_in_source"))
    if scores:
        print(f"{row}: scored {scores} = {sum(scores)}/{len(scores) * cap}"
              f"   moved-out {moved}")
    else:
        print(f"{row}: no results")
PY
