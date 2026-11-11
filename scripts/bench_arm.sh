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
# An array, not a bare string: the flag name has to travel with the value.  Storing only "1"
# hands argparse a positional it does not accept, and every 2x5 seed dies instantly with
# "unrecognized arguments: 1" -- which looks like a finished run with no results.
SOURCE_COLUMN=()
TAG="arm"
RECORD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --history) HISTORY=1 ;;
        --prompt-style) PROMPT_STYLE="${2:?--prompt-style needs a value}"; shift ;;
        --source-column) SOURCE_COLUMN=(--source-column "${2:?--source-column needs a value}"); shift ;;
        --record) RECORD="${2:?--record needs a directory}"; shift ;;
        --tag) TAG="${2:?--tag needs a name}"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Recording is main's own `--diagnostic-dir`: one mp4 per camera per seed at the env's
# native 256x256 / control-rate fps, plus trace.jsonl and result.json.  The resolution is
# the env's, not a choice -- configs/default.yaml renders at 256 and the policy was trained
# on that scene, so rendering bigger for a nicer video would change the input being measured.
#
# ONLY=2x5 / ONLY=2x10 restricts the run to one task.  It also narrows the wipe to that
# task's subdirectory, so re-recording 2x5 after a failed pass cannot delete 2x10 footage.
ONLY="${ONLY:-both}"
RECORD_ARGS_2X5=()
RECORD_ARGS_2X10=()
if [ -n "$RECORD" ]; then
    # EpisodeDiagnostic opens its directory with exist_ok=False, so a stale seed_N aborts
    # the run rather than overwriting it.
    mkdir -p "$RECORD/2x5" "$RECORD/2x10"
    [ "$ONLY" = "2x10" ] || rm -rf "$RECORD/2x5"
    [ "$ONLY" = "2x5" ] || rm -rf "$RECORD/2x10"
    mkdir -p "$RECORD/2x5" "$RECORD/2x10"
    RECORD_ARGS_2X5=(--diagnostic-dir "$RECORD/2x5")
    RECORD_ARGS_2X10=(--diagnostic-dir "$RECORD/2x10")
    echo "recording to $RECORD/{2x5,2x10}/seed_N/{front,left_wrist,right_wrist}.mp4 (ONLY=$ONLY)"
fi

PY=/root/autodl-tmp/a3-collect/.venv/bin/python
DATA=/root/autodl-tmp/datasets/a3_multitask_v6
SOCK=/tmp/bench_${TAG}.sock
LOGS=/root/autodl-tmp/logs
mkdir -p "$LOGS"

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp

# lerobot writes a checkpoint as <step>/pretrained_model/model.safetensors, so an argument
# of <step> is the natural thing to type and the wrong thing to serve.  Accept either.
[ -e "$CKPT/model.safetensors" ] || {
    if [ -e "$CKPT/pretrained_model/model.safetensors" ]; then
        CKPT="$CKPT/pretrained_model"
    else
        echo "no model.safetensors in $CKPT (nor in $CKPT/pretrained_model)" >&2
        exit 1
    fi
}

HISTORY_ARGS=()
# `HISTORY_FRAMES` must match the dataset the checkpoint trained on: the encoder does
# `window.view(-1, steps, 48)`, so a 10-frame window against a 200-step encoder is a shape
# error and 200 against 10 is silently the wrong input.  v6-v9 used 10 (480 floats); the
# long-window arm uses 200 (9600).
HISTORY_FRAMES="${HISTORY_FRAMES:-10}"
if [ "$HISTORY" -eq 1 ]; then
    HISTORY_ARGS=(--config "history_dims=$(( HISTORY_FRAMES * 48 ))")
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
    WINDOW_ARGS=(--history-window --history-frames "$HISTORY_FRAMES")
fi

for start in $(seq 0 $(( SEEDS - 1 ))); do
    if [ "$ONLY" != "2x10" ]; then
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x5 --prompt-style "$PROMPT_STYLE" "${WINDOW_ARGS[@]}" "${SOURCE_COLUMN[@]}" \
        --episodes 1 --seed-start "$start" --max-steps "${STEPS_2X5:-1000}" \
        "${RECORD_ARGS_2X5[@]}" \
        --output "$LOGS/${TAG}_2x5_${start}.json" \
        > "$LOGS/${TAG}_2x5_${start}.log" 2>&1 &
    fi
    if [ "$ONLY" != "2x5" ]; then
    "$PY" scripts/bench_pi05_on_main.py --socket "$SOCK" \
        --task 2x10 --prompt-style "$PROMPT_STYLE" "${WINDOW_ARGS[@]}" \
        --episodes 1 --seed-start "$start" --max-steps "${STEPS_2X10:-3200}" \
        "${RECORD_ARGS_2X10[@]}" \
        --output "$LOGS/${TAG}_2x10_${start}.json" \
        > "$LOGS/${TAG}_2x10_${start}.log" 2>&1 &
    fi
done
wait

"$PY" - "$TAG" "$RECORD" "$ONLY" <<'PY'
import glob, json, re, sys

tag, record, only = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ""), sys.argv[3]
for row, cap in (("2x5", 10), ("2x10", 20)):
    if only not in ("both", row):
        continue
    scores, left, ranked = [], [], []
    for path in glob.glob(f"/root/autodl-tmp/logs/{tag}_{row}_*.json"):
        # Seed is the last field of the filename; sorting the paths as strings would put
        # seed_10 before seed_2, which reads as a shuffled run when it is not.
        seed = int(re.search(r"_(\d+)\.json$", path).group(1))
        try:
            data = json.load(open(path))
        except Exception:
            continue
        # `evaluate` returns BenchmarkResult, whose to_dict is {summary, episodes}.
        for episode in data.get("episodes", []):
            score = episode.get("cookies_in_target", episode.get("score", 0))
            scores.append((seed, score))
            # `cookies_in_source` counts what is STILL IN THE SOURCE at the end.  It used
            # to be printed as "moved-out", which reads as the number carried across --
            # a 10/10 episode showed "moved-out 70" when it had moved 10 and left 70.
            left.append((seed, episode.get("cookies_in_source")))
            ranked.append((score, seed, episode.get("cookies_in_source")))
    if not scores:
        print(f"{row}: no results")
        continue
    scores.sort()
    left.sort()
    total = [s for _, s in scores]
    print(f"{row}: scored {[s for _, s in scores]} = {sum(total)}/{len(total) * cap}"
          f"   left-in-source {[m for _, m in left]}")
    # Best three per the question actually asked -- most cookies placed -- and the path that
    # holds the footage, so picking one does not mean re-deriving the mapping by hand.
    ranked.sort(reverse=True)
    print(f"{row}: best three -> "
          + ", ".join(f"seed {seed} {score}/{cap} (left {left_in})"
                      for score, seed, left_in in ranked[:3]))
    if record:
        for score, seed, _ in ranked[:3]:
            print(f"      {record}/{row}/seed_{seed}/")
PY
