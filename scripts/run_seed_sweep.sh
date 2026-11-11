#!/usr/bin/env bash
# Sweep seeds for two arms, round-robin, reusing one loaded bridge per arm.
#
# Why round-robin and not one arm at a time
# ----------------------------------------
# The point of these seeds is a *paired* comparison: v9 against the no-history ablation.
# Running arm A to completion and then arm B means that if the night is cut short -- a
# different job arrives, the box is needed -- one arm has 100 episodes and the other has 8.
# Alternating one batch per arm keeps both at equal n after every round, so the comparison
# is valid whenever the run stops, and the stopping time becomes a capacity choice instead
# of a threat to the result.
#
# Both bridges stay loaded (2 x ~10 GB on a 32 GB card).  That is what makes alternation
# cheap: `bench_arm.sh` reloads the checkpoint every time and the load costs ~2 minutes, so
# alternating through it would spend more time loading than evaluating.
#
# Why not a loop over `bench_arm.sh`
# ---------------------------------
# Same reason.  16 batches x 2 minutes of load is 32 minutes of nothing.
#
# Budgets, and the evidence for them
# ----------------------------------
# 2x5 keeps 1000 steps (50 s).  Measured on this checkpoint: all 8 seeds reach their peak
# cookie count by 23.1 s, and 1000 steps is what the original 8 seeds used, so these stay
# comparable with them.  Changing it would redefine the quantity being averaged.
#
# 2x10 drops from 3200 to 1500 steps (75 s).  Measured: every one of 8 seeds reached its
# final cookie count by 55.4 s; the score at 1200 steps was identical seed-for-seed to the
# score at 3200; and past that point the arm keeps knocking cookies out and re-placing
# them, so extra steps buy nothing.  1500 keeps 35 % headroom over the slowest observed
# finish.  Measured batch wall time at this budget: 15.9 min for 8 seeds.
#
# Stop rules
# ----------
# Every batch is preceded by a wall-clock check against the measured duration of the
# previous batch, so a batch never starts that cannot finish.  Stop the run at any moment
# and every completed episode is already on disk as its own JSON.
#
# Usage:
#   run_seed_sweep.sh --until 07:00 --target 60 --start 8
#
# Both arms start at the same seed, because the comparison is paired: the point of running two
# arms is to compare them on *the same scenes*, and independent seed sets would need a much
# larger n to reach the same power.
set -u
cd /root/a3_dual_arm_sim

PY=/root/autodl-tmp/a3-collect/.venv/bin/python
LOGS=/root/autodl-tmp/logs
DATA=/root/autodl-tmp/datasets/a3_clean_v9h
COLLECT=/root/autodl-tmp/a3-collect
PER_BATCH=8
STEPS_2X5=1000
STEPS_2X10=1500
# Which tasks a batch runs.  `2x5` alone is the useful setting when topping up a short task:
# 2x5 finishes in ~8 min per batch and 2x10 in 25-60 min, so a 2x5-only batch is 7x cheaper
# and the 2x10 seeds it would have produced are not needed for a 2x5 question.
TASKS="${TASKS:-2x5 2x10}"
# `execute` is how many of the 50 predicted steps are actually run before the policy is asked
# again.  Default 16; a smaller value re-plans more often, which is a closed-loop frequency
# change and needs no retraining.  Exposed here because it is an evaluation-side knob: the
# checkpoint is unchanged, so the same weights can be compared at several values.
EXECUTE="${EXECUTE:-16}"
EXEC_ARGS=(--execute "$EXECUTE")

UNTIL="07:00"
TARGET=60
START_SEED=8
while [ $# -gt 0 ]; do
    case "$1" in
        --until) UNTIL="${2:?--until needs HH:MM}"; shift ;;
        --target) TARGET="${2:?--target needs a count}"; shift ;;
        --start) START_SEED="${2:?--start needs a seed}"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR" "$LOGS"

seconds_left() {
    "$PY" - "$UNTIL" <<'PY'
import datetime as dt, sys
h, m = (int(v) for v in sys.argv[1].split(":"))
now = dt.datetime.now()
target = now.replace(hour=h, minute=m, second=0, microsecond=0)
if target <= now:
    target += dt.timedelta(days=1)
print(int((target - now).total_seconds()))
PY
}

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOGS/sweep.log"; }

batch_done() {
    local tag=$1 start=$2 n=0
    for s in $(seq "$start" $(( start + PER_BATCH - 1 ))); do
        # Gate on the tasks this run actually produces: requiring a 2x10 result while running
        # `TASKS=2x5` would mark every batch unfinished and re-run it forever.
        case " $TASKS " in
            *" 2x5 "*)  [ -s "$LOGS/${tag}_2x5_${s}.json" ] || return 1 ;;
        esac
        case " $TASKS " in
            *" 2x10 "*) [ -s "$LOGS/${tag}_2x10_${s}.json" ] || return 1 ;;
        esac
        n=$(( n + 1 ))
    done
    [ "$n" -eq "$PER_BATCH" ]
}

start_bridge() {
    local ckpt=$1 tag=$2 sock=$3 history=$4 frames=${5:-10}
    local hargs=()
    # `history_dims` is frames x 48 and has to match the dataset the checkpoint trained on;
    # the adapter refuses a window of the wrong width rather than truncating it.
    [ "$history" = "1" ] && hargs=(--config "history_dims=$(( frames * 48 ))")
    # Absolute script path: the subshell cds into the adapter tree, where a relative
    # `scripts/...` resolves to a path that does not exist.  That failure is instant and
    # silent, so anything that only polls for "listening on" waits out its whole timeout.
    ( cd "$COLLECT" && setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
        --adapter adapters/pi05_a3.py --listen "unix://$sock" \
        --config "ckpt=$ckpt" --config stats_root="$DATA" --config state_dims=16 \
        "${hargs[@]}" \
        > "$LOGS/bridge_${tag}.log" 2>&1 < /dev/null & )
    for _ in $(seq 1 120); do
        grep -q "listening on" "$LOGS/bridge_${tag}.log" 2>/dev/null && return 0
        if grep -qiE "Traceback|No such file|ModuleNotFoundError|ImportError|CUDA error|out of memory" \
                "$LOGS/bridge_${tag}.log" 2>/dev/null; then
            echo "bridge failed for ${tag}:"; tail -5 "$LOGS/bridge_${tag}.log"; return 1
        fi
        sleep 5
    done
    echo "bridge for ${tag} never came up"; return 1
}

run_batch() {
    local tag=$1 sock=$2 start=$3 history=$4 frames=${5:-10}
    local begin=$SECONDS
    # `--history-window` has to follow the arm, not the task: the ablation has no encoder,
    # so a window would fail its width check, and omitting it on the history arm would feed
    # a checkpoint a shape it never saw.
    local hargs=()
    [ "$history" = "1" ] && hargs=(--history-window --history-frames "$frames")
    for s in $(seq "$start" $(( start + PER_BATCH - 1 ))); do
        case " $TASKS " in
        *" 2x5 "*)
            "$PY" scripts/bench_pi05_on_main.py --socket "$sock" \
                --task 2x5 --prompt-style task "${hargs[@]}" --source-column 1 \
                --episodes 1 --seed-start "$s" --max-steps "$STEPS_2X5" \
                "${EXEC_ARGS[@]}" \
                --output "$LOGS/${tag}_2x5_${s}.json" \
                > "$LOGS/${tag}_2x5_${s}.log" 2>&1 &
            ;;
        esac
        case " $TASKS " in
        *" 2x10 "*)
            "$PY" scripts/bench_pi05_on_main.py --socket "$sock" \
                --task 2x10 --prompt-style task "${hargs[@]}" \
                --episodes 1 --seed-start "$s" --max-steps "$STEPS_2X10" \
                "${EXEC_ARGS[@]}" \
                --output "$LOGS/${tag}_2x10_${s}.json" \
                > "$LOGS/${tag}_2x10_${s}.log" 2>&1 &
            ;;
        esac
    done
    wait
    LAST_BATCH_SECONDS=$(( SECONDS - begin ))

    # Health check, because a batch that produced nothing still reaches this point and returns
    # instantly.  The failure this catches: the bridge was killed by an unrelated cleanup, every
    # driver died on `ConnectionRefusedError` within a second, and the loop advanced through
    # batch after batch at "0 min" each -- logging progress while writing no results, 103 empty
    # logs in one run.  A duration far below the slowest single episode means the work did not
    # happen, whatever the exit statuses said.
    local made=0
    for s in $(seq "$start" $(( start + PER_BATCH - 1 ))); do
        case " $TASKS " in
            *" 2x5 "*)  [ -s "$LOGS/${tag}_2x5_${s}.json" ] && made=$(( made + 1 )) ;;
            *" 2x10 "*) [ -s "$LOGS/${tag}_2x10_${s}.json" ] && made=$(( made + 1 )) ;;
        esac
    done
    if [ "$made" -eq 0 ] && [ "$LAST_BATCH_SECONDS" -lt 60 ]; then
        say "${tag}: batch produced nothing in ${LAST_BATCH_SECONDS}s; bridge is gone, aborting"
        tail -3 "$LOGS/${tag}_2x10_$start.log" 2>/dev/null | sed 's/^/    /'
        return 1
    fi
    return 0
}

# Arms, as `tag|checkpoint|history|frames`.  The window length travels with the arm because
# it is a property of the dataset each checkpoint trained on: v9 used 10 frames, the long-window
# arm 200, and the ablation has no encoder at all.  Round-robin order is this order.
#
# Override with `ARMS` to compare a different pair, e.g.
#   ARMS='v9w200|...|1|200 v9s|...|1|10' run_seed_sweep.sh ...
DEFAULT_ARMS=(
    "v9s|/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model|1|10"
    "v9w200|/root/autodl-tmp/pi05-runs/a3_clean_v9w200_30000/checkpoints/030000/pretrained_model|1|200"
)
if [ -n "${ARMS:-}" ]; then
    read -r -a ARM_LIST <<< "$ARMS"
else
    ARM_LIST=("${DEFAULT_ARMS[@]}")
fi

LAST_BATCH_SECONDS=$(( 20 * 60 ))   # measured 15.9 min; a little slack on the first gate
say "=== sweep start: soft target ${TARGET} seeds/arm, until ${UNTIL} ==="
say "budgets: 2x5=${STEPS_2X5} steps, 2x10=${STEPS_2X10} steps, ${PER_BATCH} seeds/batch, execute=${EXECUTE}"
say "arms: ${ARM_LIST[*]}"

pkill -f "a3_bridge_server.py --adapter adapters/pi05_a3.py --listen unix:///tmp/sweep_" 2>/dev/null
sleep 3
say "loading bridges (the only load in the whole run)"

# Per-arm state, in parallel arrays: next seed, soft-target-noted flag, history, frames.
declare -A NEXT_SEED NOTED FRAMES HIST
for spec in "${ARM_LIST[@]}"; do
    IFS='|' read -r tag ckpt hist frames <<< "$spec"
    FRAMES[$tag]=$frames
    HIST[$tag]=$hist
    NEXT_SEED[$tag]=$START_SEED    # both arms start at the same seed so the test is paired
    NOTED[$tag]=0
    start_bridge "$ckpt" "$tag" "/tmp/sweep_${tag}.sock" "$hist" "$frames" \
        || { say "${tag} bridge failed, aborting"; exit 1; }
    say "  ${tag} (${frames} frames) $(grep -a 'launcher patches' "$LOGS/bridge_${tag}.log" \
        | head -1 | sed 's/launcher patches *: //')"
done

while :; do
    left=$(seconds_left)
    if [ "$left" -lt $(( LAST_BATCH_SECONDS + 300 )) ]; then
        say "time gate: ${left}s left, a batch needs ${LAST_BATCH_SECONDS}s + 300s; stopping"
        break
    fi

    for spec in "${ARM_LIST[@]}"; do
        IFS='|' read -r arm _ckpt _hist _frames <<< "$spec"
        next=${NEXT_SEED[$arm]}
        sock="/tmp/sweep_${arm}.sock"

        # Re-check the clock inside the arm loop: one batch is long enough that the budget
        # can expire between the two arms of a single round.
        left=$(seconds_left)
        if [ "$left" -lt $(( LAST_BATCH_SECONDS + 300 )) ]; then
            say "time gate: ${left}s left before ${arm}; stopping"
            break 2
        fi

        if batch_done "$arm" "$next"; then
            say "${arm}: seeds ${next}..$(( next + PER_BATCH - 1 )) already on disk, skipping"
        else
            say "${arm}: batch seeds ${next}..$(( next + PER_BATCH - 1 )) start, ${left}s left"
            # A dead bridge makes every later batch a no-op, so stop the whole run rather
            # than record a night of zero-minute batches as if it were progress.
            if ! run_batch "$arm" "$sock" "$next" "${HIST[$arm]}" "${FRAMES[$arm]}"; then
                say "${arm}: batch failed, stopping the sweep"
                break 2
            fi
            say "${arm}: batch done in $(( LAST_BATCH_SECONDS / 60 )) min"
        fi
        NEXT_SEED[$arm]=$(( next + PER_BATCH ))

        # Soft target: report the milestone once, then keep going while the clock allows,
        # because a night with no other work should convert into more seeds.
        # Count against whichever task this run produces; a `TASKS=2x5` run would otherwise
        # compare a 2x10 count of zero against the target forever.
        case " $TASKS " in
            *" 2x10 "*) n=$(ls "$LOGS/${arm}_2x10_"*.json 2>/dev/null | wc -l) ;;
            *)          n=$(ls "$LOGS/${arm}_2x5_"*.json 2>/dev/null | wc -l) ;;
        esac
        if [ "$(( START_SEED + n ))" -ge "$TARGET" ] && [ "${NOTED[$arm]}" = "0" ]; then
            say "${arm}: soft target ${TARGET} reached (${n} new seeds); continuing while the clock allows"
            NOTED[$arm]=1
        fi
    done
done

say "=== sweep finished ==="
for spec in "${ARM_LIST[@]}"; do
    IFS='|' read -r a _ _ _ <<< "$spec"
    say "  ${a}: 2x5=$(ls "$LOGS/${a}_2x5_"*.json 2>/dev/null | wc -l) 2x10=$(ls "$LOGS/${a}_2x10_"*.json 2>/dev/null | wc -l) new seeds"
done
