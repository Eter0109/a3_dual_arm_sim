#!/usr/bin/env bash
# Evaluate one checkpoint on one prompt condition, sharded across processes.
#
#   A3_POLICY_CKPT=<pretrained_model dir> bash scripts/recap/eval_conditioned.sh \
#       <tag> [suffix] [seeds] [shards]
#
# Examples
# --------
#   A3_POLICY_CKPT=$CKPT_4A bash scripts/recap/eval_conditioned.sh r4a_pos " Advantage: positive"
#   A3_POLICY_CKPT=$CKPT_4A bash scripts/recap/eval_conditioned.sh r4a_nosuffix ""
#   A3_POLICY_CKPT=$CKPT_V9 bash scripts/recap/eval_conditioned.sh v9 ""
#
# Why the suffix is the whole experiment
# --------------------------------------
# Advantage conditioning lives in the *sentence*, not in the weights' shape
# (`build_task_variants.py`).  So the same checkpoint evaluated with the positive suffix and
# without it is the pair that isolates what conditioning bought, and the third arm -- v9 on the
# same seeds -- says whether RECAP as a whole moved anything.
#
# Why one harness for all three
# -----------------------------
# §9.80 of docs/pi05-training-history.md: *one harness for every arm in a comparison*.  The
# per-step history window only fills if `policy.observe_step` runs every control step, and main's
# `run_episode` never calls it, so anything scored there measures a 47 %-empty window and reads
# roughly +2 Cookies higher.  Every arm here goes through `collect_rollouts.py` for that reason,
# and the v9 numbers already in §9.74 are therefore *not* reusable as this baseline.
#
# Why `--no-record`
# ----------------
# A score needs `success`, `cookies_in_target` and `steps`, all of which come from the trace.
# The three camera streams are training data, and encoding them costs about 3x the wall time
# (measured: 1.0 -> 3.4 steps/s single, both still dominated by MuJoCo).  The control loop is
# untouched, so scores stay comparable to a recorded run; `NullRecorder` documents the argument.
#
# Why shards
# ----------
# One collector saturates one CPU core and leaves the GPU ~5 % busy: an episode needs 1500/16 =
# 94 bridge forwards at ~219 ms, i.e. ~21 s of GPU for ~440 s of wall time.  So the model is a
# server, not a bottleneck, and N processes share it.  Measured here: 3.4 steps/s for one
# process, 11.4 for four, 25.5 for eight -- 7.5x, which turns a 7-hour arm into a 55-minute one.
# Each shard owns its own output root because `LeRobotDataset.create` refuses two writers.
set -euo pipefail

TAG="${1:?usage: eval_conditioned.sh <tag> [suffix] [seeds] [shards]}"
SUFFIX="${2-}"
SEEDS="${3:-56}"
SHARDS="${4:-8}"
CKPT="${A3_POLICY_CKPT:?set A3_POLICY_CKPT to the pretrained_model directory}"

cd "$(dirname "$0")/../.."
PROJECT="$PWD"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
ROOT=/root/autodl-tmp/datasets/eval_${TAG}
SOCK=/tmp/eval_${TAG}.sock
LOG=/root/autodl-tmp/logs/eval_${TAG}.log
BLOG=/root/autodl-tmp/logs/bridge_eval_${TAG}.log

# Fixed start so every arm sees the identical seed set; that is what makes the arms paired.
SEED_START=300
SEED_END=$(( SEED_START + SEEDS - 1 ))
PER_SHARD=$(( (SEEDS + SHARDS - 1) / SHARDS ))

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp A3_MAIN_ROOT=/tmp/a3-main
mkdir -p "$TMPDIR" /root/autodl-tmp/logs

say() { printf '%s %s\n' "$(date '+%m-%d %H:%M')" "$*" | tee -a "$LOG"; }

say "=== eval $TAG: seeds ${SEED_START}-${SEED_END} in $SHARDS shard(s), suffix=[$SUFFIX] ==="
say "checkpoint: $CKPT"
[ -e "$CKPT/model.safetensors" ] || { say "missing checkpoint weights"; exit 1; }

say "freeing the GPU"
pkill -f a3_bridge_server.py 2>/dev/null || true
sleep 6

say "starting bridge on unix://$SOCK"
# The adapter module path is relative to this directory, and `--config ckpt=` is how the bridge
# learns which weights to serve.
# `A3_EXTRA_CONFIG` carries adapter options that are not the checkpoint -- today that is the
# DSRL steering actor (`actor_ckpt=...`).  It is a single string of space-separated `key=value`
# pairs so an arm is still one command line, and it is *unset* for the baseline arm on purpose:
# the control has to be the untouched `predict_action_chunk` path, otherwise the comparison is
# between two new things rather than between two noise policies.
EXTRA=()
if [ -n "${A3_EXTRA_CONFIG:-}" ]; then
    IFS=' ' read -r -a EXTRA_PAIRS <<< "$A3_EXTRA_CONFIG"
    for pair in "${EXTRA_PAIRS[@]}"; do EXTRA+=(--config "$pair"); done
    say "extra bridge config: $A3_EXTRA_CONFIG"
fi

cd /root/autodl-tmp/a3-collect
setsid nohup "$PY" /root/a3_dual_arm_sim/scripts/a3_bridge_server.py \
    --adapter adapters/pi05_a3.py --listen "unix://$SOCK" \
    --config "ckpt=$CKPT" --config state_dims=16 --config "history_dims=480" \
    "${EXTRA[@]}" \
    > "$BLOG" 2>&1 < /dev/null &
cd "$PROJECT"

for _ in $(seq 1 120); do
    grep -q "listening on" "$BLOG" 2>/dev/null && break
    grep -q "Traceback" "$BLOG" 2>/dev/null && { say "bridge failed:"; tail -8 "$BLOG" | tee -a "$LOG"; exit 1; }
    sleep 5
done
grep -q "listening on" "$BLOG" || { say "bridge never came up"; tail -8 "$BLOG" | tee -a "$LOG"; exit 1; }
say "bridge ready"

rm -rf "$ROOT"
mkdir -p "$ROOT"
say "collecting"
started=$(date +%s)
pids=()
for (( shard = 0; shard < SHARDS; shard++ )); do
    from=$(( SEED_START + shard * PER_SHARD ))
    to=$(( from + PER_SHARD - 1 ))
    [ "$to" -le "$SEED_END" ] || to=$SEED_END
    [ "$from" -le "$SEED_END" ] || continue
    "$PY" scripts/recap/collect_rollouts.py \
        --policy bridge --socket "$SOCK" \
        --output "$ROOT/part_$(printf '%02d' "$shard")" \
        --seeds "${from}-${to}" --task 2x10 --max-steps 1500 \
        --no-record --prompt-style task --task-suffix "$SUFFIX" \
        --history-frames 10 --execute 16 \
        > "/root/autodl-tmp/logs/eval_${TAG}_part$(printf '%02d' "$shard").log" 2>&1 &
    pids+=($!)
done
say "${#pids[@]} collector(s) running"

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$(( failed + 1 ))
done
say "collectors finished in $(( $(date +%s) - started ))s, $failed failed"

pkill -f "a3_bridge_server.py --adapter adapters/pi05_a3.py --listen unix://$SOCK" 2>/dev/null || true

[ "$failed" -eq 0 ] || { say "a shard failed; see /root/autodl-tmp/logs/eval_${TAG}_part*.log"; exit 1; }

# --- summary ------------------------------------------------------------------------
"$PY" - "$ROOT" "$TAG" "$SUFFIX" <<'PY' | tee -a "$LOG"
import glob, json, statistics as st, sys
from pathlib import Path

root, tag, suffix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for path in sorted(glob.glob(str(root / "part_*" / "episode_labels.jsonl"))):
    rows += [json.loads(line) for line in open(path)]
if not rows:
    raise SystemExit("no episode labels found")
rows.sort(key=lambda r: r["seed"])

n = len(rows)
strict = sum(1 for r in rows if r["success"])
# The label key is `cookies_in_target`, not `score` -- `EpisodeScore.score` is the harness's
# attribute, while the JSONL writes the field's own name.  Using the wrong one crashed the first
# arm's summary *after* 140 minutes of collection had already finished, which is the expensive
# way to learn a key name.
scores = [r["cookies_in_target"] for r in rows]
steps = [r["steps"] for r in rows]
# The suffix must actually have reached the model, or the arm is a duplicate of the control.
# `instruction` is written by the collector from the string it hands the policy, so this is a
# record of what was sent rather than of what was intended.
wrong = [r["seed"] for r in rows if not r["instruction"].endswith(suffix)]
seeds = [r["seed"] for r in rows]

print()
print(f"{tag}: n={n}  seeds {min(seeds)}-{max(seeds)}  suffix=[{suffix}]")
print(f"  strict success {100 * strict / n:.1f} %  ({strict}/{n})")
print(f"  mean cookies   {st.mean(scores):.2f}  sd {st.stdev(scores) if n > 1 else 0:.2f}"
      f"  sem {(st.stdev(scores) / n ** 0.5) if n > 1 else 0:.2f}")
print(f"  mean steps     {st.mean(steps):.0f} of 1500")
print(f"  score histogram {sorted(scores)}")
print(f"  prompts carried the suffix: {'yes' if not wrong else f'NO -- {len(wrong)} did not'}")
PY

say "=== eval $TAG done ==="
