#!/usr/bin/env bash
# Re-record 2x5 into an existing run directory, once the 2x10 pass has finished.
#
# The first attempted pass died on every 2x5 seed with "unrecognized arguments: 1" -- the
# driver's --source-column flag had its name stripped by bench_arm.sh, so argparse saw a
# bare positional.  2x10 was unaffected because it takes no source column, so the two tasks
# have to be re-run separately rather than as one job.
#
# ONLY=2x5 narrows bench_arm.sh's wipe to the 2x5 subtree, so the 2x10 footage already on
# disk survives.  SEEDS matches the pass being repaired.
set -u

CKPT=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
RECORD=/root/autodl-tmp/diag/v9_run
TAG=v9r
SEEDS=${SEEDS:-8}
LOGS=/root/autodl-tmp/logs

# Wait for the in-flight run to release the bridge.  bench_arm.sh's own `wait` only covers
# its own children, so anything still attached to the socket has to go first or the new
# bridge and the old one both answer.
for _ in $(seq 1 720); do
    pgrep -f bench_pi05_on_main > /dev/null 2>&1 || break
    sleep 10
done
if pgrep -f bench_pi05_on_main > /dev/null 2>&1; then
    echo "drivers still running after 2h; refusing to start a second pass" >&2
    exit 1
fi

pkill -f a3_bridge_server > /dev/null 2>&1
sleep 5

echo "=== 2x5 re-record, $SEEDS seeds -> $RECORD/2x5 ==="
cd /root/a3_dual_arm_sim
ONLY=2x5 setsid nohup bash scripts/bench_arm.sh "$CKPT" "$SEEDS" \
    --history --prompt-style task --source-column 1 \
    --record "$RECORD" --tag "$TAG" \
    > "$LOGS/bench_${TAG}_2x5retry.log" 2>&1 < /dev/null &

sleep 30
tail -3 "$LOGS/bench_${TAG}_2x5retry.log"
