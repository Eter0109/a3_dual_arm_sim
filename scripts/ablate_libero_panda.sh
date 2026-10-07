#!/usr/bin/env bash
# Leave-one-out ablation: how much of the 31/40 depends on each fix?
#
# Every fix that landed between the 0/40 runs and the 31/40 run has an environment switch, and
# this reverts them one at a time from the state that works.  A drop says the fix is
# load-bearing *now*; no drop says it was never what mattered, whatever it seemed to do to the
# numbers it was measured against at the time.
#
# The policy holds ~8.7 GiB, so runs go in batches of $PARALLEL rather than all at once.
#
# Usage:  bash scripts/ablate_libero_panda.sh [last_seed] [parallel]
set -u

LAST_SEED=${1:-39}
PARALLEL=${2:-2}
ROOT=/root/a3_dual_arm_sim
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
LOGS=/root/autodl-tmp/logs/ablation
mkdir -p "$LOGS"

export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# name|env assignments
CASES=$(cat <<'EOF'
baseline|A3_ABLATION_NONE=1
flip_rot180|A3_ADAPTER_FLIP=h
no_mirror|A3_IMAGE_MIRROR=0
no_sites|A3_HIDE_SITES=0
osc_once|A3_OSC_RETORQUE=once
heavy_bowl|A3_VISUAL_MASS=keep
home_only|A3_LIBERO_INIT=0
cookie_phys|A3_PHYSICS_LEGACY=1
grip_legacy|A3_GRIPPER_LEGACY=1
EOF
)

run_one() {
    local name=$1 spec=$2
    ( cd "$ROOT" && env $spec "$PY" scripts/probe_libero_props_trace.py \
        --scene a3_libero_panda --manifest configs/policies/pi05_libero_panda.yaml \
        --seeds "$(seq -s, 0 "$LAST_SEED")" --steps 300 \
        --trace "$LOGS/$name.npz" --figure "$LOGS/$name.png" \
        > "$LOGS/$name.log" 2>&1 ) &
}

n=0
while IFS='|' read -r name spec; do
    [ -z "$name" ] && continue
    echo "[$(date +%H:%M:%S)] starting $name ($spec)"
    run_one "$name" "$spec"
    n=$((n + 1))
    if [ $((n % PARALLEL)) -eq 0 ]; then wait; fi
done <<< "$CASES"
wait
echo "[$(date +%H:%M:%S)] all done; logs in $LOGS"
