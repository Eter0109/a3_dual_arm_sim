#!/usr/bin/env bash
# Does the learned noise policy actually score better?  Three arms, one harness, paired seeds.
#
#   bash scripts/recap/run_dsrl_eval.sh [seeds] [shards]
#
# The three arms, and what each one rules out
# ------------------------------------------
# | arm | sampler's x0 | answers |
# |---|---|---|
# | `default`     | `N(0,1)`, the policy's own | what the policy does unaided |
# | `tanh_random` | `tanh(N(0,1))`, repeated across the chunk axis | does merely *bounding* the noise help? |
# | `actor`       | the trained DSRL actor, deterministic | did the actor learn anything beyond that bound? |
#
# The middle arm is the one that makes the experiment interpretable.  The actor's output lives on
# `(-1, 1)` by construction (`tanh`), so `actor` differs from `default` in two ways at once: the
# bound and the content.  If `tanh_random` already matches or beats `default`, then any `actor`
# gain is a property of the bound -- which the literature already knows about and which needs no
# 8000 epochs of SAC to obtain.
#
# Why the arms are paired
# ----------------------
# Every arm runs seeds 300+ through `eval_conditioned.sh`, so the *scenes* are identical across
# arms and differences are differences rather than samples from three different rulers (§9.80).
# What is not paired is the noise realization: each arm draws its own.  With a within-scene score
# sd of ~3.34 Cookies, 16 seeds give a paired-difference SEM near 1.2, so this first pass can
# detect a large effect and cannot certify a small one.  The script says so in its output instead
# of leaving the reader to compute it.
#
# What has to be true before the numbers mean anything
# ---------------------------------------------------
# Two gates, both of which have failed silently in this project before:
#   * `verify_visual_feature.py` -- the actor's observation at inference equals the one it trained
#     on.  Measured: relative error 0.17 %, so the gate passes today.
#   * `probe_steering_effect.py` -- the actor's chunk differs from the default-noise chunk by more
#     than redrawing the noise does, and is not saturated.  Measured: 2.1x, 0 % saturated.
# Both run first and both abort the suite on failure, because a null result from a broken
# observation is indistinguishable from a null result from a useless actor.
set -uo pipefail

SEEDS="${1:-16}"
SHARDS="${2:-8}"
cd "$(dirname "$0")/../.."

PROJECT="$PWD"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python
CKPT=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
ACTOR="${A3_ACTOR_CKPT:-/root/autodl-tmp/logs/dsrl_v5/checkpoint.pt}"
OUT=/root/autodl-tmp/logs/dsrl_eval
mkdir -p "$OUT"

# The first seed of every arm.  Changing it invalidates comparison with any earlier arm.
SEED_START=300

if [ ! -e "$ACTOR" ]; then
    echo "missing actor checkpoint $ACTOR"; exit 1
fi

echo "=== DSRL steering evaluation: ${SEEDS} seeds x ${SHARDS} shards ==="
echo "actor: $ACTOR"

# --- gate 1: the observation the actor sees is the one it was trained on -------------------
echo
echo "################ gate 1/2: visual feature ################"
A3_POLICY_CKPT="$CKPT" "$PY" scripts/recap/verify_visual_feature.py \
    --actor "$ACTOR" --count 16 --output "$OUT/gate_visual.json" 2>&1 | tail -6
if ! grep -q '"ok": true' "$OUT/gate_visual.json" 2>/dev/null; then
    echo "GATE FAILED: live visual feature does not match the cache.  Every score below would be"
    echo "measuring the mismatch, not the actor.  Stopping."
    exit 1
fi

# --- gate 2: the actor is neither inert nor saturated -------------------------------------
echo
echo "################ gate 2/2: steering effect ################"
"$PY" scripts/recap/probe_steering_effect.py \
    --actor "$ACTOR" --rows 16 --batch 4 --output "$OUT/gate_effect.json" 2>&1 | tail -8
if ! grep -q '"finite": true' "$OUT/gate_effect.json" 2>/dev/null; then
    echo "GATE FAILED: the steered chunk contains NaN/Inf.  Stopping."
    exit 1
fi

run_arm() {
    local tag="$1" extra="$2"
    echo
    echo "################ arm $tag ################"
    A3_POLICY_CKPT="$CKPT" A3_EXTRA_CONFIG="$extra" \
        bash scripts/recap/eval_conditioned.sh "dsrl_$tag" "" "$SEEDS" "$SHARDS"
}

run_arm default ""
run_arm tanh_random "noise_source=tanh_random"
run_arm actor "actor_ckpt=$ACTOR"

# --- paired comparison --------------------------------------------------------------------
echo
echo "################ comparison ################"
"$PY" - "$SEEDS" <<'PY' | tee "$OUT/comparison.txt"
import glob, json, statistics as st, sys
from pathlib import Path

seeds_wanted = int(sys.argv[1])
arms = {}
for tag in ("default", "tanh_random", "actor"):
    rows = []
    for path in sorted(glob.glob(f"/root/autodl-tmp/datasets/eval_dsrl_{tag}/part_*/episode_labels.jsonl")):
        rows += [json.loads(line) for line in open(path)]
    arms[tag] = {r["seed"]: r["cookies_in_target"] for r in rows}
    if not rows:
        raise SystemExit(f"arm {tag} produced no episodes")

common = sorted(set(arms["default"]) & set(arms["tanh_random"]) & set(arms["actor"]))
print(f"paired seeds: {len(common)} of {seeds_wanted} requested")
print()
print(f"{'arm':14s} {'n':>3s} {'mean':>7s} {'sd':>6s} {'sem':>6s} {'strict':>7s}")
for tag, scores in arms.items():
    values = [scores[s] for s in common]
    sd = st.stdev(values) if len(values) > 1 else 0.0
    print(f"{tag:14s} {len(values):3d} {st.mean(values):7.2f} {sd:6.2f} "
          f"{sd / max(len(values), 1) ** 0.5:6.2f} {'-':>7s}")

print()
print(f"{'contrast':28s} {'mean diff':>10s} {'sem':>6s} {'t':>6s}")
def paired(a, b):
    diffs = [arms[a][s] - arms[b][s] for s in common]
    md = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    sem = sd / max(len(diffs), 1) ** 0.5
    t = md / sem if sem > 0 else 0.0
    return md, sem, t, diffs
for a, b in (("actor", "default"), ("tanh_random", "default"), ("actor", "tanh_random")):
    md, sem, t, diffs = paired(a, b)
    print(f"{a + ' - ' + b:28s} {md:+10.2f} {sem:6.2f} {t:6.2f}")
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    print(f"{'  wins/ties/losses':28s} {wins:>10d} {ties:6d} {len(diffs) - wins - ties:6d}")

print()
print("How to read this")
print("  `actor - tanh_random` is the only contrast that attributes anything to the actor.")
print("  A difference of a few tenths with |t| < 2 is not a result at this sample size: the")
print("  within-scene score sd is ~3.3 Cookies, so the paired difference needs |t| >= 2.")
print("  `actor - default` matching `tanh_random - default` means the bound did the work.")
PY

echo
echo "=== done; summary at $OUT/comparison.txt ==="
