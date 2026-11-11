#!/usr/bin/env bash
# The shape-correct test: does steering work when the prior keeps its Gaussian tails?
#
#   bash scripts/recap/run_dsrl_eval_meanshift.sh [seeds] [shards]
#
# What the previous two rounds established
# ---------------------------------------
# | prior | |u|mean | std | p99 | max | mass |x|>2 | score (16 seeds) |
# |---|---|---|---|---|---|---|
# | `N(0,1)` (baseline) | 0.798 | 1.000 | 2.577 | 5.27 | 4.55 % | **12.44** |
# | `tanh(N(0,1))`      | 0.556 | 0.628 | 0.989 | 1.00 |  0 %   | 7.38 |
# | `1.34 * tanh(...)`  | 0.745 | 0.841 | 1.325 | 1.34 |  0 %   | 0.70 (10 seeds) |
# | `actor` (tanh)      | 0.510 | 0.592 | 0.985 | 0.99 |  0 %   | 0.25 |
#
# Two conclusions from that table, and one of them cost a wasted round:
#   * narrowing hurts (`tanh` loses 5 points);
#   * *widening does not help* -- rescaling `tanh` to match `mean |u|` made it worse (0.70), because
#     matching a mean is not matching a shape: the rescaled prior still had a hard wall and no tail.
#     So the defect is the truncated support, not the width.  That was my first hypothesis and it
#     was wrong.
#
# The arms here
# -------------
# `mean_shift` keeps the marginal Gaussian and moves only its location:
#
#     u = eps + shift_scale * m,     eps ~ N(0,1),  m = the actor's pre-tanh mean
#
# Measured on 16 frames at `shift_scale = 0.3`: |u|mean 0.868, std 1.070, max 3.67, 5.66 % of mass
# beyond |x|>2 -- i.e. the same shape and the same tails as the baseline, 6-9 % wider.  The only
# thing that differs from `default` is *where* the noise sits, and that location varies with the
# observation (shift std 0.613 across frames).
#
# | arm | shift | answers |
# |---|---|---|
# | `ms_actor` | `0.3 * m(s)`, varies with the observation | does steering work at all? |
# | `ms_fixed` | `0.3 * m(s0)`, one constant vector for every frame | is the *observation-dependence* doing anything? |
#
# The second is the control that matters.  A constant shift is the cheapest possible "steering":
# if it scores the same as `ms_actor`, the actor's per-state information is worth nothing and any
# gain is just "the prior was moved somewhere", which needs no training.
#
# Why 8 seeds
# -----------
# The effects this is chasing are 5-12 Cookies.  The paired-difference SEM at 8 seeds is ~2.4, so
# 8 seeds resolves anything larger than ~5.  It will not certify a 2-point gain, and it is not
# meant to: this round is asking whether the direction is right, not how large it is.
set -uo pipefail

SEEDS="${1:-8}"
SHARDS="${2:-8}"
cd "$(dirname "$0")/../.."

PY=/root/autodl-tmp/a3-collect/.venv/bin/python
CKPT=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
ACTOR="${A3_ACTOR_CKPT:-/root/autodl-tmp/logs/dsrl_v5/checkpoint.pt}"
OUT=/root/autodl-tmp/logs/dsrl_eval
mkdir -p "$OUT"

echo "=== shape-correct arms (mean_shift): ${SEEDS} seeds x ${SHARDS} shards ==="

run_arm() {
    local tag="$1" extra="$2"
    echo
    echo "################ arm $tag ($extra) ################"
    A3_POLICY_CKPT="$CKPT" A3_EXTRA_CONFIG="$extra" \
        bash scripts/recap/eval_conditioned.sh "dsrl_$tag" "" "$SEEDS" "$SHARDS"
}

run_arm ms_actor "actor_ckpt=$ACTOR noise_source=mean_shift shift_scale=0.3"
run_arm ms_fixed "actor_ckpt=$ACTOR noise_source=mean_shift shift_scale=0.3 shift_fixed=1"

echo
echo "################ comparison ################"
"$PY" - <<'PY' | tee "$OUT/comparison_meanshift.txt"
import glob, json, statistics as st

arms = {}
for tag in ("default", "ms_actor", "ms_fixed"):
    rows = []
    for path in sorted(glob.glob(f"/root/autodl-tmp/datasets/eval_dsrl_{tag}/part_*/episode_labels.jsonl")):
        rows += [json.loads(line) for line in open(path)]
    if rows:
        arms[tag] = {r["seed"]: r["cookies_in_target"] for r in rows}

common = sorted(set.intersection(*[set(v) for v in arms.values()])) if len(arms) > 1 else []
print(f"paired seeds: {len(common)}")
print()
print(f"{'arm':12s} {'n':>3s} {'mean':>7s} {'sd':>6s}")
for tag, sc in arms.items():
    v = list(sc.values())
    print(f"{tag:12s} {len(v):3d} {st.mean(v):7.2f} {st.stdev(v) if len(v)>1 else 0:6.2f}")

if len(common) > 1:
    print()
    print(f"{'contrast':24s} {'diff':>7s} {'sem':>6s} {'t':>6s}  w/t/l")
    for a, b in (("ms_actor", "default"), ("ms_fixed", "default"),
                 ("ms_actor", "ms_fixed")):
        if a in arms and b in arms:
            d = [arms[a][s] - arms[b][s] for s in common]
            md = st.mean(d); sd = st.stdev(d) if len(d) > 1 else 0.0
            sem = sd / max(len(d), 1) ** 0.5
            w = sum(1 for x in d if x > 0); t_ = sum(1 for x in d if x == 0)
            print(f"{a+' - '+b:24s} {md:+7.2f} {sem:6.2f} {(md/sem if sem else 0):6.2f}"
                  f"  {w}/{t_}/{len(d)-w-t_}")

print()
print("How to read this")
print("  `ms_actor - default` > 0 with |t| >= 2: steering works once the shape is right.")
print("  `ms_actor - ms_fixed` > 0 with |t| >= 2: the actor's per-state information matters;")
print("    without it, the gain would have been obtainable from a constant shift and no training.")
print("  Both near zero: the prior was moved, and it did not matter -- so at this scale the")
print("  policy simply does not read its initial noise the way DSRL assumes.")
PY

echo
echo "=== done; summary at $OUT/comparison_meanshift.txt ==="
