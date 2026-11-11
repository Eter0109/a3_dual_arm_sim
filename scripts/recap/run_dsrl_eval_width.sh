#!/usr/bin/env bash
# The decisive test: is the failure a *width* problem or a *bound* problem?
#
#   bash scripts/recap/run_dsrl_eval_width.sh [seeds] [shards]
#
# What §9.94 measured
# -------------------
# | arm | sampler's x0 | mean | vs baseline | t |
# |---|---|---|---|---|
# | `default`     | `N(0,1)`        | 12.44 | -- | -- |
# | `tanh_random` | `tanh(N(0,1))`  |  7.38 | -5.06 | -2.42 |
# | `actor`       | actor, det.     |  0.25 | -12.19 | -7.28 |
#
# and, on 16 real frames, the width of each prior in `mean |u|`:
#
#     N(0,1) 0.763    tanh(N(0,1)) 0.569    actor 0.510
#
# The sampler is a flow-matching head pre-trained on `x0 ~ N(0,1)`.  Both alternatives are
# 25-33 % narrower, and the worse one scored worse.  That correlation is suggestive and nothing
# more: narrower is confounded with *being the actor*, so it cannot by itself explain the -12.
#
# The two arms here break the confound
# ------------------------------------
# | arm | prior, rescaled to width 0.763 | answers |
# |---|---|---|
# | `tanh_wide`  | `1.340 * tanh(N(0,1))` | does restoring the width fix the bounded prior? |
# | `actor_wide` | `1.495 * actor`        | does the actor's *information* help once its width is fair? |
#
# The first is the width control.  The second is the one that matters: if `actor_wide` beats
# `tanh_wide`, the actor learned something real and §9.94's failure was a parameterisation bug
# (fixable).  If the two are equal, the actor learned nothing about *which* noise is better and
# the whole approach is dead here -- which is a different and stronger conclusion.
#
# Scales are measured, not guessed: `noise_scale` is set so that `mean |u|` of the scaled prior
# equals `N(0,1)`'s on the same 16 frames.  1.34 and 1.495 come from that measurement.
#
# Why 16 seeds and not more
# -------------------------
# The baseline sd is 6.66, so a paired difference of 3 Cookies needs ~40 seeds for |t|=2.  These
# arms are testing a 5-12 Cookie effect, which 16 seeds resolves.  If either lands within 2 of the
# baseline the answer is "width was the problem, and 16 seeds was enough to see it"; if either
# lands near 7.4 the answer is "width was not the problem" and no more seeds are needed either.
set -uo pipefail

SEEDS="${1:-16}"
SHARDS="${2:-8}"
cd "$(dirname "$0")/../.."

PY=/root/autodl-tmp/a3-collect/.venv/bin/python
CKPT=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
ACTOR="${A3_ACTOR_CKPT:-/root/autodl-tmp/logs/dsrl_v5/checkpoint.pt}"
OUT=/root/autodl-tmp/logs/dsrl_eval
mkdir -p "$OUT"

TANH_SCALE=1.340
ACTOR_SCALE=1.495

echo "=== width-controlled arms: ${SEEDS} seeds x ${SHARDS} shards ==="
echo "tanh_wide  scale ${TANH_SCALE}   actor_wide scale ${ACTOR_SCALE}"

run_arm() {
    local tag="$1" extra="$2"
    echo
    echo "################ arm $tag ($extra) ################"
    A3_POLICY_CKPT="$CKPT" A3_EXTRA_CONFIG="$extra" \
        bash scripts/recap/eval_conditioned.sh "dsrl_$tag" "" "$SEEDS" "$SHARDS"
}

run_arm tanh_wide  "noise_source=tanh_random noise_scale=$TANH_SCALE"
run_arm actor_wide "actor_ckpt=$ACTOR noise_scale=$ACTOR_SCALE"

echo
echo "################ comparison (five arms) ################"
"$PY" - <<'PY' | tee "$OUT/comparison_width.txt"
import glob, json, statistics as st

arms = {}
for tag in ("default", "tanh_random", "actor", "tanh_wide", "actor_wide"):
    rows = []
    for path in sorted(glob.glob(f"/root/autodl-tmp/datasets/eval_dsrl_{tag}/part_*/episode_labels.jsonl")):
        rows += [json.loads(line) for line in open(path)]
    if rows:
        arms[tag] = {r["seed"]: r["cookies_in_target"] for r in rows}

common = sorted(set.intersection(*[set(v) for v in arms.values()]))
print(f"paired seeds: {len(common)}")
print()
print(f"{'arm':14s} {'n':>3s} {'mean':>7s} {'sd':>6s} {'sem':>6s}  prior")
notes = {
    "default":     "N(0,1), width 0.763",
    "tanh_random": "tanh(N(0,1)), width 0.569",
    "actor":       "actor, width 0.510",
    "tanh_wide":   "1.34*tanh(N(0,1)), width 0.763",
    "actor_wide":  "1.495*actor, width 0.763",
}
for tag, scores in arms.items():
    v = [scores[s] for s in common]
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"{tag:14s} {len(v):3d} {st.mean(v):7.2f} {sd:6.2f} {sd/max(len(v),1)**0.5:6.2f}"
          f"  {notes[tag]}")

print()
print(f"{'contrast':30s} {'mean diff':>10s} {'sem':>6s} {'t':>6s}  wins/ties/losses")
def paired(a, b):
    d = [arms[a][s] - arms[b][s] for s in common]
    md = st.mean(d); sd = st.stdev(d) if len(d) > 1 else 0.0
    sem = sd / max(len(d), 1) ** 0.5
    w = sum(1 for x in d if x > 0); t_ = sum(1 for x in d if x == 0)
    print(f"{a + ' - ' + b:30s} {md:+10.2f} {sem:6.2f} {(md/sem if sem else 0):6.2f}"
          f"  {w}/{t_}/{len(d)-w-t_}")
for a, b in (("tanh_wide", "default"), ("actor_wide", "default"),
             ("tanh_wide", "tanh_random"), ("actor_wide", "actor"),
             ("actor_wide", "tanh_wide")):
    if a in arms and b in arms:
        paired(a, b)

print()
print("How to read the decisive contrast")
print("  `actor_wide - tanh_wide` is the only one that isolates the actor's information")
print("  with the width held fixed.  Positive and |t| >= 2: the actor learned something,")
print("  and §9.94's -12 was a parameterisation bug.  Near zero: the actor learned nothing")
print("  about which noise is better, and no amount of width fixing will rescue it.")
PY

echo
echo "=== done; summary at $OUT/comparison_width.txt ==="
