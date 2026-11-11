#!/usr/bin/env bash
# Run the three arms of the RECAP stage-4 comparison and print them side by side.
#
#   bash scripts/recap/run_eval_suite.sh [seeds] [shards]
#
# The three arms, and what each one buys
# --------------------------------------
# | arm | checkpoint | suffix | answers |
# |---|---|---|---|
# | `r4b_pos` | 4B | `" Advantage: positive"` | does the conditioned policy beat its own base? |
# | `r4b_neg` | 4B | `" Advantage: negative"` | **sanity**: the wrong condition must not win |
# | `r4b_nos` | 4B | none | the 4A weights with no condition at all |
# | `v9` | v9 | none | did RECAP move anything against the baseline? |
#
# The negative arm is not decoration.  If `r4b_pos` and `r4b_neg` score the same, the sentence is
# not being read and the positive arm's number is not evidence of conditioning -- it is 4A's
# fine-tuning performance, which `r4b_nos` already measures.  §9.83 found this project's first
# advantage threshold to be backwards by exactly this kind of check, so it is worth the 55 minutes.
#
# All four share seeds 300+ and the same harness, so the arms are paired and the differences can
# be read as differences rather than as samples from four different rulers (§9.80).
set -euo pipefail

SEEDS="${1:-56}"
SHARDS="${2:-8}"
cd "$(dirname "$0")/../.."

V9=/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model
R4B=/root/autodl-tmp/pi05-runs/a3_recap_4b_10000/checkpoints/010000/pretrained_model
for ckpt in "$V9" "$R4B"; do
    [ -e "$ckpt/model.safetensors" ] || { echo "missing $ckpt"; exit 1; }
done

run() {
    local tag="$1" ckpt="$2" suffix="$3" n="$4"
    # Skip an arm that already has all its episodes.  Collection is the expensive part (an
    # episode is ~12 minutes and a full arm ~2 hours), so a later arm's bug must not cost the
    # earlier arms' work.  The count is checked rather than a marker file so a partially
    # collected arm is redone in full -- a half-arm silently averaged into a comparison is worse
    # than re-running it.
    # `|| true` inside the substitution is load-bearing: this script runs under `set -euo
    # pipefail`, and for an arm that has not run yet the glob matches nothing, so `cat` exits
    # non-zero and `pipefail` propagates it out of the assignment -- killing the suite at the
    # first arm that needs collecting.  It fails silently, right after printing the previous
    # arm's skip line.
    local have
    have=$(cat /root/autodl-tmp/datasets/eval_"$tag"/part_*/episode_labels.jsonl 2>/dev/null | wc -l || true)
    [ -n "$have" ] || have=0
    if [ "$have" -ge "$n" ]; then
        echo
        echo "################ $tag already has $have episode(s), skipping ################"
        return 0
    fi
    echo
    echo "################ $tag ($n seeds) ################"
    A3_POLICY_CKPT="$ckpt" bash scripts/recap/eval_conditioned.sh "$tag" "$suffix" "$n" "$SHARDS"
}

# The negative arm runs on a third of the seeds on purpose.  Its job is not to be compared
# numerically -- it is to answer "is the sentence being read at all", and 16 paired seeds answer
# that: if the wrong condition scores like the right one, the arm is not worth 56.
NEG_SEEDS=$(( SEEDS / 3 ))
[ "$NEG_SEEDS" -ge 8 ] || NEG_SEEDS=8

run r4b_pos "$R4B" " Advantage: positive" "$SEEDS"
run r4b_nos "$R4B" "" "$SEEDS"
run v9      "$V9"  "" "$SEEDS"
run r4b_neg "$R4B" " Advantage: negative" "$NEG_SEEDS"

echo
echo "################ summary ################"
/root/autodl-tmp/a3-collect/.venv/bin/python - "$SEEDS" <<'PY'
import glob, json, sys, statistics as st

seeds = int(sys.argv[1])
arms = ("r4b_pos", "r4b_neg", "r4b_nos", "v9")
data = {}
for arm in arms:
    rows = []
    for path in sorted(glob.glob(f"/root/autodl-tmp/datasets/eval_{arm}/part_*/episode_labels.jsonl")):
        rows += [json.loads(line) for line in open(path)]
    if rows:
        data[arm] = {r["seed"]: r for r in rows}

print(f"{'arm':10s} {'n':>4s} {'strict%':>8s} {'mean':>7s} {'sem':>6s}")
for arm in arms:
    if arm not in data:
        print(f"{arm:10s} {'--':>4s} {'--':>8s} {'--':>7s} {'--':>6s}")
        continue
    rows = list(data[arm].values())
    scores = [r["cookies_in_target"] for r in rows]
    n = len(rows)
    print(f"{arm:10s} {n:4d} {100 * sum(r['success'] for r in rows) / n:7.1f}% "
          f"{st.mean(scores):7.2f} {st.stdev(scores) / n ** 0.5:6.2f}")

# Paired differences over the seeds every pair has in common.  A paired test is the only honest
# comparison here: the bridge is stochastic, so two arms on different seeds differ by more than
# the effect, and this project has already been misled once by an unpaired n=8.
print()
print("paired differences (cookies), common seeds only:")
for a, b in (("r4b_pos", "r4b_nos"), ("r4b_pos", "r4b_neg"), ("r4b_pos", "v9"), ("v9", "r4b_nos")):
    if a not in data or b not in data:
        continue
    common = sorted(set(data[a]) & set(data[b]))
    if len(common) < 2:
        continue
    diff = [data[a][s]["cookies_in_target"] - data[b][s]["cookies_in_target"] for s in common]
    mean = st.mean(diff)
    sem = st.stdev(diff) / len(diff) ** 0.5
    wins = sum(1 for d in diff if d > 0)
    losses = sum(1 for d in diff if d < 0)
    print(f"  {a} - {b}: {mean:+.2f} +/- {sem:.2f} (n={len(common)}, W{wins}/L{losses})")
PY
