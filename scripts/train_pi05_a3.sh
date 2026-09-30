#!/usr/bin/env bash
# Fine-tune pi0.5 (lerobot/pi05_base) on an A3 cookie dataset.
#
#   bash scripts/train_pi05_a3.sh <dataset_root> <output_dir> [steps] [batch] [name]
#
# The single-GPU constraints this script exists to satisfy
# --------------------------------------------------------
# pi0.5 is PaliGemma-2B (~2B) plus a ~300M flow-matching action expert.  Full fine-tuning
# does not fit: bf16 weights are 4.6 GB, gradients another 4.6 GB, and AdamW's fp32
# moments plus master weights another ~18 GB -- about 28 GB before a single activation, on
# a 32 GB card.  So `--policy.train_expert_only=true` is not a preference here, it is what
# makes the run possible: it freezes all of PaliGemma (`self.paligemma.eval()` and
# `requires_grad = False` on every parameter) and trains the action expert and the
# projections, which is also what the openpi authors recommend for π0.5 adaptation.
#
# The rename map
# --------------
# The dataset's cameras are A3's names and the checkpoint's are openpi's:
#
#   observation.images.front             -> observation.images.base_0_rgb
#   observation.images.left_wrist        -> observation.images.left_wrist_0_rgb
#   observation.images.right_wrist       -> observation.images.right_wrist_0_rgb
#
# `lerobot-train` applies this in two places -- `make_policy(..., rename_map=...)` for the
# feature keys, and the preprocessor's `rename_observations_processor` for the batches --
# which is why it is a flag rather than a dataset edit.  Rewriting 26 GB of videos to
# change three strings would be the alternative, and it would also destroy the dataset's
# value as something GR00T can still read.
#
# Why the statistics come from the dataset rather than the checkpoint
# ------------------------------------------------------------------
# `lerobot/pi05_base` ships `policy_preprocessor.json` with `features: {}` -- no
# normalization statistics, because it is a base model and the statistics belong to
# whoever fine-tunes it.  `lerobot_train.py` therefore injects `dataset.meta.stats` into
# the normalizer, and PI05Config's `normalization_mapping` asks for QUANTILES, which reads
# `q01`/`q99`.  A3's dataset already carries both, so no extra pass over the data is
# needed.  This is the same substitution the zero-shot manifest makes, except that here it
# is the intended path rather than a workaround: after fine-tuning, A3's range *is* the
# model's range.
#
# Checkpoints
# -----------
# 15 GB per checkpoint at the default interval.  `save_freq` is 2000 here rather than the
# library's 20000 so an interrupted night's run leaves something usable, and
# `save_total_limit` keeps the three most recent.

set -euo pipefail

DATASET_ROOT="${1:?usage: train_pi05_a3.sh <dataset_root> <output_dir> [steps] [batch] [name]}"
OUTPUT_DIR="${2:?usage: train_pi05_a3.sh <dataset_root> <output_dir> [steps] [batch] [name]}"
STEPS="${3:-20000}"
BATCH="${4:-8}"
NAME="${5:-a3_pi05}"

PYTHON="${A3_PI05_PYTHON:-/root/autodl-tmp/a3-collect/.venv/bin/python}"
BASE="${A3_PI05_BASE:-lerobot/pi05_base}"
# Resolved from this script's own location, so the launcher is found no matter which
# directory the training is started from.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Offline, deliberately.  The tokenizer resolves through the cache entry that
# `setup_paligemma_tokenizer_cache.py` populates, and the gated name is only *partly*
# cached: transformers also asks for `chat_template.jinja`, which the ungated mirror does
# not carry.  Online, that request goes to the mirror and comes back a 403 -- so the run
# dies on a file that is optional.  Offline it is not requested, and the cached
# tokenizer.json / tokenizer.model are what get used.
#
# Everything else this needs is cached too (the 14.5 GB checkpoint), so nothing here
# depends on network access.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# The checkpoint's preprocessor names `google/paligemma-3b-pt-224`, which is gated, and
# `lerobot_train.py` builds the processor pipeline itself with no CLI hook to redirect the
# tokenizer.  So the *name* has to resolve: the setup script populates an HF cache entry
# for it from an ungated mirror of the same files.  Run it once; it is idempotent, and
# `--check-only` verifies the resolution without writing anything.
#
#   python scripts/setup_paligemma_tokenizer_cache.py
#
# The inference adapter takes the other route -- it passes the local directory directly --
# so this is only needed for training.

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "pi0.5 fine-tune"
echo "  base        : ${BASE}"
echo "  dataset     : ${DATASET_ROOT}"
echo "  output      : ${OUTPUT_DIR}"
echo "  steps       : ${STEPS}"
echo "  batch       : ${BATCH}"
echo "  expert-only : yes (PaliGemma frozen)"
echo

# Pre-flight on the normalization statistics, because getting them wrong is not a
# crash -- it is a loss of 1e12 at step 1 that reads as divergence.  π0.5 normalises with
# QUANTILES, so it divides by `q99 - q01`; a dimension whose stored pair is degenerate
# (the A3 dataset stores `q01 == q99 == mean` for the right arm) divides by `eps = 1e-8`
# instead, and the loss is then six orders of magnitude too high.
#
# The repair recomputes the quantiles from the recorded rows.  It is idempotent, it only
# touches `action` and `observation.state`, and it leaves `meta/stats.json.bak` behind the
# first time, so running it unconditionally here costs nothing after the first pass.
# The heredoc is quoted, so the shell expands nothing inside it; the two paths arrive as
# arguments instead.  Unquoting it would mean shell variables were substituted into Python
# source, which works until a path contains something the shell treats specially.
"${PYTHON}" - "${DATASET_ROOT}" "${PROJECT}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[2]) / "src"))
from a3_dual_arm_sim.statistics import recompute_quantiles

report = recompute_quantiles(Path(sys.argv[1]), apply=True)
for feature, info in report["features"].items():
    repaired = info["repaired_dimensions"]
    constant = info["constant_dimensions"]
    note = f"repaired {repaired}" if repaired else "nothing to repair"
    print(f"  stats {feature:<20} {note}" + (f"  (constant: {constant})" if constant else ""))
if report["backup"]:
    print(f"  original statistics saved to {report['backup']}")
PY
echo

# A bash array rather than a backslash-continued command, because the arguments below
# carry comments and a comment inside a `\`-continued command silently truncates it: the
# escaped newline joins the lines, the `#` then starts a comment, and every argument after
# it is dropped.  That happened once here -- the run proceeded with `--steps` at its
# 100000 default and no `--rename_map` -- and it is invisible in the log, because nothing
# reports which arguments were absent.  Inside parentheses each line is its own word, so
# the comments are inert.
ARGS=(
    # `policy.repo_id` is required by `TrainPipelineConfig.validate` even when nothing is
    # pushed.  It lands in the saved `train_config.json` as a provenance string, so it is
    # set to something meaningful rather than a placeholder reading like a real Hub id.
    --policy.repo_id="local/${NAME}"

    --policy.path="${BASE}"
    --dataset.repo_id="local/${NAME}"
    --dataset.root="${DATASET_ROOT}"
    --output_dir="${OUTPUT_DIR}"
    --job_name="${NAME}"
    --policy.device=cuda

    # bf16 weights, and the expert-only freeze that makes the run fit.  See the header.
    --policy.dtype=bfloat16
    --policy.train_expert_only=true
    --policy.gradient_checkpointing=true

    # `pyav`, not the default.  `get_safe_default_codec()` picks `torchcodec` whenever the
    # package is installed -- it only calls `importlib.util.find_spec` -- and here
    # torchcodec's shared libraries do not load (`libavutil.so.58: cannot open shared
    # object file`, because the container's FFmpeg has a different soname).  The symptom is
    # a DataLoader worker dying with "Caught RuntimeError in DataLoader worker process 0"
    # behind a wall of FFmpeg-version noise, which says nothing about which backend to pick.
    #
    # pyav is also the backend that reads this dataset's codec: A3 records AV1 (libsvtav1),
    # and neither decord nor opencv can decode it.
    --dataset.video_backend=pyav

    # Dataset key -> policy key.  `lerobot_train.py` applies this both to the policy's
    # feature keys and to the batches, which is why it is a flag rather than a dataset
    # edit: rewriting 26 GB of videos to change three strings would be the alternative,
    # and it would also destroy the dataset's value as something GR00T can still read.
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}'

    --batch_size="${BATCH}"
    --steps="${STEPS}"
    --num_workers=4
    --save_freq=2000
    --save_checkpoint=true
    --log_freq=20
    # No simulator evaluation during training: this run is judged by the A3-side comparison
    # afterwards, on the same ruler as every other model.  Leaving it at its default would
    # also require an environment this container does not have.
    --eval_freq=0
    --wandb.enable=false
    # Nothing is pushed.  The default is True, and the failure is not at the start but at
    # the very end, after the training is done and the checkpoint is written: the run
    # exits 1 on `OfflineModeIsEnabled: Cannot reach .../api/repos/create`, which reads as
    # a failed training when the training actually succeeded.
    --policy.push_to_hub=false
    --seed=1000
)

"${PYTHON}" "${PROJECT}/scripts/train_pi05_launcher.py" "${ARGS[@]}" "${@:6}"
