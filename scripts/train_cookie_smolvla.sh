#!/bin/bash
#SBATCH --job-name=a3_cookie_train
#SBATCH --partition=rtx2080ti
#SBATCH --account=gpulab02
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/lab/haoq_lab/cse12311731/a3_dual_arm_sim/logs/train_cookie_%j.log

# Fine-tune SmolVLA on the overnight cookie-transfer dataset.
#
# Pass steps as $1 (default 20000) and batch size as $2 (default 4).
#
# The GPU node cannot reach the Hub, so everything is pre-fetched in .runtime and
# offline mode is forced. It also lacks AVX, so MuJoCo cannot be imported there --
# training only touches the dataset and torch, which is exactly why the package
# init resolves its simulation names lazily.

set -uo pipefail

STEPS="${1:-20000}"
BATCH_SIZE="${2:-4}"

PROJECT=/lab/haoq_lab/cse12311731/a3_dual_arm_sim
DATASET="${PROJECT}/outputs/datasets/a3_cookie_overnight"
BASE_MODEL="${PROJECT}/.runtime/smolvla_base"
OUTPUT="${PROJECT}/outputs/training/cookie_smolvla"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${PROJECT}/.runtime/hf"
export HF_DATASETS_CACHE="${PROJECT}/.runtime/hf_datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl

source /lab/haoq_lab/cse12311731/miniconda3/etc/profile.d/conda.sh
conda activate a3_sim
# Drop the login node's software-Mesa injection; unused here but harmful if kept.
unset LD_LIBRARY_PATH LIBGL_DRIVERS_PATH

cd "${PROJECT}"

echo "host      : $(hostname)"
echo "steps     : ${STEPS}"
echo "batch size: ${BATCH_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python - "${DATASET}" "${BASE_MODEL}" "${OUTPUT}" "${STEPS}" "${BATCH_SIZE}" <<'PY'
import sys
from pathlib import Path

from a3_dual_arm_sim.training import train_smolvla

dataset, base_model, output, steps, batch = sys.argv[1:6]

summary = train_smolvla(
    dataset_root=Path(dataset),
    repo_id="local/a3-cookie-overnight",
    base_model=Path(base_model),
    output_dir=Path(output),
    steps=int(steps),
    batch_size=int(batch),
    seed=1000,
    device="cuda",
    dry_run=False,
    job_name="a3_cookie_smolvla",
    save_freq=max(1, int(steps) // 10),
)
print()
print("checkpoint:", summary.get("checkpoint"))
PY
