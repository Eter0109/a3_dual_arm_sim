#!/usr/bin/env bash
# Extract value-model features with several shards sharing one memmap.
#
#   bash scripts/recap/run_extract_features.sh [shards] [stride]
#
# Why sharded, and why the stride default is 5
# --------------------------------------------
# The bottleneck is **video decode, not the model**.  Measured on this box: reading frames
# through `LeRobotDataset` runs at 20 frames/s and adding the SigLIP forward does not change
# that (20 -> 16), so the GPU sits idle while `pyav` decodes three AV1 streams per frame.  Full
# coverage of 552,789 frames is therefore 7.7 hours on one process.
#
# Two consequences drove the defaults:
#
#   * decode parallelises, so shards help linearly -- but each shard loads its own copy of the
#     2B checkpoint (~6 GB with context), and the card is 32 GB, so 2 shards is the comfortable
#     maximum (3 is the ceiling; 8 would OOM).
#   * a stride costs nothing but label resolution.  The value target is **one number per
#     episode**, so consecutive frames are nearly redundant; stride 5 still leaves 300 samples
#     per 1500-frame episode, and it makes the advantage step operate on 250 ms windows, which
#     is a reasonable horizon for "did this step help".
#
# The memmap is created here rather than by a shard, so there is no race over "who writes the
# header first".
set -uo pipefail

SHARDS="${1:-2}"
STRIDE="${2:-5}"
DATASET="${DATASET:-/root/autodl-tmp/datasets/a3_recap_all}"
CKPT="${CKPT:-/root/autodl-tmp/pi05-runs/a3_clean_v9_30000/checkpoints/030000/pretrained_model}"
OUT="${OUT:-$DATASET/features_v9}"
WIDTH="${WIDTH:-8192}"
PY=/root/autodl-tmp/a3-collect/.venv/bin/python

cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1
export TMPDIR=/root/autodl-tmp/tmp A3_MAIN_ROOT=/tmp/a3-main

FRAMES=$("$PY" - "$DATASET" <<'PY'
import sys
import pyarrow.parquet as pq
t = pq.read_table(f"{sys.argv[1]}/meta/episodes/chunk-000/file-000.parquet")
print(int(max(t.column("dataset_to_index").to_pylist())))
PY
)

echo "=== extract: ${FRAMES} frames, ${SHARDS} shards, stride ${STRIDE} ==="
echo "    dataset ${DATASET}"
echo "    output  ${OUT}  (${WIDTH} wide)"

mkdir -p "$OUT"
rm -f "$OUT/features.npy" "$OUT/written.npy" "$OUT/index.json"

# Preallocate so the shards only ever open `r+`.  Doing it here, once, removes the race where
# two shards both decide the file is missing and both try to truncate it.
"$PY" - "$OUT" "$FRAMES" "$WIDTH" <<'PY'
import sys
from pathlib import Path

import numpy as np

out, frames, width = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
features = np.lib.format.open_memmap(
    out / "features.npy", mode="w+", dtype=np.float16, shape=(frames, width)
)
written = np.lib.format.open_memmap(out / "written.npy", mode="w+", dtype=bool, shape=(frames,))
features[:] = 0
written[:] = False
features.flush()
written.flush()
print(f"[prepare] allocated {frames} x {width} float16 = {frames * width * 2 / 1e9:.2f} GB")
PY

pids=()
for ((s = 0; s < SHARDS; s++)); do
    (
        "$PY" scripts/recap/extract_value_features.py \
            --dataset "$DATASET" --checkpoint "$CKPT" --output "$OUT" \
            --width "$WIDTH" --stride "$STRIDE" \
            --shard-index "$s" --shard-count "$SHARDS" \
            > "/root/autodl-tmp/logs/extract_shard$(printf "%02d" "$s").log" 2>&1
    ) &
    pids+=($!)
    echo "    shard $s launched"
done

echo "=== waiting on ${#pids[@]} shards (logs: /root/autodl-tmp/logs/extract_shard*.log) ==="
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done

"$PY" - "$OUT" "$FRAMES" <<'PY'
import sys
from pathlib import Path

import numpy as np

out, frames = Path(sys.argv[1]), int(sys.argv[2])
written = np.load(out / "written.npy")
filled = int(written.sum())
print(f"[extract] {filled}/{frames} rows filled ({100 * filled / frames:.1f} %)")
if filled < frames:
    # Not an error by itself -- a stride leaves rows unfilled on purpose -- but the trainer
    # must select on this mask, and a *truncated* run is indistinguishable from a strided one
    # without knowing the stride.  Print it so the two can be told apart.
    print("[extract] NOTE: unfilled rows exist; the trainer selects by written.npy")
PY

if [ "$failed" -ne 0 ]; then
    echo "!!! ${failed} shard(s) failed -- check the logs" >&2
    exit 1
fi
echo "=== done ==="
