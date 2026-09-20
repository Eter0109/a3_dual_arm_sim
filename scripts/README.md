# Scripts

The tools that are meant to be run by hand, in the order a dataset is built.  The
library itself lives in `src/a3_dual_arm_sim/`; anything here is a driver for it.

Two rules decide what belongs in this directory and what does not:

* **Shareable only.**  A script belongs here when someone else can run it on their
  own machine and get a meaningful result.  Notes to self, one-off diagnostics, and
  probe scripts belong in `logs/`, which is git-ignored.
* **Site facts are overridable.**  A path to a conda installation, a cluster
  partition, or a checkpoint is the one thing that differs between machines, so it
  is a default that an environment variable or argument can replace -- never a
  hard-coded constant.

## The pipeline

Collect, merge, train, evaluate, record.  Each stage's output is the next stage's
input, and the whole chain is re-runnable from any point.

| Stage | Script | What it does |
| --- | --- | --- |
| 1. Collect | `collect_cookie.sh` | Runs `a3-sim collect` in parallel shards, one dataset per shard |
| 2. Merge | `merge_cookie_shards.py` | Concatenates the shards and re-emits one summary |
| 3. Train | `train_cookie_smolvla.sh` | Slurm job that fine-tunes SmolVLA on the merged dataset |
| 4. Evaluate | `evaluate_cookie_policy.py` | Scores a checkpoint over several episodes |
| 5. Record | `render_cookie_policy_video.py` | Writes a checkpoint's rollout to an mp4 |

### 1. Collect

```bash
bash scripts/collect_cookie.sh <output_name> <scene> [shards] [episodes] [seed_base]

bash scripts/collect_cookie.sh a3_cookie_overnight a3_cookie_same_column
```

Shards must not share an output root: LeRobot writes one parquet file and one video
set per dataset, so concurrent writers need separate directories.  Seeds are
interleaved across shards, so the union is still one clean sweep.

`<scene>` names a registered scene, and which scenes can actually collect is not
obvious -- the batch scenes do, the single-Cookie one does not any more.  See
`registered_scenes()` in `src/a3_dual_arm_sim/collection.py` and each builder's
docstring before picking one.

### 2. Merge

```bash
python scripts/merge_cookie_shards.py \
  --shards outputs/datasets/a3_cookie_overnight/shard_* \
  --repo-id local/a3-cookie-overnight-merged \
  --output outputs/datasets/a3_cookie_overnight_merged
```

Wraps LeRobot's `aggregate_datasets`, then renumbers `episode_index` sequentially
and re-emits one `a3_episode_metadata.jsonl` and `collection_summary.json` for the
whole run.

### 3. Train

```bash
sbatch scripts/train_cookie_smolvla.sh 20000 4
```

A Slurm job, because training needs a GPU and MuJoCo does not run there.  The
`#SBATCH` partition and account are site facts; edit them or override the script's
variables for another cluster.

### 4. Evaluate

```bash
python scripts/evaluate_cookie_policy.py \
  --checkpoint outputs/training/cookie_smolvla/checkpoints/last/pretrained_model \
  --episodes 5 --first-seed 0 --output outputs/eval/cookie.json
```

Runs on the CPU node: MuJoCo aborts on the GPU worker, and LeRobot's action queue
makes CPU inference affordable.

### 5. Record

```bash
python scripts/render_cookie_policy_video.py --checkpoint <path> --output outputs/videos/rollout.mp4
```

## Visualisation and diagnosis

Same commands, no pipeline position -- each answers one question.

| Script | The question it answers |
| --- | --- |
| `render_cookie_batch_video.py` | What does the scripted batch expert actually do?  Composites the scene camera with the policy cameras and overlays the live Cookie counts |
| `plot_rollout_trace.py` | Is the policy following the demonstration, or cycling through it?  Charts the phase trace against step index, Pillow only |
| `measure_render_consistency.py` | Does `--fast-render` change the pixels?  Renders one state twice, with and without the shadow and reflection passes |

`measure_render_consistency.py` exists because every dataset is collected with
`--fast-render` and every evaluation and recording path calls `use_fast_render()`
unconditionally.  Training and inference agree only if the flag actually changes
the pixels, and it is worth being able to check that the agreement is real rather
than assumed.

## Conventions

* `MUJOCO_GL=egl` is set by the scripts that render.  On a headless node the
  software rasteriser needs `LD_LIBRARY_PATH` and `LIBGL_DRIVERS_PATH` pointing at
  a Mesa build; see the main `README.md`.
* `--fast-render` must match between collecting a dataset and evaluating a
  checkpoint trained on it.  A mismatch is silent: the model sees pixels it was
  not trained on and simply performs worse.
* Every script here sets its flags unconditionally rather than exposing both
  options, so the mismatch above cannot be introduced by forgetting an argument.
