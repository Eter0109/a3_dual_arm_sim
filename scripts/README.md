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
obvious.  The registered set is:

| Scene | Collects? | What one episode does |
| --- | --- | --- |
| `a3_cookie_same_column` | yes | Five-Cookie batches from one source column, twice, into one box |
| `a3_cookie_batch` | yes | The same, taking the second batch from a different column |
| `a3_cookie_two_box` | yes | Fills box A, pushes it clear, carries box B into the station, fills B |
| `a3_grasp` | yes | Lifts a single cube |

`a3_cookie_transfer` used to be here and is **retired**: its single-Cookie expert
cannot complete on any layout the repository ships, so it is no longer registered.
`cookie_transfer_scene`'s docstring has the measurements, and `a3-sim collect-cookie`
(the old entry point) still exists but prints a warning and will accept nothing.

A two-box episode is roughly twice as long as a single-box one, so budget for it:
the reference run took about 2600 control steps and 18 minutes on a loaded login
node before the second fill.

See `registered_scenes()` in `src/a3_dual_arm_sim/collection.py` and each builder's
docstring for the details, including which success criterion each scene uses.  One
of them does not use the environment's: the relay decides success itself, because a
two-box fill is not expressible as the single-bin task config.

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
| `render_collection_video.py` | What did the collected episodes actually do?  One watchable video per episode plus a combined file, with seed and outcome overlaid |
| `render_cookie_batch_video.py` | What does the scripted batch expert actually do?  Composites the scene camera with the policy cameras and overlays the live Cookie counts |
| `plot_rollout_trace.py` | Is the policy following the demonstration, or cycling through it?  Charts the phase trace against step index, Pillow only |
| `measure_render_consistency.py` | Does `--fast-render` change the pixels?  Renders one state twice, with and without the shadow and reflection passes |

`render_collection_video.py` is worth reaching for before trusting a summary:
collection writes the three policy cameras because that is what a policy trains on,
but they are 256x256 and stored per camera with every episode concatenated into one
file.  A summary says six episodes succeeded; only the pictures say whether they
succeeded the same way twice.  The overlay carries the seed and outcome, because a
run of six successes at six seeds is a different artifact from six at one seed and
the file names alone do not distinguish them.

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
