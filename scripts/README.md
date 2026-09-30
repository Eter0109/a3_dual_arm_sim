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
| `a3_cookie_transfer` | **no** | The original single-Cookie workflow; its right arm can no longer reach the tabletop target box, so it accepts nothing |

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

## External models

The bridge lets a model from another project drive this environment without sharing
its interpreter -- see `docs/external-policy-bridge.md`.  Two scripts belong to it,
and the second is meant to be copied *out* of this repository.

| Script | What it does |
| --- | --- |
| `make_replay_recording.py` | Records an action sequence (`--source hold` or `--source dataset`) for the bridge's replay transport, so the whole pipeline can be exercised with no model installed |
| `a3_bridge_server.py` | The model side of the wire protocol.  Imports nothing from `a3_dual_arm_sim` and nothing beyond numpy, because it runs in the model's environment, not this one |
| `render_bridge_policy_video.py` | Records an external model's rollout to an mp4, with the behaviour metrics the comparison uses drawn onto the frame |

```bash
# A recording, then a rollout that uses it
a3-py scripts/make_replay_recording.py --source hold --steps 200 \
    --output configs/policies/replay_hold.json
a3-sim run --scene sandbox --policy-manifest configs/policies/replay_hold.yaml --steps 100

# Audit a manifest for a model that is not installed anywhere
a3-sim policy check configs/policies/subprocess_template.yaml

# Watch what a model actually does, with its numbers on the frame
python scripts/render_bridge_policy_video.py \
    --manifest configs/policies/gr00t_bimanual.yaml \
    --output outputs/videos/gr00t_sandbox.mp4 --steps 200

# The cookie task: one source box, one target box
python scripts/render_bridge_policy_video.py \
    --manifest configs/policies/gr00t_bimanual.yaml --scene a3_cookie_same_column \
    --output outputs/videos/gr00t_cookie.mp4 --steps 400

# Put several models on one ruler
a3-sim compare --scene sandbox \
    --manifest configs/policies/cosmos_policy_aloha.yaml \
    --manifest configs/policies/gr00t_bimanual.yaml \
    --steps 200 --output report.json
```

`--scene` takes `sandbox` or any registered collection scene
(`a3_cookie_same_column`, `a3_cookie_batch`, `a3_cookie_two_box`,
`a3_cookie_transfer`, `a3_grasp`).  A registered scene is built through its own
builder rather than by repeating its task config here, because that config decides
what counts as success -- a copy would drift from the one the datasets were
collected with.

`--visual-style detailed` gives the Cookies their own materials and dims the
camera-locked headlight, so the packed blades read as separate objects instead of
one striped block.  It changes no physical parameter and is **not** the default:
the rendered pixels are a dataset contract, and every collected image so far was
produced with `plain`.  Measured on `a3_cookie_same_column`, the two styles differ
by 6.80 mean pixel value while every mass, geom size, contact bit and joint range is
identical.

```bash
python scripts/render_bridge_policy_video.py \
    --manifest configs/policies/gr00t_bimanual.yaml \
    --scene a3_cookie_same_column --visual-style detailed \
    --output outputs/videos/gr00t_cookie_detailed.mp4 --steps 200
```

The layout is left unrandomised by default (`--randomize` opts in): two models
recorded on different arrangements could not be compared, and a video is usually
watched to see what a model does with a known scene.

The adapter a model needs is one file with two functions, `load(config)` and
`infer(model, request)`; the replay recording's format and the adapter's contract
are both documented in the scripts themselves.

### What a model was asked is not the same as what the scene asked for

A model with `prompt_kind: fixed_label` ignores the scene's task string: Cosmos
Policy looks its prompt up in a precomputed table, and the released ALOHA
checkpoint carries six labels, none about cookies.  Rendered on
`a3_cookie_same_column`, GR00T is driven with the cookie instruction while Cosmos
Policy is still driven with "pick up the cube and place it in the bowl".

The frame shows each model's own `*_task` self-report for exactly this reason, so
the difference is visible in the video rather than only in a manifest.

### Episode length is not a detail

Behaviour metrics taken over a short episode are dominated by the initial transient:
the arm starts from its home pose, so a model's first commands are necessarily far
from anything useful, and the first inference also carries kernel selection and
warmup.  Measured on the same two models, going from 32 to 200 steps changed
Cosmos Policy's correction rate from 0.66 to 0.17 and GR00T's reported inference
latency from 422 ms to 196 ms.

A comparison therefore has to state its episode length, and a short one answers a
different question -- "how does the model behave the moment it takes over" -- than a
long one.

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

## Rebuilding an external model

Scripts that exist to put a model back on a machine that does not have it live next
to the model they belong to, under `integrations/<model>/`, together with the
adapter that runs in the model's own interpreter and a README recording what each
step costs and which shortcuts turned out to be wrong.  See
`integrations/cosmos_policy/` for the worked example: a parallel resumable download
for a 45 GB file, and the Hugging Face cache layout that
`from_pretrained("google-t5/t5-11b")` actually resolves against.

They are not in `scripts/` because they import nothing from `a3_dual_arm_sim` and are
run by hand, on the machine that needs the model -- the same reason
`a3_bridge_server.py` is written to be copied out of this repository.
