# A3 dual-arm cookie transfer — clean v9 base (400 expert episodes)

400 expert demonstrations on a simulated dual-arm cookie-transfer scene, split evenly
between two task variants. This is the dataset the `a3_clean_v9_30000` π0.5 checkpoint was
fine-tuned on, and it is published so that checkpoint can be reproduced.

## The task

Two Franka-style 7-DoF arms with parallel-jaw grippers work over two bins. The source bin
holds 80 rectangular biscuits ("cookies"); the target is a slotted box. The policy has to
grasp cookies in groups, carry them across, and release them upright into the slots.

| variant | episodes | instruction |
|---|---|---|
| **2x5** | 200 | `Transfer 10 cookies from source column 1 into the target box in two batches of five.` |
| **2x10** | 200 | the per-episode plan sentence (see below) |

The 2x10 variant draws a plan `[n, 10-n, n, 10-n]` per episode and the instruction names it,
so no relabelling step is needed; the plan `n` is recoverable from the instruction string.
There are **11 distinct instruction strings** in `meta/tasks.parquet` — one for 2x5 and ten
for the 2x10 plans, which is what the benchmark samples. Selection took **20 per plan** so
the plan prompts are uniform rather than skewed like the source pools (where 190 of 440
episodes used `n=5` and the rarest plan had 20).

**Parsing gotcha.** Nine of the ten plans read
`... Grasp {n} cookies first, then {10-n} ...`, but the tenth plan is `n = 10`, i.e.
`[10, 0, 10, 0]`, and the collector phrased it differently because "grasp 0 cookies
first" is not a sentence:

```
Transfer 20 cookies from source column 1 into the 2x10 target box. Skip grasp 1 and
transfer 10 cookies with grasp 2 into target column 1. Skip grasp 3 and transfer 10
cookies with grasp 4 into target column 2. Release all 20 cookies upright in the
target box and leave 60 in the source box.
```

So `re.search(r"Grasp (\d+) cookies first", task)` finds only **9 of the 10** plans and
silently drops 20 episodes. Group by the full string, not by a `first`-grasp regex.

## Contents

| | |
|---|---|
| episodes | **400**, all `success = true` |
| frames | **312,470** at 20 fps (548 / 781 / 1,087 per episode, min/mean/max) |
| duration | 4 h 20 min of control time at 20 fps |
| size | 2.9 GB (video dominates) |
| layout | LeRobot **v3.0**: `meta/info.json`, `meta/episodes/chunk-000/*.parquet`, `meta/stats.json`, `meta/tasks.parquet`, `data/chunk-000/*.parquet`, `videos/<camera>/chunk-000/*.mp4` |

Per-step features (`meta/info.json`):

| key | type | notes |
|---|---|---|
| `observation.images.front` | video 256×256×3 | scene camera |
| `observation.images.left_wrist` | video 256×256×3 | eye-in-hand |
| `observation.images.right_wrist` | video 256×256×3 | eye-in-hand |
| `observation.state` | f32[16] | `[L joints (7), L gripper, R joints (7), R gripper]` |
| `observation.velocity` | f32[16] | same order, joint velocities + gripper rate |
| `observation.eef_pose` | f32[14] | 2 × (position 3 + quaternion 4) |
| `observation.force` | f32[18] | 2 × (wrist force 3, wrist torque 3) + 4 touch sensors + 2 gripper force sums |
| `action` | f32[16] | absolute joint position targets, same order as `observation.state` |

Two files are ours rather than LeRobot's:

* **`episode_metadata.jsonl`** — one JSON object per episode: `episode_index`, `seed`,
  `task`, `controller_type`, `source_action_mode`, `stored_action_mode`, `frames`,
  `success`.
* **`scene_config.yaml`** — the scene configuration the collector ran.

## Two epochs, and why

The 400 episodes come from **two different expert controllers**, 200 each:

| `controller_type` | count | task |
|---|---|---|
| `CookieCollectionPolicy` | 200 | 2x5 |
| `cookie_2x10_variable_grasp` | 200 | 2x10 |

This is deliberate and matters for how the set is split. 50 % of *episodes* is not 50 % of
*frames*: the 2x10 episodes average ~1,014 frames against ~549 for 2x5, so the gradient sees
**35.1 : 64.9** of frames (109,752 vs 202,718).

The episode order is also blocked, not interleaved: **`episode_index 0–199` are the 2x10
episodes and `200–399` are the 2x5 ones.** A sequential train/validation split along
`episode_index` therefore separates the two task variants completely and measures task
transfer rather than generalisation — **shuffle before splitting.**

## What this set is not

* **No `observation.history`.** The history window (`observation.history`, f32[480] = ten
  frames of `[q, q_dot, a_previous]`) is a *derived* column, produced by
  `scripts/add_history_window.py` in the simulator repository. The checkpoint above was
  trained on the derived set (`a3_clean_v9h`), not on this one. If you want to fine-tune
  with the same input contract, either derive the column yourself from
  `observation.state` / `observation.velocity` / `action` (the recipe is one shifted copy of
  `action` plus a ten-frame rolling window), or fine-tune a checkpoints whose config does not
  declare a history encoder.
* **No `observation.progress`.** Two progress channels were added to later variants; they
  are not here.
* **No policy rollouts.** Every episode is expert; the set exists as a clean base precisely
  so that none of the training signal comes from a model imitating its own stopping
  behaviour.

## Reproducing the checkpoint

The full pipeline lives in the simulator repository
(`git@github.com:Eter0109/a3_dual_arm_sim.git`, branch `vla-policy-interface-v2`):

```bash
# 1. build this dataset out of the merged source pools (200 x 2x5 + 20 x 10 plans)
python scripts/build_clean_v9.py --out .../a3_clean_v9 --repo-id local/a3-clean-v9

# 2. derive observation.history -> a3_clean_v9h
python scripts/add_history_window.py --source .../a3_clean_v9 --out .../a3_clean_v9h \
    --repo-id local/a3-clean-v9h --steps 10 --max-batches 4 --cookies-per-batch 5

# 3. fine-tune pi0.5 for 30,000 steps
bash scripts/multitask_v9_pipeline.sh 30000 8
```

`scripts/chain_v9.sh` runs all three in order and refuses to continue on a short build,
because a dataset with fewer episodes silently changes what "30,000 steps" means.

## Provenance and caveats

* All 400 episodes were accepted under the collector's own success contract
  (2x10: 20 cookies in the box, 60 left in the source; 2x5: 10 in the box). The scene's
  *evaluation* grading was relaxed after collection — a leaning cookie with one end on the
  floor now counts as placed — so a small number of these episodes would be graded
  differently today than when they were accepted.
* The boxes are randomised in x/y/yaw and the arm home pose in ±0.04 rad, keeping 20 mm
  between the bins. Cookie positions inside a column are **not** jittered per cookie: the
  dense packing cannot take it (independent leans interpenetrate the 6.33 mm-spaced
  neighbours and the solver ejects the group).
* Action space is absolute joint positions, not deltas. A consumer that expects deltas must
  convert.

## Licence

CC BY 4.0 — see `LICENSE`. Simulated data, no personal information.

## Citation

If this dataset is useful, cite the simulator repository:

```
@misc{a3_dual_arm_sim,
  title  = {A3 dual-arm cookie transfer: simulation, expert collection and π0.5 fine-tuning},
  author = {Eter0109},
  year   = {2026},
  url    = {https://github.com/Eter0109/a3_dual_arm_sim}
}
```
