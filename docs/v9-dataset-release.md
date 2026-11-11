# The clean v9 base: what is in it, and how to release it

`a3_clean_v9` is the 400-episode set that `pi05-runs/a3_clean_v9_30000` was fine-tuned on.
It is a *different* dataset from the second batch (see `b2-dataset-release.md`) and it is
released for a different reason: not to add conditions, but to make that checkpoint
reproducible.  Everything here was verified against the built bundle, not against the
collection plan.

## 1. What the 400 episodes are

Two hundred from each of two existing pools, 20 per plan prompt:

| source pool | episodes | controller | task |
|---|---|---|---|
| `a3_b2_main_aligned_v2_merged`, frames ≥ 400 | 200 | `CookieCollectionPolicy` | 2x5 |
| `a3_2x10_n5` + `a3_2x10_v1`, 20 per plan | 200 | `cookie_2x10_variable_grasp` | 2x10 |

`scripts/build_clean_v9.py` does the selection; the argument for the selection is its
docstring and is not repeated here.  What matters for release is that **the set has three
properties that are invisible from the LeRobot metadata and each one misleads a consumer**:

**Both task variants in equal episodes is not equal frames.**  Measured on the built set:
202,718 frames of 2x10 against 109,752 of 2x5, i.e. **64.9 : 35.1**.  A trainer that
balances by frame weights the two tasks differently from one that balances by episode, and
the build script chose the latter deliberately.

**Episodes are blocked, not interleaved.**  `episode_index 0–199` is all 2x10 and
`200–399` is all 2x5.  A sequential train/validation split along `episode_index` therefore
separates the two task variants completely, and measures task transfer rather than
generalisation.  (The second batch has the same defect with a different block structure —
20 blocks of 50-by-condition — so this is a recurring property of how these sets are built,
not an accident of this one.)

**The tenth 2x10 plan does not parse like the other nine.**  Nine instructions read
`... Grasp {n} cookies first, then {10-n} ...`, but the `n = 10` plan is `[10, 0, 10, 0]`
and the collector had to phrase it as "grasp 0 cookies first" otherwise:

```
... Skip grasp 1 and transfer 10 cookies with grasp 2 into target column 1.
Skip grasp 3 and transfer 10 cookies with grasp 4 into target column 2. ...
```

So `re.search(r"Grasp (\d+) cookies first", task)` finds **9 of the 10 plans** and silently
drops 20 episodes.  Group by the full instruction string.  This was found by running the
bundle's own example script and noticing it printed nine plans where the card claimed ten.

## 2. Size, layout, schema

| | |
|---|---|
| episodes | 400, all `success = true` |
| frames | 312,470 at 20 fps (548 / 781 / 1,087 min/mean/max); 4 h 20 min of control time |
| size | 2.9 GB, 21 files (video dominates: 7 mp4) |
| layout | LeRobot v3.0 — `meta/{info,stats,tasks}.json|parquet`, `meta/episodes/chunk-000/`, `data/chunk-000/`, `videos/<camera>/chunk-000/` |

Features, as `meta/info.json` declares them:

| key | type | notes |
|---|---|---|
| `observation.images.front` | video 256×256×3 | |
| `observation.images.left_wrist` | video 256×256×3 | |
| `observation.images.right_wrist` | video 256×256×3 | |
| `observation.state` | f32[16] | `[L q (7), L gripper, R q (7), R gripper]` |
| `observation.velocity` | f32[16] | same order |
| `observation.eef_pose` | f32[14] | 2 × (pos 3 + quat 4) |
| `observation.force` | f32[18] | 2 × (force 3 + torque 3) + 6 sensor channels |
| `action` | f32[16] | absolute joint positions, same order as `observation.state` |

**This set has no `observation.history`.**  The checkpoint was trained on the *derived* set
`a3_clean_v9h`, where `scripts/add_history_window.py` adds
`observation.history` f32[480] (ten frames of `[q, q̇, a_previous]`) and
`observation.progress`.  A consumer handed `a3_clean_v9` alone cannot reproduce the
checkpoint's input contract without deriving those columns.  The card says so; the bundle
does not include the derived set because it is a different (and much larger) artefact.

There is also **no `collection_summary.json`**, which the second-batch bundle has.  That file
recorded the acceptance contract and randomisation for a single collection run; v9 is a
*selection* from two pools, so the honest equivalent is the two source scene configs, which
are shipped verbatim as `scene_config_2x5.yaml` and `scene_config_2x10.yaml`.  The umbrella
`scene_config.yaml` is a prose index, not a merged config — merging them would be a lie,
since the two halves have different gripper pads and different chamfers.

## 3. The bundle, as built

`/root/autodl-tmp/release/a3-dual-arm-cookie-clean-v9` — 21 files, 2.9 GB, all 21 hashed in
`MANIFEST.sha256`, bulk hard-linked so staging cost no extra disk:

```
README.md                     the card: task, prompt, both variants, feature table,
                              the three gotchas, caveats, licence, provenance
LICENSE                       CC BY 4.0
episode_metadata.jsonl        renamed sidecar (400 lines; the only place `seed` lives)
scene_config_2x5.yaml         copy of configs/cookie_batch_main_aligned.yaml
scene_config_2x10.yaml        copy of configs/cookie_2x10_recovery.yaml
scene_config.yaml             prose index of the two above; not a merged config
examples/load_example.py      loads the set, prints controller/frame/plan breakdown,
                              reports the block structure, decodes a frame
examples/train_pi05_example.sh  the recipe
upload_modelscope.sh          same script as the second-batch bundle, with the commit
                              message derived from the card instead of hard-coded
MANIFEST.sha256               hashes of the other 21
data/ meta/ videos/           the LeRobot dataset, unmodified
```

Verification actually performed before sealing, in this order:

1. `examples/load_example.py` against the bundle: 400 episodes, 312,470 frames,
   11 instructions, all 10 plans recovered, `LeRobotDataset` constructs, `observation.state`
   and `action` are `(16,)`, one frame decodes from `front`.
2. All three cameras at three positions (`0`, `156235`, `312469`): shapes `(3,256,256)`,
   `min 0.000 max 1.000`, `std 0.18–0.26`, ≥186 distinct values.  This second check existed
   because the first printed `mean = 0.3` for all six camera-position pairs, which is what a
   constant image would look like; the `.1f` rounding was the only thing wrong, but the
   lesson from the DSRL work applies — a measurement that could equally be measuring nothing
   has to be sharpened until it can distinguish the two.
3. `sha256sum -c MANIFEST.sha256` → 21 OK, 0 failed, exit 0.  (The manifest was regenerated
   after `upload_modelscope.sh` was edited, because it had been written before that edit and
   a stale manifest is worse than no manifest.)

## 4. Releasing it — done

**Published: <https://www.modelscope.cn/datasets/yunsx114/a3-dual-arm-cookie-clean-v9>**

```bash
export MODELSCOPE_API_TOKEN=...          # modelscope.cn -> 个人中心 -> 访问令牌
cd /root/autodl-tmp/release/a3-dual-arm-cookie-clean-v9
bash upload_modelscope.sh <owner>/a3-dual-arm-cookie-clean-v9
```

The run took 4.5 min for 2.9 GB at 8 workers, exit 0 on the first attempt.  No token needed to
be set: the credentials `modelscope login` had persisted under `~/.modelscope/credentials` were
a full-account token, so `whoami` succeeded and the script could create the repository itself
(no `--skip-create`).  That is the opposite of the fine-grained-token case in
`b2-dataset-release.md` §5, and `whoami` succeeding is what distinguishes them.

Verified after the upload, not before it:

| check | result |
|---|---|
| remote file count and sizes | 21 files, 2.83 GB, every byte count equal to local |
| `README.md` location | repurposed root (`/tmp/ms-verify/README.md`), hash `3d13b434…` = local |
| hash spot-check on 4 files | `meta/episodes/*.parquet`, `meta/tasks.parquet`, `examples/load_example.py`, `upload_modelscope.sh` — 4/4 match `MANIFEST.sha256` |

Two defects were found in this step, both in the inherited script rather than the data:

**`upload_modelscope.sh` executed `modelscope` by accident.**  The line

```bash
echo "  `modelscope login` persisted under ~/.modelscope/credentials"
```

is inside double quotes, so the backticks are a *command substitution*: on a machine where the
client lives only in `/root/autodl-tmp/ms-venv`, every run printed
`upload_modelscope.sh: line 57: modelscope: command not found` into a log that otherwise said
the upload succeeded.  Harmless to the transfer (it was stderr noise) and fixed by switching to
single quotes.  The script was edited *after* the first upload, so `upload_modelscope.sh` and
`MANIFEST.sha256` were re-sealed and re-uploaded — the incremental path took 9 s (2 pending,
20 skipped), which is the resume behaviour the script was written for, now demonstrated rather
than assumed.

**`right_wrist` has one mp4 where the other cameras have three.**  This looked like missing
data and is not: LeRobot splits video files by *size*, not by episode count, and
`right_wrist` totals 402 MB against the 500 MB threshold, while `front` (1011 MB) and
`left_wrist` (1396 MB) cross it.  Its single file carries timestamps out to 15,623 s, i.e. all
312,470 frames, and frame 312,469 decodes from it.  A consumer counting files per camera will
think the set is broken; the card could say so.


## 5. What is not released, and why

| | |
|---|---|
| `a3_clean_v9h`, `a3_clean_v9h200` | derived, and their videos are hard-linked to v9's, so publishing them would triple the video bytes for a column a consumer can compute |
| `pi05-runs/a3_clean_v9_30000` | the checkpoint itself; this document exists so the dataset half of reproducing it is public |
| the two source pools | they carry the defects `build_clean_v9.py` was written to escape, and publishing them would invite exactly the sampling work the v9 selection already did |
