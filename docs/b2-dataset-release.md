# The second-batch dataset: what is in it, and how to release it

1000 expert episodes on the A3 dual-arm cookie scene, collected to close one specific hole:
every earlier episode was "start empty, do batch 0, do batch 1", so a policy met the second
batch only from the state the *expert* leaves behind -- five perfectly placed Cookies -- which
is a state a policy never produces itself.  These episodes meet the second batch four
different ways, which is the point of the set.

Everything below was read out of the dataset's own files
(`meta/info.json`, `meta/episodes/`, `a3_episode_metadata.jsonl`, `collection_summary.json`).

## 1. The four conditions

Exactly 250 episodes each, assigned by `seed % 4` (that is the scene's rule, and since the
seed travels in the episode metadata the condition is derivable after the fact).

| mode | episodes | frames (min/mean/max) | what the episode contains |
|---|---|---|---|
| `full` | 250 | 530 / 561 / 874 | the whole task: empty box, both batches done by the expert |
| `prefill` | 250 | 292 / 303 / 321 | **starts at the second batch**: five Cookies already in the box, imperfectly |
| `perturb` | 250 | 530 / 558 / 599 | the whole task, and the five just placed are nudged at the moment of release |
| `prefill_perturb` | 250 | 293 / 305 / 325 | `prefill` with the wider imperfections |

The frame counts are the tell: the two `prefill` variants are ~300 frames because the first
batch is teleported into the box rather than performed by the expert (a rollout that has to be
teleported, 550-870).

What "imperfectly" is drawn from, in `prefill`:

* a **single coherent lean** for the whole group, uniform in `[0, 4°]` (`prefill_perturb`: `[0, 6°]`),
  at a drawn azimuth.  One lean for all five rather than five independent draws, because the
  column's slots are 6.33 mm apart against 6.33 mm-thick Cookies: independent leans
  interpenetrate the neighbours and the solver explodes the group out of the box (measured).
* yaw uniform in `±1°` (`±1.5°`), and per-Cookie position jitter `±0.2 mm` (`±0.3 mm`).

`perturb` re-applies, to the five just placed, a coherent lean of up to 3° at a drawn azimuth
plus per-Cookie jitter -- the state a second batch meets when the first one was inserted next
to imperfect neighbours.

## 2. Size, layout, schema

* **3.7 GB** total, of which video is 3.66 GB: `front` 1.2 GB, `left_wrist` 1.9 GB,
  `right_wrist` 560 MB.
* 1000 episodes, **432,327 frames**, 20 fps → 21.6 s per episode on average.
* LeRobot **v3.0** layout: `meta/info.json`, `meta/episodes/chunk-000/*.parquet`,
  `meta/stats.json`, `meta/tasks.parquet`, `data/chunk-000/*.parquet`,
  `videos/<camera>/chunk-000/*.mp4` (AV1, `libsvtav1`).
* Per-step features: `observation.images.{front,left_wrist,right_wrist}` video 256×256×3,
  `observation.state` f32[16], `observation.velocity` f32[16], `observation.eef_pose` f32[14],
  `observation.force` f32[18], `action` f32[16], plus `timestamp`/`frame_index`/`episode_index`/`index`.
* Two files that are ours rather than lerobot's: `a3_episode_metadata.jsonl` (one JSON object
  per episode: `episode_index`, `seed`, `frames`, `success`, `source_shard`,
  `controller_type`, `source_action_mode`, `stored_action_mode`, `task`) and
  `collection_summary.json` (task, contract, prompt, randomisation ranges, acceptance counts).
* Collection facts from that summary: 1030 attempts → **1000 accepted** (98.9 %), all
  `success=True`; `placements_randomized=True`, `cookies_jittered=False` (the dense packing
  cannot take per-Cookie jitter); the boxes move in x/y/yaw and the arm home in ±0.04 rad,
  keeping 20 mm between the boxes.

## 3. What the files do not say, and a consumer will need

1. **There is no `mode` column.**  `meta/episodes/*.parquet` carries no seed either -- the
   sidecar is the only place the mode is recoverable, via `modes[seed % 4]`.  A release must
   ship the sidecar and document that mapping, or nobody downstream can slice by condition.
2. The **scene config** (`configs/cookie_batch_second_batch.yaml`) and the code that defines
   the task are in this repository, not in the dataset: the prompt string, the 1000 Hz/20 Hz
   rates, the 2×5 target box, the finger-pad shim (0.5 mm half-thickness, without which the
   dense packing is unsolvable rather than merely hard).
3. The images are the **unmirrored A3 convention**.  A checkpoint fine-tuned on LIBERO expects
   the horizontally mirrored view, and `A3_IMAGE_MIRROR=1` is what toggles it.
4. `action` and `observation.state` are **A3 joint positions in radians** (16 wide: 7 + gripper
   per arm) -- not Panda, not LIBERO, not Cartesian deltas.
5. The videos are **AV1**: decoding needs pyav with an AV1 decoder.

## 4. Releasing it

**One dataset repo, the merged form, plus the four things that carry the semantics.**

```
<org>/a3-dual-arm-cookie-second-batch
  README.md                     # the card, below
  data/ meta/ videos/           # the lerobot dataset as-is, so LeRobotDataset(repo_id) works
  episode_metadata.jsonl        # renamed a3_episode_metadata.jsonl; the mode authority
  collection_summary.json       # contract, prompt, randomisation, acceptance
  scene_config.yaml             # copy of configs/cookie_batch_second_batch.yaml
  examples/                     # the two manifests + the render command actually used
  MANIFEST.sha256               # hashes, so a consumer can verify what they downloaded
```

* **Do not** upload the 20 raw shard directories.  They are strict supersets of the merged
  data (same frames, more files) and would double the download for no information.
* **Do not** upload checkpoints here.  Each un-freeze checkpoint is 21 GB (weights + optimizer
  state) and belongs in its own model repo; the dataset card should link to it, not contain it.
* Drop `meta/stats.json.bak` (a repair artefact of the first training run) and describe the
  repair in the card instead: the degenerate `q01 == q99` dimensions were recomputed from the
  recorded rows, which matters to anyone who re-derives normalisation statistics.
* `README.md` front-matter needs a **license**, and this repository has no `LICENSE` file --
  that is a decision to make before publishing, not after.  The card should also carry
  `task_categories: robotics`, `tags: [lerobot, mujoco, manipulation, dual-arm]` and a
  `size_categories` consistent with 1000 episodes.
* The card's body should state, in this order: the task and the exact prompt string; the four
  conditions and how to derive them; what a step contains (the feature table); the physical
  facts above (rates, joint space, camera convention, AV1); the contract that decided which
  episodes were kept; and one known caveat -- the scene's *evaluation* grading was relaxed
  after collection (a leaning Cookie with one end on the floor now counts), while these
  episodes were accepted under the stricter rule.

**Where it goes.**  Not the Hugging Face Hub -- this machine cannot reach it:

```
https://huggingface.co/api/whoami-v2   -> connection timed out (port 443)
https://hf-mirror.com/api/whoami-v2    -> HTTP 401 (reachable, but the mirror is read-only)
HF_TOKEN / stored credentials          -> none
```

`www.modelscope.cn` does answer from here, and hosts LeRobot datasets, so **ModelScope is the
target**.  Its client is installed in a virtualenv of its own
(`/root/autodl-tmp/ms-venv`) rather than in the training environment, so that pulling in its
dependencies cannot disturb the one that trains and evaluates the policies.

## 5. The bundle, as built

`/root/autodl-tmp/release/a3-dual-arm-cookie-second-batch` — 40 files, 3.7 GB (39 of them
hashed in `MANIFEST.sha256`), with the bulk hard-linked to the collected dataset so the staging
cost no extra disk:

```
README.md                     the card: task, prompt, four conditions, feature table,
                              physical facts, caveats, licence, provenance
LICENSE                       CC BY 4.0, full legal text
episode_metadata.jsonl        renamed sidecar; the only place `seed` lives
collection_summary.json       contract, prompt, randomisation, acceptance counts
scene_config.yaml             copy of configs/cookie_batch_second_batch.yaml
examples/load_and_filter_by_mode.py    loads the set, prints per-condition counts and the
                                       block structure, reads one episode of each
examples/train_pi05_example.sh         the recipe, with the four traps commented
upload_modelscope.sh          create-if-absent + upload + the verification steps
MANIFEST.sha256               hashes of the other 39 files
data/ meta/ videos/           the LeRobot dataset, unmodified
```

Both examples were run against the bundle before it was sealed: the loader reads the sidecar,
constructs `LeRobotDataset` (432,327 frames, 1000 episodes) and decodes one frame of each
condition.  Two things it turned up while being written, both now in the card:

* **The episodes are ordered in 20 blocks of 50-by-condition.**  Each of the 20 collection
  shards contributed its episodes grouped by mode, so along `episode_index` the sequence is
  50 `full`, 50 `prefill`, 50 `perturb`, 50 `prefill_perturb`, five times over.  A sequential
  train/validation split therefore separates *conditions*, and correlates with shards.  Anyone
  who splits this set without shuffling will measure something that is not generalisation.
* **`video_backend="pyav"` is not optional in practice.**  lerobot's default is torchcodec,
  which needs a matching FFmpeg build; where it is missing the failure is
  `OSError: libavutil.so.57: cannot open shared object file`, which reads as a broken
  environment rather than as a codec.  The parquet data, the sidecar and the metadata need no
  decoder at all.

**Remaining steps** (they need a ModelScope account, so they are the user's to take):

```bash
export MODELSCOPE_API_TOKEN=...          # modelscope.cn -> 个人中心 -> 访问令牌
cd /root/autodl-tmp/release/a3-dual-arm-cookie-second-batch
bash upload_modelscope.sh <owner>/a3-dual-arm-cookie-second-batch
```

**The token has to carry write permission, and that is easy to get wrong.**  A fine-grained
token created at the read-only level is accepted by the client and then refused by the write
endpoint:

```
POST /api/v1/datasets -> 403  10020000608  "只读Token无写权限"
```

The same token is also refused by the account-level endpoints (`whoami`, `users/me`) with
`OperationNotAllowed`, because a fine-grained token is scoped to particular repositories.  That
second refusal is *not* a diagnosis of the upload: the script therefore treats a failed
`whoami` as a warning and lets the upload report what actually matters.  It also means such a
token usually cannot create a repository — create the dataset on modelscope.cn first and pass
`--skip-create`.

The script creates the dataset repository if it does not exist (`--exist-ok`, so a retry is
safe), uploads with 8 workers, and resumes an interrupted transfer rather than restarting the
3.7 GB.  Afterwards, **check that the card is at the repository root** — an uploader that nests
the payload under a directory named after the local one leaves the card at
`<repo>/a3-dual-arm-cookie-second-batch/README.md`, where nobody reads it; the script prints
the download command that answers this.  Then hand consumers `MANIFEST.sha256` so they can
verify what arrived.
