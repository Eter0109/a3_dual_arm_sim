# π0.5 on A3: every fine-tune, and what each one actually changed

Reconstructed from the training logs' own launch blocks (`logs/train_*.log`,
`pi05-runs/*.log`), the commit history, and the session notes.  Every row below is a real
run whose log still exists, except where marked.

The point of writing it down: the differences between consecutive runs are the experiment.
Most of them changed exactly one thing -- a dataset, an un-freeze scope, a shim -- and the
interesting failures came from the runs that changed *nothing* and were read as if they had.

## 1. The runs

| # | date | output | dataset | base | trainable | steps / batch | loss first → last |
|---|------|--------|---------|------|-----------|---------------|-------------------|
| 0 | 09-30 | `a3_adv_smoke` | `a3_cookie_adv_1000_merged` | `lerobot/pi05_base` | frozen | 300 / 8 | smoke test |
| 1 | 09-30 | `a3_adv_40k` | `a3_cookie_adv_1000_merged` | `lerobot/pi05_base` | frozen | 40000 / 8 | 0.340 → 0.006 |
| 2 | 10-01 | `a3_adv_40k` (unfreeze 3k) | same | same | vision + 18 LLM layers | 3000 / 4 | — |
| 3 | 10-03 | `a3_single_base` / `a3_single_libero` | `a3_cookie_single_arm` | `pi05_base` vs `pi05_libero_finetuned` | frozen | 2000 / 4 | starting-point control |
| 4 | 10-03 | `one_base` | `a3_cookie_one_arm` | `pi05_base` | frozen | 6000 / 4 | 0.208 → 0.009 |
| 5 | 10-03 | `one_rel` | `a3_cookie_one_pi05` | `pi05_base` | frozen, `drop_n_last_frames=49` | 6000 / 4 | 1.901 → 0.012 |
| 6 | 10-04 | `one_rel2` | `a3_cookie_one_delta` | `pi05_base` | as above | 6000 / 4 | 1.901 → 0.012 |
| 7 | 10-04 | `one_proj` | `a3_cookie_one_pi05` | `pi05_base` | vision + projector (2.4 M) | 4000 / 4 | 0.196 → 0.009 |
| 8 | 10-05 | `one_overfit` | `a3_cookie_one_pi05` | `pi05_base` | vision + projector | 5000 / 4 | 0.202 → 0.004 |
| 9 | 10-05 | `one_single` | `a3_cookie_one_pi05` | `pi05_base` | vision + projector + 18 LLM layers | 1000 / 4 | 0.126 → 0.006 |
| 10 | 10-05 | `one_stateproj` | `a3_cookie_one_pi05` | `pi05_base` | + state projector `Linear(32→2048)` | 4000 / 4 | 0.192 → 0.008 |
| 11 | 10-10 | `a3_two_cookie_unfreeze_15k` | `a3_two_cookie_500_merged` | `pi05_base` | vision + 18 layers, state projector **off** | 15000 / 4 | 0.506 → 0.012 |
| 12 | 10-10 | `a3_two_cookie_adv_unfreeze_15k` | `a3_two_cookie_500_adv_merged` | `pi05_base` | as above | 15000 / 4 | 0.642 → 0.014 |
| 13 | 10-13 | `a3_b2_14k` | `a3_b2_merged` | `pi05_base` | frozen | 14000 / 6 | → 0.0085 |
| 14 | 10-13 | `a3_b2_unfreeze_14k` | `a3_b2_merged` | `pi05_base` | vision + 18 layers + state projector | 14000 / 6 | 0.159 → 0.0028 |
| 15 | 10-14 | `a3_b2_unfreeze_20k` | `a3_b2_merged` | **`a3_b2_unfreeze_14k/014000`** | as above | 20000 / 6 | 0.002 → 0.002 |

Run 15 is a **warm restart**, not a continuation of the same schedule: it started from a
checkpoint whose learning rate had already annealed to the floor, with a fresh 20000-step
cosine (first logged `lr` 4.3e-07 at the warmup start, 1.7e-05 at step 8000, 2.5e-06 at
the end).  It was interrupted at step 10000 by a full disk and resumed from `008000` with
lerobot's own `--resume` (step, optimizer, scheduler and RNG restored), so steps 8000-20000
are one continuous run.

## 2. The datasets

| dataset | size | what it is |
|---|---|---|
| `a3_cookie_adv_1000_merged` | 1123 eps / 537,765 frames, 16 shards | the randomisation branch's first big A3 collection (recorded success rate 92.4 % per the collection notes); 4.87× the 200-episode `gr00t-runs/b1` set that preceded it |
| `a3_cookie_one_*` (`one_arm`, `one_pi05`, `one_delta`) | 7.6 MB each | one Cookie, one arm; the small set the un-freeze scope was explored on |
| `a3_cookie_single_arm` | 37 MB | the one-armed A3 interface's own dataset |
| `a3_two_cookie_500_merged` / `_adv_merged` | ~700 MB | two Cookies, 500 episodes; the `adv` twin is the same scene with the higher randomisation profile |
| `a3_b2_merged` | 1000 eps | **second-batch** episodes, 250 each of `full` / `prefill` / `perturb` / `prefill_perturb` |

The `a3_b2` set exists because of one failure shape: every earlier episode was "start empty,
do batch 0, do batch 1", so a policy met the second batch only from the state the *expert*
leaves behind -- five perfectly placed Cookies -- which is a state a policy never produces
itself.  The four modes are the four ways to meet it imperfectly.

## 3. What each phase changed, and why

**A. Getting π0.5 to run at all (09-30, `58dceda`, `57845f9`, `cbe7610`).** The external
VLA interface, the dataset repair pass, and π0.5 support.  Three probes asked whether the
checkpoint was usable before any training: does the backbone still produce text
(`1310592`), can the subtask head be driven (`240f208`, which also found the state bug),
and does the prompt need openpi's format rather than lerobot's (`7724d8c`).  `25e461c`
measured the state encoding defect and concluded **not** to repair it: the repaired
encoding clipped the chunk spread to 0.03×, `minmax` was worse still (0.69×), and removing
state was worse again (2.41×).  The model was reading state; the defect was not the problem.

**B. The first long run (10-01, run 1).** Three things had to be built for it, and each was
a trap that a shorter run would have hidden:

* **The LR schedule does not scale for long runs.**  `scheduler_decay_steps=30000` only
  auto-scales when `--steps < 30000`, so 40000 steps would have spent its last 10000 at the
  floor.  The run passed `decay=40000, warmup=1300` explicitly.
* **`save_total_limit` does not exist in this lerobot.**  20 checkpoints × 11 GB = 220 GB
  against 56 GB of free space: the run needed a pruning daemon.  This is the origin of the
  disk pressure that later killed run 15 mid-save.
* **Padded chunk tails were being trained on.**  11.3 % of the action rows inside a
  50-step chunk are the "repeat the last action" padding; `drop_n_last_frames` (a launcher
  attribute patch, since neither `PI05Config` nor `PreTrainedConfig` declares the field)
  takes them out of the loss.

**C. Isolating the body from the task (10-03 … 10-05, runs 3-10).**  A one-armed,
one-Cookie interface and its datasets, so that "can π0.5 learn *this* task" could be asked
without the two-armed scene's confounds.  `e11ce0f` added the un-freeze scope and fused
AdamW; the scope was then walked up deliberately -- frozen → projector only → +18 LLM layers
→ + state projector -- and runs 5-6 were the **relative-actions** line, retired as a
negative result (`84d7982`).

**D. Three interface bugs that decided what the data even meant (10-05 … 10-07).**
`728fa1c` gave π0.5 its own loop and borrowed objects; `05b24fe` made the chunk get executed
the way the checkpoint says; `56b689d` found the policy had been seeing a **left-right
mirrored** world; `3838c73` found the state's orientation was 120° outside the training
distribution; `f5cc3cf` stopped feeding it an upside-down world, which took a 0/40 to 31/40.
Those four are the reason later numbers are comparable at all.

**E. Two Cookies, twice (10-10, runs 11-12).**  Same recipe, one variable: the randomisation
profile of the data.  `ef74fb4` fixed the measurement that had been reporting torques as
radians and gave this convergence run its manifests.

**F. Making A3 driveable (10-12, `b47c9a3`, `0f8906d`).**  The dense batch expert learned to
survive the packed column on either arm, and the slid-mount scenes got their state
calibration.  The lesson there is worth repeating: `state_offset` is a calibration constant,
and moving the robot without moving it told the model it was 20 cm from every training
sample -- which is why the dense scene went from ploughing 51/80 Cookies over to touching
1/80.

**G. The second-batch data (10-13, `ebd7fee`), then runs 13-15.**  The four-mode collection
above, then the frozen/un-freeze pair (13, 14) and the warm restarted extension (15).

**H. Judging and speed (10-14, `03ce33f`, `50a995e`).**  The second-batch scene now grades
containment rather than posture -- a Cookie leaning across the pile with one end on the
floor is in the box -- and the sweep machinery got one containment pass per step, one loaded
model per sweep, and a resumable driver.

## 4. The launcher's shims, and the failure each one answers

| shim | failure it exists for |
|---|---|
| processor-name alias | the checkpoint's `policy_preprocessor.json` names `relative_actions_processor`; this lerobot registers the class under another name and the lookup raises something that reads like a missing dependency |
| quantile pre-flight and repair | a degenerate `q01 == q99` divides by `eps`, and the loss is 1e12 at step 1 -- which reads as divergence, not as bad statistics |
| `rename_map` | A3's camera names onto openpi's slots, without rewriting 26 GB of video |
| partial un-freeze | full un-freeze needs 33.1 GB on a 32 GB card; vision + 18 layers is 26.8 GB |
| train/eval mode on un-freeze | `PaliGemmaWithExpertModel.train()` ends with `paligemma.eval()`; restoring only `requires_grad` trains the un-frozen blocks in eval mode, silently |
| state projector | `Linear(32 → 2048) × √2048`, one extra prefix token |
| `drop_n_last_frames` | padded chunk tails, above |
| fused AdamW | memory and step time |
| native `--resume` | `A3_PI05_BASE` swaps the starting weights, which is not resuming; `--resume=true --config_path=<ckpt>/train_config.json` restores optimizer, schedule and RNG (and must not also be given `--policy.path`) |

## 5. Numbers that moved

* **0/40 → 31/40** after the mirror fix (`f5cc3cf`).
* **51/80 → 1/80** Cookies knocked over, after the slid-mount state calibration.
* dense-scene expert placement error **4.21 mm → passing**, after `layout_offset_m`.
* frozen vs un-freeze on `a3_b2`, 14000 steps: teacher-forced loss **0.0085 vs 0.0028**.
* run 15 (warm restart, 20000 steps): **0.002**, against 0.0028 where the 14k schedule
  stopped.
* `a3_b2` at 20k, ten seeds, relaxed judging: **mean 6.6 / median 5.5**, all ten seeds
  placing something.  The same ten seeds on the 14k checkpoint re-judged: **mean 2.6**, six
  of them zero.
* In `full` mode specifically -- the model doing both batches from an empty box -- the 20k
  checkpoint reaches ten on 5 of 8 samples (mean 8.25, no zeros); the 14k checkpoint reaches
  ten on 3 of 11 (mean 5.09) and lands on **zero** in 3 of them.  So the gap is smaller than
  the all-mode numbers suggest and it is a gap in reliability rather than in capability: the
  14k checkpoint can do the whole task, it just does not do it dependably.  On the three
  seeds both checkpoints have been run on (0, 4, 8) the 14k scored 5/0/0 and the 20k 5/10/10.

## 6. Open questions

1. **Extra steps or warm restart?**  Run 15 ran **20000** more steps than run 14 -- not the
   6000 the names `14k`/`20k` invite -- starting from a checkpoint whose learning rate had
   already annealed to the floor, with a fresh 20000-step cosine: re-heated from 2.5e-06 to
   2.5e-05 over the first 1000 steps, then annealed back.  The windowed loss says what that
   cost and what it bought.  The re-heat pushed the loss from 0.0029 (where run 14 stopped)
   up to 0.0054, and it took until roughly step 12000 to get back below where run 14 had left
   off; from there the second anneal took it 0.0027 → 0.0022 in 8000 steps.  So about 60 % of
   the restart's budget went into recovering its own damage, and the gain is better described
   as *a longer anneal* than as *an excursion that found a better basin* -- the more so
   because run 14's schedule auto-scaled its decay to its own 14000-step budget, so a single
   34000-step cosine was never tried and remains a live explanation for the same result.
2. **The 14k judging baseline does not match the older 24-seed one** -- 2.6 against 4.6 on
   the same weights and scene, with the zero-count fraction going 5/24 → 6/10.  That is too
   large for seed noise; the older sweep's scene and step budget need checking before the
   comparison is used.
3. **The second batch is all-or-nothing.**  The score distribution is 5 or 10, not a spread:
   5 is "the first batch landed and the second did not".  Whatever the 20k checkpoint learned
   raised the first batch to a certainty and left the second a coin toss.  The failure that
   separates the two checkpoints is not "never places anything" so much as *losing* what is
   already there: the 14k checkpoint lands on zero on 6 of 10 all-mode seeds, including
   three `prefill` seeds that begin with five Cookies already in the box, and on 3 of 11
   `full` seeds.  The 20k checkpoint has no zero in any of its 18 scored seeds.
4. **Single seeds are not measurements.**  Seed 12 on the 20k checkpoint scored 11 and then
   5 on two runs of the identical configuration.  The policy is stochastic; ten seeds give
   the mean to about ±1.5 Cookies at this spread.
