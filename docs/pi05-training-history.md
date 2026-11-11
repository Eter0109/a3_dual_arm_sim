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
| 16 | 10-15 | `a3_b2_floor20k` | `a3_b2_merged` | **`a3_b2_unfreeze_14k/014000`** | as above | 20000 / 6 | 0.0028 → 0.003 — run 15's control: same steps, rate held at 2.5e-06 (§8) |

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
* In `full` mode specifically -- the model doing both batches from an empty box -- the two
  checkpoints are **the same on the mean** once every sample is traced to its own log
  (14k 6.38 and 20k 6.38 on the eight seeds they share; see §8's correction, the 8.25 that
  used to stand here was assembled from mixed tags).  What differs is the shape: the 14k
  checkpoint lands on **zero** in 3 of those 8 and reaches ten or eleven in 3, while the 20k
  checkpoint has no zero and two tens.  So the difference is a reliability difference, not a
  capability one -- and the six zeros in ten of the all-mode ruler say the same thing louder.  So the gap is smaller than
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
   **Answered by run 16 (§8), after a correction.**  The extra 20000 steps on their own
   move the mean by nothing measurable (6.38 → 5.38 on the eight seeds run 16 shares with
   run 14), and the second cycle does not beat the baseline it started from either (6.38 on
   the same eight) -- the "+3.2 for the cycle against +0.3 for the steps" that stood here
   first came from a run 15 mean assembled by hand across two sweep tags (§8, correction).
   What both continuations do, and run 14 does not, is remove the *collapses*: it lands on
   zero in 3 of 8 full-mode samples and 6 of 10 all-mode ones.  What a single long cosine
   would have done is still untested (§8, "What it does not settle").
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

## 7. Why the second cycle helped, as far as the logs can say

No experiment was run for this section.  It is the windowed loss of runs 14 and 15, read
against what an anneal and a restart are each supposed to do -- and it is written down because
the tempting summary ("the restart worked") does not follow from it.

**What the two mechanisms predict**

* An *anneal* is a noise-radius argument.  Stochastic optimisation does not settle on a
  minimum; it wanders inside a ball whose radius scales with the learning rate, so the loss at
  the end of training is roughly the objective averaged over that ball.  Only a decayed rate
  makes the ball small enough to land.  The terminal loss is then set by how small the final
  rate was and how far the trajectory had travelled before it -- by the *shape* of the
  schedule, not by how many cycles it is cut into.
* A *restart* has two candidate stories.  (i) **Escape**: raising the rate moves the weights
  far enough in a few hundred steps to leave the basin the previous cycle ended in, and the
  re-anneal lands in a different, possibly flatter one.  (ii) **Just a longer schedule**:
  re-raising and re-decaying is equivalent to spending longer above the floor and then
  annealing once more -- which a single long cosine also provides.  **A loss curve cannot tell
  these two apart**, and that is the whole difficulty.

**The windowed loss** (2000-step means, from the two logs)

| run 14 (cosine auto-scaled to its 14000-step budget) | run 15 (from the 14k weights, fresh 20000-step cosine) |
|---|---|
| 8-9k  0.00422 | 2-4k  0.00538 -- **worse than where run 14 stopped** |
| 11-12k 0.00322 | 6-8k  0.00390 |
| 12-13k 0.00296 | 10-12k 0.00338 |
| 13-14k 0.00292 (last window: −4e-05, after −2.6e-04 in the one before) | 12-14k 0.00272 -- back below run 14's endpoint |
| | 18-20k 0.00217 |

**Three readings.**  (1) The restart's immediate effect is negative: 0.0029 → 0.0054, +86 %.
Stopping there would have been a pure loss.  (2) About 12000 of the 20000 steps went into
repaying it -- the loss only got back below run 14's endpoint near step 12000.  (3) The whole
net gain (0.00292 → 0.00217) accrued during the second *decay*, from ~step 12000 at a rate
near 1e-05 down to the floor.

**Which one, then.**  "Just more steps at the floor" does not explain it: run 14's last window
was already at the floor (its rate was ~2.8e-06 against a 2.5e-06 floor), and the creep there
was 4e-05 per 1000 steps, falling sixfold within one window.  Extrapolating that creep at its
own constant rate for 20000 steps lands on ~0.0021 -- the observed value -- but the measured
deceleration says the within-basin creep was nearly exhausted, which puts a floor-continuation
at ~0.0029.  **The logs cannot settle which extrapolation is right.**  What they do support is
that the gain came from *going back up to a high rate and annealing down again* -- the second
cycle as a whole -- rather than from the extra steps as such, because steps spent at the floor
barely move the loss.

The restart is nevertheless not *necessary* for that process: a single 34000-step cosine
supplies the same two ingredients (a long high-rate phase, then a full anneal) while spending
nothing to repair an excursion.  That comparison is the experiment that would settle the
question and it has not been run.  Mechanistically the re-heat is expensive here because the
dataset is small next to the model -- 1000 episodes, ~250k frames, against 3.09 B trainable
parameters -- so a high rate does not only explore, it forgets, which is what a 0.0025 rise in
loss over the first 2000 steps is.

One caveat that limits all of the above: the task metric is not the loss.  The loss moved 26 %;
the task moved from 6/10 seeds landing nothing to 0/10.  A loss-level attribution does not
transfer to the task level, so the two stories would have to be compared on the task metric.

**Since run 16, they have been, and the loss reading is confirmed while its task-level
benefit is not.**  §8 has the task-side trajectory of the same cycle: by 8000 steps in, the
re-heat has cost the policy *both* batches; at 16000 steps it is no better than the
checkpoint it started from; and the last quarter -- the rate falling from about 1e-05 to the
floor -- repairs it back to that same level and stops there.  So the "escape" story is not
merely unproven: on the task metric the high-rate phase only destroys, and the low-rate phase
only undoes it.  "Just train longer" is contradicted too, by run 16 sitting a Cookie below
the baseline after 20000 flat steps.  What the second cycle does buy is the disappearance of
the zero -- in 40 samples across runs 15 and 16, not one lands on nothing, against 3 of 8
full-mode samples for run 14 -- and that is a change in reliability, not in capability.

## 8. Run 16, the control that was missing (started 10-15, ~00:10)

Every sentence in the two sections above compares run 15 against run 14 and calls the
difference "the second cycle".  That is not something run 14 can support: run 15 also ran
**20000 more steps**, and the two changed together.  Run 16 changes one of them.

| # | output | base | steps | effective learning rate | time |
|---|--------|------|-------|-------------------------|------|
| 16 | `a3_b2_floor20k` | `a3_b2_unfreeze_14k/014000` | 20000 / batch 6 | **constant 2.5e-06** | ≈ 6.4 h at 1.15 s/step |

The rate is the one run 14 annealed to, held for the whole run: `--policy.optimizer_lr=2.5e-06`
`--policy.scheduler_decay_lr=2.5e-06`, which makes the cosine flat (the scheduler is a
`LambdaLR`, so the effective rate is `optimizer_lr x lambda` and `lambda` is 1 when
`peak_lr == decay_lr`).  Two traps on the way there, both of which cost a launch:

* `--scheduler.*` and `--optimizer.*` on the command line are **overwritten**:
  `TrainPipelineConfig.validate` rebuilds both from the policy preset whenever
  `use_policy_training_preset` is left at its default (`configs/train.py:137`).  The
  surviving knobs are `--policy.optimizer_lr`, `--policy.scheduler_decay_lr`,
  `--policy.scheduler_{warmup,decay}_steps`.
* `Output directory ... already exists and resume is False`: `mkdir -p` on the output
  directory itself is enough to fail the run, so the launcher creates only its parent.

Everything else is byte-identical to run 15: `a3_b2_merged`, batch 6, seed 1000, the
partial un-freeze (vision + 18 LLM layers + state projector), `drop_n_last_frames`, pyav,
fused AdamW, and the same 20000-step auto-scaled cosine geometry (so the warmup is 667
steps in both).  The training script is `logs/floor20k_train.sh`, with a pruner that keeps
the two newest checkpoints because the disk is at 79 %.

Paths in this section: `logs/` is `/root/autodl-tmp/logs` (the runners, the per-seed logs
and the `<tag>_summary.json` files) and `pi05-runs/` is `/root/autodl-tmp/pi05-runs` (the
checkpoints).  Both live outside the repository -- the weights and the logs are too large
for git -- so the scripts that produced tonight's numbers are
`logs/floor20k_train.sh`, `logs/floor20k_evals.sh`, `logs/floor20k_topup.sh` and
`logs/matched_ruler.sh`.

**Reading it.**  The three arms differ in exactly one factor each, and the first two were
written down before the run finished.  The numbers below are the ones that survive tracing
every sample back to the log that produced it -- see the correction note after the table,
which is the reason this section reads differently from its first version.

| arm | first 14000 steps | next 20000 steps | full mode, 11 seeds | same 8 seeds | shape |
|-----|-------------------|------------------|---------------------|--------------|-------|
| 14 (`a3_b2_unfreeze_14k`) | cosine to the floor | — | 5.09 | 6.38 | 3 zeros, 3 tens or better |
| 15 (`a3_b2_unfreeze_20k`) | same | re-heat 2.5e-05 → floor | 5.82 | 6.38 | no zeros, 2 tens |
| 16 (`a3_b2_floor20k`) | same | hold at the floor | **5.36** | 5.38 | no zeros, no tens |

The "same 8 seeds" column is 12/16/20/24/28/32/36/40, the seeds all three arms have been
scored on.

**Correction: the 20k arm's 8.25 was an artifact.**  An earlier version of this table gave
run 15 "8.25 on 8 samples, 5 of them at ten", and the whole "the cycle is worth +3.2, the
steps +0.3" reading was built on it.  That figure was assembled by hand from two different
sweep tags, and seed 12 was read from `b2unf20k_seed12.log` -- a second-batch *all-mode*
sample worth 11 -- instead of `b2unf20kfull_seed12.log`, which scored 5.  Retraced from the
logs by tag, and topped up today with the seeds whose logs had been deleted, run 15's
full-mode mean is 5.82 over eleven samples, and 6.38 over the eight the other arms share --
**exactly the 14k baseline**.  The lesson is worth more than the number: this document's
arm means had been computed from per-seed logs selected by eye across tags, and a single
mislabelled sample moved the headline by 2.4 Cookies.

The three prediction bullets, marked:

* 16 ≈ 14 → the extra steps buy nothing on their own; the gain is the cycle.  (Only for the
  *second batch*, and only weakly -- see the shape below.)
* 16 ≈ 15 → the steps were the active ingredient and the re-heat was incidental.  (Not
  that either.)
* 16 < 14 → a long small-step continuation on 1000 episodes drifts.  (By 1.0 Cookie on the
  shared eight, which is inside this ruler's noise.)

What the corrected table says is narrower than the first version claimed, and it is not the
one the run was designed to return: **on the full-mode mean the cycle does not beat the
checkpoint it started from at all** (6.38 against 6.38), and on the same eight seeds the flat
continuation is a Cookie behind both.  What both continuations do, and neither the baseline
nor the loss curve showed, is remove the *collapses*: run 14 lands on zero in 3 of 8 full-mode
samples and in 6 of 10 of the all-mode ones, while runs 15 and 16 have not produced a single
zero in 19 and 21 samples respectively.  That is the effect this night actually measured.

The loss, separately, does say which extrapolation was right: run 16 ended at **0.003**
against run 14's 0.0028 and run 15's 0.002, so §7's linear extrapolation of the within-basin
creep (0.0028 → 0.0021 over 20000 steps) was the wrong one -- the deceleration was real and
the creep was exhausted.  That much of the original reading survives.

**Which of the two continuations is better is not settled by this ruler.**  The all-mode
numbers in this document (14k 2.6 with six zeros, 20k 6.6 with none, 16 at 5.10) point the
same way as the shared-eight full-mode column, and they have the same provenance problem, so
`logs/matched_ruler.sh` re-measures all three arms on both rulers in one batch with today's
driver -- which is also the only way the "same 8 seeds" column above can be called matched
rather than merely same-labelled, since a seed is not reproducible across sessions (0/4/8
scored 5/10/10 once and 5/0/8 the next time).

**The shape is the part the mean hides**, and it is the one thing that survives the
correction above.  14's distribution is three modes apart -- 0, 5, 10/11 -- while both
continuations are single-humped: 16 is a spike at 5 (eight of eleven samples at five, none
below, none above seven) and 15 is 5, 5, 5, 5, 5, 6, 10, 10 on the shared eight.  So the
same treatment that leaves the mean where it was removes the *zero*: run 14 lands on nothing
in 3 of 8 full-mode samples and 6 of 10 all-mode ones, and runs 15 and 16 together have not
produced one in 40 samples.  Two honest caveats: eleven samples put "3/11 → 0/11" at about
p ≈ 0.03 one-sided, suggestive rather than settled; and because the distribution is lumpy at
0, 5 and 10, a mean over it is a weak summary of behaviour -- a policy that always does the
first batch and never the second scores exactly what a policy that gambles on both does.

**What it does not settle.**  Neither continuation beats the 14k checkpoint on this ruler's
mean, so nothing here argues that a *cycle* is the efficient way to spend a budget: the
second cycle spent 16000 of its 20000 steps below the level it started from and finished
level with it.  A single long cosine -- same two ingredients, no repair bill -- remains the
untried alternative (§7), and it is not the comparison this arm makes: run 16 changes the
rate, not the schedule's shape.

On the loss the question is answered: extrapolating run 14's final within-basin creep at its
own rate for 20000 steps lands on 0.0021 -- run 15's endpoint -- while the measured
deceleration says the creep was nearly exhausted and a floor continuation should sit at
~0.0029.  Run 16 finished at 0.003, so the deceleration was right.  But the caution that
opened §7 bites in the other direction too, and harder than expected: run 15's loss is 30 %
below run 14's and its full-mode mean is identical to run 14's, so a teacher-forced loss
that keeps falling can be telling you about the data it is fitting and nothing about the
task.  What the task metric did move was the *failure rate*, not the score.

**Riding along, for free.**  Run 15's own checkpoints (`008000`, `012000`, `016000`) were
still on disk, so the same 8 full-mode seeds (12/16/20/24/28/32/36/40) were run against them.
This turns "the re-heat did +86 % damage and took 12000 steps to repay it" from a loss
statement into a task statement:

| run 15's step | steps into the second cycle | full-mode mean, same 8 seeds | shape |
|---------------|-----------------------------|------------------------------|-------|
| 0 (= run 14) | 0 | 6.38 | 3 zeros, 3 tens or better, 2 fives |
| 8000 | 8000 | **1.25** | 6 of 8 land nothing |
| 12000 | 12000 | 4.25 | 2 zeros, 1 ten |
| 16000 | 16000 | 5.62 | no zeros, 1 ten |
| 20000 | 20000 | 6.38 | no zeros, 2 tens |

The shape is a V, not a staircase.  At 8000 steps into the cycle the policy has lost *both*
batches -- six of eight seeds place nothing, against three tens the baseline was still
scoring -- and 16000 steps in it is still not back to where it started (5.62 against 6.38;
at n = 8 on a three-mode distribution that is indistinguishable, which is the point).  The
last quarter, in which the rate falls from about 1e-05 to the floor, undoes the damage and
stops there.  That is what the trajectory says, and it is less flattering than the first
version of this section: the high-rate phase finds nothing, the low-rate phase repairs what
the high-rate phase broke, and the endpoint ties the starting point on the mean.  "Escape"
is dead on the task metric as well as in the loss; "the second anneal buys the second batch"
is not supported either -- run 14 already reached ten or eleven in 3 of these 8 samples,
run 15 in 2.  What the endpoint does have that the baseline does not is the absence of
collapse.

**A missing flag cost the first six sweeps of that runner.**  `--serve-checkpoint` names the
checkpoint for the summary and requires `--serve-adapter`; it does **not** configure the
bridge.  The bridge command is built from `--serve-config` pairs alone
(`sweep_policy_seeds.py:314`), so a run that passes only `--serve-checkpoint` starts a
server with no `ckpt=`, the adapter falls back to its own `DEFAULT_REPO`/`DEFAULT_STATS`
(both pointing at `gr00t-runs/b1`, deleted earlier to free disk), and every sweep spends the
driver's 30-minute startup timeout before reporting "the bridge did not come up".  The fixed
runner therefore ends with a keystone step that loads the bridge once and waits for its
socket before any sweep starts, which turns five minutes of checking into insurance against
three hours of silence.

## 9. The aligned scene: 1000 episodes on main's arrangement, and 4000 steps from the 20k incumbent (10-18)

Section 8 ends with a policy that scores well on this branch's layout and **0/20 on main's
benchmark**.  The two benchmarks were not merely different code: they were different scene
regimes, and the 20k checkpoint had never seen main's arrangement.  That is the whole of the
"2-Cookie discrepancy" that took a day to chase -- not a physics difference, not a judge
difference (both were verified identical), but a train/test scene mismatch, made invisible
by an unseeded sampler that turned every comparison into a coin flip.

### 9.1 The dataset (`a3_b2_main_aligned_merged`)

Main's arrangement as shipped, pinned: bins, all 80 Cookies and the arm's home joints are
rewritten onto main's own dump after every reset (`main_alignment.apply_scene`), and the
*only* variation is the target box's yaw, sigma 0.005 rad.  `--position-noise 0`,
cookies not jittered.  The four second-batch modes (`full`, `prefill`, `perturb`,
`prefill_perturb`) still supply the variety, 250 episodes each.

* 1000 episodes, 425,354 frames (against `a3_b2_merged`'s 1000 / 432,327 -- same scale)
* 1000/1000 accepted, seeds 2000-2999, 3.9 GB, 20 shards, 20-way parallel: 40 probe
  episodes took 7 min 22 s, the full run 3 h 41 m
* audit: 1000 unique episode indices, 0 NaN/Inf in state and action, 16-wide, [-1.507, 1.709]

Two gates, because "aligned in process" and "aligned in the data" are different claims:

| gate | result |
|---|---|
| `scripts/verify_main_alignment.py` (in-process) | PASS -- bins 0.000 mm, 80 Cookies 0.000 mm, 18 joints 0.000 rad, cameras 0.000; render MAE 0.157 / 0.614 / 0.000 |
| first training frame vs main's render (the data) | MAE **2.25 / 3.34 / 1.48** of 255 |

The frame gate reads ten to twenty times larger than the in-process one, and that is not a
defect: the in-process gate compares two renders of *one* state, while the frame gate
compares episode 0 of the dataset -- which carries the 0.005 rad yaw draw and the expert's
own first step -- against main.  About 1 % mean absolute difference is the expected size of
that, and it is the number a reader should quote when asking whether training frames sit in
main's observation distribution.

### 9.2 The run (`a3_b2_aligned4000`)

From `a3_b2_unfreeze_20k/checkpoints/020000` -- not from the base.  4000 steps, one cosine
**5e-06 -> 2.5e-07**, warmup 133, batch 6, seed 1000, the same partial un-freeze (vision +
18 LLM layers + state projector), pyav, fused AdamW.  Four checkpoints kept, no pruning.

The peak is 5x below the recipe's 2.5e-05 for the reason §8 measured: re-heating an already
trained two-batch policy destroys both batches and spends 16000 steps repairing itself.  A
scene shift is a smaller change than that, and the loss agrees -- 0.022 at 1000, 0.002 at
4000, never a spike.

Measured: 1.24 s/step, 1 h 22 m, 31.6 GB VRAM, 84 GB of checkpoints.

### 9.3 The protocol, and why the old baseline does not apply

`scripts/render_bridge_policy_video.py` grew `--main-aligned` in the same place
`--without-cookies` already sits: *after* `env.reset`, because the reset is what places the
Cookies, the bins and the cameras.  All three sigmas are 0, so the scene is **frozen** --
every seed runs one identical arrangement and only the rollout differs.  `A3_PI05_SEED=0`,
20 seeds, 900 steps, one bridge and 20 workers per checkpoint.

Frozen means these numbers are not the layout-averaged numbers of §5 and §8.  The 5.35 on
this branch's `a3_cookie_batch_second_batch` scene and the 6.38s of §8 come from the scene's
own randomised layout; they cannot be compared with the table below, and mixing them is
exactly the mistake §8 had to correct.

### 9.4 What 4000 steps bought

Same stack, 20 seeds, main's arrangement frozen:

| checkpoint | mean | distribution |
|---|---|---|
| 20k incumbent (control) | **0.00** | {0: 20} |
| +1000 | 4.25 | {0: 3, 1: 1, 5: 12, 6: 4} |
| +2000 | **0.00** | {0: 20} |
| +3000 | 5.00 | {5: 20} |
| **+4000** | **7.55** | {5: 9, 9: 4, 10: 7} |

The control is the important row, and it is worse than it looks: `in_source` never leaves
80, nothing is ever held, and the arm's mean motion is 0.004 rad over 900 steps.  20k does
*not* fail at this scene so much as not attempt it -- the same 0/20 main's own benchmark
reported, now with a receipt.  Re-running it under a different sampler seed
(`A3_PI05_SEED=7`) gives 0/5 again, so the zeroes are the weights, not one unlucky draw.

7.55 is the first time this arrangement is solved at all, and 7 of 20 seeds place all ten
Cookies -- both batches -- from a state the policy produced itself.

**2000 is a V, not a fluke.**  All 20 seeds score 0 and the trajectory is distinct from the
incumbent's: it grasps five Cookies by step 100 and then holds them, rate-limited, for the
remaining 800 steps (`rate_limited_fraction` climbs 0.02 -> 0.05, motion 0.01 rad).  A
burst-then-freeze, not paralysis.  Reproduced at `A3_PI05_SEED=7` (0/5).  So the same V that
§8 found inside the 20k cycle also exists here at a twentieth of the learning rate: the
high-rate part of the cosine destroys the behaviour before the anneal rebuilds it.  The
practical reading is to keep all four checkpoints, always -- one sweep of a mid-cosine
checkpoint would have declared this arm a failure.

**The zero is the weights, and two confounds were checked.**  The control's 0.00 was not a
normalisation accident: the aligned dataset's own statistics differ from `a3_b2_merged`'s by
0.083 rad on state[0] and 0.173 against 0.113 on its std, which is ~0.7 sigma and enough to
ruin a policy, so the control was re-run with the incumbent's *own* statistics
(`stats=a3_b2_merged`) -- 0/5 again, with the same frozen arm.  Nor is it one unlucky
sampler draw: `A3_PI05_SEED=7` gives 0/5 as well.  Three independent stacks now agree on
this number (main's own benchmark without a shim, 0/20; this stack with the aligned
statistics, 0/20; this stack with the incumbent's statistics, 0/5).

### 9.5 Nothing was forgotten: the old layout improved too

The obvious risk of a scene-adaptation fine-tune is that it buys main's arrangement by
giving up the one the branch had already solved.  It did not -- same scene
(`a3_cookie_batch_second_batch`, its own randomised layout), 900 steps, 20 seeds,
`A3_PI05_SEED=0`, each model with its own training statistics:

| policy | mean | distribution |
|---|---|---|
| 20k incumbent | 5.35 | {0: 2, 3: 1, 4: 1, 5: 11, 6: 1, 9: 1, 10: 3} |
| +4000 aligned | **6.25** | {5: 15, 10: 5} |

Both ends of the distribution moved: tens 3 -> 5, zeros 2 -> 0.  The aligned data is a
narrower distribution than the randomised layout, and training on it acted as a regulariser
rather than a trade.  So the 4000-step checkpoint dominates the incumbent on *both*
arrangements, which is the reason to ship it and not a branch of checkpoints.

### 9.6 Verified on main's own benchmark -- and 7.55 does not transfer

Run in `/tmp/a3-main` (the worktree of main @ 29fe90e) with `bench_main_20k.py`, no scene
shim, 20 episodes, 900 steps, 20 workers, one bridge on `/tmp/bench-main.sock` holding the
4000-step checkpoint:

| main's benchmark | mean | distribution |
|---|---|---|
| default (`randomize_boxes` +-1 cm, cookies 0.3 mm) | **4.55** | {0: 4, 5: 13, 6: 1, 10: 2} |
| `--no-randomize` (layout frozen, closest to our training scene) | **5.00** | {5: 20} |

The incumbent's 0/20 there is unchanged, so the improvement is real on the authority stack:
+4.55 with main's randomisation, +5.00 without it.  But our own stack's 7.55 is *not*
reproduced, and the gap is not the scene -- section 9.1's gates rule that out.  It is the
evaluation itself.

**`A3_PI05_SEED` does not make a rollout reproducible.**  Same checkpoint, same frozen
scene, one worker, one seed, run twice: **5/10 and then 10/10**.  The seed pins the
flow-matching noise (`torch.manual_seed` before every `predict_action_chunk`), but the
inference is bf16 GPU work with nondeterministic kernels, and a 900-step contact task
amplifies last-bit differences into "one batch or two".  Two consequences:

* A single rollout is one draw from a distribution, so a *single-seed* comparison is not a
  comparison.  Main's frozen row above is 20 identical rollouts, which means the whole row is
  one draw repeated -- not evidence that this checkpoint always scores exactly 5.
* Our 7.55 and main's 5.00 differ by less than this measurement's own variance: 11 of our 20
  runs got the second batch, 0 of main's 20 did, and a repeat of one identical configuration
  spans 5 to 10.  Until the rollout is made deterministic (`torch.use_deterministic_algorithms`,
  autotune off, fp32) or the episode count is raised far beyond 20 on both sides, the honest
  claim is **"on main's benchmark: 0.00 -> 4.5-5.0, and up to 7.5 on our stack"**, not
  "7.55".

The 2000-step collapse and this nondeterminism point the same way: at this epsilon the task
is bimodal, and the only numbers worth quoting are distributions with the scene and the
stack both named.

### 9.7 What the 5.00 vs 7.55 gap actually was: an instrument, not a scene

Three measurements, in the order that isolates them.

**1. The simulator is deterministic; the policy is not.**  Replaying one fixed action sequence
twice gives cookie trajectories that differ by **0.000000 mm**, so every bit of the run-to-run
variation is in inference.  `A3_PI05_DETERMINISTIC=1` alone does not fix it (10 then 5);
`A3_PI05_DETERMINISTIC=1 A3_PI05_FP32=1` does (10 then 10).  The checkpoint runs in bf16, and
bf16 matmuls on this card are not bit-reproducible no matter which deterministic-kernel switch
is set.

**2. The two simulators agree.**  The same 548 actions -- episode 0 of the aligned dataset,
replayed with no policy at all -- put every Cookie within **0.454 mm** of the same place in
main's simulator and in this branch's, and both finish **10/10**.  `observation.state` agrees
to 5.4e-4 rad.  So dynamics, judge, action space and state are not the difference.

**3. What is left is pixels, and only pixels.**  At the same step of the same replay:
front 0.16-0.20, **left wrist 0.61-2.12**, right wrist **0.000** of 255.  The left wrist is
the camera that watches the grasp, and its difference grows exactly where the arm moves; the
right wrist, pointed at empty space, is bit-identical.  That is a renderer-configuration
difference (shadow/reflection passes over the arm and the lights), not a camera placement one:
the camera's world matrix matches to 6e-19.

So the earlier framing was wrong in both directions.  It is **not** a distribution difference
-- the scene is the same to 0.0004 mm -- and it is **not** an undiscovered large difference
either: the simulators agree to half a millimetre under identical commands.  It is a
**measurement** difference: a bimodal task scored with a nondeterministic sampler, on two
stacks whose renders differ by under 1 % in the one view that matters.  Main's twenty
identical 5s were one draw repeated; this branch's 7.55 was twenty draws from the same
coin, biased by that 0.6/255.

**What to do about it.**  Any comparison from now on should run with
`A3_PI05_DETERMINISTIC=1 A3_PI05_FP32=1`, which makes a rollout a function of (scene,
checkpoint) and costs speed.  Numbers produced without them -- including every sweep in
sections 5, 8 and 9.4 -- are samples, not measurements.  Chasing the last 0.6/255 is worth
doing only after both sides are scored deterministically, because until then it is below the
noise floor of the comparison it would be used for.

**And with the instrument fixed, the two stacks agree exactly.**  20 seeds each, 900 steps,
main's frozen arrangement, the same checkpoint:

| stack | mean | distribution |
|---|---|---|
| main's benchmark, deterministic + fp32 | **5.25** | {5: 19, 10: 1} |
| this branch, deterministic + fp32 | **5.25** | {5: 19, 10: 1} |

Same mean, same distribution, and the single success lands on a different seed (10 here, 19
there) -- which is what a knife-edge looks like when the only differences left are 0.2 mm of
main's own residual bin jitter and 0.6/255 in the left-wrist render.  So:

* the 7.55 was **the sampler**, not the stack: 20 draws from a bimodal outcome, averaged;
* the honest figure for this checkpoint on this arrangement at 900 steps is **5.25**, i.e. it
  completes the second batch in one run out of twenty, and 5.0-5.25 is the plateau both
  stacks reach;
* the comparison the section opened with was never a physics question at all.

The checkpoint table in 9.4 has to be read with that in mind: 4.25 / 0.00 / 5.00 / 7.55 are
*before* the fix, so the ordering between 1000 and 3000 is inside the sampling noise, and only
the 20k zero and the 2000 collapse (0/20 twice, two sampler seeds) are large enough to survive
it.  Re-running the four checkpoints deterministically is the first thing to do before any of
them is called better than another.

### 9.8 The rebuilt table, and the variance that survives determinism

Five checkpoints, `A3_PI05_DETERMINISTIC=1 A3_PI05_FP32=1`, `A3_PI05_SEED=0`, 20 seeds, 900
steps, each scored against its own training statistics (`a3_b2_merged` for the 20k incumbent,
the aligned set for the four fine-tuned ones):

| checkpoint | mean | distribution |
|---|---|---|
| 20k incumbent | **0.00** | {0: 20} |
| +1000 | 2.30 | {0: 6, 1: 3, 2: 1, 3: 2, 4: 5, 5: 3} |
| +2000 | 2.50 | {0: 10, 5: 10} |
| +3000 | 7.40 | {5: 8, 9: 12} |
| +4000 | **10.00** | {10: 20} |

This is a much cleaner story than 9.4: 20k cannot do the arrangement at all, 1000-2000 learn
one batch and stall, 3000 nearly always finishes both, and 4000 finishes both on every seed.
Notably +2000 is no longer a collapse -- the "V" in 9.4 was the sampler, not the training.

**The residual variance is between processes, not within one.**  The 4000 row was measured in
three separate launches of the same command: **5.25** {5:19, 10:1}, then **10.00** {10:20},
then **10.00** {10:20}.  Inside a launch the rollouts are reproducible (the same seed twice in
one process gives the same score), and `CUBLAS_WORKSPACE_CONFIG` is set before any CUDA work,
so this is not the usual cuBLAS-workspace trap -- fp32 GPU inference on this card simply is not
bit-identical from process to process, and this task's margin is thinner than that.

Practical reading: the **20k zero is solid** (four independent measurements, every one 0), the
ordering 1000 < 2000 < 3000 < 4000 is wider than the between-launch spread at every step except
3000->4000, and the 4000 figure should be quoted as **"10.00 on two launches out of three,
5.25 on the third"** rather than as a single number.  Getting an error bar on the other three
points costs three launches each -- 12 minutes apiece -- and is worth doing before any of them
is used to choose a training recipe.

### 9.9 Correction: 9.7's "both 5.25" was a coincidence, and the scene was under-specified

Section 9.7 closed by reporting 5.25 on both stacks and calling the gap closed.  Repeated
launches say that was one number meeting one number:

| stack, 4000 checkpoint, det+fp32 | launch 1 | launch 2 | launch 3 |
|---|---|---|---|
| main's benchmark, `--no-randomize` | 5.25 {5:19, 10:1} | **5.00** {5:6} | **5.00** {5:6} |
| this branch, aligned scene, zero jitter | 5.25 {5:19, 10:1} | **10.00** {10:20} | **10.00** {10:20} |

Both numbers are real and both are reproducible *for their stack* -- and they disagree.  The
reason is not physics and not the renderer; it is that **the two "frozen" scenes are not the
same kind of frozen.**  Main's `--no-randomize` still moves both bins by up to 0.2 mm from
episode to episode (measured from its own per-episode `target_bin_pos`: 94.8-95.0 mm in x,
99.2-99.9 mm in y), while this branch's alignment hook pins them to the dump to the last digit.
So our evaluation was strictly *more* frozen than the benchmark it was meant to reproduce, and
the exactly-frozen arrangement happens to be a favourable point: all 20 seeds are the same
rollout, and it succeeds.

Give our evaluation the same 0.2 mm of jitter main has, and the uniform tens break up:

| our aligned scene | mean | distribution |
|---|---|---|
| zero jitter | 10.00 | {10: 20} |
| `--align-bin-noise 0.0002` | **7.45** | {5: 10, 9: 1, 10: 9} |

That is the same bimodal behaviour main shows, and it closes most of the gap: with jitter, 7.45
against main's 5.00-5.25 instead of 10.00 against 5.25.  What is left (~2 cookies) is not yet
attributable, because the two jitters are not the same shape -- ours is Gaussian sigma 0.2 mm
on both bins, main's is a small systematic per-episode offset that its own reset produces.
Matching it exactly means replaying main's recorded per-episode bin offsets through our hook,
which is the next measurement to make, not an argument to have.

The methodological lesson, and the reason this section exists: **"the same scene" is not a
complete description of an evaluation.**  Two stacks can agree to 0.0004 mm on the arrangement
and still disagree by 5 cookies because one of them jitters and the other does not.  Any
comparison between these two benchmarks has to state (a) the arrangement, (b) the jitter model,
and (c) whether the sampler is deterministic -- in that order, and all three.

### 9.10 Both explanations above are wrong, and what is left

Two measurements, made after writing 9.9, remove every difference it blamed.

**Main's "frozen" is zero-jitter too.**  Resetting main's benchmark for each of the 20 seeds and
reading the bins *before* any rollout gives the same pose every time: target (0.095000,
0.100000), source (0.175000, 0.315000), **0.0000 mm spread**.  The 0.2 mm variation quoted in
9.9 came from `EpisodeScore.target_bin_pos`, which `episode_execution.py:289` reads *after* the
rollout -- it is the arm nudging the box, not the reset.  So our hook and main's reset start
from the identical arrangement, and the 7.45-vs-5.00 "convergence" that 9.9 used as evidence was
simply our evaluation perturbed away from a favourable point.  Gaussian jitter of main-like size
is a sensitivity probe, not a reproduction of main.

**The renderers agree at the same state.**  One state (step 300 of the expert replay) written
into both environments joint by joint -- 99 joints by name, qvel zeroed -- renders as:

| camera | MAE | max | pixels off by >2/255 |
|---|---|---|---|
| front | **0.000** | 1 | 0.00 % |
| left_wrist | **0.010** | 76 | 0.04 % |
| right_wrist | **0.000** | 0 | 0.00 % |

So the 0.61-2.12 left-wrist difference measured during replay was entirely the 0.45 mm of
trajectory drift that travelled with it, not the renderer.  The two stacks see the same pixels
when they are in the same state.

**What is actually left.**  Identical reset state, identical pixels at equal state, `state`
equal to 5.4e-4, and a fixed action sequence landing within 0.454 mm -- yet main reads 5.00 on
three launches and this branch reads 10.00 on two of three, with 5.25 on the third.  The one
number that does not fit is our own third launch: the same stack, same scene, same weights,
produced both 5.25 and 10.00.  That is a per-process coin flip in fp32 GPU inference, and it
means the 5-vs-10 split between stacks is, so far, **not distinguishable from that coin** --
three launches on one side and three on the other cannot separate a 50/50 coin from a biased
one.  Estimating the rate is what the `mode_rate` runs are for; until it exists, the correct
statement is the deflating one: on this arrangement the checkpoint either finishes both batches
or stalls after the first, the outcome is decided somewhere inside fp32 inference, and neither
benchmark can currently tell you which it will be.

### 9.11 The mode is per-process, and each stack has its own propensity

`mode_rate` launched each stack eight times (four rounds of 6 episodes, then the earlier
sequences) with identical flags, scene, checkpoint and settings.  Within a launch every episode
lands on the same side; across launches the side is what changes:

| launch | this branch (6 eps) | main (6 eps) |
|---|---|---|
| 1 | 10.00 {10: 6} | 5.00 {5: 6} |
| 2 | 10.00 {10: 6} | 5.00 {5: 6} |
| 3 | 10.00 {10: 6} | 5.00 {5: 6} |
| 4 | 10.00 {10: 6} | 5.00 {5: 6} |

Counting the earlier 20-episode launches as well: **this branch reaches the ten-mode in 6 of 7
launches** (the seventh scored 5.25), **main in 0 of 7**.  Under a shared 50/50 coin the chance
of seven main launches all landing on one side is ~1 %, and combined with ours 6-of-7 it is
~2e-5.  So this is not sampling noise: each stack has its own propensity, decided per process,
and main's 5.25 is what main always produces on this arrangement -- not a bad draw.

What that leaves is the only measured difference between the stacks: **0.454 mm of trajectory
drift when both are fed the same 548 actions**, with every other measured quantity equal
(reset pose 0.0000 mm, state 5.4e-4 rad, pixels at equal state 0.000-0.010 of 255).  A
sub-millimetre difference deciding five cookies sounds absurd until it is set against what this
task tolerates: the Cookies are packed 2.5 mm apart, and a 0.01 rad change of the arm's home
pose is enough to make the scripted expert fail outright.  The margin is thinner than the
difference, which is the same shape as the 2-cookie discrepancy that started all of this.

The one measurement that would settle it: record the *policy's* action sequence from a ten-mode
rollout here and replay it into main's environment.  If main then completes both batches, the
drift is not the cause and the difference is upstream in the observation path; if it stalls at
five, the 0.454 mm is the cause and this is settled.

**The drift is not the cause.**  Recording the policy's own 900 actions from a ten-mode rollout
in this branch (it reaches 10/10 at step 600) and replaying them with no policy at all:

| replayed into | result |
|---|---|
| this branch's environment | in_target **10**, in_source 70 |
| main's environment | in_target **10**, in_source 70 |

So when the same actions arrive, both simulators finish the task.  The 0.454 mm of dynamics
drift cannot explain a stack that stalls at five, because it does not stall when handed those
actions.  What differs is therefore **upstream of the actions**: the same checkpoint, at the
same seed, under the same flags, in two processes, produces two different action sequences --
"finish both batches" in most launches here, "stall after the first" in every launch on main's
side and in one of seven here.

Everything the model is given has now been measured equal -- scene (0.0000 mm), pixels at equal
state (0.000-0.010 of 255), `observation.state` (5.4e-4), prompt, statistics, checkpoint, seed,
precision.  What has *not* been compared is the tensor the adapter actually hands to the model:
the resized uint8 frames, their channel order and dtype, and the normalised state vector.  That
is the next probe, and it is a hash comparison, not a rollout: dump the inputs in both stacks
and diff.  If they are identical, then the two stacks are feeding the same model the same bytes
and the mode difference is inside fp32 inference -- at which point it is not a bug in either
benchmark but a property of the instrument, and the only honest response is to quote
distributions over launches rather than single numbers.

### 9.12 The hash comparison: the model is deterministic, the client is not

`a3_bridge_server.py` grew an env-gated `A3_LOG_IO_SHA` that prints a hash of every inference's
input arrays and output actions -- the single place both stacks pass through, so a difference
there cannot be blamed on anything downstream.

**1. The model answers identically to identical input.**  Two separate launches of *this* stack,
same scene, same seed, 150 steps = 10 chunks:

```
inputs identical across the two runs:  True
outputs identical across the two runs: True
```

Every chunk, input *and* output.  So fp32 + `A3_PI05_DETERMINISTIC` really does make inference a
function of its input, across processes; the earlier 5.25-versus-10.00 spread within this stack
has to come from the input, not from the model.

**2. Across the stacks the input differs at step 0, in the images only.**

| array | this branch | main | same? |
|---|---|---|---|
| `observation.state` | 7ad58774b8e2 | 7ad58774b8e2 | **yes** |
| `right_wrist_view` | 7da91cb49086 | 7da91cb49086 | **yes** |
| `front_view` | e26a8afa735b | f6b222d76169 | no |
| `left_wrist_view` | bd7feac43a7d | 7b64befb997a | no |

**3. Everything upstream of those images is bit-identical**, measured directly:

* bins after each stack's own reset: identical;
* **all 80 Cookie poses after each stack's own reset: max 0.000000 mm, 0 of 80 different**;
* the joint state the model receives: same hash;
* the raw rendered observation at that state: **MAE 0.0000, max 0, 0 pixels different, all
  three cameras**.

So the two simulators, at their own resets, are bit-for-bit the same scene; the model is
deterministic; and yet the two *clients* hand the bridge different `front_view` and
`left_wrist_view` tensors.  Both resize with `cv2.INTER_AREA` to 224x224 (`observation.py` and
`bench_main_20k.py:238` agree on the filter), so the difference must be in what each client
feeds that resize or how it names/orders what it feeds it.

That is where the 5-versus-10 lives: **a client-side image-preparation difference**, amplified by
900 steps of contact.  It is not the scene, not the renderer, not the physics, not the judge, and
not the sampler.  The next step is one log line with shape/dtype/mean per array on our side --
the sweep's bridge log did not capture it in the run above -- and then a diff that names the
line of code.

### 9.13 Found it: the alignment dump never carried the Cookies' orientation

The hash probe named the arrays, and the arrays named the field.  Our `front_view` and
`left_wrist_view` differed from main's while `right_wrist_view` and `observation.state` matched
-- so the difference was something visible to two cameras and invisible to the third, and not in
the state vector.  The Cookies are exactly that: 80 objects in the first two views, none in the
third, and their poses are free joints that `observation.state` (arm joints + grippers) does not
carry.

Measured at each stack's own reset, seed 0:

| quantity | this branch | main | same? |
|---|---|---|---|
| bin poses | -- | -- | yes |
| Cookie **positions** (80) | -- | -- | **0.000000 mm, 0 of 80 different** |
| Cookie **quaternions** (80) | `[1, 0, 0, 0]` | `[0.999976, 0, 0, -0.006885]` | **0 of 80 equal, up to 0.43° of yaw** |
| rendered frames | -- | -- | differ exactly in front + left wrist, 0.005 / 0.001 mean of 255 |

`/tmp/main_scene.json` has `cookie_positions` but no `cookie_quaternions`, so
`main_alignment.apply_scene` could only pin positions and `set_cookie_pose` fell back to its
default -- perfectly upright.  The scene we collected on and evaluated on therefore differed from
main's by a sub-degree rotation of every Cookie, and that is enough to move the rendered pixels
and, through 900 steps of contact, the score.

**The fix and what it costs.**  `apply_scene` now reads an optional `cookie_quaternions` field
(older dumps still load); the dump needs the field added from main's own reset.  With it pinned:

| aligned evaluation, 20 seeds, det+fp32 | mean | distribution |
|---|---|---|
| Cookies upright (as collected) | 10.00 | {10: 20} |
| **Cookies at main's orientations** | **7.25** | {5: 11, 10: 9} |
| main's benchmark | 5.00-5.25 | {5: 6} / {5: 19, 10: 1} |

So the missing field explains most of the 10.00-versus-5.00 gap and not all of it: 7.25 sits
between.  The dump also carries `gripper_state`, which `apply_scene` does not apply either, so at
least one more unpinned input remains -- and the same probe (per-array hash at step 0) will name
it in one run.

**What this means for the dataset.**  The 1000 aligned episodes were collected with upright
Cookies, i.e. on a scene that differs from main's benchmark by 0.43° of Cookie yaw.  That is a
genuine train/benchmark mismatch of exactly the kind this whole section has been chasing, and it
is a property of the data, not of the evaluation: fixing the evaluation does not change what the
policy was trained on.

### 9.14 The re-collection: same seeds, same schedule, one field more (10-19)

The dump was rebuilt with the field (`/tmp/dump_main_scene.py` writes `cookie_quaternions`
from each Cookie's free-joint `qpos[3:7]`) and re-verified before a single episode was recorded:
quaternion difference **0.000e+00, 0 of 80 different**, render means `80.070938` front /
left wrist `-0.000011` / right wrist `0.000000` against main's own frames.  The scene adapter
needed no further change -- `apply_scene` already read the optional field.

The collection is a re-run of seeds **2000-2999**, not a new seed range: the point is to change
one input and hold everything else fixed, so the new 1000 episodes are comparable episode by
episode with the old ones.  Command line identical to the first pass except the dump:

```
--main-aligned /tmp/main_scene_full.json --position-noise 0 --align-pose-noise 0 \
--align-bin-yaw-noise 0.005
```

20 shards x 50 episodes, 1000 attempts, **1000 accepted** (100.0%), both gates passed:
non-finite 0, parquet rows 425,393 == frame sum, modes 250/250/250/250.  Frame counts are the
same shape as the first pass (full 548/548.7/550, prefill 302/302.0/303, perturb 548/548.8/550,
prefill_perturb 302/302.1/303) -- which is the expected outcome: pinning the Cookies' rotation
changes what the cameras see, not how long the expert takes.

The frame gate moves in the right direction, and only on the two cameras that can see a Cookie:

| first frame vs main's render, MAE of 255 | revision 1 (upright) | revision 2 (oriented) |
|---|---|---|
| front | 2.248 | **2.198** |
| left wrist | 3.339 | **3.061** |
| right wrist | 1.475 | 1.475 |

Right wrist is bit-identical across the two revisions, which is the control: that camera sees no
Cookie at reset, so a change there would have meant something other than the orientations had
moved.

**The second arm.**  Same recipe as 9.1, base = `a3_b2_aligned4000/checkpoints/004000` (i.e. the
20k incumbent plus the first 4000 aligned steps), 4000 more steps on the re-collected set,
cosine 5e-06 -> 2.5e-07, warmup 133, batch 6, seed 1000, `--save_freq=1000`, ~1.21 s/step,
output `a3_b2_aligned_v2_4000`.  Launcher: `/root/autodl-tmp/logs/aligned_v2_finetune.sh`.
Evaluation to follow on `/tmp/main_scene_full.json` with `A3_PI05_DETERMINISTIC=1
A3_PI05_FP32=1`, 20 seeds, 900 steps.

### 9.15 The second arm did not learn anything, and the third arm asks the real question (10-19)

The second arm continued `a3_b2_aligned4000/004000` for 4000 steps on the corrected data.  Its
loss was **0.002 at step 20**, against **0.093 at step 20** for the first arm: the base already
fit a dataset that differs from the one it was trained on only in the Cookies' sub-degree yaw.
There was no gradient left to follow, so the 4000 steps were a warm restart of the cosine at
5e-06 with nothing to optimise -- drift, not training.

It shows up as a regression, measured on main's actual reset (same instrument, same scene
digest `cookies 80 sha=535695d0`, yaw noise 0, det+fp32, 20 seeds, 900 steps):

| checkpoint | trained on | main's scene | rev-1 upright scene |
|---|---|---|---|
| `a3_b2_aligned4000/004000` | rev-1, 4000 steps from 20k | **7.25** {5: 11, 10: 9} | 10.00 {10: 20} |
| `a3_b2_aligned_v2_4000/003000` | rev-2, +3000 from above | 5.00 {5: 20} | -- |
| `a3_b2_aligned_v2_4000/004000` | rev-2, +4000 from above | 5.00 {5: 20} | 9.75 {10: 19, 5: 1} |
| main's own benchmark | -- | 5.00-5.25 | -- |

Three checkpoints of the arm and its base: the collapse is present by 3000 and stays, and it is
specific to the scene where the second-batch insertion is hard (upright 9.75 against 10.00, a
0.25 difference; oriented 5.00 against 7.25, a 2.25 difference).

**What the policy does instead.**  Seed 0, step by step: the first batch is placed by step 250
in both checkpoints, and both have grabbed the second batch by step 400.  Then they part --
the base releases into the box by step 500 and reads 10/10 at 850; the drifted arm keeps
holding, is still at `in_source 75` at step 800, releases at 850, and every one of the five
lands `below_rim, on_the_floor, OUT` at floor +28.8 mm.  So it is not a forgotten skill: it is
a second-batch insertion that misses by roughly the box wall, with the Cookies held for 450
extra steps first.

**The data itself is not the suspect, and this was checked before blaming it.**  rev-1 and rev-2
were collected on the same seeds with the same command; comparing the two datasets file by file:

| check | result |
|---|---|
| first frame, `right_wrist` (sees no Cookie) | **MAE 0.000000** -- bit-identical |
| first frame, `front` / `left_wrist` (see the Cookies) | 0.691 / 1.305 |
| same state rendered upright vs oriented, same camera pattern | 0.222 / 2.116 / **0.000** |
| `meta/stats.json` means and stds, all keys | identical to 1e-6 (only `count` differs, by 39 frames) |
| episodes, frames, per-camera video files (10 / 10 / 4) | identical |

The right-wrist camera is the control: a change there would have meant something other than the
Cookie orientations had moved, and there is none.  The dataset is what it claims to be.

**So the question the second arm was meant to answer is still open**, and it cannot be answered
by continuing -- only by starting from the 20k incumbent again on the corrected data
(`aligned_v3_finetune.sh`, same recipe, same seeds, same four conditions).  That arm is running;
its 1000/2000/3000/4000 checkpoints are all scored, because the slope matters more than the
endpoint when an arm is known to be able to drift.

**The published dataset is rev 2.**  `yunsx114/a3-dual-arm-cookie-main-aligned` was overwritten
in place (same repository id, no new one), 39 files, `MANIFEST.sha256` regenerated; the card now
documents the orientation correction and states that rev 1 is superseded.  Verified after
upload: `meta/info.json` reads 1000 episodes / 425,393 frames, the card sits at the repository
root, the remote `MANIFEST.sha256` is byte-identical, and two downloaded files (`file-001.parquet`,
one left-wrist video) hash to the same values as the local bundle.

### 9.16 The third arm: the corrected data does help, and the curve is not monotonic (10-20)

Third arm: the 20k incumbent, 4000 steps, the corrected dataset -- the recipe of 9.1 with a
different one of the two inputs.  From the 20k base the loss starts at **0.109 at step 20**
(against 0.002 for the drift arm of 9.15), so this arm actually trained.  Scored at every
checkpoint on main's actual reset, 20 seeds, 900 steps, det+fp32:

| checkpoint | main's scene | rev-1 upright scene |
|---|---|---|
| 001000 | 5.20 {5: 18, 7: 2} | -- |
| 002000 | 5.00 {5: 20} | -- |
| 003000 | 0.00 {0: 20} | -- |
| **004000** | **7.75** {5: 9, 10: 11} | 9.90 {9: 2, 10: 18} |

Two things fall out of that table.

**The curve is not monotonic, and 3000 is a hole, not a trend.**  Arm 1 read 3.50 / 2.25 / 6.80
/ 10.00 on its own scene at 1k/2k/3k/4k; this arm reads 5.20 / 5.00 / 0.00 / 7.75.  The 3000-step
checkpoint grasps three Cookies in its first 300 steps and never releases them -- 80 of 80
Cookies still in the source bin after 900 steps, three in the gripper, `floor +0.4 mm`, and the
rollout runs at 0.7 steps/s instead of 1.9 because the arm is dragging them through contact.
Reading a checkpoint from the middle of a cosine is therefore not reading the arm; both arms
have a collapsed checkpoint somewhere in their history, and only the endpoint is comparable.

**At the endpoint the corrected data is ahead, by two seeds.**  7.75 against 7.25 on main's own
reset, 9.90 against 10.00 on the upright scene -- so the gain is specific to the scene where the
second-batch insertion is hard, and costs nothing where it is easy.  Both arms read the same
scene digest (`cookies 80 sha=535695d0`), the same statistics file (the two datasets' stats are
identical to 1e-6), and the same seeds, so the rollouts are paired.

Two seeds out of twenty is not a result, so both checkpoints were re-run on **seeds 20-39**
immediately afterwards to decide the claim on 40 seeds rather than 20; those numbers go in 9.17
when they land.

**Where this leaves the numbers.**  The published card's 0.00 -> 7.55 came from the bf16,
non-deterministic instrument on the upright scene; under the deterministic float32 instrument
the same checkpoint reads 10.00 there and 7.25 on the benchmark's real scene.  The best
checkpoint we now have on that scene is this arm's 004000 at **7.75**, against the benchmark's
own policy at 5.00-5.25.  Every number in this section is a deterministic single measurement of
a single checkpoint, so no seed-level error bar is being quoted.

### 9.16 On the authority stack the two data revisions are indistinguishable (10-20)

Everything in 9.14-9.15 was measured in *our* env with main's layout transplanted and *our*
relaxed contract.  The number that matters to main is the one main's own code produces, so both
checkpoints were scored there: `/tmp/a3-main` at 29fe90e, `bench_main_20k.py`, `--no-randomize`,
900 steps, 20 episodes, 20 workers, weights served through the same bridge, det+fp32.

| checkpoint | trained on | main's benchmark | our scene, 40 seeds |
|---|---|---|---|
| `a3_b2_aligned4000/004000` | rev-1 (upright) | **5.00** {5: 20} | 7.50 {5: 20, 10: 20} |
| `a3_b2_aligned_v3_4000/004000` | rev-2 (oriented) | **5.00** {5: 20} | 7.75 {5: 18, 10: 22} |
| 20k incumbent, same two | -- | 0.00 (9.6) | -- |

Both read `failure_reason: max_steps_exceeded`, `cookies_in_target: 5`, `success_rate: 0.0`,
`mean_steps: 900`.  On main's judge the Cookie-orientation correction buys **nothing measurable**,
and neither does the extra 4000-step arm (5.00, 5.00, and 5.00 for the rev-1 checkpoint in 9.6).

**Why the two columns disagree, and what each one is allowed to say.**  The 40-seed column uses
this repository's contract (`require_upright: false`, `require_exact_slots: false`, "ten released,
settled and contained"), the 20-episode column uses main's (`15 deg` upright, exact slots, wall
contact, `success_hold_steps: 5`).  A rollout that ends 10-in-the-box-but-not-exact passes one and
fails the other, so the columns are not two measurements of one quantity.  The only claim the
data supports:

* **On main's benchmark** the aligned fine-tune moved the score from 0.00 to 5.00, and that is
  where it stops: 5.00 is the one-batch plateau, and 20 of 20 rollouts sit on it in every
  configuration tried today (rev-1 data, rev-2 data, 4000-step continuation, 4000-step fresh arm).
* **On a relaxed contract** the same checkpoints reach 7.5-7.75, and the two data revisions are
  one seed apart in 40 (7.50 vs 7.75, p ~ 0.8) -- indistinguishable.
* The 7.55 / 7.25 / 7.75 numbers in 9.1-9.15 are therefore **this branch's instrument**, not
  main's score, and the dataset card was corrected to say so.

**What is still unexplained.**  Five checkpoints across three arms and two data revisions all
land exactly on the 5/10 plateau under main's judge, while the same checkpoints split ~50/50
between 5 and 10 under the relaxed contract.  So the binding constraint on main's benchmark is the
strict acceptance test (upright + exact slot + hold), not the scene, not the data revision, and not
the sampler.  Closing the remaining gap means training for *precision*, not for the arrangement:
the demonstrations are 100 % successful under the relaxed contract and were never filtered for
exact-slot placement, so the data may simply not contain the signal main's judge asks for.

### 9.17 Where the "20k scored better than 5" memory comes from, and why nothing regressed

Four runs of the 20k incumbent through main's own benchmark exist, and none of them is a mean
above 5:

| file | checkpoint | scene | n | mean |
|---|---|---|---|---|
| `bench_main_20k_scene20.json` | 20k | `--training-scene /tmp/ours_scene.json --no-randomize` | 20 | 3.65 |
| `bench_main_20k_scene20_w4.json` | 20k | same, 4 workers | 20 | **4.45** (one seed reached 10) |
| `bench_main_seed0.json` | 20k | same | 20 | 3.70 |
| `bench_main_random_seed0.json` | 20k | `--scene-mode cameras` (main's own per-seed randomisation kept) | 20 | 0.00 |

The numbers above 5 that this session has quoted for the incumbent are from **our** instrument,
on **our** arrangement: 5.35 (randomised layout, our judge, 9.5) and 7.75 on a four-seed batch of
`a3_b2_unfreeze_20k`.  They are not main's benchmark.

So the score sheet on main's judge is monotone, not regressive:

| checkpoint | scene it is scored on | main's benchmark |
|---|---|---|
| 20k incumbent | main's layout, no shim | 0.00 (9.6) |
| 20k incumbent | its own layout restored | 3.65-4.45 |
| aligned +4000 (rev-1) | main's layout (= its training layout) | 5.00 |
| aligned-from-20k +4000 (rev-2) | main's layout | 5.00 |

The aligned fine-tune is the only thing that gets off zero on main's own layout, and it does so by
+5.00.  The flat 5.00 across checkpoints is **9.11's per-stack propensity**, not a fresh
regression: the identical checkpoint reads 10.00 in four launches on this branch's stack and 5.00
in four launches on main's, reproducibly, before any of today's data work.  `cookies_in_target`
in main's code is a geometric count (`_cookie_inside_target`), so the 5 is "the second batch never
got into the box", not "it got in and was rejected by the strict criterion" -- the strict part
only gates `success`, which is 0 % everywhere.

What would still be worth trying, in order of information per GPU-hour: (1) more steps under
main's judge (1400 instead of 900) to see whether the plateau is a budget artefact; (2) a
rollout-level comparison of the two stacks on the *same* observation stream, which 9.12 shows is
already bit-identical for `right_wrist_view` and `observation.state` and differs only in the two
Cookie-facing views; (3) demonstrations filtered for exact-slot placement, which is the one thing
the current data never asked for.

### 9.18 The task is knife-edge at the micrometre, so the cross-stack gap cannot be "aligned away" (10-20)

The render now matches between the stacks to 0.0009 / 0.0136 / 0.0000 of 255 at an equal joint
state, and a 548-step identical-action replay lands every Cookie within 0.454 mm on both.  Those
are the residuals.  This section asks whether that size of residual can matter -- by perturbing
the scene on *one* stack, far below the residual, and re-running the same 20 seeds.

The perturbation is target-bin yaw noise, seeded per episode: 1e-4 rad is 0.006 degrees, i.e.
~5 micrometres at the box rim, 50x smaller than the 0.005 rad the dataset itself varies and 70x
smaller than the 0.43-degree Cookie yaw the alignment fix restored.

| scene | mean | distribution | seeds that changed verdict vs no perturbation |
|---|---|---|---|
| frozen (yaw 0) | 7.75 | {5: 9, 10: 11} | -- |
| + 1e-4 rad (5 um) | 6.60 | {0: 2, 3: 1, 4: 1, 5: 7, 10: 9} | **10 / 20** |
| + 1e-2 rad (0.5 mm) | 6.10 | {0: 3, 4: 1, 5: 8, 9: 2, 10: 6} | **14 / 20** |

Ten of twenty episodes change their score when the box is rotated by five micrometres.  Two seeds
that read 10 collapse to 0, one drops to 3; two seeds that read 5 jump to 10.  The first batch --
the part every checkpoint gets right -- is itself occasionally lost.

**Consequences, in order of how much they change what is worth doing.**

1. **Per-seed cross-stack comparisons are meaningless.**  A stack difference of 0.454 mm is ~90x
   the perturbation that flips ten seeds.  "Our stack gets 10, main's gets 5" on a given seed is
   not evidence about the policy; it is a coin flip whose bias is set by sub-micron numerics.
2. **20 seeds cannot resolve a 7.75-versus-5.00 gap on this task.**  With 10/20 seeds unstable
   under a change far smaller than the between-stack residual, the binomial noise floor of a
   20-episode comparison is of the same order as the effect.  Distributions over hundreds of
   seeds, or a continuous score, are the only honest instruments here.
3. **No amount of further scene alignment will close it.**  The scene already matches to 0.000 mm
   / 0.000e+00, the pixels to ~0.01/255, and the dynamics to 0.45 mm; each of those is *larger*
   than a perturbation that already moves the outcome.  The remaining gap is not a bug to find,
   it is the task's own sensitivity to floating-point detail across two independent simulators.
4. **What this does not excuse.**  Main's 0/20 at 5.00 is a genuine one-batch plateau in *that*
   stack, and the relaxed-contract 22/40 tens is genuine in this one; both are real observations
   of their own instrument.  They just do not estimate the same quantity, and neither is a
   measurement of "the policy" to better than the sensitivity above.

The practical reading: report the distribution, name the stack and the contract, and stop
attributing differences of this size to the model.

### 9.19 A fixed half-millimetre moves eleven of twenty seeds, and it moves them *one way* (10-20)

9.18 perturbed the scene randomly per episode and scattered the outcomes.  A real second stack
does not scatter: it applies *the same* slightly-different physics to every episode.  So the same
experiment was repeated with a single fixed offset -- the target bin moved +0.454 mm in x, which
is the size of the replay residual between the two stacks -- and nothing else changed.

| scene (same 20 seeds) | mean | distribution | seeds that changed |
|---|---|---|---|
| frozen | 7.75 | {5: 9, 10: 11} | -- |
| target bin **+0.454 mm fixed** | **8.80** | {4: 4, 10: 16} | **11 / 20** |

Seven of the nine "5" seeds become 10.  Two tens drop to 4.  Nothing about the policy changed --
only where the box is, by half a millimetre.

**What this settles.**  The 5-vs-10 outcome is not a coin flip that noise decides; it is a
deterministic function of a sub-millimetre geometric relationship between the second batch, the
slots and the box.  Each seed sits on one side of that relationship, and the side is decided by
detail at the 1e-4 m scale.  A stack-to-stack residual of 0.454 mm is therefore *large enough to
pin the whole distribution*, which is exactly what is observed: our stack reads ~50 % tens
reproducibly, main's reads 0 % tens reproducibly.  The direction is stable because the bias is
stable, not because we were lucky 4 times.

**What it does not settle.**  The sign.  The +0.454 mm offset made *our* stack better, not worse,
so "our stack scores higher" is not evidence that it is the more faithful simulator -- a shift of
the same size in the other direction could plausibly have hurt just as much.  The honest claim is
that the two stacks differ by an amount that dominates this task's outcome, with a sign we cannot
derive from first principles.

**The actionable consequence is about the policy, not the stack.**  Nine of twenty seeds miss the
second-batch insertion by less than half a millimetre, and they miss it the same way every time.
The dataset pinned the box (only a 0.005 rad yaw draw), so the policy never saw a placement error
of that size during training and has no reason to be robust to one.  Small box jitter in
collection -- ±0.5 mm, which is the scale that matters -- is the change most likely to remove the
knife-edge, and it is worth more than any further alignment of the two simulators.

### 9.20 What the last 1000 episodes actually varied (10-20)

From the collection's own log (`/tmp/collect_v2.log`), the launcher's `extra` line was:

```
--main-aligned /tmp/main_scene_full.json --position-noise 0 --align-pose-noise 0 --align-bin-yaw-noise 0.005
```

so the per-episode variation is exactly one degree of freedom:

| input | variation across the 1000 episodes |
|---|---|
| target bin **yaw** | Gaussian, sigma = 0.005 rad = 0.286 deg (from the episode seed) |
| target bin / source bin **translation** | **none** (`--position-noise 0`) |
| arm home pose | **none** (`--align-pose-noise 0`) |
| 80 Cookie positions | **none** (no jitter; `cookies_jittered: false`) |
| 80 Cookie orientations | **none** (identical to main's reset; that is the point of rev-2) |
| cameras | **none** |
| second-batch mode | `full` / `prefill` / `perturb` / `prefill_perturb`, by `seed % 4` (250 each) |
| in `prefill`, the first batch's imperfections | fixed by the mode: one coherent lean in [0, 4] deg ([0, 6] for `prefill_perturb`), yaw +-1 (+-1.5) deg, per-Cookie jitter +-0.2 (+-0.3) mm |
| physics seed, sampler | not in the data; the expert is deterministic |

**Why this matters after 9.19.**  sigma = 0.005 rad of yaw is 0.565 mm of displacement at the
box's corner (half-diagonal 0.113 m) -- but the box's **centre does not move at all**, and it is
the centre that the second-batch insertion is aimed at.  So the collection exercised rotation and
never exercised translation, while 9.19 shows that a translation of 0.454 mm is what flips eleven
of twenty seeds' outcomes.  The policy was trained on a scene whose insertion target sat at
exactly one place, and it fails by less than half a millimetre whenever it does not.

That is the change worth making in the next collection: keep everything above and add box
translation noise, so the placement error that decides the outcome is inside the training
distribution instead of outside it.

**Flag correction.**  The knob for that is `--align-bin-noise` (Gaussian sigma in metres, applied
to *both bins* on top of the aligned arrangement).  `--position-noise` is **per-Cookie** XY jitter
-- in the aligned path `cli.py` passes it to the hook as `cookie_noise_m`, not as a bin term --
and `--yaw-noise` is per-Cookie yaw.  Matching main's shipped `basic` profile therefore means

```
--align-bin-noise 0.01 --align-bin-yaw-noise 0.03 --position-noise 0.0003 --yaw-noise 0.015
--align-pose-noise 0
```

(bins +-1 cm, target yaw 0.03 rad, Cookies 0.3 mm / 0.9 deg, arm home pose pinned), not
`--position-noise 5e-4`.

**A reporting gap to close.**  `collection_summary.json` records `yaw_noise_rad: 0.0` and
`position_noise_m: 0.0`, which are the *scene config's* randomisation fields (zeroed by the two
`*_noise 0` flags).  The align-flag sigma of 0.005 rad is not in the sidecar at all -- the only
record of it is the launcher log.  Anyone reading the dataset's metadata alone would conclude the
layout was completely frozen.  The card states it correctly, but the sidecar should too.

### 9.21 The shipped benchmark is randomised, and on it the two data revisions separate (10-20)

Every previous number on main's judge was taken with `--no-randomize`, i.e. on the frozen layout
main ships as its `training_scene`.  But main's **default** benchmark is randomised: measured with
the shipped defaults, `cookies_in_target` is computed on a scene whose bins and Cookies have been
jittered.  That is the number a reader of the paper will see, so both aligned checkpoints were
scored there too -- same driver, same 20 episodes, same 900-step cap, same judge constants, no
shim, no `--no-randomize`:

| checkpoint | training data | mean | distribution | strict successes |
|---|---|---|---|---|
| `a3_b2_unfreeze_20k/020000` (incumbent) | original 20k, un-aligned | **0.00** | {0:20} | 0 / 20 (0 %) |
| `a3_b2_aligned4000/004000` (arm 1) | rev 1, pinned layout | **3.75** | {0:6, 5:12, 6:1, 9:1} | 0 / 20 (0 %) |
| `a3_b2_aligned_v3_4000/004000` (arm 3) | rev 2, main-aligned orientations | **4.05** | {0:5, 1:1, 4:1, 5:10, 6:1, 10:2} | **2 / 20 (10 %)** |

**The incumbent's zero is real, not a crash.**  Every base episode ran the full 900 steps at a
median wall time of 432 s, the same as the two aligned arms (419 s / 430 s), so the bridge was
healthy and the policy was acting for the whole episode; it simply never placed a single Cookie
inside the target.  With the layout frozen it scores 3.65 / 4.45 / 3.70 (9.17).  So the entire
0.00 -> 3.75 jump is *robustness to the randomised layout*, not a difference in the nominal task:
the incumbent only works when the bins sit exactly where it was trained to find them.

1. **The orientation fix is worth something, but not much.**  +0.30 mean and the first genuine
   strict successes (episodes 5 and 11, full upright-in-slot with the 5-step hold).  Under the
   frozen layout both checkpoints had been indistinguishable at 5.00 {5:20}, so the frozen judge
   was hiding the only difference the data revision produced -- it rewards getting the Cookies
   *near* the box and does not care whether they end up in the slots.
2. **The mode of the distribution is still "near but not in".**  Twelve of arm 1's twenty and ten
   of arm 3's are exactly 5: Cookies moved, footprint inside the box, but not upright in a slot.
   The remaining variance is the insertion itself, which is the same sub-millimetre quantity 9.19
   measured as decisive.  A 10 % strict rate is not a task the policy has learned; it is a task it
   sometimes gets away with.

**What the three rows together say.**  The aligned fine-tune is not "better at the task"; it is
the only one of the three that survives a layout it was not shown.  That reframes the next
collection: the question is no longer "does the aligned data help" (it does, 0.00 -> 4.05) but
"how much of that 4.05 is insertion and how much is luck", which is exactly what training on
jittered bins is meant to answer.

Reproduce:

```
/tmp/bench_main_rand.sh      # bridge per checkpoint, then bench_main_20k.py --episodes 20
                             # --workers 20 --max-steps 900, no --no-randomize, no shim
```

### 9.22 The aligned hook was perturbing an axis main never touches, and it cost two thirds
of the collection (10-20)

The first randomised collection (`a3_b2_main_aligned_v3_rand5mm`, 20 shards, seed 5000) was meant
to widen the layout from rev-2's pinned arrangement to main's `basic` profile.  It ran at a 34.7 %
expert acceptance rate, which was read as "5 mm of box shift is hard for the scripted expert".
That reading was wrong.

`apply_scene` drew `rng.normal(0, sigma, size=3)` for each bin, so the **vertical** axis moved too.
Main never moves it: `sim/randomization.py` draws `uniform(-s, s, size=2)` for the bins and writes
nothing in z.  And this branch's source bin is a **mocap** body (`_place_source_bin` writes
`mocap_pos`), which is kinematic -- a lifted bin never settles, the Cookies stay where gravity put
them, and the grasp becomes impossible.  Acceptance is almost exactly a function of that one
number, recovered from the seeds the logs record:

| \|source-bin z\| | attempts | acceptance |
|---|---|---|
| 0-1 mm | 394 | 80.2 % |
| 1-2 mm | 381 | 76.6 % |
| 2-3 mm | 360 | 67.8 % |
| 3-4 mm | 301 | **3.3 %** |
| 4-6 mm | 495 | **0.0 %** |
| 6-10 mm | 447 | **0.0 %** |
| > 10 mm | 104 | **0.0 %** |

Accepted episodes: \|z\| median 1.39 mm, max 3.36 mm.  The horizontal components were *not*
filtered -- accepted sd 4.78 / 4.64 mm against 4.93 / 5.09 mm over all attempts -- so the dataset
was a clean sigma=5 mm horizontal set, but 65 % of the compute went into episodes that were
unsolvable for a reason main's task does not contain.

Four other places where the hook disagreed with `basic`, found in the same audit:

| term | main `basic` | hook before | hook now |
|---|---|---|---|
| bins, horizontal | `uniform(+-10 mm)` per axis | `normal(0, 5 mm)` | `uniform(+-10 mm)`, default |
| bins, vertical | **none** | `normal(0, 5 mm)` | none (`--align-noise-3d` to restore) |
| target yaw | `uniform(+-0.03 rad)` | `normal(0, 0.03)` | `uniform(+-0.03)` |
| Cookie xy | `uniform(+-0.3 mm)` | `normal(0, 0.3 mm)` in 3D | `uniform(+-0.3 mm)`, horizontal |
| Cookie yaw | `uniform(+-0.015 rad)` | **0** -- `--yaw-noise` is overwritten by the hook | `--align-cookie-yaw-noise`, passed through |

The Cookie-yaw row is the quiet one: `post_reset` runs after `env.reset`, so it replaces the
quaternions with the dumped ones and the environment's own `yaw_noise_rad` never reaches a frame.
A flag that is silently inert is worse than a flag that is missing.

**Measured after the fix.**  Six episodes at `--align-bin-noise 0.01 --align-noise-dist uniform`
with the horizontal-only default: **6 attempts, 6 accepted**.  Against 10-13 % at the same
magnitude in 3D.  The vertical axis was the whole effect.

`collection_summary.json` now also records a `post_reset` block with every sigma the hook applied,
because this is the fourth dataset whose real distribution lived only in a launcher command line.

### 9.23 Matching the benchmark's own distribution made the model worse, not better (10-21)

9.22 produced the first dataset whose layout noise is main's `basic` term for term: 1000 episodes,
1107 attempts, 90.3 % expert acceptance, bins `uniform(+-10 mm)` horizontal, target yaw
`uniform(+-0.03)`, Cookies 0.3 mm / 0.9 deg, arm pose pinned.  Arm 4 trained on it from the 20k
incumbent with arm 3's recipe, and was scored on main's shipped randomised benchmark, same 20
episodes as every other row:

| checkpoint | steps | mean | distribution | strict successes |
|---|---|---|---|---|
| arm 4 | 2000 | 3.45 (SE 0.78) | {0:9, 5:8, 9:1, 10:2} | 0 / 20 |
| arm 4 | 3000 | 3.30 (SE 0.54) | {0:7, 5:12, 6:1} | 0 / 20 |
| arm 4 | **4000** | **1.75** (SE 0.53) | {0:13, 5:7} | 0 / 20 |
| arm 3 (no layout noise) | 4000 | 4.05 (SE 0.66) | {0:5, 1:1, 4:1, 5:10, 6:1, 10:2} | 2 / 20 |
| arm 1 | 4000 | 3.75 (SE 0.58) | {0:6, 5:12, 6:1, 9:1} | 0 / 20 |
| 20k incumbent | -- | 0.00 | {0:20} | 0 / 20 |

Three things are visible and they pull in different directions.

1. **The data works for the expert and not for the policy.**  The expert accepted 90 % of the
   episodes it was asked to solve at this noise level, so the noise is inside the task's
   solvable set.  The policy, trained on it, ends up worse than one trained on the pinned
   layout.
2. **It degrades with training, monotonically.**  3.45 -> 3.30 -> 1.75, and the zero-score
   episodes climb 9 -> 7 -> 13.  That is the arm-2 shape again, on different data, which is why
   the first suspect is the schedule rather than the distribution.
3. **The run has seen 5.6 % of one epoch.**  `ep:56 epch:0.06` at 4000 steps with batch 6 over
   425 501 frames.  Loss fell 0.089 -> 0.001-0.003, so the model fit what it saw; it did not
   learn the layout *distribution* from 24 000 samples of it.

An extra detail worth recording: two of arm 4's 2000-step episodes score 10 while `success` is
False -- every Cookie geometrically inside the box, none upright in a slot.  The judge's count and
the success contract separate at this granularity, exactly as 9.21 describes.

**What was launched instead.**  A from-scratch fine-tune of `lerobot/pi05_base` -- the same
checkpoint the 20k incumbent itself started from -- for **20 000 steps** on the same
layout-randomised data.  The earlier instinct was to add 10 000 steps on top of the incumbent at a
lower peak LR, and that was the wrong reading of item 3: 4000 steps on top of a 20k model is 24 000
samples of a 425 501-frame epoch, so "starved of steps" is true, but continuing a converged model
cannot fix it.  The run that has the same budget as the incumbent is the one that answers whether
the layout-matched distribution is *learnable* at 1000 episodes.

Recipe otherwise exactly the incumbent's: partial un-freeze (18 LLM layers, vision tower, state
projector), batch 6, seed 1000, cosine 5e-06 -> 2.5e-07 over the 20 000 steps, warmup 666,
`--dataset.video_backend=pyav`.  At 120 000 samples that is 28 % of an epoch rather than 5.6 %.

Step-20 loss is **0.190**, against 0.089 for the arms that start from the 20k incumbent -- the
base does not know the task, which is the expected difference and a useful check that this run is
not secretly continuing the old weights.

Checkpoints every 5000 steps (5k / 10k / 15k / 20k), each scored on main's shipped randomised
benchmark as soon as training ends, one line per checkpoint into `/tmp/overnight_summary.txt`.
This run is also the only arm whose intermediate points can say whether the degradation in arm 4
was the schedule (it should now be monotone) or the data (it will not be).

### 9.24 The 4.05 checkpoint is published, with its failure breakdown attached (10-21)

`yunsx114/a3-dual-arm-cookie-pi05-aligned` on ModelScope: 13 files, the weights at the repository
root as a normal LeRobot policy directory, plus the two evaluation records that produce the
numbers this line has been quoting.

| file | |
|---|---|
| `model.safetensors` | 9.35 GB, bf16 |
| `config.json`, `train_config.json` | policy config and the exact training configuration |
| `policy_{pre,post}processor*.json` / `*.safetensors` | feature keys and the dataset's quantiles (pi05_base ships none) |
| `eval/bench_main_basic_randomised.json` | the 4.05 run, per episode |
| `eval/bench_main_frozen_layout.json` | the 5.00 run, per episode |
| `README.md`, `NOTICE.md`, `MANIFEST.sha256`, `upload_modelscope.sh` | card, licence notice, integrity, the release script |

The card states 4.05 with its standard error, says plainly that the 5.00 is a plateau rather than a
score, gives the 7.75 on our own stack as a *different instrument*, and includes the failure split
that 9.23's diagnosis produced -- 11/20 stop after one batch, 9/20 attempt the second, 2/20 finish.
Shipping the checkpoint without that table would invite exactly the misreading that
`cookies_in_target` produces on every checkpoint that can place five.

Licence is **gemma**, not Apache: the weights are a derivative of `lerobot/pi05_base`, whose
PaliGemma backbone is under the Gemma Terms of Use.  The dataset stays separate at CC BY 4.0.
`NOTICE.md` carries both Gemma links, which the terms require to travel with redistribution --
the dataset release's `LICENSE` file had no equivalent obligation, so this is the one place the two
release procedures differ.

Verified after upload by downloading the card, the notice and both eval records and checking them
against `MANIFEST.sha256` (all four OK), and by listing the repository through the API: 13 files,
`model.safetensors` 9.35 GB.

### 9.25 Measured: the policy's own first batch lands 2-10 mm off the slot column, and `prefill`
only ever generates 0.2 mm (10-21)

9.23 argued from episode-level bookkeeping that `prefill` cannot be doing its job.  This measures
it.  The 4.05 checkpoint was driven on main's benchmark, one episode at a time, recording after
every step the pose of each Cookie that the judge counts as inside the target -- in the box frame,
so it is comparable with what `place_cookie_in_slot` generates.  Ten seeds; five of them reached
five Cookies in the target, which is the state `prefill` is supposed to imitate.

| seed | Cookies at first-5 | median offset from nearest slot | spread within the group | worst lean | worst \|yaw\| |
|---|---|---|---|---|---|
| 1 | 5 | 9.49 mm | 0.66 mm | 0.63° | 0.5° |
| 4 | 5 | 2.29 mm | 6.22 mm | 0.37° | 1.6° |
| 5 | 5 | 9.33 mm | 2.92 mm | 0.52° | 0.1° |
| 6 | 5 | 2.04 mm | 9.59 mm | 1.02° | 3.3° |
| 7 | 5 | 2.50 mm | 0.12 mm | 0.72° | 0.1° |

Across the 25 Cookies: median offset **5.03 mm**, quartiles 2.34 / 5.03 / 9.45 mm, and **4 % within
2 mm** of a slot centre.

`prefill`'s generator (`batch_expert._prefill_first_batch`, soft draw) produces:
lean `U(0, 4°)`, yaw `U(±1°)`, per-Cookie jitter **≤ 0.2 mm around the ideal slot centre**.  So the
two distributions differ by **25-50x on the axis that matters**, and they differ in the other
direction on the axes that do not: the policy is *more* upright than the generator ever is
(≤ 1.0° against 0-4°) and its yaw is comparable.

**Zero of five episodes fall inside `prefill`'s soft support.**  That is the covariate shift 9.23
predicted, measured.

Two structural facts make it worse than a scalar mismatch:

* **The offset is coherent, not scattered.**  Where it goes well the five land in a block with
  0.1-0.7 mm of internal spread, displaced as a unit.  That is a *group* error -- the whole batch
  is deposited a few millimetres off the slot column -- and it is the shape the judge rewards with
  5 rather than 10: footprint inside the box, nothing in a slot.  The 20-episode benchmark says the
  same thing independently (score 5 in 10-12 episodes, score 10 in 2).
* **`prefill` fixes the wrong residuals.**  It varies lean, yaw and sub-millimetre jitter around a
  correctly slotted group.  Every one of those is already inside the policy's own error budget; the
  one it does not vary is the millimetre-scale translation that dominates.

So the change `prefill` needs is not a wider lean or a bigger jitter: it is a **group-level
translation drawn from the measured error distribution** -- a per-episode offset of order 0-10 mm
applied to all five slots together, plus a smaller per-Cookie residual -- and, because the policy
also produces 5/10 episodes that never place five at all, the prefill has to remain only part of the
data rather than most of it.

The same measurement says what the *policy* needs, independent of the augmentation: it can already
carry five Cookies as a coherent block and put them in the box; what it cannot do is put that block
where the slots are.  A method that adds millimetre-accurate slot localisation to the second batch
would be worth more than every layout-noise and prefill variant tried so far -- which is consistent
with the 4.05 coming from the Cookie-orientation fix (a *localisation* change) rather than from any
noise-setting change (9.21-9.23).

### 9.26 What the 4.05 handover actually consists of (10-21)

`/root/autodl-tmp/release/handover-pi05-4.05.tar.gz` (152 KB) plus the ModelScope weights.  The
package is code and protocol only, because the weights are 9.35 GB and already published:

| file | why it has to travel |
|---|---|
| `PROMPT.txt` | the one sentence the checkpoint was trained on; with their harness's `"transfer 10 cookies into target box"` it places nothing |
| `code/a3_bridge_server.py` + `code/pi05_a3.py` | the policy-side contract: 16-wide absolute joint actions, chunk 50 / **execute 16**, 224×224 `INTER_AREA` RGB, three named camera slots, state 16 wide and *not* padded, and the determinism switches |
| `code/reference_driver_bench_main_20k.py` | their driver with the two changes; reference only |
| `CHANGES_vs_your_benchmark.md` | the two changes, so they can patch their own tree instead of taking ours |
| `eval/*.json` | the two runs behind 4.05 and 5.00, per episode |
| `HANDOVER.md` | commands, protocol table, determinism, failure breakdown, provenance |

**The benchmark does not need copying.**  Their `benchmark.py` already accepts an external policy
through `worker_policy_spec`, and their CLI already has `--max-steps`, `--seed-start`,
`--no-randomize-boxes` and `--no-randomize-cookies`.  The only two things that are genuinely ours
are (a) a `--policy` route to a socket, which the driver shows how to bypass, and (b) the
instruction override.  Handing over a forked benchmark would have implied a divergence that does
not exist.

Two details in the handover that would silently wreck a reproduction, both stated there and both
worth repeating: **execute 16 of 50 predicted steps** (not all 50), and **fp32 + a fixed sampler
seed** (bf16 without one scored 10/10, 0/10 and 5/10 on the same checkpoint and scene).

The package is `tar.gz` rather than a branch push so that the recipient needs no repository access;
`git bundle` of the same branch (167 commits, 11 MB) is at
`/root/autodl-tmp/release/vla-policy-interface-v2.bundle` if a fetchable form is preferred.

### 9.27 The gate: can the expert finish from what the policy leaves? (10-21)

The question a DAgger-shaped dataset lives or dies on is not "would more data help" but **"is the
state the policy creates recoverable at all"**.  If a second, privileged controller cannot finish
from that state, then collecting 100 grafted episodes collects 100 failures.

The protocol: the 4.05 checkpoint on main's `basic` layout, up to 700 steps of policy until the
first five Cookies are in the target box, then a further `U[100,200]` steps **provided the source
count has not gone down** (the policy must have stopped, not merely paused), then
`A3CookieBatchExpert` in `adopt` mode for up to 700 steps.  `adopt` reads the occupancy, marks the
in-target Cookies as already done, aims the second batch at the free column and **moves nothing** --
moving the policy's block would make the recording a mixture of two controllers, which is exactly
the mixture a trained policy could not parse.

Outcome tiers: **A** 10 in the target *and* five slots gained (recoverable, the demonstration is
usable), **B** 10 in the target with the expert's five not in slots (usable but must be labelled),
**C** the expert could not place five more (unrecoverable).

**Round 1, five seeds, 0/5.**  Every failure was a `CLOSE` timeout with the pads at 8 N and the
opening at **0.000** -- the jaws had closed on air.  The adopted indices said why: seed 0 adopted
`[1,2,3,4,5]`, leaving `[0,6..19]`, and the expert went for `[0,6,7,8,9]`.  `_candidate_batches`
yielded "the first five remaining by y", which after the policy took the middle of a column is a
group spanning a 31 mm hole.

The fix is two conditions on a candidate batch -- **consecutive** indices, and each within
`SOURCE_IN_PLACE_TOLERANCE_M = 1 mm` of its nominal source position, upright and still in the
source -- so a batch is five adjacent Cookies the pad can actually close around.  The selection now
yields `whole[:5]` / `whole[-5:]` of every consecutive run.

*Regression check first, because that function is used by every future collection*: 3 accepted in 4
attempts, `failed shards: 0`, on the production flags (`--align-bin-noise 0.01 --align-noise-dist
uniform --align-bin-yaw-noise 0.03 --position-noise 0.0003 --align-cookie-yaw-noise 0.015
--fast-render --main-aligned`).  With no perturbation a column is one consecutive run, so the
rewrite is a no-op for ordinary collection.

**Round 2, four seeds, 0/4 -- but the failure moved.**  Grasping air disappeared; what remains is
insertion:

| seed | adopted | at takeover (target / slots) | outcome | failure |
|---|---|---|---|---|
| 0 | [1..5] | 5 / 0 | 8 / 3 | `DESCEND_TO_PLACE` timeout, 421 steps, pads 2.55 / 2.50 N |
| 2 | [1..5] | 5 / 5 | 5 / 5 | none; 413 steps, budget spent |
| 7 | [2..6] | 5 / 5 | 5 / 5 | `CLOSE` timeout, pads 11.60 / 11.72 N, opening 0.000 |
| 9 | [1..5] | 5 / 2 | 5 / 2 | `DESCEND` blocked, pads 59.12 / 0.0 N |

The 11.7 N at `opening = 0.000` is a *different* zero-opening than round 1: the jaws are closed on
something.  And 59 N on one pad at `DESCEND` is the arm pressing into a Cookie that is in the way.
Both are consistent with 9.28: the policy's block sits in the target column.

A second sample of the same protocol was recorded as video (`/root/autodl-tmp/logs/videos/gate_v4/`,
seeds 0-3, 700/700 caps, both phases in one file): seed 0 reached **tier A** (expert `DONE`, 10/10
in target, +5 slots, `success`), seed 1 never triggered (0 in target), seeds 2 and 3 timed out in
`CLOSE` at 21.2 / 1418.9 N and 21.8 / 21.8 N.  Same seed is **not** the same episode across runs:
the bridge's sampler state is consumed per request, so a run that follows other episodes in the same
process samples a different trajectory.  The video is a sample of the protocol, not a replay of
`gate_fix.json`.

**What this says.**  Recoverability is real but rare (1 of 4 in the video run; 0 of 4 in round 2),
and the obstruction is the policy's own first batch.  Two consequences for the plan: (a) collecting
grafted episodes now would collect mostly failures, and (b) the useful version of the idea is not
"expert cleans up after the policy" but either an expert that can work around clutter, or a policy
whose block stops landing in the way.  9.28 says which of those two is cheaper.

### 9.28 Where the first batch actually differs: expert vs 4.05 vs 3.75 (10-21)

Two questions got conflated in 9.25: *does the model grasp as well as the expert*, and *does it put
the grasped block where the expert puts it*.  Measuring one rollout step by step answers both:
end-effector pose, 16 joints, the action, all eighty Cookie positions and the per-Cookie containment
verdict are written every step, for the expert and for a model, **from the same initial state**
(`reset(seed)` plus the same `basic` rewrite).

"Picked up" is a **30 mm rise**, not the containment clause `released`: the expert carries five
Cookies in one grasp and raises them together, and the clause is a placement verdict -- on the
expert's own run it marked two of the five it lifted.  A rise is a property of the object, which is
what the comparison is about.

| controller | seed | lift step | tool-to-group | path | block rot | block translation | block RMS | joint RMS at grasp |
|---|---|---|---|---|---|---|---|---|
| expert | 0 | 99 | 0.06 mm | 570.8 mm | +0.61° | 0.3 mm | 0.29 mm | -- |
| expert | 2 | 99 | 0.13 mm | 573.6 mm | +0.99° | 0.5 mm | 0.19 mm | -- |
| 4.05 | 0 | 109 | 1.43 mm | 619.3 mm | +0.18° | **12.0 mm** | 0.14 mm | 0.0198 rad (1.13°) |
| 4.05 | 2 | 108 | 0.64 mm | 604.0 mm | +0.03° | **9.0 mm** | 0.16 mm | 0.0172 rad (0.99°) |
| 4.05 | 1 | 107 | −0.07 mm | 1573.4 mm | -- | -- | -- | -- |
| arm 1 (3.75) | 0,1,2 | **none** | -- | 335 / 351 / 319 mm | -- | -- | -- | -- |

"Block" is the rigid transform taking the five placed Cookies onto the five nearest slots, fitted in
the target bin's own frame; the residual after removing the transform is the block's internal
spread.

**The window has to outlast the release.**  The first recording stopped the step the fifth Cookie
was counted inside (217 steps), and at that step the jaws had not opened yet -- it measured the
block in transit.  Re-recorded to 245 steps, past `OPEN` and the retract, the block is still 12.0 /
9.0 mm off (RMS 0.14 / 0.16 mm, rotation 0.18° / 0.03°).  The offset is where the block comes to
rest, not a frame of the carry, and that is the version this table reports.

1. **Grasping is not the bottleneck.**  At the moment the five leave the packing, the tool is
   0.6-1.4 mm from the centre of the group it is lifting; the expert is 0.1 mm by construction.  The
   path over the same phases is 6-8 % *longer* than the expert's (619 / 604 mm against
   571 / 574 mm) and the joint state at the grasp moment differs by 1.0-1.1°.  Nothing here needs
   fixing.
2. **The difference is one rigid translation of 9-12 mm.**  Both blocks are internally coherent
   (0.1-0.2 mm RMS), both are square to the slot grid (the model's rotation is *smaller* than the
   expert's), and the model's is displaced as a unit.  The tool is 13.4 / 9.5 mm from where the
   expert's was at the moment the first Cookie counts as inside -- while the block is still in the
   jaws -- with 2.8° / 1.8° of joint difference, and the gap survives to rest (12.0 / 9.0 mm).
   Over the whole rollout the two paths stay within 15-17 mm of each other at equal arc length
   (DTW 3.8 / 3.1 m), so this is not a different route: it is the same route ending in a different
   place.  `block_offset_seed0.png` in `logs/first_batch/` draws it: the expert's rings sit on the
   left slot column, the model's dots sit ~12 mm inboard of it.
3. **The 3.75 -> 4.05 step is "grasping at all", not placement accuracy.**  arm 1 never lifted
   anything on any of three seeds: 700 steps, 80 Cookies still in the source, and on seed 0 **30
   Cookies displaced more than 5 mm** with a maximum rise of 1.15 mm -- it was sweeping with the
   jaws, not closing on a group.  That is the same discontinuity 9.23 attributed to the
   Cookie-orientation fix.

Two details that look like errors and are not:

* **The model grasps `[1..5]` where the expert grasps `[0..4]`.**  Both are five adjacent Cookies in
  one column; the model's group is one pitch (8.83 mm) further along.  This is orthogonal to the
  12 mm placement error (which is along the transfer axis), and any five adjacent Cookies can go
  into five slots, so it is not itself a defect.
* **The source pitch is 8.83 mm and the slot pitch is 6.32 mm.**  The expert's placement compresses
  the group by 10 mm over its length and both controllers achieve it (0.29 mm and 0.11 mm RMS
  respectively), so the box's own geometry is doing that work, not the controller.

**What the comparison does not say.**  Two usable (seed, controller) pairs, because the scripted
expert fails on seed 1: `DESCEND`, 421 steps, pads at **0.0 N**, tool parked at 759.44 mm -- the arm
is stopped by the source bin's wall, not by a Cookie.  So seed 1 is not evidence about the model
(4.05 fails it differently: it lifts the group to 860 mm and holds it there for 590 steps).  The
offsets should be read as a direction (rigid, translational, along the transfer axis), not as
constants; 9.0 / 12.0 mm are two samples.

The consequence for the next experiment: a rigid translation is the kind of error an inference-time
setting or a calibration can shift, unlike "the policy is not smooth enough".  That distinction is
worth an hour of measurement before it is worth a night of training.

**Correction, same day -- see 9.29.**  Every number above was measured at the manifest's declared
`execute: 16` of a 50-action chunk.  Executing 48 of the same chunk instead moves the block from
9.8 mm off to 1.4 mm off and fills all five slots.  So the rigid translation is a property of the
**deployment pacing**, not of the checkpoint: the checkpoint's placement is accurate when the
chunk is allowed to finish.  The rest of this section -- grasp accuracy, block rigidity, the
3.75-arm not grasping at all -- is unaffected, because those happen before the final approach.

### 9.29 The ten millimetres were the chunk pacing (10-21)

9.28 left one number unexplained: a block that the model had grasped to 1.4 mm landed 9-12 mm from
the slots.  It was repeatable within a scene (6 seeds on the same layout gave 9.84 ± 0.68 mm) and
varied across scenes, which reads either as a policy that cannot localise the slot column or as a
deployment that never finishes the motion.  Those two have very different fixes, and one variable
separates them: **how many of the 50 returned actions are executed before the next request**.

The manifests declare `chunk: 50, execute: 16`.  `execute` is a deployment choice -- the training
data has no chunks at all -- so it is the one thing that can be changed without touching the
weights.

The yardstick is the **frozen** layout (zero noise): every seed sees the identical scene, so the
only spread left is the sampler's.  At `execute: 16` that spread is tight and the outcome never
changes: 0 of 5 slots in 6 of 6 seeds.  At `execute: 48`:

| execute | seeds | mean |dy| (per Cookie to its slot) | slots filled |
|---|---|---|---|
| 16 | 6 | 9.84 ± 0.68 mm | 0 / 5 in 6 / 6 |
| 48 | 3 | 0.06 / 0.06 / 1.38 mm | **5 / 5 in 3 / 3** |

`dy` is the direction that matters: the slot tolerance is **10 mm in x and 2.5 mm in y**, so a
block displaced along the column fails no matter how well it is centred across it.  The 10 mm was
in y.

**Why it is stable rather than converging.**  Cutting a chunk at 16 of 50 truncates the final
approach.  The next request re-plans from a state that is still ~10 mm short, produces the same
truncated motion, and is cut again at the same point -- so the error holds instead of shrinking.
That is the signature to look for: an offset that is repeatable to 1 mm on an identical scene is
not a perception limit (perception would spread), it is a control loop that is not allowed to
finish.

**What this costs the record.**  The published 4.05 (2 / 20 strict successes on main's randomised
benchmark) was scored at `execute: 16`.  If the pacing is what kept the first batch out of the
slots, then that score is a floor set by our harness, not by the checkpoint, and main reproducing
it with their own pacing would not get the same number.  The next measurement is the benchmark
itself at `execute: 48`, which is what turns 3 seeds into a score.

### 9.30 The pacing fixes the first batch and does not move the score (10-21)

9.29 ended with a prediction: if the truncation is what kept the first batch out of the slots, the
published score is a floor.  The paired benchmark tests it -- same checkpoint, same 20 seeds, same
judge, same 900-step cap, one variable (`execute` of a 50-action chunk):

| execute | mean | distribution | strict successes |
|---|---|---|---|
| 16 (published regime) | 3.85 | {0:6, 1:2, 5:9, 10:3} | 3 / 20 |
| 48 | **4.40** | {0:2, 1:1, 4:1, 5:14, 6:1, 7:1} | **0 / 20** |

The means differ by 0.55 with a paired standard deviation of 3.5 (SE 0.78), so **the score change is
not significant** -- but the shape of the distribution is not noise.  Eight of twenty episodes
improve (0 -> 5 in six of them), four get worse, eight are unchanged.  The four that get worse are
exactly the episodes that had reached ten: 4, 10 and 17 go **10 -> 5**.

So the prediction was half right.  `execute: 48` makes the first batch reliable -- the zero column
nearly empties (6 -> 2) and the mode moves from 5-with-a-tail to a tight 5 -- and it removes every
strict success, because the same long commitment that lets the final approach finish also makes the
*second* batch worse.  The first batch was never the binding constraint; it was hiding the second.

That is consistent with 9.27's gate: with `execute: 48` the takeover state now holds **5 Cookies in
the slots** (it held 0 before) and one of four seeds completes (tier A, 10/10 in target, 9 slots),
while the failures are the expert at ~39.5 N trying to insert into a box that is half full.  The
wall is the second batch, and it is the same wall from both ends -- the policy's second batch and
the expert's recovery both fail there.

**What to carry forward.**  The 10 mm was real and is explained; fixing it did not raise the score,
so it must not be sold as a win on its own.  Two open readings of the 10 -> 5 regressions: either
the second batch needs the shorter re-plan (a 2.4 s commitment is too long for its insertion), or
those three episodes were the ones where a mis-placed first batch accidentally left the box in a
shape the second batch could use.  The per-batch trace at both pacings is what separates them, and
that is the next measurement rather than another training run.

### 9.31 Four pacings: the mean peaks in the middle, the strict successes do not (10-21)

9.30 used two settings.  Four of them, same checkpoint, same 20 seeds, same judge:

| execute | mean | distribution | strict successes | episodes ending at exactly 5 |
|---|---|---|---|---|
| 16 | 3.85 | {0:6, 1:2, 5:9, 10:3} | **3 / 20** | 9 |
| 24 | **4.60** | {0:2, 1:1, 2:1, 5:13, 6:1, 8:1, 10:1} | 0 / 20 | 13 |
| 32 | 4.55 | {0:1, 1:1, 4:1, 5:16, 6:1} | 0 / 20 | 16 |
| 48 | 4.40 | {0:2, 1:1, 4:1, 5:14, 6:1, 7:1} | 0 / 20 | 14 |

Two things are stable and one is not:

* **The zero column collapses as soon as the chunk is allowed to finish**: 6 -> 2 -> 1 -> 2.  This
  is the 9.29 mechanism showing up in the benchmark.
* **The mode is pinned at exactly five at every pacing above 16**: 13, 16 and 14 episodes end there,
  and none of them finish early.  The second batch makes no progress in roughly three episodes out
  of four, whatever the pacing.
* **The mean peak at 24 is not significant**: 4.60 against 4.40 is 0.2 with a paired spread of ~3.5.
  The strict successes are also not significant in isolation (3 of 20 against 0 of 20), but they are
  the only place a 10 appears, so the reading is that short re-planning occasionally rescues a
  second batch that long commitment cannot -- not that 16 is better on average.

So the pacing is a real lever on the *first* batch and a coin-flip on the second.  The next
measurement is the second batch itself: full 900-step traces at `execute: 24` on seeds 0, 4 and 10,
recording every step the way 9.28 did for the first batch.  The question is which of three shapes it
has -- the arm keeps carrying and dropping (transfer), it stops moving (the policy believes it is
done), or it works on Cookies that never count (placement) -- because each points at a different
fix and only the last is a training problem.

### 9.32 The second batch is executable; what fails is the state the first one leaves (10-22)

9.31 left one question: when three quarters of the episodes end at exactly five with the budget
spent, is the policy unable to do a second batch, or is it refusing to?  The probe separates them.

`_prefill_first_batch` teleports five Cookies into the first column's slots (with drawn lean, yaw
and jitter) and leaves the arm at home.  That is a half-done episode with a *clean* first batch and
a *reset* arm pose.  Handing that state to the policy, `execute: 24`, 900-step cap:

| seed | steps | final in target |
|---|---|---|
| 0 | 243 | **10 / 10** |
| 4 | 247 | **10 / 10** |

Both finish in about the same time the first batch alone takes.  So **the second batch is
executable at the same speed as the first** -- the policy is not refusing to work, and it does not
treat five as the goal.

What differs between this probe and a real rollout:

* the box holds the policy's *own* block, 9-12 mm off the slots at `execute: 16`, ~1 mm at 24;
* the arm starts mid-episode, above the box, rather than at home;
* the source column has a hole where that block came from.

The one real rollout that reached a second batch (seed 4, `execute: 24`) showed the arm returning to
the source bin and hovering at **x = 110 mm while the Cookies sit at 99-101 mm** -- 11 mm outboard
-- for 250 steps, rising and descending, without lifting anything.  That is the same
"stops one approach-width short and repeats" signature 9.29 found in the first batch's placement,
now at the second batch's grasp.

**Consequence for training.**  The failure is not a missing capability that more demonstrations of
the clean state would supply -- the policy already solves that.  It is a state-distribution problem:
the policy meets its own half-done state at deployment and not in the demonstration's half-done
state.  The two ways to attack that are on-policy data (collect from the policy's own states, which
is what 9.27's gate was built to enable and could not, because the expert cannot finish from them)
and more exposure to the existing data, which is the only one that can run tonight.

**Tonight's run -- one variable, 4.05 with 4.5x the schedule.**  `a3_b2_aligned_v3_18k`:
same base checkpoint, same dataset, same un-freeze scope, same batch 6, `lr 5e-6 -> 2.5e-7` over
18000 steps (warmup 600).  The reason it is worth a night: at batch 6 the 4000-step run saw
**24,000 samples = 5.6 % of one epoch, about 56 episodes of 1000**.  The aligned arm has been judged
on five per cent of the data.

Checkpoints every 2000 steps, so the morning has a scaling curve rather than one number.

* If the "exactly five" count falls with steps (13 of 20 at 4k) and strict successes appear at 12k
  or 18k, the aligned arm was under-trained and the next night is simply longer.
* If it stays at 13 of 20, then exposure is not the lever, the covariate shift above is, and the
  next step is on-policy collection with an expert that can work from the policy's states -- not
  another training run.

Either way the run answers a question the last four days kept deferring.

### 9.33 4.5x the training, same wall (10-22)

9.32 pre-registered the test: if the "exactly five" count does not fall with more exposure, the
aligned arm is not under-trained and the next move is on-policy data, not another run.  The run
finished: 18,000 steps, 6h05m, final loss 0.002, **254 episodes seen** against the 4,000-step run's
56 -- 4.5x the data exposure, everything else identical (same base, dataset, un-freeze, batch 6,
`lr 5e-6 -> 2.5e-7`).

Same 20 seeds, same judge, `execute: 24`:

| model | steps | episodes seen | mean | strict | ending at exactly 5 | distribution |
|---|---|---|---|---|---|---|
| 4.05 | 4,000 | 56 | 4.60 | 0 / 20 | 13 | {0:2, 1:1, 2:1, 5:13, 6:1, 8:1, 10:1} |
| 18k | 18,000 | 254 | 4.30 | **2 / 20** | **12** | {0:5, 5:12, 6:1, 10:2} |

**The wall did not move.** 13 of 20 episodes still end at exactly five, and the mean is 0.30 lower,
which is well inside the paired spread of ~3.5.  Two episodes now complete (they did not at 4k) and
three more fail outright, so the model is more variable rather than better: the same 4.5x exposure
that produced two completions also produced three more total failures.

That is the answer 9.32 asked for, and it is negative: **the aligned arm is not under-trained.**  A
policy that has seen 56 and one that has seen 254 demonstrations stall at the same place, so the
missing information is not in the demonstrations -- it is the state the policy puts itself in, which
no demonstration contains (9.32 measured that directly: from a clean half-done state the same
policy finishes 10/10 in 243 steps).

Two things follow, and they are not both training problems:

* **Deployment**: the two batches want opposite pacing (9.31) -- 24 for the first, 16 for the
  second, where 3 of 20 complete.  A hybrid that switches after the fifth Cookie is untested and is
  the cheapest remaining shot at a score, because it changes no weights.
* **Data**: the state the policy creates has to be *in* the training set, with a controller that can
  finish from it.  The scripted expert mostly cannot (9.27: 39.5 N against the policy's own block),
  so this needs either an expert that works around clutter or a human/teleop pass over exactly those
  frames -- and it is the real next project, not a night's run.

The 18k checkpoints are kept at 2k, 16k and 18k (`pi05-runs/a3_b2_aligned_v3_18k`); the intermediate
ones were deleted to keep the run inside the disk budget, since this lerobot has no
`save_total_limit`.

### 9.34 Which ingredient of the half-done state blocks the second batch (10-22)

9.32's prefill probe left two candidates: the pose the arm is in at the handover, and the block the
policy itself leaves in the box.  Three probes change exactly one of them at a time, all on the same
checkpoint (4.05), same scene, `execute: 24`, same 900-step cap.

| cell | block in the box | arm at handover | result |
|---|---|---|---|
| prefill (9.32) | expert-shaped, 5 / 5 in slots (teleported) | reset (home) | **10 / 10**, 243 / 247 steps, 2 / 2 |
| C `expert-first` | the expert really grasped and placed it, 5 / 5 in slots | the expert's mid-episode pose | **10 / 10**, 224 / 225 more steps, 2 / 2 |
| B' `block` | the policy's own, **6 in the box but 4 in slots** | reset (home) | **6 -> 6**, 517 further steps, 0 added |
| normal rollout | the policy's own | the policy's own | stalls at 5, 13 / 20 |

The pose is exonerated twice over: the expert's mid-episode pose works (C) and so does the reset pose
(prefill).  What separates success from failure is **the block, and specifically the loose Cookie in
it**: the expert's five land in five slots; the policy's six land with two of them inside the box but
outside any slot, and those occupy exactly the space the second batch needs.

B' is the clean version of that.  Its count timeline moves only at steps 357-362 -- the moment the
block settles, with one Cookie flickering in and out of the box edge -- and then **nothing happens
for the remaining 540 steps**.  Same arm pose as the prefill run that finished in 245.

That also re-reads 9.27's gate: the expert was not failing on clutter in general, it was failing on
the one or two loose Cookies the policy leaves behind, which is why it pushed at 39.5 N with nothing
in its jaws and why it succeeded on the seed where the block happened to land clean.

**Why more training could not have fixed this.**  Every demonstration in the dataset was recorded by
the expert, and the expert's first batch leaves *no* loose Cookie -- by construction, since it places
into slots.  "There is a Cookie in the box but not in a slot" is therefore a state the policy has
never seen, and no amount of exposure to the same 1000 episodes puts it there.  That is a sharper
statement than "covariate shift": the shift is a specific object configuration that the data
generator cannot produce.

Two ways out, and the second is the one the numbers point at:

* make batch one slot properly so the loose Cookie never appears -- the pacing knob does move this
  (0 / 5 slots at `execute: 16`, about 4 / 5 at 24) but 9.31 showed it alone does not lift the score;
* train on states that contain loose Cookies, which needs a generator that produces them.  The
  expert cannot (the gate tried), so this is a scripted perturbation of the expert's own placement
  -- drop one Cookie beside the column before the second batch -- which is cheap to build and is the
  first data-side experiment here that does not depend on the policy's failures being expert-fixable.

### 9.35 The fourth cell had to be built twice, and the first build was wrong (10-22)

9.34's table says the block decides.  Variant A -- the expert-shaped block at the policy's own
handover pose -- is the cell that would turn "the block decides" into "the pose does not", and it
took two sweeps and one discarded conclusion to build.

Three ways the construction lies, each of which produced a number that looked like a result:

1. **The teleport can bury a Cookie.** The first version filled only the *free* slots with the
   Cookies already in the box.  With a dirty block the loose Cookie sits exactly where the block
   goes, so two Cookies ended up in one volume; the solver answered by ejecting them and seed 4
   went from six in the box to one in a single step and terminated.  A block replacement has to
   start from an emptied box.
2. **A handover below five has no pose to keep.**  The arm then sits wherever the policy left it
   after hundreds of idle steps, and phase 2 measures a stuck arm: seed 1 reached ten (`5 -> 5`),
   adding nothing.
3. **A clean block makes the cell a no-op.**  Seed 3 handed over at 226 with five in target and
   five in slots; the replacement rewrote the same five Cookies into the same five slots, and the
   run then stalled at five for another 674 steps -- the normal rollout, not a control.  The saved
   positions say the block really was clean: x = 0.0680, y = 0.0795 / 0.0858 / 0.0922 / 0.0985 /
   0.1049, pitch 6.33 mm (the slot pitch, against the source bin's 8.83), one height to 0.1 mm,
   and no sixth Cookie anywhere in the box.

Only a dirty block leaves the variant something to remove, and that is the rare handover: seeds 0,
1 and 4 of the first sweep never reached five at all.

**What the accepted sample said, and why it cannot be believed yet.**  Seed 6 handed over at 233
with five in target and four in slots.  With the block replaced by five-in-five and the arm left
where the policy put it, the policy added nothing for the remaining 667 steps (`5 -> 5`, 900
steps).  Read alone that refutes 9.34: a clean block did not rescue the run.  But the two cells
were not the same state.  Dumping the first request each cell hands the policy -- same seed,
`prefill` against the same block with the arm parked -- gives: `observation.state` identical to
the last digit, `right_wrist_view` identical, `front_view` 3.32 % of pixels different,
`left_wrist_view` 10.54 % different.  The arm is provably in the same place; the scene is not.
The replaced block is made of whichever five Cookies the policy happened to move, so the source
grid's five gaps sit in different rows than prefill's, and the model is looking at a different bin.

So at this point 9.34's conclusion is *not* refuted, A's failure is unexplained, and the cell has
to be rebuilt so the scene matches prefill exactly: run the policy, then put every Cookie back
where the episode started and lay down prefill's own five.  That is `prefill-after`.

**Two silent bugs in `park_arm_at_home`, both worth the paragraph.**  `env.data.qpos[:] =
home_qpos` restores the *bins* too -- they are free joints -- walking them back to the poses they
held before `apply_scene` randomised them; the probe then reported five in target and *zero* in
slots, which reads as a broken teleport and was a moved box.  And the reference was snapshotted
after `reset` but before `apply_scene`, so restoring the untouched Cookies put them up to ten
millimetres off inside an already-moved source bin (the saved poses show it: the source grid
splits into 0.767 and 0.7685 heights where it should be one).  Both are fixed by snapshotting
after `apply_scene` and copying hinge and slide joints only.  B''s single "0 progress in 517
steps" sample came from the old `park_arm_at_home`, so it is being re-run as well.

**`prefill-after` settles the history question, and sharpens the real one.**  Seed 6's own
rollout hands over at 233 with five in target and four in slots; the scene is then put back to
prefill's construction exactly -- every other Cookie at its start-of-episode pose, prefill's own
five laid into column 1, the arm parked -- and the policy finishes **10 / 10 in 219 further
steps**.  A prior rollout is therefore harmless: the model is stateless and the scene, not the
history, is what phase 2 reads.  That removes 9.34's least comfortable explanation.

What it does not remove is that A's two accepted samples failed.  The first-frame dump already
said the scenes were not matched, and rebuilding the block from *the policy's own five* rather
than prefill's five is the reason.  So variant A as designed cannot separate block from pose: the
identity of the five Cookies and the position of the five gaps in the source grid move together
with the block.

**`tidy` -- and the structural fact underneath the whole section.**  The single-variable version
sends only the in-box, in-no-slot Cookie back to the source, leaving the block and the arm
untouched.  Seed 6: five in target, four in slots -> one Cookie tidied -> **4 / 10, slots 4, and
nothing added for 667 steps**.  The count fell because the tidied Cookie was one of the five.

That sample also exposes an invariant that reorders 9.33-9.35: **a handover with a loose Cookie
always has at most four Cookies in slots.**  Five in the box with five in slots is clean by
definition, so there is never a loose Cookie to remove from a completed first batch.  "The policy
leaves a loose Cookie" and "the policy's first batch is not finished" are the same statement --
the fifth Cookie went into the box but not into a slot.  The stall is therefore not an extra
object blocking a free column; it is a first batch that is one placement short, and "finish the
batch you started" is a task the demonstrations never contain, since every recorded first batch
ends five-in-five.

Which leaves exactly one clean question, and it is the run in flight: a handover at five in five
slots, arm parked at home, block untouched (`block` on a clean handover).  If that finishes, the
pose the policy leaves is the blocker and A's failures are pose failures.  If it does not, the
remaining difference is the source grid's five gaps, which `prefill-after` moves and A does not.

### 9.36 What actually stops the second batch: the source grid, not the box (10-22)

Four probes now hold the same seed (6) and the same handover (233, five in target, four in
slots), and they disagree in exactly one place each:

| what is put back | box | arm | source grid | result |
|---|---|---|---|---|
| everything (`prefill-after`, seed 6) | prefill's five-in-five | home | pristine | **10 / 10**, 219 more steps |
| the box only (`tidy`, seed 6) | policy's, minus the loose Cookie | policy's | as the policy left it | 4 / 10, 0 added |
| the arm only (`block`, seed 3 at 5 / 5) | policy's, clean | home | as the policy left it | 5 / 10, 0 added |
| nothing (`normal`, `pose`) | policy's | policy's | as the policy left it | 5 / 10 |

Restoring the box does not help.  Restoring the arm does not help.  Restoring the whole scene
finishes.  The only thing the two failing cells still share is the **source bin's grid after the
policy's own five grasps**, and the saved poses say what that grid looks like: next to the neat
8.83 mm lattice there are Cookies at 0.1006, 0.0986, 0.0980 against a column at 0.0994 -- nudged
0.8 to 1.4 mm, and one Cookie at 0.1006 / 0.3215 sitting where the lattice has 0.0994 / 0.3215.
With a 7.2 mm Cookie on an 8.83 mm pitch the clearance is 1.63 mm, so a 1.4 mm nudge is very
nearly a contact, and a gripper that approaches on a fixed lattice-relative path will find that
row closed.

That also re-reads 9.33-9.35.  "The policy stalls at five" is not one behaviour with one cause:
the first batch's own grasps rearrange the source, and the sixth pick is attempted on a grid that
no longer exists.  The loose Cookie in the box was a correlate -- it appears in exactly the
episodes that stalled, because a fifth grasp that ends badly also nudges its neighbours.

`regrid` is the test: put back **only** the source grid, leave the box and the arm exactly as the
policy left them.  If that finishes, the second batch's difficulty lives in the source bin and
9.34's block explanation has to be dropped.

**`regrid` closes the decomposition, and closes it the other way.**  Restoring only the source
grid -- box and arm exactly as the policy left them -- gives **5 / 10** as well (seed 3, five in
target and five in slots at handover, 0 added).  So every single-ingredient restore fails:
box only `4 / 10`, arm only `5 / 10`, source only `5 / 10`, and only restoring *everything* to
prefill's construction finishes (`10 / 10`, and it does so after a full policy rollout).

The reading that fits all four is not "one object is in the way".  It is that the policy's margin
at the second batch is thinner than the deviation its own first batch leaves behind, whichever
component you look at: a source Cookie nudged 1.4 mm on a 1.63 mm clearance, a fifth placement
that lands beside a slot instead of in it, an arm parked over the box.  Put back any one of the
three and the other two are still outside the tolerance; put back all three and it runs.

That is a different failure from the one 9.34 described, and it changes what to do about it:

* 9.34's "the demonstrations never contain a loose Cookie" is still true, but a loose Cookie is
  one of three deviations, not the deviation;
* the pacing knob (9.31) moved the deviation and did not move the score, which is what a
  tolerance problem looks like rather than a single-cause one;
* and a handover is only viable if batch one leaves the scene within a millimetre of how the
  demonstrations leave it -- so the experiment worth running is not "teach the policy to recover
  from clutter" but "make batch one end where the demonstrations end", or drop the boundary and
  train one policy for ten Cookies in one episode.

### 9.37 Raising the takeover yield: two fixes and what they cost (10-22)

The grafted dataset is worth collecting only if the expert can finish from what the policy leaves,
and 9.27 measured that at roughly one usable episode in four.  Two things were wrong, and the
first is the more embarrassing.

**The column choice never reached the arm.**  9.27 attributed the 39.5 N jams to the expert aiming
at the column the policy had already filled, and the fix was `_free_slot_column`, which reads the
occupancy and returns the emptiest column.  It set `target_slot_index` -- and `_place_pose`, the
only thing that decides where the batch is carried, reads `self.batch_index`.  `target_slot_index`
is bookkeeping for the report and is overwritten by `_select_cookie` from `_slot_order`; nothing
downstream moves the arm.  So the report said "aimed at column 0" while the arm went to column 1,
which is exactly the jam the fix was written for, and the gate kept failing for a reason that had
already been "fixed".

`target_column` is now the single value `_place_pose` reads, `adopt` sets it from occupancy, and
every other path leaves it `None` so that `batch_index` is still the answer and non-adopted
episodes are byte-identical.  Checked without a policy, by filling a box and reading what the
expert decided:

| block in | target column | free per column | verdict |
|---|---|---|---|
| column 0 | 1 | [0, 5] | ok |
| column 1 | **0** | [5, 0] | ok (previously 1 -- the full column) |
| 3 in 0, 2 in 1 | 1 | [2, 3] | **infeasible** |

**A full column is a precondition, not a preference.**  One grasp carries all five into a single
column, so an episode whose block straddles both columns cannot be completed at all -- and the
common shape is exactly that, five in the box with four in slots.  The gate used to discover it as
a 420-step `DESCEND_TO_PLACE` timeout.  `_adopt_current_state` now counts free slots per column and
fails at takeover with `no column has five free slots (free per column: [2, 3])`, under the phase
name `ADOPT` so it cannot be mistaken for a grasp failure.  For a collection that is the right
outcome: an uncollectable episode costs three steps instead of seven hundred, and the reason is
recorded.

**A row is not the source.**  Round 2's failures were mostly grasp-side -- `CLOSE` timing out with
both pads at 11.7 N and the jaws shut in a gap the policy's picks had left, `DESCEND` blocked at
59 N against a Cookie a neighbour's grasp had nudged.  Neither is evidence about the box, and the
source holds fifteen more rows, so `_fail` now intercepts failures in the grasp phases
(`APPROACH` … `VERIFY_GRASP`), marks that five as unusable, and selects again, up to twice.
Everything else still ends the episode, which is what keeps the recorded reason worth reading.

**The neighbour outside the group is part of the grasp's problem.**  With the column target fixed,
round 4's first usable seed still failed: seed 7 reached a clean takeover (328 steps, five in the
box, five in slots) and stopped inside `DESCEND` at **75.7 N on one pad with the other at 0.0 N**.
One-sided force is not a jam between the jaws -- it is a lateral hit, and the only lateral obstacle a
row grasp has is the Cookie just past either end of the group, because the pads stand *outside* the
five.  `_graspable_in_place` cannot see that Cookie: it is not one of the five, and its own
displacement is what makes it an obstacle -- 1.4 mm of it closes a 1.63 mm clearance.

`_candidate_batches` now ranks runs by what bounds them.  A run ends at the column's edge, at a hole
(that Cookie was taken, so nothing is in the pad's path), or at a Cookie that is present but
displaced -- and only the third kind is dangerous.  Groups are yielded with the safe ends first, so
a retry walks down the ranking instead of re-trying the same geometry at another position.  Checked
without a policy, on a row of twenty with five taken from the middle and one nudged 2 mm:

```
row of 20: 0..19 | in the box: 5..9 | nudged: 12
  0: [0, 1, 2, 3, 4]        upper end is the hole at 5
  1: [15, 16, 17, 18, 19]   taken from the column's end
  2: [20, 21, 22, 23, 24]   second column
  ...
  and [13, 14, 15, 16, 17] — the row that would put a pad against the nudged Cookie — is ranked
  after both.
```

**The other half of the yield is not the expert's.**  Two of round 3's four seeds never produced a
takeover at all: the policy put nothing in the box in 700 steps (`no trigger (0 in target)`), and
seeds 0, 1, 2 and 4 of the earlier probe sweeps did the same.  Those episodes are not collectable at
any expert quality, so a collection campaign has to budget for them: they are the reason the
achievable rate is a product -- the policy finishing a first batch *times* the expert finishing the
second from it -- and only the second factor is what these fixes move.

**The episode was ending in eight steps and nothing said why.**  With the column and the row
ranking both fixed, seed 7 still came back `tier C` -- five in target, five in slots, no progress,
`expert_failed = False`, `expert_error = None`.  The loop had run **8** steps of the expert phase
and stopped without either party reporting anything, which is only possible if the environment
ended it: `terminated` is the safety stop and nothing else, and its reason lives in `info` where the
gate was not looking.  Recorded, it is one line:

```
expert_stopped_by: terminated
expert_safety_reason: joint velocity exceeded safety threshold
```

Printing the first twelve steps of the expert phase, with the worst joint named and the size of the
command step, says where it came from:

```
step+1: phase=APPROACH qvel_max=7.5  at cookie_8_free[3] command_jump=0.6240
step+2: phase=APPROACH qvel_max=28.8 at cookie_7_free[3] command_jump=0.5440
step+3: phase=APPROACH qvel_max=6.9  at cookie_22_free[3] command_jump=0.4640
step+8: phase=APPROACH qvel_max=95.2 at cookie_1_free[3]  command_jump=0.0662
```

`command_jump` falls by exactly 0.08 a step, which is the environment's own joint-step limit: the
expert's very first command is 0.62 away from what the policy last applied, and the biggest single
term is the jaw opening -- `APPROACH` asks for 1.0 while the hand is still holding the release
opening.  So the adopted hand was being **re-opened and sent to the source while it was still inside
the box**, through the block it had just released.  The Cookie velocities are the block being swept,
not an expert that cannot grasp.

`adopt` now starts with a **vertical retreat**: 60 mm up, at the jaw opening the hand already has,
before the phase machine does anything else, with the right gripper held at whatever the policy was
commanding.  Same seed, same takeover:

```
step+1:  qvel_max=1.0  at cookie_9_free[3] command_jump=0.0289
step+7:  qvel_max=21.8 at cookie_7_free[3] command_jump=0.0137
step+9:  qvel_max=0.3  at cookie_9_free[3] command_jump=0.0134
step+12: qvel_max=0.1  at L_SHOULDER_P[0]  command_jump=0.0055
```

The command step is 20x smaller, the arm's own joints stay at 0.1, and the block's settling
(21.8 m/s, one squeezed Cookie releasing) decays instead of tripping the limit.  That is the first
time this seed has reached the expert's second batch at all.

### 9.38 Two more takeover failures, and one of them is structural (10-22)

**A grasp in the jaws is not something to open.**  A policy's second batch sometimes ends with the
five *closed on* and never lifted: the jaws are on the row, the lift did not happen, and the
episode stalls there.  Takeover used to open the jaws, which drags a squeezed row of five across
its neighbours -- a failure the expert caused rather than inherited.  `adopt` now looks first at
what is already held: `_cookies_in_jaws` tests the Cookie centres against the pad-to-pad axis and
the jaw span, and when exactly five are in the jaws and the pads read ≥ 1 N the phase machine
starts at `LIFT` with the batch adopted and the jaws left at their commanded opening.  Checked by
construction -- one expert grasps a row, a *second* expert adopts that state: detects `[0..4]`,
phase `LIFT`, lifts 47 mm in 90 steps, reaches `DESCEND_TO_PLACE`, no failure.  Fewer than five is
deliberately not adopted: `_place_pose` carries the batch as a group whose offset is measured for
five, so placing three with that geometry puts them beside the column.

**The reachable band is the near end of the first two columns, and batch one consumes it.**  With
the column choice and the row ranking both fixed, the gate's seed 7 still could not start a second
batch.  Printing the candidates says why:

```
pristine [27..31] y=292.9 301.8 310.6 319.4 328.2
pristine [40..44] y=231.1 239.9 248.8 257.6 266.4     <- near end, but column 3: IK 7.46 mm short
loose    [15..19] y=363.6 372.4 381.2 390.1 398.9
```

`[20..24]` -- column 1's near end, the batch a *normal* collection takes second -- is not in the
list at all: the policy displaced those five by more than even the loose 1 mm tolerance, by
reaching over them.  Every remaining candidate is either deeper in a column (63 mm out of reach
for a vertical grasp) or in a column the arm cannot reach at all (7.5-10.6 mm).  So the episode has
no reachable five-Cookie row, and that is a property of the *task geometry plus the policy's own
first batch*, not of the expert.

Two consequences:

* the expert's usefulness now has a ceiling that no amount of expert tuning lifts -- the near ends
  of columns 0 and 1 are exactly what the policy's first batch uses and disturbs;
* what would move it is a grasp that does not need five pristine neighbours in a row: two grasps
  of three and two, or one Cookie at a time, which trades steps for reach and for tolerance.

`_select_batch` now asks the solver per candidate (three IK seeds, since the branch it picks is
seed-dependent) and moves to the next row when the approach pose is unreachable, remembering the
failure so the retry ladder does not revisit it.  When nothing is reachable the episode ends in
**two steps** with `no reachable five-Cookie batch available`, instead of three 421-step APPROACH
timeouts -- which is the difference between a cheap "drop this episode" and a wasted eight minutes.

### 9.39 The expert's own first move was destroying the source (10-22)

B-class failures -- batch one done, second batch never starts -- were being read as "the policy left
no reachable row".  Part of that is true and structural (9.38), but a measurement on a recorded
handover says the expert was also damaging the scene itself, exactly as suspected from the video.

One state (seed 7's handover), three adopt starts, counting source Cookies still within 1 mm of
their nominal position after 30 steps:

| start | untouched | delta | what it does |
|---|---|---|---|
| `none` (open the jaws and travel, the historical behaviour) | 70 -> **57** | **-13** | opens to the 5-Cookie insertion width while the hand is still among the Cookies |
| `lift` (vertical retreat, jaws held) | 70 -> 70 | 0 | leaves without touching the row |
| `park` (home configuration) | 70 -> 70 | 0 | leaves without touching the row |

Thirteen of seventy untouched source Cookies displaced by the first thirty steps.  That is the
`SELECT_COOKIE: no exposed, upright five-Cookie batch available` failure being caused by the
controller that reports it: it opens, the row it was about to use is no longer a row, and it
concludes there is no batch.  The video makes the same point in the frame at the handover -- the
panel reads `1 held`, i.e. the policy was holding *one* Cookie -- and the takeover opened on it.

So the order of the two fixes matters: **never open at the takeover pose** comes first, and it is
already what `lift` and `park` do.  What remains after that is geometry, not damage: with `lift`
the expert reaches `DESCEND` and bottoms out **2.9 mm** above a target that is itself 4.7 mm below
the Cookie centres (`_pick_eef.z = 0.7585` against a standing Cookie at 0.7685), and with `park`
the columns it can still reach solve 7.5-10.6 mm short.  Both are separate from the disturbance and
both are the next thing to fix.

### 9.40 The grafted dataset, and 2000 steps on it (10-23)

The gate said the state a policy's own first batch leaves behind is recoverable about half the
time.  This is the run that turns that into data: 200 episodes where phase one is the 4.05 policy
driving from the empty bench until it stops making progress on the source, and phase two is
`A3CookieBatchExpert` in `adopt` mode finishing from exactly that state.  Both phases are one
trajectory in one dataset, which is the whole point -- a pure-expert dataset never visits the state
a policy produces, so it cannot teach the recovery.

**Collection.**  Ten shards, 428 attempts, **200 accepted = 46.7%**, 4.3 hours wall (`collect_graft.py`;
seed sweep interleaved per shard).  Tier split: **167 A** (the expert's five all reached slots),
**33 policy_only** (the policy finished all ten itself), **0 B** -- a box that merely filled without
slotting is refused, so nothing in the dataset teaches shoving.  140,302 frames, 702 per episode,
1.5 GB, action space and |action| statistics within 1% of the v2 dataset (0.926 vs 0.929), same
shadowed rendering (the `--fast-render` runs that failed earlier were the out-of-distribution
appearance, not a broken scene).

**One measurement that changes how every rate here should be read.**  The bridge is **stochastic**:
the same observation, sent twice, returns actions differing by up to **0.086 rad**.  So the earlier
gate's 50% and a later probe's 29% were two samples of one process, not a contradiction, and a
single seed's outcome says nothing.  Ten-episode rows below are indicative, not decisive.

**The arm.**  From 4.05 (`a3_b2_aligned_v3_4000/004000`), 2000 steps, batch 6, cosine 5e-06 ->
2.5e-07, warmup 66, partial un-freeze 18 -- identical to 9.35's arm except the data.  42 minutes.
2000 x 6 = 12000 samples against 140302 frames is **0.086 of an epoch**: about seventeen of the two
hundred episodes, once.

Main's benchmark, same 10 seeds, both checkpoints, this machine:

| row | 4.05 | graft-2000 |
|---|---|---|
| A randomised (deployment) | **4.20** `{5x5, 10, 6, 1, 0x2}` | **5.10** `{10x2, 9, 5x4, 2, 0x2}` |
| B frozen layout | 3.00 `{5x6, 0x4}` | 3.50 `{10, 5x5, 0x4}` |
| main's strict judging | 0/10 | 0/10 |

Per seed (row A), base -> graft: `0: 5->2`, `1: 10->0`, `2: 0->5`, `3: 1->10`, `4: 5->5`,
`5: 0->5`, `6: 6->0`, `7: 5->9`, `8: 5->10`, `9: 5->5`; total **42 -> 51**.

The shape of that is the whole result.  The gains are exactly the seeds that used to stop at five
(3, 7, 8 -> 10, 9, 10) plus two that used to place nothing (2, 5 -> 5), which is the second batch
being finished rather than attempted.  The three regressions are the other half of the same coin:
the policy phase in this data is the model's own imperfect first batch, so 2000 steps of it also
teaches some of that imperfection back -- seed 1 went from a complete ten to nothing.

Strict success stays 0/10 on both, and that is a different axis: main's judge wants upright,
exact slots, no wall contact and a five-step hold, and none of that is what the graft data was
collected to fix.

**Why the graft arm looks slower in a benchmark run.**  It is not the model.  Isolated, the same
scene and 900 steps cost 249 ms/step on 4.05 and 282 ms/step on graft-2000, with inference at 5.8%
and 5.2% of the step.  Within one checkpoint the per-step cost then varies by up to ten times
across episodes (graftB seed 0: 260 ms, seed 9: 1430 ms; baseB seed 4: 749 ms), and it does not
track score.  That is MuJoCo's contact solve getting expensive when the arm grinds the batch into
the box wall and the pile, and the graft checkpoint spends more time in those states because it
actually performs the insertion instead of parking at five.  4.05 has the same tail available; it
visits it less often.

**Next.**  The gain per sample was large because 0.086 epoch is nowhere near fitting, so the cheap
experiment is more steps, not more data: `a3_graft_v1_15000`, 15000 steps = 0.64 epoch = 5.3 hours,
with the evaluation chained to training (motion self-check, then rows A and B on the same seeds) so
the comparison against this arm and 4.05 stays like-for-like.


### 9.41 Where the grafted arms plateau, and what the failure actually is (10-24)

Four arms were trained on the 200-episode graft collection and two derived datasets.  All rows
below are main's `CookieBatchBenchmark`, task 2x5, `randomize_boxes=True randomize_cookies=True`,
ten seeds `0-9`, 900 steps, same instrument every earlier row used.

| arm | data | steps | row A mean | per-seed | strict |
|---|---|---|---|---|---|
| 4.05 | `a3_b2_main_aligned_v2` | -- | 4.20 | 5,10,0,1,5,0,6,5,5,5 | 0/10 |
| graft-2k | graft as recorded | 2 000 | **5.10** | 2,0,5,10,5,5,0,9,10,5 | 0/10 |
| graft-15k | graft as recorded | 15 000 | 4.50 | 5,5,0,0,10,5,5,5,5,5 | 0/10 |
| **mix-9k** | 50 whole + 167 handover-only | 9 000 | **5.80** | 5,5,2,5,6,5,10,10,5,5 | **1/10** |
| mix-15k | same | 15 000 | 2.90 | 6,5,5,5,0,0,5,2,0,1 | 0/10 |
| clean-9k | 167 prefix + 167 handover-only | 9 000 | 3.80 | 5,5,0,0,5,5,2,6,5,5 | 0/10 |

**More steps is not more model.**  On the same data, 2k beat 15k (5.10 against 4.50), and on the
mix 9k beat 15k by a wider margin (5.80 against 2.90, and 15k put three seeds at zero).  The 2k
arm had seen 0.086 of an epoch and the 9k arm 0.59, so both are nowhere near fitting; what the
extra steps buy past the peak is a better imitation of whatever dominates the data, not a better
policy.  `mix-15k`'s distribution (`{5:4, 6, 2, 1, 0:3}`) is the same collapse the unmixed 15k
arm showed, one iteration later.

**The benchmark's own logs already separate the failure modes.**  Every episode records
`cookies_in_source`, so "did it pick the second batch up" is answerable without a video: in
mix-9k, eight of ten seeds took it (`in_source` 67-73 against 80) and only two did not (75).  Of
the eight, two got all ten counted inside the box and one of those passed the strict contract.  So
the bottleneck is neither the grasp nor the approach -- it is what happens after the batch is in
the air.

**The strays sit one layer too high.**  A replay that classifies all eighty Cookies at step 900
(`/tmp/endgame_geometry.py`, same benchmark, same randomisation) puts the untouched five at
`dz = +28.1 ± 0.4 mm` relative to the target bin, in the correct row (`dx = +32..+35` against a
slot row at `+28` with a ±10 mm tolerance), at the correct spacing (the five `dy` values preserve
the 6.3 mm slot pitch), and none of them between the jaws.  A correctly placed Cookie sits at
`dz = +18.5`; the box rim is at `+26`.  So the batch is carried to the right place, in the right
shape, and released **9.6 mm too high** -- above the rim, where nothing counts.

**Why it is released too high, measured on the demonstrations.**  One detector (gripper-opening
jump while the effector is over the box) applied to the collection:

| phase of the demonstration | median release height | n |
|---|---|---|
| policy phase (4.05 attempting batch two) | **0.8468 m** | 537 |
| expert phase | **0.7596 m** | 189 |

The rim is at 0.7755.  The policy releases 71 mm above the rim; the expert descends 16 mm below it,
to slot height.  And the hand *can* get there -- a probe on the mix-9k checkpoint measures its
lowest effector height over the box at 0.7576-0.7598, within 1.7 mm of the expert's 0.7593-0.7596.
The failure is the moment it lets go, not a reach it cannot make.

**The state it leaves is fully recoverable.**  Taking over from exactly that state (same scene,
same seed, `A3CookieBatchExpert` in `adopt`): 2 of 2 triggered seeds finished at **tier A** -- ten
in the target, five slots gained.  So nothing about the scene, the row or the pile is blocking;
only the policy's own release.

**Removing the bad demonstration made it worse, and that is the informative result.**  Since
every recorded episode contains the policy's own failed attempt (2.7 releases per episode, from
0.8468), the `clean` dataset cut the anchor at the handover so that no segment shows a second
batch released from above.  It scored 3.80 against the mix's 5.80, lost both 10/10 seeds, and its
endgame replay cannot pick the second batch up at all (`in_source` 75/80).  What the whole-episode
anchors were providing was not a bad habit to imitate but the **transition** -- finish batch one,
reach for batch two, attempt the insert -- and cutting it left the model with two disconnected
skills and no bridge between them.

So "the data contains its own failures" is not the binding constraint.  The binding constraint is
that the same release height is correct for an empty box (batch one lands fine from 0.8468, the box
is 26 mm deep) and wrong for a half-full one, and the data has to teach *that distinction* rather
than either height on its own.

**Best checkpoint so far: `mix-9k`, A=5.80, strict 1/10** (`pi05-runs/a3_graftmix_15000/checkpoints/009000`,
symlinked as `a3_graftmix_best`).

### 9.42 The instruction is the task, and every number before this was measured on ours (10-25)

The acceptance target is one weight set that scores on **main's** 2x5 and **main's** 2x10.  Attempting
it produced the sharpest measurement of this whole line, and it invalidates a reading I published in
9.40-9.41.

**The measurement.**  `mix-9k`, one scene, one seed, 3200 steps, changing only the sentence:

| instruction sent | 2x10 score | `cookies_in_source` |
|---|---|---|
| our 2x5 sentence ("grasp five yellow blocks together ... fill all ten slots") | **7 / 20** | 73 |
| main's 2x10 sentence ("Transfer 20 cookies into the 2x10 target box, ten per column.") | **0 / 20** | 78 |

Two Cookies moved in 3200 steps under the sentence it has never seen, seven scored under the one it
has.  Same weights, same scene, same budget.

**So the 7/20 I reported earlier was not a 2x10 result.**  It was "a 2x5 checkpoint dropped into a
larger box while being told to do 2x5", which is a transfer observation and not the benchmark.  The
honest 2x10 baseline for a model that has never seen 2x10 data is **0**, and the same is true in the
other direction: every 2x5 number in this document (4.20, 5.10, 5.80, 4.50, 2.90) was measured with
our training sentence, so under main's own 2x5 string they are all 0 as well.  Those numbers remain
valid *as a ranking of our checkpoints on our instruction* -- they are not acceptance numbers, and
were never labelled as such.

**Why it follows from the architecture rather than from a bug.**  pi0.5 here is `n_obs_steps = 1`:
one observation, no history, no recurrent state.  The instruction is therefore the *entire*
specification of which task the frames belong to -- there is no other channel.  A fine-tune on one
sentence is a model of that sentence's task, and "answer the sentence main sends" is a property that
has to be trained, not assumed.

**What main actually sends**, which decided the data design:

| benchmark | instruction | varies per seed? |
|---|---|---|
| 2x5 | "Transfer 10 cookies from source column 1 into the target box in two batches of five." | **no** -- `choose_column("1", seed)` returns 1 for every seed |
| 2x10 | "Transfer 20 cookies from source column 1 into the 2x10 target box. Grasp *n* cookies first, then *10-n* to fill target column 1 with 10. Repeat: grasp *n*, then *10-n* to fill target column 2 with 10." | **yes** -- `randomization_2x10.yaml` says `first_grasp: random`, sampled per episode, so there are ten variants |

The 2x5 correction is therefore one label, and the 2x10 coverage has to include all ten variants:
training on one of them leaves the other nine seeds answering a sentence the model has never read.

**The build.**  `scripts/build_multitask_dataset.py` relabels the 2x5 episodes to main's sentence
and leaves the 2x10 instructions alone (the collector already writes main's plan prompt verbatim),
then aggregates with one global task table.  The second half is not cosmetic: `task_index` is
per-dataset, and `aggregate_datasets` unions the task tables while copying data rows verbatim, so
merging two datasets whose index 0 means different sentences silently retrains one task on the
other's instruction.  The builder rewrites every source to the global table, then verifies by
resolving each row's index back through the column -- 1480 episodes, 853113 frames, 11
instructions, 0 mismatched rows.

**The dataset.**  2x5: 1200 episodes (1000 canonical + 200 grafted), relabelled.  2x10: 280 episodes
collected with main's own expert (`A3Cookie2x10Expert`, plan `[n, 10-n, n, 10-n]`, four batches of
five, 20/20 under its own judge, measured 386.9 s and 1071 steps per episode).  Post-collection the
2x10 set is 287418 frames after quarantining one shard: `shard_00` died in
`EGL.eglMakeCurrent` during teardown with eight processes sharing the card, leaving a parquet with
14 episodes and a metadata file with 12, so it is excluded rather than repaired.

**Still open: the acceptance eval is running.**  30000 steps from 4.05, 0.21 epoch, on the mixed
set, then 2x5 and 2x10 evaluations with **main's own instructions** -- no `--task-prompt`, because
the question is whether this checkpoint answers what the benchmark sends.  Until that lands, the
only defensible statements are the two rows above.

### 9.43 Four arms toward one checkpoint for both benchmarks, and the one that stops after a batch (10-27)

The acceptance target is one weight set that scores 3-4/10 on main's 2x5 and 7-8/20 on its 2x10.
Four arms were trained against it.  All numbers below are on **main's own instructions** and
`prompt_style="short"`, which is one constant sentence per task; the scores are not comparable with
9.40-9.41, which used our training sentence (9.42).

| arm | data | 2x5 | 2x10 | note |
|---|---|---|---|---|
| v1, 15k | mixed, 34% 2x10, plan prompts | -- | 4.50 | peak of three checkpoints |
| v1, 22.5k | " | -- | **5.00** | zeros 1/6 -- the best 2x10 so far |
| v1, 30k | " | 3.38 | 4.33 | more steps is worse |
| v2, 15k | 2x10 weighted to 50% | **5.00** | 3.25 | best 2x5 so far; 4 zeros on 2x10 |
| v4, 15k | 2x10 filtered to the n=5 plan, 52% | 3.50 | 3.75 | moved-out equals scored |
| v5, 15k | v4 plus two progress channels | 3.88 | 3.12 | first arm to move out 15 |

**The step budget is closed off.**  15k -> 4.50, 22.5k -> 5.00, 30k -> 4.33 on the same six seeds,
so the peak sits below 20k and the earlier single-seed 9/20 at 30k was noise.  The reason more steps
cannot help is arithmetic: a 15000-step arm at batch 6 sees 90000 of the set's 853113 frames, 10.5%,
about 156 of 1480 episodes once.  A full epoch would be 142185 steps and 47.8 hours.  The data is
9.5x the training budget, which is why the fix had to be composition and not collection.

**What the failures look like, which is the useful part.**  v4's per-episode `cookies_in_source`
shows the shape exactly:

```
v4 2x10  moved-out = [5, 5, 0, 0, 5, 5, 5, 5]
v4 2x10  scored    = [5, 5, 0, 0, 5, 5, 5, 5]
```

Moved-out equals scored in every episode, so whatever it picks up it places -- no drops, no partial
inserts -- but it moves **one batch of five** on a task that needs four.  Comparing the task pairs:

| | batches needed | batches done | dropped in transit |
|---|---|---|---|
| 2x5 | 2 | 2 (moves 10-11) | yes, the second batch |
| 2x10 | 4 | **1** (moves 5) | **no** |

So batch picking and insertion are solved to the standard this benchmark measures, and the missing
behaviour is "keep going until the task is done".

**The progress-channel arm (v5) is a probe, not a deliverable.**  Two entries were appended to
`observation.state` -- batches delivered / 4 and cookies placed / 20 -- derived from the effector's
y coordinate (the two boxes are 215 mm apart; the rule labels 570 of 570 20-cookie episodes as
exactly four deliveries, against 700/500 episodes of two/one on the 2x5 side, the latter being the
302-frame episodes that genuinely contain one batch).  The gripper channel was rejected first and
the rejection is measured: the jaw sits at 0.31 for most of an episode and only reaches 0.49-0.58 on
release, so 0.62 found three deliveries of four and 0.45 found a median of six.

The arm moved the number that matters: **v4 moved out 5/5/0/0/5/5/5/5, v5 moved out
5/0/5/0/15/5/6/5.**  Fifteen Cookies is the first time any arm exceeded one batch, which is direct
evidence that the history channel carries information the frame alone does not.  But the mean fell
(3.75 -> 3.12) and the spread widened (zeros 2 -> 3), and one episode that moved 15 scored 0.

Two reasons it cannot be shipped.  First, the channel is **privileged**: "cookies placed" is a fact
about the world, not something the observations contain, so it is legitimate as a probe of whether
history helps and not as an input at acceptance time.  Second, the attribution is **not clean**,
because pi0.5 gives the state two paths and this arm moved both:

* the **text path** -- the state is discretised into 256 bins and written into the prompt
  (`Task: ..., State: 128 45 ...;Action:`), so 18 numbers appear where 16 did, which the language
  tower has to re-adapt to;
* the **learned projection** -- this launcher adds `Linear(32 -> 2048)` as one extra prefix token
  (pi0.5 ships without a state projector), and the two new entries sit on input dimensions 16-17,
  which were **identically zero in every earlier arm**.  Measured on the checkpoint, `|dW|` over
  15000 steps is 0.00033 on those dimensions against 0.00028-0.00043 for the original sixteen and
  exactly 0.0 for the zero-padded dimensions 18-31: the new columns did receive gradient, but from a
  `std=0.02` random initialisation and only to 1.7% of it.  So the scalars reach the model mostly
  through the prompt, and what v5 measured is "more numbers in the prompt" as much as "the model can
  read progress".

**Where that leaves the architecture question.**  The evidence supports the premise of a recurrent
state and does not yet show that two scalars are enough.  What the scalars encode is *how many*
batches landed; what the failures need is also *how well* they landed -- an episode that moves 15 and
scores 0 is one where the count was right and the placement was not.  That is the covariate-shift
half of the problem, and a scalar cannot express it.

**Best checkpoints, by benchmark:** 2x5 `a3_multitask_v2_15000/015000` at **5.00/10** (acceptance
needs 3-4); 2x10 `a3_multitask_15000/022500` at **5.00/20** (acceptance needs 7-8).  No single
checkpoint is at both.



### 9.44 A newly initialised layer at 3e-6 is a layer that is frozen with extra steps (10-27)

v6 added a recurrent state -- a 128-wide latent of the robot's own last ten frames of `[q, q_dot,
a_prev]`, with an auxiliary head that predicts progress and a KL term toward `N(0, I)` (the design
note is at the top of `scripts/multitask_v6_pipeline.sh`).  The reason for the design is that the
encoder sees **no images**: the bottleneck is the input modality, so it cannot represent "the
Cookies look messy", only "where the arm has been and what it did", which is where task progress
lives.

At 7500 steps the check said the module had learned nothing:

```
$a3_multitask_v6_15000/007500
MSE 0.29368   Var(target) 0.29705   R^2 +0.011
corr(batches) 0.360   latent |mu| 0.1132   per-dim std 0.0055
projection token std 1.3261   distinct predicted batch values: 68
VERDICT: the latent does not carry progress
```

An R^2 of 0.011 against a two-dimensional target means the head is emitting roughly the batch mean.
The same checkpoint's training log agreed from the other side: `progress` hovered near 0.33, which
is the target's own mean, and `z|mu|` sat at 0.024 from step 1 to step 7500 without moving.

**The input was fine, and so was the initialisation.**  Reading the actual windows out of the
dataset, `std = 0.8222` over all 480 channels, no dead columns.  Training the same encoder *alone*
on the same data reached `R^2 = +0.967` from a `std=0.02` init and `+0.974` from kaiming in 500
steps, so neither the architecture nor the init explains it.

**The learning rate did.**  Three runs of the standalone recipe, identical except for the step size,
each 1500 steps at batch 6:

| encoder lr | R^2 vs progress | latent per-dim std |
|---|---|---|
| 3e-06 | **-0.010** | 0.00548 |
| 1e-04 | **+0.954** | 0.0783 |
| 1e-03 | +0.940 | 0.0400 |

The 3e-06 row reproduces the real run's numbers almost digit for digit (0.011 / 0.0055).  That is
not a coincidence to explain away: `make_optimizer_and_scheduler` feeds `policy.parameters()` to a
single AdamW at `--policy.optimizer_lr`, and that flag is **3e-06**, because it is the right step
size for nudging a fine-tuned 3B model.  The new module -- 630018 parameters over 14 tensors -- was
handed the same 3e-06.  Fresh weights have to travel much further than fine-tuned ones, and at 3e-06
they do not travel at all.

**This is not new, it is 9.43 in hindsight.**  The v5 arm added two progress entries and measured
`|dW| = 0.00033` against a `std = 0.02` initialisation after 15000 steps -- **1.7% of the initial
weight scale**.  That was read at the time as "the new columns received gradient but weakly, so the
scalars reach the model mostly through the prompt".  The correct reading is that the projection was
also being trained at 3e-06 and could not have moved.  **Every arm in this line -- v1's plan tokens,
v4's simplified prompts, v5's scalars, v6's latent -- added a new module and trained it at the
fine-tuning learning rate**, so each of them measured "a frozen random projection plus whatever the
prompt path could carry".  The arms that improved did so through the text path.

**The fix** is a separate parameter group, in `train_pi05_launcher.py`:

* `_patch_optimizer_param_groups` splits `policy.named_parameters()` on `.history_encoder.` and
  `.history_proj.`, builds `cfg.optimizer` on the inherited parameters so `param_groups[0]["lr"]`
  keeps meaning what the trainer logs and the scheduler steps, then **appends** the fresh group at
  `A3_NEW_MODULE_LR`;
* because `LRScheduler`'s constructor takes an initial step, `param_groups[*]["lr"]` already holds
  `peak / (warmup + 1)` by the time `build` returns, so the peaks have to be **recorded before**
  `build` is called and restated afterwards -- see the next paragraph for what happens otherwise;
* it patches both `lerobot.optim.factory.make_optimizer_and_scheduler` and the module attribute in
  `lerobot.scripts.lerobot_train`, because the trainer did `from ... import` and rebinding only the
  first leaves the second pointing at the original.

Confirmed at startup, which is the line to look for before trusting any new-module arm:

```
new-module lr       : 1e-04 on 630018 params (14 tensors: history encoder + projection);
                      3090175760 on the fine-tuning schedule; base_lrs=['3.0e-06', '1.0e-04']
```

The `base_lrs=[...]` tail is the part that matters and it is only there because the first version
of this patch got it wrong.  That version restated `base_lrs` from `param_groups` **after**
`build`, which reads back the already-warmup-stepped values: the peaks became `6.0e-09` and
`2.0e-07` instead of `3.0e-06` and `1.0e-04`, both divided by the warmup length of 500.  The
checkpoint at 1500 steps that read `R^2 = -0.311` was therefore not an experiment about
`1e-04` at all -- **both** groups, the fine-tuning one included, ran 500x slow, so it was a
slightly randomised 3.05.  A separate bug hid this for a while: the diagnostic read
`optimizer.param_groups` from a list captured in the factory, and `accelerator.prepare` replaces
that list, so the captured reference kept printing the values from `_initial_step` forever.  It
looked exactly like a scheduler that never advances -- same optimiser object, different list,
verified with `is`.  The log now reads through `scheduler.optimizer.param_groups`.

With the peaks recorded correctly, the schedule is right at every point that was checked:
`['6.0e-09', '2.0e-07']` at step 1, `['3.0e-07', '1.0e-05']` at 50, `['1.5e-06', '5.0e-05']` at
250, and `['3.0e-06', '1.0e-04']` at 500 -- the two peaks, in the ratio that was asked for.
The module responded within the warmup: `mu_std` 0.0310 -> 0.0314 -> 0.0473 -> **0.1091**, the
progress head's bias 0 -> -1.95e-04 -> -4.63e-03 -> **-1.01e-02**, `kl` 0.0007 -> **0.0060**, and
the weight fingerprint of one GRU matrix 0.01611 -> **0.01621**.  Nothing about that is visible
at 3e-06, where the same quantities are flat to their printed precision for 7500 steps.

**Two false alarms on the way to that conclusion, both worth remembering.**  The first: the new
module looked frozen even after the learning rate was fixed, because the log printed
`pred=[0.006, 0.003]` identically at steps 1, 50 and 500.  Three decimals is coarse enough to hide
a real drift -- `0.0060 -> 0.0064` prints the same at 3 dp -- so the log now prints four, and
`mu_std` beside it.

The second is sharper.  Reading `p.grad` from inside the forward hook reported `gn=0.00e+00`, which
looks like a module that was concatenated into the sequence but never differentiated.  It is
not: the trainer calls `optimizer.zero_grad(set_to_none=True)` **before** the forward pass, so the
gradient buffers are empty at exactly the moment the log runs, and a perfectly trained module is
indistinguishable from a dead one.  Capturing the norms with `parameter.register_hook` during
backward gives the real answer -- `gn=1.91e+00` over 12 of 14 tensors on the same model, the two
silent ones being biases of gates that never fired -- so the graph was connected the whole time.
Instrument placement is a correctness question here, not a style one.

**The rule.**  A module created by the launcher is a module being trained from scratch, and its
learning rate is a property of *that module*, not of the run.  3e-06 is not a small learning rate for
a 3B fine-tune, it is a small learning rate for a 3B fine-tune; applied to a random layer it is a
freeze with extra steps, and the failure is silent -- no NaN, no divergence, a loss that looks
plausible because the flow term is doing all the work.  Two diagnostics catch it cheaply: the
auxiliary head's R^2 against its own label (`scripts/check_history_latent.py`, target > 0.5), and
`|dW|` on the new module's tensors compared with its initialisation scale.

**A third false alarm, the expensive one: the learning rate was measured from a stale list.**  For
most of a day the encoder was believed to be running at `9.1e-06` when it had been asked for
`1e-04`, because the diagnostic printed `optimizer.param_groups` from a reference captured inside
`make_optimizer_and_scheduler`.  `accelerator.prepare` swaps that list for a new one, so the captured
reference kept reporting the values `LRScheduler.__init__` had written during `_initial_step` and
never changed again.  The tell was that `scheduler.last_epoch` advanced normally (0, 49, 249, 499)
while the printed lr did not, and that `scheduler.get_lr()` disagreed with `param_groups` -- a real
scheduler cannot produce two different answers for the same step.  `is` settled it: same optimiser
object, different list.  Reading through `scheduler.optimizer.param_groups` is the only version that
follows what the trainer actually steps.  **A diagnostic that holds a reference is not a diagnostic
of the thing; it is a diagnostic of the reference.**

**The effect at the checkpoint, same data and same probe as the 7500-step measurement above:**

| | 7500 steps @ 3e-06 | 1500 steps @ 1e-04 |
|---|---|---|
| R^2 vs progress | **+0.011** | **+0.295** |
| corr(batches) | 0.360 | 0.551 |
| latent per-dim std | 0.0055 | **0.0638** |
| latent \|mu\| mean | 0.1132 | 0.0968 |
| projection token std | 1.3261 | 2.4102 |
| distinct predicted values | 68 | 1522 |

Three things are worth reading off that table.  The latent's spread grew **11.6x** while its mean
magnitude stayed the same, which is the signature of a representation using its dimensions rather
than emitting a constant -- the KL term pins the mean, and what changed is what the dimensions carry.
The projection token is now **2.41** against image tokens at ~4.2 and language at ~9.6, i.e. inside
the range attention can see, where the broken runs had it at 0.34 and 1.33.  And 1522 distinct
predicted values against a 2-D target means the head is reading the latent continuously rather than
saturating near a per-batch mean.

**The 0.954 above is not the comparison it looks like, and the correction matters more than the
number.**  That figure came from training the encoder on the **whole 40k-frame sample at once** --
full-batch gradient descent, i.e. one optimiser step per pass over the data.  The VLA trains at
batch 6 and takes one step per six frames.  Reproducing the standalone recipe at batch 6 over 3000
steps (the same number of optimiser steps as the joint run, and 18000 frames seen against 38400)
gives **R^2 = +0.439**, not 0.954.  So the joint run's 0.310 at 3000 steps is 70% of what the same
encoder reaches with no VLA at all, and the remaining 0.13 is the flow gradient competing for the
same weights.  Nothing about that requires an architecture change; it requires the same thing the
whole line needs, which is more samples per optimiser step than batch 6 provides.

Which raises the obvious alternative explanation -- that the KL term is the ceiling, since
`scripts/check_history_latent.py` shows the latent's spread *falling* between 1500 and 3000 steps
(0.0638 -> 0.0489) while R^2 stalls.  `scripts/probe_kl_ceiling.py` settles it by training the
encoder alone at batch 6 for 3000 steps under the joint run's exact objective:

| kl_weight | R^2 | latent mu_std |
|---|---|---|
| 0.05 (the run's setting) | **+0.439** | 0.2562 |
| 0.00 | **+0.434** | 0.5887 |

**KL is not the limiter.**  Removing it entirely buys 0.005 of R^2 and quadruples the latent's
scale -- it is doing its job (bounding the channel, which is what makes the token readable at
inference) at essentially no cost in information.  What looked like a plateau at 3000 steps is the
batch-6 noise in the trajectory, not the regulariser (the 7500-step row below settles that), so the
fix was not to re-tune weights: it was to let the run reach the samples it needs, which is what the
remaining 12000 steps do.  The ablation's value is the refutation, not the diagnosis -- it cost
twenty minutes and removed a weight-tuning branch from the plan.

**The 7500-step verdict, at the same step count as the 0.011 measurement it is compared with:**

| | 7500 @ 3e-06 | 1500 @ 1e-04 | 3000 | **7500** |
|---|---|---|---|---|
| R^2 vs progress | +0.011 | +0.295 | +0.310 | **+0.573** |
| corr(batches) | 0.360 | 0.551 | 0.573 | **0.757** |
| latent per-dim std | 0.0055 | 0.0638 | 0.0489 | **0.0759** |
| projection token std | 1.3261 | 2.4102 | 2.3778 | **3.9531** |

The curve was not plateauing after all.  1500 -> 3000 moved only 0.015, which is what prompted the
KL ablation above, but 3000 -> 7500 moved **0.263**, and the latent's spread recovered from the
dip (0.0489 -> 0.0759) instead of being crushed.  The lesson is that a single 1500-step interval is
not enough evidence to call a plateau: at batch 6 the R^2 trajectory is dominated by which batches
happened to be seen, and the probe reads the training set.

The projection token at **3.95** is the number to watch when reading the benchmark numbers that
come next.  Image tokens enter the prefix at ~4.2 and language at ~9.6, so the history token is now
a first-class citizen of the sequence rather than a faint one -- compare 0.34 in the first broken
run and 1.33 in the second, where attention had the least possible reason to use it.

**Status.**  v6 reached R^2 = +0.573 at 7500 steps, past the 0.5 bar that makes a history channel
worth having, so it earns a 2x5/2x10 benchmark.  Training continues to 15000; the benchmark runs on
the final checkpoint, and 7500 is kept as the mid-run reference point.

### 9.45 Every arm's benchmark was run without the module its training created (10-27)

Found while preparing v6's evaluation, and it changes how 9.43's table should be read.

`PI05Policy.from_pretrained` cannot build `state_proj` or `history_encoder`, because pi0.5 does not
have them.  It reacts to the first in a way that is very easy to miss:

```python
if key.startswith("state_proj."):
    logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
```

One warning, scrolling past inside a four-minute model load.  And nothing on the evaluation side
re-installed the launcher's patches: `multitask_v5_pipeline.sh` exports `A3_STATE_PROJECTOR=1` as a
prefix to the **training** command, and `env A=B cmd` applies to that command only -- the bridge
server started on a later line inherits nothing.

**So the state projector has never been active at evaluation.**  Every number in 9.43 -- 4.50, 5.00,
3.38, 3.25, 3.50, 3.88, 3.12 -- came from a plain pi0.5 whose proprioception reached the model
through the prompt bins alone.  That is consistent with what 9.43 concluded ("the scalars reach the
model mostly through the prompt") but for a stronger reason than it gave: the projection was not
merely untrained and weak, it was **absent from the forward pass**.  The `|dW| = 1.7%` measurement
in 9.43 was reading a tensor in the checkpoint file, not a tensor in any model that produced a
score.

**For v6 this would not have produced an error, it would have produced a number.**  An unpatched
forward pass ignores `observation.history` completely -- the adapter dutifully packs the 480-float
window, the server ships it, and nothing reads it.  A v6 benchmark run before this fix would have
reported a score for an arm that was never run, and the honest reading would have been "the history
channel does not help", which is exactly the conclusion the whole arm exists to test.  **A silently
dropped input is more dangerous than a crash, because a crash cannot become a result.**

**The fix** is `scripts/apply_launcher_patches.py`, called from the adapter before `from_pretrained`:

* it reads the checkpoint's own key list, so a projector is created only when `model.state_proj.*`
  is present, and a stock `lerobot/pi05_base` is untouched;
* the history encoder's `width` and `per_step` are read off the stored GRU (`from_hidden.weight` is
  `(3*width, width)`, `to_gates.weight` is `(3*width, per_step)`), so a mismatch fails loudly rather
  than loading into the wrong shape;
* the number of steps is the one thing the weights cannot record -- the recurrence is
  length-agnostic -- so it comes from `history_dims` and is divided by `per_step` and checked; and
* `A3_HISTORY_SAMPLE=0`, because inference wants the mean: the bridge is already stochastic through
  the flow-matching noise, and a second source of randomness would make one seed unrepeatable.

Verified on the 7500-step checkpoint: `state_proj`, `history_encoder` and `history_proj` all exist,
the load reports "All keys loaded successfully", and `embed_prefix` returns **776 -> 778** tokens
with the history token at `|.| = 0.68`.

**What this does not retroactively change.**  The arms in 9.43 were all measured the same way, so
their relative ordering still holds and the conclusions about step budget, data composition and the
one-batch stop (9.43) rest on behaviour that does not involve the projector.  What it does change is
that the projector was never tested at all: v5's two progress scalars reached the model through a
wider prompt and nothing else, so "the projection could not move because it was trained at 3e-06"
(9.44) is only half the story -- it also was not there.

### 9.46 What the v6 failures actually are, read off the traces (10-27)

9.45 installed the launcher's modules before `from_pretrained`, which made the first honest v6
benchmark possible.  Full results, main's instructions, `prompt_style="short"`, 8 seeds:

| | 900 steps | 1000 steps (main's default) | acceptance |
|---|---|---|---|
| 2x5 | 5.00/10 | **4.50/10** | 3-4 |
| 2x10 | 4.50/20 | — | 7-8 |

**Adding the 100 steps changed nothing that mattered.**  Not one 2x5 failure converted: the same one
seed completed (616 steps), the same seven ran out of budget.  Budget is therefore not the variable,
which is worth having established rather than assumed -- the only successful seed used 809 of its 900
steps, so "900 is tight" was a reasonable hypothesis and is now a rejected one.

**The traces say the 2x5 failures are three different things, not one.**  `--diagnostic-dir` records
the scorer's own `cookies_in_source_mask` / `cookies_in_target_mask` per step, so "moved but not
scored" splits cleanly, and the distance from each unaccounted cookie to the effector at the last
step says whether it is still in the jaw:

| seed | in box | not in either | distance to effector | what it is |
|---|---|---|---|---|
| 0 | 5 | 0 | — | moved one batch, placed it, **stopped** |
| 1 | 5 | 5 | 68-71 mm | **froze with the second batch in the jaw** |
| 2 | 0 | 15 | 15-76 mm | **three batches in the jaw, none released** |
| 3 | 0 | 5 | 256-280 mm | **dropped them 270 mm from the hand** |
| 4 | 5 | 0 | — | moved one batch, placed it, **stopped** |
| 5 | 6 | 4 | 72-73 mm | froze with the second batch in the jaw |
| 6 | 5 | 5 | 66-68 mm | froze with the second batch in the jaw |
| 7 | 10 | 0 | — | **completed** |

The 68-72 mm cluster is the jaw: three seeds are holding a batch at the final step.  Seed 2 holds
fifteen, which means it never released at all after the first success.  Seed 3 is the only genuine
drop.  And two seeds (0, 4) did exactly what 9.43 described -- one batch, placed, then nothing.

**"Froze" is not the policy output collapsing, and the distinction is load-bearing.**  The commanded
actions have a standard deviation of ~1.0 rad over the last 200 steps, which looks like motion; but
the *range* they span is 0.012-0.036 rad per joint and the effector travels **67-90 mm** in those 200
steps, against 409-568 mm for the seeds that were still working.  The model is emitting a stable
"hold this pose" command and the arm obeys it.  So the failure is a **decided stop**, not a policy
that ran out of signal or a controller that stopped tracking -- and a decided stop is a behaviour
that data can teach, whereas a collapsing policy is not.

**2x10 fails differently again, and the trace rules out the obvious explanations.**  The expert
demonstrations in the dataset are a median 1067 and at most 1095 frames, so 3200 steps is three
times what the task needs -- this is not a budget problem, and it is not a placement problem either:

| seed | in box | unaccounted | distance to effector | what it is |
|---|---|---|---|---|
| 0 | 5 | 0 | — | placed batch 1 at step 365, grabbed 3 more at 737-743, **put them back**, then toured the scene for 2400 steps without touching a cookie |
| 2 | 7 | 8 | 259-308 mm | dropped 8, kept going |
| 3 | 5 | 6 | 341-377 mm | dropped 6, kept going |

Seed 0's timeline is the important one.  After returning the three cookies it picked up, the arm keeps
moving -- about 800 mm of effector travel per 400 steps for the entire remainder -- between the source
and the target without ever grasping again.  It is not frozen and not lost; it is **repeatedly
approaching and not committing**.  That is a different mechanism from 2x5's decided stop, and it is
the one the acceptance target actually needs fixed: 2x5 is at 4.50 against a 3-4 requirement, while
2x10 is at 4.50 against 7-8.

**What this changes about where to work next.**  The three 2x5 modes need three different fixes:
"stopped after one batch" and "froze holding the second" are the same underlying question (does the
instruction imply continuing?) but different failure points, and "dropped at 270 mm" is a separate
grasp-robustness issue.  The 2x10 mode is not a stop at all, so more steps of the same data will not
touch it -- it needs episodes where the approach leads to a grasp, which is what a DAgger pass on the
2x10 seeds would supply and what another 15000 steps of the current mix would not.

### 9.47 What the score actually measures, and how much of it is seed (10-27)

Two questions came out of 9.46's numbers, and both have answers that change how those numbers
should be read.

**The score does not measure slot alignment, on either task.**  `require_exact_slots` is set by
`benchmark_cookie_batch_cli.py:40` as `benchmark.task == "2x5"`, so only 2x5 declares
`success_criterion="exact_slots"` and 2x10 declares `"released_count"`.  But that flag only decides
what counts as a *finished* episode and therefore which `success` boolean is set -- the score is
`cookies_in_target` in both cases, with no alignment term anywhere.  Measured directly on the
traces:

| 2x5 seed | score | cookies_in_target | occupied slots | exact_fill |
|---|---|---|---|---|
| 0 | **5** | 5 | **0 / 10** | False |
| 1 | 5 | 5 | 5 / 10 | False |
| 4 | 5 | 5 | 4 / 10 | False |
| 6 | 5 | 5 | 5 / 10 | False |
| 7 | 10 | 10 | 10 / 10 | **True** |

Seed 0 is the case that settles it: **five cookies are inside the target box and not one of them is
in a slot.**  So "5/10" never meant "five inserted correctly" -- it meant "five are inside the
footprint", and the 2x5 task is not stricter than the 2x10 task for the purpose of scoring.  What the
alignment requirement costs is only the difference between `success=True` and a high score, which is
why 2x5's one completed seed is the only one at 10.

The consequence for the acceptance target is favourable and worth stating plainly: **"2x5 reaches
3-4" and "2x10 reaches 7-8" are both counts of cookies inside the box.**  v6's 2x5 = 4.50 meets its
requirement on that basis, and the alignment machinery is not standing in the way of a better number
-- the cookies are not getting into the box in the first place.

**The seed matters far more than is comfortable, and the inference noise matters about as much.**
Score distributions, one checkpoint, 8 seeds each:

| | scores | mean | sd | CV | completed |
|---|---|---|---|---|---|
| 2x5 | 5, 5, 0, 0, 5, 6, 5, 10 | 4.50 | 3.25 | **72%** | 1/8 |
| 2x10 | 5, 5, 0, 10, 5, 0, 6, 5 | 4.50 | 3.25 | **72%** | 0/8 |

A coefficient of variation of 72% means a single seed says nothing.  But the scene difference across
seeds is small: measuring the bin positions the benchmark reports in each episode, the source bin
spans **17.1 x 14.3 mm** and the target bin **18.7 x 18.6 mm**.  About 1.5 cm of scene variation
produces a 0-to-10 spread.

**And the same seed is not repeatable either.**  The 900-step and 1000-step sweeps are two
independent runs of the same checkpoint at the same seeds, differing only in budget:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 900 steps | 0 | 5 | 5 | 1 | 5 | 5 | 5 | 10 |
| 1000 steps | 5 | 5 | 0 | 0 | 5 | 6 | 5 | 10 |
| change | **+5** | 0 | **-5** | -1 | 0 | +1 | 0 | 0 |

Four of eight change, two of them by the full width of a single batch.  So the spread is not a scene
sweep -- it is largely flow-matching sampling noise on top of it, which is the same effect 9.x hit
once already (one checkpoint, one frozen scene, same sampler seed, 5/10 then 10/10).

**What that does to the error bars.**  The paired differences have a standard deviation of 2.6, so a
single evaluation carries about 1.8 of noise; over 8 seeds the mean's standard error is roughly 0.6.
The two sweeps agreeing on **4.50 exactly** is consistent with that and is the useful part: the mean
is more reproducible than any of its terms.  Against the acceptance targets:

* 2x5 = 4.50 +/- 0.6 against a 3-4 requirement -- **passes**, and the margin is 2.5 standard errors;
* 2x10 = 4.50 +/- 0.6 against a 7-8 requirement -- **fails by 4-5 standard errors**, which is why
  that gap is worth acting on and the 2x5 gap is not.

**On "training across seeds".**  The v6 dataset was already built that way: `collect_graft.py` applies
main's `basic` randomization with `rng=np.random.default_rng(seed)` per episode -- `bin_noise_m=0.01`,
`cookie_noise_m=0.0003`, `bin_yaw_noise_rad=0.03` -- which is the same ±1 cm the benchmark draws from.
So the training distribution covers the test distribution, and the brittleness above is not a
coverage gap that more seeds would close.  Two things would: **denser coverage of that 1.5 cm** (more
episodes, same range) so the model stops interpolating between sparsely-seen arrangements, or a
**wider range** than the benchmark uses, which forces the model to look at the image instead of
recalling a trajectory.  Neither is "retrain with different seeds" -- the seeds are already different
in every episode.

**One measurement change this implies.**  Any future arm's number needs `n >= 16` and a stated
standard error before it can be compared with another arm: at n=8 and 1.8 of per-episode noise, the
resolution is about +/- 1.3, which cannot separate v4's 3.75 from v5's 3.12 in 9.43 -- and those two
were treated as different at the time.

### 9.48 The data disagrees with its own instruction, and that is the "stop" (10-27)

9.46 left three 2x5 failure modes and one 2x10 mode with no explanation for *why* a model that
places whatever it picks up would stop.  The explanation is in the dataset, and it is arithmetic.

**41.7% of the 2x5 episodes end after one batch, under an instruction that asks for ten cookies.**
Every 2x5 episode carries the same sentence (`transfer 10 cookies into target box`), and the lengths
split into two populations with nothing in between:

| frames | episodes | batches delivered at the last frame |
|---|---|---|
| 302-303 | **500** (41.7%) | **1** |
| 506-1176 | 700 (58.3%) | **2** |

The batch count is not inferred from the length -- it is read out of `observation.progress`, which
`add_history_window.py` wrote from the effector's trajectory: every one of the 500 short episodes
ends at exactly `batches = 1.0`, and every one of the 700 long ones at exactly `2.0`.  So the model
was shown, 500 times, a demonstration that says "this sentence, when the cookies are in the box,
means stop" and 700 times one that says "keep going".  **Nine of the eight 2x5 seeds landing on one
or two batches is that ratio coming back out.**

**The 2x10 side has no such defect, which makes it worse rather than better.**  All 570 episodes are
1043-1095 frames and end at `batches = 4.0`, exactly complete.  Yet v6 scored 4.50/20 and stopped at
five cookies.  The most economical explanation is cross-task transfer of the prior the 2x5 half
taught: the model has learned "five cookies in the box is a finished task", and that prior does not
carry a task label.  If that is right, removing the contradiction fixes both benchmarks at once,
which is the reason to do it first rather than collect more 2x10 data.

**The training loss said nothing about this, and that is the lesson.**  At 15000 steps the flow loss
is 0.00256 and the progress head is at 0.04987 with `R^2 = +0.590` -- the model fits the data it was
given very well.  A loss cannot detect that half the data contradicts its own label; it can only
report that the contradiction was learned.  **Every one of the forty thousand steps in this line was
spent fitting a distribution whose mode was the wrong behaviour.**

**The fix is a sampler setting, not a data rebuild.**  `EpisodeAwareSampler` already accepts
`episode_indices_to_use`, and episode length separates the two populations with a gap of two hundred
frames, so `_patch_episode_filter` in the launcher drops everything below `A3_MIN_EPISODE_FRAMES`
(400) and prints what it did:

```
episode filter     : keep 1270 of 1770 episodes (>= 400 frames, 1022689 frames); 500 dropped as short
```

The alternative -- rebuilding the set without those episodes -- means re-encoding 12 GB of video,
about two hours, to reach the same gradient.  The threshold is a length only because the sampler
sees frame ranges; any value in (303, 506) gives the same partition, and that is worth writing down
so 400 is not later read as a tuned constant.

`A3_DROP_N_LAST_FRAMES=49` rides along: 11.3% of the action rows inside a 50-step chunk were the
"repeat the last action" padding (9.x), and dropping them costs 6% of the frames.

**v7 is the test.**  Same data, same hyper-parameters, same 15000 steps, same base -- the only
difference is that the sampler never sees the 500 contradictory episodes and the padded chunk tails
are excluded.  If the prior hypothesis is right, 2x5 should stop landing on one batch and 2x10 should
get past five cookies; if neither moves, the contradiction was not the binding constraint and the
next thing to try is 2x10 data weight or a DAgger pass rather than more steps.

### 9.49 Reproducing 4.05 on main's own pi0.5 path: three traps, and it is the scene (10-28)

main added a dedicated π0.₅ path in `80e8a7a` (`policies/pi05.py`,
`--policy pi05:<ckpt>` which requires `--task 2x5 --prompt-style pi05_aligned`).  Reproducing
4.05 through it, on main's code, gave **0/10** -- and the reason is not the model.

**Trap 1: the loader deletes what our checkpoint needs, and `from_pretrained` hides it.**
main's `PI05PolicyPlugin` comments say older checkpoints "saved a state projection that π₀.₅ never
reads" and drop `model.state_proj.{weight,bias}` as obsolete.  On this machine's lerobot 0.5.1 the
method main calls (`_convert_openpi_state_dict`) does not exist -- same version number, different
build -- so the file raises `AttributeError` before it can drop anything.  Switching to
`from_pretrained(checkpoint, config=config)` gets a different failure with the same origin:

```
Unexpected key(s) in state_dict: "model.state_proj.bias", "model.state_proj.weight".
```

lerobot's converter skips `state_proj.*` **only without the `model.` prefix**; our launcher saved the
tensors prefixed, so they survive and `strict=True` raises.  `from_pretrained` **catches the
exception**, prints that one line, and returns the model with **randomly initialised weights**.  The
run then proceeds for 900 steps and scores 0/10 while looking healthy.  Teaching the converter to
skip the prefixed form makes the load succeed and lets lerobot say `All keys loaded successfully!`,
which is the only end-to-end evidence that the weights are in the model.

**Trap 2: main's budget is 500 steps, and 2x10 cannot finish inside it.**  Measured on main's own
code: the 2x5 expert completes in **407** steps (10/10), the 2x10 expert hits the 500-step wall at
**10/20**.  main's docs say so ("不保证专家或模型完成四次抓取"), which means a 2x10 number taken at
500 is not comparable with one taken at 3200.

**Trap 3, the one that mattered: the score is entirely scene-dependent.**  Same checkpoint, same
prompt, same seed handling, only the scene differs:

| main's path | score |
|---|---|
| default (randomised boxes, main's cameras) | **0/10** |
| `--pi05-scene <fingerprint> --pi05-scene-mode cameras` | **0/10** |
| `--pi05-scene <fingerprint> --pi05-scene-mode full` | **5/10** |

So changing the cameras alone does nothing and rebuilding the arrangement recovers the number.
That is consistent with what our own harness has always done (`bench_pi05_on_main.py` installs the
training scene on every reset) and inconsistent with "the checkpoint transfers to a fresh layout".
The initial *state* is not the variable -- every seed, 0-5 and 13000 alike, resets to the same arm
pose, differing from the training q01 by at most 0.05 rad on one joint -- so the difference is where
the boxes and the 80 cookies sit.

**What this means for the handover.**  Our `HANDOVER.md` says the number is 4.05 and lists the
prompt and the 900-step budget as the two things that matter.  It does not say that the measurement
requires the training scene, because the driver we shipped installs it automatically and the
document described the driver's flags rather than its implicit behaviour.  A reader following the
README's "replace the policy and run" instructions lands on main's randomised layout and gets 0/10,
which is exactly what happened.  The missing sentence is worth more than the flags we documented.

**Reproduced, for the record.**  On main's code, with the load fixed and the scene installed, 2x5
seed 0 scores **5/10** -- the same value 4.05 has in the 9.41 table for that seed.

### 9.50 A clean base: expert-only, balanced, openpi's recipe, from the untouched pi0.5 (10-29)

Nine arms had been fine-tuned from 4.05, and 4.05 itself came from 4000 steps over 5.6 % of its
set -- about 56 episodes' worth of gradient.  Each arm then had 15000 steps (10 % of its own set)
to learn two tasks on top of that.  Every one of them equilibrated to a **trade**: v7 held 2x5 at
6.25 and lost 2x10 to 2.50, v8 did the reverse, and neither could be added to.  This arm starts
over, on the untouched release, with data that has none of the properties the earlier ones had to
work around.

**The dataset.**  `a3_clean_v9`, built by `scripts/build_clean_v9.py` (400 episodes, 312470 frames):

| | episodes | instruction |
|---|---|---|
| 2x5 | 200 | `Transfer 10 cookies from source column 1 into the target box in two batches of five.` |
| 2x10 | 200 | each episode's own plan, **20 per plan, all ten plans equal** |

Three properties, each of which was a defect somewhere earlier:

* **Whole two-batch 2x5 demonstrations only.**  `/a3_b2_main_aligned_v2_merged` has 1000 episodes
  with a 200-frame gap in the length distribution: 500 run 302-303 frames and deliver **one**
  batch under an instruction asking for two (9.48), the other 500 run 548-550.  Only the long
  half is used, 200 of it taken at even spacing so the selection spans the seed range rather than
  one shard.
* **The 2x10 plan prompts are balanced.**  In the raw collection 190 of 440 episodes carry the
  `n=5` plan and the rarest has 20, so a uniform `first_grasp: random` benchmark tests nine plans
  that were barely trained.  20 per plan fixes the training side of that mismatch.
* **Expert only.**  `a3_graft_v1_merged` is excluded: 55 % of its frames are the policy imitating
  its own stopping behaviour, which is precisely the behaviour this line is trying to remove.

50 : 50 by episode; **32 : 68 by frame**, because a 2x10 episode is 1067 frames against 549.  The
two cannot both be equal and the counts were the requirement, so this is the shape to expect from
any arm on this set.

**The recipe is openpi's, unmodified.**  `configuration_pi05.py` defaults are peak `2.5e-05`,
floor `2.5e-6`, warmup 1000, decay 30000 -- and 30000 is the step budget, so no auto-scaling
engages and the run is the shape openpi intends.  Nothing is tuned: the point of this arm is to be
a reference, so the only departures from stock are the two modules this line adds -- state
projector and history encoder, the latter at `A3_NEW_MODULE_LR=1e-04` -- and the batch size the
card allows.

```
base        : lerobot/pi05_base          <- untouched, not a local checkpoint
budget      : 30000 steps x 6 = 180000 frames = 58 % of an epoch, 230 episodes' worth
final       : step 30K  loss 0.006  lr 2.5e-06  grad_norm 0.522
```

**The latent is the story, and this is the first time it has worked properly.**  Same probe, same
data, across the four arms that have a history encoder:

| arm | R^2 | corr(batches) | latent per-dim std | `A3_NEW_MODULE_LR` | trunk lr |
|---|---|---|---|---|---|
| v6 | +0.535 | 0.735 | 0.0619 | **absent** (all params at 3e-06) | 3e-06 |
| v7 | +0.535 | 0.735 | 0.0619 | 1e-04 | 3e-06 |
| v8 | +0.510 | 0.735 | 0.0607 | 1e-04 | 3e-06 |
| **v9** | **+0.853** | **0.927** | 0.0576 | 1e-04 | **2.5e-05** |

v9 differs from v8 in four ways at once (data hygiene, data balance, from-scratch start, openpi's
learning rate), so this is not an attribution.  What it does establish is that **R^2 = 0.85 is
reachable**, and that 0.51-0.54 was not a property of the task or the encoder.  The mechanism most
likely responsible is the trunk rate: 9.44 found that fresh modules trained at 3e-06 do not move,
and the encoder's *input* is the trunk's state -- at 3e-06 the trunk was barely changing, so a
history of it was barely informative.  Encoder and trunk now train at rates that suit each.

Results, main's benchmark, `profile=basic`, boxes and Cookies randomised, 8 seeds:

| arm | 2x5 (target 3-4) | 2x10 (target 7-8) | note |
|---|---|---|---|
| v6 | 4.50 ±1.07 | 4.50 ±1.15 | no-op module at the fine-tuning lr |
| v7 | **6.25** ±1.03 | 2.50 ±0.94 | 2x5 up, 2x10 down |
| v8 | 2.88 ±0.85 | 5.43 ±2.07 | the reverse |
| **v9** | **4.88 ±0.64** | **11.14 ±1.78** | **both targets met, first arm to do so** |

**9.51 carries the protocol correction that produced v9's 2x5 number**; the 5.38 measured before
it was taken with the wrong instruction and is superseded.

**What 2x10's 11.14 means.**  Three seeds landed 20/20 (`[4, 8, 11, 10, 12, 14, 19]`), so the arm
can complete four batches, which nothing before this line could.  Two things are worth separating:
the balanced plan prompts teach the batch schedule through the instruction, and the 0.853 latent
tells the model where in that schedule it is.  The earlier arms had neither, which is why they
stopped at five Cookies regardless of how many steps they were given (9.46).

**What 2x5's 4.88 ±0.64 still needs.**  The per-seed scores are `[2, 3, 5, 6, 5, 8, 5, 5]` and the
moved-out counts are `[12, 11, 10, 11, 10, 10, 10, 9]` -- mean **10.4, standard deviation 0.9**.
The arm moves exactly what the task asks for and then fails to place all of it.  That is a
different failure from every earlier arm, all of which stopped at five: **the "how much" question
is answered and the remaining gap is placement precision.**  It is the first time in this line that
the 2x5 shortfall is downstream of the picking decision rather than identical to it.

### 9.51 `--prompt-style task` does not mean what its name says on 2x5 (10-29)

Found while checking whether v9's benchmark followed main's standard, and it invalidates the 2x5
number of every arm from v8 onward.

main's 2x5 instruction has two forms and the switch between them is **`source_column`**, not
`prompt_style`:

```python
def _episode_task(self, column, seed):
    return (
        "transfer 10 cookies into target box"                       # column is None
        if column is None
        else f"Transfer 10 cookies from source column {column} into the target box in two batches of five."
    )
```

`column` comes from `--source-column`, and **our driver did not have that flag at all**.  So
`--prompt-style task` -- a name that reads as "use main's own sentence" -- produced, on 2x5, the
short string regardless.  Verified by reading it back out of the result JSON rather than trusting
the flag:

```
2x5   prompt_style = task
      task_instruction = 'transfer 10 cookies into target box'   <- short, not the task sentence
2x10  prompt_style = task
      task_instruction = 'Transfer 20 cookies from source column 1 ... Grasp 4 cookies first ...'
```

**So v8 and v9 were measured on a sentence neither was trained on.**  Both were trained on
`Transfer 10 cookies ... in two batches of five.` and both were evaluated on the one-liner.  9.42
established that this line's checkpoints answer the sentence they saw and nothing else, so those
2x5 numbers measure the sentence rather than the policy.

**The fix and the measurement.**  `--source-column` is now a driver flag, passed to
`make_cookie_benchmark` because `EpisodeExecution` assigns `self.source_column` in `__init__` --
setting it on the benchmark afterwards does nothing, which is how the first attempt at this fix
failed silently.  Re-running v9's eight seeds with the correct sentence and main's own 1000-step
budget:

| protocol | 2x5 scores | mean | moved out |
|---|---|---|---|
| short sentence, 900 steps | `[9, 3, 1, 4, 7, 6, 6, 7]` | 5.38 ±0.91 | mean 12.9, sd **1.9** |
| **task sentence, 1000 steps** | `[2, 3, 5, 6, 5, 8, 5, 5]` | **4.88 ±0.64** | mean **10.4**, sd **0.9** |

**The mean went down and the behaviour got better, and the second fact is the important one.**
Moved-out counts collapse onto the task's own number: mean 12.9 -> 10.4, standard deviation 1.9 ->
0.9.  The short sentence let the arm over-move and occasionally over-score; the task sentence pins
it to "two batches of five", which is what the sentence says.  The score did not rise because the
extra Cookies were never being placed either -- the arm has been failing at **placement**, not at
the decision of how much to move, and that only became visible once the instruction stopped
scrambling the decision.

**Two smaller protocol gaps, both closed.**  Our 2x5 budget was 900 steps against main's CLI
default of 1000, and the driver had no `--inference-seed`, so the flow-matching sampler was
unseeded and one seed's score could drift (9.47 measured 5/10 and 10/10 on the same frozen scene).
`bench_arm.sh` now defaults to 1000 / 3200 and accepts `--source-column`; `STEPS_2X5` and
`STEPS_2X10` override when comparing against older rows.

**The rule, stated so it is checkable.**  A checkpoint's evaluation must send the exact string its
dataset sent.  On this benchmark that string depends on **two** flags, not one, and the one that
matters -- `source_column` -- is not the one whose name says "prompt".  Read the sentence back out
of the result JSON after a run; `prompt_style` alone does not determine it.

### 9.52 The history latent earns its place on the task that needs it, and only there (10-31)

The first controlled ablation of this line: 80 seeds for v9, 64 for the no-history arm, both
trained by the same script from the same base on the same data, differing only in
`A3_HISTORY_ENCODER`.  Seeds 8+ are new; the original 8 are folded in for the merged row.

| arm | 2x5 (cap 10) | 2x10 (cap 20) |
|---|---|---|
| v9, history latent | 5.65 ±0.27 (n=80) | **13.18 ±0.58 (n=80)** |
| v9, no history encoder | 5.67 ±0.37 (n=64) | 10.64 ±0.74 (n=64) |
| difference | **−0.02** | **+2.53** |
| Welch | p = 0.976 | p = 0.0072 |
| paired, same 64 seeds | +0.16, p = 0.71 | **+2.92 ±0.92, p = 0.0022** |

Both arms ran the same seeds, so the paired test is the one to read: on 2x10 v9 wins 39 seeds,
loses 16, ties 9.

**The 2x5 result is a genuine null, and the n=8 sample had it backwards.**  With 8 seeds the
ablation looked *better* on 2x5 (6.50 vs 5.12).  At 80/64 the means are within 0.02 of each
other and the effect size is d = −0.008.  That earlier reading was noise, which is the concrete
cost of CV ≈ 50 % on this benchmark.

**What the latent actually buys on 2x10 is the absence of collapses, not a higher ceiling.**

| | v9 (n=80) | no history (n=64) |
|---|---|---|
| ≤ 5 cookies placed | 6 (7.5 %) | 14 (21.9 %) — p = 0.016 |
| exactly 0 placed | 2 (2.5 %) | 4 (6.2 %) |
| ≥ 15 placed | 34 (42.5 %) | 18 (28.1 %) — p = 0.085 |
| full 20 | 9 | 5 |
| sd / CV | 5.20 / 39.5 % | 5.93 / 55.7 % |

The no-history arm produced `seed 27` with `cookies_in_source` still at 80: it never moved a
single cookie in 1500 steps.  Both arms are equally capable of a good run; the one with the
latent is less likely to produce nothing.  That is what a progress signal should look like on a
four-batch task — it converts "wandered off and kept going" into "carried on".

**On 2x5 the same latent concentrates behaviour without improving it.**  Both means are 5.65,
but the distributions differ:

| score | v9 | no history |
|---|---|---|
| 0 | 2.5 % | 9.4 % |
| exactly 5 | **53.8 %** | 29.7 % (p = 0.005) |
| ≥ 6 | 31.2 % | 43.8 % (p = 0.16, CIs overlap) |
| 10 | 15.0 % | 17.2 % |

So the "exactly one batch then stop" plateau that motivated the module (§9.43) is *more*
common with the latent, not less.  The plausible reading is that two batches is short enough
that a progress signal adds little — and that a confident "one batch done" reading can be
enough to stop.  The four-batch task is where not knowing where you are actually costs
episodes.

**A scoring bug found on the way.**  Four 2x5 episodes report `cookies_in_target = 11` against a
declared `max_score = 10` (v9 seed 86; no-history seeds 48, 49, 54).  The in-target predicate
therefore admits an eleventh cookie, so the score is not bounded by its own maximum.  Clamping
to the cap changes the means by at most 0.05 and no conclusion here, but a benchmark whose score
can exceed `max_score` will eventually be compared against one that cannot.

**Also worth recording: 6 of 144 2x5 episodes ended early.**  All six ended at 430–450 steps
with 10/10 and `success = True` — that is `require_exact_slots` firing on the 2x5 config, i.e.
the intended early exit, not a truncation.  2x10 has no such term (`terminate_on_success` is
off) and every episode runs its full budget.

**Budget.**  2x10 was run at 1500 steps rather than 3200.  Measured on v9's traces: every one of
8 seeds reached its final cookie count by 55.4 s, and the score at 1200 steps equals the score
at 3200 seed-for-seed.  1500 keeps 35 % headroom.  A batch of 8 seeds takes 15–38 min depending
on load; 2x5 at 1000 steps finishes in ~11 min and is absorbed by the 2x10 batch.

### 9.53 The 10-frame latent causes a one-batch plateau, and a 200-frame window does not fix the score (10-31)

Follow-up to 9.52: the window is 10 frames = 0.5 s, while one placement cycle (five cookies
picked, carried, released) takes a median of **264 frames = 13.2 s** off the v9 traces.  A
200-frame window therefore covers 0.76 of a cycle instead of 3.8 %.  A third arm was trained
with `A3_HISTORY_STEPS=200` and every other setting identical; its dataset
(`a3_clean_v9h200`) was verified byte-identical to `a3_clean_v9h` in every column except
`observation.history` (480 -> 9600), so v9 is a valid control.

Latent quality rose as designed:

| | v9 (10 frames) | v9w200 (200 frames) |
|---|---|---|
| R^2 on progress | 0.853 | **0.942** |
| corr | 0.927 | 0.971 |
| pred std / target std | 0.889 | **0.958** |
| ridge upper bound for that window | 0.651 | 0.966 |

The encoder genuinely integrates the whole window: keeping the *last 10 frames* untouched and
replacing the earlier 190 with frames from other episodes drops R^2 from 0.9455 to **-0.7084**.
A model that only read the tail would not move.

**The score did not.** 2x5, 57 seeds for the 200-frame arm against 88 for v9:

| | v9 (10 fr) | no history | 200 fr |
|---|---|---|---|
| mean | 5.60 | 5.76 | 5.88 |
| sd | 2.40 | 2.94 | 2.66 |
| difference vs v9 | -- | +0.16 | +0.27 (p = 0.52) |
| paired vs v9 | -- | -- | -0.04 (p = 0.97) |

Nothing separates the means.  The earlier +1.38 seen at n = 8 was noise, the same trap as
9.52 -- and this time the 8-seed subsample was `{5:2, 6:3, 7:2, 10:1}`, a run of eight
consecutive non-zero, non-5 episodes whose probability under the v9 rate is small enough to
look convincing.

**What did change is the shape, and it points at the 10-frame window as the cause of the
plateau rather than as a cure for it.**

| score band | v9 (10 fr, n=88) | no history (n=72) | 200 fr (n=57) |
|---|---|---|---|
| <= 4 (failed to finish a batch) | 14.8 % | 25.0 % | 17.5 % |
| **exactly 5** | **54.5 %** | 31.9 % | 31.6 % |
| 6..8 (partial progress) | **12.5 %** | 20.8 % | **33.3 %** (p = 0.0028 vs v9) |
| 9..11 | 18.2 % | 22.2 % | 17.5 % |
| exactly 5 *or* exactly 10 | **68.2 %** | 45.8 % (p = 0.007) | **40.4 %** (p = 0.001) |

The 10-frame arm is close to bimodal: it either delivers one clean batch or all two, and rarely
stops in between.  Both alternatives -- removing the encoder, and extending the window 20x --
smooth that out to the same extent (31.6 % vs 31.9 % on the plateau, indistinguishable).
So the plateau is a property of the *10-frame* latent, not of having a progress signal at all.

That also explains why the mean is flat while the shape moves: the plateau acts as a floor.
v9 buys its 5.60 with fewer sub-5 episodes and pays for it by rarely scoring 6-8.  A smoother
policy trades a few more low scores for a few more partial ones and lands on the same average.

**2x10 is still at 8 seeds and says nothing yet.**  13.12 (200 fr) vs 13.62 (v9) vs 11.75 (no
history), all p > 0.7.  The 10-frame arm was the only one that beat no-history there
(9.52, paired p = 0.0022), so the interesting question -- does a longer window help the task
that actually rewards progress awareness -- is still open.

**Two process notes worth keeping.**

A sweep was left running while the bridge it depended on was killed by a cleanup `pkill`.  The
driver failed instantly on `ConnectionRefusedError [Errno 111]` and the orchestrator advanced
through batch after batch at "0 min" each, producing 103 empty logs and no results.  The
resume logic keys on result files, so it looked like progress in the log and like nothing on
disk.  Kill the orchestrator before killing a bridge, and check that a batch's duration is
plausible before trusting "batch done".

The 8-seed snapshot was also read as if it were the whole arm, because the pairing was done on
the intersection of seed numbers and that intersection was 0-7 while 49 more seeds sat on disk
under the same tag.  Any paired analysis should print the arm's total n next to the paired n.

**2x10 closes the question, and the answer is the opposite of 2x5.**

104 seeds for the 200-frame arm against 88 for v9, all on the same scenes:

| | v9 (10 fr, n=88) | no history (n=72) | 200 fr (n=104) |
|---|---|---|---|
| mean | **13.22** | 10.76 | **11.05** |
| sd | 5.27 | 6.05 | 5.76 |
| vs v9, unpaired | -- | -2.46 | **-2.17 (p = 0.008)** |
| vs v9, paired | -- | -- | **-2.01 ±0.75 (p = 0.0079)**, 27 win / 51 lose |
| full 20 | 11.4 % | 8.3 % | 4.8 % |
| >= 15 | **43.2 %** | 29.2 % | 31.7 % |
| <= 5 | 8.0 % | 22.2 % | 19.2 % |
| exactly 0 | 2.3 % | 6.9 % | 8.7 % |

The 200-frame arm is statistically indistinguishable from having **no history encoder at all**
(+1.10 ±0.89 vs no-history, p = 0.23) and significantly worse than the 10-frame arm.  Twenty
times the window, R^2 0.942 instead of 0.853, and the four-batch task gets worse.

**Why this is not a paradox.**  Ten frames is exactly the length of one *grasp-and-place
primitive* at 20 Hz, which is what the flow-matching action expert has to produce: a 50-step
chunk is 2.5 s, and the last ten frames describe the phase the arm is in right now.  A 200-frame
window also contains the previous two batches, and the encoder compresses all of it into a
single 128-d vector and one 2048-d prefix token.  That token then has to serve two purposes at
once -- "which phase am I in" for the action expert and "how many batches are done" for the
progress head -- and the longer window makes the second easier (R^2 0.853 -> 0.942) while
blurring the first.  The 10-frame arm wins because the information it carries is the
information the action head needs; the 200-frame arm optimises a proxy that only the auxiliary
head reads.

This is the same lesson as 9.52 from another angle: an auxiliary metric that improves while the
task gets worse is not evidence that the module works.  Both times the honest signal came from
seed sweeps, not from the latent diagnostic.

**What the whole sequence says about the module.**

| task | no history | 10 frames | 200 frames |
|---|---|---|---|
| 2x5 (2 batches, 27 s) | 5.76 | 5.60 (p = 0.98) | 5.88 (p = 0.52) |
| 2x10 (4 batches, 53 s) | 10.76 | **13.22 (p = 0.0022)** | 11.05 (p = 0.23) |

The latent earns its place on the long task, and only at the window length that matches a
single action primitive.  Lengthening it is not an improvement; it is a different module that
happens to score better on its own supervision.

### 9.54 The progress head is what keeps the latent nonzero, not a loss term that helps it (11-01)

Follow-up to 9.53.  The 200-frame result raised a specific worry: the auxiliary progress loss
might be steering the latent toward "how many batches are done" and away from the phase
information the action expert needs.  `A3_HISTORY_PROGRESS_WEIGHT=0` tests that by removing the
supervision while keeping the channel, the architecture, the KL weight and everything else.

The arm did something the hypothesis did not anticipate.

**It collapsed.**  `mu` has 128 dimensions and the loss pushes it toward 0 only through KL
(weight 0.05).  With the progress term gone, nothing pulls it away from 0:

| step | mu_std | KL |
|---|---|---|
| 1 | 0.0315 | 7.2e-4 |
| 750 (peak) | 0.0345 | 5.9e-4 |
| 6000 | 0.0068 | 2e-5 |
| 15000 | ~0.003 | ~0 |
| 30000 | **0.0017** | **0.00000** |

v9 finished at 0.0576.  This arm ends **34x narrower with KL at exactly zero** -- the capacity
constraint is no longer binding because the latent has already been squeezed to a constant.
The check that catches this is a one-line grep of `mu_std` from the training log; it was read
at 3000 steps and the direction was already unambiguous.

**What it scored.**  32 seeds on each task, paired by seed:

| | 2x5 (n) | 2x10 (n) |
|---|---|---|
| v9, 10 frames + progress | 5.60 (88) | **13.22 (88)** |
| no history encoder | 5.76 (72) | 10.76 (72) |
| **progress_weight = 0 (collapsed)** | 5.69 (32) | **11.38 (32)** |
| paired vs v9 | -0.72 ±0.52, p = 0.21 | -1.94 ±1.14, **p = 0.10** |
| paired vs no-history | -0.50 ±0.74, p = 0.53 | +0.81 ±1.30, **p = 0.55** |

On 2x10 the collapsed arm sits between the two controls and is **statistically
indistinguishable from having no encoder at all**.  On 2x5 all three arms are equal, which is
consistent with 9.52 -- 2x5 never responded to the latent in the first place.

**So the answer to the original question is "not testable this way", and the reason is
informative.**  Removing the supervision does not produce "a free latent that the flow loss
shapes as it likes".  It produces **no latent**: the flow loss is content with a constant
prefix token (the action expert can ignore it), and a constant token is the minimiser of KL.
The progress loss is therefore not merely teaching the latent to report progress -- it is the
only term in the objective that requires the latent to carry anything at all.

**This reframes every earlier result.**  9.52 measured the value of "a latent kept alive by
progress supervision", not the value of "a latent".  The 200-frame failure in 9.53 fits the same
frame: a longer window offers more to encode and scores better on R^2, but with KL compressing
harder over a 20x larger input, the information that actually reaches the prefix token shrinks.
R^2 was measuring the encoder, not the token.

**A cost note on the measurement itself.**  The 8-seed snapshot of this arm read 14.38 on 2x10
-- above v9.  Adding seeds moved it 14.38 (n=8) -> 13.75 (n=16) -> 12.62 (n=24) -> 11.38
(n=31), monotonically down.  That is the fourth arm in this sequence where n=8 pointed the
wrong way, and the reason is always the same: CV near 50 % on a 0-20 integer score means a
handful of good scenes dominate.

### 9.55 Two harness bugs found while running these arms (11-01)

Both are silent, and both were caught only because a number looked implausible.

**The `mu_std` monitor read one line out of 120.**  `grep '^\[history\]'` matched only the
first record, because the trainer writes its progress bar with `\r` and no newline, so each
`[history]` line is appended *after* `Training: ...` rather than starting a line.  The first
record happens to be the pre-training one, so a collapse over the following 30000 steps would
have looked like a flat line at 0.0315.  Fixed by dropping the `^` anchor; the fix was
verified against v9's log, which then yields the full sequence (0.0315 -> 0.1227 by step 1000).

**A dead bridge turned a sweep into 103 no-op batches.**  `run_seed_sweep.sh` loads the
checkpoint once and reuses it.  Killing that bridge with an unrelated `pkill` made every driver
exit on `ConnectionRefusedError [Errno 111]` within a second, and the orchestrator recorded
"batch done in 0 min" and advanced -- logging progress while writing nothing.  Added a health
check: if a batch produced zero results in under 60 s, print the driver's error and abort the
whole sweep.  The rule worth keeping: **judge a batch by its duration, not by its completion
message.**

### 9.56 Why 2x10 stops at 13 of 20: a failed grasp is never retried (11-02)

Four hypotheses were tested and three were rejected.  The survivor explains every earlier anomaly,
and it is not a precision problem at all.

**What the loss is not.**  Tracking each cookie's own xy position through 8 recorded episodes
(3200 frames each) shows the arm moved 129 cookies and delivered 117: **90.7 %, one or two lost
per episode**.  An earlier count-based estimate said 21 % was lost to whole failed batches; that
was wrong, because `cookies_in_source` also falls when a cookie is nudged across the box's
internal edge, not only when it is lifted out.  Geometry cannot be fooled that way.  **Placement
precision is not the bottleneck**, which removes `cartesian_delta` from the list.

**What the loss is not, part two.**  `bench_arm.sh` has been run at 1500 and at 3200 steps and the
scores are equal (13.18 vs 13.62).  The reason: across all 8 episodes the arm makes its last grasp
between frame 490 and 1053 and its last placement between 534 and 1236, then stays idle for
**100 to 133 seconds**.  The episode budget was never the constraint.

**What the loss is not, part three.**  It is not "learned the expert's final pose".  The expert's
last frame sits at `y = 0.118 +- 0.011` (near the target box, 200 episodes); the policy parks at
`y = 0.39`, which is the **far inner edge of the source box** (source centre 0.310 + half-width
0.101 = 0.411), and it is inside the demonstration distribution (14 % of expert frames exceed
y = 0.36).  So the parking spot is a legal "reaching into the far end of the source box" pose --
which is exactly what a grasp start looks like.

**What it is.**  The arm reaches, does not take a cookie, and never reaches again.

| seed | 张-合周期 | 末次张开 | 末次闭合 | 末次放入 | 放入后还有尝试 |
|---|---|---|---|---|---|
| 5 | 4 | 1114 | 1142 | 3190 | 0 |
| 6 | 4 | 1146 | 1172 | 2938 | 0 |
| 1 | 4 | 1116 | 1141 | 2418 | 0 |
| 0 | 3 | 843 | 907 | 3132 | 0 |
| 4 | **0** | -- | -- | 2007 | 0 |
| 2 | 3 | 817 | 905 | 794 | 1 |
| 3 | 3 | 874 | 901 | 531 | 2 |
| 7 | 2 | 578 | 601 | 565 | 1 |

The jaw completes 0-4 open-close cycles per episode, all of them in the first ~60 s, and then
stops cycling.  Meanwhile the effector keeps moving with normal amplitude -- mean |action| is 0.85
during the idle tail against 0.90 while working -- and the arm tracks it to within 0.005-0.035 rad.
Action magnitude standard deviation drops from 0.123 to 0.020, so the policy is outputting a small
drifting wiggle, not zeros, and the arm obeys.  Nothing is stuck, blocked, or out of reach.

**The failure is that a miss is terminal.**  The demonstrations are expert-only: every episode in
`a3_clean_v9` is a successful run.  A policy trained on them has seen "reach, close, carry, place"
and has never seen "reach, close on nothing, try again".  When a grasp misses, the state it lands
in has no continuation in the data, and the closest thing the action expert can produce is a quiet
drift near the source box.

**Why this matters more than the previous four experiments.**  `cartesian_delta`, the window
length, the KL weight and the progress weight were all attempts to make the *policy* better at
what it already does.  This says the policy is missing a *behaviour*, and no amount of tuning the
inputs supplies a behaviour that is absent from the demonstrations.  The grafted-DAgger dataset
that 9.52 rejected for mixing policy rollouts into expert data is the correct remedy here, for the
opposite reason: not to add volume, but to add recoveries.

**Consequence for the score ceiling.**  Four grasps of five cookies is the designed route to 20.
Episodes with 4 cycles score 18-19; episodes with 2-3 score 10-15; the one with 0 scores 17 by
pulling 10 and 9 in two oversized grabs.  So 2x10's 13.2 average is mostly "3.1 grasps instead of
4", and the mechanism behind a missing grasp is a miss that was never retried.

### 9.57 Is the stall state recoverable? The gate, measured on v9's own end states (11-02)

9.56 ends with "a miss is terminal, so the demonstrations need recoveries".  The question that has
to be answered before any collection is whether a privileged controller can finish from the state
v9 actually leaves -- if it cannot, collecting grafted episodes collects failures.

**The mechanism already exists, and it is not new.**  `A3CookieBatchExpert` has an `adopt` entry
(`second_batch_mode="adopt"`, `_adopt_current_state`) written for exactly this: it reads the
occupancy, marks the in-target Cookies as already done, aims at a column with free slots and moves
nothing.  It has three start strategies (`A3_ADOPT_START`), and the measured reason for the default
is in 9.39: opening the jaws at the takeover pose displaced **13 of 70** untouched source Cookies,
while `lift` displaced 0.  `scripts/collect_graft.py` already drives it (policy phase, then expert
phase, one trajectory) and 9.40 reports the production run: 428 attempts, **200 accepted (46.7 %)**,
**167 tier A**, 0 tier B.  And 9.41 measured the specific claim: from the exact state a policy
leaves after releasing a second batch 9.6 mm too high, `adopt` finished **2 of 2** seeds at tier A.

So the answer to "can it take over" is yes, with production numbers behind it.  What had not been
checked is whether the 2x10 state v9 leaves is inside what `adopt` can accept.

**The gate, on the eight recorded 2x10 end states.**  `adopt` refuses when no column has five free
slots (`adopt_infeasible`), because one grasp carries a whole group of five into one column.  The
2x10 target is two columns of ten, so this is a real gate.  Read from
`success_diagnostics.json:final_conditions.target_slot_occupancy` on `diag/v9_run/2x10` (the idle
tail changes nothing, so the final frame *is* the stall frame):

| seed | in target | col0 taken | col1 taken | free per column | `adopt` feasible | score after |
|---|---|---|---|---|---|---|
| 0 | 13 | 8 | 3 | 2 / 7 | yes | 18 |
| 1 | 19 | 9 | 8 | 1 / 2 | **no** | 19 |
| 2 | 12 | 8 | 2 | 2 / 8 | yes | 17 |
| 3 | 10 | 9 | 0 | 1 / 10 | yes | 20 |
| 4 | 15 | 10 | 4 | 0 / 6 | yes | 20 |
| 5 | 20 | 9 | 8 | 1 / 2 | **no** | 20 |
| 6 | 19 | 9 | 9 | 1 / 1 | **no** | 19 |
| 7 | 1 | 0 | 0 | 10 / 10 | yes | 20 |

**5 of 8 feasible -- and the three that are not are already at 19-20.**  The infeasible states are
the near-misses, which do not need recovery; the states that need it (10-15 in target) all have a
column with six to ten free slots.  Replaying "add one batch of five into the emptiest column while
a whole column has five free" over these occupancies gives the ceiling:

| task | v9 mean | after takeover | gain |
|---|---|---|---|
| 2x5 (8 seeds, cap 10) | 5.12 | **10.12** | +5.00, every seed full |
| 2x10 (8 seeds, cap 20) | 13.62 | **19.25** | +5.62 |

So for 2x5 the existing expert closes the gap completely, and for 2x10 the recovery ceiling is
19.25/20 -- the remaining 0.75 is the three near-misses that have nothing left to recover.

**The free rows are contiguous, which is what makes it work.**  `_place_pose` aims a group at the
*mean* y of a set of rows, so a batch of five needs five adjacent free rows, not five free rows
anywhere.  Per column, from the same occupancy:

| seed | col0 free rows | col1 free rows | longest run | what a five-grasp can do |
|---|---|---|---|---|
| 0 | 3,4 | 3-9 | 7 | col1 rows 3-7 |
| 1 | 5 | 6,7 | 2 | nothing (already 19) |
| 2 | 2,7 | 0, 3-9 | 7 | col1 rows 3-7 |
| 3 | 4 | 0-9 | 10 | col1 rows 0-4, then 5-9 |
| 4 | -- | 0-5 | 6 | col1 rows 0-4 |
| 5 | 2 | 4,8 | 1 | nothing (already 20) |
| 6 | 4 | 9 | 1 | nothing (already 19) |
| 7 | 0-9 | 0-9 | 10 | four batches |

Every seed that needs help has a run of at least six, i.e. a five-grasp fits.  The three that do not
are the ones with 19-20 already in the box.

**What is actually missing, and it is two small things rather than a new controller.**

1. **`A3CookieBatchExpert` is the wrong expert for 2x10, and main already has the right one.**
   `A3Cookie2x10Expert` (`controllers/cookie_2x10_expert.py`) takes a `Cookie2x10Plan` with
   `batch_sizes = (n, 10-n, n, 10-n)` and `target_rows(batch_index)` naming **which rows** a grasp
   aims at, so a group of `n` lands in rows `0..n-1` and the complement in `n..9`.  Variable batch
   sizes are not a special case there -- the collected 2x10 data already contains grasps of 1 to 9
   (`a3_2x10_n5` shards, plan variants n=1..9).  What it has **no** entry for is `adopt`: its plan
   comes from the seed, not from occupancy.  The batch expert's `_place_pose` is the 2x5 version --
   it aims at the column's *centre* -- so aimed at a half-full 2x10 column it descends onto the
   Cookies the policy already placed.
2. **The collection tree cannot host a 2x10 scene at all.**  `config.py:588` asserts
   `len(target_slots_local_m) != 10 -> raise`.  The 20-slot config exists only in main's tree, where
   `batch_expert.py` does not.  So a 2x10 recovery collector needs that assertion relaxed (every
   consumer of the slot list already derives its length rather than assuming ten:
   `_target_slot_occupancy`, `_column_sizes`, `_free_slots_per_column`).

**Consequence for the plan.**  The recovery primitive is "place the longest contiguous free run in
one column, up to five".  For 2x5 that is `adopt` as it stands.  For 2x10 it is `adopt` with the
plan derived from occupancy instead of from the seed -- the same `target_rows`/`batch_size`
machinery, fed by the judge's occupancy rather than by `Cookie2x10Plan.from_seed`.  Nothing about
the scene, the arm or the source box is blocking; 9.41 already showed that for the 2x5 case and the
free-run table above shows it for 2x10.

### 9.58 The 2x10 `adopt` entry, built and measured (11-02)

9.57 concluded that the 2x10 stall state is recoverable but that the two pieces needed to
*use* that were missing.  Both now exist, and the numbers below are the first measurement of
the entry itself.

**What was missing, restated as code.**

* `A3CookieBatchExpert`'s `adopt` decides a **column** (`_free_slot_column`) because a 2x5
  column is five slots.  In a ten-row column that is not enough: `_place_pose` aims at the
  **mean y of the rows it is given**, so a group aimed at a half-full column's centre lands
  on the Cookies already there.
* `A3Cookie2x10Expert` (main's tree) has the machinery -- `plan.target_rows(batch_index)`
  names which rows a grasp goes to -- but its plan comes from the seed, not from occupancy.
* The collection tree could not build a twenty-slot scene at all: `config.py` refused any
  `target_slots_local_m` whose length was not ten.

**What was built.**

| piece | file | what it does |
|---|---|---|
| `A3Cookie2x10RecoveryExpert` | `batch_expert_2x10.py` | reads the judge's occupancy, cuts each column into **contiguous free runs**, and turns the longest runs into groups of five; `_place_pose` aims each group at *its own rows*, and `_relief_sign` leans the release away from an occupied neighbour |
| three hooks on the base class | `batch_expert.py` | `_is_final_batch`, `_batches_complete`, `_on_batch_advanced` -- the batch scene's "two batches" assumption, made answerable by a subclass; defaults are the old constants, so every existing path is unchanged |
| twenty-slot config | `configs/cookie_2x10_recovery.yaml` | `cookie_batch.yaml` character for character except the target box and its grid, which are main's `cookie_2x10.yaml` -- that is what makes an episode here comparable with main's benchmark |
| the slot-count rule | `config.py` | ten became "even and at least four"; every consumer (`_target_slot_occupancy`, `_column_sizes`, `_free_slots_per_column`) already derived the length, so the constant was the only thing pinning it |
| scene + task | `collection.py`, `tasks.py` | `a3_cookie_2x10_recovery`, contract `released_20_upright_settled_contained_60_remain` |

**The instruction is part of the experiment, and it was wrong.**  The scene's default prompt is
this branch's 2x5 sentence.  v9 was fine-tuned on main's, and 9.42 measured a v9-lineage
checkpoint at **7/20 under the sentence it had seen and 0/20 under the other one**.  So the
recovery scene uses `COOKIE_2X10_PROMPT`, which is character for character the `n=5` plan
prompt `a3_clean_v9` contains (`task_index 0`).  `n=5` rather than any other variant because
the recovery expert's grasp *is* five -- an episode whose instruction says "grasp 1 then 9"
while the controller takes five is a mismatch the policy would be right to distrust.

**The measurement, on v9's own recorded end states.**  `scripts/probe_2x10_recovery.py`
rebuilds each recorded occupancy by teleporting that many Cookies into the same slot indices,
then hands the box to the new expert.  Both numbers are reported, because they answer
different questions: `cookies_in_target` is what main scores, and slot occupancy is what a
dataset is worth.

| seed | recorded | in-box before -> after | slots before -> after | plan | outcome |
|---|---|---|---|---|---|
| 0 | 13 | 11 -> **16** | 11 -> 15 | 1x5 | ok, 278 steps |
| 1 | 19 | 17 -> 17 | 17 -> 17 | -- | infeasible (runs of 2 and 1) |
| 2 | 12 | 10 -> **15** | 10 -> 13 | 1x5 | ok, 278 steps |
| 3 | 10 | 9 -> **14** | 9 -> 13 | 2x5 | batch 1 ok, batch 2 `VERIFY_RELEASE` |
| 4 | 15 | 14 -> **19** | 14 -> 15 | 1x5 | ok, 282 steps |
| 5 | 20 | 17 -> 17 | 17 -> 17 | -- | infeasible (runs of 1, 1, 1) |
| 6 | 19 | 18 -> 18 | 18 -> 18 | -- | infeasible (runs of 1, 1) |
| 7 | 1 | 0 -> **20** | 0 -> 18 | 4x5 | ok, 1045 steps |

**Every takeover that was feasible delivered exactly five.**  Seeds 0, 2 and 4 gained +5 in-box
and +4, +3, +1 in slots -- the slot gain is smaller than five because a group of five placed
from one wide grip lands 3-5 on slot centres, which is the same reason the 2x5 contract sets
`require_exact_slots=False`.  Seed 7 started from an empty box and ran **four** consecutive
batches to 20/20 in 1045 steps with 18 slotted, which is the multi-batch state machine working
across both columns (columns 0, 0, 1, 1 -- the plan fills the longest run first and then moves
on).

**The three infeasible seeds are the ones that do not need recovery**, exactly as 9.57
predicted from occupancy alone: 17, 17 and 18 in the box with no run of five free rows
anywhere.  A `VERIFY_RELEASE` failure on seed 3's second batch is a placement that did not
settle, not a planning failure -- batch 1 landed fine from the same state.

**Two gaps this measurement exposes, neither of them in the new expert.**

1. **`adopt` plans from *slots*, and a policy's Cookies are not always in slots.**  On the 2x5
   probe, three of five triggered episodes had `slots@takeover` of 0, 0 and 2 with
   `in_target` of 6, 6 and 7: the policy put Cookies in the box without slotting them, so the
   occupancy reads "empty" while the box is physically full of a loose block.  The recovery
   then plans into rows that a Cookie is already lying across.  A feasibility check that
   counts only free slots will accept those states and fail in `LIFT` or `DESCEND_TO_PLACE`,
   which is what seeds 5 and 7 did.  The fix is a physical check -- is the group's insertion
   corridor clear -- rather than a bookkeeping one.
2. **`_should_hand_over` requires the source count to stop shrinking.**  That is right for a
   policy that stalls, and wrong for one that keeps *removing* source Cookies while failing to
   place them: seed 2 of the 2x5 probe ended with five in the box, five in slots, and
   `graft_triggered=False` because the source count never stopped falling.  The gate should
   also fire on "the target count has not increased for N steps", which is the condition that
   actually describes a stall.

### 9.59 The 2x5 takeover rate, measured on v9 with the right instruction (11-02)

The gate 9.40 reports (428 attempts, 200 accepted, 46.7 %) was measured on the 4.05
checkpoint with this branch's own instruction.  Re-measuring it on v9 first required fixing the
instruction, because `collect_graft.py` drives phase one with `scene.task.prompt` and that is
this branch's sentence -- while v9 was fine-tuned on main's.  Under the wrong sentence the same
twelve seeds gave **1 / 8 accepted** and one episode that moved **zero** Cookies in 700 steps;
9.42 measured exactly this failure mode (7/20 vs 0/20 for one weight set under two sentences).

With `--task-prompt "Transfer 10 cookies from source column 1 into the target box in two
batches of five."` (the `source_column=1` sentence, `a3_clean_v9` `task_index 10`):

| seed | tier | triggered | slots at takeover -> final | in box at takeover | accepted | expert failure |
|---|---|---|---|---|---|---|
| 0 | A | yes | 4 -> 9 | 5 | **yes** | -- |
| 1 | C | yes | 5 -> 5 | 5 | no | `DESCEND_TO_PLACE` |
| 2 | no_trigger | **no** | -- -> 5 | -- | no | -- |
| 3 | A | yes | 4 -> 9 | 5 | **yes** | -- |
| 4 | no_trigger | **no** | -- -> 10 | -- | no | -- |
| 5 | A | yes | 4 -> 9 | 5 | **yes** | -- |
| 6 | A | yes | 4 -> 9 | 5 | **yes** | -- |
| 7 | C | yes | 4 -> 2 | 5 | no | `LIFT` |
| 8 | A | yes | 4 -> 9 | 5 | **yes** | -- |
| 9 | C | yes | 4 -> 4 | 7 | no | -- (ran out of steps) |
| 10 | no_trigger | **no** | -- -> 9 | -- | no | -- |
| 11 | no_trigger | **no** | -- -> 4 | -- | no | -- |

**5 / 12 accepted (42 %), 8 / 12 triggered, and 5 of the 8 triggered reached tier A (62 %).**
That is the same rate as 9.40's 46.7 %, on a different checkpoint and a different instruction --
so the primitive is stable and this is not a lucky sample.

**The five successes are one shape, exactly.**  Every one is `slots_at_takeover = 4`,
`slots_final = 9`, `in_box_at_takeover = 5`: the policy placed its first five (four of them on
slot centres), and the expert's single group of five carried the box to ten.  That is the
behaviour the dataset is being collected for, and it is reproducible rather than incidental.

**Four seeds never triggered, and they are the interesting ones.**

* seed 4 finished the job itself -- ten in slots, eleven in the box.  Eleven violates the
  contract (`70_remain` wants exactly seventy left), which is a *policy* error the gate has no
  business rescuing, and the episode is correctly refused.
* seed 10 reached ten in the box with nine slotted and **69** left in the source -- one Cookie
  over.  Also refused, also correctly: the contract is exact.
* seeds 2 and 11 stopped at five and four with `graft_triggered = False`, which is the gate's
  own blind spot: `_should_hand_over` required `cookies_in_source` to stop falling, and these
  two were still lifting Cookies out of the source every batch while placing none of them.

**Two fixes, both in the gate rather than in the expert.**

1. **The stall trigger now needs *neither* progress signal to have moved.**  Tracking the
   target alone is not enough -- a batch cycle is 300 steps and the drawn delay is 100-200, so
   "the target has not grown" is true for most of every normal transport and would fire
   mid-carry.  Tracking the source alone is 9.40's rule and misses seeds 2 and 11.  So both are
   tracked, *either* improving counts as progress, and the trigger reads when progress last
   happened.
2. **`adopt` now checks the insertion corridor, not just free slots.**  Three of the eight
   triggered episodes had `slots_at_takeover` of 0, 0 and 2 with 6, 6 and 7 Cookies *in the
   box*: occupancy read "ten free" while a loose block of six lay across the rows the expert was
   about to use.  Those are the `LIFT` (pad forces 2.86 / 2.88 N, opening 0.185) and
   `DESCEND_TO_PLACE` ("Cookie slipped out of batch during transport") failures -- 400 steps
   spent, and a failure name that blames the controller.  `_corridor_blocked` now answers the
   geometric question ("is a Cookie inside the box standing in this group's band?") and
   `adopt_infeasible` refuses the state up front with the blocking Cookie's index.

**A third finding that is not a bug in anything.**  Seed 7 went `slots 4 -> 2`: the expert's
group *reduced* the slot count.  Its five landed in the box but pushed the policy's four off
their centres.  A takeover that trades one arrangement for another still raises
`cookies_in_target`, so it is not a lost episode -- but it is why the acceptance rule has to be
tier A (five slots *gained*) rather than "ten in the box".

### 9.60 Two gate fixes, one of which lost, and the bottleneck they exposed (11-02)

9.59 ended with two candidate fixes to `collect_graft.py`'s handover gate.  Both were built and
both were measured over the same twelve 2x5 seeds.  One is kept and one is off by default,
because it was measured to be *worse*.

| arm | accepted | triggered | tier A among triggered |
|---|---|---|---|
| baseline (source-stall rule only) | **5 / 12 (42 %)** | 8 / 12 | **5 / 8 (62 %)** |
| + corridor check + stall rule (`--stall-steps 350`) | 4 / 12 (33 %) | 10 / 12 | 4 / 10 (40 %) |

**Kept: `_corridor_blocked`.**  It fires correctly and cheaply.  Seed 9 of the fixed run is the
demonstration: `slots@takeover = 1` with **nine** Cookies in the box, and the refusal is instant
and named --

```
every column with five free slots has a Cookie lying in it
  (blockers by column: {0: 0}, in target: 9, slotted: 1)
```

-- where the old gate would have spent ~400 steps in `LIFT` or `DESCEND_TO_PLACE` and reported a
placement failure.  It also provably does not over-refuse: re-running the 2x10 probe dry-run after
the change gives byte-identical plans for all eight seeds, including the three that are infeasible
for the *other* reason (no run of five free rows).

**Rejected: the stall rule, and the reason is the interesting part.**  Adding "the policy took a
batch and never delivered it" raised the trigger count (8 -> 10) but lowered both rates.  The two
extra takeovers were the two worst states in the set:

* seed 2 triggered with **four slots already filled** and ended with **zero**.  The expert's group
  of five came apart during transport and scattered the box -- a takeover that made the episode
  strictly worse than leaving the policy alone (baseline: five in the box, five slotted).
* seed 11 triggered with one slot filled and ended at one, failing in `DESCEND_TO_PLACE`.

So "the policy is still reaching but not delivering" does not mean "the box is ready to be
rescued".  It usually means the policy is holding a batch, or has just dropped one somewhere the
expert cannot see; taking the box off it costs more than the batch it was going to lose.  The flag
is kept (`--stall-steps`, default 0 = off) with the measurement in its help text, so the idea can
be revisited without being re-derived.

**The bottleneck the fixed run exposed is not the box at all.**  Three of its six failures are the
same string:

```
Cookie slipped out of batch during transport     (seeds 1, 2, 11)
```

That is the *expert's own* grasp failing to hold five Cookies, not the box state, not the
corridor, and not the trigger.  `adopt` prefers a pristine source row
(`ADOPT_IN_PLACE_TOLERANCE_M = 0.0004`) and falls back to the loose tolerance when none is
pristine -- and a policy that has already taken Cookies from that column leaves the remaining rows
nudged by about a millimetre, which is enough for the pad to close on a misaligned row.  The five
successes are consistent with this: they are all `slots@takeover = 4`, i.e. the policy used *one*
column and left the other's rows untouched.

**What that means for the plan.**  The takeover rate is 42 %, and it is 42 % because of the source
row, not the target box.  Two consequences:

1. Collection cost is ~2.4 attempts per accepted episode.  At 4.5 minutes per attempt that is
   ~11 minutes per episode, so 200 episodes is ~36 hours -- which is affordable but means the
   quota should be set from this rate rather than from 9.40's 46.7 % on a different checkpoint.
2. If the rate needs to go up, the lever is the **source row**, not the gate: either restrict
   collection to episodes where the policy used one column (which is what every success looks
   like), or make `_select_batch` in `adopt` reject a disturbed row outright instead of falling
   back to the loose tolerance.  The second is a small change and is the obvious next experiment.

**One incidental bug.**  `collect_graft.py --help` had never been run and crashed with
`ValueError: unsupported format character ')'`: argparse `%`-formats every help string, and the
`--fast-render` text contains `(50%)`.  Escaped to `(50%%)`.  Worth noting because the same
mistake is one keystroke away in any new help string that quotes a rate.

### 9.61 How to test 2x10, now that the 2x5 pipeline is validated (11-02)

The 2x5 takeover now works end to end: the collector drives v9 with the sentence it was trained
on, the gate fires, the expert's `adopt` places a group, and 42 % of attempts are accepted.  None
of that transfers to 2x10 for free, because **the 2x10 failure is a different shape**: 2x5 stops
after *one* batch, 2x10 keeps going for four and stops at 3.1 of them (9.56).  So the test ladder
has to answer three questions in order, and each one can kill the plan.

**What is already established, and what it does not cover.**  `probe_2x10_recovery.py` rebuilt
v9's eight recorded end states by teleporting Cookies into the same slot indices, then handed the
box to `A3Cookie2x10RecoveryExpert` (9.58).  Every feasible state gained exactly five, and seed 7
ran four consecutive batches from empty to 20/20.  That establishes the **planner**: reading a
scattered twenty-slot occupancy, cutting it into contiguous free runs, aiming each group at its
own rows, and re-planning after each batch.  It does **not** cover the arm: the policy's hand is
not where the probe's teleport leaves it, the box pose is the probe's not the benchmark's, and the
source rows have not been worked around by a policy.

**Stage 1 -- live takeover on the real 2x10 scene.**  This is the measurement that matters, and
the collector can already make it: `--scene a3_cookie_2x10_recovery` selects the twenty-slot
config, `--expert` infers the recovery controller from the scene name, and the scene dump needs no
new one because robot, source box, cameras and all eighty Cookie positions are identical between
main's 2x5 and 2x10 benchmarks -- verified parameter by parameter, only the target box differs and
that is config-level.  The 2x10 config itself is byte-equivalent to main's `cookie_2x10.yaml` on
every physical field, so a number measured on it is comparable with main's benchmark.

```
A3_ADOPT_STRICT_ROW=1 python scripts/collect_graft.py \
  --root /root/autodl-tmp/datasets/graft_2x10_probe \
  --episodes 12 --max-attempts 12 \
  --socket /tmp/graft_v9.sock \
  --scene a3_cookie_2x10_recovery \
  --task-prompt "$(python -c 'import sys;sys.path.insert(0,"src");
      from a3_dual_arm_sim.tasks import COOKIE_2X10_PROMPT;print(COOKIE_2X10_PROMPT)')" \
  --policy-steps 1500 --expert-steps 1400 \
  --execute 16 --no-videos --keep-failures
```

**`--policy-steps 1500`, not the 2x5 default of 700.**  9.56 measured v9's last grasp at frames
490-1053 and its last placement at 534-1236.  At 700 the policy is cut off while it is still
working, the gate's `cookies_in_source` rule never sees a stall, and every episode would be
recorded as `no_trigger`.  `--expert-steps 1400` because one group takes 280-600 steps (the probe
measured 278, 278, 282, 596) and the plan can hold two.

**Stage 2 -- the question the 2x5 data never had to ask: one batch or the whole episode?**  For
2x5 a single recovery is the entire remaining task.  For 2x10 it is one of up to four, and the
collector's `GraftPolicy` naturally runs the expert to `DONE`, i.e. it collects the **whole
remaining sequence** including the re-plan between batches.  That is the right default for 2x10 --
seed 7 of the probe showed the re-plan works (columns 0, 0, 1, 1) -- but it also means an accepted
episode is much longer, so the quota and the wall-clock estimate have to be recomputed rather than
copied from 2x5.

**Stage 3 -- the prompt-variant trap, which can make the result uninterpretable.**  `a3_clean_v9`
contains **ten** distinct 2x10 plan sentences (`Grasp n cookies first, then 10-n ...`, n = 1..9,
plus a skip variant) and main's benchmark draws `first_grasp=random`, so one eval seed in eleven
sees the `n=5` sentence the recovery data would carry.  A model trained only on `n=5` recoveries
would have learned "when the instruction says five, recover", and 10/11 of main's seeds would
never ask it that.  Two defensible ways out, and they are different experiments:

* **Narrow, single-variable (recommended first).**  Collect and evaluate at `first_grasp=5`.  The
  question becomes "does recovery data fix the stall at all", with the instruction held constant,
  which is the same discipline 9.42 imposed on the 2x5 numbers.  It is not an acceptance number.
* **Full distribution (second).**  Collect under all ten variants.  That needs the recovery
  controller to obey the plan's batch sizes rather than always taking five -- the capability
  main's `A3Cookie2x10Expert` already has (`plan.batch_sizes`, `plan.target_rows`) and which was
  deliberately not ported (9.58) because a *recovery* only ever needs five.  So this is a real
  port, not a flag.

**The acceptance ceiling, from the 2x5 data, because it bounds the 2x10 plan too.**  Of the twelve
2x5 baseline episodes: five accepted, two *correctly refused* (the policy over-delivered -- eleven
in the box, or sixty-nine left in the source, both contract violations that no takeover should
rescue), three lost to the source row, and two that never triggered.  So fixing the source row
alone caps the rate near **8/12 = 67 %**, and the remaining headroom needs the no-trigger cases,
which is exactly what the stall rule was measured to make *worse* (9.60).  Any 2x10 quota should
be planned at ~2.4 attempts per accepted episode, i.e. ~11 minutes per episode at the 2x5 timings,
and 2x10 attempts are longer.

**The honest order of operations.**  Stage 1 first, twelve seeds, because it answers whether the
arm can do what the planner can.  If the rate is in the same 40-65 % band as 2x5, collect.  If it
is near zero, the reason will be in `graft_expert_failure_phase` and the probe has already ruled
out the planner, which narrows it to the grasp or the trigger -- two things the 2x5 run has
already instrumented.

### 9.62 Strict source rows: it removed a failure class and did not raise the rate (11-02)

9.60 ended by naming the source row as the suspected cause of the transport failures, and by
proposing to make `adopt` refuse a disturbed row instead of falling back to the loose 1.6 mm
tolerance.  Built as `ADOPT_STRICT_ROW` / `A3_ADOPT_STRICT_ROW=1`, measured over 16 seeds.

| arm | n | tier A | rate | triggered | failures by phase |
|---|---|---|---|---|---|
| baseline (loose fallback) | 12 | 5 | 42 % | 8 | `DESCEND_TO_PLACE` 1, `LIFT` 1, no_trigger 4 |
| + corridor + stall rule | 12 | 4 | 33 % | 10 | `DESCEND_TO_PLACE` 3, `ADOPT` 1, no_trigger 2 |
| + corridor + **strict rows** | 16 | 4 | 25 % | 8 | `SELECT_COOKIE` 3, `ADOPT` 1, no_trigger 8 |

**The rate comparison is not a measurement, and saying so is the point.**  The bridge is
stochastic (9.40: two samples of one observation differ by up to 0.086 rad), so each run draws a
*different policy trajectory*.  That is why the no_trigger count moves 4 -> 2 -> 8 across arms
that cannot possibly affect the trigger -- it is sampling noise, not an effect.  A 4/16 vs 5/12
difference is well inside what that noise produces, so the honest reading is **"no change
detected"**, not "strict is worse".

**What the change actually did, measured where the noise cannot reach it.**

| failure | expert steps | what the arm does |
|---|---|---|
| `DESCEND_TO_PLACE` (baseline) | 351 | carries a bad row all the way to the box |
| `LIFT` (baseline) | 547 | closes on a misaligned row, times out |
| `SELECT_COOKIE` (strict) | 254 | clears the hand away, then finds no pristine row |
| `ADOPT` (strict) | **0** | refuses at takeover, before moving |

And the failure class the change was aimed at is gone:

| arm | `Cookie slipped out of batch during transport` |
|---|---|
| baseline | 1 / 12 |
| corridor + stall rule | **3 / 12** |
| corridor + strict rows | **0 / 16** |

So the hypothesis was right -- the loose fallback *is* what produced the transport slips -- and
fixing it removed them entirely.  It just does not add episodes, because refusing a row is still a
failure; it converts a 350-550-step corruption into a 254-step or 0-step named refusal.  For a
collection run that is worth having (it is ~40 % of the expert budget on the failing half of the
attempts, and it stops writing episodes whose second group was scattered), but it is a **cost**
lever, not a rate lever.  Kept, off by default, documented as such.

**The rate lever is the no-trigger cases, and that is what 2x10 makes urgent.**

### 9.63 2x10, first live takeovers: the pipeline works and the gate is the problem (11-02)

9.61 proposed the ladder; this is stage 1, three seeds, on the real twenty-slot scene with v9
driving for 1500 steps.

| seed | tier | in box @ takeover | slots @ -> final | frames | failure |
|---|---|---|---|---|---|
| 0 | **B** | 5 | 0 -> 3 | 1111 | `VERIFY_RELEASE`: released Cookies not upright, settled and contained: [21..25] |
| 1 | C | 6 | 5 -> 5 | 914 | `LIFT` timeout, pad forces 10.92 N |
| 2 | no_trigger | -- | -- -> 5 | 1501 | the policy never stalled |

**Seed 0 is the result that matters: the whole chain works.**  The policy placed five (none on
slot centres), the gate fired after its drawn 116-step delay, `A3Cookie2x10RecoveryExpert` took
over, planned against the occupancy, grasped five from the source, carried them in, and the box
went **5 -> 10 Cookies** -- the exact +5 the probe predicted.  It failed only at
`VERIFY_RELEASE`: the group landed in the box but not upright and settled, which is tier **B**,
"usable but must be labelled", not an unrecoverable state.  So the planner, the arm, the clearing
lift, the multi-column aiming and the re-plan are all confirmed on the real scene, and the
remaining defect is placement quality -- the same axis the 2x5 data already has a tier for.

**Seed 2 is the structural finding, and it inverts a 2x5 conclusion.**  The policy ran the full
1500 steps; `first_five` was reached at step 230, so **1270 steps passed with the box stuck at
five** while `cookies_in_source` kept falling the whole time.  The gate's rule is "hand over when
the source stops shrinking", and on 2x5 that rule works because a stalled policy *stops reaching*.
On 2x10 it does not: the policy keeps lifting Cookies out of the source, missing, and lifting
again, which is exactly the behaviour 9.56 identified as the bottleneck.  So the source rule is
blind to the 2x10 failure by construction, and **the stall rule that lost on 2x5 (9.60) is the one
2x10 needs** -- its target population is not "the policy gave up" but "the policy is still trying
and failing", which on 2x10 is most episodes and on 2x5 was 4 of 12.

That reframes the flag: `--stall-steps 0` is the right default for a 2x5 collection and the wrong
one for a 2x10 collection.  The window still has to be a whole attempt cycle (9.60's lesson
stands -- a 2x10 grasp-to-release is 280-600 steps, so >= 700 is the honest value here, not 350).

**The 2x10 ladder, restated with these numbers.**

1. **Gate first, not data.**  Re-run stage 1 with `--stall-steps 700`.  If the no-trigger cases
   convert to takeovers, the rate should land near the 2x5 band; if they convert to fast refusals,
   the bottleneck is the source row on 2x10 too, and the answer is `A3Cookie2x10Expert`'s variable
   batch sizes rather than a five-Cookie group.
2. **Then placement quality.**  Seed 0's tier B is a release that did not settle.  Two
   candidates, both cheap to test: the group's `_relief_sign` lean (4 mm) may be too small for a
   ten-row column whose neighbour was placed by a policy, and `_place_pose`'s wall clearance
   (5 / 3 mm) was measured for the 2x5 box.
3. **Then the prompt distribution.**  9.61's stage 3 stands: `a3_clean_v9` carries ten 2x10 plan
   sentences and main's benchmark draws `first_grasp=random`, so a model trained only on `n=5`
   recoveries has learned a conditional that 10 of 11 eval seeds never ask for.  Narrow first
   (`first_grasp=5`), full distribution second -- and the second needs the variable-batch port.

### 9.64 2x10 with the stall gate: the takeover works and cannot chain (11-02)

9.63 ended with "fix the gate first, and the stall rule that lost on 2x5 is the one 2x10 needs".
That prediction was right, and the result is the most important negative finding of this line:
**the recovery primitive is reliable per batch and the 2x10 task needs four batches.**

`--scene a3_cookie_2x10_recovery --stall-steps 700 --policy-steps 1500 --expert-steps 1400`,
twelve seeds, v9, `A3_ADOPT_STRICT_ROW=1`:

| seed | tier | triggered | in box @ -> final | slots @ -> final | frames | expert failure |
|---|---|---|---|---|---|---|
| 0 | C | yes | 5 -> 10 | 5 -> 10 | 907 | `MOVE_TO_SLOT` |
| 1 | C | yes | 0 -> 0 | 0 -> 0 | 1075 | `SELECT_COOKIE` |
| 2 | no_trigger | no | -- -> 14 | -- -> 4 | 1501 | -- |
| 3 | **A** | yes | 5 -> **20** | 5 -> 17 | 1769 | -- (3 batches) |
| 4 | C | yes | 5 -> 10 | 5 -> 10 | 1194 | `LIFT` |
| 5 | C | yes | 5 -> 10 | 5 -> 10 | 1172 | `LIFT` |
| 6 | C | yes | 5 -> 10 | 4 -> 4 | 1249 | `DESCEND_TO_PLACE` |
| 7 | C | yes | 5 -> 15 | 0 -> 12 | 1451 | -- (DONE after 2) |
| 8 | C | yes | 6 -> 16 | 2 -> 12 | 944 | -- (DONE after 2) |
| 9 | no_trigger | no | -- -> 14 | -- -> 11 | 1501 | -- |
| 10 | no_trigger | no | -- -> 15 | -- -> 11 | 1501 | -- |
| 11 | C | yes | 2 -> 2 | 2 -> 2 | 1501 | `DESCEND_TO_PLACE` |

**The gate fix worked.** Triggered 9/12, against 2/3 in the three-seed smoke and -- more to the
point -- the three `no_trigger` seeds are the ones where the *policy* reached 14, 14 and 15 in the
box. So the stall rule now fires exactly on the episodes that stall and leaves alone the episodes
that are still progressing, which is what 9.63 predicted it would do.

**And the chain degrades, in a way that is arithmetic rather than mysterious:**

| batch | seeds completing it |
|---|---|
| 1st | 7 / 9 |
| 2nd | 3 / 9 |
| 3rd | 1 / 9 |

Mean gain per takeover is **+6.1 Cookies (1.22 batches)**, and reaching 20 from a stall at five
needs **+15 (3 batches)**.  The 2x5 recovery needs one batch and succeeds at 62 % *given a
trigger*; the 2x10 recovery needs three to four and lands at 1/12.  A per-batch rate `p` compounds
as `p^n`, so the identical primitive is worth 62 % on one task and ~10 % on the other.  Nothing
about the controller changed between those two numbers.

**Two distinct causes, and they need different fixes.**

1. **The grasp/transport fails from batch two onward (5 seeds).**  Seeds 0, 4, 5, 6 and 11 each
   placed a clean first group and then failed in `MOVE_TO_SLOT`, `LIFT` or `DESCEND_TO_PLACE` --
   the same source-row failure 9.60/9.62 measured on 2x5, except that on 2x5 it costs the episode
   and here it costs the *chain*.  The expert's own first grasp disturbs the rows beside it
   (the pads straddle the group and stand outside it), so by batch two the column it is working
   has been worked twice.
2. **The plan runs out of groups before the task does (2 seeds).**  Seeds 7 and 8 stopped at
   `DONE` after two batches, at 1091 and 591 expert steps -- they did not hit the 1400-step
   budget, so `_plan_targets` had nothing left to offer.  Both had **zero or two slots filled and
   5-6 loose Cookies in the box**: `_corridor_blocked` refused the remaining groups because those
   loose Cookies lie across the corridors.  That check is doing its job (those groups *would* have
   scattered), but on a box where the policy leaves its Cookies loose it also removes most of the
   plan.

**So the honest conclusion for 2x10 is that "expert cleans up after the policy" does not work
here**, for the same structural reason 9.41 found for the 2x5 box and then escaped: one privileged
grasp is reliable, four in a row are not.  The 2x5 dataset works because its failure needs exactly
one recovery.

**What that leaves, in order of cost.**

1. **Do not chain: hand back after one batch.**  The collector already has the machinery
   (`GraftPolicy` owns both phases), and a policy-expert-policy rotation is what DAgger is
   supposed to be.  Each cycle gives the policy a fresh chance to continue, and the recorded
   trajectory teaches exactly one recovery.  This is the cheapest experiment and it is the one
   that matches what the data can carry.
2. **Fix the source row for repeated batches**, which is the same problem 9.62 attacked and which
   strict mode made *worse* by refusing rows the expert could still use -- after its own first
   grasp, nothing is within 0.4 mm.  A tolerance measured against the row's *own* last observed
   position rather than the nominal layout is the obvious next attempt.
3. **Port the variable batch sizes** from main's `A3Cookie2x10Expert` (`plan.batch_sizes`), so one
   batch can carry the eight to ten Cookies that a fragmented box leaves, turning three grasp
   attempts into one.

**One measurement that got worse and has to be stated.**  The twelve episodes average **11.8/20**
in the box against v9's 13.2 at 1500 steps -- but that is not like-for-like, because nine of the
twelve had their policy cut off at the stall.  What *is* like-for-like: at the stall point those
nine held five or six Cookies, and the takeover carried them to a mean of 11.0, i.e. **the expert
added ~6 and removed none**.  So the primitive is not harmful, it is simply worth less than the
ten Cookies a full 2x10 recovery needs.

### 9.65 The collector never sent the history channel, and nothing said so (11-02)

Found while checking whether a hand-back-after-one-batch rotation would be feasible, which is
the wrong order to find it in: the fix comes first because **every takeover rate in 9.59-9.64 was
measured on a model that was not v9**.

**The defect.**  `collect_graft.py`'s `Client.infer` sent four arrays -- three camera views and
`observation.state` -- and no `history`.  The chain from there is silent at every step:

1. the adapter's `_history_vector` catches the missing key and returns `None`, deliberately, so a
   pre-v6 checkpoint still loads;
2. the batch therefore has no `observation.history`, so the launcher's `_stash` sets nothing;
3. the patched `embed_prefix` begins `if window is None: return embs, pad_masks, att_masks`.

So **the history token was never appended**.  This is not "a policy with a zeroed history" -- it is
a *different architecture*: a prefix one token shorter than the one v9 was trained with, whose
attention mask and positional layout are both off by one.  Nothing raised, no episode failed, and
the only symptom was a takeover rate that happened to look plausible: 42 % against 9.40's 46.7 %,
on a checkpoint (4.05) that had no history encoder at all and so could not have been measuring the
same thing.  Two numbers agreeing was the coincidence that hid the bug.

**The fix, in three places.**

* `Client` takes `history_frames` (default 10) and builds the window itself, term for term the
  benchmark's: ten frames of `[q(16), q_dot(16), a_previous(16)]`, oldest first, leading frames
  **repeated from the first** rather than zero-filled, and `a_previous` being the action the
  environment actually executed one step earlier -- which is why `infer` now takes it as an
  argument (`GraftPolicy.last_action`) instead of guessing.
* The collector prints the width it will send at startup, and the adapter now prints a **WARNING**
  when a checkpoint *with* an encoder receives no vector.  A warning rather than an exception
  because the same adapter serves the benchmark path, where running without the window is a
  legitimate (if confounded) choice; the point is that the choice can no longer be invisible.
* `--history-frames 0` is the documented opt-out for pre-v6 checkpoints.

**Verified end to end, not by inspection.**  Bridge restarted on v9 with `history_dims=480`, one
episode collected: the collector printed `history: 10 frames (480-wide observation.history)`, the
bridge logged **zero** missing-history warnings, no shape error was raised (a wrong width is a hard
error in this adapter), and the episode finished tier A at 10/10 with slots 4 -> 9 in 670 steps.

**What this invalidates.**  Every takeover rate below was measured on the shorter-prefix model and
has to be re-measured before it is quoted:

| result | status |
|---|---|
| 2x5 baseline 5/12, corridor+stall 4/12, strict 4/16 (9.59-9.62) | **invalid as a v9 number** |
| 2x10 smoke 0/3, stall=700 1/12 (9.63-9.64) | **invalid as a v9 number** |
| 2x10 batch-chain decay 7/9 -> 3/9 -> 1/9 (9.64) | **stands** -- the expert's own grasping, no policy call |
| `_corridor_blocked`, strict rows, the tier rule (9.60-9.62) | **stands**, all mechanism |
| `probe_2x10_recovery` +5 per batch, 4 batches from empty (9.58) | **stands**, no policy phase |
| 9.40's 46.7 % on 4.05 | **stands**, that checkpoint has no encoder |

**One thing the fix does not settle, and it is the next question.**  The benchmark appends **one
history frame per chunk request**, not per control step -- the call sits inside `if not
self._queue` -- so its window spans `frames * execute` = 160 steps while the training data's spans
ten.  Its own docstring says "this runs every step, not every `execute`" and the code does not.
The collector now replicates the *code*, deliberately, because the 13.2/20 baseline and every
ablation under it were measured that way and changing both at once would move the baseline.  That
means collection and evaluation are consistent with each other and both inconsistent with training
-- which is a real finding about v9, not a bug being hidden: it may be part of why the latent
carried so little (9.54).  Fixing both together is a clean single-variable experiment, and it
should be run before any more collection.

### 9.66 After the history fix: the 2x5 numbers move, and move against the project (11-02)

`--history-frames 10`, per-step window (`9.65` + the A change), sixteen seeds, v9, every other flag
at its 9.59 value.  **2 / 16 accepted, both `policy_only`.  Zero expert recoveries.**

| seed | tier | triggered | in box @ -> final | slots @ -> final | policy steps | expert failure |
|---|---|---|---|---|---|---|
| 0 | C | yes | 5 -> 5 | 4 -> 4 | 308 | `LIFT` timeout, pads 2.52 N, opening 0.224 |
| 1 | C | yes | 5 -> 5 | 5 -> 5 | 603 | `slipped out of batch during transport` |
| 2 | no_trigger | no | -- -> 5 | -- -> 0 | 700 | -- |
| 3 | no_trigger | no | -- -> **11** | -- -> 8 | 700 | -- (over-delivered, contract refused) |
| 4 | C | yes | 5 -> 5 | 0 -> 0 | 335 | `LIFT` timeout, pads 5.75 N, opening 0.255 |
| 5 | no_trigger | no | -- -> **10** | -- -> 6 | 700 | -- (69 left in source, contract refused) |
| 6 | no_trigger | no | -- -> 6 | -- -> 3 | 700 | -- |
| 7 | C | yes | 5 -> 5 | 0 -> 5 | 311 | `APPROACH` not converging, 4.1 mm short after 108 steps |
| 8 | **policy_only** | no | -- -> **10** | -- -> 7 | 516 | -- (**accepted**, 70 left) |
| 9 | C | yes | 7 -> 7 | 6 -> 6 | 618 | `ADOPT`: free per column [3, 1] |
| 10 | **policy_only** | no | -- -> **10** | -- -> 5 | 503 | -- (**accepted**, 70 left) |
| 11 | no_trigger | no | -- -> 5 | -- -> 4 | 700 | -- |
| 12 | no_trigger | no | -- -> 5 | -- -> 5 | 700 | -- |
| 13 | C | yes | 5 -> 5 | 4 -> 4 | 344 | **none named**, expert budget 700 exhausted |
| 14 | C | yes | 5 -> 5 | 4 -> 4 | 578 | `slipped out of batch during transport` |
| 15 | no_trigger | no | -- -> 8 | -- -> 3 | 700 | -- |

**The policy is much stronger than it was.**  Four episodes reached ten or eleven in the box under
the policy alone (seeds 3, 5, 8, 10) where the pre-fix runs typically stalled at five.  Two of those
are clean successes; the other two violate the exact contract (eleven moved, or sixty-nine left).

**The expert got worse, and that is the number that matters: 0 / 7 against 5 / 8 before.**  With
seven triggered episodes and a stochastic bridge this is a small sample, so it is not yet a
measurement of an effect -- but the *failure modes* are informative, because they are all on the
**source** side:

| failure | count | what it means |
|---|---|---|
| `LIFT` timeout at 2.5-5.8 N with the jaws part-closed | 2 | the pads closed on a row that does not hold together |
| `Cookie slipped out of batch during transport` | 2 | the grasp formed and then came apart |
| `APPROACH` not converging, 4.1 mm short | 1 | the chosen row is outside the vertical-grasp envelope |
| `ADOPT` refused, free slots [3, 1] | 1 | correct refusal |
| **no failure named, expert budget exhausted** | 1 | see below |

The first three are the same family 9.60/9.62 measured, and the plausible mechanism is *causal*:
a stronger policy takes more Cookies out of the source, so by the time it stalls the rows the expert
needs have been worked harder and are more disturbed.  That is a hypothesis, not a result -- the
controlled test is to replay the expert from each episode's recorded takeover state.

**A real logic gap the run exposed: the expert can loop without ever naming a failure.**  Seed 13
burned all 700 expert steps, moved nothing, and reported `failure_phase = None`.  The per-phase
guard is 420 steps and it never fired, so the machine was cycling
(`SELECT_COOKIE -> ... -> SELECT_COOKIE`) with `phase_steps` reset at every advance.  Every other
failure in this project is *named*; this one is not, and a budget exhaustion that says only
"`failed`" cannot be told apart from "the arm was thinking".  A global "no target progress in N
steps" guard belongs in `_act_step` alongside the per-phase one.

**Two judgment bugs, now confirmed on data.**

* **Budget exhaustion is stamped `failed`, which erases `policy_only`.**  `tier` derives
  `no_trigger` from `failed`, so seeds 3 and 5 -- where the *policy* delivered ten and eleven --
  are recorded as `no_trigger` rather than `policy_only`.  Their refusal is correct (both violate
  the exact contract), but the recorded reason is wrong, and a genuinely successful
  `policy_only` that happens to finish at step 700 would be silently discarded.
* **The `--stall-steps` default is wrong for 2x5 in this regime.**  Five of the nine
  `no_trigger` episodes ended with the policy still removing Cookies from the source
  (`in_source` 69-70 while `source_at_first_five` was 70-71) and the box stuck at 5-8.  That is
  exactly the "keeps taking, never delivers" state the stall rule exists to catch, and it is off.

**What this changes about the plan, stated plainly.**  The premise of the whole line was 9.56's
"a failed grasp is never retried".  With the history channel finally wired, the policy retries
enough to finish the task 4 times in 16 on its own, and the recovery rates measured in 9.59-9.64
are not the recovery rates of this model.  Before any further collection or training, three things
have to be re-established on the fixed pipeline: the benchmark score (9.56's 13.2/20 was measured
on the old window), the trigger rate, and the per-batch expert success rate from recorded states.

### 9.67 What the policy actually does when it fails: it closes its jaws in mid-air (11-02)

> **RETRACTED -- see 9.68.**  The events this section calls `air_close` are grasps whose jaws
> began closing during the transport and completed at the row.  The probe anchored its window on
> the close rather than on the descent bottom, so it measured the descent and missed the grasp.
> The true failure is a release 9 mm above the box floor, and "closed on air" never happens.
> Kept because the mechanism of the mistake is worth more than the claim was.

9.56 counted jaw open-close cycles and concluded "a failed grasp is never retried".  That is a
statement about **attempts**, and it cannot distinguish two failures with nothing in common:

* the jaws close **around the row** and nothing comes up -- a grasp that failed to hold;
* the jaws close **in the air** and there was never a grasp at all.

`scripts/probe_grasp_events.py` separates them on the recorded rollouts.  For every close event
it measures (a) how far the effector is above the nearest Cookie's own height, (b) which Cookies
then move with the hand, and (c) which of those rise more than 8 mm.  A grasp lifts a row ~12 mm
in these traces, so the classes are cleanly separated rather than threshold-tuned:

| class | meaning |
|---|---|
| `grab_lift` | closed at the row and carried Cookies up |
| `grab_no_lift` | closed at the row and they did not come up |
| `air_close` | closed with the effector **30 mm or more above** the row |
| `missed` | descended to the row, closed, nothing between the pads |

**2x10, eight rollouts, 22 close events:**

| class | count | share |
|---|---|---|
| `grab_lift` | 6 | 27.3 % |
| `grab_no_lift` | **1** | **4.5 %** |
| `air_close` | **10** | **45.5 %** |
| `missed` | 5 | 22.7 % |

Failed closes have a **median height gap of 77.2 mm**.  The first successful grasp of seed 0
closes at 12.0 mm, and seeds 1, 2 and 7 close at 77-85 mm -- three quarters of a metre above the
table, on a pile that stands 0 mm tall relative to itself.

**2x5, the same eight seeds, the same checkpoint:**

| class | count | share |
|---|---|---|
| `grab_lift` | **13** | **81.2 %** |
| `grab_no_lift` | 0 | 0 % |
| `air_close` | **0** | **0 %** |
| `missed` | 3 | 18.8 % |

**So the answer to "grab and fail to lift, or hover with an empty gripper" is: neither, and the
distinction matters.**  It is not a half-held grasp (4.5 %); it is a close performed in mid-air
(45.5 %).  And it is **specific to 2x10** -- the same weights, the same source box, the same eighty
Cookies produce zero air-closes on 2x5.

**Two consequences, and the second one is why this matters for the recovery plan.**

1. **"Never retried" was the wrong diagnosis.**  The policy does keep cycling; what it does not do
   is *descend*.  A recovery dataset built on "what to do after a grasp missed" addresses a failure
   that accounts for 1 of 22 events.  The failure that accounts for 10 of 22 is "close at 80 mm",
   and no amount of recovery-from-a-missed-grasp data teaches the hand to come down.
2. **It explains the history result in 9.66 without needing a new mechanism.**  A policy that stops
   with its jaws shut 80 mm above the pile leaves the expert an inherited state
   `_cookies_in_jaws() == 0` -- which is exactly the `jaws_held = 0` the source report measured on
   the failing takeover (seed 1).  The expert then takes the "clear away and re-grasp" path, and
   that path is what fails in `LIFT`.  The two findings are one finding.

**What this does not yet explain, stated so it is not overclaimed.**  *Why* 2x10 air-closes and 2x5
does not is open.  The source box, the Cookies and the weights are identical; what differs is the
instruction and the target box.  `a3_clean_v9` carries ten 2x10 plan sentences whose `n` varies
from 1 to 9, and a plan that says "grasp 1 cookie" is satisfied by a jaw that closes on one -- the
model may have learned a close as a *plan step* rather than as the end of a descent.  That is a
hypothesis with a cheap test: score 2x10 with `first_grasp=5` only, where the sentence matches what
2x5 asks for, and see whether air-closes disappear.

### 9.68 RETRACTION of 9.67, and the failure mode it was hiding (11-02)

**9.67 is wrong and this replaces it.**  It claimed that 45 % of the policy's closes on 2x10 happen
"in mid-air" with the effector 78-84 mm above the row, and concluded the policy "never descends".
The measurement that produced that was anchored at the wrong event.

**What went wrong.**  The probe detected a close as the gripper channel crossing 0.45 downward and
called the following 25 steps its "lift window".  But the gripper channel starts closing *during*
the transport, tens of steps before the hand arrives at the row: on 2x10 seed 1 the channel goes
`1.000 -> 0.263` at frame 288 while the effector is at z = 0.847, and the hand then **descends from
0.847 to 0.768 by frame 320 and grasps four Cookies there**.  The window (288-313) contained the
descent and none of the grasp, so a perfectly ordinary grasp was recorded as "closed on air".

**The corrected measurement.**  Anchor on the *descent bottom* -- a local minimum of the effector's
height within 20 mm of the Cookies' own height, inside the source box -- and ask what left the
source and what entered the target over the following 250 steps.  Restricted to the episode's
active phase (a parked arm drifts at Cookie height and otherwise registers as an "attempt"):

| | 2x5 (20 attempts) | 2x10 (29 attempts) |
|---|---|---|
| grabbed and **delivered** | 9 (45 %) | 24 (83 %) |
| **grabbed, delivered nothing** | **7 (35 %)** | **4 (14 %)** |
| grabbed 1-2 only | 4 (20 %) | 1 (3 %) |
| **grabbed nothing at all** | **0 (0 %)** | **0 (0 %)** |

**"Closed on air" never happens -- not once in 49 attempts across both tasks.**  Every descent that
reaches the row takes at least one Cookie out of the source.  (The 110 `nothing` events in the raw
2x10 count are all in the idle tail, where the parked arm's drift crosses the Cookie-height plane
forty times.)

**And "grabbed but delivered nothing" is not a transport failure either.**  Tracking the five
Cookies through the 400 steps after each such grasp gives their final resting height:

| task | seed | step | left source | entered target | final z | vs a placed Cookie |
|---|---|---|---|---|---|---|
| 2x5 | 0 | 320 | 5 | 0 | 0.7776 | **+9.1 mm** |
| 2x5 | 2 | 52 | 5 | 0 | 0.7794 | **+10.9 mm** |
| 2x5 | 3 | 59 | 5 | 0 | 0.7789 | **+10.4 mm** |
| 2x5 | 4 | 73 | 5 | 0 | 0.7776 | **+9.1 mm** |
| 2x5 | 6 | 311 | 5 | 0 | 0.7772 | **+8.7 mm** |
| 2x5 | 7 | 71 | 5 | 0 | 0.7771 | **+8.6 mm** |
| 2x10 | 3 | 626 | 5 | 0 | 0.7773 | **+8.8 mm** |

A correctly placed Cookie rests at z = 0.7685; the box rim is at 0.7755.  Every one of those
batches is at **0.777 +/- 0.001, i.e. 1.5-2 mm above the rim, 8.6-10.9 mm above the floor**.  The
full trace of 2x5 seed 0's batch is the demonstration:

```
f320  y=0.284-0.309  z=0.769   grasped at the far end of the source
f360                z=0.852   lifted
f420  y=0.091-0.116  z=0.827   carried to the target box, above it
f480  y=0.098-0.123  z=0.777   released -- 8.5 mm high
f560+  unchanged              rests on the rim for the remaining 500 frames
```

`cookies_in_target_mask` counts **0 of the 5** at every one of those steps, and the episode's score
stays at 5 to the end.  **So the batch is grasped correctly, carried correctly, aimed correctly,
and let go 9 mm too high, where it balances on the rim.**  This is 9.41's release-height finding,
reproduced on the v9 checkpoint with the history channel working -- and it is the *dominant* 2x5
failure (35 % of attempts), not a grasp failure at all.

**Two cases are different and are worth separating.**  2x5 seed 1 and 2x10 seed 7 end at
`z = 0.766-0.768` (correct height) with a **negative** target delta: the batch reached the box and
*knocked cookies out of it*.  That is a placement-collision failure, not a release-height one.

**What this changes for the recovery plan.**  9.56's premise -- "a failed grasp is never retried"
-- was already weakened by 9.66; it is now dead in this form.  The policy does not fail to grasp
(0 of 49 attempts), it fails to **release at the right height**, and it fails to **stop** doing
that.  A DAgger dataset of "recover from a missed grasp" addresses a failure that does not occur.
What the data supports is either (a) teaching the release, which is a *height* correction and would
show up as a change in the last 20 frames of each demonstration, or (b) leaving the release alone
and teaching the expert to clean up after it -- which is the same +5-per-batch primitive 9.58
measured, and which 9.66 showed is worth ~45 % of attempts when the history channel is present.

**The comparison the pictures actually show**, since the frames were rendered for it.  All three
views are the policy's own inputs, so this is what the model sees, not a diagnostic overlay.

* **`front`** answers *where and how high*.  In the delivered grasp the forearm is low and the
  fingers are down among the Cookies at the **near** end of the source box.  In the lost grasp the
  arm is over the **far** end and visibly higher -- and after the release the batch is a yellow
  cluster sitting on the box's far rim, above the ones already inside.
* **`left_wrist`** answers *what is between the fingers*.  Delivered: the two fingers are
  interdigitated with the Cookie rows, with Cookies on both sides right up to the pads.  Lost: the
  fingers are at the bottom of frame with a wide brown gap between them and the rows -- and at the
  release frame the view contains the **target box's white rim** with the batch balanced on it.
* **`right_wrist`** shows neither, in either case: the right arm only holds its own gripper shut
  and never enters the workspace, so that camera is table and shadow.  Worth saying plainly because
  it means two thirds of the image tokens in every observation carry no information about this
  task -- which is a training-side observation, not a failure.

**A caveat about what a single frame can prove.**  The release-height failure is invisible in the
*close* frame: at that moment the two cases look similar (fingers in the pile, jaws near shut).  It
only becomes visible ~160 frames later, at the release, and only in the `front` view as "the batch
is above the rim rather than in the box".  This is exactly why the failure went undiagnosed for so
long -- every convenient frame, including the ones a person would naturally screenshot, is the
frame before the mistake happens.

### 9.69 The failure, measured properly: a batch that lands leaning 27 degrees (11-02)

9.68 was directionally right ("release too high, batch perched") and factually wrong on the
mechanism and on both of its key numbers.  Two measurement errors caused it, and both are worth
recording because they are the kind that produce confident wrong answers.

**Error 1: the demonstrations were the wrong task.**  `a3_clean_v9` holds two tasks in one file --
**episodes 0-199 are the 2x10 demonstrations, 200-399 are the 2x5 ones** -- and every "expert"
number in 9.67/9.68 sampled `np.unique(episode_index)[:80]` without checking.  All of it was 2x10.
That is why the demonstrations appeared to release four times per episode and at 0.825: they were
2x10 episodes with four batches, and the detector was wrong on top of that.

**Error 2: the detector marked the retreat, not the release.**  The expert raises the gripper
channel to ~1.0 when it *starts its retreat*, and the earlier detector called the crossing of 0.45
the release.  The actual letting-go is a small rise -- 0.28 (the width the jaws descend at) to
0.38-0.43 -- with the effector still inside the box.  A release is now: the channel rising by
>0.03 over 4 steps while the effector is over the target box **and at or below the rim**.

**With both corrected (`scripts/measure_release_task.py`, `scripts/census_final_states.py`):**

| | releases / ep | release z | vs rim | off-column | above rim |
|---|---|---|---|---|---|
| expert, 2x5 | **2.00** | 0.7628 | −12.1 mm | **0.3 mm** | 0 / 40 |
| v9, 2x5 | 2.12 | 0.7651 | −9.9 mm | 1.4 mm | 0 / 17 |
| expert, 2x10 | **4.00** | 0.7692 | −5.8 mm | 4.4 mm | 0 / 80 |
| v9, 2x10 | **3.00** | 0.7711 | −3.9 mm | 4.5 mm | 3 / 24 |

So the policy does **not** release "50 mm in the air": on 2x5 all 17 releases are below the rim, and
the height differs from the expert by 2.3 mm.  **Release height is not the failure.**

**What the failure actually is.**  Every Cookie that left the source and was refused, in all eight
2x5 rollouts:

| seed | counted (and its column) | refused near the box | that batch's z | implied tilt |
|---|---|---|---|---|
| 0 | 5 @ x 0.063-0.074 | 5 @ x 0.120-0.121 | 0.7776-0.7777 | **27.5°** |
| 1 | 0 | 11 @ x 0.063-0.123 | 0.7668-0.7695 | 0-? (scattered) |
| 2 | 5 @ x 0.120-0.121 | 5 @ x 0.061-0.062 | 0.7792-0.7793 | **33.4°** |
| 3 | 6 @ x 0.118-0.119 | 5 @ x 0.061-0.062 | 0.7788-0.7789 | **31.9°** |
| 4 | 5 @ x 0.128 | 5 @ x 0.065-0.074 | 0.7772-0.7775 | **26.2°** |
| 5 | **10** (both columns) | 0 | -- | -- |
| 6 | 5 @ x 0.062-0.063 | 5 @ x 0.123-0.124 | 0.7771-0.7772 | **25.6°** |
| 7 | 5 @ x 0.123-0.124 | 5 @ x 0.068 | 0.7773-0.7774 | **26.4°** |

The refused batch is always in the **opposite column** from the counted one, and its height is
always 0.777-0.779.  Inverting the height through the Cookie's own half-extents
(0.025 x 0.0032 x 0.0125, resting on the box floor) gives the tilt:

> **33 of the 41 refused Cookies are leaning 25.6°-33.8°, median 27.3°. The judge allows 15°.**

They are not on the rim: their x is 0.120 while the box's x wall is at 0.152, so they are 32 mm
inside.  They are standing at the right column and the right rows, **tilted about 12° past the
limit**, because there is nothing to stand up straight against.

**Why the second batch cannot stand up, in three numbers.**  The box is 26 mm deep; a Cookie is
25 mm tall; so after one batch the placed Cookies' tops are at 0.7803 -- **5.3 mm above the rim**.
The next batch must descend into a 50 mm slot with **4 mm** of clearance to the outer wall and
**6 mm** to that protruding neighbour.  There is no room to land upright, so it lands leaning.
That is why it is always the *second* batch, whichever column it goes to, and why 85 % of the 2x5
point deficit (33 of 39) is this one event.

**2x10 fails differently, and this is the sharpest new number.**  Its deficit is 51 points, but only
7 of them (14 %) are refused Cookies; the rest are Cookies **never picked up**.  The policy makes
**3.00 releases per episode where the expert makes 4.00** -- it never attempts the fourth batch at
all.  Seed 0 is the clean case: 13 left the source, 13 were counted, 0 were refused, and the 7
missing points are seven Cookies still in the source box.  So 2x5 loses points to *placement* and
2x10 loses them to *throughput*, and the two need different remedies.

**One caveat that limits how far the comparison can be pushed.**  The demonstrations are the
*accepted* set -- the collector discarded every failed episode -- so "the expert never fails" is
survivorship, not a measurement.  Its perfect 0.3 mm off-column figure is a property of a scripted
controller, and the right reading is "the demonstrations contain only upright outcomes", not "the
expert is 5x more precise and that is why it works".

**What this changes.**  The failure is not a release that is too high and not a grasp that misses.
It is a **second batch that cannot seat because the box has 1 mm of headroom**.  That is a
geometry-and-tolerance problem, and the remedies that follow are different from the ones the
recovery plan was built on:

1. **Teach the batch to be wider apart when released** (open the jaws further, or release after a
   small downward settle), so the five Cookies arrive already spanning the column instead of
   collapsing into a lean.  This is a modification of the last 20 frames of each demonstration and
   needs no new collection.
2. **Or accept the lean and make the judge's 15° limit the target**: an upright-tolerance is
   arguably too strict for a box that is 26 mm deep and a Cookie that is 25 mm tall, but changing
   the judge changes the task, so it needs to be an explicit decision rather than a tuning.
3. **2x10 needs the fourth batch attempted at all**, which is a throughput fix (9.56's retry
   behaviour), not a placement fix.

### 9.70 Force-triggered release intervention: built, and the trigger does not separate (11-02)

The hypothesis from 9.69 was that the second batch jams against the Cookies already in the box,
and that letting go as soon as the fingers register the squeeze would let the group settle.  The
intervention was built (`BridgeChunkPolicy._intervene`, `--release-intervention RISE_N`) and the
trigger was measured against the eight recorded rollouts.  **It does not work, and the reason is
that the signal is not there.**

**What was built.**  A wrapper that only ever raises the two gripper channels to 0.49; every other
entry is the policy's own.  That restriction is deliberate -- the point is to test one hypothesis,
and moving the arm too would confound it.

**First version, and why it had to be gated.**  A sustained touch rise with no positional gate
fired **34-40 times per episode**, all of them at the *grasp*: closing the jaws on a row produces
30-70 N of fingertip impulse, an order of magnitude more than any placement.  Gating on "over the
target box and at or below the rim" removes those, and is also the honest statement of what the
trigger means.

**With the gate, the signal that remains cannot separate the two cases.**  Peak sustained rise in
the placement window, per batch:

| seed | batch 1 | batch 2 | batch 2's outcome |
|---|---|---|---|
| 0 | 0.23 | **1.15** | leaned, refused |
| 1 | 0.14 | **15.62** | catastrophic collision |
| 2 | 0.20 | **0.35** | leaned, refused |
| 3 | 1.12 | **0.54** | leaned, refused |
| 4 | 0.62 | **0.49** | leaned, refused |
| 5 | 0.07 | **0.20** | **clean, scored** |
| 6 | 0.23 | **0.38** | leaned, refused |
| 7 | 0.63 | **0.37** | leaned, refused |

A clean placement (seed 5, batch 2) reads **0.20**; four of the six refused batches read 0.35-0.54,
and one reads 0.49 -- *below* seed 3's clean first batch at 1.12.  There is no threshold that
separates them.  The only outlier is seed 1 at 15.62, which is a different event: that is a collision
hard enough to scatter eleven Cookies, not the gentle lean being targeted.

**Run live, it also changed nothing.**  One episode with `--release-intervention 0.8` scored 5/10,
identical to the baseline -- and inspection showed why: that run's policy produced a *different
trajectory* (75 Cookies left in the source, one batch instead of two), so the intervention never
had the state it was written for.  The bridge is stochastic (9.40: 0.086 rad between two samples of
one observation), so a single-episode test of a rare event is not a test at all.

**Why the signal is weak, mechanically.**  The jam is a *geometric* obstruction: the group's outer
face rests on the neighbour with a normal force of order the group's weight distributed over a
contact patch, not a squeeze.  The fingertips read the *gripping* force (2-5 N here), and the jam
modulates it by a few tenths of a newton.  The two quantities are three orders of magnitude apart --
a 5 N grip against a ~0.02 N contact -- so the jam is a rounding error on the channel being read.

**What would carry the signal instead.**  The one that is directly sensitive to the jam is not force
but **motion**: the trace shows the effector descending 4-11 mm while the Cookies stay put, i.e. the
hand-to-Cookie distance growing from 6 mm to 17 mm.  That is a rigid-body measurement, visible in
`observation.eef_pose` plus the camera, and it is exactly what a policy could learn.  Force was the
wrong channel to reach for.

**Status of the code.**  `_intervene` and `--release-intervention` are kept, defaulted off, with
this measurement recorded in the docstring.  They are a working harness for a *motion-based* trigger,
and re-deriving the negative result would be the expensive part.

### 9.71 The motion trigger does not separate either, and the reason is instructive (11-02)

9.70 ended with "force was the wrong channel; the signal is in *motion* -- the effector descends
4-11 mm while the Cookies stay put".  That was measured, on the same eight rollouts, and **it fails
for the same reason**: the quantity is real but it is not exclusive to the failure.

**The measurement.**  For each descent over the target box, count the effector travel during which
the nearest carried Cookie's height did not change by 1 mm:

| unfollowed travel | outcome | batch |
|---|---|---|
| 126.7 mm | FAIL | seed 1, batch 2 (the collision) |
| **28.1 mm** | **OK** | **seed 4, batch 1** |
| 16.6 mm | FAIL | seed 0, batch 2 |
| 14.2 mm | FAIL | seed 7, batch 2 |
| 10.2 mm | FAIL | seed 2, batch 2 |
| 8.3 mm | FAIL | seed 6, batch 2 |
| **8.0 mm** | **OK** | **seed 7, batch 1** |
| **7.7 mm** | **OK** | **seed 2, batch 1** |
| 5.1 mm | FAIL | seed 4, batch 2 |
| 2.2 mm | FAIL | seed 3, batch 2 |
| 0.0 mm (x6) | OK | every other clean batch |

**A clean batch reads 28.1 mm, worse than four of the eight failures**, and two more clean batches
(8.0, 7.7) sit inside the failure range.  No threshold separates them.

**Why, and this is the useful part.**  The measure conflates two events that produce the same
signature in the channels used:

1. **the jam** -- the group is held up by a neighbour while the hand descends;
2. **the release and fall** -- the jaws open, the hand starts to rise, and the Cookies fall the last
   few millimetres on their own.

In a clean placement the second dominates; in a jammed one the first does.  Both read as "the
effector moved and the Cookies did not", and separating them needs a discriminator that knows
whether the jaws are still closed -- which is available (the gripper channel is part of the action),
but which turns a one-line test into a state machine, and the state machine is what
`batch_expert.py` already is.

**Two negatives now stand, and together they close this direction.**  Force cannot see the jam
(5 N grip against 0.02 N contact, 9.70) and motion cannot distinguish it from a release (this
section).  What that leaves is not a better trigger but a different question: the jam is a
*planning* failure -- the second batch is aimed at a column whose neighbour already protrudes 5 mm
above the rim -- and both experiments were attempts to fix a planning error with a reactive
controller.

**So the honest summary of the intervention thread is: a real, measured failure, two plausible
reactive remedies, both falsified on the same eight rollouts, for reasons that are now specific
rather than mysterious.**  The pieces are kept (`_intervene`, the constants, the measurements) so a
future attempt starts from the numbers rather than from the hypothesis.

### 9.72 The 200-frame arm, re-measured with the per-step window: 5.33, i.e. nothing (11-03)

**The claim under test.** 9.69 proposed that the 200-frame arm's regression was an artefact of the
per-chunk history window: 200 frames at 16 steps/chunk needs 3200 control steps to fill, while a
2x5 episode is 1000 steps and a 2x10 episode 1500--3200, so the 200-frame window *never filled* and
the arm measured latent absence rather than window length.  Prediction: with `_advance_history`
called from `act` every step, the arm recovers to about the 10-frame level.

**Re-measured** (`--task 2x5 --episodes 24 --seed-start 0 --max-steps 1000 --history-window
--history-frames 200`, bridge `history_dims=9600`, log `w200_2x5_24.log`):

| arm | n | mean | sd |
|---|---|---|---|
| 10 frames (v9) | 88 | **5.60** | -- |
| 200 frames, per-chunk window (old) | 57 | 5.88 | -- |
| 200 frames, per-step window (this) | 24 | **5.33** | 3.03 |

5.33 vs 5.60 is a difference of 0.27 with a standard error of about 0.68.  Nothing.

**And that means the 2x5 arm could never have answered the question.**  With sd 3.03 the coefficient
of variation is 57%; detecting a half-point difference at 80% power needs roughly 570 episodes per
arm.  Even the old 5.88 is not distinguishable from this 5.33 (0.55 +/- 0.74).  The 2x5 comparison
should never have been read as evidence in either direction, including in 9.69.

**Where the power actually is.**  The 2x10 effect that motivated all of this was 2.17 points, which
at this variance needs n ~ 31 per arm -- so n=24 was adequate there and inadequate here. 2x10 is
also the only task that ever responded to the latent (9.52, paired p=0.0022).  The question is
therefore still open, and it is only open on 2x10.

**Two scoring details worth recording.**  `EpisodeScore.score = cookies_in_target` and
`max_score = required_cookies`, so a 2x5 episode can legitimately print `11/10`: over-delivery into
the target is counted, it just cannot satisfy the exact-count contract.  Two of the 24 seeds did
this (12 and 17); capping them at 10 gives 5.25 and changes nothing.  The `/10` is a denominator,
not a clamp.

### 9.73 The per-chunk window predates every history result, so the whole ablation is confounded (11-04)

**Found while setting up the 2x10 re-measurement.**  `git show b3bacac` (10-27, the commit that
introduced the history channel) has `_history_vector(observation)` *appending* a frame and being
called from inside `if not self._queue:` -- that is, once per 16-step chunk.  The bug fixed on
11-02/11-03 was therefore not introduced after the sweeps; **it was present from the first commit
in which a history arm was ever benchmarked.**

**The magnitude, from the logs rather than by inference.**  2x10 was run at 1500 steps, and the
driver prints `bridge forwards` at the end: `v9nokl_2x10_47.log` says **94 forwards** for 1500
steps.  A 200-frame window advanced once per chunk therefore received 94 of its 200 frames --
**47 % full** -- and the 10-frame window received ten samples spanning 160 steps instead of ten.

| arm | trained window | frames actually delivered at 1500 steps |
|---|---|---|
| v9 (10 frames) | 10 consecutive (0.5 s) | 10 samples over **160 steps (8 s)** |
| v9w200 (200 frames) | 200 consecutive (10 s) | **94** of 200, i.e. 4.7 s |

So every row in 9.52, 9.53 and 9.54 -- including the v9 baseline `13.22 (n=88)` that all of them
are quoted against -- measures a checkpoint being served a time-stretched, under-filled version of
the channel it was trained on.  Nothing has been measured with the window the checkpoints were
trained against.

**This does not retract the results, it re-labels them.**  They are still valid measurements of
*those deployed configurations*, and the 2x10 ordering (10 frames > no history, p = 0.0022) still
holds of them.  What cannot be claimed is any statement about window length per se, and in
particular 9.53's conclusion -- "lengthening the window is not an improvement" -- was drawn from a
200-frame arm that never saw more than 47 % of a window.  The honest status of that conclusion is
*untested*, and 9.69's artefact hypothesis stays alive for 2x10 even though 9.72 killed it for 2x5
(where the arm has no power anyway).

**What is running now.**  2x10, v9w200, `--prompt-style task --source-column 1 --max-steps 1500`,
seeds 0--55, with the per-step window, so 200 frames are delivered from step 200 onward.  Chosen
over v9 first because it is the arm whose conclusion is in question, and because its own number
gives a two-sided read either way: unchanged near 11.05 says the window length barely matters and
9.53 stands; a move to about 13.2 says the arm was measuring under-fill and 9.53 is void.

**The confound that this makes unavoidable:** if the fix matters, it matters for v9 as well, and
tonight's arm cannot then be compared against the stale 13.22.  v9 on 2x10 with the per-step
window, same seeds, is the next night's run and is required before any final claim.

**Two prompt deviations recorded, because both are sentences the checkpoints never saw.**
`--prompt-style short` sends 'transfer 10 cookies into target box', and `a3_clean_v9h`'s task
table contains only the eleven `task`-style sentences (indices 0--9 the 2x10 plan variants,
index 10 the 2x5 two-batch sentence).  Last night's 2x5 arm (9.72) used `short`; the historical
rows it was compared against used `task --source-column 1`.  9.51 measured the 2x5 prompt flag as
a red herring, so this is very unlikely to explain 5.33 vs 5.60, but 9.72's null is now a null
under an unseen instruction and should be read that way.

**Power, so the next two nights are read correctly.**  Paired differences on 2x10 are *not*
correlated by seed: 9.53 reports a paired effect of -2.01 with SE 0.75 at n = 104, which implies
sd_diff = 7.65 against raw sds of 5.27 and 5.76 -- i.e. essentially the independent value
sqrt(5.27^2 + 5.76^2) = 7.81.  Pairing buys nothing on this benchmark, so n is set by the
two-sample variance: SE at n = 56 is 1.04, enough to separate 0 from 2.2 at p about 0.03, and not
enough for anything smaller.

### 9.74 All ablations to date, in one table (11-04)

Collected from the per-seed logs rather than from earlier prose, so the numbers here are the ones
the runs actually printed.  `n` is the count of 2x10 episodes found under that tag; where the
prose quoted a slightly different n (5.60 at n=88 vs 5.65 at n=80 for v9 on 2x5) the difference is
eight seeds that live under an older tag, and it changes nothing.

**Training-side ablations, all on the same data, base and recipe, differing in one flag.**

| ablation | flag | 2x5 | 2x10 | delta vs 10-frame, paired |
|---|---|---|---|---|
| no encoder | `A3_HISTORY_ENCODER=0` | 5.67 +/-0.37 (64) | 10.64 +/-0.74 (64) | **-2.92 +/-0.92**, W16/L39 |
| **10 frames (base)** | `A3_HISTORY_STEPS=10` | 5.65 +/-0.27 (80) | **13.18 +/-0.58 (80)** | -- |
| 200 frames | `A3_HISTORY_STEPS=200` | 5.88 +/-0.35 (57) | 10.96 +/-0.54 (112) | **-2.16 +/-0.80**, W25/L48 |
| progress weight 0 | `A3_HISTORY_PROGRESS_WEIGHT=0` | 5.69 +/-0.48 (32) | 11.38 +/-0.81 (32) | -2.83 +/-1.33, W10/L13 |
| KL weight 0 | `A3_HISTORY_KL_WEIGHT=0` | 6.04 +/-0.38 (48) | 12.95 +/-0.83 (43) | -0.40 +/-1.27, W16/L19 |

Supporting latent scale, `mu_std` read off each arm's `train_*_30000.log` at step 30000:

| arm | base | 200 fr | KL=0 | progress=0 |
|---|---|---|---|---|
| `mu_std` | 0.0529 | 0.0382 | **0.1700** | **0.0017** |

Two readings that survive: removing the progress supervision collapses the latent by 31x and lands
the arm between the base and no-encoder (9.54), and removing KL quadruples the latent for -0.40
points, i.e. buys nothing (9.54 and the encoder-only sweep agreeing at R^2 0.439 vs 0.434).

**Architecture evolution, 8 seeds each, kept only as background** -- every one of these differs
from its neighbour in more than one way, so none of them is an ablation:

| version | 2x5 | 2x10 | what changed |
|---|---|---|---|
| v2 | 5.00 | 3.25 | 2x10 weighted to 50 % of the data |
| v4 | 3.50 | 3.75 | 2x10 filtered to the n=5 plan |
| v5 | 3.88 | 3.12 | two privileged progress channels |
| v6 | 4.50 | 4.50 | history encoder at 3e-06, i.e. no-op |
| v7 | **6.25** | 2.50 | encoder lr to 1e-04 |
| v8 | 2.88 | 5.43 | rebalanced data |
| **v9** | 4.88 | **11.14** | trunk lr to 2.5e-05, four changes at once |

The v6-to-v9 step is the one that mattered, and 9.45 attributes it to the trunk rate rather than to
the module: v6-v8 sit at `mu_std` 0.057-0.062 with R^2 0.51-0.54, v9 at 0.0576 with R^2 0.853.  An
encoder trained on a history of a trunk that is not moving has nothing to read.

**Collection-side ablations, 2x5, different metric (accepted episodes, not score):**

| arm | n | tier A | rate | triggered | failure cost |
|---|---|---|---|---|---|
| baseline | 12 | 5 | 42 % | 8 | 351 / 547 steps |
| + corridor + stall 350 | 12 | 4 | 33 % | 10 | -- |
| + corridor + strict rows | 16 | 4 | 25 % | 8 | 254 or **0** steps |
| after the history-channel fix | 16 | **2** | 13 % | -- | both `policy_only`, **0 expert frames** |

2x10 counterpart: 1 of 12, batches completed 7/9 -> 3/9 -> 1/9.

The two reactive interventions are both negatives and both specific: force cannot see the jam
(5 N grip against 0.02 N contact, 9.70) and displacement cannot distinguish a jam from the release
and fall that follows it (9.71).

**The caveat that governs every training-side row.**  All of them except the two running tonight
were measured with the per-chunk history window (9.73), so "200 frames is worse than 10" is a
statement about two *mis-served* configurations, not about window length.  The two arms in flight
are the first measurement of either window as trained.

### 9.75 Strict success, measured across every arm, and why "in the box" is not "success" (11-04)

The question was how often each arm *actually* satisfies the contract, not how many Cookies it gets
in the target.  Read off the `Status:` field rather than the score, and the answer is much harsher
than any mean suggests.

**2x10 (20 in target, 60 remain in source, all released, 20-step hold):**

| arm | n | strict | 95 % CI | full basket (score 20) | within one (>=19) |
|---|---|---|---|---|---|
| **KL=0** | 43 | **6 (14.0 %)** | 6.6--27.3 | 7 (16.3 %) | 8 (18.6 %) |
| **10 fr (base)** | 80 | 5 (6.2 %) | 2.7--13.8 | 9 (11.2 %) | 18 (22.5 %) |
| 200 fr | 112 | 4 (3.6 %) | 1.4--8.8 | 5 (4.5 %) | 14 (12.5 %) |
| no encoder | 64 | 3 (4.7 %) | 1.6--12.9 | 5 (7.8 %) | 7 (10.9 %) |
| progress=0 | 32 | 1 (3.1 %) | 0.6--15.7 | 1 (3.1 %) | 3 (9.4 %) |
| 10 fr per-step (running) | 40 | 0 (0.0 %) | 0--8.8 | 2 (5.0 %) | 8 (20.0 %) |
| 200 fr per-step (running) | 46 | 2 (4.3 %) | 1.2--14.5 | 3 (6.5 %) | 6 (13.0 %) |

**2x5 (exact ten slots, 70 remain, 20-step hold):**

| arm | n | strict | 95 % CI | full basket (>=10) |
|---|---|---|---|---|
| KL=0 | 48 | 1 (2.1 %) | 0.4--10.9 | 7 (14.6 %) |
| 10 fr (base) | 80 | 5 (6.2 %) | 2.7--13.8 | 12 (15.0 %) |
| 200 fr | 57 | 3 (5.3 %) | 1.8--14.4 | 7 (12.3 %) |
| no encoder | 64 | 1 (1.6 %) | 0.3--8.3 | 11 (17.2 %) |
| progress=0 | 32 | 1 (3.1 %) | 0.6--15.7 | 5 (15.6 %) |

Pooled over every arm with n >= 30: **2x10 strict 21/416 = 5.0 %**, 2x5 strict 11/281 = 3.9 %.

**KL=0 is the best 2x10 arm on this metric and the difference is not significant.**  6/43 against
5/80 is z = +1.43, p = 0.15; the full-basket counts are 7/43 against 9/80, p = 0.43.  On 2x5 the
same arm is the *worst* on strict (1/48) while its full-basket count is identical to base
(14.6 % against 15.0 %, p = 0.95).  So the honest reading is that KL=0 moves nothing measurable,
and the 14 % is the upper end of what n = 43 can resolve, not evidence.

**The two contracts are not equally hard, and that explains the whole shape of this table.**

| | 2x5 | 2x10 |
|---|---|---|
| `require_exact_slots` | **True** | False |
| `require_released` | False | **True** |
| extra clauses | every one of the 10 slots occupied by a one-to-one match, and the collection's bounding box within 30 mm of **all four** inner walls | none beyond the count |

So 2x5 asks for *slot-accurate placement* and 2x10 asks only for *twenty Cookies released on the
floor of the box*.  The measured consequences:

| | full basket | of which FAIL | why |
|---|---|---|---|
| 2x10 | 32 | 11 (34 %) | counts satisfied, but the 20-step hold never completed |
| 2x5 | 44 | **33 (75 %)** | counts satisfied, but the ten slots were not all occupied |

Confirmed on named episodes: `v9s_2x10_39/42/52` all report `cookies_in_target=20,
cookies_in_source=60` -- the count predicate is satisfied exactly -- and all three still end
`max_steps_exceeded`.  Since `success_hold_steps = 20`, those episodes put the twentieth Cookie in
place inside the last second of a 1500-step budget.  They are not placement failures, they are
**settling-time failures**, and 11 of 32 full baskets are of that kind.

The 2x5 gap is the opposite: 33 of 44 full baskets fail because one of the ten slots is empty even
though ten Cookies are inside, which is the same 27-degree, few-millimetre-offset population 9.69
characterised.  "In the box" and "in its slot" differ by the placement accuracy the policy does not
have.

**What this means for a reward.**  Strict success at 4--5 % is too sparse to learn from, but the
gap between the two events is large and structured: on 2x10 a third of the full baskets are one
second of settling away, and on 2x5 three quarters are one slot away.  A shaped reward should
therefore score *per-Cookie slot occupancy and settle time* rather than the terminal contract --
which is also what makes the dense signal available on every episode instead of on 5 % of them.

### 9.76 Every benchmark episode fed the encoder a dead `a_previous` channel (11-04)

**Found while writing the RECAP collector**, by asking which code path calls
`BridgeChunkPolicy.observe_step`.

The history window is 48 floats per frame: `[q(16), q_dot(16), a_previous(16)]`.  The
`a_previous` entry is written by `_advance_history` from `self._applied_action`, and
`_applied_action` is set in exactly one place -- the `observe_step` hook.  Two callers exist
in this repo:

| caller | calls `observe_step` | used by |
|---|---|---|
| `src/a3_dual_arm_sim/runner.py:216` | **yes** | `collect_graft.py`, this repo's own collection |
| main's `workflows/episode_execution.py` | **no** | every `bench_pi05_on_main.py` run |

So on the benchmark path `_applied_action` stays `None`, `_advance_history` takes its
zeros branch, and the encoder is fed a window whose third channel is sixteen zeros for the
whole episode -- while `a3_clean_v9h` trained it on real applied actions
(`add_history_window.py` shifts the recorded `action` column by one row).  `test_collect.py`
pins both directions: filled when the hook is called, all zeros when it is not.

**What this does and does not invalidate.**  It does not overturn any arm-to-arm
comparison: every arm measured through the benchmark saw the same dead channel, so the
ordering -- 10 frames beating no encoder on 2x10, 200 frames landing near no-encoder, tonight's
per-step arms agreeing at about 12.0 -- was obtained under one consistent handicap.  What it
does invalidate is any claim about the *absolute* value of the channel, including the
natural reading that "the latent is worth 2.5 Cookies on 2x10".  The true figure is
unknown and, if anything, larger: the checkpoint is being served an input it never saw.

**Why it stayed invisible for a week.**  The window has the right width, two of its three
channels are correct, and nothing errors.  A checkpoint whose encoder mostly reads `q` and
`q_dot` degrades gracefully rather than failing, so the symptom is a slightly worse score
that nobody would attribute to a specific channel.  The collector written today computes the
same window through `observe_step` and shows `a_previous` non-zero on 59.5 % of entries,
which is what surfaced it.

**The fix is in the collector, not in main.**  `collect_rollouts.py` drives its own loop
precisely so it can call `observe_step` after every step; patching main's runner would have
been the alternative, and it is the wrong layer -- `observe_step` is this repo's contract,
not main's.  Rollout data written from now on carries the real channel, so RECAP's value
model and the next round of training see the input as trained.

**One consequence to keep in mind when reading §9.52-§9.75.**  The v9 and w200 benchmarks
that produced them were run with this channel dead.  A re-measurement with the collector's
loop is the honest way to get absolute numbers, and it is cheap -- the same 56 seeds, the
same bridge, only `observe_step` added.

### 9.77 `a_previous` is worth about +2 Cookies, and fixing the window alone was a regression (11-04)

Follow-up to 9.76.  The collector from §11 was used to re-run seeds 0--11 of v9 through a loop
that calls `observe_step`, so for the first time the encoder got a history window with the real
applied action in its third channel.  Everything else is identical: same bridge, same checkpoint,
same `execute=16`, same 1500 steps, same instructions.

| configuration | n | mean | sd |
|---|---|---|---|
| 10 fr, per-step window, **`a_previous` = 0** (the benchmark path) | 12 | 11.17 | 6.90 |
| 10 fr, per-step window, **`a_previous` correct** (the collector) | 12 | **13.25** | 5.12 |

Paired, same seeds: **+2.08 +/- 2.27, z = 0.92, p = 0.36**, 6 wins / 5 losses / 1 tie.

**Not significant, and it should not be reported as if it were.**  What makes it worth
recording anyway is the effect size relative to everything else measured this week: 2.08 is
larger than the -2.16 that the whole 200-frame investigation was about, and it comes from a
channel that was silently zero.  The variance also moved in the direction a real effect
predicts (sd 6.90 -> 5.12); noise usually does not narrow a distribution.

**The uncomfortable part: fixing the window by itself appears to have been a regression.**

| 2x10, v9, all with `a_previous` = 0 unless noted | n | mean |
|---|---|---|
| 10 fr, per-chunk window (the pre-11-02 state) | 80 | **13.18** |
| 10 fr, per-step window (11-02 "fix") | 46 | 11.74 |
| 200 fr, per-step window | 51 | 11.94 |
| 10 fr, per-step, **`a_previous` correct** | 12 | **13.25** |

Under the old per-chunk behaviour the window advanced once per 16-step chunk, so its ten
entries spanned **160 steps (8 s)** instead of ten consecutive steps (0.5 s).  That was called
a bug, and as a *sampling rate* it is one -- but as a *horizon* it was the only long-range view
of `q` and `q_dot` the model ever received.  Fixing it to the training definition shortened the
horizon by 16x, and the 10-frame arm lost about 1.4 Cookies doing so.  Adding `a_previous` back
recovered that and returned the arm to 13.2, which is where the per-chunk version already was.

**So the two bugs partly cancelled, and neither arm has ever been measured as trained.**  The
checkpoint was built on 10 consecutive frames *with* real applied actions; the benchmark has
neither, in two different ways depending on when it ran.

**What this does not change.**  Every arm-vs-arm comparison inside one measurement era still
holds, because both sides carried the same handicap: 10 frames beating no-encoder on 2x10, the
200-frame arm landing between them, and tonight's per-step pair agreeing at 11.74 / 11.94.  The
*absolute* numbers are all depressed by an unknown amount, and 9.72's conclusion -- that the
2x5 200-frame arm has no power -- is unaffected.

**What it does change.**  9.73 concluded that "the window artefact cannot explain the v9-vs-w200
gap".  That still holds (both per-step arms had `a_previous` = 0).  But 9.73's *premise* -- that
the per-step arms are the correct configuration -- is now doubtful: they are correct in sampling
rate and still wrong in content, and the content error costs more than the sampling-rate error
did.  The 200-frame arm has never been run with `a_previous` either, and its 10-second horizon
may interact with the finding above in the opposite direction.

**Next measurement, and it is cheap.**  Re-run both arms for 56 seeds through the collector's
loop.  At 10 workers that is about 1.5 h per arm, and it gives the first numbers in this repo
measured on the distribution the checkpoints were trained on.  Until then the honest baseline for
RECAP is *unknown* rather than 13.18.

### 9.78 `cookies_in_target` is not monotone, so the reward has to be the running maximum (11-04)

The plan (§2.3) defined RECAP's reward as the frame-to-frame difference of
`cookies_in_target`.  The first 12 collected rollouts show that definition is unusable.

**Every episode's count falls.**  100 % of them, 18.5 times per episode on average.  The
excursions are almost all flicker rather than losses:

| length of the run below the running maximum | share |
|---|---|
| 1 frame | **80.9 %** |
| <= 2 frames | 89.3 % |
| <= 5 frames | 94.9 % |
| <= 50 frames | 97.2 % |

A Cookie being regrasped, nudged, or re-settled inside the box leaves the containment
region for a frame -- `_cookie_inside_target` needs the footprint inside *and* the bottom
within 4 mm of the floor -- and comes straight back.  The two long excursions (up to 699
frames) are the real ones.

**What the raw difference would have taught the value function.**  Over 12 episodes the
raw difference gives **294 positive and 222 negative rewards**: 43 % of the non-zero signal
is a false "this made things worse", aimed at exactly the moments the policy is working
inside the box.  RECAP's advantage is `A_t = r_t + V(t+1) - V(t)`, so those 222 frames
would have been labelled low-advantage -- and they are the placement frames, which are the
behaviour the whole method is supposed to strengthen.

**The fix is `maximum.accumulate`.**  Reward becomes the change in the running maximum:
**73 positive, 0 negative** on the same data.  It is potential-based, so the episode's
cumulative reward equals the peak with no double counting.

**The cost, stated with its number.**  The running maximum ends at the episode's *peak*,
not its final count, so on the 3 of 12 episodes that lost a Cookie for good it
overestimates: 6 peaks against 159 finals summed over 12 episodes is 4.4 % high.  Real
losses are rare and small compared with the flicker they would otherwise be buried in, and
`value_target` is therefore also the peak -- a value head trained on the final count while
the reward spoke about peaks would be fitting two different functions.

**A cheaper alternative that was considered and not needed.**  Debouncing per Cookie, with
the same five-frame rule the progress channel already uses (`add_progress_channels.py`),
would keep real losses and remove 94.9 % of the flicker.  It needs the per-Cookie mask
(`cookies_in_target_mask`) in the trace, which this collector does not currently store, and
it introduces a five-frame latency into the label.  Running-max needs neither and removes
100 % of the false negatives, so it is what `build_value_labels.py` implements.  If the 4.4 %
overestimate ever matters, the mask is one field away.

**The invariant is now enforced.**  `--check` fails on any negative reward, which is the
check that would have caught this definition before a single training step.

### 9.79 Scope of the dead `a_previous` channel: proven, and narrower than it first looks (11-04)

9.76 asserted the channel was dead on the benchmark path from static reading.  This section
records the empirical proof and then narrows the blast radius, because "every number is
contaminated" is both too strong and too vague to act on.

**Proof, from a live 48-step benchmark episode** (v9 bridge, `history_window=True`,
96 forward passes' worth of steps, spying on the hook):

```
run_episode 来自: a3_dual_arm_sim.workflows.benchmark
episode 48 步
observe_step 调用次数: 0
_applied_action is None: True
a_previous 通道非零: 0.0%
```

Not "rarely called" -- zero times in 48 steps, and the channel is 0.0 % non-zero at the end.
The one occurrence of the string `observe_step` in `bench_pi05_on_main.py` is the method
*definition* (line 379), not a call site.

**Why: main has no hook.**  `grep -rn observe_step /tmp/a3-main/src/` returns nothing.  The
hook is this repo's contract (`src/a3_dual_arm_sim/runner.py:216`), and the benchmark driver
imports main's `make_cookie_benchmark`, so its episodes run main's
`workflows/episode_execution.run_episode`, which has no such call.

**So the split is by driver, not by arm:**

| driver | calls `observe_step` | what ran through it |
|---|---|---|
| `bench_pi05_on_main.py` → main's `episode_execution` | **no** | every ablation sweep (§9.45-§9.75): v9s, v9w200, v9nokl, v9np, v9nhs, tonight's per-step pair |
| `src/a3_dual_arm_sim/runner.py` | **yes** | `collect_graft.py` (all DAgger data), `probe_expert_dwell.py`, `probe_error_tolerance.py`, `run_expert_gate.py` |
| `scripts/recap/collect_rollouts.py` | **yes** (own loop, same call) | tonight's 12-episode pilot and all RECAP data |

**Two rows are exempt for a different reason: the channel does not exist for them.**

* v1-v5 took no history input at all.
* `v9nhs` (`A3_HISTORY_ENCODER=0`) has no encoder, so nothing reads the channel, and
  `run_seed_sweep.sh` passes `--history-window` only for arms whose `history` field is 1.

That exemption rescues the most important comparison in the sequence.  §9.52's headline --
v9 beats no-encoder by 2.92 on 2x10 -- compares a checkpoint served a two-thirds-correct
window against one with no window at all.  **That result stands**: a history encoder reading
`q` and `q_dot` while being blinded to `a_previous` was still worth about 2.5 Cookies over
having no encoder.  What 9.76 doubted, and what is now unknown, is only the *additional*
value of the third channel.

**And it makes the §9.77 pilot's +2.08 the missing term rather than a contradiction.**  If
`q + q_dot` alone is worth +2.5 over nothing, and adding the real `a_previous` is worth another
+2.1 on 12 paired seeds, the two are additive and both plausible: the window has three
channels and the arms measured so far have only ever had two of them working.

**Practical consequence.**  The ablation ordering in §9.74 is intact and needs no re-run for
internal validity.  What needs a re-run is any claim about the *absolute* level of v9 on 2x10,
and any claim that compares a history arm against a non-history arm at face value -- because
the history arms were all handicapped and the non-history arms were not.  §9.52's paired
+2.92 is therefore a **lower bound**.

### 9.80 Does the dead `a_previous` channel threaten the RL direction? Measured, and no (11-04)

The question was whether the channel bug biases the strict-success estimate enough to steer
RECAP the wrong way, and whether more seeds would settle it.  The answer has three parts and
only the last one is a real risk.

#### 1. The model is not contaminated; only the ruler is

The decisive check, run against the training data itself:

| check | result |
|---|---|
| `a_previous` non-zero share on `a3_clean_v9h` | **100 %**, one sampled frame |
| same, over 200 randomly sampled frames | mean 99.8 %, median 100 % |
| frames with an all-zero channel | **0 of 200** |

So v9 trained on a live third channel.  The bug lives in main's `episode_execution`, which
only the benchmark driver uses (`collect_graft.py` goes through this repo's `runner.py`,
which calls the hook).  Consequence: **every number the RL pipeline will consume is clean**
-- the 400-episode expert set, the DAgger data, and every rollout the new collector writes.
The value model cannot inherit the bias.

#### 2. A few dozen seeds cannot resolve strict success, and more seeds would not help anyway

Baseline: v9 2x10 strict = 5/80 = 6.2 %, Wilson 95 % **2.7–13.8 %** -- already a five-fold
range before any new measurement.

| n per arm | true rate 10 % vs 6.2 % | true rate 15 % vs 6.2 % |
|---|---|---|
| 30 | z 0.53, p 0.59 | z 1.11, p 0.27 |
| 40 | z 0.62, p 0.54 | z 1.28, p 0.20 |
| 56 | z 0.73, p 0.47 | z 1.52, p 0.13 |
| 100 | z 0.97, p 0.33 | **z 2.03, p 0.04** |
| 430 | **z 2.02, p 0.04** | z 4.21, p < 0.001 |

80 % power needs **191 seeds/arm** if the true rate is 15 %, **390** if 12 %, **829** if 10 %.

**And the deeper point: seeds shrink variance, not bias.**  Running 30 more seeds on the
biased harness yields a tighter interval around a wrong centre -- a number that looks
authoritative and is not.  A wide honest interval is safer than a narrow biased one.

#### 3. The pilot gives no evidence that the bug depresses strict success at all

Same 12 seeds, benchmark (bug) against collector (fixed):

| | bug | fixed |
|---|---|---|
| episodes reaching 20 | **0** | **0** |
| episodes reaching >= 19 | 3 | 2 |
| best score | 19 | 19 |

The +2.08 mean gain did not lift the ceiling.  Breaking the 12 seeds down by their *bug-era*
score gives +9.6 on the sub-10 group, -3.8 on 10-17 and -2.7 on >= 18 -- a pattern that looks
like mechanism and is not.  That is regression to the mean: the grouping variable shares a
noise term with the delta.  A simulation of the null (true effect +2.08, per-arm noise sd 5)
reproduces the shape by itself, expecting +5.2 / +0.8 / -3.3 for those three groups.  Nothing
can be inferred from a single-measurement split, and the honest reading is **the pilot is
consistent with any effect on the ceiling between strongly positive and mildly negative**.

#### 4. The one mechanism that would actually mislead the RL direction

Not the bias -- **switching rulers between arms**.  If the RECAP arm is evaluated with the
collector loop (+2.08 relative to the old harness) while the baseline stays on the old
benchmark, the comparison manufactures a phantom gain of about **+2 Cookies**, which is the
same size as the effect the whole method is supposed to produce.  A real +2 and a phantom +2
are then indistinguishable.

**Rule, therefore: one harness for every arm in a comparison.**  The collector loop is the
right choice -- it is the only path that feeds the encoder what the checkpoint was trained on
-- and the old benchmark numbers simply stop being comparable once it is adopted.

#### 5. What to use as the RL metric

| metric | power at n = 56 | verdict |
|---|---|---|
| mean score | detects +2.08 at p = 0.045 | **primary** |
| peak count (running max) | similar, and it is the value target | **primary** |
| strict success | needs 191–829 seeds | long-term only |

This is not only a power argument.  9.75 showed strict success is a *compound* event: 34 % of
full 2x10 baskets fail on the 20-step hold, 75 % of full 2x5 baskets fail on slot occupancy.
Optimising a metric that mixes placement accuracy, settling time and the exact-count contract
gives the value function an ambiguous target.  The peak count is what the value head predicts
anyway, so training and reporting agree.

**Summary.**  The bug does not corrupt the RL data, cannot be resolved as a strict-success
question by any affordable sample size, and has no evidence behind the claim that it suppresses
strict success.  Its only real danger is inconsistency between arms, which is a process rule
rather than a modelling one.

### 9.81 The history-module bugs, fixed (11-04)

Four defects were found while planning RECAP's code landing.  All four are now closed.

**1. `_a3_history_latent` could be stale (`train_pi05_launcher.py:445`).**
`embed_prefix` returns early when the batch carries no window, and it used to return without
clearing the cached latent.  `policy_forward` reads that cache to build the progress loss, so
a windowless batch was supervised against the *previous* batch's latent -- right shape, finite
value, loss falls, nothing errors.  Unreachable today (every batch has a window) and
guaranteed to fire the moment RECAP mixes datasets.  Fixed by clearing the cache on the early
return.

**2. `reduction != "mean"` silently dropped both auxiliary terms (`:492`).**
`lerobot_train.py` calls `policy.forward(batch, reduction="none")` whenever RA-BC sample
weighting is on, and the old guard `if ... reduction != "mean": return` skipped the progress
and KL terms entirely.  A run that loses its only supervision on the latent is
indistinguishable from a run where the module is not learning.  Now computed per sample and
reduced three ways, with the `"mean"` path kept numerically identical -- verified against the
old expression on the real tensors, **KL diff 0.0**, so `A3_HISTORY_KL_WEIGHT` still means what
it meant.

**3. `history_encoder.out` was dead weight (`history_encoder.py:169`).**
`nn.Linear(128, 2048)` = **264,192 parameters** that no forward pass ever called.  The claimed
"630,018 new parameters" was therefore overstated by 42 %: the module was really 365,826, of
which the encoder was 101,634 and `history_proj` 264,192.  Every checkpoint carried the dead
tensor.  Removed; measured encoder parameters now **101,634**, and old checkpoints still load
clean (`strict=False` drops the two unexpected keys with a warning).

**4. Two packages named `a3_dual_arm_sim` cannot share a pytest process.**
This repo installs `src/a3_dual_arm_sim/` (flat, no `data/`); main's tree at
`/tmp/a3-main/src/a3_dual_arm_sim/` has `data/`, `workflows/`, `tasks/`.  Python keys modules
by name, so `pytest tests scripts/recap/tests` fails with

```
ModuleNotFoundError: No module named 'a3_dual_arm_sim.data'
```

which reads as a missing file rather than as a collision.  Both suites pass alone.

**This one took three attempts, and the first two are the instructive part.**

| attempt | what it did | outcome |
|---|---|---|
| 1 | conftest purges `sys.modules`, prepends main's path | **broke all 29 tests in `tests/`** |
| 2 | conftest only *checks* `a3_dual_arm_sim.data` is resolvable, refuses to start otherwise | **broke the standalone recap run** |
| 3 | conftest prepends main's path, no purge; guard fires only if the *wrong* package is already loaded | correct |

Attempt 1 failed because pytest loads conftest files **before** importing test modules, so the
purge ran before `tests/` imported the package it needs.  Attempt 2 failed for the mirror-image
reason: the path insert that makes `a3_dual_arm_sim.data` resolvable happens at import time
*inside the test module*, which is after conftest, so a check there always fails.

The resolution is that the two operations are not symmetric: **inserting a path is harmless,
deleting cached modules is not.**  And a conftest only applies to its own directory, so
prepending main's path here cannot disturb `tests/` -- that suite never loads this file.  The
guard that remains distinguishes by module *origin* rather than by presence, because main's
package may legitimately be loaded already.

`scripts/recap/run_tests.sh` runs the two suites in sequence.  Its first version grouped
`tests/test_history_encoder_patches.py` with the recap tests, which fails for the same reason
-- that file lives in `tests/` and imports this repo's package.  The split is by **which
package each suite needs**, not by what the test is about.

**Test status, with attribution checked rather than assumed:**

| suite | result | note |
|---|---|---|
| `scripts/recap/tests` | **15 passed** | includes the `a_previous` both-directions pin and the progress-tracker drift guard |
| `tests/test_history_encoder_patches.py` | **5 passed** | |
| `tests` (full) | 517 passed, **7 failed** | |

The 7 failures are pre-existing and unrelated.  Verified by stashing the six touched files
and re-running: the same 6 cookie-expert tests fail on the untouched tree, and the seventh
(`cosmos_policy_aloha.yaml`) is a missing-transport environment issue.  My own changes touch
`_adopt_current_state`/`__init__` in `batch_expert.py` and nothing on those tests' paths.

### 9.82 The collector now records the expert too, and merging parts is verified pixel-wise (11-04)

**The collector gained `--policy expert`.**  RECAP's positive class is supposed to be
genuinely-better behaviour, and `v9`'s own rollouts supply it only 6 % of the time.  The
expert supplies it always, so both now go through one loop, one schema and one trace format --
which is what makes them loadable by one trainer without a branch.

Smoke test, 2 seeds each, 2x10:

| source | seed | score | steps | strict |
|---|---|---|---|---|
| expert | 910 | **20/20** | 1091 | **True** |
| expert | 911 | **20/20** | 1074 | **True** |
| v9 rollout | 900 | 0/20 | 200 (budget capped) | False |
| v9 rollout | 901 | 0/20 | 200 (budget capped) | False |

Three things had to change for the expert to work, and each was a real failure first:

1. **The adapter is constructed, not named.**  `make_policy_adapter("variable_grasp")` returns
   the string; the construction happens in `bench._episode_policy`.  Passing the spec straight
   through gave `AttributeError: 'str' object has no attribute 'act'`.
2. **One adapter per episode.**  `bind_env` builds the controller *around* the environment, and
   `run_episode` closes its environment.  Reusing an adapter drives a closed scene, so the
   parameter is a factory called per seed.
3. **The expert has no history channel, and the schema has one anyway.**  Writing zeros keeps
   both sources in one dataset; raising instead would have forced two layouts.  A zero window
   is the honest representation of "this policy has no history", not a fabricated trajectory --
   and it is distinguishable from a rollout's, which measured 100 % non-zero from frame 1.

Verified in the merged mix: episodes 0-1 (expert) show `a_previous` **0.0 %** non-zero,
episodes 2-3 (rollout) show **100 %**, and the labels record 2 successes and 2 failures.

**`merge_parts.py` and `verify_merge.py` are written and validated.**  The merge is more than a
concatenation because four things are per-part:

| per-part quantity | what goes wrong if copied |
|---|---|
| `task_index` | **silently wrong instructions** -- `part_00`'s index 0 is "Grasp 4", `part_01`'s index 0 is "Grasp 7" |
| `episode_index`, `index` | duplicated row numbers, and `dataset_from/to` no longer bound the episodes |
| video `file_index` | every episode after the first shows the wrong video |
| `stats` | quantiles averaged across parts are not the union's quantiles |

On a real 6-part merge the task table collapsed 9 local indices to **7 distinct sentences**,
which is the check that matters: equal sentences map to one global index, unequal ones never
share one.

**The verification is pixel-level, not structural.**  A merge that copies every part into
`file-000.mp4` and rewrites only `file_index` passes every count and boundary check while
showing every episode the wrong footage.  So `verify_merge.py` compares frame 0 of each merged
episode against frame 0 of its source part:

```
part_00: merged row     0 vs part row 0 -> max|diff|=0.000000 OK
part_01: merged row  3000 vs part row 0 -> max|diff|=0.000000 OK
part_02: merged row  6000 vs part row 0 -> max|diff|=0.000000 OK
part_03: merged row  9000 vs part row 0 -> max|diff|=0.000000 OK
part_04: merged row 12000 vs part row 0 -> max|diff|=0.000000 OK
part_05: merged row 15000 vs part row 0 -> max|diff|=0.000000 OK
```

plus a live-window check at two interior samples (100 % non-zero) so a merge cannot quietly
zero the history column.

**One environment note, because it will bite again.**  Loading any of these datasets needs
`video_backend="pyav"`.  The default picks torchcodec, which fails with
`libavutil.so.58: cannot open shared object file` against this container's FFmpeg -- the same
reason `train_pi05_a3.sh` passes `--dataset.video_backend=pyav`.  Without it the failure is a
wall of FFmpeg version noise that says nothing about the backend.

### 9.83 RECAP stages 2 and 3 built, and the global advantage threshold is backwards (11-05)

Collection finished: 300 rollouts + 100 expert episodes, 552,789 frames, and the value head
cleared its gate.  Two things in this section are results and one is a defect worth its own
paragraph.

**The data.**

| set | episodes | frames | strict success | mean score |
|---|---|---|---|---|
| rollout (v9, 1500-step budget) | 300 | 450,000 | 11 (3.7 %) | 12.98 |
| expert (2x10 variable_grasp) | 100 | 102,789 | 94 (94.0 %) | 19.50 |

The expert's 94 % is the positive class RECAP needs; v9 supplies 3.7 %.  Note that the
rollout mean is **12.98 against the benchmark's 11.74**, i.e. +1.24 with p = 0.16 -- the
direction of §9.77's `a_previous` pilot, still not significant because the benchmark arm is
n = 46 and carries its own 0.84 standard error.

**Three merge defects surfaced on real data, all silent if unhandled.**  §9.82 validated the
merge on six parts of one camera layout; 300 episodes found the rest:

1. **A part's video is not one file per camera.**  `left_wrist` sharded at episode 28 (two
   files, 28 + 10) while `front` and `right_wrist` did not, because LeRobot rolls each camera
   to `file-001.mp4` independently when it hits its size cap.  The first merge refused; a merge
   that copied only `file-000` would have silently truncated ten episodes' video to the wrong
   frames.
2. **Two roots both name their directories `part_00..part_07`.**  `build_value_labels.py`
   keyed on `part.name`, so the expert parts overwrote the rollout parts in the lookup and 12
   `(seed, episode_index)` pairs read the wrong root's trace.  Fixed by keying on
   `"{root}/{part}"`; the part count went 8 -> the correct 14.
3. **Both tracks used seeds 1000+**, because a seed picks the *scene* and the policy is what
   differs.  The merged trace directories collided on every seed; each source part now gets its
   own parent directory.

**The value model.**  Encoder frozen, only a 3.4 M-parameter head trained, on cached features
(SigLIP patch-embedding mean per camera + masked language-embedding mean, 8192 wide, stride 5
-> 110,558 rows).

| | |
|---|---|
| holdout Spearman (gate: 0.6) | **0.699** |
| early-quarter Spearman | **0.481** (5,489 frames) |
| predicted mean vs actual | 14.64 vs 14.66 Cookies |

The early-quarter number is the one that matters.  The overall figure is generous by
construction -- late in an episode the current count nearly determines the peak, so a head that
merely counts passes 0.6.  0.48 on the first quarter says it carries something a counter does
not.  Leakage checks: `observation.progress` is hard-zeroed in the inputs (it is a privileged
simulator quantity that would not exist on a robot) and the cache layout contains no progress
channel.

**The defect: a global 30th-percentile threshold labels successful episodes *worse*.**

The obvious implementation -- and the one both upstream projects use -- takes one percentile
over all frames.  Measured here:

| rule | positive rate, successful episodes | positive rate, failed episodes | verdict |
|---|---|---|---|
| global percentile | **23.3 %** | **31.7 %** | **backwards** |
| per-episode percentile | 74.2 % | 32.0 % | correct |

The per-episode *mean* advantage was already right (+0.98 successful vs +0.44 failed); what
broke was the frame-level rate.  The reason is episode length: a successful episode spends most
of its frames in "everything is placed, nothing new" where `A_t ~ 0`, and a global cut puts
those frames in the negative class.  Conditioning a policy on that teaches **"frames spent
carrying are good"** -- a property of the trajectory's *phase*, not its quality.

Taking the top 30 % *within each episode* removes the length dependence: every episode
contributes the same fraction of positive frames regardless of whether it finished early.
`compute_advantages.py` now writes both columns (`positive_global` retained for inspection,
`positive` used for training) and **prints the success-vs-failure rate check on every run**, so
this cannot regress unnoticed.

### 9.84 RECAP stage 4, twice: the reward was too coarse, then the sentence turned out inert (11-06)

Stage 4 was run twice.  The first run (4A) lost 5.4 Cookies against v9; the second (4B) fixed
the reward and recovered to within 3.1, but **the advantage condition itself measured zero in
both runs**.  Both conclusions are load-bearing, and the second one is the more expensive fact.

**All four arms, seeds 300+ on one harness (56 seeds, 18 for the sanity arm):**

| arm | weights | suffix | strict | mean | sem |
|---|---|---|---|---|---|
| `r4b_pos` | 4B | `" Advantage: positive"` | 5.4 % | 9.84 | 0.74 |
| `r4b_nos` | 4B | none | 3.6 % | 9.00 | 0.79 |
| `r4b_neg` | 4B | `" Advantage: negative"` | 0.0 % | 8.61 | 1.21 |
| `v9` | v9 | none | 12.5 % | **12.96** | 0.75 |

Paired, common seeds only:

| comparison | delta | n | W/L |
|---|---|---|---|
| `r4b_pos − r4b_nos` | **+0.84 ± 0.88** | 56 | 30/21 |
| `r4b_pos − r4b_neg` | +0.39 ± 1.80 | 18 | 10/8 |
| `r4b_pos − v9` | −3.12 ± 0.97 | 56 | 15/39 |
| `v9 − r4b_nos` | +3.96 ± 1.07 | 56 | 34/14 |

**The conditioning is inert, and the three-condition comparison is what shows it.**  On the 18
seeds all three arms share: positive 9.00, no-suffix 8.61, negative 8.61.  For conditioning to be
working the order has to be positive > none > negative; instead *none and negative are identical
to two decimal places* (paired `nos − neg = +0.00 ± 1.59`, W6/L8).  The model is not reading the
tail of the instruction at all.  `build_task_variants.py` is verified to write the suffix --
every episode's `instruction` field ends with it -- so this is the policy ignoring text it was
fine-tuned on, not a plumbing bug.

This is exactly the check `run_eval_suite.sh` was written to perform, and it is why the negative
arm was worth 18 seeds: without it, `+0.84 ± 0.88` against no-suffix invites the reading "small
positive effect, need more seeds", when the honest reading is "the mechanism does nothing."

**Why the reward change still mattered.**  4A→4B isolated the label change: same base, same
steps, same lr, only `task_index` differs.  Mean went 8.32 → 9.84 (+1.52), and against v9 the
gap narrowed from −5.41 ± 1.09 to −3.12 ± 0.97.  So a reward that can see slot precision and wall
coverage is worth about 1.5 Cookies even when the conditioning built on it is inert -- the
labelled data trains better as plain BC, which is a real (if unglamorous) result.

**The layered reward.**  Counting Cookies in the box cannot separate the task's two halves:
31.2 % of the 400 collected episodes put all 20 in the box, but only 19.5 % also seat all 20 in
valid slots and 70.5 % touch all four walls.  The gap `in_box − slotted` averages **+1.04 and is
non-zero in 62.5 % of episodes (max 5)**.  `layered_reward.py` replaces the count with

$$\Phi(s) = 1.0\,n_{\text{box}} + 0.5\,n_{\text{slot}} + 1.0\,\mathbb{1}_{\text{walls}}, \qquad \max\Phi = 31.0$$

which raises episode-level separation from AUC 0.959 / ρ +0.713 to **AUC 0.991 / ρ +0.752**, and
removes the tie at 20 that made the old target blind at the top of its range.

Two properties had to be corrected during the build, both caught by measurement rather than by
reading the code:

* **`slotted ≤ in_box` is false**, in 355 of 400 episodes.  `_target_slot_occupancy` tests a
  Cookie's *centre* against `TARGET_SLOT_TOLERANCE` (10 mm × 2.5 mm) while `cookies_in_target`
  requires the whole oriented bounding box inside; a tilted Cookie gives `slotted = in_box + 1`.
* **The `source` layer is vacuous.**  `cookies_in_source` is 80 at frame 0 of all 400 episodes
  and only empties, so after the running maximum it contributes a constant.  It is excluded from
  the potential and reported as `dropped` instead.

**The hindsight advantage.**  The RECAP-style bootstrapped advantage is degenerate here, and the
arithmetic says why: rewards are non-zero on **2557 of 552789 frames** (6.4 per episode), and `V`
predicts the episode's final potential, so `V(t+N) − V(t) ≈ 0` over 50 steps and `A` collapses to
"reward in the next 2.5 s" -- zero for ~80 % of frames.  Measured discrimination at the 70th
percentile was **−4.6 pp, i.e. backwards**.

Because every trace runs to completion, the future does not need to be estimated:

$$\text{headroom}(t) = \Phi_{\text{peak}} - \Phi_{\text{running}}(t)$$

This measures **+76 pp** (positive class: success 88.0 % vs failure 12.2 %; negative class:
4.1 % vs 39.2 %), which survives the obvious confound -- `Spearman(trel, success) = +0.000`, the
separation holds within every time quintile (+5 to +13 potential points), and the partial
correlation controlling normalized time (+0.599) is *higher* than the raw one (+0.417).
Labelling is a per-time-bucket 70th percentile with a `trel < 0.9` guard, because a successful
episode ends by holding still at its peak and the unguarded rule labels that hold as negative.

**A label-alignment bug that had been silently costing accuracy.**  `build_value_labels.py`
iterated `sorted(labels)` (seed order) while the dataset, and therefore `train_value.py`'s
positional mapping, uses **file order**.  The rollout part happened to be seed-ascending so its
300 rows were fine; the expert part is ordered 1000, 1006, 1012, … and **98 of its 100 rows were
attached to the wrong episode**.  The value head trained on that corruption still reached
Spearman 0.699 -- which is why it went unnoticed.  After the fix the same code reaches 0.6994, so
the number that "worked" was fitting a shuffled target, and the true figure is the same only
because the shuffle was confined to the easier quarter of the set.  `--dataset` now verifies
label rows against the episodes table's frame counts (400/400) and refuses to write on a mismatch.

**What this rules out, and what it does not.**

Ruled out, by measurement rather than argument:

* *Evaluation artefact* -- all arms share seeds, script and harness; the suffix is present in the
  recorded `instruction` of every episode.
* *Normalisation drift* -- 4A's de-normaliser is 6–23 % wider on joints 2–5, which looked like a
  plausible cause for its deficit, but the measured `|requested action|` is 0.8321 (v9) against
  0.8192 (4A): the policy is not moving further, it is moving the same amount to less effect.
* *Dirty labels from a mis-sorted set* -- fixed and re-verified.

Not ruled out:

* *A small real effect.*  `+0.84 ± 0.88` has an upper bound near +2.6, so a weak effect is
  compatible with this data.  But `nos − neg = 0.00` argues that whatever the suffix does, it is
  not acting as a graded condition.
* *Insufficient training.*  10 000 steps at batch 6 is 10.85 % of an epoch over 553k frames.  The
  conditioning may need more passes to be used.
* *The text channel being the wrong mechanism.*  This is now the leading hypothesis.  π0.5 reads
  its instruction through a frozen PaliGemma prefix; appending a two-word tail may be too weak a
  signal to steer a 50-step action chunk.  The alternative designs are a dedicated conditioning
  token, a numeric channel in the history latent, or CFG guidance with `w > 1` -- none of which
  have been tried.

**Direction.**  Conditioning did not work, so the next lever is not the threshold.  Either the
condition has to be a first-class input rather than a sentence suffix, or the RL step has to be
abandoned in favour of the plain-supervised reading this data supports: 4B's labelled set trains
to 9.84 against v9's 12.96, still below baseline, with 81 % of its frames still coming from failed
rollouts.  The next experiment should therefore cut that 81 % -- expert episodes plus successful
rollouts only -- before any further work on advantages.

### 9.85 Where the failures actually are, and why policy-gradient RL does not fit this task (11-07)

Stage 4 failed as a *labelling* mechanism (9.84).  This section is about what replaced it, and
about why every RL formulation tried against this task bounced -- including two published ones
whose code was read line by line.

**Diagnosis first: the bottleneck was not where four weeks of work assumed it was.**

`diagnose_failures.py` splits each episode's 20 Cookies by outcome using the trace's
`target_slot_occupancy`, which names the occupying Cookie per slot:

| per episode (24 scenes, v9) | count |
|---|---|
| seated at the final frame | 11.54 / 20 |
| ever seated | 12.92 / 20 |
| **never seated at all** | **7.08 / 20 (35 %)** |
| in the box but never aligned to a slot | 1.33 / 20 |
| seated then pushed back out | 1.38 / 20 |

So **precision costs 2.71 Cookies (14 %) and delivery costs 7.08 (35 %)**.  The working assumption
had been "the task lives in the last 2.5 mm", because the slot tolerance is 10 mm x 2.5 mm and the
insertion margin is 0.8 mm.  Tolerance is small, but the policy mostly fails *before* reaching it.

**The scene is not the variable.**  The expert reaches **94/100 strict success** on exactly the
randomisation the evaluation uses, against v9's 12.5 %.  And the randomisation is narrow: `±10 mm`
on both bins, `±1.7°` yaw, `±0.3 mm` on Cookies, which over 24 seeds spans **17.0 mm in x and
18.1 mm in y**.  A one-centimetre box displacement is tens to hundreds of pixels; no vision model
needs to generalise across it.  An earlier reading of this project argued the SuperMimic line was
worth pursuing *for generalisation* -- that was wrong, and this measurement is what settles it.

Three of the expert's six failures are not precision failures at all: `seed 1048/1060/1084` seat
exactly the ten even-indexed slots and stop, i.e. the scripted expert's 2x10 phase machine
sometimes omits the second column.  That is a behaviour bug contaminating ~6 % of the expert set,
and it is cheaper to fix than any learning result.

**Data filtering is what worked.**  `subset_dataset.py` cuts whole episodes only, because LeRobot
v3 concatenates ~28 episodes per mp4 and addresses them by absolute `from_timestamp` /
`to_timestamp`; dropping an episode leaves a gap nobody seeks to, while truncating one would
require re-encoding the shard.  Verified pixel-wise (8/8 frames identical to the source episode)
and by asserting the video timestamp columns are byte-identical to the source subset.

| arm | data | episodes | frames | 10k-step epochs | strict | mean |
|---|---|---|---|---|---|---|
| `4B` (`r4b_pos`) | everything | 400 | 552,789 | 11 % | **5.4 %** | **9.84** |
| `4B` (`r4b_nos`) | everything, no suffix | 400 | 552,789 | 11 % | **3.6 %** | **9.00** |
| **`succ`** | strict successes only | 105 | 112,857 | **56 %** | **10.9 %** | **12.56** |
| `good` | + near misses (Φ ≥ 28) | 161 | 196,512 | 32 % | 5.4 % | 11.86 |
| `v9` | expert 2x5 + 2x10 | 400 | 312,470 | 61 % | 12.5 % | 12.96 |

> The `4B` row previously read `3.6 % / 9.84`, which mixed `r4b_nos`'s strict rate with
> `r4b_pos`'s mean.  `eval_suite_4b.out` has nos at 3.6 % / 9.00 and pos at 5.4 % / 9.84.  Both
> rows are listed now because the 3.6-to-5.4 difference *is* the +0.84 suffix effect, and the
> comparison against `succ` must use the same arm the filtering replaced.

Paired: `succ − 4B = +3.62 ± 1.08` (W39/L15), `succ − v9 = −0.34 ± 0.99` (W27/L26).  Filtering
recovers everything 4B lost and lands on the baseline.  Adding the 56 near misses *costs*
0.70, which is a useful negative: those episodes score 20 Cookies in the box but leave 16-19
slots occupied, and training on them teaches the imprecision.

An arithmetic point that reframes all earlier comparisons: 4B saw 60,000 samples against 533k
frames = **11 % of an epoch**, while v9 saw 180,000 against 293k = **61 %**.  Filtering fixes this
as a side effect, because a smaller set at a fixed step count is a larger epoch fraction.

**Retrying beats every trained mechanism here.**  The policy is a stochastic map: the flow head
starts from a fresh `x0 ~ N(0, I)` on every forward pass, and an episode consumes
1500/16 = 94 of them, fed back into the next observation.

Measured, 24 scenes x 3 independent attempts on v9:

| quantity | value |
|---|---|
| within-scene standard deviation | **4.18 Cookies** |
| scenes that repeated identically | 0 / 24 |
| single-attempt mean | 13.22 |
| **best-of-3 mean** | **16.79 (+3.57)** |
| scenes improvable by retrying | 24 / 24 |
| scenes where all three attempts stayed ≤ 12 | 4 / 24 (17 %) |

`best_of_n.py` and `run_retry_suite.sh` implement it.  Two caveats are reported alongside the
number and must stay there: it does **not** raise the policy's ceiling (the 17 % unreachable
scenes stay unreachable), and an exact reset exists only in simulation, so the realisable gain on
hardware is smaller.

A closed-loop measurement was needed to see this at all.  An earlier open-loop probe sampled 8
action chunks per observation and found a standard deviation of **0.43 % of the action range**,
which was read as "the sampler is nearly deterministic, so choosing among samples cannot help".
That inference was wrong -- 0.43 % per step compounds over 94 closed-loop steps in a contact task
with a positive Lyapunov exponent.  The tell was already in this document's own notes: the same
seed and scene have scored 10/10, 0/10 and 5/10 across sessions.

**Why the two published RL formulations do not transfer.**  `ajwagen/dsrl` and
`Yyshadow/openpi-RLT` were cloned and read (`/root/refs`).  Both freeze the VLA and train a small
actor-critic; DSRL acts in the sampler's **noise** space, RLT edits the VLA's **action chunk**
through a residual actor with reference dropout.  Both need a critic that answers "at this state,
which action is better".

Our critic cannot answer that, and the reason is structural rather than a training shortfall.
Both failures were measured, in that order:

* **Time leakage, then its removal.**  Absolute headroom correlates with position in the episode
  at |rho| 0.900 (half the episodes above 0.9), so a critic trained on it reaches 100 % pairwise
  accuracy by learning "earlier is better".  Dividing by the peak does not help (the peak is a
  per-episode constant that cancels).  A per-time-bucket z-score reduces the leak to
  **|rho| 0.040, with 0 % of episodes above 0.9**, and it is the target that makes the question
  askable at all.
* **With that target the critic ranks well but does not read the action.**  On held-out episodes
  it reaches **85.6 % pairwise accuracy (679 pairs)** and Spearman 0.478 -- real signal, so the
  critic is learning something.  But `measure_q_action_sensitivity.py` holds the state fixed and
  varies only the action:

| perturbation | within-state variance | fraction of Q's total variance |
|---|---|---|
| **nearby frames** (same episode, ±10 cached rows) | 0.0026 | **1.53 %** |
| random frames from anywhere | 0.0126 | 7.47 % |

  **98.5 % of Q's variance comes from the state.**  And the random-action column is 4.9x the
  near one, so what little action dependence exists is mostly out-of-distribution displacement
  rather than a learned preference.  Steering a sampler with this critic would produce a flat
  gradient in the noise.

  **Corrected after the `potential` indexing bug (§9.85's own retraction list).**  The table
  above was trained on a target built from the mis-indexed `potential`, so the critic was
  regressed against the wrong curve.  Retraining with the fixed target (`q_fixed/report.json`)
  makes the negative result **stronger, not weaker**:

| version | Spearman | pairwise | action fraction |
|---|---|---|---|
| `q_head_v3` (pre-fix target) | 0.478 | 85.6 % | **1.53 %** |
| **`q_fixed` (post-fix target)** | **0.574** | **97.4 %** | **0.0128 %** |

  The ranking improves and the action dependence **collapses by 120x**.  So the honest reading is
  not "the critic is a bit state-biased" but "with a correctly built target the critic is very
  nearly a pure state-value function": 99.99 % of its variance is the state.  The horizon sweep
  agrees and is the reason this is not a capacity problem -- 20 steps gives 0.055 %, 400 steps
  1.08 %, the whole episode 0.745 %, so lengthening the window 20x buys three orders of magnitude
  less than it needs to.

  A first version of this experiment reported 100 % pairwise accuracy from **4 pairs** -- a
  bucket-indexing bug (cached rows multiplied by `stride`, collapsing nearly everything into the
  last bucket).  The pair count is now printed on every line so that cannot recur silently.

Five properties of this task explain why, and none of them is a defect that could be fixed by
tuning:

1. **Only 2.2 % of frames can change the score within one second.**  `Φ` moves only when a Cookie
   enters the box or a slot, about 20 events per 1500-step episode; within any 20-step window
   2.2 % of frames see `Φ` change (11.6 % at 50 steps, 46.7 % at 200).  A frame-level gradient is
   therefore 98 % signal-free by construction.
2. **Large outcome variance with small action sensitivity** is the worst pairing for RL: enough
   variance to estimate an advantage, not enough action dependence to act on it.
3. **Twenty-way conjunction.**  Strict success requires all 20 Cookies seated, so a per-Cookie
   reliability `p` gives `p^20`: 0.90 -> 12.2 %, and 0.99 -> 81.8 %.  RLT's tasks (screwdriver,
   zip tie, Ethernet, power cord) have **one** success event, where a single frame of improvement
   is legible.  Ours has twenty, and one failed episode does not say which of them failed.
4. **The proprioceptive state repeats; the state that decides the outcome does not.**  Across
   110,558 cached frames the nearest neighbour from a *different* episode is **0.0000 rad** away
   in joint space (every episode starts at home and the trajectory is periodic), yet frames at
   that distance still differ by **4.0 Cookies of headroom, 46 % of the total standard deviation**.
   What separates them is which Cookies sit in which slots -- a visual fact that never repeats
   because every episode draws a fresh layout.
5. **The variance is a property of the trajectory, not of the action.**  Same policy, scene and
   seed, three runs: `seed 302: 12 -> 7 -> 20`, `seed 306: 7 -> 15 -> 10`, `seed 315: 3 -> 17 -> 11`.

That last point is why retrying works and RL does not: best-of-N needs no credit assignment, it
scores a finished attempt with the task's own verifier.  **The variance lives at the trajectory
level, and RL looks for it at the frame level.**

**What follows.**

* **Actually tried and measured zero or negative:** sentence-suffix conditioning (4A/4B, and
  the three-arm check above proves the mechanism inert rather than merely weak), trained value
  heads used as credit assignment, and critic-guided sampling (blocked by the 0.0128 % action
  fraction).  Data filtering and retrying have both measured positive.
* **Named in the plan but never implemented**, and this document should stop counting them as
  evidence: **CFG guidance with `w > 1`** (`recap-plan.md` §6.2 defines 4B as
  `unconditional_prob=0.1` plus a dual-forward extrapolation and an `w` sweep; what actually ran
  as "4B" was still the text suffix with a different reward label, so CFG has no result either
  way), **paraphrase-based conditioning**, **advantage weighting (AWR/RWR)** -- the code has no
  weight column and the landing was binary filtering via `subset_dataset.py` -- and a
  **dedicated conditioning token** or **numeric channel in the history latent**.  The larger
  point stands without them: the channel that *was* tested failed a test that distinguishes
  "weak" from "not read at all".
* The remaining frame-level lever is not a better critic but a denser reward: score each of the
  20 Cookies separately using `target_slot_occupancy`, so one episode yields 20 signals instead
  of one aggregate.  That is the only change that attacks the conjunction and the attribution
  problem at the same time.
* Beyond that, the deficit is data: `succ` with 105 episodes matches v9 with 200 expert 2x10
  episodes, which suggests 105 is the ceiling of what that much expert data supports.  Collecting
  200 more expert episodes costs about 5 hours at the measured rate and is a more certain
  investment than any of the algorithms rejected here.

---

## 9.86 物理频率：降频不能省时间，1000 Hz 必须保留

> **标题更正。** 本节最初的标题是「500 Hz 是免费的 1.92 倍」，那是 3 个种子的结论，
> 补到 6 个种子后已被推翻（5/6，seed 1030 从 20 掉到 17）。**最终结论是降频不可用。**
> 下面保留了推翻的过程，因为失败的方式比结论本身更有用。

**问题。** 剖析显示 `env.step` 的 60.3 ms 里有 57.2 ms 是 `mj_step`，而 `mj_collision` 只占
0.233 ms（`mj_step` 的 20%）。剩下的钱花在约束求解器上，规模由 `ncon=340`、`nefc=1980` 决定。
在动 `ncon` 之前，先问一个更便宜的问题：**能不能少跑几步物理？**

当前基准用 1000 Hz（50 子步 @ 20 Hz 控制）。仓库默认是 500 Hz。降频是纯收益——如果任务行为不变。

**判据是先验设定的，不是事后挑的：** "加速可用当且仅当专家均值与成功率不降"。脚本内的
专家是 `variable_grasp` 适配器，与生成训练数据用的是同一个，所以"不降"意味着降频没有改变任务。

**结果**（同一批种子 1000/1006/1012，`max_steps=1500`，每场景 2×10 块饼干）：

| Hz | 子步 | 步/秒 | 加速 | 静止漂移 (mrad) | 专家均值 | 专家成功 | 专家步数 |
|---|---|---|---|---|---|---|---|
| 1000 | 50 | 16.7 | 1.00× | 4.4e-15 | 20.00 | **3/3** | 1086 / 1085 / 1073 |
| 500 | 25 | 32.0 | **1.92×** | 9.7e-16 | 20.00 | **3/3** | 1093 / 1085 / 1075 |
| 400 | 20 | 38.0 | 2.28× | 3.2e-15 | 20.00 | **2/3** | 1089 / 1086 / **1141 ✗** |

静止漂移在所有频率下都是零量级（机械臂靠自身重力保持姿态，没有下垂），所以降频不引入任何
数值不稳定。250 Hz 被 `SimConfig.validate` 正确拒绝：250/20 = 12.5 不是整数。

### 判读（第一次，3 种子 → 已被推翻）

3 个种子下 500 Hz 全过、400 Hz 挂 1 个，当时写下的结论是"500 Hz 零代价"。**这个结论是错的，
因为种子太少。** 补到 6 个种子（1000/1006/1012/1018/1024/1030）后：

| Hz | 子步 | 步/秒 | 加速 | 专家均值 | 专家成功 | 逐种子饼干数 |
|---|---|---|---|---|---|---|
| 1000 | 50 | 16.3 | 1.00× | 20.00 | **6/6** | 20,20,20,20,20,20 |
| 500 | 25 | 29.5 | 1.81× | 19.50 | **5/6** | 20,20,20,20,20,**17** |

**500 Hz 也不是免费的。** seed 1030 从 20 掉到 17，而 1000 Hz 下它 1066 步就成功、500 Hz 下
同样跑了 1066 步——不是超时，是专家在降频的接触动力学下少放进去 3 块。

**最终结论：降频不能用。**

证据是 6 个种子里 1 个失败（≈17%），而且失败方式不是"整段崩掉"而是"少 3 块"——这种损失
最危险，因为它不会在少量种子上暴露，却会在 100+ 场景的评估里稳定地压低所有数字。更关键的
是：**保真度检查必须保守。** 一个种子掉就是接触动力学变了，而评估的可比性依赖于"同一场景
在两种配置下判分相同"。任何降频收益都要用这个可比性来换，1.81× 不值。

### 两条方法论教训

1. **3 个种子不足以下保真度结论。** 第一轮 3 个种子的失败率是 0/3，第二轮同样的 500 Hz 在
   6 个种子下是 1/6。真实失败率若是 ~15%，3 个种子看到 0 个的概率是 0.85³ ≈ 61%——所以
   "3/3 通过"几乎不提供信息。判据是先验设定的，但**样本量没有先验设定**，这是这套流程的漏洞。
2. **降频最先破坏的是最精细的那条判据。** 400 Hz 下 seed 1012 `cookies_in_target = 20` 却
   `success = false`：饼干全进去了，卡在 `slot_occupancy`（10 mm × 2.5 mm）或 `column_balance`。
   所以**不能用"搬进几块"判断降频可用**——总量对、对齐错，正是 20 子步不足以分辨 2.5 mm 容差
   的表现。而槽位对齐恰好是 v9 本来就做不好的那一条（§9.82：对齐失败 2.71/20），用降频的
   仿真去评估一个卡在对齐上的策略，会把仿真误差和策略缺陷搅在一起。

### 顺带修掉的两个测量 bug

* `substeps` 是 `SimConfig` 上的 property（`physics_hz // control_hz`），不是 env 的属性。
  之前探测 env 得到的是一列 0。修好后 1000→50 子步、500→25 子步、`exact_division` 全 True。
* `static_drift_mrad` 在 `--bench-steps 20` 下读出 514 mrad，在 `--bench-steps 100` 下读出
  4.4e-15。差的不是物理，是**预热步数不够**：漂移量在 `bench_steps` 之后才取基准姿态，
  20 步时机械臂还在从 reset 位往目标位走，测到的是残余行程，不是漂移。只有 100 步版本的
  读数可信——它说明在 500–1000 Hz 全部频率下，机械臂靠自身重力能稳住姿态，降频不引入数值
  不稳定。**所以降频的问题不在数值稳定性，在接触分辨率。**

**下一步剩下的钱在求解器，不在频率。** 500 Hz 之后 `mj_step` 仍是 ~28 ms，`mj_collision` 仍只
占五分之一，主体是 `nefc=1980` 的约束求解。而场景里同一时刻只有 1–2 块 Cookie 在动，
其余饼干静止却仍在参与碰撞与约束。要动的是这些静止饼干（把远处饼干的 `contype` 置 0，或改用
mocap），不是采样率。

---

## 9.87 速度账本：哪些测量还算数，还剩多少提速空间

把所有和速度有关的测量放在一起，逐条标注可用性。**判据是「这个数字现在还算数吗」，
不是「这个数字有多漂亮」。**

### 可以保留的

| 测量 | 数值 | 为什么还算数 |
|---|---|---|
| `env.step` 分解 | 60.3 ms = `mj_step` 57.2 ms (92.4%) + 3.1 ms 其余 | 纯计时，与降频结论无关。**物理占 92%，所以优化策略侧最多省 8%** |
| `mj_collision` 占比 | 0.233 ms/子步 × 50 = 11.65 ms ≈ `mj_step` 的 20% | 这是碰撞削减的**理论上限**：全免也只有 1.24× |
| 频率吞吐曲线 | 1000→16.3–16.7 步/秒；500→29.5–32.2；400→38.0–39.4 | 吞吐本身是真的；**废掉的是「可以拿它换时间」这个推论** |
| 静止漂移 | 预热充分时 ~1e-15，三个频率相同 | 降频**不破坏数值稳定性**，失败纯粹来自接触分辨率。这条排除了「积分炸了」这个备选解释 |
| 真实 rollout 步成本 | 2x5、900 步下 249 ms/步，推理只占 5.2–5.8% | 关键：**249 ≫ 60**。空转 60 ms，真跑 249 ms，差值全是接触求解 |
| 步成本的场景依赖 | 同一 checkpoint 跨 episode 260 / 749 / 1430 ms | 慢的不是固定开销，是「机械臂在箱子壁上磨」。**所以「ms/步」是策略的函数，不是常数** |
| 采集吞吐 | 8 分片 × 6.03 h = 245 条 trace ≈ 40.5 条/时 ≈ 12 min/条 | 与脚本自己的「~12 min/条」估计吻合，说明分片是**线性**的 |
| 分片上限 | 8 采集 + 1 trainer；**12 个采集会把 trainer 饿死** | 实测过的硬边界（`run_dsrl_night.sh:28`），只在「采集与训练同机」时成立 |
| DSRL 反传成本 | batch 1/2/4 = 0.69/0.52/0.63 s，9.7/13.3/17.8 GB | 共存判断的依据（bridge 9.7 + 训练 10 ≈ 20/32 GB） |

### 必须作废的

| 结论 | 状态 | 作废原因 |
|---|---|---|
| 500 Hz 省 1.81–1.95× | **作废** | 6 种子 5/6，seed 1030 从 20 掉到 17 |
| 400 Hz 省 2.28–2.34× | **作废** | 3 种子 2/3；失败形态是「20 块进箱但不过严格判定」 |
| 「3 个种子 3/3 所以零代价」 | **作废** | 真实失败率 ~15% 时，3 个种子全过的概率 ≈61%。判据先验定了，样本量没有 |
| `static_drift` 514 mrad | **作废** | `--bench-steps 20` 预热不足，测到的是残余行程 |
| `substeps: null` | 已修 | 它是 `SimConfig` 的 property，不是 env 属性 |

### 从未跑过的

| 杠杆 | 状态 | 天花板 |
|---|---|---|
| 静止饼干碰撞削减（Lever B） | `bench_physics_levers.py` 写好了，**一次没跑** | ≤1.24×（碰撞全免才这么多）；而慢的状态恰恰是活跃列在磨的那些，那里一个接触都省不掉，真实收益远小于此 |

### 剩下的提速空间

| 杠杆 | 上限 | 现实 |
|---|---|---|
| 频率降一半 | 1.81× | **0×（作废）** |
| 碰撞全免 | 1.24× | 不可达；静止饼干只省空转段 |
| 推理全免 | 1.06× | 不可达 |
| 分片 8→16 | ~2× | 只在「不跑 trainer」时成立；12 已实测饿死 trainer |

**单步仿真已经没有免费的钱了。** 单步杠杆全乘起来的天花板是 1.24 × 1.06 ≈ **1.31×**，
其中碰撞那 1.24 只在空转态成立，而真实 rollout 的成本由接触重的状态决定（249 ms 对 60 ms），
那些状态一块饼干都省不掉。**逐条算下来，现实可得的单步提速 ≈ 1.0×。**

唯一还站着的杠杆是**分片吞吐**，而它不是「让仿真变快」，是「同时跑更多仿真」：已实测线性
（8 分片 ≈ 40.5 条/时），受 16 核限制，与 trainer 共机时 8 是硬顶。

**另有一条不是加速的加速：** 因为步成本随接触状态涨 5 倍以上（260→1430 ms），**一个更好的
策略本身就是提速**——v9 的 12.5% 成功率意味着大量步数花在「在箱壁上磨」的状态里，而专家
1073 步就干净收场。这条是推论不是测量，但方向与 §9.85 一致：剩下的收益在策略，不在仿真。

---

## 9.88 DSRL 修复后的 3000 epoch：约束生效了，actor 学习没有

`|mean|` 的爆炸（0.267 → 52728，§9.85 之后的那次夜间运行）被修掉了。**但修好约束之后，
暴露出来的才是真问题：actor loss 的"下降"100% 来自 alpha 衰减，Q 是往下走的。**

**修复内容。** 从「约束采样输出」改为「约束均值」：

```python
# 改前（错）—— tanh 饱和后梯度消失，mean 无限增长
sampled = clip * torch.tanh((mean + std * noise) / clip)

# 改后（对）—— 均值有界，采样仍是纯高斯，log_prob 干净
mean = clip * torch.tanh(raw_mean / clip)
sampled = mean + std * noise
```

**结果**（3000 epoch，batch 2，16.4 min，离线 552,789 transitions，`/tmp/dsrl_fix2.log`）：

| epoch | critic | noise_critic | actor | alpha | Q | logp | minQ | \|mean\| |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0295 | 1.9703 | −2281.2 | 1.000 | **+0.016** | −2279.6 | 1.618 | 0.262 |
| 500 | 0.0000 | 0.0077 | −1952.7 | 0.861 | −0.682 | −2265.7 | 1.870 | 1.538 |
| 1000 | 0.0000 | 0.0054 | −1704.8 | 0.741 | −0.632 | −2297.5 | 1.461 | 1.657 |
| 1500 | 0.0000 | 0.0010 | −1456.2 | 0.639 | −0.606 | −2278.7 | 1.225 | 1.675 |
| 2000 | 0.0000 | 0.0022 | −1249.4 | 0.550 | −0.579 | −2271.4 | 0.166 | 1.666 |
| 2500 | 0.0001 | 0.0103 | −1073.3 | 0.474 | −0.549 | −2263.4 | 0.739 | 1.700 |
| 3000 | 0.0000 | 0.0024 | **−927.2** | **0.408** | **−0.527** | −2268.7 | 0.765 | **1.734** |

### 约束生效，这是真的

`|mean|` 从 0.262 涨到 1.734 后**停住**，全程 7 个采样点落在 1.53–1.73，没越过 2.0 的帽子。
对比修复前的同区间 916 → 2598 → 20500 → 52728，这是干净的修复。`logp` 全程在
−2298…−2263 之间（极差 34，即 0.02%），不再发散。这两条证明了 tanh-on-mean 是对的做法。

### actor loss 下降是假的，99.9% 来自 alpha

把 actor loss 拆成它自己的两项：

$$\mathcal{L}_{\text{actor}} = \alpha \cdot \log\pi(a|s) - Q(s,a)$$

| 量 | epoch 1 | epoch 3000 | 变化 |
|---|---|---|---|
| $\alpha \cdot \log\pi$ | −2279.6 | −926.5 | **+1353.1** |
| actor loss | −2281.2 | −927.2 | +1354.0 |
| **占有率** | | | **99.9 %** |

$\alpha$ 从 1.000 掉到 0.408，而 `logp` 几乎不动（−2279.6 → −2268.7）。所以 actor loss 上升
的 1354 里，**1353 是熵项自己缩水**，留给 Q 的只有 0.9。而 Q 在同一段时间：

$$Q: +0.016 \;\longrightarrow\; -0.527 \qquad (\Delta = -0.543)$$

**Q 是单调下降的**，7 个采样点没有一次回头。

### 这正是 §9.85 预判的那一行

`train_dsrl.py` 自己打印的判读表里，第二行就是：

> *actor loss 下降 但 Q 不升 = critic 平坦，actor 在拟合噪声（预期结果，见 9.85）。*

实际发生的甚至比这更差——不是"Q 不升"，是 Q 在降。机制也吻合：`critic loss ≈ 0.0000`
（TD 残差被拟合到零，因为 reward 只在 **0.46 %** 的帧上非零，其余全是 0 奖励 + 一致的
bootstrap），所以 critic 对动作几乎不敏感（§9.85 修正后实测：**99.99 %** 的 Q 方差来自状态，
命中系数 0.0128 %）。actor 在
一个近乎常数的 Q 上做梯度上升，得到的只能是噪声；而 alpha 因为熵项持续下降，最后连随机性
都不剩，只剩下在一个平坦曲面上乱走。

### 这份结果能说明什么，不能说明什么

**能说明：** DSRL 的训练**机制**现在是正确的——梯度穿过冻结的 10 步 ODE 到达 noise、均值
有界、log_prob 干净、不爆炸。§9.85 归因的"梯度消失/爆炸"这一类工程问题不再是障碍。

**不能说明：** 任何关于任务表现的事。三条硬限制：

1. **没有存 checkpoint。** 这次运行没给 `--output`，权重没落盘，无法评估。
2. **纯离线。** `train_dsrl.py` 只在 400 条原始 episode 的回放表上训练，从未在仿真里跑过
   actor。237 条新轨迹还没提特征、没进回放表。
3. **3000 epoch 是 0.5 个 epoch 的数据量级。** 552,789 transitions / batch 2 = 276k 步为一
   个 epoch，3000 只走了 1%。§9.84 的教训（4B 在 11% epoch 上就没学会读指令）在这里同样
   适用，而这次的 1% 更极端。

**所以结论停在工程层面：** 修好了，能跑长，但"能跑长"和"能学到"是两件事。§9.85 给出的
结构性理由（20 路合取、2.2% 的帧能改变分数、Q 99.99% 由状态决定）没有被这次运行反驳，也
没有被它检验。**要做检验，下一步必须是一个能落盘、能进仿真评估的在线协议**，而不是继续
加 epoch。

---

## 9.89 速度账本（修正版）：原仓库用的是 0.57M MLP，不是 VLA

前面几节反复说"我们做不了在线 RL，因为物理仿真太慢"。**那句话是错的**，错在两处，这一节
把两处都修掉，并给出一个可以复现的账本。

### 错误一：DSRL 的 base policy 是 0.57M 参数的 MLP

一直说"4.14B vs 0.8M，五千倍"，但没核实过那个 0.8M 是什么。`dsrl_can.yaml`：

```yaml
network:
  _target_: model.diffusion.mlp_diffusion.DiffusionMLP
  mlp_dims: [512, 512, 512]   time_dim: 16
obs_dim: 23   action_dim: 7   cond_steps: 1   act_steps: 4
```

输入 $28 + 16 + 23 = 67$，输出 28，逐层算下来 **574,492 参数**。它**不是 VLA**，是从零训练
的 MLP 扩散策略（robomimic can，约 200 条演示，`state_5000.pt`，23 维本体感知，无图像）。

所以"DSRL 在小模型上成立"和"DSRL 在 VLA 上成立"是两个命题，参考实现只证明了前者。真正
把 DSRL 搬到 VLA 的是 `openpi-RLT`（冻结 openpi/pi0.5 + RL token + 小 actor-critic），而它
依赖**人工按键标注 success/failure** 和**单事件任务**（Ethernet 插入）——两条我们都没有。

### 错误二：没有反向传播穿过 VLA，我算的 0.85 s 里 66 % 是视频解码

`train_dsrl.py` 走的是 DSRL-NA 路径：`critic` 用 TD 训练，`critic_noise` 由 `critic` **蒸馏**
，actor loss 用 `critic_noise`。原仓库 `dsrl.py:273,350` 两处 `diffusion_policy(...)` 都在
`no_grad` 里，我们的实现一致。**梯度只穿过一个 2.6M 的 MLP。**

于是吃掉了 833 ms/epoch 的那个东西必须重新找。实测（batch 8，30 epoch，包住真实调用）：

| 项 | 次数 | 每次 |
|---|---|---|
| `embed_prefix` | 30 | 126.8 ms |
| `paligemma_forward` | 330 | 28.0 ms |
| `denoise_step` | 300 | 12.7 ms |
| `preprocess` | 40 | 5.0 ms |

330 次 forward = 30 次 prefix + 300 次 denoise 内部，所以每 epoch 的**策略前向**是
$155 + 127 = 282$ ms。而实际 833 ms——差的 550 ms 是 `dataset[i]`：

```python
# train_dsrl.py:385 —— 注释自己写着
# A `LeRobotDataset` item is a full frame, which needs video decoding; that is
# ~0.1 s per frame and dominates everything else
```

**66 % 的时间在从 mp4 解码。在线 RL 没有这个成本**（观测是仿真直接渲染的）。

### 两个优化，A/B 实测

**1. 解码池**（`--prepared-cache`，默认 512 帧）

| | 每 epoch |
|---|---|
| `--prepared-cache 0` | **833 ms** |
| `--prepared-cache 512` | **500 ms**（1.67×）|

池子建设 512 × 46 ms = **23 s 一次性**。tb0 GB 内存（uint8 保在 CPU）。

**2. prefix 缓存 + 让 `--utd` 真的生效**

发现 `--utd` 之前**只被解析、从未被使用**——每 epoch 只做一次更新。接进循环后：

| utd | 20 epoch | 每次更新 |
|---|---|---|
| 1 | 10 s | 500 ms |
| 4 | 18 s | **900 ms** |

4 倍梯度步只花 1.8 倍时间。模型：

$$\text{每 epoch} = \underbrace{155}_{\text{prefix}\times1} + \underbrace{127k}_{\text{denoise}} + \underbrace{218}_{\text{critic MLP}}$$

utd=1 解出 critic MLP = 218 ms，代入 utd=4 得 881 ms，实测 900 ms。

### 修正后的在线 RL 账本

$$\text{每决策} = 0.055_{\text{物理}} + 0.155_{\text{prefix}} + k(0.127_{\text{denoise}} + 0.218_{\text{critic}})$$

| 配置 | 每决策 | 10 万决策 | 20M 决策 |
|---|---|---|---|
| utd=1 | 0.56 s | **15.4 h** | 129 天 |
| utd=4 | 1.6 s | 43 h | 364 天 |
| **utd=20** | **7.1 s** | **197 h** | **4.5 年** |

### 每个杠杆的真实权重

| 杠杆 | utd=1 | utd=20 |
|---|---|---|
| 饼干焊接 3.09×（物理 55→18 ms） | **6.6 %** | 0.5 % |
| 解码缓存 1.67× | **离线特有，在线无关** | 同左 |
| prefix 缓存 | 0 %（只算一次） | **2.2 %** |
| critic MLP 218→? | 若降到 50 ms：**30 %** | 若降到 50 ms：**47 %** |

**prefix 缓存的价值被高估了。** 它只省一次 155 ms，而 utd=20 的总成本 7.1 s 里
$20 \times 0.345 = 6.9$ s 才是主体，其中 critic MLP（218 ms，比 denoise 的 127 ms 还大）
占一半以上。**下一个有意义的杠杆是 critic MLP，不是 prefix，更不是物理。**

### 留一个自相矛盾处，供后来的人查

`denoise_step` 内部也调用了一次 `paligemma_with_expert.forward`（`modeling_pi05.py:890`），
所以 `paligemma_forward` 的 330 次已经包含了 prefix 的 30 次和 denoise 内部的 300 次，
两行不能相加。本节的分项计时互有重叠，只有"每 epoch 总时间"和"A/B 的差值"是可靠的。

§9.89 的结论是"critic MLP 218 ms 是下一个杠杆"。**这个结论本身是错的**，因为它默认了
"critic 必须和大 VLA 一起在循环里"。下一节按论文 §C.5 重做之后，循环里根本没有 VLA，
也没有 218 ms 的 MLP。

## 9.90 按论文 §C.5 重写 DSRL：循环里不再有 VLA（11-09）

### 论文到底喂进去什么

arXiv 2506.15799v2 附录 C.5 原文（π0 实验）：

> We input the proprioceptive state, the final token's last hidden feature from π0's VLM backbone
> (**a 2,048-dimensional vector**), and visual features into the noise policy.

以及处理 1,600 维噪声的办法：

> …we instead train a **single-step** actor π_single(s) and critic Q_single(s, w_single) where
> W_single = R^d. At inference time, we simply sample w_single ~ π_single and **repeat it across
> the action chunk axis**.

再加一句决定了整条账本的部署事实：

> We run π0 inference on a **remote policy server**, but run DSRL training locally on a NVIDIA
> GeForce RTX 3070 GPU with **8GB of VRAM**.

8 GB 装不下 3.3 B 的 π0（bf16 就是 6.6 GB，加激活和优化器不行）。所以"训练在本地、策略在远端"
不是省钱技巧，而是**训练循环里没有 VLA 前向**的必然结果。这三点合起来推翻了 §9.89 的账本：

| §9.89 的假设 | 论文的做法 |
|---|---|
| 每次更新都要 prefix + denoise 前向 | actor/critic 只吃**缓存特征**，循环里无 VLA |
| noise 空间是 R^1600 | actor/critic 在 **R^32**，推理时才沿 chunk 轴复制 |
| critic MLP 218 ms 是瓶颈 | critic 只有 ~0.2 ms 前向 |

### 改了什么（`scripts/recap/train_dsrl.py`）

1. **观测** = `[proprio(16), VLM final-token hidden(2048), visual(256)]`，共 2320 维。
   2048 维取自 `encode_prefix` 已经算出来的 `prefix_hidden`，用 `pad_masks.sum()-1` 取
   **最后一个非 pad** 位置（取 `[:, -1]` 会在 tokenizer 补齐时读到 pad）。维度 2048 与论文
   完全一致——gemma-2b 的 width 就是 2048。
2. **噪声**：`--noise-mode single`（默认）= R^32，`repeat_noise` 沿 chunk 轴复制成 1600 维再送
   采样器；`--noise-mode full` 保留旧的 1600 维。`target_entropy` 随之从 $-(50\times32)$ 变成
   $-d\log$ 项。
3. **两级缓存**（`<dataset>/dsrl_cache/`，按 checkpoint + 池子几何命名，可复用）：
   - `obs_*.pt`：每个 transition 的 VLM 特征 + 归一化 proprio（current 和 successor 各一份）；
   - `distill_*.pt`：`transitions × distill_per_frame` 组 `(ε, π0(s,ε))`。**这是唯一依赖噪声的
     量**，所以它必须跑采样器；在缓存阶段一次批量生成，循环里只做 gather。
4. **transition 取相邻行对**。原先"随机抽行 + 查 `row+1`"需要二次解码一批不相交的帧；抽
   `(row, row+1)` 对则后继免费。校验条件：`frame_index` 连续且 `episode_index` 相同。
5. `--utd` 默认 20（论文附录 B：UTD ≥ 20）。
6. 删掉了 `--prepared-cache`、`--pool-on-gpu`、`--samples`（后者从未被读过）。

### 顺手修掉的一个真 bug：80 % 的观测是全零

旧代码：

```python
obs_by_frame = np.zeros((len(episodes), projection_dim), np.float32)
obs_by_frame[cached_rows] = cache
for offset_index in range(1, stride):
    source = cached_rows - offset_index
    valid = source >= 0
    obs_by_frame[cached_rows[valid]] = np.where(          # 源和目的都是 cached_rows
        obs_by_frame[cached_rows[valid]].any(axis=1, keepdims=True),
        obs_by_frame[cached_rows[valid]],                 # ← 已经有值，np.where 永远保留它
        cache[valid],
    )
```

`cached_rows` 同时出现在赋值号两边，所以每次赋值都是"把已有值写回已有值"，循环是**空操作**。
`features_v9` 每 `stride = 5` 行才写一次（`written.npy` 实测 110,558 / 552,789），于是
**每 5 行里有 4 行的观测是精确的 0**。

实测确认：`written.sum() = 110558`，`cached_rows[:12] = [0, 5, 10, ...]`。也就是说
§9.88 那次 3000 epoch 的 DSRL，actor 和两个 critic 有 4/5 的样本看的是零观测。

修法是注释一直在说的那件事——**前向填充到最近的前一个已缓存行**：

```python
nearest = np.clip(np.searchsorted(cached_rows, np.arange(len(written)), "right") - 1,
                  0, len(cached_rows) - 1)
obs_by_frame = cache[nearest]
```

修完打印 `552789/552789 rows non-zero`。帧间隔 5 帧（0.25 s @ 20 Hz），用前一帧的特征是可接受
的近似；用零不是。

### 一次性成本（实测）

| 阶段 | 512 transition / 8 noise |
|---|---|
| 解码 + VLA（current 行，含 8 次采样） | **84 s** |
| 解码 + VLA（successor 行，不采样） | **34 s** |
| 合计（写盘后复用） | **~2 min**，`distill_*.pt` 26.7 MB |

缓存内容自检：`noise (512, 8, 32)` 标准差 0.998（标准正态）；`chunk (512, 8, 1600)`；
**同一帧内换噪声造成的动作偏移 0.0379，不同帧之间 0.1308**（比值 0.29）——说明噪声确实在
起作用，不是被采样器忽略。相邻 chunk 位置相关性 0.9998，符合"同一噪声沿 chunk 轴复制"的
预期。动作本身 std ≈ 0.98，所以 0.0379 约等于 3.9 % 的动作标准差。

### 循环成本

原有的 500 ms/epoch 是 utd=1；重写后单次更新由三次 MLP 前向+反向加两次 gather 组成。

拆解（batch 64、纯 MLP 部分）：`critic` 前向 0.21 ms、`NoiseActor.sample` 0.19 ms、
`noise critic` 前向 0.22 ms，单个 MLP 前向+反向 1.11 ms。剩下的都是 Python 与 kernel 启动，
所以 batch 8 和 batch 64 的单次更新成本几乎一样（23 ms vs 26 ms）——**成本由"一次更新"
决定，不由 batch 规模决定，所以 batch 越大越划算**。

这直接改写了在线账本：

$$\text{每决策} = 0.055_{\text{物理}} + \underbrace{0.004 \times \text{utd}}_{\text{三次 MLP}} + \underbrace{\epsilon}_{\text{缓存摊销}}$$

with utd = 20 时约 **0.14 s/决策**，比 §9.89 的 7.1 s 低 **50 倍**；10 万决策从 197 h 降到
**~4 h**，且这个 4 h 里没有 VLA。

### 仍然存在的退让，写在这里而不是藏起来

1. **噪声覆盖是有限的。** 论文每个梯度步现抽 ε；这里只有 `transitions × distill_per_frame`
   组预算好的 `(ε, π(s,ε))`。`--distill-per-frame` 是买回覆盖的唯一旋钮，它会打印出来。
   缓解因素：actor 初始化就是 N(0,1)，和被预算的 ε 同分布；`noise_clip = 2` 把它限制在两倍
   标准差内。
2. **`alpha` 现在是摆设。** `NoiseActor` 的 `std` 固定为 1（与参考实现
   `jnp.full_like(mu, self.fixed_std)` 一致），因此熵恒定，`alpha` 只是给 actor loss 加常数。
   保留它是为了对齐参考实现，日志里照旧打印漂移。
3. **TD 仍然用录制的下一个动作**，不是 actor 重采样。离线场景下后者会给自己的目标注入当前
   意见。论文 §5.3 的 offline→online 结果也说明直接搬在线目标不行（RLPD/Cal-QL "not able to
   learn at all"）。

## 9.91 critic 目标：三种写法，三种失败，以及哪一种是真问题（11-09）

§9.90 把 VLA 从循环里拿掉了，但拿掉之后暴露出一件更基本的事：**循环里的 critic 从来没有学过
任何东西**。这次把它当成一个可证伪的问题来查，而不是再调一轮超参。

### 先说最关键的那个数

`train_dsrl.py` 第一次跑通（旧的散点池，512 transition，γ=0.99）时，日志是：

```
critic 0.0210 -> 0.0002      Q -0.406 -> -0.978
```

critic loss 掉到 $2\times10^{-4}$，Q 停在初始值附近。§9.88 把这个读作"critic 平坦"。
**这个读法是错的**：那时候池子里 512 个 transition 里**只有 1 个带非零奖励**，`done` 全为 0，
所以每个 TD 目标的唯一自洽解就是 $Q=0$。不是 critic 学不会，是它没有东西可学。

同一批 512 行上真实折扣回报的均值是 **6.5**，最大 **25.9**。

### 三个数，依次排除了三种解释

**1. 折扣因子选错了。** 这套数据 episode 长 1382 步（20 Hz，约 69 秒），第一个放置发生在中位数
第 226 步。而 γ=0.99 的半衰期是 69 步：

| γ | 半衰期 | 每 episode 折扣回报 | 第 226 步权重 |
|---|---|---|---|
| 0.99 | 69 步 | **0.87** | 10 % |
| 0.995 | 138 步 | 3.26 | 32 % |
| 0.998 | 346 步 | 9.01 | 64 % |
| **0.999** | **693 步** | **13.83** | **80 %** |
| 1.0 | ∞ | 22.76 | 100 % |

γ=0.99 把 22.76 的回报压成 0.87——和随机初始化的 Q 同量级。gym 的默认值在这里没有意义，
因为 gym 的 episode 是几百步，这里是 1382 步。默认改成 **0.999**。

**2. 池子里没有奖励锚点。** 0.46 % 的帧带奖励，均匀抽 512 个 transition 期望只命中 1 个。
改成"所有带奖励的行必进池，其余随机填"后，池子从 1 个锚点变成 2557 个（62.4 %）。

**3. 池子在时间上是散的。** 这是结构性的一条，也是前面所有失败真正的原因：

| 池 | 行数 | 覆盖 episode | 连续段数 | 平均段长 | 最长段 |
|---|---|---|---|---|---|
| 512 散点 | 512 | 277 | 512 | **1.00** | **1** |
| 4096 散点 | 4096 | 400 | 3687 | **1.11** | **5** |

TD 要靠 $Q(s')$ 把奖励往回传，而 γ=0.999 需要约 **693 步**的传播链。一个平均长度 1.11 的
散点集合**连一步都传不了**——`rows_next = rows_cur + 1` 取到的后继永远不在训练集里，所以每个
bootstrapped 目标都是在让 critic 去拟合自己在其训练分布之外的外推值。

这不是"训练不够久"，是**目标在数学上没有固定点**。

### 直接测表征上限，把"学不会"和"读不出"分开

新增 `scripts/recap/probe_value_ceiling.py`：同一批行、同一批特征、同一架构，只换目标。

| 目标 | 散点池 (512) | 锚点池 (4096) |
|---|---|---|
| 恒 0 | −1.015 | −1.821 |
| **Monte-Carlo $G(s)$** | **+0.552** | **+0.426** |
| TD（实际用的） | **−1.173** | **+0.059** |

Monte-Carlo 目标是纯回归，没有固定点可退化。它拿到 +0.55，说明**特征完全读得出回报**——
`proprio(16) + π0 VLM final-token(2048) + SigLIP summary(256)` 这个组合是有信息的。

于是结论被钉死了：**问题在目标与采样，不在表征。** 换 critic 结构、加大网络、换激活函数都不会
有用。

### 三种 critic 目标的实测行为

同一套数据、同一套特征，只换 critic 的回归目标，各跑 3000 epoch（batch 128, utd 20）：

| | loss | Q 轨迹 | minQ 终点 | 真实上限 26.9 |
|---|---|---|---|---|
| TD + 散点池 + 锚点 | 2.5 → 6.2 | 1.8 → 7.6 → 12.5 → **25.0** | **49.7** | **越界 1.8 倍** |
| TD + 散点池（无锚点） | → 0.0002 | −0.4 → **−1.0** | −1.0 | 塌到 0 |
| Monte-Carlo | 37.5 → 11.9 | 8.6 → 9.4 → 9.1 → **8.6** | 9.0 | 命中（真值均值 8.92）|

三行的失败方式各不相同，而且每一种都能从机制上解释：

* **有锚点的 TD 单调发散。** Q 一路涨过真实最大值，`minQ` 到 49.7，同时 actor loss 从 −46
  掉到 −96。actor 在追一个不断膨胀的假价值——这就是离线 Q-learning 的经典外推高估，而
  `min(·)` 双 critic 没有挡住它（两个 head 一起漂）。
* **无锚点的 TD 塌到常数。** 没有奖励的链上 $Q=0.999Q$ 的唯一解是 0，所以整个 critic 被拉到
  一个常数（实测 −1.0）。
* **Monte-Carlo 数值正确，但拟合被动作噪声稀释。** Q 稳定在 8.6–9.4，与真值均值 8.92 吻合。

### 一个差点写错的结论

看到 MC 那行，我本来要写"MC 目标与动作无关，所以 actor 没有梯度"。这个推论**不成立**：
SAC 本身就是每个状态只有一个动作的样本，它靠函数逼近器在相邻 $(s,a)$ 之间插值学出动作
依赖。所以这句话必须实测。

`scripts/recap/probe_critic_gradient.py` 在同一批特征上拟合两种 critic，然后量三件事：
$|\partial Q/\partial a|$、同状态下换动作造成的变化（`within-state`）、以及它与跨状态变化的
比值。

| 目标 | held-out R² | $\lvert\partial Q/\partial a\rvert$ | within-state std | across-state std | **within/across** |
|---|---|---|---|---|---|
| Monte-Carlo | **+0.122** | **3.19** | **3.05** | 4.73 | **0.646** |
| TD | −0.219 | 1.57 | 1.53 | 4.73 | 0.323 |

**MC 的动作敏感度是 TD 的两倍，R² 也更好。** 我原本的猜测完全反了——如果照那个猜测写进
文档，就会把一个正确的方案误判为死路。

真正被稀释的是**状态**信号：MC 用同一个目标监督同一状态下的多个不同动作，逼着网络对这些
动作取平均，于是 R² 从散点池小样本上的 +0.55 降到 +0.12。这是一个真实代价，但它比 TD 的
−0.22 还是好得多。

所以当前证据支持的结论是：**MC 目标是三种写法里唯一同时具备正确数值和可用梯度的。**

### 正在验证的第四条：锚定连续段 + 段尾精确回报

机制上讲得通的做法是把两者拼起来：**段内用 TD（保留动作依赖），段尾用精确的 MC 回报**
（让 bootstrap 有锚点，而不是自由外推）。

```
target(s) = r + γ (1-done) Q(s')        s 不是段尾
target(s) = G(s)                         s 是段尾，G 是已算好的 MC 回报
```

段尾不需要长到 693 步：一段 128 步的链，只要它内部含一个奖励、且末端接上精确回报，头部的
目标就是一个正确的 128 步截断回报。

实现时踩了两个自己的 bug，都写在这里因为它们都会"看起来在工作"：

1. **锚点只统计、没使用。** 第一版算了 `anchored` 集合、打印了它的大小 2557，然后
   `rng.shuffle(np.union1d(anchored, usable))`——而 `anchored ⊆ usable`，所以并集就是 `usable`，
   一个锚点都没被优先，打印出来的数字描述了一个根本没建出来的池子。结果 32 个段里大部分
   无奖励，Q 塌到 −1.6。
2. **缓存键没含池子几何。** 段池 128×64 和散点池都是 4096 行，签名只含 `t{transitions}`，
   于是段池那次**直接复用了散点池的 `obs_*.pt`**：观测来自散点池、奖励和动作来自段池，
   三张表各自看都对，合起来全错。现在签名里加了行集合的 CRC32
   （`pool_hash = zlib.crc32(rows_cur)`）。

修正后：7816 行、64 个段、全部锚定在第 2557 个奖励候选行上。

**实测结果：还是塌了。**

```
  1  critic 1.4836   Q -0.147
250  critic 1.6029   Q -1.130
500  critic 0.1645   Q -1.145
750  critic 0.0024   Q -1.273
```

池子里真实折扣回报的均值是 **12.26**（std 5.53，最大 26.13），**0 % 的行低于 1**。而 critic
输出 −1.2，损失同时掉到 0.0024——这意味着**目标本身就塌到了 −1.2**，不是拟合不动，是
自洽地坍缩了。

机制是这样的：段尾只有 61 个（7816 行的 0.8 %），每个 batch 平均只命中 1 个；它们的
$Q=G\approx+12$ 是对的，但要把这个数往链条上游传，需要目标网络把它"拉过去"。
目标网络每 epoch 更新一次、$\tau=0.005$，所以信息前沿大约每 $1/\tau = 200$ 个 epoch 前进
一个链节——128 节的链需要天文数字的 epoch。实测 1000 个 epoch、2 万次梯度步之后，Q 仍然停在
−1.2，最深的段内行一点都没被点亮。**这是慢目标网络与长链的组合问题，不是调参能解决的**：
要传 693 步，$\tau$ 必须大到接近 1，而那时目标网络就失去意义了。

对照之下 Monte-Carlo 根本不需要传播——它对每一行直接给出 $G$，所以第一次评估就命中
（Q 8.6 vs 真值 8.92）。

### 三种池子 × 两种目标 = 六次测量，只有一种站得住

| 池 | 目标 | Q 终值 | 与真值 | 判定 |
|---|---|---|---|---|
| 散点 512（1 个奖励） | TD | −1.0 | 真值均值 6.5 | **坍缩**（唯一解是 0） |
| 散点 4096（2557 锚点） | TD | **25.0**，minQ 49.7 | 真值最大 26.9 | **发散**（外推高估） |
| 段 128×64（全锚定 + 精确段尾） | TD | −1.2 | 真值均值 12.26 | **坍缩**（传播太慢） |
| 散点 4096 | **MC** | **8.6–9.4** | 真值均值 8.92 | **命中** |

TD 的三种失败不是同一个 bug，而是同一个结构性事实的三个面：**在这份数据上，价值需要跨
数百步传播，而每种池子形状都会在某个环节切断它**——散点切断了链，锚点不足让固定点变成 0，
慢目标网络让链传不动。Monte-Carlo 是唯一不需要传播的目标。

### 结论与代价

采用 `--critic-target mc`（现为默认）。代价必须写清楚：

1. **MC 只能评估行为策略。** 它评估的是"录下来的那个动作带来的后果"，不能给没执行过的动作
   记分。这里可接受，因为 actor 搜索的是已经会做这件事的策略的噪声空间。
2. **状态信号被稀释。** 同一状态下的不同动作共享一个目标，逼着网络平均它们，R² 从散点小样本
   的 +0.55 降到 +0.12。
3. **动作敏感度是插值出来的，不是学出来的。** `within/across = 0.646` 量的是"换动作时 Q 变
   多少"，不是"变得对不对"。要判断变得对不对，得拿 §9.89 之前那个分支数据集做成对排序检验
   （那里测到的行为策略成对准确率是 0.611）。**这一步还没做**，是下一步必须先做的事。

### 这一节的方法论价值


这一串排查里，每个结论都曾被一个"看起来合理"的读数骗过一次：

| 读数 | 第一解释（错） | 实际原因 |
|---|---|---|
| critic loss → 2e-4 | "critic 平坦，actor 在拟合噪声"（§9.88） | 池里只有 1 个奖励，TD 唯一的解是 0 |
| Q 单调上升 | "critic 学到了价值" | 外推高估，越界到真实上限的 1.8 倍 |
| MC 的 Q 对得上 | "critic 对了，可以训 actor" | 数值对，梯度为零 |
| 段池跑完 Q 塌 | "段池不行" | 锚点没生效 + 缓存串了 |

共同点：**每一个都是"损失或均值看起来正常/合理"的状态**。能区分它们的不是损失曲线，而是
外部量——奖励锚点计数、连续段长度、真实回报的数值、以及 `∂Q/∂a` 是否非零。

## 9.92 对照开源实现：我漏掉的是时间尺度，不是 critic 目标（11-09）

论文只写了方法，真正的实现细节在 `/root/refs/dsrl`（开源仓库，CoRL 2025）和
`/root/refs/sb3-dsrl`（其 fork 的 SB3，`dsrl/dsrl.py`）。注意 `dppo/` 和
`stable-baselines3/` 两个 submodule 是空的，实际可读的是 `train_dsrl.py`、`env_utils.py`、
`utils.py` 和 `cfg/*.yaml`，加上单独的 `sb3-dsrl` 副本。

### 一处结构性差异，它才是 §9.91 真正的原因

`env_utils.py` 里的 `ActionChunkWrapper.step`：

```python
def step(self, action):
    for i in range(action.shape[0]):     # action.shape[0] == act_steps
        self.count += 1                  # count 以 env step 计
        obs_i, reward_i, ... = self.env.step(action[i])
    ...
    if self.count >= self.max_episode_steps: done = True
```

**一个 RL step 就是执行一整个动作块。** 而 `train_dsrl.py` 里
`MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)` 把这个换算写在了明面上。
robomimic can：`max_episode_steps: 300`、`act_steps: 4` → **一个 episode 是 75 个 RL step**，
而 `discount: 0.99` 的半衰期是 69 个 RL step——**折扣几乎覆盖整个 episode**。

我这边的对应数字：

| 量 | 参考（can） | 我原来的实现 | 我修正后 |
|---|---|---|---|
| 一个 RL step | `act_steps=4` 个 env step | **1 个控制步** | `execute=16` 个控制步 |
| episode | 75 RL step | 1382 RL step | **87 RL step** |
| γ | 0.99 | 0.999（被迫） | **0.99** |
| 半衰期占 episode | 92 % | 50 % | **79 %** |
| 需要传播的距离 | ~69 步 | **693 步** | **69 步** |
| 折扣后保留的回报 | — | 13.83 / 22.76 | **16.57 / 22.76** |

我原来把**同一个决策切成了 16 份**当成 16 个独立 transition。证据是动作本身：

```
mean |a[t+1]  - a[t]|  = 0.0021    ← 相邻控制步
mean |a[t+16] - a[t]|  = 0.0250    ← 相隔一个 RL step
ratio = 11.8x
```

相邻控制步的动作差只有 0.0021，而动作本身 std ≈ 0.98——**相差 0.2 %**。因为 `execute=16`
意味着这 16 个动作来自**同一次策略调用、同一个噪声 ε**，它们只是同一个 50 步 chunk 的
16 个切片。把它们当成 16 个独立决策，一方面凭空造出 16 倍的样本量，另一方面把价值需要传播的
距离也放大了 16 倍。

**§9.91 说"TD 在这份数据上结构性地无法工作"，这个结论对，但归因错了。** 不是 TD 不行，是我
在错误的粒度上建了池子。修正后：`--step-stride 16`，奖励按 16 个控制步求和，后继取
`row + 16`，段长度以 RL step 计（64 RL step = 1009 行），γ 回到 0.99，σ 传播距离 69 步。

### 其他逐项对照

| 项 | 参考实现 | 我的实现 | 判断 |
|---|---|---|---|
| γ | 0.99（按 RL step） | 0.99（修正后） | ✓ |
| `utd` | 20 | 20 | ✓ |
| `batch_size` | 256 | 256 | ✓ |
| `n_critics` | 2 | 2 | ✓ |
| `tau` | 0.005 | 0.005 | ✓ |
| 网络 | 3 层 × **2048** | 3 层 × 512 | 容量 1/4，可接受（输入是 2048 维 VLM 特征而非 23 维本体） |
| `action_magnitude` | 1.5（噪声上下界） | `noise_clip` 2.0 | 相近 |
| 噪声空间 | R^(4×7) = R^28 | R^32（论文 §C.5 的单步噪声） | 按论文 |
| 观测 | 23 维本体（robomimic） | 2320 维（proprio+VLM+visual） | 按论文 |
| `noise_critic` 更新 | **额外 10 步**，每步独立采 batch | 与 utd 同步 1:1 | **覆盖少 30×**，未修 |
| `target_ent` | **0.0** | +45.4 | 需对齐 |
| actor | tanh 压缩 + **可学习 log_std** | 无界 + 固定 std=1 | α 在我的实现里是常数偏移，参考实现里是真在调节 |
| critic 目标 | 在线 TD，next action 由 actor 现算，**带熵项** `-α·log π` | 离线 TD，next 用录制动作，无熵项 | 有意的离线偏离 |

### 还没对齐的三件事，按重要性排序

1. **`noise_critic` 的更新量。** 参考实现每个 env step 给它 10 个独立 batch 的梯度步，我在
   utd 循环里只给 20 次中的 20 次共用同一批状态。它决定蒸馏链的拟合速度，而 actor 直接吃这个
   critic——所以这是下一步该先对齐的。
2. **`target_ent = 0.0` 与可学习 `log_std`。** 我的固定 std=1 让 α 完全失效（§9.90 已记）。
   换成可学习 log_std 才能让 α 真正起作用。
3. **`--step-stride` 的验证。** 修正后的 TD 正在跑；在它给出 Q 之前，本节只主张"时间尺度是
   §9.91 的归因错误"，不主张"TD 现在已经能用了"。

### 一句话总结这一节

§9.91 我写了一张"三种池子 × 两种目标"的表，把失败归给 TD 本身。对着 `ActionChunkWrapper`
读十行代码就发现：**池子的形状不是主要问题，池子的时间单位才是。** 参考实现没有做任何特别的
事情来让 TD 工作——它只是让一个 RL step 等于一次真实的决策。

### 修正后的实测：TD 第一次表现出健康行为

`--step-stride 16 --segment-length 64 --gamma 0.99`，64 段，全部锚定，段尾用精确回报：

| epoch | critic loss | Q | minQ | actor loss |
|---|---|---|---|---|
| 1 | 2.02 | 0.12 | 0.12 | −45.44 |
| 250 | 0.48 | −0.13 | −0.12 | −46.01 |
| 500 | 0.61 | 0.44 | 0.33 | −45.36 |
| 1000 | 0.56 | 1.25 | 1.09 | −45.27 |
| 1500 | 1.04 | 1.44 | 0.84 | −44.23 |
| 2000 | 0.61 | 2.17 | 1.04 | −42.41 |
| 2500 | 0.90 | 2.35 | 0.80 | −43.94 |
| 2750 | 0.49 | 2.66 | 1.70 | −44.73 |
| 3000 | 0.34 | **2.77** | 1.35 | −45.10 |

这是整个 §9.89–9.92 里**第一次没有出现任何一种病态**：

* critic loss **稳定在 0.5–1.0**，不塌到 $10^{-4}$（无锚点的坍缩）也不发散；
* Q **单调上升** 0.12 → 2.77，方向正确（真值 16.57）；
* **`minQ < Q` 始终成立**，两个 head 没有一起漂——对照散点池那次 `minQ` 冲到 49.7 而真值上限
  只有 26.9。

actor loss 这一栏**不算健康**，要单独说：它非单调（−45.44 → −46.01 → −42.41 → −45.10），
3000 个 epoch 后的终点（−45.10）与起点（−45.44）几乎相同。脚本自己打印的判读表里第三行
——"Q 持续上升而 actor loss 不动"——恰好适用于这里。也就是说 critic 在缓慢爬升，但 actor
**还没从中取得可用的方向**。考虑到 Q 只有真值的 17 %，这与"critic 还不足以支撑 actor 改进"
是一致的，而不是另一个独立的问题。

**但 Q 只到 2.77/16.57 = 17 %，且增长在明显放缓**（750→1500 涨 0.5，1500→2250 涨 0.84，
2250→3000 涨 0.49）。这仍然不是能用的 critic。原因是链长：64 段 × 64 RL step，段尾的精确回报要
往上游传 64 个链节，而 $\tau=0.005$ 的目标网络每个 epoch 只让前沿前进一小步。走完 64 节需要
数万 epoch；而且 $Q$ 的覆盖面受段尾稀疏度限制（64 个尾 / 4096 个 transition，1.6 %）。

**所以修正后的结论是：** 参考实现的 TD 在这份数据上能工作，前提是时间尺度对、锚点足、段够长，
且训练量够大。§9.91 的"TD 结构性地无法工作"是错的；正确说法是"我在错误的粒度上建池子，导致
需要的传播距离是可行值的 10 倍"。但**"能工作"和"够用"是两件事**——2.77 对 16.57 的 Q 还不足以
支撑 actor 改进。

### 下一步该按这个顺序

## 9.93 补齐参考实现的其余设置，以及一个自己引入的相位 bug（11-09）

§9.92 的对照表列出三项未对齐。这一节补完其中两项，并在验证过程中发现我自己写的一个
bug，它使 §9.92 的结论偏低了近 3 倍。

### 补上的两项

**1. `critic_noise` 独立的 10 步。** 参考 `dsrl.py:326-335` 把它放在 SAC 循环**之外**，
每次重新 `replay_buffer.sample`。我原来放在 `--utd` 循环里，20 次更新共用同一批 128 个状态。
现在按参考拆成独立循环（`--noise-critic-steps 10`），每个 batch 新抽。额外成本几乎为零
（450 → 442 ms/epoch，因为 critic 前向只要 0.2 ms）。

**2. actor 换成 SB3 的 `SquashedDiagGaussian`。** 参考里 `DSRL.actor` 就是
`self.policy.actor`，即：

* **可学习**的逐维 `log_std`（初值 0）；
* `action = tanh(mean + std * eps)`；
* `log_prob = Normal(gaussian).log_prob − Σ log(1 − action² + 1e-6)`。

第三行是我之前漏掉的关键。原实现算的是 **pre-squash** 变量的密度，而那是另一个随机变量。
单元测试直接量出来：

| | 旧（无界） | 新（tanh 压缩） |
|---|---|---|
| `log_prob` 均值 | −45.4 | **−19.4** |
| `log_prob ≤ 0` | 否 | **是**（有界区间上的密度不能超过均匀分布） |
| `|u|` 上界 | 无界 | **0.9996**（`tanh` 的界） |
| `alpha` 在 `target_ent=0` 下 200 步 | 0.99 → 0.99（无作用） | 0.99 → **0.135**（真在调节） |

`target_entropy` 随之从 +45.4 改成参考的 **0.0**。SB3 的 `-dim` 假设 tanh 压缩后每维密度可以
超过 1；这里直接用参考值 0。

`critic_noise` 看到的输入也按 `scale_action` 的语义改成单元尺度（`noise / action_magnitude`），
采样器则收到 `action_magnitude × tanh(·)`，两者不再混用。

### 一个自己引入的 bug：回报表与池子相位不一致

跑完 8000 epoch，`Q` 收敛到 **3.93**，而 episode 级回报是 16.57，看起来只有 24 %。我准备
把它写成"又一个停滞"。先做了一件本该更早做的事：**把池子的真实回报和 `Q` 直接对比**。

结果很刺眼：

```
TRUE discounted return G on this pool:
  mean 1.11   median 0.00   p10 0.00   p90 0.00   max 24.81
```

池子里有 63 个段，每段内部都含奖励（例如某段有 4 个），而**真实回报的中位数是 0**。这不可能是
数据的问题，只能是代码的问题。

原因是相位：

```python
# 旧代码
for episode in range(len(labels)):
    start, stop = bounds[episode]
    path = list(range(start, stop, stride))   # 只覆盖与 start 同余的行
    ...
    return_all[row] = running
```

而池子是从 `adjacent` 里抽的，`adjacent` 只要求 `frame_index` 相差 `stride`，**与 episode 起点
没有相位关系**。所以 16 行里有 15 行落在网格之外，`return_all` 保持初始化值 0。而段尾锚点正是
读 `return_all`——**64 个锚点里有 63 个被告诉"答案是 0"**。

修法是在**每一行**上递推：

```python
for row in range(len(episodes) - 1, -1, -1):
    successor = row + stride
    if successor < episode_end[episodes[row]]:
        return_all[row] = step_rewards[row] + gamma * return_all[successor]
    else:
        return_all[row] = step_rewards[row]
```

修复前后：

| | 修复前 | 修复后 |
|---|---|---|
| 池内真实回报均值 | 1.11 | **9.56** |
| 中位数 | **0.00** | 10.26 |
| 非零比例 | 8 % | **72 %** |
| **锚点（段尾）回报均值** | ~0 | **12.80（100 % 非零）** |

### 修复后的曲线：Q 追上参考量级

`--transitions 4096 --segment-length 64 --step-stride 16 --gamma 0.99 --utd 20
--noise-critic-steps 10`，8000 epoch：

| epoch | Q | minQ | actor loss | alpha | σ | critic loss |
|---|---|---|---|---|---|---|
| 1 | 0.16 | −0.25 | −21.03 | 0.994 | 0.994 | 3.82 |
| 1000 | 2.38 | 2.14 | −2.14 | 0.006 | 0.748 | 0.87 |
| 2000 | 3.35 | 1.44 | −1.46 | 0.005 | 0.742 | 0.99 |
| 3000 | 4.87 | 1.64 | −1.64 | 0.011 | 0.740 | 0.66 |
| 4000 | 5.76 | 3.16 | −3.18 | 0.014 | 0.731 | 0.31 |
| 5000 | 6.99 | 5.27 | −5.28 | 0.020 | 0.731 | 0.09 |
| 6000 | 7.78 | 6.63 | −6.62 | 0.023 | 0.732 | 0.03 |
| 7000 | 7.86 | 8.29 | −8.31 | 0.022 | 0.730 | 0.02 |
| 7500 | 8.85 | 9.05 | −9.06 | 0.023 | 0.730 | 0.02 |
| 8000 | 7.72 | 8.64 | −8.62 | 0.023 | 0.730 | 0.02 |

对照同一配置但带相位 bug 的 v4：epoch 4500 时 v4 的 Q 是 3.77，v5 是 6.35。**同一个"停滞"
的读数是 bug 造成的，不是方法的上限。**

这一版第一次让三个指标同时朝正确方向走：

* `Q` 升到池内真实均值的 **81 %**（峰值 8.85，即 **93 %**），不再是 24 %；
* `minQ` 从 1.4 升到 8.6 并紧跟 `Q`（末段 `minQ/Q` 在 1.0–1.1），两个 head 一致；
* `actor loss` 从 −21.0 降到 −8.6，**方向与 Q 一致**——这是 §9.92 里缺失的那一条，现在有了；
* `alpha` 从 0.99 降到 0.023 后自行稳定，σ 从 0.99 降到 0.73 后稳定，说明熵控制器到位；
* `|u|` 稳定在 0.62，`tanh` 没有饱和；
* `critic loss` 从 3.8 单调降到 0.02。

**`minQ > Q` 在末段（6500 之后）出现，需要谨慎解读，不能当成"actor 变好了"。** 两者不是同一个
量：`Q` 是 TD critic 在**录制动作**上的输出，`minQ` 是 noise critic 在 **actor 采样噪声**上的输出。
`minQ/Q ≈ 1.05` 只说明 noise critic 对 actor 的噪声给出了略高于录制动作的估值，而这既可能来自
actor 真的找到了更好的噪声，也可能只是蒸馏链的残差。区分这两者只能靠仿真。


### 新增的自检，因为三个 bug 都"看起来像在学习"

| bug | 表面现象 | 靠什么发现 |
|---|---|---|
| 回报表相位错位 | Q 单调升到 3.93，曲线漂亮 | 池子真实回报的中位数是 0 |
| `stride` 被 `feature_stride` 覆写 | `row + stride` 变成 `row + 5` | 审计只有一处赋值 |
| 段尾掩码只覆盖一部分段 | Q 停在 −1.6 | 对照 `minQ < Q` 是否成立 |

三者共同点：**损失曲线和 Q 曲线都不报警**。所以现在每次启动都会打印

```
[dsrl] pool return: mean 9.56  median 10.26  p90 21.66  non-zero 72 %
[dsrl] anchor return (the tails): mean 12.80  median 13.80  non-zero 100 %  <- must not be ~0
```

并在锚点非零比例 < 50 % 时直接 WARNING。这两行是这一节唯一可复用的产出。

### 还剩什么

`Q` 到 74 % 的真实池均值，且仍在上升，但 5500→6000 只增 0.05，已接近该池能支撑的上限——
池内回报的覆盖只来自 64 个段尾。要再往上：加长段（更多链节）、或增大 `--transitions`
（更多锚点）。两者都只增加特征缓存的一次性成本。

真正的验收仍未做：**actor 产出的噪声在仿真里是否真的得分更高**。这需要一个评估回路，
把 actor 的 ε 送到采样器、在 2x10 bench 上跑成功率和 episode 得分，与"标准正态噪声"的基线比。
在此之前，上面所有数字只能说明 critic 学会了估值，不能说明策略变好了。

## 9.94 评估回路：把 actor 的噪声送进采样器（11-09）

### 参考实现是怎么做的

论文只写了"at inference time, we simply sample `w_single ~ π_single` and repeat it across the
action chunk axis"。落到代码是 `dsrl.py:452 predict_diffused`：

```python
unscaled_action, _ = self.policy.predict(observation, deterministic)
scaled_action = self.policy.scale_action(unscaled_action)   # -> buffer_action
action = self.policy.unscale_action(scaled_action)          # 与上一步互逆，回到 unscaled
action = self.diffusion_policy(obs, action.reshape(-1, C, d))
```

三点值得记下来：

1. **`scale_action` 与 `unscale_action` 互为逆运算**，所以采样器收到的就是 `tanh(gaussian)`，
   落在 `(-1, 1)`。`action_magnitude` 真正影响的是**回放缓冲区/critic 看到的尺度**，不是采样器。
2. **评估就是完整跑一遍** `utils.py:113 evaluate`：`obs → actor → noise → diffusion_policy → env.step`，
   报 success_rate 与 avg_reward。它**不查 critic**，这正是"评估独立于训练 actor 的那个 critic"的原因。
3. `deterministic_eval` 是个开关，两条曲线都记录（`success_rate` 与 `success_rate_deterministic`）。

### 我这边的一个结构性优势

参考实现的 actor 吃 23 维本体（robomimic）。我的 actor 吃
`[proprio(16), VLM 2048, visual(256)]` —— 而**这三项都是采样器本来就要算的东西**：

* VLM 2048 维 = `prefix_hidden[:, 最后一个非 pad]`，就在 `embed_prefix` 的那次前向里；
* proprio = `preprocess` 已经归一化好的 `observation.state`；
* visual 256 = `features_v9` 的 256 维投影，而它的 8192 维原值可以从 `prefix_embs` **切片得到**
  （三段相机嵌入 + 一段语言嵌入，语言那段要除掉 `embed_prefix` 乘上的 `sqrt(2048)`）。

所以**转向不需要额外的 VLA 前向**，代价只有一个小 MLP。这一点使评估成本与基线几乎相同。

### 抽出共享模块，而不是复制

`scripts/recap/steering.py` 现在是 `encode_prefix` / `sample_with_prefix` / `repeat_noise` /
`NoiseActor` / `build_observation` 的**唯一实现**，训练脚本与 bridge 适配器都从它导入。

这不是整洁问题。两份 `encode_prefix` 必然漂移，而漂移在这里**不会报错**——它会表现为
"actor 没有帮助"，恰好是这次实验要测的那个结论。

重构后做了一次回归：同种子 1 个 epoch 的指纹是
`critic 3.8249 / noise_critic 0.0791 / actor −21.0285 / alpha 0.9943 / Q 0.155 / |u| 0.555 /
sigma 0.994`，与重构前逐位相同。

### 两道 gate，都必须先过

评估一个"没有差异"的结果有三种完全不同的解释：actor 真的没学到、actor 退化了、或者 actor 看到的
观测和训练时不一样。episode 得分无法区分它们，所以先过两道 gate。

**Gate 1：实时特征 == 缓存特征**（`verify_visual_feature.py`）

| 层次 | 误差 |
|---|---|
| 8192 维 raw 向量最大绝对误差 | 0.0621 |
| 256 维投影后最大绝对误差 | 0.0197 |
| **相对误差（÷投影后标准差）** | **0.0017** |

0.17 %。raw 那层的 0.06 是 float16 存储的量化步长，不是算法差异。

**Gate 2：actor 不是惰性的、也没有饱和**（`probe_steering_effect.py`）

| 量 | 值 | 含义 |
|---|---|---|
| chunk 尺度（默认噪声下的 std） | 0.5178 | 动作空间的自然尺度 |
| **换一次默认噪声造成的变化** | **0.0150（2.9 %）** | 噪声对动作的总杠杆 |
| actor 噪声 vs 默认噪声 | 0.0318（6.1 %） | actor 的影响 |
| 跨观测差异 | 0.1277（24.7 %） | 观测的影响 |
| **actor 影响 / 换噪声的影响** | **2.1×** | 量级合理 |
| tanh 饱和比例 (\|u\| > 0.999) | 0.00 % | 没有 §9.88 那种饱和 |
| \|u\| 均值 | 0.5051 | 不贴边界 |

第二行是这次最该记住的数字：**换一次噪声只让动作变化 2.9 %**，与 §9.90 记录的
"同一帧换噪声造成 3.9 % 的动作偏差"一致。也就是说，**DSRL 在这个任务上能动用的杠杆本身就只有
3–4 %**。这给整个方法定了一个上限——与 §9.85 的"失败点在哪"是同一类约束，但这是从
噪声侧量出来的。

### 三臂设计

| 臂 | 采样器的 `x0` | 回答 |
|---|---|---|
| `default` | `N(0,1)`，策略自己的 | 策略本来就怎样 |
| `tanh_random` | `tanh(N(0,1))` 沿 chunk 轴复制 | **仅仅把噪声有界化**是否有用 |
| `actor` | 训练好的 DSRL actor（确定性） | actor 是否学到了边界之外的东西 |

中间那臂是让实验可解释的关键。actor 的输出天然落在 `(-1,1)`，所以 `actor` 与 `default` 同时
差在两件事上：边界和内容。如果 `tanh_random` 已经追平或超过 `default`，那么 `actor` 的任何
增益都是"有界"带来的，而那不需要 8000 epoch 的 SAC。

三臂共用 `eval_conditioned.sh`（即 §9.80 那条"一个 harness 给所有臂"），种子 300+ 固定，
所以场景逐配对，差异是差异而不是三把尺子的采样。

**这一版能测什么、不能测什么，写在这里**：每个臂自己抽噪声，所以噪声实现不配对。
场景内得分的 sd 约 3.34 Cookies，16 个种子给出的配对差值 SEM 约 1.2。也就是说，
**它能发现大效应，不能证明小效应不存在**。

### 结果：负面，且是大幅负面

| 臂 | 平均 | sd | sem | 严格成功率 |
|---|---|---|---|---|
| `default`（`N(0,1)`） | **12.44** | 6.66 | 1.67 | 18.8 %（3/16）|
| `tanh_random` | 7.38 | 4.73 | 1.18 | 0 % |
| `actor` | **0.25** | 0.68 | 0.17 | **0 %**（14/16 得 0，2/16 得 2）|

| 对比 | 差值 | sem | t | 胜/平/负 |
|---|---|---|---|---|
| `actor − default` | **−12.19** | 1.67 | **−7.28** | 0 / 1 / 15 |
| `tanh_random − default` | −5.06 | 2.09 | −2.42 | 2 / 3 / 11 |
| `actor − tanh_random` | **−7.12** | 1.21 | **−5.90** | 0 / 3 / 13 |

`|t| = 5.9` 与 `7.3` 远超 2，不是噪声。**actor 不是"没提升"，是基本把策略弄坏了。**

### 先排除"actor 坏了"这个解释

得 0 分有几种可能，逐一测过：

| 检查 | 结果 | 结论 |
|---|---|---|
| actor 的输出塌成常数？ | 末层 `|b|` 范围 `[-0.53, 0.57]`，sigma 0.72 | 没有塌 |
| actor 忽略观测？ | 跨帧 std **0.366**；变化 <0.02 的维度占 **0.00 %** | 确实随观测变 |
| 饱和在 tanh 边界？ | `|u|>0.999` 占 **0 %**，`|u|` 均值 0.465 | 没有饱和 |
| 出现了 NaN/Inf？ | `finite: true` | 没有 |
| 观测与训练时不一致？ | 相对误差 **0.17 %** | 一致 |
| actor 对动作的影响太小？ | 是重抽噪声的 **2.1 倍** | 不是惰性 |

六项全部正常。**所以问题不在 actor 的实现，而在它输出的分布本身。**

### 真正的原因：tanh 把噪声收窄了，而策略是在 `N(0,1)` 上训练的

同一个观测、同一批 8 帧，三种先验的实际形状：

| 先验 | `\|u\|` 均值 | 逐维 std | `\|u\|>0.9` 的维数（共 256） |
|---|---|---|---|
| `N(0,1)`（策略训练时的） | **0.7848** | **0.9719** | 94 |
| `tanh_random` | 0.5616 | 0.6312 | 37 |
| actor | **0.4646** | **0.5537** | 38 |

`tanh(N(0,1))` 的 `|u|` 均值是 0.56，只有 `N(0,1)` 的 72 %。actor 更窄，0.46，只有 59 %。
flow-matching 的采样器是在 `x0 ~ N(0,1)` 上预训练的，把初始噪声系统性收窄等于把它推到分布外：
逐步去噪的轨迹整体偏向"接近条件均值"的那一支，动作幅度被压小，二十个饼干搬不动几个。

这个解释同时命中了三件事：

* `tanh_random` 只跌 5 分（形状对了一半，仍是有界且偏窄）；
* `actor` 跌 12 分（更窄，且额外带着 critic 的偏好）；
* `probe_steering_effect` 里"actor 影响 = 2.1 × 重抽噪声"——影响够大，方向是错的。

### 一个必须与参考实现对照的细节

参考实现也是 tanh 输出的。`dsrl.py:452 predict_diffused` 里 `scale_action` 与
`unscale_action` 互为逆运算，所以 `diffusion_policy` 收到的确实是 `tanh(·) ∈ (-1,1)`，
而不是 `N(0,1)`。那它为什么不崩？

差别在**宽度**。参考的 `action_magnitude = 1.5`，而后处理 `unscale_action` 把
`tanh` 的输出映射到 `[-1.5, 1.5]`：

```
unscaled = tanh(.)                    # [-1, 1],  |u| ~ 0.56
scaled   = scale_action(unscaled)     # = unscaled / 1.5,  |u| ~ 0.37  <- critic 看到的
action   = unscale_action(scaled)     # = unscaled,  |u| ~ 0.56        <- 采样器看到的
```

**这是我自己早先的判断，现在看是错的**：我此前把 `action_magnitude` 读成"只影响 critic"，
但 SB3 的 `scale_action`/`unscale_action` 对**对称**区间恰好互逆，所以真正到达采样器的是
`tanh` 输出，幅值 1.0。而 `N(0,1)` 的 `|u|` 均值是 0.785——也就是说参考实现喂给采样器的
噪声比 `N(0,1)` **更窄**，与我的实现同一类偏差。

差别只在程度：参考的 diffusion policy 是 DPPO 在 robomimic 上预训练的，容忍度未知；
π0.5 的 flow-matching 头在这里实测不容忍（`tanh_random` 就掉 5 分）。
**这是"参考实现在其任务上可行"与"同样做法在本任务的 π0.5 上可行"之间的差别，不是我的实现 bug。**

### 结论

**在这套 π0.5 + 2x10 设置上，用有界噪声替代 `N(0,1)` 会直接损害性能，DSRL 学到的 actor 在
此基础上进一步恶化。** `actor − default = −12.19`，`|t| = 7.28`。

这不是"训练不够久"或"超参没调好"能解释的：actor 的六项健康检查全部通过，它对动作的影响
是重抽噪声的 2.1 倍（够大），而它的输出分布与策略预训练时的先验差了 41 %（`|u|` 0.465 vs
0.785）。**方向对不上，幅度越大越糟。**

### 下一步该测什么，以及为什么不是我现在的猜测

上面所有推论都指向"噪声太窄"，但那仍是推论。能把它变成结论的最小实验是**加宽**：
`tanh_random` 乘以一个 `s` 使 `|u|` 均值回到 0.785（需要 `s ≈ 1.4`），
如果得分回到 12 分附近，就确认是宽度问题；如果仍然低，那"有界"本身才是问题，加宽无用。

这个实验只需一臂（约 50 分钟），且它决定后面所有工作：
* 若是宽度问题 → actor 的训练目标里必须加一项把输出分布锚在 `N(0,1)` 上（参考实现在
  附录 B 里提到 `bW ∈ [1,3]` 的动作幅值约束，正是这个作用的另一种写法）；
* 若是有界本身的问题 → DSRL 在 π0.5 上需要换参数化，即让 actor 输出 `N(0,1)` 的
  **均值偏移**而不是有界动作，这也更贴近 flow-matching 的用法。

## 9.95 判定实验：宽度假设被证伪，问题是"尾部"不是"宽度"（11-09）

### 先做的那个判定实验，以及它推翻的东西

§9.94 的推断是"两边都比 `N(0,1)` 窄，而更窄的那个更差，所以宽度是原因"。这个推断有一个
明摆着的漏洞：**窄和"是 actor"是混杂的**，窄本身不能解释那 −12 分。

判定实验就是把宽度对齐再看：

| 臂 | 先验 | 平均 | n |
|---|---|---|---|
| `default` | `N(0,1)` | **12.44** | 16 |
| `tanh_random` | `tanh(N(0,1))` | 7.38 | 16 |
| `tanh_wide` | **`1.340 · tanh(N(0,1))`**（宽度已对齐 0.763） | **0.70** | 10 |

**加宽不但没救回来，还更差**（0.70 对 7.38）。宽度假设被证伪。

### 为什么加宽会适得其反：一阶矩不等于形状

同一批 16 帧上四种先验的矩和分位数：

| 先验 | `\|u\|`均值 | std | p99 | **max** | **`\|x\|>2` 占比** |
|---|---|---|---|---|---|
| `N(0,1)` | 0.798 | 1.000 | 2.577 | **5.27** | **4.55 %** |
| `tanh(N(0,1))` | 0.556 | 0.628 | 0.989 | **1.00** | **0 %** |
| `1.34 · tanh` | 0.745 | 0.841 | 1.325 | **1.34** | **0 %** |
| `actor`（tanh） | 0.510 | 0.592 | 0.985 | 0.99 | 0 % |

`1.34 · tanh` 的 `|u|` 均值 0.745，与 `N(0,1)` 的 0.798 已经很接近——**但它的支撑是
`[-1.34, 1.34]`，一条尾巴都没有**。而且加宽把原本堆在 `|u|≈1` 的质量（tanh 在 0.99 以外只有
0.8 %）推到 `|u|≈1.33`，正好是**硬边界**上。`N(0,1)` 在那个位置有 18 % 的质量继续向外延伸。

所以 `1.34 · tanh` 在 `|u|` 意义上的"宽度"对上了，在**分布意义上离高斯更远了**：
一阶矩匹配，尾部为零，边界处有堆积。

**这就是它得 0.70 的原因。** flow-matching 采样器在 `x0 ~ N(0,1)` 上预训练，ODE 的积分路径
依赖初始点的完整分布，不是一个均值。截断支撑等于把轨迹的每一条都拉向"接近条件均值"的
那一支，而且拉得比未加宽的版本更狠（因为边界更远、质量更集中）。

### 一个我该更早做的检查

`tanh` 的支撑是 `(-1, 1)`，这在写代码的第一分钟就是已知的。`N(0,1)` 有 4.55 % 的质量在
`|x| > 2`，也是查表可得。**"用 tanh 的输出替代高斯噪声"这件事，在任何测量之前就能看出分布
不匹配**，而我先测了均值、发现对不上、然后去修均值——把两个明显不同的分布当成"宽度不同"。

代价是一轮约 50 分钟的实验和一个被证伪的假设。教训不是"多测几个矩"，而是：
**替换一个概率分布的支撑时，先画出两个分布，而不是先算它的一个统计量。**

### 形状正确的方案：只平移，不截断

```
u = ε + shift_scale · m(s),      ε ~ N(0, 1),   m = actor 的 pre-tanh 均值
```

实测（α = 0.3）：

| 先验 | `\|u\|`均值 | std | p99 | max | `\|x\|>2` |
|---|---|---|---|---|---|
| `N(0,1)` | 0.791 | 1.009 | 2.749 | 3.56 | 4.88 % |
| **`mean_shift` α=0.3** | 0.868 | 1.070 | 2.443 | **3.67** | **5.66 %** |
| `actor`（tanh） | 0.510 | 0.592 | 0.985 | 0.99 | 0 % |

**尾部保住了，整体只宽了 6–9 %**，而位置的偏移随观测变化（跨帧 std 0.613）。

actor 的 pre-tanh 均值本身就落在几乎正确的尺度上（`|m|` 均值 0.766、std 1.012），所以
`α` 只需要 0.3 就能让边际分布几乎不变。这是这个方案唯一真正的优点：
**它不需要修改 actor 的训练目标，只需要改推理时怎么用它的输出。**

### 正在跑的两臂，以及为什么第二臂是关键

| 臂 | shift | 回答 |
|---|---|---|
| `ms_actor` | `0.3 · m(s)`，随观测变化 | 形状对了之后，转向有没有用？ |
| `ms_fixed` | `0.3 · m(s₀)`，**第一次请求后冻结** | 增益是否只来自"把先验挪了地方"？ |

第二臂是必须有的对照：冻结的向量与 actor 在某个状态上的输出**完全相同**（就是那个输出），
唯一的区别是它不再随观测变化。若两臂得分相同，则 actor 的逐状态信息毫无价值，
"训练 8000 epoch 得到的只是一个大致的偏移"。

### 结果：两个方向相反的强信号

8 个配对种子（300–307），同一 harness：

| seed | `default` | `ms_actor` | `ms_fixed` |
|---|---|---|---|
| 300 | 19 | 5 | 0 |
| 301 | 4 | 6 | 0 |
| 302 | 18 | 12 | 0 |
| 303 | 15 | 5 | 0 |
| 304 | 20 | 16 | 0 |
| 305 | 20 | 16 | 0 |
| 306 | 5 | 6 | 0 |
| 307 | 10 | 13 | 0 |
| **平均** | **13.88** | **9.88** | **0.00** |

| 对比 | 差值 | sem | t | 胜/平/负 |
|---|---|---|---|---|
| `ms_actor − default` | **−4.00** | 2.11 | −1.89 | 3 / 0 / 5 |
| `ms_fixed − default` | **−13.88** | 2.36 | −5.89 | 0 / 0 / 8 |
| `ms_actor − ms_fixed` | **+9.88** | 1.73 | **+5.72** | **8 / 0 / 0** |

### 这两半各自说明什么，以及合起来说明什么

**`ms_actor − ms_fixed = +9.88`，8/8 全胜，t = 5.72。** 这是整轮实验里最强的信号。冻结臂与
actor 臂的偏移向量在幅度和形状上完全相同（它**就是** actor 在某个状态上的输出），
唯一差别是它不随观测变。它掉到 0 说明：**actor 的逐状态信息确有价值**，
"训练出来只是一个大致的常数偏移"这个假设被排除。

**`ms_actor − default = −4.00`，t = −1.89。** 未过 2，严格说不能断定负效应。但胜负的构成
（3 胜全落在基线低的种子上，5 负全落在基线高的种子上）指向一个具体的机制：

| seed | 基线 | `ms_actor` | 变化 |
|---|---|---|---|
| 300 | 19 | 5 | −14 |
| 302 | 18 | 12 | −6 |
| 303 | 15 | 5 | −10 |
| 304 | 20 | 16 | −4 |
| 305 | 20 | 16 | −4 |
| 301 | 4 | 6 | +2 |
| 306 | 5 | 6 | +1 |
| 307 | 10 | 13 | +3 |

**偏移把采样点从策略偏好的位置推开**：好场景被拉低，差场景被略救。净效应为负。

合起来：**actor 学到了"与观测相关"的东西，但没学到"该往哪里推"。** 一个随观测变化的
无意义偏移（在这份数据上）和一个常数偏移会得到几乎一样的 0 分吗？不会——`ms_fixed` 得 0，
`ms_actor` 得 9.88，说明 actor 的输出不是噪声。但它朝着一个平均而言不如原点的地方推。

### 一个必须一起考虑的约束

`probe_steering_effect` 测出：**重抽一次噪声只让动作变化 2.9 % 的动作标准差**。
这给整个方法定了一个上限——噪声这个旋钮本来就很轻。`shift_scale = 0.3` 是从"保持分布形状"
推出来的（边际 std 1.070，比基线宽 6 %），不是从"最大化收益"推出来的；
没有任何理由相信它是收益最大的那个值。

但这一轮已经能排除的，比"这个 α 不好"更重要：**三种参数化（tanh、加宽 tanh、均值平移）
在正确配置下全部不如不转向**。而其中最好的那个（`ms_actor`, 9.88）恰恰是唯一保持
`N(0,1)` 形状的那个。所以参数化不是无关的，但它也不是充分的。

### 三轮实验的完整账，写在一起

| 先验 | 形状 | 平均 | vs 基线 | 结论 |
|---|---|---|---|---|
| `N(0,1)` | 高斯，有尾 | **13.88** | — | 基线 |
| `tanh(N(0,1))` | 截断 @1.0，**无尾** | 7.38 | −5.06 | 截断本身有害 |
| `1.34·tanh` | 截断 @1.34，**无尾** | 0.70 | — | **加宽反而更差**（证伪宽度假设）|
| `actor`(tanh) | 截断 @1.0，无尾 | 0.25 | −12.19 | 截断 + 错误方向 |
| `ε+0.3m(s)` | **高斯，有尾** | 9.88 | −4.00 | 形状对，方向不对 |
| `ε+0.3m(s₀)` 冻结 | **高斯，有尾** | 0.00 | −13.88 | 常数偏移完全失效 |

**形状是必要条件，不是充分条件。** 两个保持高斯尾部的臂（9.88 和 0.00）差距 9.88 分，
说明形状对了之后，"偏移什么"仍然决定一切；而两个截断的臂都 ≤ 7.38，说明形状不对时
连"偏移什么"都救不回来。

### 当前判断

在这套 π0.5 + 2x10 设置上，**DSRL 学到的 actor 尚未带来任何收益**，三种参数化、共
约 4 小时的评估都指向同一结论。可复现的正面证据只有一条：`ms_actor` 显著优于
`ms_fixed`（t = 5.72），即 actor 的输出确实携带观测信息。

要让它变成收益，最可能的缺口不在参数化，而在 critic 的目标——§9.91–9.93 已经把
critic 的 TD 目标换成 MC、修过相位 bug，`Q` 也追到了池内真实均值的 81 %，但
**`argmax_ε Q(s,ε)` 给出的 ε 未必真的更好**。这一点从未被独立验证过：
现在的证据只说明 Q 的*数值*对，不说明它的*排序*对。








---

## 9.96 排序探针：离线 critic 的排序能力等于抛硬币，这不是 bug（11-09）

§9.95 结尾留下的那句话是"`argmax_ε Q(s,ε)` 给出的 ε 未必真的更好，这一点从未被独立验证过"。
这一节就是那个验证，而它否掉的是整条离线路线，不是某一个超参。

### 探针

`probe_critic_ranking.py` 用 `branches/*.npz`（14 个分片、100 个分支点、其中 93 个有效、
共 787 个分支、45 个"结局不同"的点、823 个 discordant pair），问一个二选一的问题：

> 同一个状态的两条分支，critic 打分高的那条，真实回报更高吗？

ties 记 0.5，bootstrap 按**点**重采样（823 个 pair 来自 45 个点，按 pair 重采样会把相关
性当成独立证据）。

| checkpoint | 同状态成对准确率 | 95% CI |
|---|---|---|
| v5 (tanh) | **0.508** | [0.443, 0.573] |
| v6 (tanh, 修好相位) | **0.508** | [0.440, 0.571] |

**两个都是 0.508，区间都盖住 0.5。**

### 为什么这不是可以修的 bug

离线池里**每个状态只有一个动作**。critic 在那个 (s, a) 上可以拟合到小数点后三位
（§9.93：Q 0.285 → 7.98，critic loss 3.91 → 0.003），但"同一状态下两个动作谁更好"这个
问题，池里根本没有信息可以回答它。这不是收敛不够，是**数据结构里没有这个自由度**。

论文也没有这么训：§5.1 全程在线，§5.3 明确说 naive offline baseline
"not able to learn at all"。所以 0.508 是那个实验的**正确答案**，而不是需要调参的症状。

### 一个反复出现的模式

这个项目至今找到的三个"看起来像正常曲线"的 bug——空填充、被遮蔽的 `stride`、相位错位的
回报表——和这次的 0.508 属于同一类。防御手段不是看 loss 曲线，是**外部量**：

- 池内真实回报、锚点回报、锚点非零比例（离线训练器启动时打印）
- 同状态成对准确率（本节）
- 与基线策略相同的奖励率（§9.97）

---

## 9.97 重建为在线：先量清楚代价，再写循环（11-09）

决定做在线之后，第一件事不是写循环，是把"在线到底有多贵"量出来。四个数决定了整份实现：

| 问题 | 测量 | 结论 |
|---|---|---|
| 物理一步多贵 | `mj_step` **1.12 ms**，但控制周期 `timestep=0.001` × 50 substep | 一个控制步 ≈ **61 ms**，这是 benchmark 自身的配置，不能改 |
| MuJoCo 放 GIL 吗 | 4 线程 **7.94×** 于串行，轨迹**逐位一致** | 物理可以并行 |
| 渲染能进线程吗 | 第二个线程渲染 `eglMakeCurrent` → **`EGL_BAD_ACCESS`**，线程直接死 | 不能。MuJoCo 只有一个全局 EGL context |
| 自建 renderer 和 env 自带的图一样吗 | 三路相机 **max\|diff\| = 0** | 一样，可以主线程按需渲染 |

外加批量前向的次线性：batch 1 = 183 ms，4 = 299 ms，**8 = 437 ms**。所以八个场景基本只付
一次前向的钱。

### 于是架构是

物理在 `ThreadPoolExecutor` 里跑、`render_cameras=False`；主线程每轮渲染 8×3 张图并做一次
批量前向；梯度步和物理等待重叠。

### 但物理的代价不是常数

第一轮 8 个场景的物理是 **1,164 ms**（≈ 一个场景的钱，说明并行完美），到 episode 中段
变成 **4–9 s**。第一版实现里这就是"吞吐从 78 掉到 16 control steps/s"的全部原因。

`probe_step_slowdown.py` 用重放定死了归因：把每一步的 `(qpos, qvel, ctrl)` 存下来，之后
把最快和最慢的六步原样喂回求解器——

| | 原始 | `mj_step`×50 | `mj_forward` | `env.step` |
|---|---|---|---|---|
| 最快 6 步 | 60.3 ms | **57.3** | 1.1 | 64.9 |
| 最慢 6 步 | 580.7 ms | **570.9** | 12.7 | 483.1 |

**95–98 % 的控制步就是求解器本身，而且变慢的是求解器。** `nefc` 只从 1,980 涨到 2,379
（+20 %）却带来 10× 的耗时，所以约束行数不是原因；是 Newton + `iterations=100` 在夹爪拖着
80 个自由刚体时不能快速收敛。

靠对照排掉的三个候选：**时间**（合成动作跑满 1500 步恒为 61 ms）、**线程争用**（单场景同样
退化）、**CPU 配额**（16 核，无其他进程）。

### 因此诚实的吞吐（实测，不是推算）

- 一轮 = 8 场景 × 16 控制步。**episode 开始时 1.2 s**（并行完美），**中段 4–10 s**，
  **episode 结束那一轮 ~30 s**——因为要关掉 8 个 env 再建 8 个新的（每个 1.36 s，串行），
  而这正是"训练和评估看到同一个场景分布"要付的代价（§9.97 开头的漂移测量）。
- 前 50 轮平均 **18.3 s/轮** → **7 scene-steps/s** → `--scenes 8` 约 **135 episodes/hour**。
- 200 分钟 ≈ **100 个 episode**。

§9.96 之前写过的"81 control steps/s、211 episodes/hour"是**首轮**的数字，
"~20 control steps/s、50–60 episodes/hour"是把中段物理当稳态的数字。两个都错了，
错在同一个地方：**拿一个时刻的物理耗时去外推整集**。真实值随 episode 相位变化 4 倍。

另一个自己踩过的测量坑：用 `grep -c "^\[online\] r"` 数轮次，而这个模式同时匹配了
`[online] replay` 和 `[online] reward` 两行，于是"300 秒跑了 1 轮"这个结论是假的。
数轮次要锚定行尾，或者直接用 `--log-every 1`。

### 梯度步是免费的

`--utd 20 --batch 256` 一批梯度步 120–160 ms，全部藏在 4–9 s 的物理等待里。所以没有任何
理由为了速度降 `--utd`。

---

## 9.98 在线循环跑的第一次，以及两个我自己引入的错（11-09）

管线跑通（60 轮 / 480 transition / 16 episode / 7.8 分钟），但过程中暴露出两个错，都是
**离线版已经修过、我在新文件里又犯了一遍**的同一类错。

### 错一：把熵项打开，Q 被抬了两个数量级

`train_dsrl.py` 里 `--entropy-in-target` 默认**关闭**，理由是：该项是每次 bootstrap 的奖励，
在折扣下复合成 `alpha·|log pi|/(1-gamma)`；这里 `alpha≈1`、`log pi≈-21`、`gamma=0.99`，所以
和是 **≈ 2100**，而真实回报只有几十个饼干。我把它设成默认打开了，于是：

| 轮 | `q` | `target_q` | 奖励均值 |
|---|---|---|---|
| 8 | 20.41 | 26.16 | 0.000 |
| 16 | 35.46 | 35.82 | 0.188 |
| 24 | 41.29 | 41.36 | 0.146 |

Q 跟着 target 走，说明 TD 是"对"的——它正确地拟合了一个被平移过的目标。这不破坏 actor
（固定状态下加常数不改 argmax），但让日志里每个 Q 都无法读数，而这正是这份文件想守住的东西。
已改回默认关闭，理由写在 `--help` 里。

### 错二：`--max-control-steps` 完全没生效

环境自己的 `horizon` 来自 config，2x10 是 **6000**；benchmark 的 1500 步上限是在
`run_episode` 里靠计数实现的，那段代码这个循环不走。所以 `--max-control-steps 400` 跑了 640
步仍然 `ep 0`。已改成在 `Scene.play` 里显式计数。

**训练 episode 比被评估的 episode 长 4 倍**，这个错误的性质就是 §9.96 结尾说的那种：没有
报错、曲线正常、结论全错。

### 奖励的定义也换了，理由不是"照抄参考"

环境的 reward 是 `min(cookies_in_target / required, 1.0)`——一个**水平量**，不是增量。
`chunk_sum`（参考实现 `ActionChunkWrapper` 的行为）对它做积分，于是"把饼干早点放进箱子里"
会在之后每一步都被再付一次，回报跑到 ~1500，而最有价值的策略里有一部分是"把饼干留在箱里"。

默认改成 `increment`，即 `count[t+1] - count[t]`，单位是**饼干个数**。这样：

- 一个 episode 的回报**就是**评分（最高 20）
- Q 可以和离线池直接比（池均值 9.56，整集未折扣 22.76）

不除以 `required_cookies`，因为一除就等于把这份文件里每个数字都暗地里除以二十。

### 奖励信号的自检

唯一有意义的判据不是"非零比例有多高"，而是**和基线策略在同一个任务上的比率**：
`a3_clean_v9_30000` 每 1500 步得 13.18，即 0.00879 饼干/控制步、`execute=16` 下
**0.1406/RL 步**。一个未训练的噪声策略**就是**那个基线（它的噪声本来就是采样器的噪声）。

实测 **0.1300/RL 步 = 基线的 93 %**，非零比例 3.5 %。3.5 % 是这个任务的正确稀疏度，不是
异常——这一点一开始被我写成了 `if 非零 < 5%: WARNING`，是错的判据，已改成与基线比较。

---

## 9.99 在线训练开跑前必须写下来的天花板（11-09）

`probe_steering_effect` 已经量过：**重新抽一次噪声只让动作移动 2.9 % 的标准差**。这就是 DSRL
在这套设置上全部的权限。

在线能修的是 critic 的**排序**——那正是离线结构上做不到的（§9.96）。在线**不能**放大
那 2.9 %。

所以先写下来：**现实的天花板是 1–2 分**（基线 13.18）。如果跑完看到 +15 分，那是管线里还有
一个 bug，不是方法奏效了；如果看到 ±0.5 分，那是这个权限上限的必然结果，不是训练失败。
这两句话必须在看结果之前写下来，否则事后怎么说都"解释得通"。

---

## 9.100 物理代价的 83 % 是饼干本身，不是管线（11-09）

§9.99 之前那个"在线很贵"的结论需要拆开：贵在哪里。`probe_episode_cost.py` 用同一段动作
比对两种模型——原场景，以及把 80 个 cookie body 的 `contype`/`conaffinity` 清零（对求解器
不可见，动作完全相同）：

| 场景 | ms/control step | 六段计时 |
|---|---|---|
| 完整 | **60.5** | 61 61 60 61 60 60 |
| 饼干不参与求解 | **10.2** | 10 10 10 10 10 10 |
| 差值 | **50.3（83 %）** | |

**六个时间桶完全平坦**（61/61/60/61/60/60），所以这段动作没有产生"中段变慢"——它只是
**一开始就慢**，因为它从第 1 步就把夹爪推进饼干堆。而移除饼干后不但快了 6 倍，还**恒定**。

这就把 §9.97 的归因补全了：

- 不是"时间让求解器变慢"，是**接触数量**让它变慢
- 而接触数量由任务决定，不由管线决定
- 所以 83 % 的物理预算是 **2x10 这个任务的定义**，不是实现开销

### 这否掉了三条"优化"路径

1. **降 `substeps`**（50 → 更少）：会改控制周期和动力学，策略是在 50 上训的，不能用。
2. **降 `solver` 迭代**（100 → 更少）：同上，改的是被训练的那个动力学。
3. **截短 episode** 是唯一合法的杠杆，但要看奖励落在哪里。基线 v9 在 1500 步里得 13.18，
   首次放置的中位数在第 **226** 步、最后一次在第 **834** 步——所以 1500 步里有约 45 % 是
   最后一次放置之后的**空转**。截到 900 步会去掉这段时间而几乎不掉奖励，省 40 % 墙钟。

第 3 条是可用但没有采用的：训练和评估必须用同一个截断点，而评估沿用 1500 是既有基线，
换掉就没法和 13.18 比。这个取舍写在这里，不是遗漏。

---

## 9.101 预算的现实：200 分钟 = 5.6 个 episode 等价（11-09）

把所有实测拼起来：

| 量 | 值 | 来源 |
|---|---|---|
| 一轮（8 场景 × 16 控制步） | episode 开始 1.2 s，中段 10–13 s，换集那轮 ~30 s | 日志 `round cost` |
| 平均（机器空闲） | **11.2 s/轮** | 100 轮 / 1,113 s（扣掉 ~95 s 加载） |
| 一集 | 94 轮 = **17.5 分钟** | `max_control_steps=1500` / `execute=16` |
| 340 分钟 | **~19 个 episode 等价**（8 场景并行 → 共 ~155 个 episode） | 除法 |

> **这张表改过一次，原因是测量被自己污染。** 第一版写的是 22.9 s/轮、5.6 个 episode 等价，
> 那个数字是在**同时跑着三个探针**（`probe_episode_cost` 的两段 1200 步课程、
> `probe_online_critic` 的成对 rollout）的时候测的。8 个物理线程和探针的物理线程抢同样的
> 16 个核，于是训练的每一轮都被拖慢一倍。把机器让给训练之后，同样跑到第 100 轮只需要
> 1,113 秒而不是 2,300 秒。
>
> 这个错误的性质和 §9.96 结尾列的那些一样：**数出来了，但是数的是错的东西**。以后测吞吐必须
> 说明"测量时机器上还有什么在跑"，否则数字没有意义。

对照参考实现：robomimic can 一集 **75 个 RL 步**（`max_episode_steps=300` / `act_steps=4`），
它的 `init_rollout_steps=1501` 就是**约 20 个 episode** 的暖启动。这里 5.6 个 episode 等价等于
**530 个 RL 步**，连参考的暖启动都不到。

原因不是实现慢，而是两个任务的**每一步根本不是一个量级的东西**：

| | robomimic can | 这里 |
|---|---|---|
| 一步的物理 | 4 ms（一个 `mj_step`） | **1,700 ms**（50 substep × 34 ms 平均） |
| 一集 | 75 RL 步 | 94 RL 步 |
| 一集的实际耗时 | ~0.3 s | **36 分钟** |

**740 万倍的单步差距**，其中 83 % 是饼干（§9.100）。这是把 DSRL 搬到这个任务上必须承认的
事实：论文的算法可以直接用，论文的**预算**不能直接搬。

### 因此这次实验能回答什么、不能回答什么

- **能回答**：在线 critic 能否学出同状态的动作排序（§9.96 里离线做不到的那件事）。
  ~19 个 episode 等价、1,800 个 RL 步、`--utd 20` 下约 36,000 次梯度更新，足够看出
  `q` 是否跟随 `target_q`、`alpha` 是否退火、critic 在噪声方向上是否有曲率。
- **不能回答**：DSRL 能否把基线 13.18 提高。§9.102 量到 critic 的全部权限只有分数的
  2.2 %，而这个预算连跑完一个完整对照（8–16 个评估种子）都不够。

这两个结论必须分开陈述，否则一次 200 分钟的实验会被读成"方法无效"，而它根本没有检验
方法的有效性——它检验的是**管线是否连得上**。

---

## 9.102 critic 在噪声方向上有结构，但结构只有分数的 2.2 %（11-09）

### 为什么换了一个测法

`probe_online_critic.py` 的设计是成对 rollout：两种噪声各跑一段，比谁放进箱子的饼干多。
实测这条路走不通，而且失败得很干净：

| 窗口 | 10 个状态 × 2 种噪声 | 结果 |
|---|---|---|
| 120 步 | 20 次比较 | **20 次全部 0/0** |
| 300 步 | 20 次比较 | **20 次全部 0/0** |

原因是量级：训练中一集 1500 步才放进 5–7 个饼干，300 步里平均不到一个。于是每次比较都是
平局，而平局记 0.5 就让准确率**恰好**显示成 0.500——一个看起来像"抛硬币"但其实是
"没有发生任何事件"的数字。这是这个项目第四次踩同一个坑：**探针没有机会和探针测到零效应
长得一模一样**。

一次有效比较需要一整集（约 10 分钟），10 个状态就是 200 分钟——超过全部训练预算。

### 换成不需要 rollout 的问法

原本要回答的问题可以拆成两半：

- **(a)** `Q_ns(s, u)` 在固定 `s` 下，**跨 u 有没有变化**？
- **(b)** 有变化时，被偏好的 u 是否真的更好？

(a) 是网络和权重的性质，每个噪声一次前向，**几乎免费**；而如果 (a) 的答案是"几乎不变"，
(b) 无论怎样都不可能是真的。所以 (a) 是门。

`probe_critic_noise_sensitivity.py` 的结果（10 个状态 × 256 个噪声向量）：

| 量 | 值 |
|---|---|
| `std_u Q(s,·)` | **0.0541** |
| `max − min over u` | **0.2954** |
| 占 5.00 饼干一集的 | 5.9 % |
| 占 13.18 基线一集的 | **2.2 %** |

单位是饼干：一个饼干在 Q 里值 1.0。所以 `max−min` 就是 **critic 通过选噪声能拿到的全部
收益上限**——不是平均收益，是最好情况减最坏情况。

### 这个数字和 §9.95 是同一个

§9.95 用完全不同的方法量到：**重抽一次噪声只让动作移动 2.9 % 的标准差**。

| 测量 | 方法 | 上限 |
|---|---|---|
| §9.95 动作权限 | 重抽噪声，量动作位移 | **2.9 %** |
| §9.102 critic 权限 | 量 `Q` 在噪声方向的全幅，除以分数 | **2.2 %** |

两个互不相干的方法给出同一个量级。这不是巧合，是同一件事的两种看法：**噪声在这个采样器里
的杠杆只有几个百分点，所以价值函数在这个方向上的曲率也只有几个百分点，所以 `argmax_ε Q`
能拿到的也只有几个百分点。**

§9.99 写下的"1–2 分天花板"因此不是保守估计，是被两次独立测量支持的定量结论。

### 这次在线实验能回答的那半个问题

回到 §9.101 列的两件事：

- **能回答的**：在线 critic 是否学到了噪声方向上的排序。答案是**学到了结构**
  （`std_u Q = 0.054`，不为零），而且这个结构的大小与动作权限一致。离线版的 0.508 不是
  "学得不够"，是**离线数据里没有可学的曲率**——现在有了，只是曲率本身就小。
- **不能回答的**：能否把 13.18 提上去。2.2 % 的上限摆在这里，任何超过 +2 分的提升都必须先
  解释这 2.2 % 是怎么被打破的，否则是管线 bug。

### 训练到第 200 轮的实际数字

停在第 200 轮（1600 个 transition，16 个 episode，49 分钟）时的状态，全部健康：

| 量 | 第 25 轮 | 第 200 轮 |
|---|---|---|
| `q` | 0.997 | **1.545** |
| `target_q` | 0.918 | 1.726 |
| `critic_loss` | 0.325 | 0.330 |
| `noise_critic_loss` | 0.169 | 0.208 |
| `alpha` | 0.898 | **0.364**（在退火） |
| `log_prob` | −21.76 | −21.82（与离线版 −20.9 一致） |
| `u_abs` | 0.527 | 0.518（**未饱和**） |
| episode 回报 | — | 5.00 → **6.94** |

`q` 跟随 `target_q`、`alpha` 单调退火、`u_abs` 稳定在 0.52，说明 TD、熵控制和 tanh 参数化
三者都在正常工作。checkpoint 302 MB，`SteeringActor` 可直接加载（`actor_dim=32`,
`chunk=50`, `obs=2320`, `hidden=2048`）。

---

## 9.103 每集第一个 chunk 的历史窗口：训练是零，评估是重复首帧（11-09）

这是在这份实现里找到的第三个"同一份输入、两种含义"的错误，而且它只在 94 个 chunk 里的
**第 1 个**出现，所以任何看平均指标的方式都发现不了。

### 不一致在哪里

评估路径（`BridgeChunkPolicy.act`）的调用顺序是：

```python
def act(self, observation, task=""):
    self._advance_history(observation)        # ← 先追加当前帧
    if not self._queue:
        arrays["history"] = self._history_vector()   # 再取窗口
```

所以 episode 的第一步，窗口里已经有当前帧，`_history_vector` 把它**重复 10 次**填满。

而"每步之后追加"的循环结构（`Scene.play` 里 `env.step()` 之后才 append）在算第一个 chunk
时窗口是**空的**，`history_vector` 于是返回**全零**。

同一个 shape（480 维），两种含义：一个是"机器人静止在初始位姿"，一个是"10 帧都被重复的
当前位姿"。这正是这份代码库反复警告的那类输入。

### 修法

`Scene.seed_history(observation)`：窗口为空时追加当前帧，之后 `history_vector` 自动重复填满。
只在 episode 起点生效（`observe_all` 每轮都调，非空时是 no-op）。

验证：

```
empty -> all zeros: True
seeded: True
after seed, shape (480,) first frame [0 1 2 3] all same: True
second call is a no-op: True len 1
```

### 为什么为此重启了训练

它只影响 1/94 的 chunk，看起来可以"先让它跑完"。但这是**每集第一个决策**的系统性偏差，
而本项目的天花板只有 2.2 %（§9.102）——在这种量级下，"已知的 1 % 偏差"和"结果"分不开。

代价是重启丢掉约 30 分钟（占 450 分钟预算的 7 %）。换来的是一个可以在文档里写成"这次运行
用的是修好的版本"的结果，而不是"这次运行有个已知偏差，但大概不要紧"。

`dsrl_online_run2_prefix.pt` 保留了修复前的 checkpoint，运行记录分别在
`logs/online_run1.log`、`online_run2.log`、`online_run3.log`。

---

## 9.104 评估之前的预测：这个 actor 应当**低于**基线，而不是高于（11-09）

这一节必须写在看评估结果之前，否则事后怎么解释都通。

### 预测的两个数

在线训练器用的是 `NoiseActor`：`u = tanh(m(s) + sigma·z)`，`sigma` 学习得到，实测收敛到
**0.88**（§9.102 的日志）。所以它的输出分布大致等价于 `tanh(N(0,1))` 那一族。

§9.95 已经量过这一族：

| 臂 | 噪声分布 | 2x10 得分 |
|---|---|---|
| `default` | `N(0,1)` | **12.44** |
| `tanh_random` | `tanh(N(0,1))` | **7.38** |
| `tanh_wide` | `1.340·tanh(N(0,1))` | 0.70 |
| `actor`（v5, tanh 确定性） | `tanh(m(s))` | 0.25 |

**所以预测是：在线 actor 的得分在 0–7.4 之间，低于 12.44 的基线。**

### 为什么这条预测重要

如果评估结果是 **5±3**，那它和"在线训练有效/无效"**无关**——它测的是 tanh 参数化，而 tanh
参数化在 §9.95 就已经被单独判定为有害（因为流匹配头从没见过 `|eps|>1` 的区域）。

把这两件事混在一起是这类实验最容易犯的错：用一个已知有问题的参数化去测一个算法，然后把
参数化的问题记在算法账上。

### 因此这次在线实验真正检验的只有一件事

**在线数据能否让 critic 学到同状态的动作排序。** 这一点的答案是：能，但幅度只有分数的
2.2 %（§9.102）。两者合起来意味着——

- 即便 critic 完全学会排序，用一个 tanh actor 去用它，结果仍会落在 `tanh_random` 附近；
- 要用上那 2.2 %，actor 必须是 `mean_shift` 那一族（`u = eps + a·m(s)`，保持高斯尾部），
  而那不是 SAC——`mean_shift` 的输出不是一个分布，没有 `log_prob`，熵正则和目标熵都失去
  意义。它需要一个不同的目标函数。

这是一个方法层面的结论，不是调参能解决的：**DSRL 的论文形式（tanh + SAC 熵）在这个采样器
上有一个二选一——要么参数化对但算法不适用，要么算法适用但参数化有害。**

### 如果要真正验证 DSRL 在这套设置上有没有用

需要的不是更长的训练，而是：

1. 把 actor 换成 `mean_shift` 参数化；
2. 给它一个有意义的训练信号（例如对 `m(s)` 做确定性策略梯度，用一个学出来的
   `Q(s, m)`，而不是对 `tanh` 输出做 SAC）；
3. 在这个前提下跑够 8–16 个评估种子的对照。

这三步都不在本次运行的预算内，也不在本次运行的问题范围内。

---

## 9.105 撤回 §9.102 的 2.2 %，并撤回"actor 比随机好 4.7"（11-10）

这一节撤回我自己在 §9.102 写的结论，以及三小时前刚得到的一个"好消息"。两个都被同一件事
推翻：**critic 只在行为分布附近有定义。**

### 先说结果

`probe_actor_vs_random.py`（16 个状态，TD critic 打分）：

| 比较 | 差值 | t | 胜负 |
|---|---|---|---|
| actor − random | **+4.73 饼干** | +4.71 | 14 胜 / 2 负 |
| actor − tanh(mean) | −0.22 | −0.89 | 10 / 6 |

看起来是个大发现：+4.73 是 13.18 基线的 36 %。但同一张表里有八个**负数**：
`−5.64, −2.22, −1.94, −1.67, −1.49, −1.35, −0.51, −0.16`。

而这个任务的奖励是**饼干计数增量**，恒 ≥ 0，所以

$$
\text{reward}_t \ge 0 \;\Rightarrow\; G_t \ge 0 \;\Rightarrow\; Q(s,a) \ge 0
\quad\text{对任意 } s, a
$$

**负的 Q 是不可能的。** 六个负数不是评估，是外推。

### 定量确认

`probe_score_sanity.py`（12 个状态）：

| 噪声来源 | TD critic 打出负 Q 的状态数 | 动作范围 min/max | sd | 与 actor chunk 的距离 |
|---|---|---|---|---|
| actor | **0 / 12** | −1.041 / +1.018 | 0.559 | — |
| random | **6 / 12** | −1.114 / +1.198 | 0.572 | 0.122 |
| tanh(mean) | **0 / 12** | −1.040 / +1.014 | 0.561 | 0.032 |

关键在最后一列：随机噪声产生的 chunk 与 actor 的 chunk 只差 **0.12**（动作单位），动作范围
几乎一样。**它并不"离谱"**——但 critic 仍然打出负值。

参照同一个项目的既有测量：相邻控制步的动作差 **0.0021**，相隔 16 步差 **0.0250**。所以
0.12 是"同一段 chunk 内部变化"的 5 倍——对 16 维动作空间里的一个 MLP 来说，这就是外推。

**结论：`actor − random = +4.73` 无效。random 那一臂不在 critic 的定义域里。**

但它留下了一条有用的信息：**actor 的 chunk 全部落在 critic 能打分（非负）的区域**（0/12 负），
说明 actor 没有把自己的动作推到离谱的地方。这一点是好的。

### 撤回 §9.102

§9.102 说"critic 在噪声方向上的全部权限只有 0.30 饼干（2.2 %）"。现在在**同一个 checkpoint
系列**上重测：

| checkpoint | 训练轮次 | `std_u Q` | `max − min` | 占 13.18 基线 |
|---|---|---|---|---|
| `run1` | **98** | 0.0549 | 0.316 | **2.4 %** |
| `run3` | **808** | **3.112** | **14.517** | **110 %** |

`run1` 训练了 98 轮，那时 `critic` 还接近零初始化，所以 `critic_noise` 蒸馏的是一个**没有
幅度的目标**，它自身也就没有幅度。§9.102 测出的 0.055 是那个状态的产物，不是这个方法的
性质。

**所以 §9.102 必须撤回，不能引用。** 正确的说法是反向的：在训练过的 checkpoint 上，critic
在噪声方向上的极差是 **14.5 饼干，比整集得分（13.18）还大**。

$$ \max_w Q_{ns}(s,w) - \min_w Q_{ns}(s,w) = 14.5 \;>\; 13.18 = \text{整集得分} $$

**一个真实的价值函数不可能有这种幅度。** 改变一个 32 维噪声向量不可能值 14 个饼干，因为整集
也只有 13 个。所以 `critic_noise` 在噪声空间里是一个**剧烈外推的面**，actor 就是在爬它。

这解释了训练日志里一直在扩大的 `min_q − q`（§9.101 记到 1.0）：

- `q` = TD critic 对**真实执行过的** chunk 打分 → 0.68，有界、可信
- `min_q` = 蒸馏 critic 对 actor 的噪声打分 → 1.61，爬到 14.5 幅度面上的一个高点

### 与官方实现的两处差异，正是原因

逐行对照 `refs/sb3-dsrl/stable_baselines3/dsrl/dsrl.py:230-365` 和
`refs/dsrl/cfg/robomimic/dsrl_can.yaml`，两处**实质性**差异：

| # | 官方实现 | 我的实现 | 后果 |
|---|---|---|---|
| **1** | `update_noise_critic` 抽**全新 `randn`** → 去噪 → `Q(s, D(s,w))` → 拟合 `Q_ns(s, scale(w))` | 用 **buffer 里的行为噪声** → `Q_ns(s, w_buf)` 拟合 `Q(s, chunk_buf/1.5)` | **覆盖范围**：官方覆盖整个先验，我只覆盖行为策略去过的地方。σ 收缩时覆盖范围跟着收缩，actor 的 argmax 因此可以跑出 `Q_ns` 的训练域 |
| **2** | TD bootstrap 用**当前 actor** 在 `s'` 的噪声，每次梯度步重新去噪 | 用 buffer 里的 `chunk_next`（采集时的**行为** actor 采样） | 官方的目标是当前策略的价值，我的是行为策略的 |

两处的共同原因：**一次去噪在这里要 40 ms/样本**，而官方在**每个**梯度步都调用一次（`utd=20`
+ `noise_critic_grad_steps=10`）。robomimic 的 `act_steps=4, action_dim=7` 让这件事很便宜
（28 维、DDIM 8 步）；我这里是 50×32 的流匹配、10 步去噪、batch 256。

### 其余八项对照都是对齐的

`target_update_interval=1`、`actor_gradient_steps=-1`、`target_entropy=0`、`ent_coef='auto'`
初值 1.0、`batch 256`、`utd 20`、`noise_critic_grad_steps 10`、`tau 0.005`、`gamma 0.99`、
`layer_size 2048`、`num_layers 3`、`action_magnitude 1.5`、`n_critics 2`、min-combine——
全部一致。两处差异已写进 `train_dsrl_online.py` 的 docstring 表格。

另外两处是**有意的适配**，不是偏差：

- **actor 宽度**：官方用 `act_steps * action_dim = 28`（每个 chunk 位置独立噪声）；我用
  `action_dim = 32` 并沿 chunk 轴重复。这正是论文 §C.5 针对 `C=50, d=32`（1600 维）给出的
  做法。官方不用是因为 `act_steps=4` 让完整空间只有 28 维。
- **奖励**：官方是 robomimic 的每步 −1；这里用饼干计数增量。因为本环境自己的奖励是**水平量**
  （`min(count/required,1)`），对一集积分会到 ~1500。

### 现在的判断

- **actor 把策略改坏了吗？** 直接测不出来（TD critic 对随机 chunk 外推）。但 actor 的 chunk
  本身是**有界的、和真实回报一致**的（4.8–9.0，均值 6.6，对应实测 episode 回报 8.45）。
- **actor 的目标函数坏了吗？** **是。** 它优化的 `Q_ns` 在噪声方向上的幅度是 14.5 饼干，
  超过整集得分。这是**差异 #1** 的直接后果。
- **§9.104 的预测仍然成立**：tanh 参数化 + 坏的蒸馏面 → 评估应当低于 13.18 基线。

### 要修的东西

优先级排序，都需要"每梯度步一次去噪"这个成本：

1. **差异 #1**：`update_noise_critic` 改成抽 `randn` 再去噪。可以降频（例如每 10 个
   noise-critic 步里只对 1 个抽新噪声，其余用缓存），或者用 `--distill-per-frame` 那种
   预生成缓存的思路——离线训练器已经实现过这个。
2. **差异 #2**：TD bootstrap 改成当前 actor。同样可以降频。
3. 在此之前，**`min_q − q` 的扩大是差异 #1 的度量，不是学习的度量**。日志里不该把它当作
   进展来读。

---

## 9.106 完整审计：4 个 bug，1 个被撤回的修复，1 个未解（11-10）

在 run5 之后再跑之前做了一次逐段通读，而不是继续用运行时间换静态可得的结论。以下每条都有代码位置和证据。

### 已修：BootstrapSet 的 cursor 越界（这是 run5 里 #2 修复失效的原因）

```python
index = torch.arange(cursor, cursor + count) % replay.capacity
cursor = (cursor + count) % replay.capacity
index = index[index < replay.size]          # ← 把整批滤空
```

`cursor` 每轮前进 `count`（32），而 `size` 每轮只增长 `scenes`（8）。**第 11 轮起整个切片
都落在 `size` 之外，过滤器把它们全部删掉，`refresh` 从此返回 0 行。**

| 轮 | size | cursor | 通过过滤的行数 |
|---|---|---|---|
| 8 | 64 | 0 | 32 |
| 9 | 72 | 32 | 32 |
| 10 | 80 | 64 | 16 |
| **11** | 88 | 96 | **0** |
| 50 | 400 | 1344 | **0** |
| 225 | 1800 | 6944 | **0** |

所以 run5 里 bootstrap 只在第 8–10 轮生效过，之后**静默关闭**——而这个类存在的唯一理由
就是修 #2。修法是在 `[0, size)` 内轮转（对增长中的缓冲区单调），现在 `boot_rows 16`
连续 9 次都不为 0。

### 已撤回：我把 truncation 当成 terminated 的"修复"是错的，方向反了

上一轮我按教科书改成"截断保留自举"，docstring 写"丢掉的是
`gamma^93 * V(s')` ≈ `0.39 × 4 = 1.6` cookies"。**这个算式用了起点状态的价值去乘折扣，
是错误的状态。**

$$V(s_0) = r\frac{1-\gamma^{94}}{1-\gamma} = 3.90, \qquad V(s_{93}) = r = 0.064$$

截断处真正丢掉的是最后一个状态的价值 **0.064**，我高估了 25 倍。

更严重的是这样做**引入了更大的错**：循环在读完下一个观测前就已经重建了场景，所以
"保留自举"时用到的 `s'` 是**下一集的起始帧**，其价值是 3.90。误差 `0.99 × 3.84 = 3.80`，
是它试图修的东西的 **60 倍**。

参考实现本身就把超时当终止：`ActionChunkWrapper.step` 在 `count >= max_episode_steps`
时置 `done = True` 并永远返回 `truncated=False`，SB3 的
`reward + (1-done) * gamma * next_q` 也就切断了自举。已回退到与参考一致。

### 已修：噪声 critic 的更新频率是参考的 3 倍

参考 `dsrl.py:326-335` 在**所有** `utd` 梯度步之后跑一个独立的
`for gradient_step in range(noise_critic_grad_steps)` 循环，每步重新采样一个 batch。
我原来的实现**在 `utd` 循环里也更新一次**，于是每轮 `20 + 10 = 30` 次，是参考的 3 倍，
而且更新的是 actor 直接优化的那个网络。离线训练器是同样的结构，所以这是被继承下来的，
不是新引入的。已改为只在独立循环里更新。

### 已修：两个日志指标算错了

**`q_bias` 的基准状态错了。** 日志里的 `q` 是 batch 平均，而 batch 是从 replay 里**均匀
采样**的，覆盖整集的状态；我拿来比的是**起点状态**的理论值。对常数每步奖励 `r`、`T` 步：

| 量 | 公式 | T=94, r=0.064 | T=20, r=0.15 |
|---|---|---|---|
| 起点状态 $V_0$ | $r\frac{1-\gamma^T}{1-\gamma}$ | 3.90 | 2.73 |
| 均匀状态平均 $\bar V$ | $\frac{r}{T}\sum_t\frac{1-\gamma^{T-t}}{1-\gamma}$ | 2.27 | 1.48 |

两者差 1.72 倍。旧代码用 $V_0$ 比，凭空造出约 −1.63 的固定偏差。已改为 $\bar V$。

**`ctrl/s` 被变量覆盖弄错 300 倍。** bootstrap 块里写了 `started = time.monotonic()`，
而 `started` 是整轮的起点，于是 `wall = time.monotonic() - started` 只剩 bootstrap 的时长，
`ctrl/s` 报出 2486 而真实值约 8。已改为 `boot_started`。这是本会话第三次同类错误
（前两次是 `stride` 和 `started`），已用 AST 脚本确认 `main` 自身现在只在循环外赋值
`started` 一次。

### 未解：`q` 在奖励为正的情况下变成负的

审计用的短跑里 episode 回报 **+2.75**、逐轮奖励均值全为正（26 轮中 0 次负值），
但 `q = −3.5`、`target_q = −3.2`，且两者互相跟随。

**已排除**：不是初始化噪声——实测 `Q` 初始化 `mean −0.183, std 0.312, min −1.061`，
−3.5 需要 10 个标准差。也不是熵项（关掉它只会让 Q 更大）。

**未查清**：负值如何从自举里自我维持。`target = r + γ(1−done)·min Q_target`，当
`min Q_target ≈ −3.3` 时 `target ≈ 0.11 − 2.86 = −2.75`，与观测吻合，说明它是一个已经
稳定下来的负不动点，而不是逐轮漂移。这个必须在下一次长跑前查清，否则所有 `q_bias` 读数
都没有意义。

### 对齐表的最终状态

| 项 | 状态 |
|---|---|
| `tau 0.005` / `gamma 0.99` / `utd 20` / `noise_critic 10` / `batch 256` / `n_critics 2` / min-combine | ✅ |
| `layer_size 2048` / `num_layers 3` / `Tanh` / `LayerNorm`（除输出层） | ✅ |
| `target_entropy 0` / `ent_coef auto` 初值 1.0 / `target_update_interval 1` / `actor_gradient_steps -1` | ✅ |
| polyak 在每梯度步（不是每轮） | ✅ |
| 噪声 critic 用**新鲜 `N(0,1)`** 去噪后拟合 | ✅ 每轮 8×场景数，残差：不是每梯度步 |
| bootstrap 用**当前 actor** | ✅ 每轮 32 行，残差：不是每梯度步 |
| 超时按终止处理 | ✅ 与参考一致 |
| 噪声 critic 更新 10 次/轮 | ✅ |
| actor 宽度 `action_dim` 沿 chunk 重复 | 有意（论文 §C.5；参考因 `act_steps=4` 用 28 维） |
| 奖励 = 饼干增量 | 有意（参考是 robomimic 每步 −1） |
| critic 输入除以 `action_magnitude` | 保留，可 `--action-magnitude 0` 关闭；参考 `dsrl_na` 路径不除 |

---

## 9.107 负不动点的根因：`min` 的双头下偏 × 稀疏奖励（11-11）

§9.106 留了一个未解问题：`q` 在奖励为正的情况下稳定在 **−3.5**，而且**第 10 轮就到了
−4.31，那一轮所有奖励都是零**。这一节把它查完了。

### 排除法

零奖励处的 TD 映射是 `target = 0.99 · min Q_target`，**只可能把 Q 吸向 0，不可能穿过 0**。
所以 −4.31 必定来自某个已经为负的输入。三个候选逐个排掉：

| 候选 | 检验 | 结论 |
|---|---|---|
| 初始化 | 实测 `Q` 初始化 `mean −0.183, std 0.312, min −1.061`（4 个种子 −0.45…+0.24） | 到不了 −4.3 |
| 观测分布 | `probe_q_init_drift` 把 obs 换成标准正态 | 仍为负（−2.94） |
| chunk 分布 | 同上，把 chunk 换成 `N(0,1)` | 仍为负（−3.87） |
| **奖励** | 换成常数 +1 | **转正（+7.62）** |

奖励是唯一有效的开关，但换 +1 只是把量级抬上去，没有解释**方向**。

### 真因：双头 `min` 的下偏

`probe_min_bias.py` 用同样的 2048 宽 critic、同样的稀疏奖励（32 个状态里 3 个为 1），只改聚合
方式：

| 聚合 | 第 1 步 | 第 400 步 |
|---|---|---|
| **`min`（官方配置）** | −1.441 | **−0.631** |
| `mean` | +0.646 | **+0.668** |
| 单头 | +0.524 | **+0.564** |

**只有 `min` 会变负。**

$$\mathbb{E}\big[\min(a,b)\big] = \mu - \frac{\sigma}{\sqrt{\pi}} \approx \mu - 0.56\,\sigma$$

两个**独立初始化**的 head 取小值，期望比单个低 $0.56\sigma$。这个偏置在**每次自举时被应用
一次**：

$$\text{target} = r + \gamma\big(\mu - 0.56\sigma\big)$$

在不动点处被放大 $\frac{1}{1-\gamma} = 100$ 倍。当 $r$ 小（稀疏）而 $\sigma$ 不小（2048 宽 +
Tanh + LayerNorm），偏置就统治了 target。

`head0 − head1` 的演化是旁证：`min` 那行从 **+0.094 收缩到 −0.035**——两个 head 被拟合到
同一个被压低的 target 上，逐渐同向偏移；`mean` 那行从 +0.67 收敛到约 0，是正常收敛。

### 为什么官方没遇到

`dsrl_can.yaml` 里同样是 `critic_backup_combine_type: min`。区别在奖励：

| | robomimic can | 这里 |
|---|---|---|
| 奖励 | 每步 **−1**（稠密，量级 1） | 96 % 为 0，非零值 1 |
| 每步 $\mathbb{E}[r]$ | −1.0 | +0.09 |
| `min` 偏置（约 0.6）相对奖励 | 微小 | **同量级，主导** |

**所以这不是 bug，是参数与奖励密度不匹配**——和 §9.106 里那三处“真差异”的性质不同。

### 修法 A：放大奖励（采用）

`--reward-scale` 三点扫描（26 轮，8 场景，其他条件固定）：

| `--reward-scale` | 第 20 轮 `q` | 该轮 `reward_b` |
|---|---|---|
| 1 | −3.39 | 0.113 |
| 10 | −2.68 | 1.31 |
| **30** | **+3.27** | 3.38 |

$\mathbb{E}[r]$ 提到约 3.4 后，`min` 的 0.6 偏置被淹没。**已设为默认值 30。**

代价：Q 的量级变成约 340（$3.4/(1-0.99)$），所以 `q_bias` 的绝对值不再可跨设置比较。已新增
**`q_rel = q_bias / |theory|`** 作为跨设置的验收指标。

### 修法 B：`--critic-combine mean`（未采用）

直接去掉偏置（`probe_min_bias` 显示 `mean` 稳定为正），但偏离官方配置，且会引入反方向的
过估计。在稀疏奖励下过估计通常更安全，但不如 A 干净：A 只改奖励单位，算法结构完全不动。

### 记一笔方法论

这次排查的顺序值得固化：**先排除法定位到“奖励”这个开关，再用最小对照分离出“聚合方式”这个
机制**。中间我曾写下“零奖励处不可能穿过零”的推理——那条推理是对的，而且正是它把搜索范围
锁定到“必有一个负输入”，才没有继续在数据分布上打转。

run6 已按 `--reward-scale 30` 启动（480 分钟，8 场景，`--fresh-noise 8`，`--bootstrap-refresh 32`）。
验收标准：`q` 全程为正，且 `q_rel` 不单调恶化。

## 9.108 稠密奖励之后仍然恶化：alpha 的梯度变号（11-11）

run8 用论文的 timer 奖励跑完 641 轮后仍未成功，但它暴露了一个**可以算术证明**的机制。

### 观察到的事实

| round | `log_prob` | `alpha` | `sigma` | `u_abs` | `ret` |
|---|---|---|---|---|---|
| 250 | −14.29 | 0.2969 | 0.8149 | 0.5484 | 6.81 |
| 450 | −1.65 | 0.1368 | 0.7496 | 0.6213 | 8.05 |
| 475 | −1.04 | **0.1285**（最低） | 0.7435 | 0.6263 | 6.30 |
| 500 | +1.80 | 0.1313 ↑ | 0.7289 | 0.6448 | 6.30 |
| 525 | +2.41 | 0.1473 | 0.7210 | 0.6424 | 6.30 |
| 625 | — | 0.1615 | — | — | 5.75 |

`alpha` 在前 475 轮单调下降（0.898 → 0.1285），在 `log_prob` 穿过 0 之后**掉头上升** 26%。

### 机制

`alpha` 的损失是 `-(log_alpha * (log_prob + target_entropy))`，`target_entropy = 0`：

* `log_prob < 0` → 梯度推低 alpha；
* `log_prob > 0` → 梯度推**高** alpha；
* alpha 升高 → 熵项权重变大 → 更推高 `log_prob` → 正反馈。

### `log_prob > 0` 从哪来

从一个 641 轮的 checkpoint 直接读 `log_std`：32 维**全部**在 σ ∈ [0.557, 0.847]，**没有一维塌缩**。

于是手算：

* `Σ log N(u; μ, σ)` 上界 `-0.919 d = -29.4`，实测 **−35.2**；
* 要得到 +2.41，雅可比项 `Σ -log(1 - a²)` 必须贡献 **+37.6**；
* 但 `u_abs = 0.642`，若所有 |a| 都等于 0.642，雅可比最多 **+16.9** → `log_prob` 只能是 −18.3。

差 20.8 nats，唯一能补的只有雅可比项，所以**必有相当一部分样本的 |a| 被推过 0.83**
（`-log(1-0.83²) = 1.176`/维）。数值验证：|a| ~ U[0.3, 0.98] 时均值正好 0.640、
`E[jac] = 0.765`/维 → 合计 +24.5，量级吻合。

**结论：tanh 的饱和被 change-of-variables 项记成了"密度大"。** 这是目标函数层面的病态，
不是日志 bug。

### 为什么这里比参考更容易发生

参考的 actor 宽度是 `act_steps * action_dim = 4 × 7 = 28`，**每个 chunk 位置独立采样**；
我们是 32 维、用 `repeat_noise` 铺到 50 个 chunk 位置。于是：

* `log_prob`、`alpha`、`target_entropy` 全部按 **32 维**记账；
* 实际扰动覆盖 **50 × 32 = 1600** 个数。

`target_entropy = 0.0` 是照抄参考、**不是背离**，但它的前提（28 维、独立、`log_prob < 0`）
在这里不成立。SB3 的默认是 `-dim`，参考特意改成 0.0。

### 与稀疏奖励时代的关系

`--entropy-in-target` 当初被关掉是**对的**，离线训练器的 docstring 自己写着当时的算术：
`alpha` 起始 1.0、`log pi` ≈ −21，补偿项 `alpha·|log pi|/(1−gamma) ≈ 2090`，对比真实
return 9.56 —— **219 倍**，必须关。

论文奖励下这个前提反转了：真实 return ≈ **−978**，补偿项 `0.147 × 2.41 / 0.01 ≈ +35`，
**3.6 %**。所以它应该按参考打开 —— 这已在 `train_dsrl_online.py` 里完成。

### 同一轮复核出的三个真实缺陷

1. **`q_bias` / `q_rel` 是量纲错配，不是偏差。** `calibration` 里的 `q` 是奖励单位
   （约 −300），而 `theory` 取自 `episode_return`，在 `--reward paper` 下是**饼干数**
   （约 1.5）。所以这一整轮对话里我读到的 "−258 的偏差" 其实是 400 倍的错配。修法：
   `Scene` 增加 `reward_return`（奖励单位）专供 calibration，`episode_return` 继续用饼干。
2. **`--grad-clip` 的理由是假的。** 帮助文本说"参考继承 SB3 默认 10.0"；实测
   `clip_grad_norm_` 只出现在 `a2c`/`dqn`/`ppo`，`sac.py` 和 `dsrl.py` **一次都没有**。
   run8 的 `--grad-clip 1000` 行为上接近参考，却被当成背离来记录。默认已改为 0（不裁剪）。
3. **`--help` 本身是坏的。** 帮助文本里有未转义的 `%`（`100 % of` → argparse 解析成 `%o`），
   `--help` 直接抛 `TypeError`。这不是新引入的：`--reward` 那段一直是坏的，只是没人跑过。

另外两条对齐表里的 "SUBSTANTIVE GAP"（噪声 critic 用 buffer 的 w、bootstrap 用采集时的
chunk）**早已被 `NoiseBank` 和 `BootstrapSet` 关掉**，表却还标着 open。剩下的只是数量：
新鲜样本 64/轮 vs 参考 2560/RL 步（40 倍），bootstrap 32 行/轮 vs 每步全量。已按数字改写。

### run9 的配置与预注册判据

与 run8 相同的一切：`--scenes 8 --minutes 480 --reward paper --reward-offset 1.0
--utd 20 --batch 256 --warmup-rounds 8 --learning-starts 256 --bootstrap-refresh 32
--fresh-noise 8 --log-every 25 --save-every-minutes 20`，`--seed-offset 45000`。

变化：`entropy_in_target=True`（对齐参考）、`grad_clip=0`（对齐参考）、
`--eval-scenes 1`（新增确定性评测）、checkpoint 原子写。

**预注册判据。** 这次的核心问题是"alpha 上升是否伴随策略退化"，而 run8 无法回答，因为
`ret` 带探索噪声。run9 同时记录 `greedy`（1 个场景走 `tanh(mean)`）和 `ret`（7 个场景注入噪声），
两者相减即可分离：

* `greedy` 保持而 `ret` 下降 → 是噪声变宽，不是策略变差，alpha 反转**无害**；
* `greedy` 与 `ret` 同步下降 → 策略真的退化，届时把 `--target-entropy` 改成 `-32`
  （SB3 默认，让平衡点永远落在 `log_prob < 0` 一侧），并记录为**故意的背离**；
* 到 r600 时若 `alpha > 0.25` 且 `log_prob > 5`，视为跑飞，直接停。
