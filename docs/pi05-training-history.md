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
