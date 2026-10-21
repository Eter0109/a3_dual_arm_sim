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
