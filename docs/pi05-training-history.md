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
