# Going back to the Cookie scene without fooling ourselves

The LIBERO reproduction works (31/40, then 35/40 on a re-run; LIBERO itself is 10/10).  The
question now is the reverse direction: walk the scene back towards A3's Cookie task, one step
at a time, and end up able to say **"π0.5 failed here because of the model"** and have that
be true rather than hopeful.

Zero-shot on the Cookie scene is *expected* to fail, and a fine-tune may fail too.  The danger
is not the failure; it is attributing the failure to the model while a bug is in the pipe.
Every bug in `docs/bug-catalogue.md` was silent, and four of them were individually fatal.
None would have been caught by looking at a rollout.

This document is the procedure that makes the attribution a measurement.

---

## 0. The one principle

> **A run that looks reasonable is not evidence that the pipe is right.  Compare each channel
> against an external reference, and make the comparison exact enough that it can fail.**

Self-consistency checks -- the sim runs, the trajectory is smooth, the images render, the
gripper closes, the numbers are finite -- passed with **every single** bug in the catalogue in
place.  What found them was:

| reference | found |
|---|---|
| LIBERO's compiled model | A3 (quaternion order), B1 (sites), B2 (leftover collision), B3 (contact params), B4 (mass), B5 (start pose) |
| LIBERO's own evaluation image | A4 (the image was upside down) |
| LIBERO's own `observation.state` | A1, A2, B5 |
| the A3 dataset's own frames | A5 (mirror scope), A6 (action semantics) |
| a rendered video | C1 (the predicate required failure) |
| forward kinematics of our own command | D1 (the arm delivered half) |

So the transitions below are not "run it and see".  They are **gates**: a change lands only
when the invariants it can break have been re-measured against their reference.

---

## 1. The ledger: seven channels, each with a reference and a test

Everything a policy can see or do goes through one of these.  Each has an owner, a reference,
a test, and a tolerance.

| # | channel | reference for the Cookie side | test | what a failure means |
|---|---|---|---|---|
| 1 | **images**: orientation, mirror, mount pose, fov, resolution, exposure | the dataset's own video frames | orientation fingerprint + MSE vs a recorded frame | a bug |
| 2 | **state** (16-wide): order, frame, units, quaternion convention, gripper polarity | the dataset's `observation.state` | element-wise compare at a recorded frame | a bug |
| 3 | **action** (16-wide): units, frame, scaling, joint order | the dataset's `action` | one-step: apply `action_t` at `state_t`, compare to `state_{t+1}` | a bug |
| 4 | **execution**: does commanded motion happen, how much, how fast | the dataset's own delivery ratio | tracking probe: request *n* mm, measure travel; compare ratio and shape | a bug if the ratio moved |
| 5 | **gripper**: command -> opening -> state, both directions | the dataset | round trip: command open, command close, read the state back | a bug |
| 6 | **physics**: timestep, contact model, damping, gravity compensation | the config that produced the dataset, then the dataset's dynamics | open-loop replay divergence over a short horizon | a bug |
| 7 | **task**: object positions, prompt, success predicate | recorded episodes, including successful ones | predicate fires on a replayed **successful** episode | a bug |

Two rules for the table itself:

* **The reference is per scene.**  LIBERO's reference is LIBERO's runtime; the Cookie scene's
  reference is the dataset.  Neither is "the right convention" -- the mirror is real on both
  sides and opposite (A5).  A scene declares its reference, and the renderer obeys the scene.
* **Never restate a measurement** (C2).  Import the function; do not rewrite the rule.

---

## 2. The two tests that carry most of the weight

### 2.1 Orientation fingerprint -- cheap, and it catches the class that killed us

Two of the four fatal bugs were image orientation.  A single scalar test catches all of them:

```
# for a fixed state, with the scene at rest:
ours   = env.render or observation.images.<cam>      # what the policy will receive
record = the dataset's frame at the same state       # or LIBERO's own policy input

for each of {as-is, flip H, flip W, flip H+W}:
    mse(candidate, record)

assert argmin == as-is, and assert a margin (the winner must beat the runner-up by, say, 15%)
```

Thresholds measured on this repository:

| comparison | MSE |
|---|---|
| LIBERO's policy input vs ours, **as-is** | **5131** (best) |
| ... vs flip H / flip W / both | 6078 / 6014 / 6175 |
| Cookie dataset frame vs our render, **as-is** | **2452** (best) |
| ... vs **mirrored** | 2702 |

The absolute number is not the signal -- the *argmin and the margin* are.  When the winner is
not "as-is", stop: it is a bug, whatever else the run says.

**With one gate in front of it, because this test lied once.**  The first version scored an
older dataset's wrist camera at `flip_hw` three times better than `as_is` and read as a
rotated camera; the winner's absolute error was still MSE 2490 where a matching scene scores
42, so the content did not agree in any orientation and the ranking was noise.  The dataset
was from a different scene configuration.  So the fingerprint is only meaningful when the
comparison is:

```
comparable  :=  mse(ours, recorded)  <=  mse(recorded, another_episode's frame)
```

-- our render must be closer to *its* episode than two different episodes are to each other.
The scale comes from the data, so it needs no hand-picked constant, and it turns "the winner
is flip_hw" into one of three verdicts rather than two: **ok**, **WRONG ORIENTATION**, or
**NOT COMPARABLE**.  Implemented in `scripts/check_invariants.py`.

### 2.2 Teacher-forced replay against the dataset -- the whole pipe in one test

The dataset is the specification: `a3_cookie_adv_1000_merged` has **1123 episodes, 537 765
frames, 16-wide state and action, and three cameras** (`front`, `left_wrist`, `right_wrist`).
If the current environment reproduces it, then the pipe *is* the training pipe, and a policy
failure cannot be laid at the pipe's door.

At a recorded frame *t* of episode *e*:

```
1. set the environment's qpos/qvel to state_t                  # teacher forcing, no drift
2. render all cameras
   -> compare to the dataset's frame t      (fingerprint + MSE)   -> channels 1, 2
3. apply action_t for exactly one control step
   -> compare the measured state to state_{t+1}                   -> channels 3, 4, 5, 6
4. compare ||state_{t+1} - state_t|| against ||action_t - state_t||
   -> the recorded "delivery ratio"; ours must match it
```

Why teacher forcing rather than a plain open-loop replay: it isolates **per-step pipeline
fidelity** from error accumulation, so a small modelling difference in the dynamics does not
hide a large wiring difference.  Run both, and report them separately:

* **TF check** -- tolerances tight, horizon infinite (every frame is its own test).
* **Open-loop check** -- tolerances grow with the horizon; use 20-50 steps and report the
  divergence curve, not a pass/fail.  This is the only test that exercises the dynamics
  channel (6), and the ablation says that channel is worth 10-15 points and is invisible
  without it.

### 2.3 Calibrate the tolerances from the dataset, not from taste

The dataset contains its own notion of "as expected", and the tolerances should come from it:

* **servo lag**: `state_{t+1} - state_t` vs `action_t - state_t` over all frames gives the
  recorded delivery ratio and its spread.  Ours must sit inside that spread -- this is exactly
  the check that D1 (half the command delivered) and D3 (torque once per step) change.
* **image tolerance**: the frame-to-frame MSE within a static span of an episode, which is the
  noise floor of "the same scene rendered twice".  Ours must be within a small multiple of it.
* **state tolerance**: the dataset's own sensor noise per dimension -- `std` over frames where
  the arm is commanded to hold (`|action - state|` small).  The A3 data has a lot of those
  (57.2% of frames dwell).

---

## 3. Capability or bug: a decision procedure, not a judgement call

Run the gates in order.  The first gate that fails decides the answer.

### Gate 0 -- the policy-free ledger

Run §2.1, §2.2 (TF, every 20th frame of 5 episodes, all three cameras), §2.3, plus:

* **tracking ratio** matches the value the dataset was collected at (measure it, do not recall
  it: `scripts/probe_tracking.py`);
* **gripper round trip** in both directions, including the state's reading;
* **success predicate fires** on a replayed *successful* recorded episode (C1);
* **no NaN, no safety stop, no termination before the horizon** in a 300-step no-op rollout.

**If any of this fails: it is a bug.  Stop; do not run the policy.**

### Gate 1 -- the canary

Run a policy that is **known to work in this scene**: the A3 checkpoint that was fine-tuned on
exactly this data.  Its expected rate is a recorded number, not a hope.

* Canary collapses, ledger green -> something changed that the ledger does not cover: timing,
  chunk scheduling, episode bookkeeping, the bridge's queue.  Treat it as a bug until the
  difference is named.
* Canary holds -> the pipe is live end to end, including everything the ledger does not see.

The canary is the only test that covers the *scheduling* channels (how many rows of a chunk are
replayed, when a replan happens, how the state is formed at the replan).  The LIBERO
investigation found `execute: 50` vs `10` to be harmless there by measuring both; the same
question has to be asked again on the Cookie side.

### Gate 2 -- is the policy's own output in distribution?

With the ledger green, ask what the policy *did*, not what happened:

* per-dimension quantiles of the first predicted chunk, against the dataset's action
  quantiles.  In range but unproductive is capability; **far outside** is a domain problem
  (which is a fact about the observation, and is itself worth measuring).
* the fraction of the episode spent commanding "hold" (the identity component).  The Cookie
  data is 57.2% dwell; a policy that never commands zero, or always does, is telling you
  something specific.

### Gate 3 -- the failure signature

Measure the same three things the LIBERO probe measures: distance to the object, whether the
object ever moves, whether the gripper ever closes.  The signatures differ, and they are
distinguishable:

| signature | reading |
|---|---|
| arm never moves; commands large | execution bug (D4-shaped). Gate 0 should have caught it |
| arm moves smoothly, object never touched; commanded travel in range | capability: it is driving somewhere, just not there |
| object moves but the verdict says failure | predicate bug (C1) |
| arm approaches, grasps, carries, but the drop misses | capability *and* a specific sub-skill; measure which |
| everything looks right but the episode ended at step 0 | sentinel bug (C4) |

### The one thing that turns "we think it is capability" into a measurement

**A ladder.**  Render the *same* task on axes that interpolate between the working scene and
the Cookie scene, and measure where behaviour degrades:

1. camera fov and mount pose (LIBERO's own grid already exists:
   `probe_libero_reference.py --perturb`, which measured `grey` = free and `fovy 45 -> 30` =
   0/50);
2. object size in pixels (the Cookie is **1-5 px wide** -- `7e76fa1`), e.g. by narrowing the
   view or moving the camera closer;
3. table and background appearance;
4. object identity: LIBERO's bowl -> the Cookie, same camera, same arm.

Axis 4 is **built**, as of `25de1fe`.  It is the coarse one and the decisive one, because it
changes exactly one thing: the scene keeps LIBERO's cameras, LIBERO's Panda in the same mount,
the same physics block, the same `pruned_init` start poses for every episode, the same plate in
the same place -- and the reachable object goes from `libero_bowl_object`, which
`pi05_libero_finetuned` was trained to pick up, to `cookie_0`, A3's own 6.3 mm Cookie, which
appears in no public dataset.  It is `a3_libero_panda_cookie`: the bowl and the distractor bowl
are removed, the Cookie is placed where the bowl's *mesh* settles (0.150, 0.130), and the
instruction changes with the objects --
`"pick up the yellow block and place it on the plate"`, since a prompt that names a black bowl
next to a Cookie would be a different experiment.  It is a flag on `libero_props_scene`, like
`matched_cameras` and `panda`, so `configs/libero_props.yaml` stays the one statement about what
that scene is, and `cookie_on_table=True` is the only difference between the two ends.

Both ends are measured by one script and one yardstick, `scripts/probe_ladder_object.py`, which
reports per step: the nearest fingertip's distance to the object's measured centre, how far the
object moved and how high it rose, whether the arm and whether the table were in contact with
it (MuJoCo's own contact list, not a proximity threshold), the gripper's opening, the eef's
displacement, and the **measured** joint motion of the working arm.  That last one is the
"did the arm move at all" evidence, and where it is read from mattered -- see F5 and F6 in
`docs/bug-catalogue.md`, two instrument bugs that the first run of this probe exposed by
printing a grasp as "never touched" and a torque as an angle.

#### What axis 4 measured

Four seeds, 300 control steps each, `configs/policies/pi05_libero_panda.yaml`:

| seed | nearest fingertip | arm contact | object moved | rose | ended from plate centre | signature |
|---|---|---|---|---|---|---|
| **bowl 0** | 11.5 mm | 56 steps | 114.6 mm | +92.5 mm | 30.0 mm | grasped it and carried it |
| **bowl 1** | 13.5 mm | 53 steps | 115.7 mm | +87.5 mm | 31.8 mm | grasped it and carried it |
| **bowl 2** | 16.1 mm | 87 steps | 188.3 mm | +159.3 mm | 24.3 mm | grasped it and carried it |
| **bowl 3** | 23.2 mm | 50 steps | 122.9 mm | +95.9 mm | 17.2 mm | grasped it and carried it |
| Cookie 0 | 12.7 mm | 0 recorded | 15.9 mm | +0.0 mm | 120.2 mm | never recorded in contact, contact between samples |
| Cookie 1 | 92.0 mm | 0 | 0.1 mm | +0.0 mm | 117.5 mm | never touched the object |
| Cookie 2 | 54.2 mm | 0 | 0.1 mm | +0.0 mm | 117.5 mm | never touched the object |
| **Cookie 3** | 16.1 mm | 41 steps | 152.4 mm | **+107.2 mm** | **1.4 mm** | grasped it and carried it |

The plate's radius is 73.7 mm, so a reading under that is a placement.

**The shape is not a cliff.**  The one seed that found the Cookie carried out the *entire* chain
-- reach, close on it, lift it 107 mm, carry it 152 mm, and put it down 1.4 mm from the plate's
centre.  A wiring bug cannot do that: every bug in the catalogue is fatal to the whole chain,
not to its rate.  The arm's kinematics are the same on both ends (median peak joint motion 45.8
against 43.5 mrad), which rules out the execution and controller classes outright.

**It is not a clean slope either.**  The three failures are two different things: seed 0 got to
12.7 mm, closed, and pushed the Cookie 15.9 mm without ever holding it, while seeds 1 and 2 never
came within 92.0 and 54.2 mm.  That split is what one expects if the binding constraint is
**whether the Cookie is resolved in the image at all** -- at 1-5 px wide (`7e76fa1`) detection is
marginal, and each seed either notices it or does not.  That is axis 2 of the ladder, and it is
where the next measurement should go.

**The caveat is the sample size.**  Four seeds per end puts a wide interval on 1/4, and this
policy samples fresh flow-matching noise per chunk, so the spread across seeds is itself a
finding.  What n=4 supports is the *existence* of the capability, which is a yes-or-no question
that one success answers; it does not support the rate.  Raising both ends to twenty seeds is the
next run, and one success in twenty would still say "capability is present, rate is 5%".

**None of this is scorable yet**, and it should not be reported as a success rate for the Cookie
task: the Cookie scene's success predicate is still A3's, and the instruction here ends at the
plate, which A3's contracts do not mention.  `probe_ladder_object.py` reports **placement against
a target body**, which is the weaker, honest claim.

A **capability** failure degrades **monotonically** along a continuous axis: performance is a
function of the scene, and it crosses the failure threshold somewhere you can point at.  A
**bug** appears as a **cliff at a wiring boundary** -- a change that has no behavioural
justification (an axis flip, a scene switch, a code path) and takes the rate from "works" to
"nothing" with nothing in between.  All four fatal bugs were cliffs.

This is the strongest available answer, because it does not rely on the absence of bugs: it
relies on the *shape* of the degradation.

---

## 4. The staged transition

Change **one axis at a time**, in this order, and land each step only when its gates are green.
Each step names the invariants it can break, which are the ones to re-measure.

| step | change | invariants at risk | gate |
|---|---|---|---|
| 0 | record the baseline | -- | `python scripts/guard.py --record` -- writes `configs/invariants_baseline.json`, which is **tracked on purpose**: under `logs/` it would be gitignored, and a baseline that cannot be diffed is a number somebody read once |
| 1 | **mirror off** (`reference_mirror: false`) | 1 (images) | fingerprint vs the dataset frame |
| 2 | **cameras -> A3's own** (mount, fov, resolution) | 1 | fingerprint + MSE at a recorded frame, per camera |
| 3 | **arm -> A3** (joints, torque/servo, gripper semantics) | 3, 4, 5 | TF replay step 3-4; tracking probe; gripper round trip |
| 4 | **physics -> the Cookie scene's block** (contact, multiccd, damping) | 6 | open-loop replay divergence over 20-50 steps |
| 5 | **props -> Cookie and the bins** | 6, 7 | object positions; predicate fires on a replayed successful episode |
| 6 | **prompt and task** | 7 | predicate reachability (a scripted expert still succeeds) |
| 7 | run the policy | -- | gates 1-3 |

Three rules for the walk:

* **Never change two axes in one commit.**  The ablation's lesson in reverse: four bugs
  composed, and each was invisible while the others were in place.
* **Re-run the other scene's guard after every step** (§5).  The mirror-scope bug (A5) was
  introduced by a change made for the *other* scene and was invisible from the scene being
  edited.
* **A step that makes things worse is information only if the invariants are green.**  If they
  are not, the step is incomplete, not unsuccessful.

---

## 5. The guard set

Two invariants that must hold after **every** edit, and a runner that checks them together:

```
python scripts/guard.py --record    # once, on a state that is known good
python scripts/guard.py             # after every edit; exits non-zero
python scripts/guard.py --cheap     # the half that needs no GPU and no checkpoint
```

1. `a3_libero_panda`, seeds 0-9, expect **>= 9/10**.
2. The Cookie fingerprint: for a fixed seed, `mse(ours, dataset_frame)` as-is beats every flip
   by >= 15%, and is within a small multiple of the within-episode noise floor.

Each is checked against the recorded baseline and, where an absolute invariant exists, against
the invariant rather than against the baseline -- `|ours - recorded|` is exact arithmetic and
is checked at `1e-6`, not at "whatever it was last time".

Every row prints **measured, expected, and the difference**.  One consequence is worth naming
because it is the whole reason the tool is shaped this way: the replay residual is checked
twice, once against the tolerance the tool documents and once as drift from the baseline, in
two rows.  They are different claims -- a residual 70x its baseline can sit happily under an
absolute tolerance -- and a single row that averaged them would answer neither.

Plus, cheap and worth it:

3. `probe_tracking.py` -- the delivery ratio.  (Also a row in the guard, from
   `check_invariants.py`'s channel 4.)
4. `python -m pytest` or whatever the smoke set is; the bridge's `policy check` for every
   manifest still parses.

The guard set is what makes "I fixed A and did not break B" a claim rather than a hope.
**A guard that cannot fail is not a guard**: the failure path was tested by tampering with a
copy of the baseline (a wrong orientation, a 58x image budget, a tracking drift, a 70x replay
drift) and confirming each one lands on its own row with the reason printed.

---

## 6. Symptom -> suspect -> test

For the day something fails and the question is where to look first.

| symptom | suspect first | test |
|---|---|---|
| policy output looks mirrored / swapped | image orientation (A4, A5) | fingerprint |
| the arm never moves | two controllers, or a goal nobody reads (D4) | step response: request 25 mm, measure travel |
| commanded motion is smooth but nothing arrives | servo delivery (D1) or contact geometry (B2) | tracking probe; list world-level collision geoms |
| grasps slip, or the object is shoved instead of held | gripper polarity (A2, A3), object mass (B4), contact params (B3) | gripper round trip; `body_mass` vs the reference; per-geom contact diff |
| the state looks nearly right | quaternion order (A1), gripper polarity (A2) | element-wise compare against the dataset |
| it works on seed 0 and never again | start-pose assumptions (B5) | compare the *first observation* of every episode against the dataset's |
| the report says 0/40 and the video says otherwise | the measurement (C1, C2, C3) | render it; re-read the verdict enum |
| it worked yesterday | cross-scene leakage (G) | run the guard set |

---

## 7. What to build

Small, and mostly assembly of things that already exist:

1. `scripts/check_invariants.py --scene <name> --dataset <root> --episode e` -- **written, and
   it runs in seconds with no policy and no GPU.**  It checks four of the seven channels:
   state convention (loads the recorded state, compares element-wise -- exact to 0.00e+00 on
   `a3_cookie_one`), the image fingerprint with the comparability gate (§2.1), the gripper
   round trip, and tracking (43% of a step per control step at the default damping, which is
   the number `probe_tracking.py` recorded).  It exits non-zero and prints
   `INVARIANT FAILED -- treat as a bug`.
2. `scripts/replay_vs_dataset.py --scene <name> --dataset <root> --episode e --frames ...`
   -- the teacher-forced comparison of §2.2 (channels 3 and 6).  **Written.**  It loads a
   recorded state, applies the recorded action for exactly one control step, and compares with
   the recorded next state; the free-space residual is expressed as a fraction of one recorded
   step, and the open loop that follows is reported as peak / settled peak / endpoint.
   Measured, both datasets pairing with their own scene: **0.00** of a step
   (`a3_cookie_single_piece`/`a3_cookie_one`) and **0.07** (`a3_cookie_batch`/
   `a3_cookie_adv_1000_merged`), gripper 0.003 and 0.029, open loop exactly 0.0 and 0.021 rad.
   The thresholds are measured rather than picked: allowing 0.5 of a step, where the *smallest
   deliberate error* this channel can make -- dropping the recorded velocity from the initial
   condition, `--rest-start` -- scores 0.76.

   Two limits are part of the result and are printed with it.  **It does not certify the
   scene**: the arm is the same robot in every scene here, so a deliberately wrong pairing also
   scores 0.00 -- this channel certifies the action, state, unit and servo conventions, which
   is where the mirrored image, the wrong picture and the legacy gripper all showed up as
   fatal.  And **contact steps are reported separately and not judged**, because the dataset
   does not record where the props were; on a scene whose props it cannot reproduce, the arm
   can be teleported into one and the environment's own safety stop fires, which is an
   `INVALID` verdict rather than a number (F1 in `bug-catalogue.md`).
3. `configs/invariants_baseline.json`
   -- `scripts/guard.py --record` writes it, and `guard.py` reads it: the recorded
   expectations, so drift is a diff and not a discussion.  It is a config and not a log
   because `logs/` is gitignored, and the only baseline worth having is one `git diff` can
   show.

Cost: the ledger is cheap -- no policy, no GPU -- which is the point.  It can run on every
commit; the policy runs only when it is green.
