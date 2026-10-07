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
| 0 | record the baseline | -- | save the current `a3_libero_panda` numbers and the Cookie fingerprint to `logs/invariants/baseline.json` |
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

Two invariants that must hold after **every** edit, run from a script:

1. `a3_libero_panda`, seeds 0-9, expect **>= 9/10**.
2. The Cookie fingerprint: for a fixed seed, `mse(ours, dataset_frame)` as-is beats every flip
   by >= 15%, and is within a small multiple of the within-episode noise floor.

Plus, cheap and worth it:

3. `probe_tracking.py` -- the delivery ratio.
4. `python -m pytest` or whatever the smoke set is; the bridge's `policy check` for every
   manifest still parses.

The guard set is what makes "I fixed A and did not break B" a claim rather than a hope.

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
2. `scripts/replay_vs_dataset.py --root <dataset> --episode e --frames ...`
   -- the teacher-forced comparison of §2.2 (channels 3 and 6), printing per-channel residuals
   and the divergence curve.  **Not written yet**, and it is the one that covers the dynamics.
3. `logs/invariants/baseline.json`
   -- the recorded expectations, so drift is a diff and not a discussion.

Cost: the ledger is cheap -- no policy, no GPU -- which is the point.  It can run on every
commit; the policy runs only when it is green.
