# Bug catalogue: the failures that looked like the policy, and were not

Every entry here was **silent**.  No exception, no NaN, no crash, no assertion.  The arm
moved, the cameras rendered, the gripper closed, the trajectory was smooth, and the numbers
were finite and plausible.  Each was found only by comparing one channel of the interface
against an external reference -- LIBERO's own compiled model, its own evaluation images, its
own recorded states, the dataset's own frames -- and never by looking at the run.

That is why this file is organised by **class of failure** rather than by date: the point is
to recognise the shape next time, not to remember the story.

The headline, from the leave-one-out ablation in `scripts/ablate_libero_panda.sh`: of nine
fixes, four were each **individually fatal** and every one of those four was a bug of this
kind.  Everything done under the heading "make the configuration more like LIBERO's" was
worth 0-15 points together.

| fix reverted from the working configuration | placements |
|---|---|
| -- (baseline) | 35/40 |
| image orientation: the adapter's extra flip | **1/40** |
| image mirror, environment side | **0/40** |
| torque once per control step instead of every substep | **0/40** |
| the old gripper (frozen outer jaw, inverted opening) | **0/40** |
| the Cookie task's contact model / multi-contact CCD | 29/40 |
| one fixed home instead of LIBERO's per-episode poses | 30/40 |
| the bowl at 45.9 g instead of 5.6 | 33/40 |
| four rendered debug sites | 37/40 |

---

## A. Silent semantic mismatches

The value is a number, the number is finite, and it means something other than what both
sides think it means.

### A1. A quaternion read in the wrong order is a half-turn

`LiberoProcessorStep._quat2axisangle` takes `(x, y, z, w)`.  This code read `(w, x, y, z)`.
A quaternion read that way looks almost like the identity with a small tilt -- the state
reported `[0, 0, 0]`-ish and nobody looked twice -- when the actual orientation was **180.03
degrees about world +x**.  Rounds ① through ⑦ of the LIBERO investigation all fed the policy a
half-turn of error on its first input.

*Found by*: printing LIBERO's own `observation.state` and comparing, element by element.
*Rule*: a state vector is a **spec**, not a value.  Write down which convention each half
uses (order, frame, units, sign) and test the round trip against the reference's own output.

### A2. The two grippers disagree about which end of the travel is zero

A3's jaws read **zero when they are apart** and close towards their travel; robosuite's read
zero when they are **together** and open to ±0.04.  The polarity, not the value, was wrong, so
`opening` was inverted: an open gripper reported "shut" and vice versa.

*Found by*: comparing the state's last two numbers against LIBERO's (`[0.03999, -0.03999]`
where LIBERO says `[0.020833, -0.020833]`).
*Rule*: for every actuator, check the *zero*, the *sign*, and the *units* against the
reference.  Then check that the same convention is used for state, action, and rendering.

### A3. A sign computed from `upper >= 0` on a range of `[-0.04, 0]`

The outer jaw's drive sign came from "is the upper limit non-negative".  For a mirrored range
whose upper bound **is** zero, that is true, so the jaw was commanded out of its own control
range, clipped to zero, and **never moved at all** -- a two-jaw gripper acting as a one-finger
press.  Every number remained plausible.

*Found by*: reading the jaw's `qpos` after a reset and seeing one of the two at 2e-05.
*Rule*: derive mechanism properties from the mechanism (the sign of the travel end, the
magnitude of the range), not from a convenient comparison.

### A4. "Where does the image point" has two conventions in one repository

LIBERO's buffer is bottom-up and `LiberoProcessorStep` flips **both** axes, so the policy is
trained on the **horizontal mirror** of the upright view:

```
LIBERO:  obs = U[::-1]  ->  flip both axes  ->  U[:, ::-1]
```

This harness rendered `U` and flipped **rows** in the adapter, twice over -- first `[0, 1]`
(a 180-degree rotation, rounds ①-⑧ of the LIBERO work), then `[0]` once the environment
started mirroring -- so the policy was served `U[::-1, ::-1]`: the world, upside down.
Measured against the image LIBERO's own evaluation feeds, as-is scores MSE **5131** and the
three alternatives 6078 / 6014 / 6175.

*Found by*: rendering the actual tensor and looking at it.  The arm came into frame from the
bottom.
*Rule*: a flip is a claim about **both** ends of the pipe.  Verify by fingerprint -- MSE
against a reference frame for all four orientations, argmin must be "as-is" -- and re-verify
after **every** change to either end.  Two "corrections" compose into a new error.

### A5. Two interfaces, one scene, opposite camera conventions

A3's own datasets were recorded **unmirrored**; LIBERO's checkpoints expect the **mirror**.
A mirror added for the LIBERO scene was placed in the shared renderer, so it silently
re-mirrored A3's data for A3's checkpoints too.

*Found by*: exporting a frame from `datasets/a3_cookie_adv_1000_merged` and comparing -- its
front camera puts the source bin on the **right**; the unmirrored render agreed (MSE 2452)
and the mirrored one did not (2702).
*Rule*: a convention belongs to the **scene** (`cameras.reference_mirror`), never to the
shared renderer.

### A6. An action in one coordinate system, read in another

`pi05_libero_zs_single.yaml` reads the checkpoint's seven outputs -- end-effector **deltas in
LIBERO's action space** -- as **joint angles for A3's joints**.  Measured open-loop error
ratio: **47.2×** the demonstrations' own per-step motion, against 2.07 for the properly wired
fine-tune.  Zero-shot failure here is not a capability fact; the wiring says the model was
asked a different question.

*Rule*: before attributing a zero-shot failure to the model, check that the action space,
units and frame are the ones it was trained on.  The manifest's own docstring says which.

---

## B. Silent geometry mistakes

### B1. Rendered debug geometry that the reference never shows

robosuite hides sites (`Robot.visualize` -> `set_sites_visibility`) and LIBERO's compiled
model carries 31 sites with **none of them showing**.  This scene drew four: a **green
cylinder** at the grip point, two red spheres, and a red burner marker -- in every frame, at
the centre of the action.

*Found by*: listing sites with non-zero alpha on both sides.
*Rule*: "what is in the picture" is part of the interface.  Enumerate the *visible* set on
both sides and diff it, rather than looking at the picture and recognising what you expect.

### B2. Collision geometry that is not in the reference, standing where the robot is

The A3 post (`stand_mast`, 1.1 m tall, at exactly the Panda's shoulder) and the A3 base
(`base_link`) were both still in the scene.  Their **collision** boxes passed through the
robot, so the arm spent every episode fighting geometry that does not exist in LIBERO -- it
commanded 8 metres of travel and never arrived.

*Found by*: walking the model and listing world-level geoms; then measuring that hiding them
changed the fingertip's reach from "213-348 mm away, never touching" to "40/40 in the bowl's
radius".
*Rule*: diff the **collision** sets, not just the visible ones.  An invisible obstacle is
worse than a visible one.

### B3. Contact parameters inherited from the wrong scene

`build_model` takes `scene="cookie_transfer"`, so the LIBERO reproduction inherited the Cookie
task's contact class for every robot geom: `solref 0.04 0.85`, `solimp 0.75 0.97 0.002`
against LIBERO's `0.02 1` and `0.9 0.95 0.001`, plus multi-contact CCD on (`enableflags` 16
against 0).  Measured: **65 of 65** robot geoms differed.

*Found by*: iterating every geom under `robot0_base` on both models and comparing solref,
solimp, friction and condim.
*Rule*: compare **all** geoms, not the interesting ones.  Objects that declare their own
parameters were always right; the ones that *inherit* were all wrong.

### B4. Mass, from a rule nobody wrote down

robosuite's object loader sets every **visual** geom's mass to `1e-8`
(`robosuite/models/objects/objects.py:418`), so a scanned object weighs what its hand-built
collision geoms weigh.  This loader restored the meshes and skipped that step: the bowl
arrived at **0.0459 kg** against LIBERO's 0.0056 (8.2×) with 8× the inertia.  Fixing it took
0/40 to 3/40 at the time.

*Found by*: measuring `body_mass` on both sides.
*Rule*: an asset XML is not the runtime model.  Compare the **compiled** numbers.

### B5. Wrong start pose, because the reference does not use one

LIBERO's evaluation gives episode *i* the *i*-th of **fifty** start poses from its
`.pruned_init` archive; each joint varies with a standard deviation near a degree and a range
of four to six.  This scene used one fixed home.  The jaws were also wrong: `0.0387` is where
they end up *after* the first open command, and LIBERO's first observation says
`[0.020833, -0.020833]`.

*Found by*: comparing the deliverable's first observation, field by field, against the
archive.
*Rule*: the reference's *first observation* is a specification.  Reproduce it exactly, and
per episode if it varies.

---

## C. Self-consistency traps

Tests that pass, and mean nothing.

### C1. The success predicate that required failure

`_verdict` first asked for contact with the **table**.  A correctly placed bowl rests on the
plate, 15 mm up, touching nothing else -- so the predicate reported a rollout that picked the
bowl up, carried it to the plate and left it there as a **failure**.  Every number in the
report was consistent; they were being read against a rule that could not be satisfied.

*Found by*: a rendered video, not by the numbers.
*Rule*: validate the success predicate against **recorded successful episodes**, and against
a scripted expert.  A predicate that no trajectory can satisfy is a measurement bug, not a
result.

### C2. Two measurements that were not the same measurement

The A3 probe defines the bowl's rim as the top of its measured bounding box.  The reference
probe restated that rule, looked for a site named `akita_black_bowl_1_top_site`, found none,
and **silently fell back** to the bowl body's origin -- 71 mm lower.  Every height it reported
was wrong by that much, and the two sides were being compared on different quantities.

*Found by*: importing the first probe's `prop_frame` into the second instead of restating it.
*Rule*: share the measurement, do not restate it.  When a fallback can be silently taken,
make the fallback an error.

### C3. Counting the wrong string

The summary counted `placed on the plate`; the verdict actually printed `PLACED`.  One run was
reported as 0/40 -- and a genuine 3/40 result, the first non-zero in the whole investigation,
was nearly lost.

*Rule*: parse the verdict enum, not a remembered phrasing.  Print the histogram of verdicts
rather than counting one of them.

### C4. `steps_taken == 0` meant "never terminated"

LIBERO's wrapper returns `truncated=False` unconditionally, so a failed episode never sets
`terminated` and the array kept its initial zero.  It read as "the episode ended at step 0".

*Rule*: define the sentinel, and print what it means next to the value.

### C5. A fingerprint with no comparability gate (this one is ours, and it happened)

The orientation fingerprint -- "render, score against the reference frame in all four
orientations, the winner must be as-is" -- was written and immediately gave a confident
answer that was **wrong**: on a dataset recorded a few days earlier the wrist camera scored
`flip_hw` three times better than `as_is`, which reads as "the camera is rotated 180°".  The
absolute numbers said otherwise -- the *best* orientation was still MSE 2490 where a matching
scene scores 42 -- so the content did not agree in any orientation and the ranking was noise.
The dataset had been collected on a scene-configuration branch; the two were not the same
scene.

*Found by*: reading the absolute value rather than the ranking, which is the thing the
ranking test exists to make unnecessary.
*Rule*: a comparison needs a **comparability gate**, and it should be calibrated from the
data rather than chosen.  The one that works here: our render must be closer to *its* episode
than two **different episodes** are to each other.  If it is not, the verdict is "cannot
judge", not "flipped".  `scripts/check_invariants.py` implements exactly this, and reports
three states -- ok, WRONG ORIENTATION, NOT COMPARABLE -- because two were not enough.

---

## D. Execution and dynamics

### D1. The arm delivered half of what it was told

`probe_tracking.py`: at the default damping the arm covers **43%** of a step command and
**56%** of a 50 ms command, because the servos need 150-200 ms and the policy issues a fresh
command every 50.  The interface was exact (forward kinematics of `applied_action` agreed with
the request to 101%) and the arm delivered half of it, flat across a 6× range of magnitudes.

*Found by*: driving one axis at several amplitudes and measuring the end-effector's travel.
*Rule*: measure **command -> motion**, not command -> command.  An interface that is exact
and an arm that is slow look identical in a log.

### D2. Chasing tracking bought ringing

Lowering the damping to 0.40 bought 91% tracking and cost **12.2% overshoot with a 300 ms
settle** -- while the policy issues a new command every 50 ms, so the arm is never out of its
transient.  At 1.0 it delivers 55% and settles in 250 ms with no overshoot.

*Rule*: a single scalar "how much of the command arrives" hides the shape of the response.
Report overshoot and settling time next to tracking.

### D3. The control loop that was not robosuite's

robosuite's `SingleArm.control` sets the goal **once per policy step** and calls
`run_controller` on **every physics substep**, so the arm is inside a 500 Hz loop under a
20 Hz goal.  This harness computed the torque once and held it for all 25 substeps.  Measured
step response for a 25 mm command: 3.9 mm on the first step against the reference's 2.8.  The
ablation says reverting this is fatal (0/40).

*Rule*: the control law is not just the equation; it is the equation **and** how often it is
evaluated.  Compare the step response, not the gains.

### D4. Two controllers, one goal, nothing moving

The bridge built its own adapter while the environment's substep loop re-computed torques from
*its* adapter's latched goal.  Every command the bridge converted was overwritten before the
first physics step: the arm moved **0.00 mm** for a command worth 25.

*Found by*: the step-response check, which is why that check exists.
*Rule*: one owner per piece of state.  If a loop recomputes from a goal, everything that sets
the goal must set **that** goal.

---

## E. Order of operations

### E1. Editing an XML tree before the names exist

Two sweeps to hide leftover geometry matched **nothing**: the props they looked for are merged
into the scene *after* the arm is built, and the collision geoms only acquire their
`*_collision` names in a later renaming pass.  The visible effect was none at all -- the black
post and the green shells stayed, and the visible geom count was unchanged at 127.

*Found by*: nothing in the run.  Found by re-checking the count and finding it identical.
*Rule*: when a pass edits by name, assert that it matched something.  `hidden = {...}` with
`raise if not all(...)` is why this is now loud.  And prefer editing the **compiled model**,
where every name is final.

### E2. Renaming `name` and leaving `joint=` behind

Merging a robot's XML renamed elements but not their `joint=` / `site=` / `body=` references,
so a lookup silently fell through to the default and the wrong object was driven.

*Rule*: a rename is a rewrite of every reference.  Assert the reference resolves.

### E3. Closing the loop after writing the state

A reset wrote joint angles into the command slot and then settled with a closed loop: the jaws
were driven back to whichever opening the command slot held, so the carefully written start
pose did not survive to the first observation.  Setting **both** the state and the command
that holds it is what makes a start pose stick.

*Rule*: in a closed-loop system, a state is not a state until the controller is holding it.

### E4. `mju_quat2Mat` wants a flat 9, not a 3x3 view

Passing a reshaped view is rejected outright; the first rotation command a policy ever sent
killed the run at its first step, and the teardown noise in the log made it look like a
rendering failure.

*Rule*: when a crash's log is loud and unrelated, read the first traceback, not the last.

---

## F. Our own measurement bugs

The three above (C1, C2, C3) are this class.  One more rule covers them:

* **A measurement is a claim with two ends.**  State which reference it was taken against, and
  whether that reference is the same thing on both sides.  The `top_site` fallback and the
  `PLACED` string are both "the two ends were not the same measurement".

The four that `scripts/replay_vs_dataset.py` produced are worth keeping because **every one of
them was wrong in the same direction**: they all reported "the scene's dynamics differ from the
recording", which is the finding the tool exists to rule out.  A harness does not fail
randomly; it fails toward the conclusion you built it to check.

### F1. The arm that had stopped, reported as an arm that did not move

*Symptom*: free-space residual 0.0391 rad, `4.45 of a recorded step`, `moved/expected median
0.000` on `a3_cookie_batch` -- the largest number the tool had ever produced.

*Cause*: nothing kinematic.  Writing the arm's state while letting `mj_resetData` drop the
**props** onto the model's default pose put the arm inside a prop at frame 98; the solver
ejected it at 97.88 rad/s, the environment's own safety stop fired ("joint velocity exceeded
safety threshold"), and `step()` then applies `self._current_joint_action()` forever -- the arm
holds position for the remaining hundred frames.  Every step after that is a zero delta against
a moving recording, i.e. a wrong-dynamics verdict of exactly the size of the recorded motion.
Restoring the props to their post-reset pose instead was no better: the arm dives into a
stationary cookie on the frames where the recording is closing the gripper, and the same stop
fires on the same frame.  The props are now left to our own physics.

*Rule*: **a body that has stopped is not a body that did not move**, and a tool that reports a
verdict must know which of the two it is looking at.  A safety stop is now an `INVALID` verdict
with the frame and the reason, not a number.  Any environment with a latch needs its latch
checked in the measurement loop, and the latch's *frame* is part of the result.

### F2. A window that silently crossed an episode boundary

*Symptom*: gripper error 0.75 while the arm channel reported nothing at all.

*Cause*: episode 0 of `a3_cookie_one` is 190 frames and the tool asked for 200.  The reader
indexed the dataset's global row index, so frames 190-199 were the first ten frames of episode
1.  Teacher forcing **teleports the arm onto the recorded pose at every frame by
construction**, so a step discontinuity in the recording is invisible in the arm channel -- and
the gripper, which cannot teleport in one control step, was the only channel that could see it.

*Rule*: an episode is a container, and a reader that takes a range must know its ends.  Asking
for more frames than an episode has is an error, not a clamp.  This is also a general property
of teacher forcing worth remembering: **it tests one-step faithfulness and hides every
discontinuity the recording contains.**

### F3. Half of a second-order initial condition

*Symptom*: free-space residual 0.0033 rad = 0.76 of a recorded step, on a scene whose open-loop
replay was **exactly 0.0 rad** for fifty steps.  That contradiction was the whole diagnosis: the
same dynamics cannot both track perfectly and miss by three quarters of a step.

*Cause*: teacher forcing wrote the recorded **position** and left the velocity at zero.  A
trajectory is an initial value problem in `(position, velocity)`; the dataset records both
(`observation.velocity`), and the only reason it was being dropped was that an earlier attempt
had estimated the velocity by finite difference across one frame -- which is the interval
*average*, not the instantaneous value -- and that estimate made things worse, so "no velocity"
looked like the safe default.  Both were wrong; the recorded velocity is the right one and it
drives the residual to 0.0000.

*Rule*: when a check disagrees with another check of the same quantity, one of them is not
measuring what you think.  Do not average two suspicions into a tolerance -- find the
contradiction and read it.

### F4. The gripper judged against the command instead of the recording

*Symptom*: a gripper error of 0.69 on a scene where the gripper tracks the recording exactly.

*Cause*: the achieved opening was compared with `action[7]`, the **commanded** opening, instead
of with `state[7]`, the recorded achieved one.  It measures servo lag while looking exactly like
a convention mismatch -- the same class as C2, with the same shape: one end is a command and the
other is a measurement.

### F5. The arm that was not under the body the search started from

*Symptom*: a policy rollout that displaced the bowl **136.6 mm** and lifted it **90.4 mm** --
unmistakably a grasp and a carry -- was reported as `arm on the object for 0` steps, and the
failure signature printed was `never touched the object`, the branch that exists for a rollout
which never reached the object at all.

*Cause*: `props.robot_bodies` walked down from `ARM_BASE_BODY = "base_link"` and returned every
descendant.  In `a3_libero_panda` the left arm is not one of those descendants: `_add_libero_panda`
splices LIBERO's Panda in as a tree of its own, rooted at `robot0_base`, which is a sibling of
the table rather than a child of `base_link`.  The search found 19 bodies -- the A3's *right*
arm, which in that scene is inert -- and none of the arm doing the work.  A contact test against
an empty body set answers `False` forever and cannot tell that it is answering nothing.

*Rule*: **a set that comes back empty because the search started in the wrong place looks exactly
like a set that is legitimately empty.**  Any helper whose job is "gather the things of kind X"
should be run once against every scene it will be used in and the count printed; here the reply
was 19 where 35 was right.  Found by moving an object and noticing the tool denied it.

### F6. A torque recorded under the unit of an angle

*Symptom*: `peak arm motion per step: 17129.87 mrad`, i.e. **17.1 rad in one control step**, in a
scene whose `max_joint_step_rad` caps a step at 0.1 rad.  Physically impossible, printed without
comment, in the same table as the real measurements.

*Cause*: the quantity was taken from `info["applied_action"][:7]` and labelled in radians.  That
is correct only for a joint-target arm.  `env._limit_joint_action` has an early return for
`OperationalSpaceAdapter` under which those seven entries are **torques in N.m** -- not joint
angles, not rate-limited, not clipped to any joint range, because there is no trajectory to
limit.  `a3_libero_panda` is such a scene.  Both probes had been logging this number, which
means every "mrad" figure they had printed for a Panda scene was the step-to-step change of a
torque vector.  The motion is now measured where it is unambiguous: the difference of
`data.qpos` at the working arm's seven joints, addressed by name.

*Rule*: **a number is only a measurement together with the unit it actually has**, and a unit is
a claim about the code path that produced the value, not about the field's name.  The cheap
check that catches this class: compare every recorded quantity against a physical bound before
believing it.  The right bound here was not `max_joint_step_rad` (0.1 rad), which the OSC path
deliberately bypasses -- there is no trajectory to rate-limit -- but the environment's **own**
safety stop, which fires above 80 rad/s: 17 rad in one 50 ms control step is 340 rad/s, a
velocity the simulator would have refused.  F5 and F6 were both found that way, in the same run,
from output that had been printed as plausible for weeks.

### F7. A threshold calibrated in another scene

*Symptom*: `gripper closed on 0/150 steps`, and the signature
`moved it without ever closing the gripper on it`, for a rollout that had closed on step 47,
held the hold, and carried the bowl 114.7 mm of the 117.6 mm to the plate.

*Cause*: a close was `opening < 0.05`, and 0.05 was taken from the A3's finger servo, which
reaches 0.027 at the end of its travel **when it is empty**.  In `a3_libero_panda` the jaws stop
at **0.0823** because there is a 112 mm bowl between them, and a gripper gripping something
cannot reach the value it reaches gripping nothing.  The threshold was not merely conservative;
it was **structurally unsatisfiable in that scene**, so it could only ever return "never closed".
It is now a fraction of the run's own maximum opening, which needs no constant per scene.

*Rule*: **an absolute threshold is a claim about a machine, and a scene is allowed to contain a
different machine.**  Calibrate against the quantity's own range in the run that produced it.
The sharper version: a threshold that *cannot be met* is worse than no threshold, because it
looks like evidence -- here it looked like proof that the policy cannot grasp, which is the
hypothesis under test.

### F8. A duration read off a step boundary

*Symptom*: `closed the gripper after leaving it`, on a rollout with **84 steps of the gripper
closed and touching the object at the same time**.

*Cause*: the grasp test was `was the object in contact on the step before the first closed
step`.  On seed 2 of `a3_libero_panda` the gripper shut across the approach -- 0.99, 0.81, 0.57,
0.35 over steps 34-37 -- so it crossed the close threshold on the **same step** it first touched
the bowl, the lookback landed on `no contact`, and a 84-step hold was classified as a gripper
that closed on nothing.  The test was also answering a different question from the one the
summary line printed next to it (`_grasp_timing` asked "did the close come before the last
contact") and the two disagreed in the same line of output.

*Rule*: **ask whether the state held, not what happened one step before it began.**  "Closed and
on the object" is a conjunction over steps, and it is measured by counting the steps where both
hold; a boundary lookup silently converts a duration into an instant, and the instant is where
the noise is.  Both functions now share `_held_steps`, so the signature cannot disagree with the
line above it.

### F9. A sampled quantity reported as a verdict, with the bias against the smaller object

*Symptom*: `never touched the object`, on a rollout where the object slid **15.9 mm** -- the
gripper closing from 0.81 to 0.15 across the six steps it happened in.

*Cause*: `touch_arm` is MuJoCo's contact list read **once per control step**, and a control step
is 25 physics substeps (`physics_hz 500`, `control_hz 20`).  A contact that begins and ends
inside one control step is invisible to it, so the field is a *lower bound* on contact, and how
loose that bound is depends on the object: a 112 mm bowl is in contact for 50-87 consecutive
steps, while a 6.4 mm-thin Cookie can be brushed and released within one.  Eight of twenty seeds
of `a3_libero_panda_cookie` displaced the Cookie 14.5-54.8 mm with no contact recorded at any
sample; in most of them the Cookie was **toppled** (see F11), which is a single-step event.

*Rule*: **a sampled field is not a verdict, and the size of what it misses is a property of the
system, not of the sampling rate.**  Where a quantity is sampled coarsely, state what it can miss
in the units of the thing being studied -- here "contact lasting less than 50 ms" -- and let a
channel that cannot miss it (displacement) decide the sentence.  The trap is not the sampling,
it is that the sampling error is **correlated with the variable the experiment is about**: every
instrument bug in this file has been biased the same way, toward "the policy cannot do it", so a
coarse channel is not a neutral channel.

### F10. A hold of one control step reported as a grasp

*Symptom*: `grasped it and carried it`, on two seeds of `a3_libero_panda_cookie` that recorded
**1 and 2 steps** of contact, never raised the Cookie above its starting height (+0.0 and +0.3
mm), and displaced it 18.5 and 21.7 mm.

*Cause*: the grasp test was "is there a step where the gripper is closed **and** the object is in
contact".  The policy closes the gripper early in every rollout -- at step 45 on the Cookie end
against step 43-47 everywhere -- and then leaves it closed, so the second half of every episode
satisfies the closure half of the test, and a single brushed sample then manufactures a "hold".
A brush of one 50 ms step is not a grasp.

*Rule*: **a grasp is not a coincidence of two booleans, it is a state that persists and has a
consequence.**  The consequence here is unambiguous and needs no threshold on contact: **a
carried object leaves the surface it was resting on.**  `LIFT_MM` (5 mm, against a measured
0.63 mm settling noise) now decides it, and the two false positives became
`displaced it without lifting it off the surface`.  Note the shape of the error: a conjunctive
test with a *permissive* reading of each half is how a 1-step event passes as a 40-step one.

### F11. `max` of a signed quantity hides the fall

*Symptom*: `rose +0.0 mm` on sixteen of twenty Cookie seeds, next to `moved 18.4 mm`.

*Cause*: the Cookie is a 25 mm-tall, 6.4 mm-thin slab standing on its edge; when the closing jaws
catch it, it **topples**, and its centre drops **9.2 mm** (from 12.5 mm above the surface to 3.2).
`object_height_mm.max()` over the episode reports the *peak rise*, so a fall -- the dominant
event in the whole run -- appears as `+0.0`, indistinguishable from "nothing happened".  The
number that revealed it was the displacement: 18 mm of horizontal motion with no lift is only
possible if the object rotated about its own base.

*Rule*: **report both ends of a signed quantity, or you have reported half of it.**  `rose` and
`sank` are now printed and recorded together, and every one of those sixteen seeds shows
`sank = -9.2 mm`.  This is the same family as F4 and C2 -- one end of a comparison standing in
for both -- and it is the one that cost the most to notice, because the missing half was the
half that explained the failure.

### F12. A topple with a tip passes a one-sided lift test

*Symptom*: `grasped it and carried it`, on a rollout whose object moved **16.5 mm**, rose
**+6.7 mm** and then fell **9.4 mm** -- while the target box was 131 mm away.  Nothing was
carried anywhere.

*Cause*: the grasp test was "did the object's centre rise above 5 mm at any step".  A slab that
is prised up by one jaw before it falls satisfies that; it is a *tip*, and its signature is the
fall that follows, which a `max` does not show.  The 5 mm value had itself been chosen against
the settling noise (0.63 mm) rather than against the impostor.

*Rule*: **measure the threshold against the impostor, not against the noise**, and state the
separation.  The impostor here is a tip-over, and what separates it from a carry is not the
height but the **duration**: re-reading the four stored traces, real carries hold the object
above 15 mm for **21 to 247 steps**, and every rollout that rose at all by less than that --
axis 4's two topples, the unaimed run's one, the aimed run's three -- sits at **0 steps above
15 mm**, with peak rises of 2.1, 4.1, 6.7, 8.8 and 2.4 mm.  The thresholds now sit in a 12x gap
in height and a 21-step gap in duration.  A test that requires both a height and a duration is
also the honest statement of the physics: a carried object has left the surface **and stays
away from it**.

---

## G. Cross-scene leakage

A change made for one scene changed another, and only an external reference could see it.

* The mirror added for LIBERO re-mirrored A3's datasets (A5).
* `arm_actuator_damping: 0.40` is scene-local **on purpose**; the default is 1.0 for a reason
  that is written down (D2).

*Rule*: after every change, re-run the **other** scene's invariant.  Keep a guard set --
  `a3_libero_panda` seeds 0-9 (expect >= 9/10) and the cookie dataset fingerprint -- and run
  both after every edit that touches shared code.

---

## The one habit that found all of these

**Compare a channel against an external reference, at the level where the two are supposed to
be equal, and make the comparison exact enough to fail.**

The four fatal bugs were invisible to every self-consistent check: the simulation ran, the
trajectory was smooth, the images rendered, the gripper closed.  What found them was LIBERO's
compiled model (A3, B1, B2, B3, B4, B5), LIBERO's own evaluation image (A4), LIBERO's own
`observation.state` (A1, A2, B5), the A3 dataset's own frames (A5, A6), and a rendered video
(C1).
