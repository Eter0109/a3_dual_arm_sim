# Collecting demonstrations for a Cookie policy

The scene stays what it is today -- LIBERO's matched cameras, LIBERO's Panda on the left arm,
A3's own props, LIBERO's physics block (`a3_libero_panda_single_cookie`, the one branch of
`libero_props_scene` whose `scorable` is true).  What changes is the *driver*: the scripted
expert, what it is allowed to emit, and what the driver puts in front of it.

The point of collecting at all is the finding in
[`scene-transition-and-bug-triage.md`](scene-transition-and-bug-triage.md): the instruction
pathway is live and load-bearing -- the right sentence solves the bowl scene 20 of 20 and a wrong
one 11 of 20, Fisher p about 0.0006 -- while **no phrasing binds this policy to this object**,
Cookie displacement 0.17 mm in fifteen episodes across three phrasings.  The words `block`,
`source bin` and `target box` do not occur in LIBERO's vocabulary (`block` 0, `small` 0,
`cookie` 0 across its templates), so they have to be taught rather than prompted.

---

## 1. What is already in place

Checked rather than assumed, because most of this plan is configuration:

| Piece | Where | State |
|---|---|---|
| Scene: LIBERO camera + Panda + A3 props, `scorable` | `collection.py:1174` (the `a3_single_cookie` branch), registered at `collection.py:1310` | works |
| Task env | `A3CookieTransferEnv(loaded, task_config=...)` -- `collection.py:1174`; the scene **is** a cookie-transfer env, not a bare `A3DualArmEnv` | works |
| Multi-Cookie support | `CookieTransferTaskConfig.cookie_count` / `required_cookies` / `require_exact_slots` / `require_released`, `cookie_transfer.py:88-128` | config-driven, no N is hardcoded |
| Sequential single-piece state machine | `A3CookieTransferExpert` (`expert.py:243`), 13 phases, retry logic, IK | exists |
| "Box stays on the table" variant | `A3SinglePieceExpert` (`single_piece_expert.py:69`) | exists, is what this scene uses |
| Scene randomization: box XY + yaw, arm home, clearance rejection | `SceneRandomization` (`config.py:292`), applied in `_apply_scene_randomization` (`cookie_transfer.py:367`) | exists, needs ranges |
| Per-Cookie placement jitter | `position_noise_m` / `yaw_noise_rad`, `set_cookie_pose` (`cookie_transfer.py:666`) | exists |
| Collection driver, sharding, summary, discard rule | `collect_dataset` (`collection.py:187`) | exists |
| LeRobot v3 writer | `LeRobotV3Recorder` (`recording.py:102`) | exists |

**And the reach question is already answered for this layout.**  The registered scene slides A3's
arrangement by `a3_layout_offset_m=(0.020, -0.200)`, and the code comment records that *A3's own
arm* solves the offset layout 0 times in 3 -- which is why the scene carries that note about
fixing the expert "only when it is actually needed -- for collecting data".  It is needed now, and
the answer is that we are not using the A3 arm: the two behaviour probes measured the **Panda**
reaching all three objects in this exact layout.  It got its fingertip to **10.9-28 mm** of the
Cookie, recorded 23-28 steps of contact (`docs/scene-transition-and-bug-triage.md`), and carried
the target box **170-217 mm** into the source bin.  So the layout is reachable by the arm we are
collecting with, and A3's own arm's failure at this offset is not on the critical path.

---

## 2. Three blockers, in order of how silently they would corrupt the data

### 2.1 The expert speaks joint angles and the Panda listens in torque

This is the measured reason `a3_libero_panda_single_cookie` scores 0 of 40.  Same scene, same
layout, only the arm swapped:

| left arm | controller the env builds | expert result |
|---|---|---|
| `a3` | `CartesianDeltaAdapter` (`env.py:196`) | **3 of 3 solved, 143 steps** |
| `panda` | `OperationalSpaceAdapter` (`env.py:192`) | **0 of 3**, 640 steps, and it reports the misleading `no reachable Cookie remains in the source bin` |

`OperationalSpaceAdapter.convert` (`osc.py:154`) takes a **Cartesian goal** and solves for joint
torques; `_limit_joint_action` (`env.py:357`) deliberately does **not** clip or rate-limit the
left arm's seven entries under it, because they are torques.  Every expert in the repository
returns `action_mode = "joint_position"`, a 16-vector, so on the Panda path its "joint targets"
are written to motor actuators as Newton-metres.

**The fix is smaller than it sounds, because the expert's plan is already in Cartesian space.**
`A3CookieTransferExpert` computes EEF **position + quaternion** waypoints (`_waypoints`, and
`_solve_l(pos, quat, q0)` at `expert.py:418` is a least-squares IK to exactly such a pose).  The
IK is the *last* step before the command, not the plan.  So:

- keep the state machine, the waypoints, the phases and the retry logic unchanged;
- keep `_solve_l` as a **reachability oracle** -- it is what implements the 4 mm acceptance that
  the placement re-planning depends on (`single_piece_expert.py`, "4.53 mm past the IK's 4 mm
  acceptance") -- but stop using its output as the command;
- emit instead `delta = (goal_pose - current_pose) / scale`, clipped to [-1, 1], as the same
  14-wide `cartesian_delta` the policy interface already declares
  (`scripts/adapter_libero_iface.py:381`).

One expert then drives **either** arm: the A3 arm through `CartesianDeltaAdapter`, the Panda
through `OperationalSpaceAdapter`.  That is worth having for its own sake -- it makes the arm an
ablation rather than a fork.

### 2.2 The dataset would record torques in a field labelled joint positions

`runner.py:147` records `info["applied_action"]`, and `recording.py:220` **hardcodes**
`"stored_action_mode": "joint_position"` in the episode metadata.  Under the Panda that vector's
first seven entries are Newton-metres from the OSC solve.  So a dataset collected on this scene
today would ship torques -- magnitudes of order 1-10 -- inside a schema that says they are joint
angles, and nothing in the pipeline would object.

This is the same failure mode as F5-F13 in `bug-catalogue.md`: not a crash, a **silent lie in the
instrument**, always in the direction of "the data looks fine".

The fix has two halves and both are needed:

1. **Record what the policy consumed.**  The dataset's action has to be the policy-space action
   (`cartesian_delta`, 14 wide) -- that is what a fine-tune trains on and what the manifest
   declares.  Either record `info["requested_action"]` when the policy space is Cartesian, or add
   a `record_action` hook so the policy decides, and state the choice in the run's summary.
2. **Refuse the impossible combination.**  `stored_action_mode` must be derived from the arm
   actually in use and asserted, not written as a literal.  A scene whose `left_arm` is `panda`
   combined with a stored mode of `joint_position` should raise at setup.

### 2.3 Two Cookies at random positions break two assumptions

- **The expert orders the Cookies from the configured layout.**  `_cookie_preference`
  (`expert.py:410`) sorts `(SOURCE_POSITIONS[i][1], x)` -- the *configured* 80-slot grid, not the
  live poses.  With random placement that list describes nothing.  Ordering must come from
  `env.cookie_positions` (live, `cookie_transfer.py:268`) with a reachability-aware tie-break.
- **The layout is a leaning stack, and we are asking for two free-standing pieces.**  The
  configured grid is 4 columns x 20 rows at 50.4 mm in x and **8.83 mm** in y, which is closer
  than the 25 mm slab is wide: the Cookies lean on each other, which is exactly why the batch
  experts grasp a *contact chain* of five.  Two Cookies placed at random are a different problem
  -- each one is grasped on its own, and the whole batch machinery is not involved.  That is
  closer to `A3SinglePieceExpert` than to `A3CookieBatchExpert`, which is the variant this scene
  already uses.

---

## 3. Randomization: the numbers

### 3.1 The two boxes

The user's request is 10 mm and 30 degrees.  The reference branch's own `basic` profile is
**10 mm and 0.030 rad, which is 1.7 degrees** (`docs/randomization.md`, and its
`configs/randomization.yaml`); `advanced` reaches 20 mm and 0.080 rad, 4.6 degrees.  So "30
degrees" is **17 times** the reference's loosest setting, and that gap is the first thing to
settle -- see section 6.

Translating the request into `SceneRandomization`, the shapes differ per axis because the arm's
workspace is not centred on the nominal pose -- that is what the existing ranges in
`configs/cookie_same_column.yaml` encode:

```yaml
randomization:
  source_bin_x_m: [-0.010, 0.010]
  source_bin_y_m: [-0.010, 0.010]
  source_bin_yaw_rad: 0.10        # placeholder, see 6.1
  target_bin_x_m: [-0.010, 0.010]
  target_bin_y_m: [-0.010, 0.010]
  target_bin_yaw_rad: 0.10        # placeholder, see 6.1
  arm_home_rad: 0.04
  min_box_clearance_m: 0.020
```

**The existing measured reach limits do not carry over and must be re-measured.**  The comment in
that YAML (`target +x 30 / -x 140 / +y 90 / -y 55 mm`, `source +x 80 / -x 170 / +y 230 /
-y 245 mm`) was measured for the **A3 arm on the same-column layout**.  This scene has the
**Panda at LIBERO's base**, a different nominal layout, and a 200 mm y offset.  What the numbers
are here is a measurement, not an inheritance, and it is the first cheap thing to run.

Yaw is the axis that needs care, and the two boxes are not symmetric:

- **source bin yaw** rotates the whole Cookie layout with it (the Cookies are placed relative to
  the bin body, `cookie_transfer.py:430`), so it is free variation for the arm;
- **target box yaw** rotates the ten slots, and the expert reads the *live* box frame, so it
  should cope -- what needs checking is the insertion, which is aimed at 2.5 mm gaps;
- 30 degrees on a 237 x 202 mm bin swings its corners about **60 mm**, which eats the
  `min_box_clearance_m` budget against a target box that is only 66.9 mm away nominally.  The
  rejection sampling already handles this by redrawing, but if the ranges make an acceptable
  layout rare, every episode pays for it in draws -- and `max_clearance_attempts` turns "the
  ranges ask for a layout that cannot exist" into a loud error, which is the right failure.

### 3.2 Two Cookies in the bin, not too close

Geometry first: the Cookie is a 50 x 25 x 6.4 mm slab standing on its edge, half-diagonal
27.95 mm.  Two of them are guaranteed not to overlap at **any** pair of yaws once their centres
are 55.9 mm apart; side by side at equal yaw the true bound is 25 mm.  The bin interior is
237 x 202 mm, so inset by the half-diagonal the centres may live in 181 x 146 mm.

Acceptance rate of rejection sampling in that region, 200k draws, measured just now:

| min centre separation | first draw accepted |
|---|---|
| 40 mm | 84.9% |
| 50 mm | 77.8% |
| 56 mm (hard no-overlap bound) | 73.3% |
| 70 mm | 61.6% |
| 80 mm | 53.0% |
| 100 mm | 36.5% |

So a separation bound costs almost nothing: even 80 mm is accepted on the first draw half the
time, and a redraw loop converges immediately.  **Recommended starting value: 60 mm** -- 4 mm
clear of the mathematical no-overlap bound, and enough for the fingertips to come in on one piece
without dragging the other.  The bound is the knob to turn if the expert starts disturbing the
piece it is not aiming at.

Two things the draw must also respect, and neither is in the table above:

- **Wall clearance.**  A Cookie 20 mm from a 26 mm wall may be ungraspable, because the fingers
  have to straddle it.  The inset above is the slab's half-diagonal, which is a *collision*
  bound, not a *grasp* bound.  The usable region has to be inset by whatever the fingertip needs,
  measured, and the expert must be able to refuse a draw rather than fail on it -- `collect_dataset`
  already treats an expert's raised refusal as a rejected attempt rather than a crash
  (`collection.py:300-320`), so refusing is cheap and is the honest option.
- **Reachability of both pieces.**  With a random draw, some pairs are simply outside the arm's
  window.  Same mechanism: refuse, redraw, count.

### 3.3 Cookie yaw: the decision that decides whether this is easy

The reference branch keeps Cookie yaw at **±0.015 rad (0.86 degrees)** -- essentially fixed --
because its insertion is aimed at measured clearances; only the *boxes* rotate.  For a slab that
must be grasped across its 25 mm width, approach direction is the whole grasp: a Cookie at 45
degrees to the nominal needs a 45-degree wrist rotation, and the expert has no such degree of
freedom in its plan today.

So the honest options are:

- **yaw aligned to the bin** (each Cookie's yaw = the bin's yaw, drawn position only).  The
  Cookies are then parallel, the min-separation bound is 25 mm rather than 56 mm, the approach is
  the one already implemented, and the *position* variation is still real.  **This is the
  recommended first version.**
- **yaw free** (a draw, then the expert has to plan the approach into the slab's own frame).  More
  variation, and a new capability in the expert -- worth doing second, not first.

---

## 4. Speed

What the action space allows, from `config.py:392-440`:

| quantity | value | meaning |
|---|---|---|
| `control_hz` | 20 | 50 ms per control step |
| `cartesian_translation_scale_m` | 0.025 | full-scale delta = 25 mm/step = **0.5 m/s** |
| `cartesian_rotation_scale_rad` | 0.10 | full-scale = 5.7 deg/step = 114 deg/s |
| `max_gripper_step` | 0.08 | jaw travel per step |
| `horizon` | 6000 | the cap, not the target |

The measured baseline is **143 steps for one Cookie** on this scene with the A3 arm, and the
reference branch's batch expert takes **1855-1860 steps for ten Cookies**, about 186 each.  So the
two agree, and the budget to hold is:

- **<= 200 steps per Cookie**, **<= 450 steps for a two-Cookie episode** (about 22 s of
  simulated time),
- and the episode must stop on `DONE` rather than run the horizon.  `CookieCollectionPolicy`
  already does this (`collection.py:30`, `hold_steps`), with one trap recorded in the code: at
  `hold_steps=0` the stopping condition is true on the first step and every episode ends one step
  long with `terminated=False` and no error, which is how `a3_libero_panda_single_cookie` reported
  0 of 20.

A slow expert is not a cosmetic problem: it is 450 frames of "the arm creeping" that the policy
has to imitate, and it wastes wall-clock and disk.  Where the time goes should be measured per
phase rather than assumed -- the phase enum makes that a per-phase histogram.

---

## 5. Plan

Each stage has a gate and lands only when the gate is green.  One GPU process at a time, ten seeds
per iteration at most.

### Stage 0 -- the two instrument fixes (blocking, no data)

1. `stored_action_mode` derived from the arm and asserted, not a literal (`recording.py:220`).
2. The action stored is the policy-space action, and the choice is in the summary.

**Gate:** a one-episode smoke run on `a3_libero_panda_single_cookie` whose metadata says
`cartesian_delta`, whose `action` array has 14 columns, and whose values are in [-1, 1].  And a
deliberate-rejection test: a Panda scene asked to store `joint_position` raises.

### Stage 1 -- the Cartesian expert, one Cookie, today's fixed layout

Port the command stage, not the state machine.  Drive the Panda, one Cookie, no randomization.

**Gate:** 3 of 3 solved; per-phase step histogram with the total at or under 200; and the *same
expert with the same code* solving the A3 arm 3 of 3, which is what proves the port did not
change the plan.

### Stage 2 -- two Cookies, sequential, fixed layout

`cookie_count=2`, `required_cookies=2`, `require_exact_slots=False` (required: `__post_init__`
raises when `require_exact_slots` and `required_cookies != 10`), `require_released=True`,
`terminate_on_success=True`.  Ordering from live positions, not the configured grid.

**Gate:** 3 of 3 with both Cookies in the box and released; total at or under 450 steps; the
second Cookie's grasp not disturbed by the first one's removal.

### Stage 3 -- box randomization

Re-measure the reach window for this scene and arm first, then set the ranges inside it.  Start at
the reference's own strength (10 mm, 0.03-0.10 rad yaw) and step the yaw up towards the requested
30 degrees only while the gate holds.

**Gate:** 9 of 10 on the randomized layout, with the failures kept in the denominator and
described.  Report per-axis, not just the total: an asymmetric range that is green overall but
red in one direction is a range that is too wide in that direction.

### Stage 4 -- two Cookies at random positions

Position draw with a measured minimum separation and wall inset; yaw aligned to the bin in v1.
Refusals counted, not crashed.

**Gate:** 9 of 10, and the failures understood -- the failure modes here are new (a piece near a
wall, a piece the first grasp nudged) and each one needs a name before it can be fixed.

### Stage 5 -- pilot collection

Twenty episodes, written but not trained on, with the discard rule on (`save_failed_episodes`
defaults to False -- a demonstration of a failure teaches the policy the wrong thing).  Look at
the *shape* of the data: per-episode step counts, per-phase time, and whether the two Cookies give
two genuinely different trajectories or one trajectory twice.

**Gate:** the acceptance rate and the step distribution are known and stable, and `collection_summary.json`
records the ranges actually in force.

### Stage 6 -- collection and first fine-tune

Volume is a decision for section 6.  Two things to carry in from what is already measured:

- train and test seeds must be disjoint; the reference branch's own warning is that a saved-episode
  success rate is not a success rate, so failures stay in the denominator;
- the words have to be taught, not prompted.  The prompt for this task names `two yellow blocks`,
  `the source bin` and `the target box`, and none of those words occur in LIBERO's vocabulary.

---

## 6. What I need decided

### 6.1 Is 30 degrees real, or did `±0.030 rad` read as 30 degrees?

The reference's `basic` profile is `±0.030 rad`, which rounds to 1.7 degrees.  If the intent was
"the reference's basic profile", this is a typo and the yaw is 1.7 degrees.  If 30 degrees is
real, it is 17x the reference's loosest profile and it is the single biggest source of risk in the
plan -- it swings the bin's corners 60 mm into a 67 mm nominal gap, and it changes the grasp
approach for the target slots.  My recommendation is to **implement the knob, start at the
reference's strength, and step up with a gate at each step**, so the number ends up measured
rather than chosen.

### 6.2 Cookie yaw: aligned to the bin, or free?

Aligned (recommended first) keeps the existing approach and still gives real position variation.
Free yaw needs the expert to plan the approach into the slab's own frame -- a new capability --
and it drops the min-separation requirement from 56 mm to 25 mm.

### 6.3 Where do the two Cookies go -- slots, or anywhere in the box?

The target box is A3's ten-slot box, and the environment's criterion is "in the target region"
with `require_exact_slots=False`.  The expert's existing slot order aims at slots 8 and 9 and
re-plans when unreachable.  Aiming at slots is the harder, more informative target and it makes
the success criterion strict; "anywhere in the box" is easier and looser.  The two produce
different datasets.

### 6.4 Volume and the prompt

How many accepted episodes for the first fine-tune, and is the prompt "pick up the two yellow
blocks from the source bin and put them in the target box" -- or a LIBERO-shaped sentence with
LIBERO's container words?

---

## 7. Risks, ranked

1. **The Panda under OSC is a different controller from the one the expert was tuned against.**
   Not a blocker -- the probes measured it reaching all three objects in this layout -- but its
   gains, its settling and its steady-state error are its own, and the expert's tolerances
   (4 mm IK acceptance, arrival tolerances, contact thresholds) were tuned on a joint-servo arm.
   Stage 1 is where they get re-tuned, and the per-phase histogram is how.
2. **The grasp of a 6.4 mm slab is the one capability that has never worked.**  Every measurement
   in this repository says the same thing: the policy reaches this Cookie and topples it
   (`sank -9.2 mm`), and axis 4 scores 2 of 20.  The expert grasps it today *on the leaning
   stack*, held by its neighbours; a free-standing piece at a random position is a harder grasp,
   and that is the thing most likely to fail in Stage 4.
3. **30-degree yaw**, per 6.1.
4. **Rendering/disk cost at collection volume**, measured in Stage 5 rather than guessed now.

---

## 8. What is deliberately not in this plan

- **Modifying the expert's state machine.**  It is the reference controller, it is tested, and the
  evidence says the plan is not what is broken -- the *output stage* is.  Change one thing.
- **Re-aiming LIBERO's camera, or changing the layout offset.**  Both were tried and priced
  (`a99c876` for the re-aim; the offset's cost is recorded in the scene).  Staying on LIBERO's
  geometry is the point of this configuration.
- **The two-box relay and the batch experts.**  They solve a different task (ten pieces via a
  contact chain) and the scene here is not that task.
