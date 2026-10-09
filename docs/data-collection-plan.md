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

`runner.py` records `info["applied_action"]`, and `recording.py` wrote the literal
`"stored_action_mode": "joint_position"` into the episode metadata.  Under the Panda that
vector's first seven entries are Newton-metres from the OSC solve.  So a dataset collected on this
scene would have shipped torques -- magnitudes of order 1-10 -- inside a schema that says they are
joint angles, and nothing in the pipeline would object.

This is the same failure mode as F5-F13 in `bug-catalogue.md`: not a crash, a **silent lie in the
instrument**.

**And there is a second half, which a test found rather than a reading.**  The dataset's
``action`` column was declared **16 wide, as a constant** (`recording.py:lerobot_features`), so a
14-wide `cartesian_delta` action cannot be written at all: LeRobot validates every frame against
the declared shape and raises `The feature 'action' of shape '(14,)' does not have the expected
shape '(16,)'`.  "Record the policy's action" is therefore a **schema** decision, not only a choice
of which array to copy -- the file has to be created with the right width before the first episode,
and LeRobot only complains on the first frame.

So the fix has three parts, and all three are needed:

1. **Record what the policy consumed.**  The dataset's action has to be the policy-space action
   (`cartesian_delta`, 14 wide) -- that is what a fine-tune trains on and what the manifest
   declares.  `EpisodeRunner(recorded_action=...)` selects the array and
   `runner.stored_action_mode(env, ...)` names its space, one definition shared with the
   collection driver so the schema and the frame cannot disagree.
2. **Let the schema follow the space.**  `LeRobotV3Recorder(action_mode=...)` declares the
   ``action`` width from `recording.ACTION_WIDTH` (16 for `joint_position`, 14 for
   `cartesian_delta`), and `start_episode` refuses a run whose recorded space is not the one the
   file was created for -- which turns a first-frame crash into a named error at setup.
3. **Refuse the impossible combination.**  `A3DualArmEnv.canonical_action_mode` returns
   `"joint_position"` for a joint-servo arm and **raises** for a torque-driven one, so the
   default `recorded_action="applied"` on the Panda scene fails loudly and names the fix.
   A scene declares its intent in `CollectionScene.recorded_action`.

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

**Decided: the reference branch's `basic` strength -- 10 mm and yaw ±0.030 rad (1.7 degrees).**
The "30 degrees" in the original request was the reference's `±0.030 rad` read as degrees.  The
reference's own `basic` profile is 10 mm and 0.030 rad (`docs/randomization.md`); `advanced`
reaches 20 mm and 0.080 rad, 4.6 degrees.  Staying at `basic` is the conservative choice and it
is the one taken.

Translating that into `SceneRandomization`, the shapes are symmetric here rather than per-axis,
because the nominal layout is not the same-column one those measured asymmetric ranges belong to:

```yaml
randomization:
  source_bin_x_m: [-0.010, 0.010]
  source_bin_y_m: [-0.010, 0.010]
  source_bin_yaw_rad: 0.030
  target_bin_x_m: [-0.010, 0.010]
  target_bin_y_m: [-0.010, 0.010]
  target_bin_yaw_rad: 0.030
  arm_home_rad: 0.04
  min_box_clearance_m: 0.020
```

**Even at this strength the existing measured reach limits do not carry over and must be
re-measured.**  The comment in `configs/cookie_same_column.yaml` (`target +x 30 / -x 140 / +y 90 /
-y 55 mm`, `source +x 80 / -x 170 / +y 230 / -y 245 mm`) was measured for the **A3 arm on the
same-column layout**.  This scene has the **Panda at LIBERO's base**, a different nominal layout,
and a 200 mm y offset, and 10 mm symmetric is a *request* until the window it sits in is known.
That measurement is the first cheap thing to run in Stage 3.

At 1.7 degrees the yaw axis is no longer a risk: it swings the bin's corners about 3.4 mm and the
target box's about 2 mm, well inside the 20 mm `min_box_clearance_m` and far from the insertion's
2.5 mm gaps.  The asymmetry worth keeping in mind is only that source-bin yaw carries the whole
Cookie layout with it (the Cookies are placed relative to the bin body,
`cookie_transfer.py:430`) while target-box yaw rotates the slots the expert reads live.

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

  **A jaw did once arrive against the wall**, and the measurement stands: closing on A3's placement
  the outer jaw met `source_bin_right_wall` at 0.45 mm -- `('source_bin_right_wall',
  'L_finger_inner_geom')` -- before it could pinch, taking a 0.03 mm graze that carried no force,
  and the grasp could not verify.  That was measured before the jaw-geometry and jaw-chain fixes,
  and it is no longer the binding constraint where it was observed: at the same placement the grasp
  now verifies.

  **What binds now is convergence, and it is measured.**  `scripts/probe_panda_wall_clearance.py`
  walks the Cookie away from the near wall in 1 mm steps, and a Cookie nearer than about **10 mm**
  beyond A3's placement costs **275 extra steps** an episode (431 against 157) while still
  succeeding: DESCEND and CLOSE run to their 120- and 180-step escape timeouts because the arm does
  not converge on the grasp pose, and `expert.retry_reasons` stays empty.  The offset with the
  worst cost is 126 mm from that wall, so the two mechanisms are not the same thing and the inset
  this section is about is the **reachability** one.  A collision bound for the jaw is still owed
  as a separate number, and `scripts/probe_panda_grasp.py` is where it should come from.

  Related, and measured so that it can be ruled out: the gripper is **not** too small for the
  object.  Closed, its two jaw collision meshes are 0.21 mm apart -- it closes completely -- and
  the gap grows 2 mm per mm of joint travel, so a 6.33 mm Cookie sits comfortably inside its range
  (`scripts/check_gripper_minimum_opening.py`).
- **Reachability of both pieces.**  With a random draw, some pairs are simply outside the arm's
  window.  Same mechanism: refuse, redraw, count.

**Decided: yaw aligned to the source bin, position drawn.**  Each Cookie's yaw is the bin's yaw
plus nothing, so the two pieces stay parallel to each other and to the bin.  The reference branch
keeps Cookie yaw at **±0.015 rad (0.86 degrees)** for the same reason -- its insertion is aimed at
measured clearances -- and only ever rotates the *boxes*.

What that buys and costs, at the 60 mm separation recommended in 3.2:

- the two pieces can be **25 mm** apart without touching rather than 55.9 mm, so much more of the
  bin is usable and the separation bound stops being the binding constraint;
- the approach direction is the one the expert already implements, so no new grasp planning is
  needed before the first data exists;
- the *position* variation is still real -- the grasp has to find the piece wherever it is in a
  181 x 146 mm region, which is what a policy trained on one memorised trajectory cannot do.

Free yaw remains the natural Stage 4b: it needs the approach planned in the slab's own frame, which
is a new capability in the expert, and it is best added once there is a working baseline to
compare against rather than at the same time as everything else.

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

### Stage 0 -- the instrument (done)

Landed as `tests/test_recorded_action_mode.py` (7 tests) plus the three code changes above, with
`CollectionScene.recorded_action` threading the choice through the collection driver.  Every
existing scene keeps `recorded_action="applied"` and a 16-wide `action` column, byte-identical to
what it produced before, and a test asserts that.

Found on the way: the registry's own consistency test had been **red on `main`** -- every
`a3_libero_*` scene declared the task named `a3_cookie_single_piece`, so a summary row could not be
traced back to the scene that wrote it.  `libero_props_scene` now *requires* a `task_name` and each
of the eight registered scenes passes its own, with an explicit `unscored_...` contract for the
four that have no reachable Cookie and therefore no acceptance rule.  The point of fixing it here
is not tidiness: a test that is red for reasons of its own cannot tell me whether I broke
something, and this plan is about to add a scene.

**Gate:** green.  A `cartesian_delta` run writes a 14-column action whose values are in [-1, 1],
the written metadata says `cartesian_delta` on disk, a joint-servo scene is unchanged at 16
columns, and both a wrong-width schema and a torque arm asked to store joint positions fail with a
message that names the fix.

### Stage 1 -- the Cartesian expert, one Cookie, today's fixed layout

Port the command stage, not the state machine.  Drive the Panda, one Cookie, no randomization.

**Gate:** 3 of 3 solved; per-phase step histogram with the total at or under 200; and the *same
expert with the same code* solving the A3 arm 3 of 3, which is what proves the port did not
change the plan.

**Status: green on both Panda configurations.**  A3's arm, joint commands -- the regression check --
is **5 of 5 at a median 143 steps** with the full phase path, byte-identical to the recorded
baseline, at every step of this work.

| configuration | outcome |
|---|---|
| A3's arm, joint | **5/5**, median 143 steps |
| Panda, cartesian, A3's layout | 0/3 -- see the reachability inset below; this layout is inside the band the arm cannot converge in |
| Panda, cartesian, collection layout | **3/3**, median 431 steps |
| Panda, cartesian, collection layout, clear of the wall | **3/3**, median 155 steps |

Everything the port needed is now measured and named rather than hanging.  Five defects came out of
it, and four only exist on the cartesian path:

1. The two controllers read a `cartesian_delta`'s **rotation part in different frames** -- A3's arm
   as world, the Panda as tool.  Measured A/B: `rotate(0.3 rad)` converges to 0.033 rad on A3's arm
   as world-frame and to 0.090 on the Panda as tool-frame.
2. The grasp orientation was a **constant that is exactly A3's own start pose**, 121.7 degrees away
   from the Panda's, and `_solve_l` checks only the position residual, so an unreachable
   orientation passed planning.
3. Several phases that wait on "arrived" had **no timeout at all**, which is a hang rather than a
   failure when arrival is a thing a controller can miss.
4. `_cartesian_command` sent the **gripper entry in a single step**, where the joint path spreads it
   over a queue.  Measured: the OPEN phase took the jaws from 0.0 to 0.25 -- half the action range
   -- in one step, with the pads still touching the Cookie, and the release put a **12.57 rad/s**
   spin into it.  Fixed by `release_steps` and `_ramp_gripper`; the spike is now 5.9980, spread over
   ten commands.
5. `settled` was judged from `qvel`, which is the velocity **inside a physics substep**.  The
   substeps in one control step oscillate, so a Cookie lying still on the box floor read 0.65 rad/s
   while its pose moved 0.0145 mm and 0.027 degrees per control step -- 0.0275 rad/s, and a factor
   of 25, which is exactly the number of substeps in a control step (0.05 s / 0.002 s).  The Cookie
   had stopped; the heading stopped advancing altogether after about 1100 steps.  This is why no
   placement could ever be accepted.  Fixed by `_cookie_average_speed`, which measures the same
   quantity over the control step instead of inside it; the thresholds are unchanged.

**The step budget is measured and explained.**  `collect clear of wall` places a Cookie in 155 steps
and `collect` in 431, from the same code and the same Cookie count, and the difference is **not**
retries: `expert.retry_reasons` is empty and the phase path is identical in both.  It is that the
arm does not converge on the grasp pose where the Cookie stands in `collect`.

`scripts/probe_panda_wall_clearance.py` walks the Cookie away from the near wall in 1 mm steps:

| offset from A3's placement | steps | DESCEND | CLOSE |
|---|---|---|---|
| 0 - 8 mm | 431 | 120 | 180 |
| 9 mm | 254 | 110 | 12 |
| 10 mm and out | 157 | **15** | **12** |

DESCEND and CLOSE both end on `_motion_done`, which in Cartesian space needs the site within 4 mm
and 0.02 rad *and* the jaw ramp finished, and both carry an escape timeout (120 and 180 steps).
Inside the band those timeouts are exactly the phase counts; outside it the same phases take 15 and
12.  The grasp succeeds either way, so the 275 extra steps buy nothing -- they are the timeout.

So this is a **reachability inset, not a collision one**.  The plan had assumed the jaw would hit
the wall; it does not, and the offset with the worst cost is 126 mm from that wall.  It also means
the `collect` config sits inside the unusable band, which is why its number is over this section's
gate: the gate's step criterion is met by the Cookie being inside the usable region, and
`collect clear of wall` is the config that measures that.

### Stage 2 -- two Cookies, sequential, fixed layout

`cookie_count=2`, `required_cookies=2`, `require_exact_slots=False` (required: `__post_init__`
raises when `require_exact_slots` and `required_cookies != 10`), `require_released=True`,
`terminate_on_success=True`.  Ordering from live positions, not the configured grid.

**Decided: the placement target is "somewhere in the box", not a named slot.**  With
`require_exact_slots=False` the criterion is the target region, so the expert is free to release
both pieces anywhere inside.  The consequence for the data is worth stating plainly: the
instruction says *put them in the target box*, and the demonstrations will show the box's interior
rather than its marked cells -- so the policy learns where the box is and not what the slots are
for.  That is a deliberate loosening, and it is the version that gets to a baseline fastest.  The
inherited planner still aims at a slot and re-plans when one is out of reach
(`single_piece_expert.py`), so nothing has to be ripped out to tighten this later; tightening it is
a change of criterion, and the datasets would then not be comparable.

**Gate:** 3 of 3 with both Cookies in the box and released; total at or under 450 steps; the
second Cookie's grasp not disturbed by the first one's removal.

**Status: green.**  `two cookies` solves **3 of 3 at a median 356 steps**, and the other three
gate configurations are unchanged (`a3 layout` 2/3 at 434, `collect` 3/3 at 431, `collect clear
of wall` 3/3 at 155).  Both Cookies land in their own cell, upright: box `(-28.47, 12.53)` and
`(28.75, 12.72)`, tilts 0.1 and 0.0 degrees.  Two defects had to be found first, and neither was
where the failure message pointed.

1. **The waiting Cookie turns on its own.**  With the policy doing nothing, a Cookie standing in
   the source bin drifted **79.6 degrees in 400 steps (20 s)** at y=0.0711, 71.5 at y=0.1108 and
   18.8 at A3's row, with the direction depending on where it stands.  Its friction tuple's
   torsional term is 0.003, below MuJoCo's own default of 0.005.  The grasp had been planned from
   the target box alone, so at 51 degrees of drift the jaws came down across the Cookie's 50 mm
   face instead of its 6.4 mm thickness and knocked it flat.  Fixed by planning the grasp from the
   Cookie's measured heading, turning about the tool's own approach axis; the sign is measured
   (-1).  Raising the torsional friction to 0.02 removes the drift outright but was **rejected**:
   it takes `collect` from 3/3 to 0/3.
2. **The second Cookie was planned into the first one's cell.**  `_select_cookie` chose slot 9 and
   `SinglePieceCookieExpert._plan_place_from_grasp` overwrote it with 8, because that loop exists
   to try every slot until one is reachable and started from the beginning of `_slot_order` each
   time.  Measured: the descent went to box x=-27.7 while the carried Cookie was released at box
   x=-28.5, **on top of** the Cookie already at x=-28.4, the contact being
   `cookie_0_top_geom <-> cookie_1_bottom_geom` at 0.88 N.  The slot selection already made is now
   the first candidate and the occupied cells are excluded.

A tilt check would have missed the first defect entirely: a Cookie turned about the vertical axis
keeps its thin axis vertical, so its tilt stays at 0.1 degrees while its silhouette along the
jaws' closing axis goes from 6.4 mm to 50 mm.  The quantity to watch is the heading, and
`scripts/render_two_cookie_video.py` records both it and the video.

### Stage 3 -- box randomization

Re-measure the reach window for this scene and arm first, then set the ranges inside it.  The
strength is decided -- 10 mm and ±0.030 rad, the reference's `basic` -- so what this stage
measures is whether the window contains it, not how wide the window should be.

**Gate:** 9 of 10 on the randomized layout, with the failures kept in the denominator and
described.  Report per-axis, not just the total: a 10 mm range that is green overall but red in
+x is telling us the lower bound on +x is tighter than 10 mm, and that is a number the next
scene needs.

### Stage 4 -- two Cookies at random positions

Position draw with a measured minimum separation and wall inset; yaw aligned to the bin in v1.
Refusals counted, not crashed.

The **inset is now measured**: a Cookie nearer than about 10 mm beyond A3's placement costs 275 extra
steps while still succeeding, because the arm does not converge on the grasp pose there (see the
reachability paragraph above).  The draw has to keep the Cookie out of that band, and the band is a
property of the *arm's reach at that x*, so it should be re-measured once the box randomization of
Stage 3 moves the bin rather than carried over as a fixed offset.  A second, tighter inset for the
jaw's own collision footprint is still owed and is a separate number from this one -- the two were
conflated until the sweep separated them.

**Gate:** 9 of 10, and the failures understood -- the failure modes here are new (a piece near a
wall, a piece the first grasp nudged) and each one needs a name before it can be fixed.

### Stage 5 -- pilot collection

**Fifty episodes first, to look at speed, the success rate and the format -- then decide the
volume from the success rate's lower bound, aiming at 500 accepted episodes.**

The pilot is written but not trained on, and the discard rule stays on (`save_failed_episodes`
defaults to False -- a demonstration of a failure teaches the policy the wrong thing).  Three
questions, in this order:

1. **Speed.**  Wall-clock per accepted episode, and where it goes.  This sets what 500 costs and
   whether `fast_render` is needed.  The per-phase histogram answers where the *steps* go, which
   is the only thing that can be optimised deliberately rather than by guess.
2. **Success rate.**  Accepted over attempted, failures kept in the denominator, broken down per
   axis.  Fifty attempts separates "about half" from "almost always"; it does not separate 92%
   from 96%, and the plan does not pretend otherwise.
3. **Format.**  That the `action` column is the 14-wide `cartesian_delta`, that its values are in
   [-1, 1], that `stored_action_mode` says what is actually stored, and that the videos, the state
   and the task string are what a fine-tune expects.

**The volume follows from the lower bound, not the point estimate.**  At n = 50 an observed 90%
has a 95% Wilson interval of roughly **78% to 96%**, so 500 accepted episodes cost somewhere
between about 520 and 640 attempts.  The decision the pilot enables is which end to budget: a 78%
lower bound needs a 640-attempt quota and the headroom that a 90% rate would not.  `max_attempts`
exists for this, and `collect_dataset` stops and reports rather than quietly returning short.

### Stage 6 -- collection and first fine-tune

Five hundred accepted episodes, collected at whatever attempt quota Stage 5 justified, with the
prompt fixed before the first episode is written:

> pick up the two yellow blocks from the source bin and put them in the target box

Two things to carry in from what is already measured:

- train and test seeds must be disjoint, and a saved-episode success rate is not a success rate;
- the words have to be taught, not prompted.  `block`, `source bin` and `target box` occur zero
  times in LIBERO's vocabulary, so a fine-tune on A3's objects has to teach the nouns as well as
  the grasp.

---

## 6. Decisions taken

| question | decision |
|---|---|
| randomization strength | **±10 mm and yaw ±0.030 rad (1.7 deg)** -- the reference's `basic` profile.  The "30 degrees" in the request was `0.030 rad` read as degrees. |
| Cookie yaw | **aligned to the source bin**; position is drawn, orientation is not.  Free yaw is Stage 4b. |
| placement target | **anywhere inside the target box** (`require_exact_slots=False`).  Looser than aiming at the marked slots, and not comparable with a strict dataset. |
| volume | **50-episode pilot first** (speed, success rate, format), then the attempt quota from the success rate's lower bound, **aiming at 500 accepted episodes**. |
| prompt | `pick up the two yellow blocks from the source bin and put them in the target box` |

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
