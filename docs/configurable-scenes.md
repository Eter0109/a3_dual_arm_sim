# Configurable cookie scenes: one spec instead of a scene per layout

Status: plan, 2026-09-23 (branch `feature/configurable-cookie-scenes`)

## Goal

Today there are two tuned, hand-calibrated scenes -- one box and two boxes -- and
everything that makes them work is written down twice: the box geometry is a pair
of hand-typed tuples, the batch size is the literal `5`, the relay's lane is the
literal `0.090`, and the randomization ranges are a comment explaining which
measurement bounded them.  A third layout means a third copy of all of it.

The target is a *scene spec*: a handful of key parameters, with everything else
derived from them by a formula that is itself testable.  The key parameters are

| parameter | meaning | today |
|---|---|---|
| `boxes` | how many boxes the run fills | 1 or 2, as two code paths |
| `per_grasp` | Cookies taken in one grasp | the literal `5` |
| `box_capacity` | Cookies one box holds | the literal `10` |
| `source_cookies` | Cookies in the source bin | `len(cookie_source_positions_m)` = 80 |
| `randomize` | whether the scene varies between episodes | per config, ranges hand-measured |

and the acceptance rule is not "every layout works" but **"every layout inside
the declared bounds collects at a usable yield"** -- the bounds are part of the
feature, and a spec outside them is refused with a reason rather than collected
badly.  That is what makes the search space finite and the testing finite with it.

Two of the five are deliberately *later*: `source_cookies` and `box_capacity` are
in the table as key parameters because they are what the derivation has to be built
around, but the first milestone only has to hold them at their current values (80
and 10) while the other three move.  They are listed together anyway, because a
derivation that has to be extended later is a different derivation from one that
was built for them and only tested at two points.

The measurements this plan rests on are re-takeable:
`scripts/measure_scene_envelopes.py` prints all three sweeps.

## What is hard-coded today

| file | symbol | the hard-coded thing |
|---|---|---|
| `config.py` | `COOKIE_SOURCE_POSITIONS` | a module constant: 4 columns x 20 rows |
| `config.py` | `TARGET_SLOTS_LOCAL` | a module constant: 2 columns x 5 rows |
| `config.py` | `CookieSceneConfig.target_bin_half_size_m` | hand-typed per config |
| `config.py` | `SimConfig.__post_init__` | refuses unless exactly 10 slots |
| `cookie_transfer.py` | `CookieTransferTaskConfig.__post_init__` | refuses unless `required_cookies == 10` |
| `cookie_transfer.py` | `_apply_scene_randomization` | one target box + one optional spare |
| `model.py` | `_add_cookie_scene` | builds `target_bin`, optionally `spare_target_bin` |
| `batch_expert.py` | `_candidate_batches`, `_select_batch` | `5`, `5 * pitch` |
| `batch_expert.py` | `_place_pose` | batch index -> column index, 1 batch per column |
| `batch_expert.py` | `_validate_target_workspace` | `for column in range(2)` |
| `batch_expert.py` | `VERIFY_RELEASE` | `batch_index == 1` is "the last batch" |
| `same_column_batch_expert.py` | `_candidate_batches` | `5`, and the neighbour at `+1` |
| `two_box_batch.py` | `TwoBoxBatchExpert` | two boxes, `A`/`B`, `+0.090`, `85.0`, `8.0`, `n - 20` |
| `tasks.py` | `_BATCH_CONTRACT` | the string `"released_10_upright_settled_contained_70_remain"` |
| `collection.py` | `cookie_*_scene` | one builder per scene |

The pattern is that the *counts* are literals in eight files and the *geometry* is
literals in two.  The counts are the easy half; the geometry is where the
derivation has to be honest about what is measured rather than computed.

## The measured boundaries

Everything below is measured on this machine, and each number is what sets a
bound in the spec.  They are recorded here because the spec's job is to *derive*
these, and a derivation is only as good as the numbers it reproduces.

### The station: where a box can be filled

The fill's own reachability pre-check is the authority (`_validate_target_workspace`,
which refuses a pose whose IK misses by more than 2 mm / 0.035 rad).  Sweeping it
by rebuilding the env at each pose:

```
  y\x     30      60      75      90     110      (mm, offset from the relay's nominal)
     0      .       .       .       .       .
    30      o       o       o       .       .
    60      o       o       o       o       .
    90      o       o       o       o       o
   120      o       o       o       o       o
   150      o       o       o       o       o
```

The usable set is a diagonal band: at x = 0.075 a box is fillable from y = 0.030
up, at x = 0.110 only from y = 0.090 up.  The relay's station (0.075, 0.030) sits
exactly on the corner, which is why it works and why it has almost no room in -y
(the shipped range is `[-0.006, +0.040]`).  This is also the earlier per-direction
measurement restated: the working box has +118 mm in y and -8 mm.

**Consequence.**  A station is one box, not a region: the usable band is ~80 mm
wide in x and the box is 126 mm.  Boxes therefore cannot be filled in place
side by side except in y, and only two of them (`2 x 62 + clearance = 149` against
a ~130 mm band) -- so N >= 3 must be a lane with a relay, not a wider station.

### The right arm: the lane it can work in

Both the push and the carry go through `BoxSupportController.solve`, whose own
acceptance rule (12 mm of position error, 0.04 rad) is the envelope.  Swept at the
two heights the controllers use (0.755 for the push, 0.780 for the carry -- the
two grids came out identical):

```
  y\x      0      25      50      75     100     125      (mm)
   100     o       o       o       .       .       .
    75     o       o       o       o       .       .
    50     o       o       o       o       o       .
    25     o       o       o       o       o       o
     0     o       o       o       o       o       o
   ...     o       o       o       o       o       o
  -520     o       o       o       o       o       o
```

What matters for a lane is not the site's reach but the **pads'**, because the pads
are what touch the box's rear wall.  Re-solved in the pad frame, in 1 mm steps:

```
  station x    0.030  0.040  0.050  0.060  0.075  0.090  0.100  0.110  0.120
  pad +y cap   124    119    114    109     99     80     72     68     57   (mm)
```

Three facts come out of this and they are the whole reason the lane is shaped the
way it is:

* **+y is capped at about +0.100, and falls off a cliff past the shipped station.**
  +y and +x compete for the same arm reach.  The push that parks a filled box works
today because the pads only travel to the box's rear wall -- the box's centre
reaches +0.120 while the pads stop at +0.089, using 86 mm of the 99 available at
x = 0.075.  One box, and no more: parking a second box at +0.207 would need the
pads at +0.176.
* **A lane wants its station as far in as the fill's band allows.**  124 mm at
x = 0.030 against 99 at the shipped 0.075, and 80 at the single-box scene's 0.095 --
where no ten-Cookie box could be pushed at all.  Since the fill band's floor is
flat at y = 0.030 for every x from 0.030 to 0.075, there is no reason to sit at
0.075 except that the relay already does.
* **-y is open past -0.520** (checked past the sweep's edge; the table runs to
-0.750).  A queue of six boxes at the derived pitch fits, so it is not what bounds
N -- but the *queue gap* can be: the shipped relay places its spare 160 mm back
against a derived pitch of 87 mm, and at that gap five boxes run off the end.

**Consequence.**  The parked boxes cannot be pushed one at a time to their own
slots.  The mechanism that does work is the one a person would use: the pads push
the box at the station into the line already parked, and **the line advances by
one pitch per fill**.  Pad travel is unchanged, and the parked boxes are never
touched directly.  That is the derived rule:

```
lane_pitch      = box_extent_y + min_box_clearance     (87 mm for a 31 mm-half box)
push_destination = station_y + lane_pitch              (the station box's target)
parked_i         = station_y + lane_pitch * (k - i)    (after filling box k)
```

### The source bin: how far it can get out of the way, and how much it can supply

Two questions, and the second one is the one that bounds N.

**Getting out of the way.**  The parked line grows towards +y and the source bin is
in the way.  Moving the bin is free: every 5-Cookie batch position of the two
usable columns still solves at every offset tested, with no error at all.

```
  source bin +y offset     +0    +58   +148   +238   +280   +340   (mm)
  usable batches solving  16/16  16/16  16/16  16/16  16/16  16/16
```

and the offsets the lane needs are +58 / +148 / +238 mm for N = 3 / 4 / 5.  The
table's own edge binds first, at about +328 mm (the bin is 107 mm deep and the
table ends at y = 0.750), so the source bin is never the reason a lane fails.

**Supplying the Cookies -- this is the real bound.**  The source grid is not
uniformly reachable, and assuming it was is the kind of error this document exists
to prevent.  Sweeping every 5-Cookie batch start of every column:

```
  first row\col     99    150    200    251      (mm, the layout's four columns)
        0           o      o      .      .
       ...          o      o      .      .
       15           o      o      .      .
```

The two columns nearest the arm are graspable at every batch position; **the two
far columns are refused outright** -- 7.24 mm of IK error against a 4 mm tolerance
at x = 0.200, 19.12 mm at x = 0.251 -- and the map is identical at every +y offset
above.  So the shipped 4 x 20 layout offers **40 graspable Cookies, not 80**: the
expert's candidate rule silently skips the far columns, and they are decoration.

That is the binding constraint:

```
  usable_source = usable_columns * rows        = 2 * 20 = 40
  N * box_capacity <= usable_source            -> N <= 4 for C = 10
```

**Consequence: `boxes <= 4`**, and it is a *source supply* bound rather than a
geometry bound -- which matters, because it says the way to raise N is to give the
source bin more usable Cookies (a 2-column, 30-row layout supplies 60), not to move
it further.  `N = 5` needs 50 against 40 and is refused by the spec with that
number in the message.

### Speed: what the profile costs

Measured last session, and the numbers the adaptive-profile phase is judged
against:

| | steps | rate |
|---|---|---|
| single box, `fast` | 469 (mean of 5) | ~2.3 steps/s |
| single box, `baseline` | 1657 | |
| relay, `baseline` | 4823 (mean of 3, re-measured) | ~1.1 steps/s |
| relay, `fast` | 2053 -- runs, not yet reliable | |

Per-step cost is 428 ms of which rendering is 307 ms (71%), so the profile buys
step count and shards buy wall clock: 16 shards give 18.8 steps/s against 1.6 for
one, near-linear.

## The derived parameters

Each of these is a pure function of the key parameters, and each is pinned by an
anchor: **the spec for (N=1, K=5, C=10, S=80) must regenerate
`configs/cookie_same_column.yaml`'s `cookie_transfer` block, and (N=2, ...) must
regenerate `configs/cookie_two_box_batch.yaml`'s**, within a stated tolerance.
That anchor is what keeps the derivation honest, and it is the first test to
write.

### 1. The target box's slot lattice, from capacity and grasp size

The two shipped lattices decode to three numbers, and they are all derivable:

```
y pitch between slots = cookie thickness                      0.0063333  (0.019/3)
x spacing between columns = cookie width + column_gap         0.056      (0.050 + 0.006)
inner_half_x = x_spacing/2 + cookie_half_x + x_margin         0.057      (0.028 + 0.025 + 0.004)
inner_half_y = (rows-1)/2 * y_pitch + cookie_half_y + y_margin
                                                              0.025      (0.012667 + 0.003167 + 0.009167)
```

The y pitch is the *nominal* thickness, not the source pitch: the source lays the
Cookies out with a clearance so they do not interpenetrate at reset, while the
target slots are where the batch comes to rest after the jaws compressed it.  The
y margin is the pad clearance -- the jaws descend into the box outside the batch,
so the box has to be wider than the batch by the pads' own thickness plus room to
move.

Both margins are *chosen*, not derived, and the spec carries them as fields: the x
margin is 4 mm in both shipped configs, and the y margin is 9.167 mm in the relay
(the tightest that leaves the 6.35 mm pads room) against 13.167 mm in the
single-box scene, which has the room.  So the derivation reproduces each scene by
taking the margin from the scene, and what it *derives* is everything the margin is
added to.

The *shape* is then a choice of `columns`, with `rows = ceil(box_capacity /
columns)`.  Two constraints pick it:

* `per_grasp <= rows` -- a batch is placed into one run of slots inside a column, so
  it cannot be longer than the column.  (A batch cannot span two columns at all:
  the jaws hold it in a line, so continuing into the next column would need a
  sideways move mid-place.)
* `per_grasp * source_row_pitch_y + pad_thickness <= 0.085` -- the jaw travel.  This
  caps `per_grasp` at 10 with the shipped 2.5 mm row gap.
* growing `columns` grows the box in x, and the fill reaches 30 mm past the shipped
  outermost slot, so **three columns fit with 2 mm to spare and four do not**.
  `columns = 2` stays the default because it is what both shipped scenes use.

So for C=10, `columns=2`, `rows=5`, and `per_grasp` can be 1..5.  To grasp more
than 5 the box has to be a single column of 10, which is a different box shape --
worth offering, and worth testing, but not the default.

### 2. The batch plan, from capacity and grasp size

```
per column: sizes = [per_grasp] * (rows_used // per_grasp) + [rows_used % per_grasp]
slots_of(batch b) = the next `sizes[b]` slots of that column, from the -y end
```

With C=10, K=5 this is `[5, 5]` and batch 0 takes column 0's five slots -- exactly
today's behaviour, which is why the anchor holds.  With C=10, K=3 it is
`[3, 2, 3, 2]`, not `[3, 3, 3, 1]`.

The first version of this plan was `[per_grasp] * (capacity // per_grasp) +
[remainder]`, spreading the remainder across the *whole box*, and it is wrong for
the reason above: there is nowhere to place a second batch of three when column 0
has two rows left.  The remainder is per column.  The plan is implemented in
`batch_plan.py` and pinned by a test that partitions every capacity from 1 to 24
against every grasp size from 1 to 10.

This is also where "how many Cookies to take next" is decided, and it is a running
counter rather than a fixed list: `min(per_grasp, remaining_in_this_column)`.
Stated that way the same code resumes a partially filled box, which is what the
user's later milestone (a box that arrives with Cookies already in it) needs.

`_place_pose` becomes `_place_pose(clearance, slots)`, and
`_validate_target_workspace` iterates the batch plan instead of `range(2)`.

### 3. The source layout, from source count and grasp size

```
col_pitch = cookie_width + 0.4 mm            (0.0504 -- the columns are not entered)
row_pitch = cookie_thickness + 2.5 mm        (0.0088333 -- this is the jaw's lane)
rows      = ceil(source_cookies / columns)
positions = the `columns`-column lattice at that pitch, centred on the nominal centre
source_bin_half = grid_extent/2 + cookie_half + margin + wall
require N * box_capacity <= usable_columns * usable_rows
```

With S=80, K=5, columns=4 this gives rows = 20 and reproduces the shipped layout
exactly.  The row pitch is a correction: the plan first said it was
`0.019/3 + 0.0004 = 0.0067333`, copying the module constant `COOKIE_PITCH`, but the
shipped configs use `0.0088333` -- the Cookie's thickness plus the 2.5 mm lane the
fingertip descends into, which they widened when the Cookies were made thinner.
The anchor test is what caught it, which is the argument for having one.

The last line is the constraint that sets `boxes <= 4`, and it is a *supply*
constraint rather than a geometry one: the far columns are unreachable, so they
contribute nothing, and whether the layout keeps them for the look of a full bin is
a presentation choice.  The spec therefore carries the usable counts as *measured*
inputs -- `usable_source_column_max_x_m` and a per-column y cap -- rather than
inferring them from `columns` and `rows`, because inferring them is exactly how a
spec would end up advertising N = 6 and refusing half of them at collection time.

**One consequence is not obvious and cost a whole box.**  The band bounds a batch's
*centre*, and a one-Cookie batch's centre is the Cookie itself, which sits closer to
the band's edge than a five-Cookie batch's centre does.  So at the shipped layout,
dropping `per_grasp` to 1 loses the last three rows of every column: the supply
falls from 40 to 34 and `boxes` from 4 to 3.  `per_grasp` is not free.

### 4. The lane, from the box count

```
lane_pitch   = box_extent_y + min_box_clearance      # what the line COMPACTS to
queue_gap    > lane_pitch                            # what it is LAID OUT at
station      = the nominal station, clamped into the fillable band
queue_i      = station_y - queue_gap * i             for i in 1..N-1
source_offset = max(0, source_front_clearance_need(N))
```

`queue_gap > lane_pitch` is not a preference, it is the reason the derived default
is refused at load: the pitch is exactly the clearance between *outer* extents, so a
lane laid out at it has nothing left for the draws.  The gap also needs room for the
queue's own y range on top of that -- measured, the relay's 160 mm gives 86 mm of
outer gap, which absorbs the 55 mm the queue's ranges can close with 31 mm to spare.

`source_offset` is the formula from the measurement above; the spec also derives
the source bin's *yaw* range from zero, because a yawed bin changes the pick pose.
Note that it depends on `lane_pitch` and not on `queue_gap`: the parked line is
where the *fills* put the boxes, and the initial spacing is washed out by the first
shove.

### 5. The randomization ranges, from everything above

Three bounds apply to each box, and the range is the tightest of them:

1. **The fill's own window** (2 mm / 0.035 rad) for the station box.  This is what
   bounds the relay's spare box to +/-4 mm in x: the carry corrects y only, so
   whatever x offset a box starts with is the offset it arrives with, and
   `sqrt(8^2 - 4^2) = 6.9 mm` is the whole budget.
2. **The right arm's envelope** for the queue and parked boxes (the grid above).
3. **The pairwise clearance**: `sum(extent) + (N-1)*min_clearance` must fit the
   lane, so the y ranges shrink as N grows.  A formula, not a table:

```
free_y = lane_length - N*box_extent_y - (N-1)*min_clearance
range_y_per_box = min(right_arm_y_reach_at(that box), free_y / (2*N))
```

and `_boxes_are_clear` (already pairwise, already yaw-aware) stays the referee.

**This is the part where a formula is not enough**, and the plan should not pretend
otherwise: bound 1 is pose-dependent, and the earlier session measured that
copying the single-box scene's ranges into the relay scene produced a run that
failed at 2.07 deg against a 2.005 deg threshold.  So the spec carries a
**validation step** rather than a promise: draw M layouts from the ranges, run the
expert's own reachability check and the clearance check on each, and refuse the
spec if any draw fails.  A Monte-Carlo gate, not a proof -- cheap (no physics),
and it catches exactly the class of error that was made by hand last time.

### 6. The profile

The adaptive rule is not "pick fast when it looks safe".  The measurement is that
`fast` *requires* `arm_actuator_damping = 1.0` (its arrival test needs a settled
response) and the relay's push and carry were calibrated against damping `0.0`, so
today the two settings are one choice and the relay is pinned to `baseline`.
The win is to make them independent: the carry already needs a tighter
`GRASP_OPENING` (0.085) under damping, which is a re-tune of the controller rather
than a different model.  Once that lands, one model serves every layout and the
profile selector is `auto -> fast` with no exceptions.

Until then the spec's `profile` is `auto` = `baseline` when `boxes >= 2`, `fast`
otherwise -- and the reason is recorded in the config, not in a comment.

## Architecture

```
src/a3_dual_arm_sim/scene_spec.py     DONE  pure derivation, no MuJoCo import
src/a3_dual_arm_sim/batch_plan.py     DONE  the batch plan and slot grouping
tests/test_scene_spec.py              DONE  the two anchors, and every bound
scripts/generate_scene_config.py      DONE  spec -> config, for a new layout
tests/test_scene_lanes.py             DONE  N boxes: config, model, reset, clearance
src/a3_dual_arm_sim/scene_spec_validate.py  NEW  the Monte-Carlo gate
src/a3_dual_arm_sim/model.py              DONE  N target boxes instead of target+spare
src/a3_dual_arm_sim/cookie_transfer.py    DONE  N boxes in randomization and reset
src/a3_dual_arm_sim/batch_expert.py        batch plan instead of the literal 5
src/a3_dual_arm_sim/same_column_batch_expert.py   same
src/a3_dual_arm_sim/relay_batch_expert.py  NEW  TwoBoxBatchExpert generalised to N
src/a3_dual_arm_sim/tasks.py               contract strings derived from the spec
src/a3_dual_arm_sim/collection.py          one spec-driven scene builder
src/a3_dual_arm_sim/cli.py                 --scene-spec key=value,...
configs/generated/*.yaml                   the specs the shipped scenes decode to
```

`scene_spec.py` must not import MuJoCo, for the same reason `tasks.py` does not:
`training` imports on hosts where the simulator wheels will not load.  It imports
`batch_plan` and nothing else.

The shipped YAML configs stay.  They become the *frozen artifacts* of two specs,
and a test asserts `spec.derive_config()` reproduces them -- so the configs keep
working for anyone who has them, and the derivation cannot silently drift.  The
anchor compares the derived key set exactly and the *whole* shipped key set against
`DERIVED_KEYS | TUNING_KEYS`, so adding a key to a config fails the test until
someone decides whether it is derived or tuning.  That is the mechanism that makes
"the derivation covers the config" a fact rather than a hope.

## Phases and gates

Each phase is a commit with a measured gate.  Nothing is measured by "it looks
right"; every gate is a number, and a phase that cannot reach its gate is a phase
whose finding gets written down instead.

**Phase 0 -- baseline.**  No code.  Re-record the reference numbers on this branch
(single box 5/5 mean 469, cross-column 479, relay 3/3 mean 4823) and the full
suite.  *Gate:* 154 tests pass, numbers recorded in the commit message.

**Phase 1 -- the derivation, as pure functions.**  `scene_spec.py` +
`batch_plan.py`, no simulator.  *Gate:* the two anchor configs are regenerated
field by field within tolerance; the derived `boxes <= 4` and `per_grasp <= rows`
bounds are asserted from the measurement tables in this document; a spec outside
them raises with the number that was exceeded.

*Done.*  24 tests in `tests/test_scene_spec.py`.  Both anchors reproduce to
3.3e-11, which is the precision the configs are printed at.  The derivation is a
function of eight geometry inputs, and the validation collects every problem rather
than raising on the first.  What it found on the way is above: the source row pitch
was wrong in the plan, the batch plan's remainder is per column, the push envelope
falls off a cliff past the shipped station, and a small `per_grasp` costs source
rows.  What it also settled: the station band is independent of the box's height,
because a placement is aimed at a *column's* centre and a centred lattice has its
column centres at the body's own y whatever the row count -- so a capacity of 18 is
as reachable as a capacity of 10.

The bounds it derives at the shipped station, for reference:

```
  per_grasp   <= min(target_rows, jaw limit) = min(5, 10) = 5   (10 at capacity 20)
  capacity    <= 18      (34 at station x = 0.030, 26 at 0.050, 0 at 0.095)
  boxes       <= 4       (source supply 40 over capacity 10)
  source      <= 40 graspable of 80 laid out   (34 if per_grasp = 1)
```

**Phase 2 -- N boxes in the model and the env.**  *Gate:* the existing two-box
tests pass unchanged, the relay is still 3/3, and a 3-box scene builds, resets and
passes the Monte-Carlo gate (no fill attempted yet).

*Done.*  `queue_target_bin_world_positions_m` carries the boxes past the second,
`model._add_cookie_scene` builds them in a loop, and `_apply_scene_randomization`
draws every box from the ranges its *role* implies -- the station from
`target_bin_*`, everything behind it from `spare_bin_*`, because whatever ends up
in the queue has to arrive at the station inside the fill's window.  The clearance
check is the same pairwise test as before, now over a set.

The "relay is still 3/3" gate was met **by construction rather than by a re-run**,
which is worth recording as a method: for a two-box config the draw *order* is
unchanged -- the source bin's three, then each box's x, y and yaw in turn -- so the
RNG stream is identical and every episode is byte-identical.  Checked by hashing the
drawn layouts over six seeds in a worktree at the previous commit against the
working tree: same digest, `b67959a1...`.  That is a two-minute check against an
80-minute relay run, and it is a stronger statement than 3/3 -- it says no seed
changed, not just the three that were run.

**The derived queue gap turned out to be wrong, and the plan had it backwards.**
The plan said a lane at the derived pitch "exhausts its redraws".  It does not: the
redraw loop finds a layout, so the scene *looks* like it works.  What is actually
wrong is that the derived pitch is not a legal *layout* spacing at all --
`lane_pitch = box_extent_y + min_box_clearance`, so two neighbours laid out at it
have zero gap between their outer extents, and the committed three-box lane measures
13 mm against the 25 mm the scene draws within.  The scene would then only ever draw
layouts where the queue *spread*, which is a loss of variation rather than a loss of
episodes: measured over 40 seeds at the pitch, the queue's offsets are non-increasing
on 40 of 40, against 18 of 40 at the relay's gap.

So `queue_gap_m` has to exceed `lane_pitch`, and a config that violates it is now
refused at **load** with the number in the message.  The distinction that was
missing is between three quantities: the pitch is what the line compacts to *after*
a shove, the gap is what the lane is *laid out* at, and the ranges need room on top
of the gap.  Two of the three are now derived or checked; deriving the gap from the
ranges is still Phase 5's job.

One more thing this phase settled: `worst_nominal_box_gap_m` had to be measured
between **outer** extents (half size plus wall thickness), matching
`_boxes_are_clear`, because a check using the inner half size passes exactly the
layout the draw check refuses.  And the check only applies when the scene actually
draws -- `configs/cookie_two_box.yaml` has its two boxes 12 mm apart against a 15 mm
clearance it never draws against, and nothing is wrong with it.

**Phase 3 -- variable `per_grasp` in the fill.**  *Gate:* `per_grasp=5` is
step-identical to today (the anchor again, this time in the simulator); 3 seeds
each at `per_grasp` 1, 2 and 3 all reach ten in the box.

**Half met, and the half that failed is the more useful result.**  The anchor is
exact and the plan is right for every size; the *grasp* only works at five, so the
config now refuses anything below it rather than collecting at a yield nobody has
measured.

*Done, and the anchor is exact.*  `batch_expert_per_grasp` is a config field, the
env derives `batch_plan` from its own slot lattice and that field (so the expert
reads one source of truth rather than deriving its own), and the fill takes each
batch's size from the plan rather than from `per_grasp` -- a capacity that does not
divide by the column count gives a *short* batch, and grasping the full size for a
short batch would leave a Cookie with nowhere to go.

The anchor is checked by hashing the whole action stream of an episode against the
previous commit: 498 steps, ten Cookies placed, **identical digest**
(`ead5b159...`).  The phase trace alone would not have been enough, and getting
there found two real differences that a phase trace hides:

* **`_place_pose` must not average the x.**  Every slot of a column shares its x, so
  reading the batch's first slot is exact and is the same number the old per-column
  formula took out of `np.unique`.  A mean of five identical values is one ulp off,
  and that one ulp propagates through the IK into *every* action of the episode.
* **The source row pitch is not the layout's nominal pitch.**  The configs write row
  positions to seven decimals, so the differences between the written values are not
  all equal -- they alternate 0.0088334 and 0.0088333 -- and their median differs
  from `rows[1] - rows[0]` in the eighth significant figure.  That is 5 nm, but it is
  a real difference rather than rounding, and the expert keeps its old expression.
  A one-Cookie batch has no spacing to measure (`np.median` of an empty difference is
  NaN), so it falls back to the layout's pitch.

Both are the same lesson as Phase 2's queue gap in a different register: the
"obviously equivalent" rewrite of an expression is not always equivalent, and the
only way to know is to measure the thing rather than the shape of it.

`_place_pose` now aims at the batch's own slots rather than at the column's centre.
For the shipped plan the two are the same point, which is *why* the anchor holds
here; for a plan with two batches in a column they differ, and using the centre for
both would drop the second batch in the wrong place.

The same-column expert's first-batch-versus-the-rest tuning is now named
(`_is_later_batch`, in the base class) rather than spelled `batch_index == 0` in one
place and `batch_index == 1` in four.  For the shipped two-batch plan those are the
same two statements, so the anchor is unaffected -- and the docstring records that
generalising it to *every* later batch of a longer plan is untested, because the
anchor only pins the two-batch case.

**What failed: the grasp does not scale down.**  A latent bug was found and fixed on
the way -- the CLOSE phase's narrowest opening was a flat `0.28` of the stroke, which
is 23.8 mm of jaw gap against a three-Cookie batch 24.0 mm wide, so the phase *opened*
the jaws from 26.6 mm to 30.2 mm and read 0.00 N for its whole budget.  That floor is
now scaled by the batch size, exactly reproducing the shipped value at five.

But the grasp still does not form below five, and the remaining cause is not
plumbing.  Measured, three seeds each on the same-column scene:

```
  per_grasp   placed   steps   outcome
         5       10     483-543   ok
         3        0     ~475      CLOSE timed out; pad forces <= 0.06 N
         2        0     ~477      CLOSE timed out; pad forces <= 0.06 N
         1        0     ~477      CLOSE timed out; pad forces 0.00 N
```

and the geometry at the moment CLOSE starts is the *same* shape in both cases -- the
pads sit 2.56 mm wider than the batch's outer extent, at 44.23 mm against 41.67 mm for
five and 26.56 mm against 24.00 mm for three.  What differs is how far they have to
close to build force: five reaches 1.27 N after 11.68 mm of pad travel, three reaches
only 3.35 mm before its floor.  So the force a squeeze builds is a property of how
many bevels are in the line, and the floor -- now proportional to the batch -- is
still not the right shape for it.

**The bound is therefore declared at five**, with the measurement in the message:
`min_verified_batch_expert_per_grasp`.  The plan derives correctly for every size
from 1 to 10 (a test partitions all of them), so the two claims are kept apart: the
plan is a parameter, the grasp is a measurement.  Opening the range up means
measuring what the pads have to close to, not relaxing the check.

**Phase 4 -- the N-box relay.**  `TwoBoxBatchExpert` becomes `RelayBatchExpert`
over the lane, with the line-shove push and the carry's timeout scaled by the
slide distance.  *Gate:* N=2 unchanged at 3/3; N=3 measured, and the parked line's
x drift and yaw recorded (they are not currently graded, so they have to be
*measured* to know whether they can be ignored).

**Phase 5 -- derived randomization and the gate.**  *Gate:* for every config in
the test matrix, 500 drawn layouts pass reach + clearance + lane checks; and a
deliberately over-wide range is refused with a reason naming the box and the
distance.

**Phase 6 -- one profile for every layout.**  Re-tune the carry for
`arm_actuator_damping = 1.0`, then flip the relay to `fast`.  *Gate:* relay 3/3 at
about 2053 steps; single-box numbers unchanged.

**Phase 7 -- the matrix, then the docs.**  Run the table below, write the results
into the README's speed section and a new "configurable scenes" section, and
update `_BATCH_CONTRACT` to a derived string.

## The test matrix

Exhaustive is out: `per_grasp` x `boxes` x `randomize` x `capacity` x
`source_cookies` is hundreds of scenes at tens of minutes each.  What is in is
every *mechanism* and every *boundary*, at 3 seeds each:

| axis | representative | boundary | why that boundary |
|---|---|---|---|
| `per_grasp` | 2, 5 | 1 | 1 is a single-Cookie grasp: the pads close on one Cookie, the batch chain is one node, and the insertion overlap (3.8 mm) is larger than the Cookie's own bevel |
| `boxes` | 1, 2 | 4 | 4 is the source-supply bound (2 usable columns x 20 rows = 40 graspable Cookies, against 4 x 10 needed); 3 is the first layout where the line-shove push happens twice |
| `randomize` | off | on (full ranges) | off is byte-identical to the pre-feature behaviour; on is where the gate lives |
| `box_capacity` | 10 | 20 | 20 forces `rows=10`, which is also `per_grasp=10` -- the jaw-travel boundary |
| `source_cookies` | 80 | `N*C` exactly | an exactly-empty source is the relay's `source == n - N*C` check at its edge |

That is 5 axes with 2-3 points each, but the matrix is not the cross product:
the boundary runs are the *same scene* as a representative run with one parameter
changed, so the table is ~12 runs rather than ~40.  At 3 seeds and 8 shards this
is a couple of hours, not a day.

Plus the per-phase unit gates above, which are all sub-minute.

## Risks and rejected alternatives

**Rejected: fill N boxes in place, no relay.**  The station band is ~80 mm wide in
x and ~130 mm in y, and a box is 126 x 62 mm, so exactly two boxes fit side by
side in y -- but only two.  It would be a *simpler* N=2 (no push, no carry, no
lane), and it is worth knowing that; it does not generalise, so it is not the
mechanism.

**Rejected: push each parked box to its own slot.**  Measured impossible: +y is
capped at +0.100 for the pads, and the second parked box needs them at +0.176.
The line-shove is the only push that fits the envelope.

**Rejected: move parked boxes sideways into a second lane in x.**  The carry's own
measurement from the last session: a pinch moves the box *with* the pads, so a
sideways command is a sideways displacement whose torque is what the next fill's
pre-check reads.  Three steering attempts all traded the two errors against each
other.  A second lane in x needs a new transfer capability, not a parameter.

**Risk: the parked line's yaw.**  Nothing grades the parked boxes' yaw today, and
the line-shove is a collision between free bodies.  Phase 4 measures it; if it
turns out to be large, the parked boxes need a yaw bound in the spec and possibly
a nudge, which is the same bounded re-nudge loop the carry still needs.

**Risk: `source_offset` moves the source bin into the fill's workspace.**  The
source bin is a mocap body with infinite mass, so a collision shoves the target
box rather than being resolved.  `_boxes_are_clear` covers the pair, but the
clearance budget shrinks as N grows and the Monte-Carlo gate is what proves it
still holds at the range extremes.

**Risk: the derivation looks right and is not.**  This is the risk the anchor test
exists for, and it is not hypothetical -- the relay's randomization was hand-copied
from the single-box scene last time and cost two of three seeds.  Every derived
number in this document is either reproduced from a shipped config or measured in
this session; anything else is marked as a parameter with a default, not a formula.

## Out of scope for this branch

* **`boxes >= 5`.**  Bounded by source *supply*, not geometry: 5 boxes need 50
  graspable Cookies and the shipped 4 x 20 layout offers 40 (two of its four
  columns are out of reach).  Raising it means a source layout with more usable
  rows -- a 2-column, 30-row grid supplies 60 and is the obvious next step -- not a
  longer lane, since the lane and the source bin's +y offset both have room to
  spare.  The source grid's own y reach still has to be measured before a taller
  layout is advertised, because the reach map was taken on a 20-row column.
* **A second target lattice orientation** (a box whose columns run along x rather
  than y).  It would raise the `per_grasp` ceiling past `rows` but changes the
  insertion geometry in the direction the arm has least reach (+30 mm).
* **Per-Cookie jitter in the batch scenes.**  Still off, still for the same
  measured reason: a 2 mm jitter is wider than the 2.5 mm gaps the insertion
  enters.
* **Learned-policy work.**  The spec changes what can be collected, not what is
  trained on it; the datasets stay comparable because the prompt is unchanged.
