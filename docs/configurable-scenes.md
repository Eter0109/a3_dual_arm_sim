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
  -350     o       o       o       o       o       o
```

Two facts come out of this and they are the whole reason the lane is shaped the
way it is:

* **+y is capped at about +0.100.**  Everything from y = +0.125 up is refused at
  every x.  The push that parks a filled box works today because the pads only
  travel to the box's rear wall -- the box's centre reaches +0.120 while the pads
  stop at +0.089.  One box, and no more: parking a second box at +0.207 would need
  the pads at +0.176.
* **-y is open to at least -0.520** (checked past the sweep's edge; the table runs
  to -0.750).  The queue has room for six boxes at the derived pitch, so it is not
  what bounds N.

**Consequence.**  The parked boxes cannot be pushed one at a time to their own
slots.  The mechanism that does work is the one a person would use: the pads push
the box at the station into the line already parked, and **the line advances by
one pitch per fill**.  Pad travel is unchanged, and the parked boxes are never
touched directly.  That is the derived rule:

```
lane_pitch      = box_extent_y + min_box_clearance
push_destination = station_y + lane_pitch        # the station box's target
parked_i         = station_y + lane_pitch * (k - i)   # after filling box k
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
y pitch between slots = cookie_thickness                     0.0063333  (0.019/3)
x spacing between columns = cookie_width + column_gap        0.056      (0.050 + 0.006)
inner_half_x = x_spacing/2 + cookie_half_x + wall_margin     0.057      (0.028 + 0.025 + 0.004)
inner_half_y = (rows-1)/2 * y_pitch + cookie_half_y + pad_clearance
                                                             0.025      (0.012667 + 0.003167 + 0.009167)
```

The y pitch is the *nominal* thickness, not the source pitch: the source lays the
Cookies out with 0.4 mm of clearance so they do not interpenetrate at reset, while
the target slots are where the batch comes to rest after the jaws compressed it.
The y margin is the pad clearance -- the jaws descend into the box outside the
batch, so the box has to be wider than the batch by the pads' own thickness plus
room to move.

`pad_clearance` is the one number here that is not arithmetic.  It is bounded
below by the pad geometry (0.0032 m) and its shipped value is 0.0092; the spec
carries it as a parameter with that default, and the anchor test is what says
whether a change is allowed.

The *shape* is then a choice of `columns`, with `rows = ceil(box_capacity /
columns)`.  Two constraints pick it:

* `per_grasp <= rows` -- a batch is placed into one run of slots inside a column,
  so it cannot be longer than the column.
* `per_grasp * source_pitch_y + pad_thickness <= 0.085` -- the jaw travel.  This
  caps `per_grasp` at 11 regardless of the box.
* growing `columns` grows the box in x, and the station has +30 mm of x reach at
  the shipped pose, so **`columns = 2` is the practical ceiling** unless the
  station also moves -x.

So for C=10, `columns=2`, `rows=5`, and `per_grasp` can be 1..5.  To grasp more
than 5 the box has to be a single column of 10, which is a different box shape --
worth offering, and worth testing, but not the default.

### 2. The batch plan, from capacity and grasp size

```
sizes = [per_grasp] * (box_capacity // per_grasp) + ([remainder] if remainder else [])
slots_of(batch b) = the next `sizes[b]` free slots, column-major
```

With C=10, K=5 this is `[5, 5]` and batch 0 takes column 0's five slots -- exactly
today's behaviour, which is why the anchor holds.  With C=10, K=3 it is
`[3, 3, 3, 1]`: column 0 takes two batches (rows 0-2, rows 3-4) and column 1 takes
two more, the last one a single Cookie.

This is also where "how many Cookies to take next" is decided, and it is a running
counter rather than a fixed list: `min(per_grasp, remaining_capacity_of_this_box)`.
Stated that way the same code resumes a partially filled box, which is what the
user's later milestone (a box that arrives with Cookies already in it) needs.

`_place_pose` becomes `_place_pose(clearance, slots)`, and
`_validate_target_workspace` iterates the batch plan instead of `range(2)`.

### 3. The source layout, from source count and grasp size

```
usable_columns = the columns the arm can actually reach   (measured: 2 of 4)
rows           = per_grasp * max(1, round(ceil(source_cookies / columns) / per_grasp))
positions      = the `columns`-column lattice at source_pitch, centred on the nominal centre
source_bin_half = grid_extent/2 + bin_margin
require N * box_capacity <= usable_columns * rows
```

With S=80, K=5, columns=4 this gives rows = 20 and reproduces the shipped layout
exactly.  The rounding is not cosmetic: `_candidate_batches` requires `per_grasp`
consecutive available Cookies in a column, so a column whose length is not a
multiple of `per_grasp` leaves a tail the expert skips -- and a skipped tail breaks
the relay's `source == n - N*C` check.  Rounding the column up to a whole number
of batches is what keeps the source exactly divisible.

The last line is the constraint that sets `boxes <= 4`, and it is worth being
explicit about why it is a *supply* constraint and not a geometry one.  The far
columns are unreachable, so they contribute nothing; whether the layout keeps them
for the look of a full bin is a presentation choice, and the spec should carry
`usable_columns` as a measured parameter rather than infer it from `columns` --
inferring it is exactly how a spec would end up advertising N = 6 and refusing
half of them at collection time.

### 4. The lane, from the box count

```
lane_pitch   = box_extent_y + min_box_clearance
station      = the nominal station, clamped into the fillable band
queue_i      = station_y - lane_pitch * i        for i in 1..N-1
source_offset = max(0, source_front_clearance_need(N))
```

`source_offset` is the formula from the measurement above; the spec also derives
the source bin's *yaw* range from zero, because a yawed bin changes the pick pose.

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
src/a3_dual_arm_sim/scene_spec.py     NEW  pure derivation, no MuJoCo import
src/a3_dual_arm_sim/batch_plan.py     NEW  the batch plan and slot grouping
src/a3_dual_arm_sim/scene_spec_validate.py  NEW  the Monte-Carlo gate
src/a3_dual_arm_sim/model.py               N target boxes instead of target+spare
src/a3_dual_arm_sim/cookie_transfer.py     N boxes in randomization and reset
src/a3_dual_arm_sim/batch_expert.py        batch plan instead of the literal 5
src/a3_dual_arm_sim/same_column_batch_expert.py   same
src/a3_dual_arm_sim/relay_batch_expert.py  NEW  TwoBoxBatchExpert generalised to N
src/a3_dual_arm_sim/tasks.py               contract strings derived from the spec
src/a3_dual_arm_sim/collection.py          one spec-driven scene builder
src/a3_dual_arm_sim/cli.py                 --scene-spec key=value,...
configs/generated/*.yaml                   the specs the shipped scenes decode to
```

`scene_spec.py` must not import MuJoCo, for the same reason `tasks.py` does not:
`training` imports on hosts where the simulator wheels will not load.

The shipped YAML configs stay.  They become the *frozen artifacts* of two specs,
and a test asserts `spec.derive_config()` reproduces them -- so the configs keep
working for anyone who has them, and the derivation cannot silently drift.

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

**Phase 2 -- N boxes in the model and the env.**  *Gate:* the existing two-box
tests pass unchanged, the relay is still 3/3, and a 3-box scene builds, resets and
passes the Monte-Carlo gate (no fill attempted yet).

**Phase 3 -- variable `per_grasp` in the fill.**  *Gate:* `per_grasp=5` is
step-identical to today (the anchor again, this time in the simulator); 3 seeds
each at `per_grasp` 1, 2 and 3 all reach ten in the box.

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
