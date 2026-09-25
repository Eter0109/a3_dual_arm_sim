"""The two validated tunings of the batch expert, as data rather than literals.

Both are measured; neither is a guess.  They differ in *how* the arm is driven
as well as how fast, which is why they are a pair of profiles and not a speed
multiplier:

``FAST``
    The rate table the README's speed section documents.  The servo plans from the
    pose the arm was *commanded* to hold, so it needs no anti-windup clamp and can
    hold a high rate safely; the arrival is made gentle by a terminal ramp and
    confirmed by a joint-velocity test.  Requires actuator damping
    (``SimConfig.arm_actuator_damping``), because without it the arrival test
    never passes -- the arm rings indefinitely and the plan has always arrived.
    Measured: the single-box scenes complete in 415-461 steps, 5/5.

``BASELINE``
    The tuning this expert shipped with, kept because the two-box relay is
    calibrated around it.  The servo plans from the *measured* pose and clamps how
    far the plan may lead it; the transits are joint-space trajectories rather
    than Cartesian servos.  Slower, but its fill leaves the Cookies in the poses
    the relay's push and carry controllers were measured against.  Measured: the
    relay completes with this profile and, with ``FAST``, does not.

A scene picks one with ``SimConfig.batch_expert_profile``.  Keeping the baseline
reachable is the point: the relay's failure under ``FAST`` is a calibration gap
in the relay, not a defect in the fast profile, and this way fixing that gap
starts by flipping one line instead of by recovering an implementation from
history.

**What that gap turned out to be** (Phase 6 of ``docs/configurable-scenes.md``),
because the two obvious guesses were both wrong.  It is not the fill's landing
pose and it is not the carry: the relay's *push* is what fails, and it fails
under actuator damping whatever the profile -- ``BASELINE`` + damping fails the
same way ``FAST`` + damping does, at "right box IK unreachable".

The measured mechanism is not the yaw compensation's *gain*.  Pushing the box
slides it sideways -- about 10 mm with the shipped dynamics and 20 to 25 mm with
a damped, rigid arm, and the same 20 mm arrives with the compensation turned off
entirely, so it is the push's physics rather than its steering that moves the
box.  The compensation is clipped to +/-10 mm of *where the push started*, so
that drift saturates it: measured directly, it sits on the bound from a yaw of
0.94 degrees, where the yaw term contributes 1.3 mm of the 10 -- it is following
the box's translation, not its rotation.  Which is why gains from 0.02 to 0.08
produce bit-identical IK residuals on the failing push, and why the clip's width
is the only lever that does anything.

Widening it to 20 mm is enough to get the push *and* the carry through, and that
is where it stops being a one-line change: ``BASELINE`` + damping with the wider
clip then fails the push on its own 25 mm sideways guard (the box reaches 25.3 mm),
and ``FAST`` + damping runs to the last fill and dies in ``LIFT`` with both pads
holding 3 N.  So the push needs a contact strategy that keeps the box straight
rather than a bound that follows it, and the carry's placement needs re-measuring
against the fill's 2 mm pre-check before this scene can run fast at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchProfile:
    """One tuning of every rate, threshold and geometry the expert chooses."""

    #: Plan from the commanded pose (needs no anti-windup clamp, allows a high
    #: rate) rather than from the measured pose (needs both).
    commanded_reference: bool
    #: Whether transits are joint-space trajectories seeded from the measured
    #: pose.  The baseline's, kept because its first motion stays on the nearby
    #: mirrored IK branch; the fast profile servos in Cartesian space instead.
    joint_transit: bool

    transit_m_per_step: float
    descend_far_m_per_step: float
    descend_near_m_per_step: float
    close_free_m_per_step: float
    close_contact_m_per_step: float
    lift_near_m_per_step: float
    lift_far_m_per_step: float
    place_free_m_per_step: float
    place_far_m_per_step: float
    place_near_m_per_step: float
    #: Rate of the retreat, and of opening the jaws.  Both are the tool *leaving* a
    #: batch it has just placed inside a box, so both are the two rates to slow
    #: first if a scene finds its box being dragged: measured on the two-box relay
    #: (which does not use this profile), a 0.015 open with a 0.0050 retreat pulled
    #: the box 7.8 mm sideways as the pads came off the batch, and 0.004 with
    #: 0.0020 held it to 2.5 mm.  The single-box scenes are verified 5 of 5 at the
    #: faster values below, so those are what this profile ships.
    open_m_per_step: float
    retract_m_per_step: float
    #: Rate of the ALIGN stage's base push, which meets the Cookie edges side-on.
    push_descend_m_per_step: float
    push_sweep_m_per_step: float
    push_retract_m_per_step: float
    #: Rate of a move that only breaks contact and returns home.
    hold_m_per_step: float

    #: How far above the grasp the tool travels, and above the slot it carries.
    grasp_clearance_m: float
    transport_clearance_m: float
    #: Distance inside which a two-speed move drops to its slow rate.
    near_m: float

    #: How far a placement pose may miss before the layout counts as out of
    #: reach.  Measured, not derived: the pre-check refuses layouts the run
    #: cannot finish, and both edges were found by trying them.  Too loose (the
    #: IK's own 4 mm) admits a pose that then times out in
    #: ``DESCEND_TO_PLACE``; too tight (the placement servo's 1.5 mm arrival)
    #: refuses a pose the run completes, because the measured tool position
    #: depends on gravity and contact as well as on the IK solution.
    reach_position_m: float
    reach_angle_rad: float

    #: Steps the pre-close opening test must hold, and the tolerance it is held to.
    preclose_stable_steps: int
    preclose_tolerance: float
    #: Steps the closed grasp must hold before lifting, and the descent-step
    #: count after which a blocked insertion may be accepted at its reached depth.
    close_stable_steps: int
    descend_jam_window: int
    #: Steps the release test must hold: the last batch uses the scene's own
    #: success window, the first only has to be released.
    release_stable_first: int


FAST = BatchProfile(
    commanded_reference=True,
    joint_transit=False,
    transit_m_per_step=0.015,
    descend_far_m_per_step=0.0035,
    descend_near_m_per_step=0.0010,
    close_free_m_per_step=0.0060,
    close_contact_m_per_step=0.0030,
    lift_near_m_per_step=0.0018,
    lift_far_m_per_step=0.0035,
    place_free_m_per_step=0.0050,
    place_far_m_per_step=0.0030,
    place_near_m_per_step=0.0012,
    open_m_per_step=0.015,
    retract_m_per_step=0.0050,
    push_descend_m_per_step=0.0018,
    push_sweep_m_per_step=0.0012,
    push_retract_m_per_step=0.0030,
    hold_m_per_step=0.001,
    grasp_clearance_m=0.082,
    transport_clearance_m=0.085,
    near_m=0.015,
    reach_position_m=0.002,
    reach_angle_rad=0.035,
    preclose_stable_steps=2,
    preclose_tolerance=0.008,
    close_stable_steps=3,
    descend_jam_window=100,
    release_stable_first=6,
)

BASELINE = BatchProfile(
    commanded_reference=False,
    joint_transit=True,
    transit_m_per_step=0.015,
    descend_far_m_per_step=0.0006,
    descend_near_m_per_step=0.0006,
    close_free_m_per_step=0.0015,
    close_contact_m_per_step=0.0015,
    lift_near_m_per_step=0.0008,
    lift_far_m_per_step=0.0008,
    place_free_m_per_step=0.0010,
    place_far_m_per_step=0.0005,
    place_near_m_per_step=0.0005,
    open_m_per_step=0.003,
    retract_m_per_step=0.0016,
    push_descend_m_per_step=0.0006,
    push_sweep_m_per_step=0.0003,
    push_retract_m_per_step=0.0010,
    hold_m_per_step=0.0010,
    grasp_clearance_m=0.105,
    transport_clearance_m=0.085,
    near_m=0.015,
    reach_position_m=0.002,
    reach_angle_rad=0.035,
    preclose_stable_steps=3,
    preclose_tolerance=0.006,
    close_stable_steps=8,
    descend_jam_window=185,
    release_stable_first=20,
)

PROFILES = {"fast": FAST, "baseline": BASELINE}
