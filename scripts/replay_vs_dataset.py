#!/usr/bin/env python
"""Teacher-forced replay against a recorded episode: the dynamics, on both sides of one step.

Why this exists
---------------
`check_invariants.py` covers four of the seven channels: state convention, image orientation,
the gripper round trip, and how much of a joint step arrives.  It cannot see the **dynamics**
-- the contact model, the damping, the integrator, the timestep -- and the ablation says that
channel is worth 10-15 points and is invisible without a test.  This is that test.

The idea is the one thing a dataset gives that nothing else does: **what the world did next**.
At every recorded frame the dataset holds a state, the action the expert issued from it, and
the state that resulted.  So for each frame:

    load state_t   ->   apply action_t for one control step   ->   compare to state_{t+1}

Teacher forcing, not open loop: resetting to the recorded state at every step isolates
**per-step fidelity** from error accumulation, so a small difference in the dynamics cannot
hide a large difference in the wiring.  Both are reported, and they answer different questions:

* **Teacher-forced residual** -- how well one control step reproduces the recorded one.  Tight,
  and it is the channel that catches "the arm delivers half the command" or "the contact model
  changed" before either has a chance to compound.
* **Open-loop divergence** -- how long a trajectory stays close when it is *not* corrected.
  This is the honest measure of whether the scene behaves like the one that was recorded, and
  it should be read as a curve, not as pass/fail.

Units
-----
The arm's fourteen joints are radians; the two grippers are openings in `[0, 1]` (A3's own
convention -- the dataset is the specification for this scene, see
`docs/scene-transition-and-bug-triage.md`).  The delivery ratio is reported per step and as a
distribution, because a mean hides a bimodal failure.

Usage::

    MUJOCO_GL=egl python scripts/replay_vs_dataset.py \
        --scene a3_cookie_single_piece --dataset /root/autodl-tmp/datasets/a3_cookie_one \
        --episode 0 --frames 200 --output /tmp/replay.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

#: The dataset's state layout is `[left arm 0..6, left gripper 7, right arm 8..14, right
#: gripper 15]`; these two selections keep the arm's radians and the grippers' openings in
#: separate statistics, because a single `max` over both mixes units.
ARM_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
GRIPPER_INDICES = [7, 15]


def load_invariants_module():
    """`check_invariants` by path: it is a script, not a package member, and its dataset
    readers are the ones this file must use rather than re-derive -- the lesson of the
    `top_site` fallback is that two restatements of one measurement will drift."""

    spec = importlib.util.spec_from_file_location(
        "check_invariants", PROJECT / "scripts" / "check_invariants.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_invariants"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="a3_cookie_single_piece")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_cookie_one"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--first", type=int, default=0, help="first frame of the episode")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1,
                        help="teacher-force every Nth frame; the open-loop tail always starts "
                             "at --first")
    parser.add_argument("--open-loop", type=int, default=50,
                        help="control steps to run uncorrected from --first")
    parser.add_argument("--rest-start", action="store_true",
                        help="start every teacher-forced step from zero velocity, i.e. keep "
                             "the position and drop half of the initial condition")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


class Scene:
    """The environment plus the index maps every test here needs."""

    def __init__(self, name: str) -> None:
        import mujoco

        from a3_dual_arm_sim import props
        from a3_dual_arm_sim.collection import scene_by_name
        from a3_dual_arm_sim.contracts import ARM_JOINTS

        self.mujoco = mujoco
        self.env = scene_by_name(name).build_env(0.0, 0.0)
        self.env.reset(seed=0, options=None)
        self.model, self.data = self.env.model, self.env.data

        # **Whether the arm is touching a prop decides whether the step is testable at all.**
        # The dataset stores the robot's sixteen numbers and nothing about the props, so a
        # step in free space is fully determined by `(state, action)` and a step in contact is
        # not: the same command against a cookie 5 mm to the left is a different trajectory.
        # Without this split a replay reports contact-phase mismatch as a dynamics defect.
        self.prop_geoms: set[int] = set()
        for body in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if body_name.startswith("libero_") or "cookie" in body_name:
                self.prop_geoms.update(props.prop_geoms(self.model, body))
        try:
            self.arm_bodies = props.robot_bodies(self.model)
        except ValueError:
            self.arm_bodies = set()
        if not self.prop_geoms or not self.arm_bodies:
            raise SystemExit(
                f"{name}: the contact split needs the props and the arm; found "
                f"{len(self.prop_geoms)} prop geoms and {len(self.arm_bodies)} arm bodies"
            )

        address = {}
        for joint in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_name:
                address[joint_name] = int(self.model.jnt_qposadr[joint])
        self.arm = [name for name in ARM_JOINTS if name in address]
        if len(self.arm) != 14:
            raise SystemExit(f"{name} drives {len(self.arm)} arm joints; expected 14")
        self.qpos_of = [address[name] for name in self.arm]
        dof_address = {}
        for joint in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_name:
                dof_address[joint_name] = int(self.model.jnt_dofadr[joint])
        self.dof_of = [dof_address[name] for name in self.arm]

    def reset_scene(self, seed: int) -> None:
        """Put the props where a reset puts them, before a measurement phase.

        **A replay has to start from a reset scene, and that is a finding rather than a
        habit.**  The dataset records the robot's sixteen numbers and nothing about the props,
        so their positions are not recoverable; what a replay can do is put them where this
        scene's own reset puts them and then declare the consequences (see the contact split).
        Leaving them wherever the last phase left them is not neutral: measured on
        `a3_cookie_one`, the same open loop scores **0.0 rad** divergence from a freshly reset
        scene, 0.0096 after an earlier phase, and 0.1524 after a teacher-forced sweep -- a
        factor of fifteen in a channel (free space) where the props are supposed to be
        irrelevant.  The variation is the harness, not the scene.
        """

        self.env.reset(seed=seed, options=None)

    def load(self, state: np.ndarray, qvel: np.ndarray | None = None) -> None:
        """Write a recorded state into the simulator, exactly.

        The arm's fourteen values are joint angles in the dataset's order.  The two grippers
        are openings, not joint angles, so they go through the same conversion the environment
        uses -- the one the state convention check verifies round-trips to 0.

        `qvel` is the fourteen joint velocities.  **A teacher-forced replay has to supply it,
        and it has to be the recorded one.**  The obvious substitute --
        `(state_{t+1} - state_t) / dt` -- is the *average* over the interval rather than the
        instantaneous value, and used as an initial velocity it front-loads a whole step of
        motion: measured on `a3_cookie_one`, it diverges by 0.036 rad and then decays, where
        starting from rest reproduces the recording to 0.0 rad.  That result was read at the
        time as "rest is correct", and it is not -- rest is only better than a *wrong*
        velocity.  The dataset records `observation.velocity`, and with it the residual falls
        to the float-precision floor; without it, 0.0033 rad, which is three quarters of a
        recorded step and points at the dynamics instead of at this function.
        **The robot is teleported and the props are left exactly where they are.**  That
        asymmetry is forced: the dataset records the robot's sixteen numbers and says nothing
        about the props, so a replay can reproduce the arm and has to let the props follow from
        its own physics.  Two attempts to have it otherwise were measured and both were worse:

        - `mj_resetData` before writing the arm - this was the original form.  It resets the
          props onto the model's default pose, the arm arrives at a recorded pose that assumes
          the props have been *moved*, and in `a3_cookie_batch` the resulting interpenetration
          is ejected at 97 rad/s.  The environment fires its own safety stop at frame 98 and
          then holds position for the rest of the sweep, which the tool reported as
          `moved/expected 0.000` and a residual of 4.45 recorded steps -- three times the
          largest real signal it has ever measured here, entirely manufactured by the harness.
        - restoring the props to their post-reset pose every frame - the arm dives into a
          stationary cookie on the frames where the recording is closing the gripper, and the
          same stop fires on the same frame.

        So nothing in the data, and nothing outside the data either: the arm is written, the
        rest of the world is simulated.  The cost is that the contact phases are a replay of a
        grasp that our own physics has to re-perform, which is why they are reported separately
        and not judged -- they cannot be reproduced by construction.
        """

        for index, address in enumerate(self.qpos_of):
            # Dataset layout: [left 0..6, left grip, right 0..6, right grip].
            self.data.qpos[address] = state[index if index < 7 else index + 1]
        for side, index in (("L", 7), ("R", 15)):
            opening = float(state[index])
            fraction = self.env._finger_travel_fraction(side, opening)  # noqa: SLF001
            travel = self.env._finger_travel[side]  # noqa: SLF001
            for sign, joint in zip(self.env._finger_sign[side],  # noqa: SLF001
                                   self.env._finger_joints[side]):  # noqa: SLF001
                self.data.qpos[self.model.jnt_qposadr[joint]] = sign * fraction * travel
        if qvel is not None:
            for index, address in enumerate(self.dof_of):
                self.data.qvel[address] = qvel[index]
        self.mujoco.mj_forward(self.model, self.data)

    def state(self) -> np.ndarray:
        """What the environment reports for its own observation, in the dataset's layout."""

        import numpy as np

        observation = self.env._observation()  # noqa: SLF001
        return np.asarray(observation["observation.state"], dtype=np.float64)

    def touching(self) -> bool:
        from a3_dual_arm_sim.props import touching

        return touching(self.model, self.data, self.prop_geoms, self.arm_bodies)

    def qpos_now(self) -> np.ndarray:
        return np.asarray([self.data.qpos[address] for address in self.qpos_of])


def command(scene: Scene, action: np.ndarray) -> None:
    """Apply one recorded action for exactly one control step.

    The dataset's action is the state vector's own layout with the gripper holding its
    commanded opening, so it goes straight in: no normalisation, because the dataset stores
    physical units (`docs/scene-transition-and-bug-triage.md` §2.3), and no clipping, because
    a clipped command would hide an out-of-range expert.
    """

    scene.env.step(np.asarray(action, dtype=np.float64))


def main() -> int:
    args = parse_args()
    ci = load_invariants_module()

    # The window is clipped to the episode here and printed, rather than clipped inside the
    # reader: reading past the last frame of an episode yields the *next* episode's frames, and
    # the only channel that reports it is the gripper (see `read_episode_frames`).  A tool that
    # quietly returns fewer frames than asked for would hide the boundary instead.
    start, stop = ci.episode_bounds(args.dataset, args.episode)
    length = stop - start
    first = min(args.first, max(length - 1, 0))
    count = min(args.frames, 10_000, length - first)
    states, actions, velocities = ci.read_episode_frames(args.dataset, args.episode, first,
                                                        count)
    print(f"episode {args.episode}: {length} frames; replaying {first}..{first + count - 1}"
          + (f"   (clipped from --first {args.first} --frames {args.frames})"
             if (first, count) != (args.first, args.frames) else ""))
    args.first = first
    scene = Scene(args.scene)
    report: dict = {
        "scene": args.scene,
        "dataset": str(args.dataset),
        "episode": args.episode,
        "first_frame": args.first,
        "frames": int(states.shape[0]),
        "width": int(states.shape[1]),
    }

    # ---- teacher-forced, every `stride`-th frame ----
    residuals: list[float] = []
    gripper_errors: list[float] = []
    ratios: list[float] = []
    per_joint: list[np.ndarray] = []
    steps = range(0, len(states) - 1, max(args.stride, 1))
    contact_flags: list[bool] = []
    step_sizes: list[float] = []
    stopped: dict | None = None
    scene.reset_scene(args.episode)
    for index in steps:
        # **The initial condition is `(position, velocity)`, and the dataset records both.**
        # Keeping the position and dropping the velocity is the tempting simplification and it
        # is wrong by a measurable amount -- see `read_episode_frames`.  `--rest-start` drops
        # it on purpose, to keep the size of that mistake on record rather than in a comment.
        start_velocity = None if args.rest_start else velocities[index][ARM_INDICES]
        scene.load(states[index], qvel=start_velocity)
        in_contact = scene.touching()
        contact_flags.append(in_contact)
        before = scene.qpos_now()
        command(scene, actions[index])
        # **An arm that has stopped is not an arm that did not move.**  The environment holds
        # position after its own safety stop, so every step from there on produces a zero
        # delta against a moving recording, which reads as a wrong-dynamics verdict of exactly
        # the size of the recorded motion.  It is a measurement failure and it has to say so --
        # and the frame that fired it is dropped, because a step that trips a velocity limit is
        # not a step whose residual means anything either.
        if scene.env._safety_stop:  # noqa: SLF001
            stopped = {"phase": "teacher-forced", "frame": index,
                       "reason": scene.env._safety_reason}  # noqa: SLF001
            contact_flags.pop()
            break
        after = scene.qpos_now()
        expected = states[index + 1]
        #: The recorded **absolute** joint angles for the next frame.  Comparing `after` to
        #: this is the test; comparing a delta to it is not, and the first version of this
        #: file made exactly that mistake -- it reported a 1.56 rad residual on a scene whose
        #: open-loop replay tracked the recording to 0.0005 rad, which is the contradiction
        #: that gave it away.
        target = np.asarray([expected[i if i < 7 else i + 1] for i in range(14)])
        residual = after - target
        per_joint.append(residual)
        residuals.append(float(np.abs(residual).max()))

        # How much of the recorded next step the environment produced: the ratio the
        # dataset's own dynamics imply, step by step.  Delta over delta, so it is a property
        # of the motion rather than of the offset.
        recorded_delta = target - before
        denominator = float(np.abs(recorded_delta).max())
        step_sizes.append(denominator)
        if denominator > 1e-9:
            ratios.append(float(np.abs(after - before).max() / denominator))

        # The gripper is compared the way the arm is -- the achieved opening against the
        # recorded achieved opening, not against the commanded one.  The dataset's `state[7]`
        # is where the jaw *is* and its `action[7]` is where it was told to go, and judging the
        # first by the second measures servo lag while looking like a convention mismatch.
        observed_gripper = scene.state()
        gripper_errors.append(float(np.abs(observed_gripper -
                                            expected)[GRIPPER_INDICES].max()))

    residuals_array = np.asarray(residuals)
    per_joint_array = np.vstack(per_joint)
    free = np.asarray([not flag for flag in contact_flags])
    free_residuals = residuals_array[free]
    gripper_array = np.asarray(gripper_errors)
    #: How big a step the expert takes, as the scale for the residual.  A residual measured
    #: against nothing is a number; against the motion it is meant to track it is a property.
    #: `a3_cookie_one` records a median step of ~0.02 rad, so 0.0033 is a sixth of one.
    median_step = float(np.median(step_sizes)) if step_sizes else None
    ratio_to_step = (float(np.percentile(free_residuals, 90)) / median_step
                     if (free_residuals.size and median_step) else None)
    report["teacher_forced"] = {
        "steps": len(residuals),
        "steps_in_free_space": int(free.sum()),
        "steps_in_contact": int((~free).sum()),
        "residual_max_rad": float(residuals_array.max()) if len(residuals_array) else None,
        "residual_median_rad": float(np.median(residuals_array)) if len(residuals_array) else None,
        "residual_p90_rad": float(np.percentile(residuals_array, 90)) if len(residuals_array) else None,
        #: The test that is actually a test: with no contact the next state follows from
        #: `(state, action)` and the simulator, so any residual is a channel difference.  The
        #: contact steps are reported too -- they are most of a demonstration -- but they
        #: cannot be reproduced exactly, because the dataset does not record where the props
        #: were.
        "free_space_residual_max_rad": float(free_residuals.max()) if free_residuals.size else None,
        "free_space_residual_p90_rad": (float(np.percentile(free_residuals, 90))
                                       if free_residuals.size else None),
        "median_step_rad": median_step,
        "free_space_residual_over_step": ratio_to_step,
        "per_joint_median_rad": np.median(np.abs(per_joint_array), axis=0).round(5).tolist(),
        "worst_joint": scene.arm[int(np.argmax(np.median(np.abs(per_joint_array), axis=0)))],
        "gripper_error_max": float(np.max(gripper_errors)),
        #: The gripper is judged in free space, like the arm: in contact the jaw can be pushed
        #: by a prop that is not where the recording had it, and that is the harness again.
        "gripper_error_free_space_max": (float(gripper_array[free].max())
                                        if free.any() else None),
        "moved_over_expected_median": float(np.median(ratios)) if ratios else None,
    }

    # ---- open loop ----
    #
    # **Two channels, two units, two numbers.**  The state is fourteen joint angles in
    # radians and two gripper openings in `[0, 1]`; taking one `max` over all sixteen mixes
    # them, and a gripper travelling its whole range contributes 1.0 against an arm that moves
    # in hundredths.  The first version of this file did exactly that and reported a 0.94 rad
    # "divergence" that was a gripper closing.
    scene.reset_scene(args.episode)
    scene.load(states[0])
    arm_divergence: list[float] = []
    grip_divergence: list[float] = []
    open_contact: list[bool] = []
    for index in range(min(args.open_loop, len(actions))):
        if scene.touching():
            open_contact.append(True)
        command(scene, actions[index])
        if scene.env._safety_stop:  # noqa: SLF001
            stopped = {"phase": "open loop", "frame": index,
                       "reason": scene.env._safety_reason}  # noqa: SLF001
            break
        measured = scene.state()
        recorded = states[index + 1]
        difference = np.abs(measured - recorded)
        # The dataset's layout: `[left 0..6, left grip, right 0..6, right grip]`.  The arm is
        # everything except the two grippers -- index 7 is a gripper, and leaving it inside
        # `range(14)` is how the first version of this reported one number twice.
        arm_divergence.append(float(difference[ARM_INDICES].max()))
        grip_divergence.append(float(difference[GRIPPER_INDICES].max()))
    #: Where the open loop starts tracking the recording rather than the transient of its own
    #: initial condition.  `load` sets position and velocity only; anything else the recording
    #: never mentions (the props) comes from the scene reset, so the first steps carry a
    #: disagreement that is not the dynamics.  The settled peak is the quantity to judge by,
    #: and `reset_scene` is what makes it reproducible: without it the same loop scored 0.0096
    #: or 0.1524 rad depending only on what ran before it.
    settled = arm_divergence[10:] or arm_divergence
    report["open_loop"] = {
        "steps": len(arm_divergence),
        "arm_per_step_rad": [round(v, 5) for v in arm_divergence[:50]],
        "arm_peak_rad": round(max(arm_divergence), 5) if arm_divergence else None,
        "arm_peak_at_step": int(np.argmax(arm_divergence)) + 1 if arm_divergence else None,
        "arm_settled_peak_rad": round(max(settled), 5) if settled else None,
        "arm_at_end_rad": round(arm_divergence[-1], 5) if arm_divergence else None,
        "gripper_peak": round(max(grip_divergence), 5) if grip_divergence else None,
        "gripper_at_end": round(grip_divergence[-1], 5) if grip_divergence else None,
        "steps_before_first_contact": (open_contact.index(True)
                                        if True in open_contact else len(open_contact)),
    }

    # ---- verdict ----
    #: The residual is a **fraction of one recorded step**, not a number of radians: the two
    #: datasets here differ by an order of magnitude in step size (median 0.0042 and 0.0088 rad)
    #: and an absolute threshold cannot separate "tracks the recording" from "does not".
    #:
    #: The value is measured on both sides rather than picked.  Correct pairing, recorded
    #: velocity: **0.00** of a step (`a3_cookie_single_piece` / `a3_cookie_one`) and **0.07**
    #: (`a3_cookie_batch` / `a3_cookie_adv_1000_merged`).  The smallest deliberate error this
    #: channel can make -- `--rest-start`, dropping the recorded velocity from the initial
    #: condition -- scores **0.76**.  Half a step sits between them.
    #:
    #: **What this test does not certify: the scene.**  The arm is the same robot in every
    #: scene here, so a *deliberately wrong pairing* (`a3_cookie_batch` replaying
    #: `a3_cookie_one`) also scores 0.00.  This channel certifies the action, state, unit and
    #: servo conventions on their way to the arm -- which is where the mirrored image, the
    #: wrong picture and the legacy gripper all showed up as fatal -- and says nothing about
    #: whether the props, materials or camera are right.  Those are `check_invariants.py` and
    #: the contact split below.
    RESIDUAL_TOLERANCE_OF_STEP = 0.5
    #: Measured free-space maximum: 0.003 (single piece) and 0.029 (batch).  The gripper's
    #: failure mode that this is here for is a convention error, which scores ~1.0.
    GRIPPER_TOLERANCE = 0.10
    worst = report["teacher_forced"]["free_space_residual_over_step"]
    gripper = report["teacher_forced"]["gripper_error_free_space_max"]
    ok = (worst is not None and worst <= RESIDUAL_TOLERANCE_OF_STEP
          and gripper is not None and gripper <= GRIPPER_TOLERANCE)
    if stopped is not None:
        ok = False
    report["verdict"] = ("INVALID -- the environment stopped itself" if stopped else
                         ("DYNAMICS MATCH" if ok else "DYNAMICS DIFFER -- treat as a bug"))
    report["tolerance_of_step"] = RESIDUAL_TOLERANCE_OF_STEP
    report["gripper_tolerance"] = GRIPPER_TOLERANCE
    report["stopped"] = stopped

    print(f"scene   {args.scene}  episode {args.episode}  frames {args.first}.."
          f"{args.first + len(states) - 1}")
    tf = report["teacher_forced"]
    print(f"one step  {tf['steps']} frames: {tf['steps_in_free_space']} in free space, "
          f"{tf['steps_in_contact']} in contact")
    print(f"          free-space residual p90 {tf['free_space_residual_p90_rad']:.4f} rad = "
          f"{tf['free_space_residual_over_step']:.2f} of a recorded step "
          f"(median {tf['median_step_rad']:.4f} rad, tolerance "
          f"{RESIDUAL_TOLERANCE_OF_STEP})   worst joint {tf['worst_joint']}")
    print(f"          gripper error max {tf['gripper_error_max']:.4f} "
          f"({tf['gripper_error_free_space_max']:.4f} in free space, tolerance "
          f"{GRIPPER_TOLERANCE})   moved/expected median {tf['moved_over_expected_median']:.3f}")
    ol = report["open_loop"]
    # Peak, not endpoint: this scene's arm is position-controlled, so an uncorrected replay
    # converges back onto the recording once the servo is driving the same targets -- the
    # endpoint of a long open loop says very little, and the excursion on the way says how
    # far apart the two worlds ever were.
    print(f"open loop {ol['steps']} steps: arm peak {ol['arm_peak_rad']} rad at step "
          f"{ol['arm_peak_at_step']}, settled peak {ol['arm_settled_peak_rad']}, "
          f"{ol['arm_at_end_rad']} at the end;  gripper peak {ol['gripper_peak']}, "
          f"end {ol['gripper_at_end']}")
    if stopped is not None:
        print(f"STOPPED at frame {stopped['frame']} of the {stopped['phase']}: "
              f"{stopped['reason']}\n        Every number above comes from before that frame; the "
              f"environment holds position afterwards, so the sweep is truncated, not wrong.")
    print(f"\n{report['verdict']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json    {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
