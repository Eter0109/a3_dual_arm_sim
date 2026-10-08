"""One yardstick for any object on the table -- the ladder's instrument.

Axis 4 of the ladder changes the object and nothing else, so the two ends have to be
measured by the *same code in the same units* or the comparison is between two probes
rather than between two scenes.  This runs a policy in a scene, watches one named body,
and reports the quantities Gate 3 of `docs/scene-transition-and-bug-triage.md` asks for:

  * the nearest fingertip's distance to that body's centre;
  * how far the body ever moved, and whether anything ever touched it;
  * whether the gripper closed *while it was on the body* or after it had left it.

Why a second probe rather than a flag on the first
--------------------------------------------------
`probe_libero_props_trace.py` is about a *bowl*, and not incidentally: its central
quantities are the rim's radius and the fingertip's height against the rim, because the
bowl is a hollow open-topped shell and "25 mm from the centre" describes a fingertip above
the opening and one reaching into the cavity equally well.  A 6.3 mm-thick Cookie standing
on its edge has no mouth, so those two numbers are undefined rather than small.

Retrofitting it would also change the probe the guard's LIBERO canary runs on.  That
canary's numbers are a recorded baseline (`configs/invariants_baseline.json`) and the
format of its output is part of what is compared, so a probe that reports different things
after an edit would move the baseline instead of catching a regression.

The two ends
------------
    # the working end: the policy places LIBERO's bowl on LIBERO's plate, 35/40
    python scripts/probe_ladder_object.py --scene a3_libero_panda \\
        --object libero_bowl_object --target libero_plate_object \\
        --manifest configs/policies/pi05_libero_panda.yaml \\
        --task "pick up the black bowl between the plate and the ramekin and place it on the plate"

    # axis 4: the same scene, camera, robot and physics, with A3's Cookie instead
    python scripts/probe_ladder_object.py --scene a3_libero_panda_cookie \\
        --object cookie_0 --target libero_plate_object \\
        --manifest configs/policies/pi05_libero_panda.yaml \\
        --task "pick up the yellow block and place it on the plate"

A **capability** failure degrades monotonically along such an axis; a **bug** is a cliff at
a wiring boundary.  The numbers this prints are what tells the two apart, so they are
printed rather than summarised into one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np

from a3_dual_arm_sim.props import (
    EEF_SITE,
    FINGERTIPS,
    arm_qpos_addresses,
    prop_frame,
    prop_geoms,
    robot_bodies,
    touching,
)

PROJECT = Path(__file__).resolve().parents[1]

#: One unit of action, under the scene's `cartesian_translation_scale_m`.  Same constant as
#: the bowl probe, and for the same reason: the model's command is in normalised OSC units
#: and a number in those units is not readable.
ACTION_UNIT_M = 0.05

#: `observation.state[7]` -- the left gripper's opening, in the scene's own convention.
GRIPPER_STATE_INDEX = 7

#: The cameras a policy sees, and the ones a clip shows beside the scene view.  Same three
#: `render_libero_props_video.py` stacks, because a clip is for comparing against the ones
#: already recorded rather than for a second opinion about which cameras matter.
POLICY_CAMERAS = ("front", "left_wrist", "right_wrist")

#: A close is a *fraction of this run's own open value*, not an absolute opening.
#:
#: The absolute version of this existed and was wrong.  0.05 was calibrated on the A3's
#: finger servo, which reaches 0.027 over an **empty** jaw travel.  In `a3_libero_panda`
#: the jaws stop at **0.0823** because they are holding a 112 mm bowl, and a gripper
#: gripping something cannot reach the value it reaches gripping nothing.  Against 0.05 the
#: probe reported `gripper closed on 0/150 steps` and
#: `moved it without ever closing the gripper on it` for a rollout that closed on step 47,
#: held the hold, and carried the bowl 114.7 mm of the 117.6 mm to the plate.  The error is
#: not merely conservative: it asserts "the policy cannot grasp", which is the hypothesis
#: this ladder exists to test, so a threshold that can only fail is worse than none.
#:
#: The reference is the **maximum opening seen in the run** -- the open value in both
#: scenes (A3 1.0; the Panda's `pruned_init` poses 0.6755, rising to 0.994) -- and the rule
#: is "below half of it".  Both real holds (0.027 and 0.0823) are far below that and both
#: idles are above, so no scene needs a constant of its own.
CLOSED_FRACTION_OF_OPEN = 0.5

#: How far the object has to move before it counts as moved at all.  The props settle to
#: within 0.63 mm of their start height in this scene (measured, `libero_props_baseline12`),
#: so this is a settled-noise allowance rather than a physical tolerance.
MOVED_MM = 2.0

#: How far the object has to **rise above where it started**, and for how many steps, to
#: count as carried.
#:
#: This is the grasp test that cannot be argued with: a carried object has left the surface
#: it was resting on, and it stays away from it.  Both halves are needed and each rules out a
#: different impostor, measured over the four traces to hand (`.npz` files in
#: `/root/autodl-tmp/logs/ladder`, re-read offline):
#:
#: * height alone is passed by a **topple with a tip** -- seed 27 of the aimed A3-task run
#:   recorded a peak rise of +6.7 mm and a fall of -9.4 mm while never moving more than
#:   16.5 mm, and `LIFT_MM = 5` called that "grasped it and carried it";
#: * duration alone is passed by a **brush**, which can hold the object a centimetre up for
#:   a step or two.
#:
#: The separation is not marginal in either direction.  Real carries: the bowl's 20 seeds at
#: 232-266 steps above 5 mm and 21-247 above 15 mm; axis 4's two real carries at 21 and 34
#: steps above 15 mm.  Every other rollout that rose at all -- axis 4's two topples, the
#: unaimed run's one, the aimed run's three -- sat at **0 steps above 15 mm**, with peak rises
#: of 2.1, 4.1, 6.7, 8.8 and 2.4 mm.  So the thresholds sit in a 12x gap in height and a
#: 21-step gap in duration rather than at a value chosen to make a number look right.
LIFT_MM = 15.0
LIFT_HOLD_STEPS = 5


def _lifted_steps(trace: dict[str, np.ndarray]) -> np.ndarray:
    """The steps on which the object was **carried**: off the surface, and staying off.

    Returns the steps above ``LIFT_MM``, which the caller compares against
    ``LIFT_HOLD_STEPS``.  Returned rather than reduced to a boolean because the count is
    the evidence -- it is printed, and it is what separates the sixteen-step carry from the
    four-step tip in the traces this threshold was measured against.
    """

    return np.flatnonzero(trace["object_height_mm"] > LIFT_MM)


def _closed_steps(trace: dict[str, np.ndarray]) -> np.ndarray:
    """The steps on which the gripper counts as closed, against its own open value.

    Returned rather than reduced to a boolean because both callers need to know *which*
    step, and comparing a closure time against a contact time is the only way to tell "it
    picked the object up" from "it put the object down and then closed".
    """

    reference = float(np.max(trace["gripper"]))
    return np.flatnonzero(trace["gripper"] < CLOSED_FRACTION_OF_OPEN * reference)


def _held_steps(trace: dict[str, np.ndarray]) -> np.ndarray:
    """The steps on which the gripper was **closed and on the object at the same time**.

    This is the whole grasp test, and it used to be a one-step lookback instead: `was the
    object in contact on the step before the first closed step`.  On `a3_libero_panda` seed
    2 the gripper crossed its close threshold on the same step it first touched the bowl --
    it had been shutting over the approach, 0.99 -> 0.81 -> 0.57 
    -> 0.35 across steps 34-37 -- so the step before the close was `no contact`, and a
    rollout with **84** steps of closed-and-touching was reported as `closed the gripper
    after leaving it`.  Reading a *duration* off a step boundary is the error; a state that
    holds for 84 steps cannot be inferred from the one step before its first frame.
    """

    closed = np.zeros(len(trace["gripper"]), dtype=bool)
    closed[_closed_steps(trace)] = True
    return np.flatnonzero(closed & (trace["touch_arm"] > 0))


class _Recorder:
    """A clip of the rollout, drawn from the same numbers the summary prints.

    Why this is here rather than a flag on `render_libero_props_video.py`: that recorder's
    chart is the bowl's mouth against the plate's edge and its scene lookup needs
    `BOWL_MESH_BODY`, so pointing it at this scene draws a picture about a plate that is not
    there.  This one draws **these** quantities -- the nearest fingertip's distance to the
    watched object, the object's distance to its target, and the band where the object was
    off the surface -- which means the clip and the table cannot disagree: they are one
    computation, not two that happen to be about the same scene.

    Layout: the scene view on the left at ``view_size``, the policy's three cameras stacked
    at the same height to its right, and the chart along the bottom.
    """

    def __init__(
        self,
        path: Path,
        *,
        view_size: int,
        panel_size: int,
        chart_height: int,
        fps: int,
        crf: int,
        steps: int,
        object_radius_mm: float,
        target_radius_mm: float | None,
        title: str,
    ) -> None:
        import av

        self.view_size = view_size
        self.panel_size = panel_size
        self.chart_height = chart_height
        self.width = view_size + panel_size
        self.height = view_size + chart_height
        self.steps = steps
        self.object_radius_mm = object_radius_mm
        self.target_radius_mm = target_radius_mm
        self.title = title
        #: `(gap_mm, target_mm, off_the_surface)` per frame, which is the chart.
        #:
        #: The second series is the **target's own displacement**, not its distance from the
        #: watched object.  It was the latter, and that is a curve about a fixed reference:
        #: it is the right thing to draw while the target is scenery, and in this scene the
        #: target is a free body the policy picks up and carries, so the curve held a flat
        #: 131 mm through the one event that mattered (F13).  The distance to the target is
        #: still in the trace; what the picture needs is the thing that moved.
        self.history: list[tuple[float, float, bool]] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "medium"}

    def _x(self, index: int) -> float:
        return (index / max(self.steps - 1, 1)) * (self.width - 1)

    def _chart(self):
        from PIL import Image, ImageDraw

        chart = Image.new("RGB", (self.width, self.chart_height), (16, 16, 20))
        draw = ImageDraw.Draw(chart)
        values = [v for pair in self.history for v in pair[:2]]
        top_mm = max([60.0, self.object_radius_mm * 1.4] + [v * 1.05 for v in values])
        baseline = self.chart_height - 5
        span = self.chart_height - 16

        def y(value_mm: float) -> float:
            return baseline - (value_mm / top_mm) * span

        # The band where the object was off the surface: the one thing the clip is here to
        # settle, so it is drawn behind everything else and as an area, not a line.
        for index, (_, _, off) in enumerate(self.history):
            if off:
                draw.line((self._x(index), 0, self._x(index), self.chart_height),
                          fill=(34, 62, 40))
        for radius, colour, label in (
            (self.object_radius_mm, (255, 210, 120), "object radius"),
            (self.target_radius_mm, (150, 200, 255), "target radius"),
        ):
            if radius is None:
                continue
            draw.line((0, y(radius), self.width, y(radius)), fill=colour)
            draw.text((4, y(radius) - 11), f"{label} {radius:.0f} mm", fill=colour)
        for column, colour in ((0, (120, 255, 160)), (1, (255, 140, 140))):
            points = [(self._x(i), y(pair[column])) for i, pair in enumerate(self.history)
                      if np.isfinite(pair[column])]
            if len(points) > 1:
                draw.line(points, fill=colour, width=2)
        draw.text((4, 2), "green: nearest fingertip to the object   "
                          "red: the target's own displacement   band: object off the surface",
                  fill=(200, 200, 200))
        return chart

    def add(self, view, cameras: dict, hud: list[str]) -> None:
        """One frame: the pictures, the chart so far, and the numbers as text."""

        import av
        import numpy as np
        from PIL import Image, ImageDraw

        frame = Image.new("RGB", (self.width, self.height), (10, 10, 12))
        frame.paste(Image.fromarray(view).resize((self.view_size, self.view_size)),
                    (0, 0))
        for index, (name, image) in enumerate(cameras.items()):
            frame.paste(
                Image.fromarray(np.asarray(image)).resize((self.panel_size, self.panel_size)),
                (self.view_size, index * self.panel_size),
            )
        frame.paste(self._chart(), (0, self.view_size))

        draw = ImageDraw.Draw(frame)
        draw.text((6, 6), self.title, fill=(255, 230, 120))
        for index, line in enumerate(hud):
            draw.text((6, 20 + index * 12), line, fill=(235, 235, 235))

        # The text sits on top of the scene view, so it needs something behind it.
        self.container.mux(self.stream.encode(
            av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")
        ))

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="a3_libero_panda")
    parser.add_argument("--object", default="libero_bowl_object",
                        help="the body to watch; `cookie_0` in the axis-4 scene")
    parser.add_argument("--target", default="libero_plate_object",
                        help="the body a placement is against; empty to skip")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT / "configs/policies/pi05_libero_panda.yaml")
    parser.add_argument("--task", default="",
                        help="the instruction; empty uses the scene's own prompt")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None,
                        help="write a clip of the whole run: scene view, policy cameras, "
                             "and the chart of the ladder's own quantities")
    parser.add_argument("--view-size", type=int, default=480)
    parser.add_argument("--panel-size", type=int, default=160)
    parser.add_argument("--chart-height", type=int, default=150)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--shadows", action="store_true",
                        help="keep the shadow passes; four times slower for a human view")
    return parser.parse_args()


#: The closed set of shapes a rollout can take.  A **fixed set** rather than a sentence with
#: the numbers baked in, because the whole purpose of naming a shape is to count them across
#: seeds and across the two ends of the ladder -- and a label that changes with the
#: measurement ("moved it 18.5 mm but never lifted it" against "...21.7 mm...") cannot be
#: counted, diffed, or compared.  The numbers belong next to it, not inside it.
SHAPES = (
    "the arm never moved (execution bug, not capability)",
    "never touched the object",
    "never recorded in contact, but displaced it",
    "touched it, never moved it",
    "displaced it without lifting it off the surface",
    "lifted it without the gripper ever closed on it",
    "grasped it and carried it",
)


def _signature(trace: dict[str, np.ndarray]) -> str:
    """Which of :data:`SHAPES` this rollout is.

    Read off the observations, not off a threshold that decides pass or fail: whether
    anything touched the object, whether it moved, and whether it left the surface.  The
    shapes mean different things -- `never touched it` and `displaced it without lifting
    it` are both "it failed" and are opposite findings, one about perception and one about
    the hold.
    """

    if float(np.abs(trace["joint_motion_mrad"]).max()) <= 0.0:
        return SHAPES[0]
    touched = int(trace["touch_arm"].sum())
    moved = float(trace["object_mm"].max())
    lifted = _lifted_steps(trace)
    held = _held_steps(trace)
    if touched == 0:
        # `touch_arm` is MuJoCo's contact list read **once per control step**, so it is a
        # lower bound, and the bound is loosest for the smallest object.  A 112 mm bowl
        # presents a contact that lasts tens of steps and cannot be missed; a 6.4 mm-thin
        # Cookie brushed within one 50 ms control step can be, and it was -- on
        # `a3_libero_panda_cookie` eight of twenty seeds displaced the Cookie 14-55 mm with
        # no contact recorded at any sample.  Saying "never touched the object" there is not
        # conservative, it is false, and it is false in the direction that flatters the
        # working end of the ladder.  Displacement is the channel that cannot miss it, so it
        # decides which of the two shapes this is.
        return SHAPES[1] if moved <= MOVED_MM else SHAPES[2]
    if moved <= MOVED_MM:
        return SHAPES[3]
    if lifted.size < LIFT_HOLD_STEPS:
        return SHAPES[4]
    if held.size == 0:
        return SHAPES[5]
    return SHAPES[6]


def _grasp_timing(trace: dict[str, np.ndarray]) -> str:
    """Did the gripper close while it was on the object, or after it had left it?

    Reported as a *timing* rather than as a rate because a grip rate cannot separate those
    two, and they are different failures with the same summary number.
    """

    closed = _closed_steps(trace)
    held = _held_steps(trace)
    contacts = np.flatnonzero(trace["touch_arm"] > 0)
    if closed.size == 0:
        return "the gripper never closed"
    if contacts.size == 0:
        return (f"the gripper first closed on step {int(closed[0])}, on an object it had "
                f"never touched")
    if held.size:
        return (f"the gripper was closed on the object for {held.size} steps, first on step "
                f"{int(held[0])} (first contact {int(contacts[0])}, first close "
                f"{int(closed[0])})")
    return (f"the gripper first closed on step {int(closed[0])} and never while it was on "
            f"the object, whose last contact was step {int(contacts[-1])}")


def main() -> int:
    args = parse_args()

    from a3_dual_arm_sim.bridge.gateway import ExternalPolicyGateway
    from a3_dual_arm_sim.bridge.manifest import load_manifest
    from a3_dual_arm_sim.collection import scene_by_name
    from a3_dual_arm_sim.contracts import STATE, EpisodeContext

    scene = scene_by_name(args.scene)
    task = args.task or scene.task.prompt
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gateway = None
    runs: list[dict[str, np.ndarray]] = []
    recorder: _Recorder | None = None

    for seed in seeds:
        env = scene.build_env(0.0, 0.0)
        if not args.shadows:
            env.use_fast_render()
        observation, _ = env.reset(seed=seed, options=None)

        model = env.model
        watched = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object)
        target = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.target)
                  if args.target else -1)
        tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in FINGERTIPS]
        eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        if watched < 0 or min(eef, *tips) < 0:
            raise SystemExit(
                f"{args.scene} has no body {args.object!r}, or is missing the arm's sites"
            )

        if gateway is None:
            gateway = ExternalPolicyGateway(load_manifest(args.manifest), env=env)
        else:
            gateway.rebind(env)
        gateway.reset(EpisodeContext(seed=seed, task=task, action_mode="joint_position"))

        # `geom_xpos` is derived, and a fresh reset has not necessarily been forwarded.
        mujoco.mj_forward(model, env.data)
        start, radius_m, size_m = prop_frame(model, env.data, watched)
        target_centre = None if target < 0 else prop_frame(model, env.data, target)[0]
        target_start = target_centre
        watched_geoms = set(prop_geoms(model, watched))
        arm_bodies = robot_bodies(model)
        arm_qpos = arm_qpos_addresses(model)

        if seed == seeds[0]:
            print(f"scene    {args.scene}")
            print(f"watched  {args.object}   centre {np.round(start, 4)}  "
                  f"radius {radius_m * 1000:.1f} mm  size "
                  f"{np.round(size_m * 1000, 1)} mm")
            if target_centre is not None:
                print(f"target   {args.target}   centre {np.round(target_centre, 4)}  "
                      f"radius {prop_frame(model, env.data, target)[1] * 1000:.1f} mm   "
                      f"start distance "
                      f"{np.linalg.norm(start[:2] - target_centre[:2]) * 1000:.1f} mm")
            print(f"task     {task!r}")
            print()

        if args.video is not None and recorder is None:
            # Built from the first seed's measured geometry: the object's radius and the
            # target's are the same in every seed, and they are what the chart draws its
            # reference lines at.
            recorder = _Recorder(
                args.video,
                view_size=args.view_size,
                panel_size=args.panel_size,
                chart_height=args.chart_height,
                fps=args.fps,
                crf=args.crf,
                steps=args.steps,
                object_radius_mm=radius_m * 1000.0,
                target_radius_mm=(None if target_centre is None else
                                  prop_frame(model, env.data, target)[1] * 1000.0),
                title=f"{args.scene}  seed {seed}  {args.object} -> {args.target or '-'}",
            )

        rows = []
        previous_joints: np.ndarray | None = None
        for step in range(args.steps):
            action = gateway.act(observation, task)
            observation, _, terminated, truncated, info = env.step(action)

            centre = prop_frame(model, env.data, watched)[0]
            fingertips = [np.asarray(env.data.site_xpos[t]) for t in tips]
            gap = min(float(np.linalg.norm(tip - centre)) for tip in fingertips)
            eef_position = np.asarray(env.data.site_xpos[eef], dtype=np.float64)
            joints = np.asarray(env.data.qpos[arm_qpos], dtype=np.float64)
            gripper = float(np.asarray(observation[STATE], dtype=np.float64)[
                GRIPPER_STATE_INDEX])
            on_arm = touching(model, env.data, watched_geoms, arm_bodies)

            if recorder is not None:
                height_mm = float(centre[2] - start[2]) * 1000.0
                target_mm = (float("nan") if target_start is None else
                             float(np.linalg.norm(
                                 prop_frame(model, env.data, target)[0] - target_start))
                             * 1000.0)
                recorder.history.append(
                    (gap * 1000.0, target_mm, height_mm > LIFT_MM)
                )
                recorder.add(
                    env.render_view("front", height=args.view_size, width=args.view_size),
                    {name: observation[f"observation.images.{name}"]
                     for name in POLICY_CAMERAS},
                    [
                        f"seed {seed}   step {step}",
                        "fingertip " + f"{gap * 1000.0:6.1f} mm"
                        + ("   target moved " + f"{target_mm:6.1f} mm"
                           if np.isfinite(target_mm) else ""),
                        "height " + f"{height_mm:+6.1f} mm"
                        + f"   off surface {int(height_mm > LIFT_MM)}"
                        + f"   arm on it {int(on_arm)}",
                        "gripper " + f"{gripper:.3f}",
                    ],
                )

            rows.append(
                {
                    "step": step,
                    "gap_mm": gap * 1000.0,
                    "object_mm": float(np.linalg.norm(centre - start)) * 1000.0,
                    "object_height_mm": float(centre[2] - start[2]) * 1000.0,
                    # **Two different distances, and conflating them cost a reading.**
                    #
                    # `object_target_mm` is against where the target *started*, which is the
                    # right reference only while the target is scenery.  In `a3_libero_panda`
                    # the target is LIBERO's plate and never moves, so it was right for every
                    # scene this probe was written for.  In `a3_libero_panda_single_cookie` the
                    # target is A3's ten-slot box, which is a **free rigid body** -- and the
                    # policy picks it up and carries it into the source bin.  The trace then
                    # said "the object never moved (median 0.3 mm), the target gap is flat at
                    # 131 mm, nothing is happening" while the one thing actually being
                    # manipulated was the box.  A clip showed it in the last frames.
                    #
                    # So the target's own displacement is recorded, and `live_gap_mm` is the
                    # distance to where the target is *now*.  A probe cannot report on a
                    # scene that moves a reference it assumed was fixed.
                    "object_target_mm": (float("nan") if target_start is None else
                                         float(np.linalg.norm(
                                             centre[:2] - target_start[:2])) * 1000.0),
                    "target_mm": (float("nan") if target_start is None else
                                  float(np.linalg.norm(
                                      prop_frame(model, env.data, target)[0] - target_start))
                                  * 1000.0),
                    "live_gap_mm": (float("nan") if target_start is None else
                                    float(np.linalg.norm(
                                        centre[:2]
                                        - prop_frame(model, env.data, target)[0][:2]))
                                    * 1000.0),
                    # MuJoCo's own contact list, not a proximity threshold.
                    "touch_arm": float(on_arm),
                    "touch_table": float(touching(model, env.data, watched_geoms, {0})),
                    "gripper": gripper,
                    "eef_x_mm": float(eef_position[0] - start[0]) * 1000.0,
                    "eef_y_mm": float(eef_position[1] - start[1]) * 1000.0,
                    "eef_z_mm": float(eef_position[2] - start[2]) * 1000.0,
                    # The arm-moved signature: **measured** joint motion, in mrad.  It used
                    # to be read off `info["applied_action"]`, which is wrong in the Panda
                    # scene in two separate ways -- `env._limit_joint_action` returns those
                    # seven entries *unclipped* when the arm is driven by an
                    # `OperationalSpaceAdapter`, and under that adapter they are **torques
                    # in N.m, not joint angles**.  The difference is not subtle: it reported
                    # a peak of 17 129 "mrad" per step, which is 17 rad in one 50 ms control
                    # step, i.e. 340 rad/s.  The bound that makes that absurd is not
                    # `max_joint_step_rad` -- which the OSC path deliberately bypasses, there
                    # being no trajectory to rate-limit -- but the environment's own safety
                    # stop, which fires above **80 rad/s**.  A quantity recorded under a unit
                    # it does not have is how a rollout that never moved gets mistaken for
                    # one that did, and the reverse.
                    "joint_motion_mrad": 0.0 if previous_joints is None
                    else float(np.abs(joints - previous_joints).max()) * 1000.0,
                    # Kept, but named for what it is: the raw entry the servo received,
                    # in whatever unit the scene's action mode puts there.  Compare it with
                    # nothing across scenes -- `joint_motion_mrad` is that quantity.
                    "command_max": float(np.abs(
                        np.asarray(info["applied_action"], dtype=np.float64)[:7]).max()),
                }
            )
            previous_joints = joints
            if terminated or truncated:
                print(f"seed {seed}: episode ended at step {step}", flush=True)
                break
        env.close()

        trace = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        runs.append(trace)
        closest = int(np.argmin(trace["gap_mm"]))
        open_reference = float(np.max(trace["gripper"]))
        print(
            f"seed {seed}: nearest fingertip {trace['gap_mm'][closest]:.1f} mm from the "
            f"object centre at step {closest}  |  object moved "
            f"{trace['object_mm'].max():.2f} mm, peak height "
            f"{trace['object_height_mm'].max():+.1f} mm (deepest "
            f"{trace['object_height_mm'].min():+.1f} mm, off the surface for "
            f"{_lifted_steps(trace).size} steps)  |  gripper closed on "
            f"{_closed_steps(trace).size}/{len(rows)} steps and closed *on the object* for "
            f"{_held_steps(trace).size}, arm on "
            f"the object for {int(trace['touch_arm'].sum())}, on the table for "
            f"{int(trace['touch_table'].sum())}  |  {len(rows)} of {args.steps} steps run\n"
            f"        the target moved {np.nanmax(trace['target_mm']):.1f} mm"
            f" (its start is the only reference the old run had)\n"
            f"        gripper: open value this run {open_reference:.4f}, so closed means "
            f"below {CLOSED_FRACTION_OF_OPEN * open_reference:.4f}, and its minimum was "
            f"{float(trace['gripper'].min()):.4f} on step "
            f"{int(trace['gripper'].argmin())}\n"
            f"        {_grasp_timing(trace)}   ->  {_signature(trace)}",
            flush=True,
        )

    if gateway is not None:
        gateway.close()

    if recorder is not None:
        recorder.close()
        print(f"video    {args.video}")

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.trace, **{f"{key}_s{index}": trace[key]
                                for index, trace in enumerate(runs) for key in trace})

    # The three Gate-3 quantities, per seed, in the units the decision is made in.  Printed
    # as a list rather than a mean because this policy samples fresh flow-matching noise per
    # chunk: the spread across seeds is itself a finding (`probe_libero_props_trace.py` says
    # the same thing at more length).
    closest = [float(t["gap_mm"].min()) for t in runs]
    print()
    print(f"nearest fingertip to {args.object}, per seed: "
          + ", ".join(f"{value:.1f}" for value in closest)
          + f" mm   (min {min(closest):.1f}, spread {max(closest) - min(closest):.1f})")
    print("object displaced, per seed: "
          + ", ".join(f"{float(t['object_mm'].max()):.2f}" for t in runs) + " mm")
    print("peak height above its start, per seed: "
          + ", ".join(f"{float(t['object_height_mm'].max()):+.1f}" for t in runs) + " mm")
    print("steps with the arm touching the object, per seed: "
          + ", ".join(f"{int(t['touch_arm'].sum())}/{len(t['touch_arm'])}" for t in runs))
    if runs and not np.isnan(runs[0]["object_target_mm"]).all():
        print(f"object to {args.target} at the last step, per seed: "
              + ", ".join(f"{float(t['object_target_mm'][-1]):.1f}" for t in runs)
              + " mm;  best "
              + ", ".join(f"{float(t['object_target_mm'].min()):.1f}" for t in runs))
    print("peak arm motion per step, per seed: "
          + ", ".join(f"{float(np.abs(t['joint_motion_mrad']).max()):.2f}" for t in runs)
          + " mrad")
    print("signature, per seed: "
          + " | ".join(_signature(trace) for trace in runs))

    if args.trace is not None:
        print(f"trace   {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
