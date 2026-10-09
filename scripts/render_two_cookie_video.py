"""Record the two-Cookie attempt, and name whatever moves the Cookie that was already down.

Stage 2 gets as far as carrying the second Cookie and fails with "packed Cookie 0
disturbed".  That message is the expert's own guard -- it fails as soon as a Cookie
it has already placed has moved more than 6 mm, or has tipped more than 12 degrees
in the box's frame -- and the guard says nothing about *what* moved it.  Numbers
alone cannot: a displacement is the same whether the jaws clipped the Cookie, the
second Cookie was laid on top of it, or it slid when the box flexed under the arm.

So this does two things at once, because they answer the same question.  It writes
a video of the whole attempt, with the front scene camera next to the wrist camera
and both Cookie states on the overlay, and it prints the contact list of the
disturbed Cookie on the step the guard trips -- the pair of geoms, the distance,
and the normal force, so "the jaws brushed it" and "the other Cookie hit it" are
different lines.

Usage::

    python scripts/render_two_cookie_video.py --seed 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from a3_dual_arm_sim.collection import libero_props_scene

PROJECT = Path(__file__).resolve().parents[1]

#: Kept in step with `run_expert_gate.py`, so the clip shows that configuration.
LAYOUT = {
    "matched_cameras": True,
    "a3_single_cookie": True,
    "a3_layout_offset_m": (0.020, -0.200),
    "a3_cookie_positions_m": ((0.1194, 0.04875), (0.1194, 0.11941667)),
}

VIEW = 480
PANEL = 240
STRIP = 96


def load_font(size: int):
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def cookie_state(env, cookie: int) -> tuple[np.ndarray, float, float]:
    """Position in the box's frame, tilt from vertical, and heading."""

    box_position = env.data.xpos[env._target_bin_body]
    box_matrix = env.data.xmat[env._target_bin_body].reshape(3, 3)
    relative = box_matrix.T @ (env.data.xpos[env._cookie_bodies[cookie]] - box_position)
    rotation = env._cookie_rotation(cookie)
    tilt = np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), -1.0, 1.0)))
    heading = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return relative, float(tilt), float(heading)


def name_of(env, kind, index: int) -> str:
    return mujoco.mj_id2name(env.model, kind, int(index)) or "?"


def describe_contacts(env, cookie: int) -> list[str]:
    """Every contact the Cookie is in, as geom pair, distance and normal force."""

    geoms = set(env._cookie_collision_geoms[cookie])
    lines = []
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        if not {int(contact.geom1), int(contact.geom2)} & geoms:
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, index, force)
        names = [
            name_of(env, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for geom in (contact.geom1, contact.geom2)
        ]
        lines.append(
            f"      {names[0]:<30} <-> {names[1]:<30}"
            f" dist {contact.dist * 1000:+8.3f} mm   normal {force[0]:+8.4f} N"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/videos/two_cookies.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument(
        "--trace-slot",
        action="store_true",
        help="Print the call stack every time target_slot_index changes",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Also print every phase change and every placement descent step",
    )
    args = parser.parse_args()

    scene = libero_props_scene(
        task_name="a3_libero_panda_two_cookies",
        hold_steps=40,
        panda=True,
        cartesian=True,
        **LAYOUT,
    )
    env = scene.build_env(0.0, 0.0)
    env.use_fast_render()
    policy = scene.build_policy(env)
    expert = policy.expert
    observation, _ = env.reset(seed=args.seed, options=None)
    policy.reset()

    cookies = range(len(env._cookie_bodies))
    if args.trace_slot:
        # The slot index is a plain attribute with one writer in the source, and it is nonetheless
        # observed changing where that writer does not run.  A setter with a stack trace is the
        # only way to find out who else is writing it.
        import traceback

        class Spy(type(expert)):
            @property
            def target_slot_index(self):
                return getattr(self, "_slot_spy", None)

            @target_slot_index.setter
            def target_slot_index(self, value):
                if getattr(self, "_slot_spy", "missing") != value:
                    print(f"\n*** target_slot_index -> {value}")
                    traceback.print_stack()
                self._slot_spy = value

        expert.__class__ = Spy
    font = load_font(15)
    small = load_font(12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.output), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = VIEW + PANEL
    stream.height = VIEW + STRIP
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(args.crf)}

    # The guard the expert itself uses, repeated here so the clip can be read against it.
    limit_m = 0.006
    tilt_limit = np.cos(np.deg2rad(12))
    reported = False
    frames = 0
    previous_phase = None
    #: Largest the Cookie has ever sat off the tool's closing axis, so the clip and the trace agree
    #: about whether the grip was centred when it was taken or slid out of the jaws on the way.
    grip_report = [0.0]

    for step in range(1, args.steps + 1):
        action = policy.act(observation)
        observation, _, terminated, truncated, info = env.step(action)

        # --- the disturbance, on the step it happens ---------------------------------
        if not reported:
            for index, placed in expert._packed_positions.items():
                live = env.privileged_cookie_target_position(index)
                displacement = float(np.linalg.norm(live - placed))
                box_rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
                rotation = box_rotation.T @ env.data.xmat[env._cookie_bodies[index]].reshape(3, 3)
                tilt = float(np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), -1.0, 1.0))))
                if displacement > limit_m or abs(rotation[2, 2]) < tilt_limit:
                    reported = True
                    print(
                        f"\n*** step {step}: packed Cookie {index} is over the limit"
                        f" -- moved {displacement * 1000:.2f} mm (limit 6.00)"
                        f", tilt {tilt:.1f} deg (limit 12.0)"
                    )
                    print(f"    phase {expert.phase.name}, expert is carrying Cookie"
                          f" {expert._current_cookie()}")
                    print(f"    what Cookie {index} is in contact with at this instant:")
                    for line in describe_contacts(env, index) or ["      (nothing)"]:
                        print(line)
                    for other in cookies:
                        if other == index:
                            continue
                        distance = float(
                            np.linalg.norm(
                                env.data.xpos[env._cookie_bodies[other]]
                                - env.data.xpos[env._cookie_bodies[index]]
                            )
                        )
                        print(
                            f"    Cookie {other} is {distance * 1000:.2f} mm away"
                            f" (centres), tilt {cookie_state(env, other)[1]:.1f} deg"
                        )
                    break

        # --- how the Cookie sits in the jaws, as soon as it is being carried ------------
        if expert._current_cookie() is not None and expert.phase.name in (
            "VERIFY_GRASP", "LIFT", "VERIFY_LIFT", "MOVE_TO_SLOT", "DESCEND_TO_PLACE",
        ):
            tool_matrix = np.asarray(env.data.site_xmat[env.ik.left_site]).reshape(3, 3)
            offset = tool_matrix.T @ (
                np.asarray(env.data.xpos[env._cookie_bodies[expert._current_cookie()]])
                - np.asarray(env.data.site_xpos[env.ik.left_site])
            )
            current = np.linalg.norm(offset[1])
            # Only the records that matter: a centred grip is under 2 mm, and what the placement
            # needs to know is whether the Cookie was ever held near one of its ends instead.
            if current > 0.003 and current > grip_report[0]:
                grip_report[0] = current
                print(
                    f"    step {step:4d}  {expert.phase.name:16s}"
                    f"  Cookie {expert._current_cookie()} sits {current * 1000:6.2f} mm off the"
                    f" tool's y (its closing axis)"
                )

        # --- which Cookie, which slot, on every phase change ----------------------------
        if args.trace and expert.phase.name != previous_phase:
            print(
                f"    step {step:4d}  -> {expert.phase.name:16s}"
                f"  current_cookie_index={expert.current_cookie_index}"
                f"  target_slot_index={expert.target_slot_index}"
                f"  completed={expert.completed_cookie_indices}"
            )
            previous_phase = expert.phase.name

        # --- where the placed Cookie is being set down, in the box's own frame -----------
        if args.trace and expert.phase.name == "DESCEND_TO_PLACE" and step % 20 == 0:
            box_position = env.data.xpos[env._target_bin_body]
            box_matrix = env.data.xmat[env._target_bin_body].reshape(3, 3)
            carried = expert._current_cookie()
            tool_box = box_matrix.T @ (
                np.asarray(env.data.site_xpos[env.ik.left_site]) - box_position
            )
            cookie_box = env.privileged_cookie_target_position(carried)
            slot = expert.target_slot_index
            slot_xy = None if slot is None else np.asarray(env.TARGET_SLOTS_LOCAL[slot]) * 1000
            print(
                f"    step {step:4d}  setting Cookie {carried} down: tool at box"
                f" [{tool_box[0] * 1000:7.1f} {tool_box[1] * 1000:7.1f} {tool_box[2] * 1000:7.1f}]"
                f"   Cookie at box [{cookie_box[0] * 1000:7.1f} {cookie_box[1] * 1000:7.1f}]"
                f"   slot {slot} wants {None if slot_xy is None else np.round(slot_xy, 1)}"
            )

        view = env.render_view("front", height=VIEW, width=VIEW)
        wrist = env.render_view("left_wrist", height=PANEL, width=PANEL)
        canvas = Image.new("RGB", (VIEW + PANEL, VIEW + STRIP), (22, 22, 26))
        canvas.paste(Image.fromarray(view), (0, 0))
        canvas.paste(Image.fromarray(wrist), (VIEW, 0))
        draw = ImageDraw.Draw(canvas)

        slot = "-" if expert.target_slot_index is None else str(expert.target_slot_index)
        lines = [
            f"seed {args.seed}  step {step}   phase {expert.phase.name}",
            f"carrying Cookie {expert._current_cookie()}  ->  slot {slot}",
        ]
        for cookie in cookies:
            relative, tilt, heading = cookie_state(env, cookie)
            jaws = expert.env.privileged_left_finger_contacts(cookie)
            packed = expert._packed_positions.get(cookie)
            moved = "-" if packed is None else f"{np.linalg.norm(env.privileged_cookie_target_position(cookie) - packed) * 1000:.1f} mm"
            lines.append(
                f"c{cookie} box [{relative[0] * 1000:6.1f} {relative[1] * 1000:6.1f}]"
                f" tilt {tilt:4.1f}  yaw {heading:+7.1f}  jaws {jaws}  moved {moved}"
            )
        for order, text in enumerate(lines):
            draw.text((10, VIEW + 8 + order * 17), text, font=font if order < 2 else small, fill=(232, 232, 236))

        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        frames += 1

        if terminated or truncated:
            print(f"episode ended at step {step} (terminated={terminated} truncated={truncated})")
            break

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"\nwrote {args.output}  ({frames} frames at {args.fps} fps)")
    print(
        f"  the worst the carried Cookie ever sat off the tool's closing axis was"
        f" {grip_report[0] * 1000:.2f} mm"
    )
    print("  where each Cookie was aimed, and where it was recorded as dropped:")
    for cookie in cookies:
        for order, slot in enumerate(expert._slot_order):
            if order >= len(expert.completed_cookie_indices) + 1:
                break
        slot = expert._slot_order[cookie] if cookie < len(expert._slot_order) else None
        target = None if slot is None else env.TARGET_SLOTS_LOCAL[slot]
        packed = expert._packed_positions.get(cookie)
        print(
            f"    Cookie {cookie}: slot {slot}"
            f"  slot local {None if target is None else np.round(np.asarray(target) * 1000, 2)} mm"
            f"  recorded drop {None if packed is None else np.round(np.asarray(packed) * 1000, 2)} mm"
        )
    box_position = env.data.xpos[env._target_bin_body]
    box_matrix = env.data.xmat[env._target_bin_body].reshape(3, 3)
    drift = np.linalg.norm(box_position - expert._initial_target_position)
    turn = np.degrees(
        np.arccos(np.clip((np.trace(box_matrix.T @ expert._initial_target_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    print(
        f"  the target box itself has moved {drift * 1000:.2f} mm and turned {turn:.2f} deg"
        f" since the episode began"
    )
    # Where the expert thought it was putting the Cookie, against where the tool got to.  The
    # place waypoint is a joint vector, so it has to go through forward kinematics to become a
    # position the tool can be compared with.
    reference = mujoco.MjData(env.model)
    reference.qpos[:] = env.data.qpos
    measured = np.asarray(env.data.site_xpos[env.ik.left_site], dtype=np.float64)
    for stage in ("preplace", "place"):
        waypoint = expert._waypoints.get(stage)
        if waypoint is None:
            print(f"    {stage}: not planned")
            continue
        reference.qpos[expert._l_qpos] = waypoint
        mujoco.mj_kinematics(env.model, reference)
        commanded = np.asarray(reference.site_xpos[env.ik.left_site], dtype=np.float64)
        print(
            f"    {stage}: commanded tool {np.round(commanded * 1000, 2)} mm"
            f"   tool actually reached {np.round(measured * 1000, 2)} mm"
            f"   short by {np.linalg.norm(commanded - measured) * 1000:.2f} mm"
        )
    print(
        "  the Cookie the expert is carrying sits"
        f" {(np.asarray(env.data.xpos[env._cookie_bodies[1]]) - measured) @ np.asarray(env.data.xmat[env.ik.left_site]).reshape(3, 3)[:, 0] * 1000:.2f} mm"
        " along the tool's x, and"
        f" {(np.asarray(env.data.xpos[env._cookie_bodies[1]]) - measured) @ np.asarray(env.data.xmat[env.ik.left_site]).reshape(3, 3)[:, 1] * 1000:.2f} mm"
        " along its y"
    )
    print(f"final: phase={expert.phase.name} reason={expert.failure_reason}")
    print(f"  success={info.get('success')}  cookies placed={len(expert._packed_positions)}")
    for cookie in cookies:
        relative, tilt, heading = cookie_state(env, cookie)
        print(
            f"  Cookie {cookie}: box [{relative[0] * 1000:.2f}, {relative[1] * 1000:.2f},"
            f" {relative[2] * 1000:.2f}] mm  tilt {tilt:.1f} deg  yaw {heading:+.1f} deg"
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
