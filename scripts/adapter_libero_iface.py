"""π0.5's own LIBERO interface, driving the A3 arm.

Why not the manifest we already have
------------------------------------
`configs/policies/pi05_libero_zs_single.yaml` sends this checkpoint's seven outputs to the
arm as **joint angles**.  LIBERO's action is not joint angles: it is an end-effector
**delta**, and the two are different physical quantities in different units.  That run
measured `rate_limited 0.9975` with `motion_mean` pinned at the 0.05 rad/step cap -- the
model asked for roughly seven times the travel the robot permits, every step, and the arm
vibrated in place.  Nothing about it is a statement about the model.

This adapter gives the checkpoint the interface it was trained on, carried across to a
different robot:

    state  = [eef_pos(3), axis_angle(eef_quat)(3), gripper_qpos(2)]      8 wide
    action = [dx, dy, dz, drx, dry, drz, gripper]                        7 wide
    images = agentview + eye-in-hand, both flipped 180 degrees

and the manifest declares `action.space: cartesian_delta`, so the bridge solves the delta
into joint targets with the scene's IK rather than pretending it is a joint angle.

The three places this is deliberately approximate, and why
----------------------------------------------------------
An end-effector **delta** is the one part of this that transfers honestly: "move the tool
2 cm along +x and close the gripper" means the same thing on any arm, so the action needs
no adaptation beyond undoing the world frame.  The *state* does not transfer that
cleanly, and three choices had to be made:

* **Position is translated, not rotated.**  LIBERO's Panda hovers at (−0.211, −0.011,
  1.174) in its own base frame; this arm hovers at (0.10, 0.25, 0.92).  `state_offset`
  is that difference, so the policy sees its familiar hover pose at rest.  A pure
  translation is the right operation for this, because the action is a world-frame delta
  and a translation does not change world-frame directions.
* **Orientation is reported in LIBERO's end-effector frame.**  This has now been got wrong
  twice, in opposite directions, and both times the *numbers looked plausible*.

  First it was passed through unchanged -- this arm's frame has no relation to LIBERO's, so
  it read 120 degrees at rest.  Then it was expressed relative to this arm's own home, on
  the argument that "LIBERO's frame is world-aligned at hover, measured at 3.2 degrees, so
  the policy has only ever seen this input read about zero at rest".  That argument rested
  on reading LIBERO's quaternion as ``(w, x, y, z)``.  `LiberoProcessorStep._quat2axisangle`
  takes ``(x, y, z, w)`` -- its own docstring says so -- and read correctly LIBERO's
  **starting** orientation is ``(3.1409, -0.0028, -0.0877)``: **180.03 degrees about world
  +x**.  So the second version replaced a 120 degree error with a 180 degree one, and it
  looked like a fix, because both readings are "a rotation" and only one of them is what the
  policy was trained on.

  There is a second trap in the same three numbers: the position half is
  `site_xpos[gripper0_grip_site]` while the orientation half is `get_body_xquat` of
  `robot0_right_hand` -- a *different* body, 97 mm above the grip point.  One state vector,
  two frames, and neither of them named in the observation.

  What belongs here is the constant relating the two arms' end-effector frames, measured at
  their homes by `scripts/measure_state_convention.py`: 121.8 degrees, carrying this arm's
  tool axis (+y) onto LIBERO's approach axis (+z).  With it the eight-wide state equals
  LIBERO's at the start of an episode, position and orientation both.  See
  `libero_frame_axis_angle` and the `state_rotation_reference` config key.
* **The gripper is scaled into metres and mirrored.**  LIBERO's `gripper_qpos` is two
  finger joints in metres, mirrored about zero (`[+0.04, −0.04]` fully open); this arm
  reports one normalised opening in `[0, 1]`.  `gripper_scale` converts, because the
  checkpoint's own normaliser is in metres and sending a 0..1 value would be a 25x scale
  error in a dimension the model reads directly.

What a failure here can and cannot mean
---------------------------------------
The scene, the renderer, the arm and the state convention all change at once, so a failure
is **not attributable** to any one of them.  What the run does establish, if it fails, is
that the A3 arm in the A3 renderer is not something this checkpoint can drive zero-shot --
which is a floor, not an explanation.

Config (``--config key=value``)
-------------------------------
``ckpt``           checkpoint, default ``lerobot/pi05_libero_finetuned``
``state_offset``   ``x,y,z`` translation from A3's eef frame into LIBERO's
``gripper_scale``  metres per unit of A3 jaw opening, default 0.04
``state_rotation_reference``  ``home`` | ``absolute`` | ``w,x,y,z`` -- which frame the three
orientation numbers are measured from.  ``home`` is this arm's home inverted, i.e. the
behaviour every run before this key existed was measured under; four numbers are the
constant that makes the reading equal LIBERO's.
``device``         ``cuda`` (default) or ``cpu``
``cameras``        request slot names in LIBERO's order: agentview first, wrist second
"""

from __future__ import annotations

import os
import math
import sys
from typing import Any

#: LIBERO's order: `image` is the agentview, `image2` the eye-in-hand.  Reversing these
#: feeds the wrist view to the camera the model believes is across the room, and nothing
#: raises -- the shapes match and the policy simply sees a scene it cannot explain.
DEFAULT_CAMERAS = ("image", "image2")

#: `libero_spatial`'s hover pose in LIBERO's own base frame, measured by stepping the
#: benchmark with no-op actions.  Subtracted the other way it is added to this arm's.
LIBERO_HOVER_POSITION = (-0.2114, -0.0110, 1.1741)

#: This arm's end-effector orientation in the scene's own home pose, measured off the model
#: as an axis-angle: **120.0 degrees**, about the (-1, 1, -1) axis, which maps +x to -y, +y to
#: -z and +z to +x.
#:
#: It exists because the state's orientation needs the same treatment the state's position
#: already had.  LIBERO's end-effector frame is **world-aligned at hover** -- measured, its
#: axis-angle is 3.2 degrees and its axes come out as the world's to three decimals -- so the
#: policy has only ever seen this input read about zero at rest.  Ours reads 120.  Reporting
#: the orientation relative to this arm's own home pose, in world axes, is what makes the two
#: agree, and it is a constant rather than a calibration.
#:
#: The position was already mapped this way and nobody noticed the asymmetry: the offset
#: below takes (0.100, 0.250, 0.920) to (-0.211, -0.011, 1.174) exactly, so the policy was
#: being told it was standing where it expects and facing a direction it had never seen.
THIS_ARM_HOME_ROTATION = (-1.2092, 1.2092, -1.2092)  # axis-angle, radians


def quaternion_from_axis_angle(rotation: tuple) -> tuple[float, float, float, float]:
    """A (w, x, y, z) quaternion from an axis-angle triple."""

    angle = math.sqrt(sum(value * value for value in rotation))
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(angle / 2.0) / angle
    return (math.cos(angle / 2.0),
            rotation[0] * scale, rotation[1] * scale, rotation[2] * scale)


#: The same rotation as a quaternion, built once so the per-inference path is three multiplies.
HOME_QUATERNION = quaternion_from_axis_angle(THIS_ARM_HOME_ROTATION)


def quaternion_product(a: tuple, b: tuple) -> tuple[float, float, float, float]:
    """Hamilton product of two (w, x, y, z) quaternions."""

    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def libero_frame_axis_angle(quaternion: tuple, reference: tuple) -> list[float]:
    """An end-effector orientation in LIBERO's own frame: ``current * reference``.

    A *right* multiplication, so that a rotation applied in world axes is reported about
    those world axes; the left-multiplied form reports a turn about world z as a turn about
    whatever direction z happens to lie along in the reference frame, which here is 120
    degrees away.  Verified by feeding it a known 90 degree world rotation.

    ``reference`` is what the three numbers are measured *from*, and it is not a free
    choice.  It used to be this arm's own home quaternion inverted, on the argument that
    LIBERO's report is also about zero at its hover -- which rested on reading LIBERO's
    quaternion as ``(w, x, y, z)`` when `LiberoProcessorStep._quat2axisangle` documents and
    reads ``(x, y, z, w)``.  Read correctly, LIBERO's **starting** orientation is
    ``(3.1409, -0.0028, -0.0877)`` -- 180.03 degrees about world +x -- and this arm was
    reporting ``(0, 0, 0)``.  Half a turn, on the first input the policy sees.

    What belongs here is the constant relating the two arms' end-effector frames, measured
    at their homes by `scripts/measure_state_convention.py`: for this arm it is a 121.8
    degree rotation that carries the tool axis (+y here) onto LIBERO's approach axis (+z
    there), and applying it makes the eight-wide state equal LIBERO's at the start of an
    episode -- both halves, not just the position.
    """

    return axis_angle_from_wxyz(quaternion_product(quaternion, reference))


def rotation_reference(text: str) -> tuple[float, float, float, float]:
    """Parse ``state_rotation_reference``: ``home``, ``absolute``, or ``w,x,y,z``.

    ``home`` is this arm's home quaternion **inverted** -- what the previous version of this
    adapter always used, kept so manifests that do not name a reference keep the behaviour
    their numbers were measured under.  ``absolute`` reports the frame itself, which is what
    a robot whose end-effector frame already is LIBERO's wants.  Four numbers are a
    right-multiplied constant, which is the general form.
    """

    value = text.strip()
    if not value or value.lower() == "home":
        return (HOME_QUATERNION[0], -HOME_QUATERNION[1],
                -HOME_QUATERNION[2], -HOME_QUATERNION[3])
    if value.lower() == "absolute":
        return (1.0, 0.0, 0.0, 0.0)
    parts = tuple(float(v) for v in value.split(",") if v.strip())
    if len(parts) != 4:
        raise ValueError(
            "state_rotation_reference wants 'home', 'absolute' or a (w, x, y, z) "
            f"quaternion; got {text!r}"
        )
    return parts


#: Fully-open finger travel of LIBERO's Panda gripper, in metres, and its sign flip: the
#: two fingers are mirrored, so an open gripper reads `[+0.04, -0.04]`.
PANDA_FINGER_TRAVEL_M = 0.040


def parse_vector(text: str, expected: int, what: str) -> tuple[float, ...]:
    values = tuple(float(v) for v in text.split(",") if v.strip())
    if len(values) != expected:
        raise ValueError(f"{what} wants {expected} comma-separated numbers, got {len(values)}: {text!r}")
    return values


def axis_angle_from_wxyz(quaternion: tuple[float, float, float, float]) -> list[float]:
    """Rotation vector from a MuJoCo-order quaternion, matching LIBERO's own conversion.

    Reimplemented from `lerobot.processor.env_processor.LiberoProcessorStep._quat2axisangle`
    rather than called, because the order differs on the way in: MuJoCo hands us
    ``(w, x, y, z)`` and that function takes ``(x, y, z, w)``.  Swapping the wrong way
    produces a perfectly finite, perfectly wrong rotation, which the policy would read as
    a wrist pointing somewhere the arm has never been.
    """

    w, x, y, z = quaternion
    w = min(1.0, max(-1.0, w))
    denominator = math.sqrt(max(1.0 - w * w, 0.0))
    if denominator <= 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.acos(w)
    return [x / denominator * angle, y / denominator * angle, z / denominator * angle]


def load(config: dict[str, Any]) -> dict[str, Any]:
    """Build the checkpoint and its own processor pipeline.  Called once."""

    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    torch.set_grad_enabled(False)

    checkpoint = str(config.get("ckpt", "lerobot/pi05_libero_finetuned"))
    device = str(config.get("device", "cuda"))
    offset = parse_vector(str(config.get("state_offset", "-0.3114,-0.2610,0.2541")), 3, "state_offset")
    gripper_scale = float(config.get("gripper_scale", PANDA_FINGER_TRAVEL_M))
    reference = rotation_reference(str(config.get("state_rotation_reference", "home")))
    names = [n.strip() for n in str(config.get("cameras", ",".join(DEFAULT_CAMERAS))).split(",") if n.strip()]
    if len(names) != 2:
        raise ValueError(f"cameras wants two names (agentview, eye-in-hand), got {names}")

    policy = PI05Policy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return {
        "torch": torch,
        "np": np,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "device": device,
        "checkpoint": checkpoint,
        "offset": np.asarray(offset, dtype=np.float64),
        "gripper_scale": gripper_scale,
        "rotation_reference": reference,
        "cameras": tuple(names),
        "calls": 0,
    }


def _image_tensor(frame: Any, torch) -> Any:
    """One 256x256 uint8 HWC frame as the (1, 3, 256, 256) float tensor LIBERO's step makes.

    Flipped in **both** axes.  `LiberoProcessorStep` does exactly this and calls it "the
    HuggingFaceVLA/libero camera orientation convention"; it is part of the interface, not
    a rendering detail, and leaving it out feeds the policy an upside-down world.
    """

    import numpy as np

    tensor = torch.from_numpy(np.array(frame, dtype=np.uint8))
    # **An ablation knob, and nothing that ships sets it.**  `none` is the correct value;
    # `h`, `w` and `hw` re-introduce the historical errors so that "what did that fix
    # actually buy" can be measured instead of argued about.
    legacy = os.environ.get("A3_ADAPTER_FLIP", "none").strip().lower()
    if legacy in ("h", "hw"):
        tensor = torch.flip(tensor, dims=[0])
    if legacy in ("w", "hw"):
        tensor = torch.flip(tensor, dims=[1])
    # **No flip in the shipped configuration.  The environment already hands over the
    # policy's orientation.**
    #
    # This is the one place the whole chain can be checked, so here it is.  LIBERO's camera
    # buffer is bottom-up: `obs["agentview_image"]` equals `sim.render()` byte for byte
    # (measured, MSE 0.0), and `LiberoProcessorStep` -- the step the checkpoint was trained
    # and evaluated through -- then does `torch.flip(img, dims=[2, 3])`, both axes.  So the
    # image the policy is given is the *horizontal mirror* of the upright view:
    #
    #     LIBERO:  obs = U[::-1]  ->  flip both  ->  U[:, ::-1]
    #
    # `mujoco.Renderer` returns `U` (it flips the rows back itself, `np.flipud(out)` in its
    # own source), and `A3DualArmEnv._reference_orientation` mirrors it, so this env's
    # `observation.images.*` is already `U[:, ::-1]` -- the same thing.  Nothing further is
    # needed, and anything further is a bug.
    #
    # Two were.  This function used to flip rows, on the argument that LIBERO's buffer is
    # bottom-up and ours is not, so one flip made up the difference -- but the difference to
    # make up was the *mirror*, not the vertical flip, and `U[::-1]` is the one thing the
    # policy never sees.  Adding the mirror in the environment on top of that left the policy
    # being handed `U[::-1, ::-1]`: the world, rotated 180 degrees.  Measured against the
    # image LIBERO's own evaluation feeds, as-is beat every alternative; rendered, the robot
    # came into frame from the bottom.
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0)


def infer(model: dict[str, Any], request: Any) -> dict[str, Any]:
    """One inference: A3 observation in, a LIBERO-shaped delta chunk out."""

    import numpy as np

    torch = model["torch"]
    agentview, eye_in_hand = model["cameras"]

    batch = {
        f"observation.images.{agentview}": _image_tensor(request.image(agentview), torch),
        f"observation.images.{eye_in_hand}": _image_tensor(request.image(eye_in_hand), torch),
    }

    eef = np.asarray(request.vector("eef_pose"), dtype=np.float64)  # left arm pos + quat(wxyz)
    opening = float(np.asarray(request.vector("gripper"), dtype=np.float64).reshape(-1)[0])
    position = eef[:3] + model["offset"]
    # Expressed in LIBERO's frame, from whatever constant relates the two end-effector
    # frames.  See `libero_frame_axis_angle`.
    rotation = libero_frame_axis_angle(
        tuple(float(v) for v in eef[3:7]), model["rotation_reference"]
    )
    fingers = model["gripper_scale"] * opening
    state = np.r_[position, rotation, fingers, -fingers].astype(np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(model["device"])
    batch["task"] = [request.task or ""]

    with torch.inference_mode():
        chunk = model["policy"].predict_action_chunk(model["preprocessor"](batch))
        chunk = model["postprocessor"](chunk)
    chunk = chunk[0].detach().float().cpu().numpy()  # (50, 7)
    model["calls"] += 1

    # `robosuite`'s own OSC does this -- "Clip action to avoid out of bound errors" -- so
    # clipping here is part of the interface rather than a guard bolted on.  How far out of
    # range the *raw* output goes is the diagnostic that matters: the checkpoint's MEAN_STD
    # unnormaliser can only leave [-1, 1] when the state it was handed is outside the
    # statistics it was fitted on, so the overshoot measures how far this arm's state is
    # from LIBERO's.
    raw_low = float(chunk.min())
    raw_high = float(chunk.max())
    raw_first_row = chunk[0].copy()
    chunk = chunk.clip(-1.0, 1.0)
    model["raw_low"] = min(model.get("raw_low", raw_low), raw_low)
    model["raw_high"] = max(model.get("raw_high", raw_high), raw_high)
    if model["calls"] % 20 == 1:
        print(
            f"[libero-iface] call {model['calls']}: axis-angle "
            f"{['%.3f' % v for v in rotation]} position "
            f"{['%.3f' % v for v in position]} | raw out of range "
            f"[{model['raw_low']:.2f}, {model['raw_high']:.2f}]",
            file=sys.stderr,
            flush=True,
        )

    # The bridge wants 14 wide: two arms.  LIBERO drives one, so the right arm's six
    # motion terms stay zero -- and a zero delta is exactly what
    # `CartesianDeltaAdapter.convert` reads as "hold the pose you are in", so the right arm
    # is inert rather than following dynamic coupling.
    actions = np.zeros((chunk.shape[0], 14), dtype=np.float64)
    actions[:, :6] = chunk[:, :6]
    # Negated: LIBERO's gripper convention is +1 close / -1 open, this arm's is +1 open.
    # `CartesianDeltaAdapter` maps the value to a joint target as `(a + 1) / 2`, so passing
    # it through unchanged inverts every grasp.
    actions[:, 6] = -chunk[:, 6]

    return {
        "actions": actions,
        "space": "cartesian_delta",
        # Already physical: the checkpoint's own postprocessor inverted MEAN_STD with
        # LIBERO's statistics, and what is left is a normalised OSC delta, not radians.
        "units": "raw",
        "info": {
            "action_units": "normalised_osc_delta",
            "libero_calls": model["calls"],
            "libero_checkpoint": model["checkpoint"],
            "libero_state_offset": [round(v, 4) for v in model["offset"]],
            "libero_state_axis_angle": [round(v, 4) for v in rotation],
            "libero_position": [round(v, 4) for v in position],
            "libero_gripper_opening": round(opening, 4),
            "libero_chunk_available": int(chunk.shape[0]),
            "libero_raw_low": round(float(model.get("raw_low", 0.0)), 3),
            "libero_raw_high": round(float(model.get("raw_high", 0.0)), 3),
            # The model's own first row, before clipping.  Exposed because the bridge hands
            # the caller *joint targets* -- `gateway.act` returns the post-IK 16-wide
            # vector -- so a caller that wants to know how big a Cartesian step the model
            # asked for has no other way to see it, and reading `act`'s first three
            # entries as metres is a factor of about fifty out.
            "libero_command": [round(float(v), 4) for v in raw_first_row],
        },
    }
