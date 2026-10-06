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
* **Orientation is passed through unchanged.**  The two grippers' mounting conventions
  differ -- ours reads 120 degrees at rest, LIBERO's about 180 -- and the difference is a
  property of how each gripper is attached, not of a shared world frame.  Rotating the
  whole state to hide it would make the state look right while the *images* still showed
  this robot, and the two would then disagree.  So the axis-angle this adapter produces is
  **out of the checkpoint's training distribution** and that is stated here rather than
  papered over.
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
``device``         ``cuda`` (default) or ``cpu``
``cameras``        request slot names in LIBERO's order: agentview first, wrist second
"""

from __future__ import annotations

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
    tensor = torch.flip(tensor, dims=[0, 1])  # H and W, still HWC
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
    rotation = axis_angle_from_wxyz(tuple(eef[3:7]))
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
