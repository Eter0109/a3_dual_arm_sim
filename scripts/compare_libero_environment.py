"""What LIBERO's test environment is, and how far ours is from it, parameter by parameter.

"Match LIBERO's environment" is only a plan if each parameter can be named, read off the
compiled model, and compared -- otherwise it is a direction of travel.  This prints that
comparison: the same dump taken from LIBERO's own scene and from ours, side by side, with the
difference, so every axis is either driven to zero or marked as one that cannot be.

Both sides are read from a raw ``mujoco.MjModel`` after compilation, because the declared
value and the effective one differ in exactly the places that matter.  LIBERO's
``_setup_camera`` sets a camera's position and quaternion and says nothing about ``fovy``;
robosuite's arena sets the lights as XML elements that the compiler resolves into vectors;
and the physics options are robosuite defaults that no LIBERO file mentions.  Reading the
source answers "what was asked for" and the model answers "what happened".

    MUJOCO_GL=egl python scripts/compare_libero_environment.py

Nothing here needs a policy, so it costs no GPU memory and no model download.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Before robosuite or mujoco are imported: the offscreen context has to be chosen first.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]

#: Bodies whose pose is worth comparing, by the substring that finds them in each scene.
PROPS = ("bowl", "plate")

#: The cameras that matter, paired by *role*.  The names differ -- LIBERO's scene is
#: robosuite's, ours is this repository's -- and a comparison keyed on names reports that
#: there is nothing in common and stops, which is exactly what the first version of this
#: script did.  What has to match is the pose and the projection, not the label.
CAMERA_ROLES = (
    ("agentview", "front", "the view of the table"),
    ("robot0_eye_in_hand", "left_wrist", "the view from the wrist"),
)


def role_pairs(left: dict, right: dict) -> list[tuple[str, dict, dict]]:
    """The role-matched camera pairs that exist on both sides."""

    pairs = []
    for libero_name, ours_name, role in CAMERA_ROLES:
        if libero_name in left and ours_name in right:
            pairs.append((role, left[libero_name], right[ours_name]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--scene", default="a3_libero_props_matched")
    parser.add_argument("--libero-root", type=Path, default=None,
                        help="Override the LIBERO asset root shown in the report")
    parser.add_argument("--render", type=Path, default=None,
                        help="Write a side-by-side image of what each policy camera sees, "
                             "showing LIBERO's raw render, LIBERO's as the policy receives "
                             "it, ours, and ours flipped 180 degrees.  The flip is the one "
                             "interface choice that no scalar in the report can check.")
    return parser.parse_args()


def _name(model, kind, index: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, kind, index) or f"<{index}>"


def dump_cameras(model) -> dict[str, dict]:
    """Every camera's pose and projection, as compiled."""

    import mujoco

    out = {}
    for index in range(model.ncam):
        out[_name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)] = {
            "pos": np.asarray(model.cam_pos[index], dtype=np.float64),
            "quat": np.asarray(model.cam_quat[index], dtype=np.float64),
            "fovy": float(model.cam_fovy[index]),
            "mode": int(model.cam_mode[index]),
        }
    return out


def dump_lights(model) -> dict[str, object]:
    """The scene's lights, plus the renderer's own headlight.

    The headlight is not a ``light`` element -- MuJoCo adds it from ``vis.headlight`` -- and
    it is the one axis of the rendering that measured free to remove in LIBERO, which is
    worth knowing before spending effort matching it.
    """

    head = model.vis.headlight
    return {
        "count": int(model.nlight),
        "pos": np.asarray(model.light_pos, dtype=np.float64).copy(),
        "dir": np.asarray(model.light_dir, dtype=np.float64).copy(),
        "diffuse": np.asarray(model.light_diffuse, dtype=np.float64).copy(),
        "specular": np.asarray(model.light_specular, dtype=np.float64).copy(),
        "ambient": np.asarray(model.light_ambient, dtype=np.float64).copy(),
        "castshadow": np.asarray(model.light_castshadow, dtype=np.float64).copy(),
        "headlight": {
            "diffuse": np.asarray(head.diffuse, dtype=np.float64).copy(),
            "ambient": np.asarray(head.ambient, dtype=np.float64).copy(),
            "specular": np.asarray(head.specular, dtype=np.float64).copy(),
            "active": int(head.active),
        },
    }


def dump_physics(model) -> dict[str, float]:
    """The solver settings, which no LIBERO file declares and every result depends on."""

    opt = model.opt
    return {
        "timestep": float(opt.timestep),
        "control_hz": 1.0 / float(opt.timestep),
        "integrator": int(opt.integrator),
        "solver": int(opt.solver),
        "iterations": int(opt.iterations),
        "tolerance": float(opt.tolerance),
        "impratio": float(opt.impratio),
        "cone": int(opt.cone),
        "jacobian": int(opt.jacobian),
        "noslip_iterations": int(opt.noslip_iterations),
        "gravity": np.asarray(opt.gravity, dtype=np.float64).copy(),
    }


def dump_bodies(model, data) -> dict[str, np.ndarray]:
    """World positions of the props and of the root body, for the layout comparison."""

    import mujoco

    out = {}
    for index in range(model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if any(key in name.lower() for key in PROPS):
            out[name] = np.asarray(data.xpos[index], dtype=np.float64).copy()
    return out


def dump_scene_bounds(model, data) -> dict[str, float]:
    """A few scalars that describe the workspace's scale and height."""

    import mujoco

    geoms = []
    for index in range(model.ngeom):
        if int(model.geom_type[index]) in (
            int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            geoms.append(np.asarray(data.geom_xpos[index], dtype=np.float64))
    if not geoms:
        return {}
    stacked = np.vstack(geoms)
    return {
        "geom_origin_min": stacked.min(axis=0),
        "geom_origin_max": stacked.max(axis=0),
    }


def _focus(model, data) -> np.ndarray:
    """The object the table camera exists to look at, by its measured centre.

    Measured through ``prop_frame`` rather than read off ``xpos``, because LIBERO nests each
    object's mesh inside a wrapper and the two bodies disagree by 100 mm -- the same trap
    that made the trace probe measure to a point below the bowl.  The wrapper is the right
    body to *name* and the wrong one to measure from, so the caller passes whichever it has
    and this walks down to the geometry.
    """

    import mujoco

    from a3_dual_arm_sim.props import prop_frame

    for index in range(model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if "bowl" in name.lower():
            centre, _, _ = prop_frame(model, data, index)
            return centre
    raise SystemExit("no body matching 'bowl' in this scene")


def camera_geometry(camera: dict, focus: np.ndarray, table_up: np.ndarray) -> dict:
    """A camera's pose broken into quantities that mean something on either scene.

    World coordinates cannot be compared between these two, because the robot is different
    and the table is in a different place.  What *can* be compared is where the camera sits
    **relative to the object it is there to look at**: how far away, at what elevation above
    the table plane, on what bearing, and rolled how much about its own optical axis.  Those
    four numbers plus ``fovy`` are the whole of a camera as far as an image is concerned, and
    they are what "restore the camera" has to mean.
    """

    from scipy.spatial.transform import Rotation

    position = np.asarray(camera["pos"], dtype=np.float64)
    # MuJoCo camera frames look down their own -z, with +y up in the image.
    rotation = Rotation.from_quat([camera["quat"][1], camera["quat"][2],
                                   camera["quat"][3], camera["quat"][0]])
    forward = rotation.apply([0.0, 0.0, -1.0])
    up = rotation.apply([0.0, 1.0, 0.0])

    to_focus = np.asarray(focus, dtype=np.float64) - position
    distance = float(np.linalg.norm(to_focus))
    horizontal = to_focus - float(np.dot(to_focus, table_up)) * table_up
    flat = float(np.linalg.norm(horizontal))
    elevation = float(np.degrees(np.arctan2(float(np.dot(to_focus, table_up)), flat)))

    # Bearing is measured in the table plane against the focus's own x axis, so it is the
    # camera that moves and not the object's position that is being compared.
    basis_x = np.array([1.0, 0.0, 0.0]) - float(table_up[0]) * table_up
    basis_x /= max(np.linalg.norm(basis_x), 1e-12)
    basis_y = np.cross(table_up, basis_x)
    azimuth = float(np.degrees(np.arctan2(float(np.dot(horizontal, basis_y)),
                                          float(np.dot(horizontal, basis_x)))))

    # Roll: how far the image's up vector is tilted from the table's up, about the optical
    # axis.  This is the one that a 180-degree flip in the interface cannot fix for one
    # camera and leave alone for another, and it is invisible in every scalar above.
    image_up = up - float(np.dot(up, forward)) * forward
    image_up /= max(np.linalg.norm(image_up), 1e-12)
    projected = table_up - float(np.dot(table_up, forward)) * forward
    projected /= max(np.linalg.norm(projected), 1e-12)
    roll = float(np.degrees(np.arctan2(
        float(np.dot(np.cross(projected, image_up), forward)),
        float(np.clip(np.dot(projected, image_up), -1.0, 1.0)))))

    return {
        "distance to the focus": distance,
        "elevation above the table": elevation,
        "bearing in the table plane": azimuth,
        "roll about the optical axis": roll,
        "fovy": camera["fovy"],
    }


def render_comparison(pixels: dict, ours_env, destination: Path) -> None:
    """What each pipeline actually hands the policy, side by side.

    The LIBERO side comes from the environment's own observation (``pixels/image`` and
    ``pixels/image2``) rather than from ``sim.render``, because the two are not the same
    picture: robosuite flips the renderer's output before the policy sees it, and rendering
    the camera directly would compare a quantity no policy has ever been given.  This is also
    the check on the interface's 180-degree flip, which is a vertical *and* horizontal flip --
    if the two pipelines differ by a vertical flip alone, then flipping both is wrong by a
    horizontal one, and no number in the parameter report above would say so.
    """

    from PIL import Image, ImageDraw

    roles = (("image", "front", "the view of the table"),
             ("image2", "left_wrist", "the view from the wrist"))
    tiles = []
    for libero_key, ours_name, role in roles:
        theirs = np.asarray(pixels[libero_key])[0]
        ours = np.asarray(ours_env.render_view(ours_name, height=theirs.shape[0],
                                               width=theirs.shape[1]))
        delta = float(np.abs(theirs.astype(int) - ours.astype(int)).mean())
        flipped = float(np.abs(theirs.astype(int)
                               - ours[::-1, ::-1].astype(int)).mean())
        print(f"  {role}: mean |LIBERO - ours| = {delta:.1f}, "
              f"|LIBERO - ours flipped 180| = {flipped:.1f}"
              + ("   -> flipping is closer" if flipped < delta else "   -> no flip is closer"))
        tiles.append((role, {
            "LIBERO, as the policy gets it": theirs,
            f"ours ({ours_name})": ours,
            "ours, flipped 180": np.ascontiguousarray(ours[::-1, ::-1]),
            "LIBERO, flipped 180": np.ascontiguousarray(theirs[::-1, ::-1]),
        }))

    cell = pixels["image"].shape[1] + 22
    sheet = Image.new("RGB", (cell * 4, cell * len(tiles)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for row, (role, images) in enumerate(tiles):
        draw.text((4, row * cell + 6), role, fill=(160, 255, 180))
        for column, (label, image) in enumerate(images.items()):
            sheet.paste(Image.fromarray(np.asarray(image, dtype=np.uint8)),
                        (column * cell, row * cell + 22))
            draw.text((column * cell + 4, row * cell + 6), label, fill=(255, 255, 160))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(f"  wrote {destination}")


def report_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def compare(label: str, a, b, tolerance: float = 1e-9) -> None:
    """One parameter, both sides, and whether they differ."""

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        left = np.atleast_1d(np.asarray(a, dtype=np.float64))
        right = np.atleast_1d(np.asarray(b, dtype=np.float64))
        same = left.shape == right.shape and np.allclose(left, right, atol=tolerance)
        print(f"  {label:<26} LIBERO {np.array2string(left, precision=4, floatmode='fixed')}")
        print(f"  {'':<26} ours   {np.array2string(right, precision=4, floatmode='fixed')}"
              f"   {'SAME' if same else 'DIFFERENT'}")
    else:
        same = abs(float(a) - float(b)) <= tolerance
        print(f"  {label:<26} LIBERO {a!r:<28} ours {b!r:<28} "
              f"{'SAME' if same else 'DIFFERENT'}")


def libero_model(args):
    """One compiled LIBERO environment, without touching a policy."""

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env

    cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id])
    # One environment, not the ten a task list would build: each carries its own EGL context.
    envs = make_env(cfg, n_envs=1)
    vec = envs[args.suite][args.task_id]
    sub = vec.envs[0]
    inner = getattr(sub._env, "env", sub._env)  # noqa: SLF001
    inner.hard_reset = False
    # A freshly constructed robosuite scene has the arm at home and every object at its
    # spawn pose.  One reset is what puts it in the state an episode actually starts from --
    # and it is also the only way to see the pixels the policy is handed, which robosuite
    # flips on the way out and `sim.render` does not.
    observation, _ = vec.reset(seed=[0])
    return inner.sim.model._model, inner.sim.data._data, inner, observation["pixels"]  # noqa: SLF001


def our_model(args):
    from a3_dual_arm_sim.collection import scene_by_name

    env = scene_by_name(args.scene).build_env(0.0, 0.0)
    # Not fast-rendered: the render flags are part of what is being compared.
    env.reset(seed=0, options=None)
    return env.model, env.data, env


def main() -> int:
    args = parse_args()

    print(f"LIBERO {args.suite} task {args.task_id}   vs   {args.scene}")
    libero_m, libero_d, libero_env, libero_pixels = libero_model(args)
    try:
        ours_m, ours_d, ours_env = our_model(args)
        try:
            report_section("cameras, paired by role")
            left, right = dump_cameras(libero_m), dump_cameras(ours_m)
            print(f"  LIBERO has {sorted(left)}")
            print(f"  ours has   {sorted(right)}\n")
            for role, lcam, rcam in role_pairs(left, right):
                print(f"  --- {role} ---")
                for key in ("pos", "quat", "fovy", "mode"):
                    compare(key, lcam[key], rcam[key])

            # The same two pairs again, reduced to the five numbers that describe a camera
            # independently of where either scene happens to be.
            report_section("camera pose relative to the object it looks at")
            libero_focus = _focus(libero_m, libero_d)
            ours_focus = _focus(ours_m, ours_d)
            up = np.array([0.0, 0.0, 1.0])
            print(f"  focus: LIBERO {np.round(libero_focus, 4)}   ours {np.round(ours_focus, 4)}")
            for role, lcam, rcam in role_pairs(left, right):
                print(f"  --- {role} ---")
                one = camera_geometry(lcam, libero_focus, up)
                two = camera_geometry(rcam, ours_focus, up)
                for key in one:
                    unit = "deg" if "fovy" in key or "deg" in key or key in (
                        "elevation above the table", "bearing in the table plane",
                        "roll about the optical axis") else "m"
                    print(f"      {key:<28} LIBERO {one[key]:8.2f} {unit:<4} "
                          f"ours {two[key]:8.2f} {unit:<4} "
                          f"delta {two[key] - one[key]:+8.2f}")

            report_section("lights")
            left, right = dump_lights(libero_m), dump_lights(ours_m)
            compare("count", left["count"], right["count"])
            for key in ("pos", "dir", "diffuse", "ambient", "castshadow"):
                compare(key, left[key], right[key])
            for key in ("diffuse", "ambient", "active"):
                compare(f"headlight {key}", left["headlight"][key], right["headlight"][key])

            report_section("physics")
            left, right = dump_physics(libero_m), dump_physics(ours_m)
            for key in left:
                compare(key, left[key], right[key])

            report_section("props and layout (world metres)")
            left, right = dump_bodies(libero_m, libero_d), dump_bodies(ours_m, ours_d)
            print(f"  LIBERO bodies matching {PROPS}: {sorted(left)}")
            print(f"  ours   bodies matching {PROPS}: {sorted(right)}")
            for name, position in sorted(left.items()):
                print(f"  LIBERO {name:<38} {np.round(position, 4)}")
            for name, position in sorted(right.items()):
                print(f"  ours   {name:<38} {np.round(position, 4)}")

            report_section("renderer globals")
            for label, model in (("LIBERO", libero_m), ("ours", ours_m)):
                flags = model.vis.global_
                print(f"  {label:<7} offwidth={flags.offwidth} offheight={flags.offheight} "
                      f"free_camera_fovy={flags.fovy}")
            print("  shadows are not a renderer global in MuJoCo 3.3 -- they are per-light,"
                  " and the `castshadow` row above is where they live.")

            if args.render is not None:
                report_section("what the two pipelines hand the policy")
                render_comparison(libero_pixels, ours_env, args.render)
        finally:
            ours_env.close()
    finally:
        libero_env.close()

    print()
    print("Every line marked DIFFERENT is either drivable to zero or is a property of the")
    print("robot and the table, which cannot be: the arm is ours and its base is where it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
