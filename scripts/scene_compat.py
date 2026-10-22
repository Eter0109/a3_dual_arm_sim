"""Rebuild the training scene inside main's benchmark, at run time, and prove it.

Three things in main's benchmark differ from the scene the checkpoints were trained on, and
only the third is obvious:

  1. the `front` camera: pos [0.5, -0.18, 1.25] fovy 48, aimed at z 0.77, against
     [0.7076, -0.4232, 1.5748] fovy 54 aimed at z 0.81;
  2. both wrist cameras: mount offsets and look-at points differ (right wrist is even
     mirrored in z: +0.085 against -0.085);
  3. the two bins: the robot is at the same place, but `source_bin` sits 22.9 mm further in
     +y and `target_bin` 41.6 mm further in +y and 12.2 mm in +x.

The cameras come from the config (main builds them from `config.cameras`), so they are
overridden with a generated YAML; the bins and the 80 Cookies are placed after every reset,
because the environment re-applies its own layout each time.

`verify()` is the point of this file: it renders main's scene both ways and reports the mean
absolute pixel difference against this branch's own render, so "the observation now matches
training" is a measurement rather than a hope.

    MUJOCO_GL=egl PYTHONPATH=/tmp/a3-main/src python scene_compat.py --verify
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from pathlib import Path

import numpy as np

MAIN_ROOT = os.environ.get("A3_MAIN_ROOT", "/tmp/a3-main")
MAIN_SRC = f"{MAIN_ROOT}/src"
MAIN_CONFIG = f"{MAIN_ROOT}/configs/cookie_batch.yaml"
FINGERPRINT = os.environ.get(
    "A3_SCENE_FINGERPRINT",
    str(Path(__file__).resolve().parent / "scene_fingerprint.json"),
)
OVERRIDE_CONFIG = os.environ.get("A3_OVERRIDE_CONFIG", "/tmp/cookie_batch_training_scene.yaml")

#: Camera keys main's schema knows about.  This branch's config also carries
#: `reference_mirror` and `front_quat_wxyz`, which main has never heard of, and passing them
#: is an error rather than a no-op.
SHARED_CAMERA_KEYS = (
    "workspace_target_m",
    "front_position_m",
    "front_fovy_deg",
    "left_wrist_position_m",
    "left_wrist_target_m",
    "right_wrist_position_m",
    "right_wrist_target_m",
    "wrist_fovy_deg",
)


def load_fingerprint(path: str = FINGERPRINT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_override_config(fingerprint: dict) -> str:
    """main's benchmark config with this branch's cameras spliced in."""
    cameras = fingerprint["camera_config"]
    body = "\n".join(
        f"  {key}: {json.dumps(cameras[key])}"
        for key in SHARED_CAMERA_KEYS
        if key in cameras
    )
    text = Path(MAIN_CONFIG).read_text(encoding="utf-8")
    # Drop a `cameras:` block if one is ever added there, then append ours.
    text = re.sub(r"^cameras:\n(?:[ \t]+.*\n)*", "", text, flags=re.MULTILINE)
    Path(OVERRIDE_CONFIG).write_text(text.rstrip() + f"\n\ncameras:\n{body}\n",
                                     encoding="utf-8")
    return OVERRIDE_CONFIG


def _place_body(model, data, name: str, pose: dict) -> None:
    """Move one body to its training **world** pose, whichever way it is attached.

    Two traps, both of which cost a debugging round:

    * the fingerprint's `body_pos` is the XML-local value, not where the bin ends up --
      the environment re-places both bins at run time, so copying `body_pos` copies the
      default and changes nothing.  `xpos`/`xquat` is the pose that was actually rendered;
    * `source_bin` is welded to the world while `target_bin` carries a free joint, so one
      is moved through `body_pos`/`body_quat` and the other through `qpos`, and writing
      `body_pos` at a free body is silently ignored.

    Both bins hang off the world body, so their `xpos`/`xquat` can be written directly.
    """
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"main's scene has no body {name!r} to move")
    position = np.asarray(pose["xpos"], dtype=float)
    orientation = np.asarray(pose["xquat"], dtype=float)
    parent = int(model.body_parentid[body_id])
    if parent != 0:
        raise RuntimeError(
            f"body {name!r} hangs off "
            f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)!r}, not the world; "
            "its world pose cannot be written through body_pos"
        )
    joint_id = int(model.body_jntadr[body_id])
    if joint_id >= 0 and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address:address + 3] = position
        data.qpos[address + 3:address + 7] = orientation
        dof = int(model.jnt_dofadr[joint_id])
        data.qvel[dof:dof + 6] = 0.0
    else:
        model.body_pos[body_id] = position
        model.body_quat[body_id] = orientation


def _aim_wrist_cameras(model) -> None:
    """Give the wrist cameras back their training look-at.

    main resolves the wrist mount into a rigid `xyaxes` orientation; this branch aims the
    camera at a marker body (`mode="targetbody"`).  The mount point is the same either way,
    but the image comes out rolled relative to training, which is why the original
    comparison showed a 70/255 difference for an otherwise identical camera.
    """
    import mujoco

    target_mode = int(mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY)
    for side, prefix in (("left", "L"), ("right", "R")):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{side}_wrist")
        if camera_id < 0:
            raise RuntimeError(f"main's scene has no {side}_wrist camera")
        target_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_wrist_camera_target"
        )
        if target_id < 0:
            raise RuntimeError(f"main's scene has no {prefix}_wrist_camera_target body")
        model.cam_mode[camera_id] = target_mode
        model.cam_targetbodyid[camera_id] = target_id


def _set_joint_pose(model, data, joints: dict) -> None:
    """Put the arms back on the training start pose, by joint name.

    main's benchmark jitters the reset pose by up to 0.04 rad, and the two branches' RNG
    streams differ, so the arms start ~2 deg apart on every joint.  That is negligible in the
    front view and dominant in the wrist views: after matching everything *except* the pose,
    the wrist cameras still differed by 28/255 while the front camera was at 2.7/255.  Names,
    not indices -- the two models do not order the arm joints the same way.
    """
    import mujoco

    missing = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None or name not in joints:
            continue
        data.qpos[int(model.jnt_qposadr[jid])] = joints[name]
    for name in joints:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            missing.append(name)
    if missing:
        raise RuntimeError(f"main's model has no joint(s) {sorted(missing)[:4]}")


def apply(env, fingerprint: dict) -> None:
    """Place the arms, the bins and the Cookies where the training data had them.

    Called *after* `env.reset`, because the environment lays out its own arrangement there
    and would overwrite anything set before.  The Cookies go first: `set_cookie_pose`
    re-derives the layout, and would undo the bins if it ran afterwards.
    """
    import mujoco

    model, data = env.model, env.data
    if fingerprint.get("joints"):
        _set_joint_pose(model, data, fingerprint["joints"])
        mujoco.mj_forward(model, data)
    for index, position in enumerate(fingerprint["cookie_positions"]):
        env.set_cookie_pose(index, tuple(position))
    for name, pose in fingerprint["bodies"].items():
        _place_body(model, data, name, pose)
    _aim_wrist_cameras(model)
    mujoco.mj_forward(model, data)


def wrap_reset(env, fingerprint: dict) -> None:
    """Make every reset land on the training arrangement."""
    original = env.reset

    def reset(*args, **kwargs):
        out = original(*args, **kwargs)
        apply(env, fingerprint)
        return out

    env.reset = reset  # type: ignore[method-assign]


def verify(fingerprint: dict, out_dir: str = "/tmp/bench_shots") -> dict:
    """Render main's scene (patched) and this branch's scene, and compare."""
    import mujoco
    from PIL import Image

    sys.path.insert(0, MAIN_SRC)
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    frozen = dict(randomize_boxes=False, randomize_cookies=False)
    bench = make_cookie_benchmark(task="2x5", max_steps=1,
                                  config_path=write_override_config(fingerprint), **frozen)
    env = bench.create_env(render_cameras=True)
    env.model.vis.global_.offwidth = 640
    env.model.vis.global_.offheight = 640
    renderer = mujoco.Renderer(env.model, height=256, width=256)

    env.reset(seed=0)
    plain = {}
    for name in ("front", "left_wrist", "right_wrist"):
        renderer.update_scene(env.data, camera=env._camera_ids[name])
        plain[name] = np.ascontiguousarray(renderer.render()).astype(float)

    apply(env, fingerprint)
    patched = {}
    for name in ("front", "left_wrist", "right_wrist"):
        renderer.update_scene(env.data, camera=env._camera_ids[name])
        patched[name] = np.ascontiguousarray(renderer.render()).astype(float)
        Image.fromarray(np.asarray(patched[name], dtype=np.uint8)).save(
            f"{out_dir}/compat_{name}.png")

    # The reference: this branch's own render, at the same seed and frozen layout.
    import subprocess

    refs = {}
    for name in ("front", "left_wrist", "right_wrist"):
        path = f"{out_dir}/ours_{name}.png"
        if Path(path).exists():
            refs[name] = np.asarray(Image.open(path).convert("RGB"), dtype=float)

    report = {}
    for name in ("front", "left_wrist", "right_wrist"):
        if name not in refs:
            continue
        report[name] = {
            "mae_plain": float(np.abs(plain[name] - refs[name]).mean()),
            "mae_patched": float(np.abs(patched[name] - refs[name]).mean()),
        }
    renderer.close()
    env.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fingerprint", default=FINGERPRINT)
    ap.add_argument("--out-dir", default="/tmp/bench_shots")
    args = ap.parse_args()

    fingerprint = load_fingerprint(args.fingerprint)
    print(f"fingerprint: {args.fingerprint}  (randomize={fingerprint.get('randomize')})")
    if args.verify:
        report = verify(fingerprint, args.out_dir)
        for name, values in report.items():
            print(f"  {name:12s} MAE plain={values['mae_plain']:6.2f}  "
                  f"-> patched={values['mae_patched']:6.2f}")
    else:
        print(f"override config written: {write_override_config(fingerprint)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
