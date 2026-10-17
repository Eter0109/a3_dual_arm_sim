"""Reproduce main's benchmark scene inside this branch, to the pixel and to the millimetre.

Why this exists: the 20k checkpoint scores 5.35 ± on this branch's own scene but 0/20 on main's
benchmark as it ships -- and the failure is a millimetre-scale grasp miss (`cookies_in_source`
never leaves 80), not a hover or an unclosed gripper.  Main's arrangement differs from this
branch's in four ways at once, each measured:

* bins at [0.175, 0.315, 0] and [0.095, 0.100, 0.749] against this branch's [0.1696, 0.3379] and
  [0.1072, 0.1416, 0.7487] -- 23 mm and 42 mm;
* the arm's home pose, by up to 0.037 rad per joint (29 mm at the flange);
* the wrist cameras: main resolves the mount into a **fixed** orientation (`xyaxes` in
  `sim/robot_builder.py`) while this branch aims a `targetbody` marker, and that roll difference
  alone was worth 70/255 of mean absolute pixel error;
* the front camera: [0.5, -0.18, 1.25] at fovy 48 against [0.7076, -0.4232, 1.5748] at 54.

So the three things that have to be pinned at run time are the two bins, the 80 Cookies, the 18
arm joints **by name** (this branch orders qpos differently: our L arm sits at 9-17, main's at
0-8), and every camera's mode/pos/orientation.  A config file cannot express the last one, which
is why this is code and not YAML.

`apply_scene` is called *after* `env.reset`, because the environment re-derives its layout there,
and the optional noises exist so the same scene can be used to *collect* training data rather
than only to evaluate: pin the arrangement, then jitter it slightly so the policy does not learn
one exact pose.

    from a3_dual_arm_sim.main_alignment import apply_scene, load_scene, scene_fingerprint
    scene = load_scene("/tmp/main_scene.json")
    env.reset(seed=0)
    apply_scene(env, scene, rng=np.random.default_rng(0), cookie_noise_m=0.002)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

#: Cameras main's benchmark exposes, in its own order.
CAMERAS = ("front", "left_wrist", "right_wrist")


def load_scene(path: str | Path) -> dict[str, Any]:
    """Read the dump written by `dump_main_scene.py`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(matrix, dtype=np.float64).reshape(9))
    return quaternion


def _yaw_from_quat(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _place_source_bin(env, position: np.ndarray, quaternion: np.ndarray) -> None:
    """This branch's `source_bin` is a **mocap** body, so `body_pos` would move nothing."""
    body = env._source_bin_body  # noqa: SLF001 - documented attribute of the environment
    mocap_id = int(env.model.body_mocapid[body])
    if mocap_id < 0:
        raise RuntimeError("source_bin is not a mocap body; this branch's layout changed")
    env.data.mocap_pos[mocap_id] = np.asarray(position, dtype=float)
    env.data.mocap_quat[mocap_id] = np.asarray(quaternion, dtype=float)


def _place_target_bin(env, position: np.ndarray, quaternion: np.ndarray) -> None:
    """The target box carries a free joint, so its pose lives in `qpos`."""
    body = env._target_bin_body  # noqa: SLF001
    joint = int(env.model.body_jntadr[body])
    if joint < 0:
        raise RuntimeError("target_bin has no joint; this branch's layout changed")
    if int(env.model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("target_bin is not a free body; this branch's layout changed")
    address = int(env.model.jnt_qposadr[joint])
    env.data.qpos[address : address + 3] = np.asarray(position, dtype=float)
    env.data.qpos[address + 3 : address + 7] = np.asarray(quaternion, dtype=float)
    dof = int(env.model.jnt_dofadr[joint])
    env.data.qvel[dof : dof + 6] = 0.0


def _set_joints(env, joints: dict[str, float], rng: np.random.Generator | None,
                noise_rad: float) -> None:
    model = env.model
    for name, value in joints.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise RuntimeError(f"this branch has no joint {name!r}")
        jitter = 0.0 if (rng is None or noise_rad <= 0.0) else float(rng.normal(0.0, noise_rad))
        env.data.qpos[int(model.jnt_qposadr[joint])] = value + jitter


def _set_camera(env, name: str, camera: dict[str, Any]) -> None:
    """Rebuild one camera exactly: parent, mount, fovy, and a *fixed* world orientation.

    Main's wrist cameras are `mode=fixed` with an `xyaxes` orientation; this branch aims a
    `targetbody` marker instead.  Copying `cam_pos` alone therefore keeps the position and loses
    the roll (70/255 MAE).  The world matrix is the ground truth, so the local quaternion is
    derived from it: `local_R = parent_R^T * world_R`.
    """
    model = env.model
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        raise RuntimeError(f"this branch has no camera {name!r}")
    parent = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, camera["parent"])
    if parent < 0:
        raise RuntimeError(f"this branch has no body {camera['parent']!r} to mount {name!r} on")
    model.cam_bodyid[camera_id] = parent
    model.cam_mode[camera_id] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
    model.cam_pos[camera_id] = np.asarray(camera["local_pos"], dtype=float)
    model.cam_fovy[camera_id] = float(camera["fovy"])
    # The parent's world orientation is already correct at this point (bins, Cookies and joints
    # were applied above and the scene is settled), so the local quaternion can be derived now.
    mujoco.mj_forward(model, env.data)
    parent_rotation = np.asarray(env.data.xmat[parent], dtype=float).reshape(3, 3)
    world_rotation = np.asarray(camera["world_xmat"], dtype=float).reshape(3, 3)
    model.cam_quat[camera_id] = _quat_from_matrix(parent_rotation.T @ world_rotation)


def apply_scene(
    env,
    scene: dict[str, Any],
    *,
    rng: np.random.Generator | None = None,
    bin_noise_m: float = 0.0,
    cookie_noise_m: float = 0.0,
    bin_yaw_noise_rad: float = 0.0,
    pose_noise_rad: float = 0.0,
) -> dict[str, Any]:
    """Put `env` on main's arrangement, optionally with a small perturbation.

    All noises default to zero: that is the evaluation scene, and it is what the pixel gate checks.
    The collection run passes non-zero values so the policy sees more than one exact pose.
    """
    def perturb(scale: float, dims: int = 3) -> np.ndarray:
        if rng is None or scale <= 0.0:
            return np.zeros(dims)
        return np.asarray(rng.normal(0.0, scale, size=dims))

    source = scene["bodies"]["source_bin"]
    target = scene["bodies"]["target_bin"]
    _place_source_bin(
        env,
        np.asarray(source["world_pos"], dtype=float) + perturb(bin_noise_m),
        _quat_from_matrix(_yaw_matrix(_yaw_from_quat(source["world_quat"]))),  # keep axis-aligned
    )
    target_position = np.asarray(target["world_pos"], dtype=float) + perturb(bin_noise_m)
    target_yaw = _yaw_from_quat(target["world_quat"])
    if rng is not None and bin_yaw_noise_rad > 0.0:
        target_yaw += float(rng.normal(0.0, bin_yaw_noise_rad))
    _place_target_bin(env, target_position, _quat_from_matrix(_yaw_matrix(target_yaw)))
    mujoco.mj_forward(env.model, env.data)

    for index, position in enumerate(scene["cookie_positions"]):
        offset = perturb(cookie_noise_m)
        env.set_cookie_pose(index, tuple(np.asarray(position, dtype=float) + offset))
    _set_joints(env, scene["joints"], rng, pose_noise_rad)
    mujoco.mj_forward(env.model, env.data)

    for name in CAMERAS:
        _set_camera(env, name, scene["cameras"][name])
    mujoco.mj_forward(env.model, env.data)
    return {"cookies": len(scene["cookie_positions"]), "joints": len(scene["joints"])}


def _yaw_matrix(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def post_reset_hook(
    path: str | Path,
    *,
    bin_noise_m: float = 0.0,
    cookie_noise_m: float = 0.0,
    bin_yaw_noise_rad: float = 0.0,
    pose_noise_rad: float = 0.0,
):
    """Build an `EpisodeRunner.post_reset` callback for collection.

    One scene for every episode, plus a perturbation drawn from that episode's own seed -- so a
    dataset collected this way is reproducible from its seeds and states its perturbation in
    exactly the place a reader would look for it.
    """
    scene = load_scene(path)

    def hook(env, seed: int) -> None:
        apply_scene(
            env,
            scene,
            rng=np.random.default_rng(seed),
            bin_noise_m=bin_noise_m,
            cookie_noise_m=cookie_noise_m,
            bin_yaw_noise_rad=bin_yaw_noise_rad,
            pose_noise_rad=pose_noise_rad,
        )

    return hook


def scene_fingerprint(env) -> dict[str, Any]:
    """The same fields as main's dump, from this branch's live environment, for diffing."""
    model, data = env.model, env.data
    cameras = {}
    for name in CAMERAS:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        cameras[name] = {
            "parent": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.cam_bodyid[camera_id])),
            "local_pos": np.asarray(model.cam_pos[camera_id], dtype=float).tolist(),
            "local_quat": np.asarray(model.cam_quat[camera_id], dtype=float).tolist(),
            "fovy": float(model.cam_fovy[camera_id]),
            "mode": int(model.cam_mode[camera_id]),
            "world_pos": np.asarray(data.cam_xpos[camera_id], dtype=float).tolist(),
            "world_xmat": np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(9).tolist(),
        }
    bodies = {}
    for name in ("source_bin", "target_bin"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        bodies[name] = {
            "world_pos": np.asarray(data.xpos[body], dtype=float).tolist(),
            "world_quat": np.asarray(data.xquat[body], dtype=float).tolist(),
        }
    joints = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name is None or int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        joints[name] = float(data.qpos[int(model.jnt_qposadr[joint])])
    return {
        "cameras": cameras,
        "bodies": bodies,
        "joints": joints,
        "cookie_positions": np.asarray(env.cookie_positions, dtype=float).tolist(),
    }


def diff_scene(reference: dict[str, Any], candidate: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Compare two fingerprints.  Returns `(field, unit, value)` for every non-zero difference."""
    report: list[tuple[str, str, float]] = []
    for name, ref in reference["bodies"].items():
        got = candidate["bodies"][name]["world_pos"]
        delta = float(np.abs(np.asarray(got) - np.asarray(ref["world_pos"])).max())
        report.append((f"body {name} world_pos", "mm", 1000.0 * delta))
    ref_cookies = np.asarray(reference["cookie_positions"], dtype=float)
    got_cookies = np.asarray(candidate["cookie_positions"], dtype=float)
    report.append(
        ("cookies (max of 80)", "mm", 1000.0 * float(np.abs(got_cookies - ref_cookies).max()))
        if got_cookies.shape == ref_cookies.shape
        else ("cookies (shape)", "count", float(abs(got_cookies.size - ref_cookies.size)))
    )
    joint_delta = [
        abs(candidate["joints"][name] - value)
        for name, value in reference["joints"].items()
        if name in candidate["joints"]
    ]
    report.append(("joints (max of 18)", "rad", float(max(joint_delta)) if joint_delta else 0.0))
    for name, ref in reference["cameras"].items():
        got = candidate["cameras"][name]
        report.append((f"camera {name} world_pos", "mm",
                       1000.0 * float(np.abs(np.asarray(got["world_pos"]) - np.asarray(ref["world_pos"])).max())))
        # World orientation, not the local quaternion: main's front camera is `targetbody` and this
        # branch pins it `fixed`, so the local quats differ by construction while the camera that
        # actually renders is identical.  The world matrix is the quantity that has to match.
        report.append((f"camera {name} world_xmat", "abs",
                       float(np.abs(np.asarray(got["world_xmat"]) - np.asarray(ref["world_xmat"])).max())))
        report.append((f"camera {name} fovy", "deg", abs(got["fovy"] - ref["fovy"])))
    return report
