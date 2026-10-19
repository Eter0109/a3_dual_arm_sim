"""Explicit training-camera profile, including target-body wrist semantics."""
from dataclasses import fields, replace
import json
from pathlib import Path


def apply_camera_reference(benchmark, path):
    from a3_dual_arm_sim.config import CameraConfig
    ref = json.loads(Path(path).read_text())
    camera = ref["cameras"]
    if camera.get("reference_mirror") or camera.get("front_quat_wxyz"):
        raise ValueError("Expected an unmirrored look-at camera profile")
    values = {f.name: camera[f.name] for f in fields(CameraConfig)}
    benchmark.config = replace(benchmark.config, cameras=CameraConfig(**values),
        image_height=ref["image_height"], image_width=ref["image_width"])
    original = benchmark.create_env

    def create_env(*, render_cameras=False):
        import mujoco
        from a3_dual_arm_sim.sim.randomization import AppearanceRandomizer
        env = original(render_cameras=render_cameras)
        for side, prefix in (("left", "L"), ("right", "R")):
            cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, side + "_wrist")
            bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "_wrist_camera_target")
            if min(cid, bid) < 0:
                env.close()
                raise ValueError("Camera target body missing")
            env.model.cam_mode[cid] = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
            env.model.cam_targetbodyid[cid] = bid
        mujoco.mj_forward(env.model, env.data)
        env._appearance_randomizer = AppearanceRandomizer(env.model)
        return env

    benchmark.create_env = create_env
