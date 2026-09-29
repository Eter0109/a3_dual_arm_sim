"""Episode-stable randomization, independent of expert/source-column selection."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class RandomizationProfile:
    source_bin_noise_m: float
    target_bin_noise_m: float
    target_bin_yaw_noise_rad: float
    camera_position_m: float = 0.0
    wrist_position_m: float = 0.0
    camera_rotation_deg: float = 0.0
    fovy_deg: float = 0.0
    light_fraction: float = 0.0
    palette: bool = False


PROFILES = {
    "basic": RandomizationProfile(0.010, 0.010, 0.030),
    "medium": RandomizationProfile(0.015, 0.015, 0.050, 0.004, 0.002, 0, 1, 0.10),
    "advanced": RandomizationProfile(0.020, 0.020, 0.080, 0.009, 0.003, 2, 2, 0.20, True),
}
COOKIE_PALETTE = {
    "golden": (0.90, 0.62, 0.12, 1),
    "cocoa": (0.38, 0.18, 0.08, 1),
    "matcha": (0.40, 0.60, 0.20, 1),
    "berry": (0.78, 0.30, 0.38, 1),
}


def profile_parameters(name):
    return asdict(PROFILES[name])


def choose_column(choice, seed):
    """Public columns 1..4 are sorted by source X; independent RNG stream."""
    if str(choice) == "random":
        group, offset = divmod(int(seed), 4)
        return (
            int(np.random.default_rng(np.random.SeedSequence([group, 7301])).permutation(4)[offset])
            + 1
        )
    if str(choice) not in ("1", "2", "3", "4"):
        raise ValueError("source column must be 1, 2, 3, 4 or random")
    return int(choice)


class AppearanceRandomizer:
    """Restore nominal arrays before every sample; no accumulation across reset."""

    FIELDS = ("cam_pos", "cam_quat", "cam_mode", "cam_fovy", "light_diffuse", "geom_rgba")

    def __init__(self, model):
        self.base = {key: getattr(model, key).copy() for key in self.FIELDS}
        self.cookie_geoms = [
            i
            for i in range(model.ngeom)
            if (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
            ).startswith("cookie_")
        ]

    def apply(self, model, data, name, seed):
        for key, value in self.base.items():
            getattr(model, key)[:] = value
        mujoco.mj_forward(model, data)
        profile = PROFILES[name]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7302]))
        cameras = {}
        for i in range(model.ncam):
            camera = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            wrist = "wrist" in (camera or "").lower()
            radius = profile.wrist_position_m if wrist else profile.camera_position_m
            offset = rng.uniform(-radius, radius, 3)
            angle = profile.camera_rotation_deg * (0.5 if wrist else 1)
            rotvec = np.deg2rad(rng.uniform(-angle, angle, 3))
            theta = np.linalg.norm(rotvec)
            delta = np.array([1.0, 0.0, 0.0, 0.0])
            if theta > 0:
                delta = np.r_[np.cos(theta / 2), rotvec / theta * np.sin(theta / 2)]
            nominal = self.base["cam_quat"][i].copy()
            if angle and model.cam_mode[i] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
                # Target-body cameras ignore cam_quat. Resolve their nominal
                # orientation into the mounting body's frame, then freeze it.
                body_rotation = data.xmat[model.cam_bodyid[i]].reshape(3, 3)
                local_rotation = body_rotation.T @ data.cam_xmat[i].reshape(3, 3)
                mujoco.mju_mat2Quat(nominal, local_rotation.ravel())
                model.cam_mode[i] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            model.cam_pos[i] += offset
            mujoco.mju_mulQuat(model.cam_quat[i], nominal, delta)
            model.cam_fovy[i] += rng.uniform(-profile.fovy_deg, profile.fovy_deg)
            cameras[camera] = {
                "position": model.cam_pos[i].tolist(),
                "quaternion": model.cam_quat[i].tolist(),
                "fovy": float(model.cam_fovy[i]),
                "mode": int(model.cam_mode[i]),
            }
        scales = rng.uniform(
            1 - profile.light_fraction, 1 + profile.light_fraction, (model.nlight, 1)
        )
        model.light_diffuse[:] = np.clip(self.base["light_diffuse"] * scales, 0, 1)
        color = None
        if profile.palette:
            color = list(COOKIE_PALETTE)[int(rng.integers(len(COOKIE_PALETTE)))]
            model.geom_rgba[self.cookie_geoms] = COOKIE_PALETTE[color]
        return {
            "profile": name,
            "parameters": asdict(profile),
            "cookie_color": color,
            "cameras": cameras,
            "light_diffuse": model.light_diffuse.tolist(),
        }
