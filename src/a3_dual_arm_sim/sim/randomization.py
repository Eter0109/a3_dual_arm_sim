"""Episode-stable randomization, independent of expert/source-column selection."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from a3_dual_arm_sim.paths import resource_root


@dataclass(frozen=True)
class RandomizationProfile:
    source_bin_noise_m: float
    target_bin_noise_m: float
    target_bin_yaw_noise_rad: float
    camera_position_m: float = 0.0
    wrist_position_m: float = 0.0
    camera_rotation_deg: float = 0.0
    wrist_rotation_deg: float = 0.0
    cookie_noise_m: float = 0.0003
    cookie_yaw_noise_rad: float = 0.015
    fovy_deg: float = 0.0
    light_fraction: float = 0.0
    palette: bool = False


DEFAULT_RANDOMIZATION_CONFIG = resource_root() / "configs" / "randomization.yaml"


def load_randomization_config(path=None, *, config=None):
    """Validate and normalize a snapshot; never mutate module-wide profiles."""
    if config is None:
        with Path(path or DEFAULT_RANDOMIZATION_CONFIG).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {
        "version",
        "profile",
        "source_column",
        "profiles",
        "colors",
    }:
        raise ValueError(
            "randomization config requires version, profile, source_column, profiles, colors"
        )
    if config["version"] != 1:
        raise ValueError("unsupported randomization config version")
    if not isinstance(config["profiles"], dict) or not config["profiles"]:
        raise ValueError("profiles must be a nonempty mapping")
    profiles = {}
    for name, values in config["profiles"].items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ValueError("each profile must be a named mapping")  # noqa: TRY004
        try:
            profile = RandomizationProfile(**values)
        except TypeError as exc:
            raise ValueError(f"invalid profile {name}: {exc}") from exc
        values = asdict(profile)
        for key, value in values.items():
            if key == "palette":
                if not isinstance(value, bool):
                    raise ValueError(f"{name}.palette must be true/false")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name}.{key} must be finite and nonnegative")
        if profile.light_fraction > 1 or profile.fovy_deg >= 45:
            raise ValueError(f"{name}: light_fraction must be <=1 and fovy_deg <45")
        profiles[name] = values
    colors = config["colors"]
    if not isinstance(colors, dict) or not colors:
        raise ValueError("colors must be a nonempty RGBA mapping")
    for name, rgba in colors.items():
        if not isinstance(name, str) or not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
            raise ValueError("colors must contain named four-channel RGBA lists")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in rgba
        ):
            raise ValueError(f"color {name} channels must be finite values in [0, 1]")
        if rgba[3] != 1:
            raise ValueError("cookie colors must be opaque (alpha=1)")
    if config["profile"] not in profiles:
        raise ValueError("selected profile is not defined in profiles")
    column = str(config["source_column"])
    if column not in ("1", "2", "3", "4", "random"):
        raise ValueError("source_column must be 1, 2, 3, 4 or random")
    return {
        "version": 1,
        "profile": config["profile"],
        "source_column": column,
        "profiles": profiles,
        "colors": {k: list(v) for k, v in colors.items()},
    }


def profile_parameters(name, config=None):
    return dict((config or load_randomization_config())["profiles"][name])


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

    def apply(self, model, data, name, seed, config=None):
        config = load_randomization_config(config=config)
        for key, value in self.base.items():
            getattr(model, key)[:] = value
        mujoco.mj_forward(model, data)
        profile = RandomizationProfile(**config["profiles"][name])
        palette = config["colors"]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7302]))
        cameras = {}
        for i in range(model.ncam):
            camera = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            wrist = "wrist" in (camera or "").lower()
            radius = profile.wrist_position_m if wrist else profile.camera_position_m
            offset = rng.uniform(-radius, radius, 3)
            angle = profile.wrist_rotation_deg if wrist else profile.camera_rotation_deg
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
            color = list(palette)[int(rng.integers(len(palette)))]
            model.geom_rgba[self.cookie_geoms] = palette[color]
        return {
            "profile": name,
            "configuration": config,
            "parameters": asdict(profile),
            "cookie_color": color,
            "cameras": cameras,
            "light_diffuse": model.light_diffuse.tolist(),
        }
