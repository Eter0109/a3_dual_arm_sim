"""Calibrated RGB-D observations for a repeatable batch skill.

The detector's inputs are images, depth, calibration and nominal workspace
geometry, never object poses or MuJoCo segmentation labels. Rendering is a
simulation sensor adapter and depth is an explicit extra sensing modality.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    fovy_degrees: float
    name: str = "camera"

    def world_points(self):
        h, w = self.depth.shape
        if self.rgb.shape != (h, w, 3):
            raise ValueError("RGB and depth must have the same resolution")
        focal = h / (2 * np.tan(np.deg2rad(self.fovy_degrees) / 2))
        v, u = np.mgrid[:h, :w]
        # MuJoCo camera axes: right, up, backward; rendered rows go down.
        ray = np.stack(((u + .5 - w / 2) / focal,
                        -(v + .5 - h / 2) / focal, -np.ones((h, w))), axis=-1)
        return (ray * self.depth[..., None]) @ self.camera_rotation.T + self.camera_position


class SimRGBDSensor:
    def __init__(self, env, resolution=512):
        import mujoco
        self.env = env
        # This changes buffer capacity, not the existing policy cameras' poses
        # or 256x256 outputs. The RGB-D localizer has a declared 512px sensor.
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), resolution)
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), resolution)
        self.renderer = mujoco.Renderer(env.model, height=resolution, width=resolution)

    def capture(self, names=("front", "left_wrist")):
        frames = []
        env = self.env
        for name in names:
            camera = env._camera_ids[name]
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(env.data, camera=camera)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(env.data, camera=camera)
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            frames.append(RGBDFrame(rgb, depth, env.data.cam_xpos[camera].copy(),
                env.data.cam_xmat[camera].reshape(3, 3).copy(),
                float(env.model.cam_fovy[camera]), name))
        return frames

    def close(self):
        self.renderer.close()


def save_frames(path, frames):
    from pathlib import Path
    import json
    import cv2
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    calibration = []
    for f in frames:
        cv2.imwrite(str(path / (f.name + ".png")), f.rgb[..., ::-1])
        np.save(path / (f.name + "_depth.npy"), f.depth, allow_pickle=False)
        calibration.append(dict(name=f.name, position=f.camera_position.tolist(),
            rotation=f.camera_rotation.tolist(), fovy_degrees=f.fovy_degrees))
    (path / "calibration.json").write_text(json.dumps(calibration, indent=2))


def load_frames(path):
    from pathlib import Path
    import json
    import cv2
    path = Path(path)
    return [RGBDFrame(cv2.imread(str(path / (c["name"] + ".png")))[..., ::-1],
        np.load(path / (c["name"] + "_depth.npy"), allow_pickle=False),
        np.asarray(c["position"]), np.asarray(c["rotation"]), c["fovy_degrees"], c["name"])
        for c in json.loads((path / "calibration.json").read_text())]


def colored_cookie_points(frames, *, lower, upper):
    """Color/height workspace filter; no simulation IDs or object masks."""
    import cv2
    points, colors = [], []
    for frame in frames:
        xyz = frame.world_points()
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        colored = (hsv[..., 1] > 70) & (hsv[..., 2] > 35)
        valid = colored & np.all(np.isfinite(xyz), axis=-1)
        valid &= np.all(xyz >= lower, axis=-1) & np.all(xyz <= upper, axis=-1)
        points.append(xyz[valid])
        colors.append(hsv[valid])
    return np.concatenate(points), np.concatenate(colors)


class LocalizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CookieEstimate:
    position: np.ndarray
    rotation: np.ndarray
    fit_error_m: float
    points: int


def choose_contiguous_batch(candidates, batch_size=5):
    """Find a usable run after fragmented leftovers; never bridge a large gap."""
    ordered = sorted(candidates, key=lambda item: item.position[1])
    for start in range(len(ordered) - batch_size + 1):
        batch = ordered[start:start + batch_size]
        gaps = np.diff([item.position[1] for item in batch])
        if np.all(gaps >= .003) and np.all(gaps <= .016):
            return batch, start
    raise LocalizationError("No continuous five-cookie group; leftovers require regrouping or another view")


def _components(points, radius):
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    unseen = set(range(len(points)))
    while unseen:
        pending = [unseen.pop()]
        group = []
        while pending:
            i = pending.pop()
            group.append(i)
            neighbours = unseen.intersection(tree.query_ball_point(points[i], radius))
            unseen.difference_update(neighbours)
            pending.extend(neighbours)
        yield np.asarray(group)


def _rotation(roll, yaw):
    c, s = np.cos(roll), np.sin(roll)
    cr, sr = np.cos(yaw), np.sin(yaw)
    return np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, c, -s], [0, s, c]])


def fit_cookie(points, half_size, floor_z, top_bevel=.0025, bottom_bevel=.001):
    """Fit the known beveled slab to observed surfaces, with a floor prior.

    Roll/yaw model is intentionally limited to source slabs; a fallen or poorly
    visible group is rejected instead of silently using simulator object poses.
    """
    from scipy.optimize import least_squares
    p = np.asarray(points, dtype=float)
    h = np.asarray(half_size)
    span = np.ptp(p, axis=0)
    if len(p) < 15 or span[0] < h[0] * .9:
        raise LocalizationError("Insufficient visible cookie surface")
    if len(p) > 500:
        p = p[np.linspace(0, len(p) - 1, 500).astype(int)]
    center_x = (np.quantile(p[:, 0], .01) + np.quantile(p[:, 0], .99)) / 2
    # The near side is visible in the existing front/wrist cameras.
    center_y = np.quantile(p[:, 1], .01) + h[1]
    initial = np.array([center_x, center_y, 0., 0., 0.])

    def decode(v):
        rotation = _rotation(v[2], v[3])
        center = np.array([v[0], v[1], floor_z + np.abs(rotation[2]) @ h + v[4]])
        return center, rotation

    def residual(v):
        center, rotation = decode(v)
        local = (p - center) @ rotation
        x, y, z = local.T
        planes = np.stack([np.abs(x) - h[0], np.abs(y) - h[1], np.abs(z) - h[2],
            (np.abs(y) + z - h[1] - h[2] + top_bevel) / np.sqrt(2),
            (np.abs(y) - z - h[1] - h[2] + bottom_bevel) / np.sqrt(2)])
        signed = planes.max(axis=0)
        return np.r_[signed, (v[0] - center_x) * 2, v[4] * 2]

    result = least_squares(residual, initial,
        bounds=(initial + [-.004, -.02, -.8, -.12, -.001],
                initial + [.004, .02, .8, .12, .004]),
        loss="soft_l1", f_scale=.0005, max_nfev=120,
        ftol=1e-8, xtol=1e-8, gtol=1e-8)
    center, rotation = decode(result.x)
    error = float(np.median(np.abs(residual(result.x)[:-2])))
    if not result.success or error > .0015 or abs(result.x[2]) > .7:
        raise LocalizationError(f"Cookie pose fit is unreliable: {error * 1000:.2f} mm")
    return CookieEstimate(center, rotation, error, len(points))


def localize_source_batch(frames, nominal_positions, half_size, *, source_column=0,
                          floor_z=.756, top_bevel=.0025, bottom_bevel=.001,
                          batch_size=5):
    """Locate the exposed five in a selected column from current RGB-D.

    Nominal positions specify the known tray layout, not randomized object
    states. Global tray translation is reestimated from observed cookie bounds;
    individual remaining slabs are clustered by color and reconstructed depth.
    No history of previously completed object IDs is needed for selection.
    """
    nominal = np.asarray(nominal_positions)
    half_size = np.asarray(half_size)
    lo = np.r_[nominal.min(axis=0) - half_size[:2] - .035, floor_z + .0003]
    hi = np.r_[nominal.max(axis=0) + half_size[:2] + .035, floor_z + .065]
    points, hsv = colored_cookie_points(frames, lower=lo, upper=hi)
    if len(points) < 100:
        raise LocalizationError("No reliable source-cookie cloud")
    lower = np.quantile(points[:, :2], .001, axis=0)
    upper = np.quantile(points[:, :2], .999, axis=0)
    measured_mid = (lower + upper) / 2
    nominal_mid = (nominal.min(axis=0) + nominal.max(axis=0)) / 2
    offset = measured_mid - nominal_mid
    columns = np.unique(nominal[:, 0])
    if not 0 <= source_column < len(columns):
        raise ValueError("Invalid source column")
    column_x = columns[source_column] + offset[0]
    selected = np.abs(points[:, 0] - column_x) < half_size[0] + .0002
    points, hsv = points[selected], hsv[selected]
    candidates = []
    for bright in (False, True):
        color_mask = (hsv[:, 0] >= 19) == bright
        p = points[color_mask]
        if not len(p):
            continue
        # Retain 3D samples for fitting, cluster in the thin y-z cross-section.
        bins = np.floor(p[:, 1:] / .0005).astype(int)
        _, inverse = np.unique(bins, axis=0, return_inverse=True)
        yz = np.zeros((inverse.max() + 1, 2))
        counts = np.bincount(inverse)
        np.add.at(yz, inverse, p[:, 1:])
        yz /= counts[:, None]
        for group in _components(yz, .0018):
            cloud = p[np.isin(inverse, group)]
            if len(cloud) < 15 or np.ptp(cloud[:, 0]) < half_size[0] * .9:
                continue
            try:
                candidates.append(fit_cookie(cloud, half_size, floor_z, top_bevel, bottom_bevel))
            except LocalizationError:
                continue
    candidates.sort(key=lambda item: item.position[1])
    if len(candidates) < batch_size:
        raise LocalizationError(f"Only {len(candidates)} reliable source slabs; need {batch_size}")
    batch, skipped = choose_contiguous_batch(candidates, batch_size)
    return batch, dict(source="calibrated RGB-D with slab geometry prior", points=len(points),
        detected_slabs=len(candidates), tray_offset_xy_m=offset.tolist(),
        skipped_front_estimates=skipped,
        residual_front_positions=[e.position.tolist() for e in candidates[:skipped]],
        max_fit_error_m=max(item.fit_error_m for item in batch))
