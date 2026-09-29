"""Default front camera is consistent across direct and YAML construction."""

from dataclasses import asdict

import numpy as np
import yaml

from a3_dual_arm_sim.core.paths import project_root
from a3_dual_arm_sim.envs.config import CameraConfig, load_config


def test_default_and_batch_share_front_camera_defaults():
    directory = project_root() / "configs/envs"
    assert load_config(directory / "default.yaml").cameras == CameraConfig()
    assert load_config(directory / "cookie_batch.yaml").cameras == CameraConfig()
    assert CameraConfig().front_position_m == (0.38, -0.02, 1.30)
    assert CameraConfig().workspace_target_m == (0.16, 0.24, 0.80)
    assert CameraConfig().front_fovy_deg == 42.0


def test_front_refit_preserves_wrist_defaults_and_resolution():
    camera = CameraConfig()
    assert camera.left_wrist_position_m == (0.0, 0.045, 0.085)
    assert camera.left_wrist_target_m == (0.0, 0.150, 0.0)
    assert camera.right_wrist_position_m == (0.0, -0.045, 0.085)
    assert camera.right_wrist_target_m == (0.0, -0.150, 0.0)
    assert camera.wrist_fovy_deg == 70.0
    config = load_config(project_root() / "configs/envs/cookie_batch.yaml")
    assert (config.image_width, config.image_height) == (256, 256)


def test_front_defaults_are_closer_and_more_top_down():
    camera = CameraConfig()
    old_offset = np.asarray((0.95, -0.55, 1.28)) - (0.15, 0.34, 0.80)
    new_offset = np.asarray(camera.front_position_m) - camera.workspace_target_m
    assert np.linalg.norm(new_offset) < 0.55 * np.linalg.norm(old_offset)
    assert np.degrees(np.arctan2(new_offset[2], np.linalg.norm(new_offset[:2]))) > 50


def test_explicit_front_camera_override_is_still_supported(tmp_path):
    source = tmp_path / "explicit_camera.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "cameras": {
                    "front_position_m": [0.95, -0.55, 1.28],
                    "workspace_target_m": [0.15, 0.34, 0.80],
                    "front_fovy_deg": 52.0,
                }
            }
        )
    )
    camera = load_config(source).cameras
    assert camera.front_position_m == (0.95, -0.55, 1.28)
    assert camera.front_fovy_deg == 52.0
    before, after = asdict(CameraConfig()), asdict(camera)
    for name in ("workspace_target_m", "front_position_m", "front_fovy_deg"):
        before.pop(name)
        after.pop(name)
    assert before == after
