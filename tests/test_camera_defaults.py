from a3_dual_arm_sim.config import CameraConfig, load_config
from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark


def test_front_camera_defaults_agree():
    expected = CameraConfig()
    for cameras in (load_config().cameras, CookieBatchBenchmark().config.cameras):
        assert cameras.front_position_m == expected.front_position_m
        assert cameras.workspace_target_m == expected.workspace_target_m
        assert cameras.front_fovy_deg == expected.front_fovy_deg
        assert cameras.left_wrist_position_m == (0.0, 0.045, 0.085)
        assert cameras.right_wrist_position_m == (0.0, -0.045, 0.085)
        assert cameras.wrist_fovy_deg == 70.0
