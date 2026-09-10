from __future__ import annotations

import pytest

from a3_dual_arm_sim.expert import A3GraspExpert
from a3_dual_arm_sim.grasp import A3GraspEnv
from a3_dual_arm_sim.recording import MemoryRecorder
from a3_dual_arm_sim.runner import EpisodeRunner


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_privileged_grasp_expert_physically_grasps_and_lifts(seed: int) -> None:
    env = A3GraspEnv(render_cameras=False)
    expert = A3GraspExpert(env)
    recorder = MemoryRecorder()
    runner = EpisodeRunner(
        env,
        expert,
        task="pick up the red cube",
        recorder=recorder,
        save_failed_episodes=False,
    )
    try:
        result = runner.run(seed=seed, max_steps=450)
        assert result.success
        assert result.terminated
        assert env.is_grasped()
        assert env.target_position[2] - env.initial_object_z >= 0.08
        assert env.success_hold_count >= env.task_config.success_hold_steps
        linear_speed, angular_speed = env._object_speeds()
        assert linear_speed <= env.task_config.max_linear_speed_m_s
        assert angular_speed <= env.task_config.max_angular_speed_rad_s
        assert env._grasp_center_error() <= env.task_config.max_grasp_center_error_m
        assert not env._object_touches_table()
        assert len(recorder.episodes) == 1
        assert recorder.episodes[0][-1]["action"].shape == (16,)
    finally:
        runner.close()
