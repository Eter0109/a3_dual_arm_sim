import mujoco
import numpy as np
import pytest

from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
from a3_dual_arm_sim.workflows.policy_adapters import policy_needs_cameras


@pytest.mark.parametrize("task,wrong_score", [
    ("2x10-paired-columns", 0), ("2x10-four-source", 5),
])
def test_full_boxes_with_wrong_sources_do_not_terminate(task, wrong_score):
    benchmark = make_cookie_benchmark(task, strict_policy_completion=True)
    env = benchmark.create_env()
    try:
        env.reset(seed=0, options={"randomize_boxes": False, "randomize_cookies": False})
        xs = np.array(env.SOURCE_POSITIONS)[:, 0]
        cols = [np.flatnonzero(xs == x).tolist() for x in np.unique(xs)]
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        ordered = np.lexsort((slots[:, 1], slots[:, 0]))
        cookies = cols[1][:10] + cols[0][:10] if task == "2x10-paired-columns" else cols[0]
        for cookie, slot in zip(cookies, ordered, strict=True):
            env.set_cookie_pose(cookie, env.privileged_target_slot_world(slot, env._target_cookie_center_z))
        for _ in range(250):
            mujoco.mj_step(env.model, env.data)
        for _ in range(benchmark.success_hold_steps + 1):
            _, _, terminated, _, info = env.step(env.DEPLOYMENT_HOME)
            assert not terminated
        assert info["target_column_counts"] == (10, 10)
        assert info["cookies_in_target"] == wrong_score
        assert not info["success"]
        assert info["success_hold_count"] == 0
    finally:
        env.close()


def test_cartesian_plugin_gets_rgb_by_default():
    class Plugin:
        action_mode = "cartesian_delta"
    plugin = Plugin()
    assert policy_needs_cameras(plugin)
    plugin.requires_cameras = False
    assert not policy_needs_cameras(plugin)
