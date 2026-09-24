from __future__ import annotations

from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.smolvla_policy import SmolVLAPolicyPlugin


def test_reset_seeds_action_sampling_and_clears_chunk_queue():
    calls = []

    class TorchStub:
        def manual_seed(self, seed):
            calls.append(("seed", seed))

    class PolicyStub:
        def reset(self):
            calls.append(("reset", None))

    plugin = object.__new__(SmolVLAPolicyPlugin)
    plugin._torch = TorchStub()
    plugin._policy = PolicyStub()
    plugin.reset(EpisodeContext(seed=17, task="transfer cookies", action_mode="joint_position"))

    assert plugin.requires_camera_rendering
    assert calls == [("seed", 17), ("reset", None)]
