from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from a3_dual_arm_sim.contracts import EpisodeContext
from a3_dual_arm_sim.workflows.shared_smolvla_grid import InferenceStream


class TinyPolicy:
    """A stochastic chunk policy: sufficient to expose cross-client queue/RNG leakage."""

    def __init__(self):
        self._device = torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self._policy = SimpleNamespace(_queues={"action": deque()}, weight=self.weight)
        self.inference_seed = 0

    def reset(self, context):
        torch.manual_seed(self.inference_seed)
        self._policy._queues = {"action": deque()}

    def act(self, observation, task):
        actions = self._policy._queues["action"]
        if not actions:
            for noise in torch.randn(3):
                actions.append(float(noise) + observation["state"])
        return np.asarray([actions.popleft()])


def test_interleaved_inference_matches_serial_across_multiple_chunk_refills():
    base = TinyPolicy()
    reference = []
    for lane in range(2):
        base.inference_seed = lane + 7
        base.reset(EpisodeContext(seed=lane + 7, task=str(lane), action_mode="joint_position"))
        reference.append([base.act({"state": lane + step}, str(lane)) for step in range(9)])
    rng_before = torch.get_rng_state().clone()
    clients = [InferenceStream(base, lane + 7, str(lane)) for lane in range(2)]
    actual = [[], []]
    for step in range(9):
        for lane in range(2):
            actual[lane].append(clients[lane].act({"state": lane + step}, str(lane)))
    assert np.array_equal(actual, reference)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert clients[0].policy._policy._queues is not clients[1].policy._policy._queues
    assert clients[0].policy._policy.weight.data_ptr() == base._policy.weight.data_ptr()
    assert clients[1].policy._policy.weight.data_ptr() == base._policy.weight.data_ptr()
