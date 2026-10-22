"""Execute the existing trained network through an actual left8-only API.

The historical benchmark adapter's act() requires 16D and appends a right hold.
Use its SAME processors/network/queue directly; right-arm holding belongs to
the robot driver and must not leak back into the learned observation.
"""
import numpy as np
from .sensor_boundary import policy_observation, SensorInputError


class Left8Executor:
    action_mode = "joint_position"

    def __init__(self, loaded_adapter):
        self.adapter = loaded_adapter
        self.config = loaded_adapter.config

    def reset(self, context=None):
        self.adapter.reset(context)

    def act(self, observation, task=""):
        import torch
        from pi05_adapter import observation_batch
        # Measured status was established when constructing ExecutionPacket.
        clean = policy_observation(observation, gripper_position_measured=True)
        batch = self.adapter.pre(observation_batch(clean, task))
        with torch.inference_mode():
            action = self.adapter.post(self.adapter.policy.select_action(batch))
        value = action.detach().float().cpu().numpy().reshape(-1)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise SensorInputError("Model produced an invalid left8 action")
        return value.copy()

    def close(self):
        self.adapter.close()
