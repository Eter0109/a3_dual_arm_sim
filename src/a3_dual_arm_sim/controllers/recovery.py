"""Explicit human takeover; recovery examples are never mixed into training automatically."""

from a3_dual_arm_sim.controllers.teleop import KeyboardTeleopPolicy


class RecoveryPolicy:
    action_mode = "joint_position"

    def __init__(self, policy):
        self.policy = policy
        self.teleop = KeyboardTeleopPolicy()
        self.manual = False
        self.events = []
        self.step = 0
        self._was_manual = False

    def reset(self, context):
        self.policy.reset(context)
        self.teleop.reset(context)
        self.context = context
        self.manual = self._was_manual = False
        self.events = []
        self.step = 0

    def handle_key(self, key):
        if key in (ord("t"), ord("T")):
            self.manual = not self.manual
        else:
            self.teleop.handle_key(key)

    @property
    def stop_requested(self):
        return self.teleop.stop_requested

    @property
    def emergency_requested(self):
        return self.teleop.emergency_requested

    @property
    def discard_requested(self):
        return self.teleop.discard_requested

    def act(self, observation, task):
        manual = self.manual
        if manual != self._was_manual:
            self.events.append({"step": self.step, "controller": "human" if manual else "model"})
            if manual:
                self.teleop.reset(self.context)
                self.teleop.grippers = [float(observation["observation.state"][i]) for i in (7, 15)]
            else:
                self.policy.reset(self.context)
            self._was_manual = manual
        self.step += 1
        return (self.teleop if manual else self.policy).act(observation, task)

    def close(self):
        self.policy.close()
