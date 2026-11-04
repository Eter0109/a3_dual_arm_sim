import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from a3_continuability.native_sensor_policy import NativeSensorPolicy


class SceneMemoryPolicyTests(unittest.TestCase):
    def reset_policy(self, **flags):
        policy = NativeSensorPolicy()
        policy.sensor = SimpleNamespace(protocol={})
        defaults = dict(A3_SENSOR_SCENE_MEMORY='0', A3_SENSOR_VISUAL_MEMORY='0',
                        A3_SENSOR_SETTLE_OBSERVE='0', A3_SENSOR_ACTIVE_RIGHT='0',
                        A3_SENSOR_RIGHT_EACH_BATCH='0')
        defaults.update(flags)
        with patch.dict(os.environ, defaults):
            policy.reset()
        return policy

    def test_new_mode_is_opt_in(self):
        policy = self.reset_policy()
        self.assertIsNone(policy.visual_memory)
        self.assertFalse(policy.settle_observe)
        self.assertFalse(policy.sensor.protocol['observation_state_memory'])

    def test_scene_memory_requires_real_observation_hold(self):
        policy = self.reset_policy(A3_SENSOR_SCENE_MEMORY='1')
        self.assertIs(policy.visual_memory.is_scene_state_memory, True)
        self.assertTrue(policy.settle_observe)
        self.assertTrue(policy.sensor.protocol['observation_state_memory'])
        self.assertFalse(policy.sensor.protocol['historical_frames_retimestamped'])

    def test_conflicting_legacy_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Choose scene-state memory'):
            self.reset_policy(A3_SENSOR_SCENE_MEMORY='1', A3_SENSOR_VISUAL_MEMORY='1')

    def test_active_right_arm_cannot_accidentally_enter_the_new_protocol(self):
        with self.assertRaisesRegex(ValueError, 'fixed right arm'):
            self.reset_policy(A3_SENSOR_SCENE_MEMORY='1', A3_SENSOR_ACTIVE_RIGHT='1')


if __name__ == '__main__':
    unittest.main()
