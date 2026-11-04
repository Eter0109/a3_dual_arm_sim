import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from a3_continuability.native_sensor_policy import NativeSensorPolicy


class CalibrationRetryPolicyTests(unittest.TestCase):
    def fixture(self,retry):
        policy=NativeSensorPolicy()
        packet=SimpleNamespace()
        policy.handeye_done=False
        policy.handeye=SimpleNamespace(steps=10,stable=0,done=True,failure=None,
            samples=[],motion=SimpleNamespace(),act=Mock(return_value=np.zeros(8)),
            retry_views=Mock(return_value=retry),calibration_trials=[])
        policy.sensor=SimpleNamespace(capture=Mock(return_value=packet),nominal_frames=[None,None],
            install_handeye=Mock(side_effect=ValueError('Insufficient measured rim views')))
        policy.spec={};policy.refine=True;policy.settle_observe=True;policy.placement_planning=True
        policy.guarded_release=False
        policy._event=Mock()
        obs={'observation.state':np.r_[np.zeros(8),np.ones(8)*.2]}
        return policy,packet,obs

    def test_real_retry_keeps_calibration_active_without_creating_failure_loop(self):
        policy,packet,obs=self.fixture(True)
        with patch('a3_continuability.native_sensor_policy.SensorBatchLoop') as loop:
            action=policy.act(obs)
        self.assertFalse(policy.handeye_done)
        self.assertFalse(policy.finished)
        loop.assert_not_called()
        policy.handeye.retry_views.assert_called_once_with(packet)
        np.testing.assert_array_equal(action[8:],obs['observation.state'][8:])

    def test_uncertified_or_exhausted_retry_remains_an_explicit_failure(self):
        policy,packet,obs=self.fixture(False)
        failed=SimpleNamespace(fail=Mock())
        with patch('a3_continuability.native_sensor_policy.SensorBatchLoop',return_value=failed):
            action=policy.act(obs)
        self.assertTrue(policy.handeye_done)
        failed.fail.assert_called_once_with('Calibration rejected: Insufficient measured rim views')
        np.testing.assert_array_equal(action[8:],obs['observation.state'][8:])
