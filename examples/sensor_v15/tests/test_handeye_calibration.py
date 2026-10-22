import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from a3_continuability.handeye_calibration import transform,solve_handeye,observation_ready


class HandeyeTests(unittest.TestCase):
    def test_actual_settled_view_is_valid_but_safety_lift_or_motion_is_not(self):
        self.assertTrue(observation_ready(3,50,0,10,.98,.776))
        self.assertFalse(observation_ready(0,50,0,10,.98,.776))
        self.assertFalse(observation_ready(3,50,0,2,.98,.776))
        self.assertFalse(observation_ready(3,50,0,10,.80,.776))
    def fixture(self):
        x=transform(Rotation.from_rotvec([.02,-.01,.005]).as_matrix(),[.03,-.04,.06])
        nominal=x@transform(Rotation.from_rotvec([.006,-.004,.003]).as_matrix(),[.002,-.002,.001])
        target=transform(Rotation.from_rotvec([0,0,.03]).as_matrix(),[.1,.2,.77])
        robots=[transform(Rotation.from_rotvec(v).as_matrix(),[.1+.02*i,.2,.99])
                for i,v in enumerate(([0,0,0],[.23,0,0],[0,.23,0],[-.23,0,0],[0,-.23,0],[.14,.12,0]))]
        seen=[np.linalg.inv(r@x)@target for r in robots]
        return robots,seen,x,nominal

    def test_recovers_mount_and_rejects_parallel_motion(self):
        robots,seen,expected,nominal=self.fixture()
        actual,report=solve_handeye(robots,seen,nominal)
        np.testing.assert_allclose(actual,expected,atol=1e-8)
        self.assertLess(report['holdout_residual_m'],1e-8)
        with self.assertRaisesRegex(ValueError,'diversity'):
            solve_handeye([robots[0]]*6,seen,nominal)

    def test_wrong_heldout_observation_is_rejected(self):
        robots,seen,_,nominal=self.fixture()
        seen[-1]=seen[-1].copy();seen[-1][:3,3]+=[.02,0,0]
        with self.assertRaises(ValueError):solve_handeye(robots,seen,nominal)
