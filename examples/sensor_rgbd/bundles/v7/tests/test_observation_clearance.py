import unittest
import numpy as np
from a3_continuability.observation_clearance import observation_lift


class ObservationClearanceTests(unittest.TestCase):
    def test_retreated_gripper_does_not_request_another_high_pose(self):
        p=np.array([.13,.10,.99])
        target,delta=observation_lift(p,[.13,.10,.89],.795)
        np.testing.assert_array_equal(target,p)
        self.assertEqual(delta,0.)

    def test_low_fingers_are_lifted_even_when_eef_already_clears_rim(self):
        p=np.array([.13,.10,.90])
        target,delta=observation_lift(p,[.13,.10,.80],.795)
        self.assertAlmostEqual(delta,.060)
        np.testing.assert_allclose(target,[.13,.10,.96])
        np.testing.assert_array_equal(p,[.13,.10,.90])

    def test_missing_or_nonfinite_sensor_geometry_is_rejected(self):
        for pad in (None,[0,0,np.nan],[0,0]):
            with self.assertRaises(ValueError):observation_lift([0,0,1],pad,.8)


if __name__=='__main__':unittest.main()
