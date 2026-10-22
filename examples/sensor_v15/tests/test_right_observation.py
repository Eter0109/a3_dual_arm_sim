import unittest
import numpy as np
from a3_continuability.right_observation import joint_step,scan_encoder_state,scan_due


class RightObservationTests(unittest.TestCase):
    def test_velocity_is_bounded_and_direction_is_preserved(self):
        initial=np.array([-.9,-1.,1.2,1.,.3,-.3,1.5])
        delta=np.array([.3,-.1,.1,.0,.2,.0,-.1])
        command=joint_step(initial,initial+delta)
        self.assertLessEqual(np.max(abs(command-initial)),.018+1e-10)
        np.testing.assert_allclose(command-initial,delta*.06)
        self.assertEqual(initial[0],-.9)

    def test_arrival_does_not_overshoot_or_zero_other_joints(self):
        initial=np.array([-.9,-1.,1.2,1.,.3,-.3,1.5])
        target=initial.copy();target[0]+=.001
        np.testing.assert_array_equal(joint_step(initial,target),target)

    def test_missing_or_invalid_encoder_cannot_drive_the_scan(self):
        for initial in (np.zeros(8),np.array([np.nan]*7)):
            with self.assertRaises(ValueError):joint_step(initial,np.zeros(7))

    def test_left8_packet_needs_explicit_measured_right_encoders(self):
        left=np.arange(8,dtype=float);right=-left
        np.testing.assert_array_equal(scan_encoder_state(left,right),np.r_[left,right])
        with self.assertRaises(ValueError):scan_encoder_state(left,[])
        with self.assertRaises(ValueError):scan_encoder_state(np.zeros(16),right)

    def test_reobservation_only_at_safe_batch_boundary_and_once_per_batch(self):
        self.assertTrue(scan_due(0,'INITIALIZING',set()))
        self.assertTrue(scan_due(1,'LOCATE',{0},each_batch=True))
        self.assertFalse(scan_due(1,'LOCATE',{0}))
        self.assertFalse(scan_due(1,'LOCATE',{0,1},each_batch=True))
        for phase in ('PINCH','INSERT','HELD','VERIFY','DONE','FAILED'):
            self.assertFalse(scan_due(1,phase,{0},each_batch=True))


if __name__=='__main__':unittest.main()
