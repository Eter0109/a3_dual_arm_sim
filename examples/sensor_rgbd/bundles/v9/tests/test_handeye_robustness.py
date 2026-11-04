"""Independent mount certification under quantized/noisy visual measurements."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from a3_continuability.handeye_calibration import transform,solve_handeye,HandEyeScan


class IndependentlyValidatedNominalTests(unittest.TestCase):
    def fixture(self,seed=0):
        actual=transform(Rotation.from_rotvec([.02,-.01,.005]).as_matrix(),[.03,-.04,.06])
        target=transform(Rotation.from_rotvec([0,0,.03]).as_matrix(),[.1,.2,.77])
        robots=[transform(Rotation.from_rotvec(v).as_matrix(),[.1+.02*i,.2,.99])
            for i,v in enumerate(([0,0,0],[.23,0,0],[0,.23,0],[-.23,0,0],[0,-.23,0],[.14,.12,0],[.1,-.14,0]))]
        seen=[np.linalg.inv(robot@actual)@target for robot in robots]
        rng=np.random.default_rng(seed)
        # Independently detected rim poses have small rotation quantization;
        # camera mount remains unchanged. It should not be distorted to fit it.
        for measured in seen:
            measured[:3,:3]=measured[:3,:3]@Rotation.from_rotvec(rng.normal(0,.0003,3)).as_matrix()
            measured[:3,3]+=rng.normal(0,.00008,3)
        return robots,seen,actual

    def test_small_visual_noise_can_preserve_an_independently_valid_mount(self):
        used_nominal=0
        for seed in range(12):
            robots,seen,nominal=self.fixture(seed)
            actual,report=solve_handeye(robots,seen,nominal)
            self.assertLessEqual(report['position_residual_m'],.002)
            self.assertLessEqual(report['holdout_residual_m'],.003)
            if report['selected_mount']=='independently_validated_nominal':
                np.testing.assert_array_equal(actual,nominal)
                used_nominal+=1
        self.assertGreater(used_nominal,0)

    def test_independent_bad_view_remains_a_rejection(self):
        robots,seen,nominal=self.fixture()
        seen[-1][:3,3]+=[.015,0,0]
        with self.assertRaises(ValueError):solve_handeye(robots,seen,nominal)

    def test_wrong_nominal_does_not_replace_a_correct_handeye_fit(self):
        robots,seen,actual=self.fixture()
        nominal=actual@transform(Rotation.from_rotvec([.01,0,0]).as_matrix(),[.004,-.002,.001])
        solved,report=solve_handeye(robots,seen,nominal)
        self.assertEqual(report['selected_mount'],'park_handeye')
        self.assertLess(np.linalg.norm(solved[:3,3]-actual[:3,3]),.002)


class BoundedAdditionalObservationTests(unittest.TestCase):
    def fixture(self):
        class Motion:
            reachable=True
            def pose(self,state):return np.array([.1,.2,1.1]),np.eye(3)
            def pose_error(self,state,position,rotation):return (0.,0.) if self.reachable else (.1,0.)
            def command(self,*args,**kwargs):return np.zeros(8),True
            def hold(self,*args,**kwargs):return np.zeros(8)
        motion=Motion();scan=HandEyeScan(motion,{})
        scan.rim_z=.776
        scan.observation_basis=(np.array([.1,.2,.776]),np.eye(3),np.eye(3),np.zeros(3))
        robots,targets,mount=IndependentlyValidatedNominalTests().fixture()
        scan.samples=[(robot,SimpleNamespace(world_from_optical=robot@mount,index=i,timestamp=1.+i))
            for i,robot in enumerate(robots)]
        packet=SimpleNamespace(observation={'observation.state':np.zeros(16)},
            pad_midpoint_world=np.array([.1,.2,1.1]))
        return scan,motion,packet,targets

    def fail_from_insufficient_visible_rims(self,scan,targets):
        def observed(frame,*args,**kwargs):
            if frame.index>=4:raise ValueError('Calibration requires four observed rim sides')
            return targets[frame.index],0.
        with patch('a3_continuability.handeye_calibration.observe_rim',side_effect=observed):
            with self.assertRaisesRegex(ValueError,'held-out'):scan.solve(.75)

    def test_no_retry_without_measured_insufficient_view_diagnosis(self):
        scan,motion,packet,targets=self.fixture()
        self.assertFalse(scan.retry_views(packet))
        self.assertFalse(scan.retry_used)

    def test_one_real_retry_preserves_timestamps_and_total_steps(self):
        scan,motion,packet,targets=self.fixture()
        self.fail_from_insufficient_visible_rims(scan,targets)
        original=list(scan.samples);scan.steps=260;scan.done=True
        self.assertTrue(scan.retry_views(packet))
        self.assertEqual(scan.steps,260)
        self.assertEqual(scan.max_steps,550)
        self.assertEqual(len(scan.goals),6)
        self.assertFalse(scan.done)
        self.assertEqual([frame.timestamp for _,frame in scan.samples],[frame.timestamp for _,frame in original])
        # The near box edge moves above the gripper mask rather than filling
        # it from a dimension prior. A physical camera movement is required.
        np.testing.assert_allclose(scan.goals[1][0],[.1,.165,1.136])
        self.assertFalse(scan.retry_views(packet))

    def test_unreachable_retry_does_not_create_a_partial_motion_plan(self):
        scan,motion,packet,targets=self.fixture()
        self.fail_from_insufficient_visible_rims(scan,targets)
        motion.reachable=False
        self.assertFalse(scan.retry_views(packet))
        self.assertIsNone(scan.goals)
        self.assertFalse(scan.retry_views(packet))

    def test_fewer_observations_alone_cannot_trigger_early_switch(self):
        scan,motion,packet,targets=self.fixture();scan.samples=scan.samples[:2]
        with patch('a3_continuability.handeye_calibration.observe_rim') as observed:
            self.assertFalse(scan.early_visibility_retry_needed(.75))
        observed.assert_not_called()
        self.assertFalse(scan.retry_eligible)
        self.assertFalse(scan.initial_visibility_checked)

    def prepare_third_capture(self,scan):
        nominal=scan.samples[2][1]
        scan.samples=scan.samples[:2]
        scan.goals=[(np.array([.1,.2,1.1]),np.eye(3))]*8
        scan.index=3;scan.stable=7;scan.steps=100
        return nominal

    def test_three_genuinely_unsupported_views_switch_without_renewing_time(self):
        scan,motion,packet,targets=self.fixture()
        third=self.prepare_third_capture(scan)
        with patch('a3_continuability.handeye_calibration.observe_rim',side_effect=ValueError('missing near edge')):
            scan.act(packet,third)
        self.assertTrue(scan.retry_used)
        self.assertEqual(scan.index,0)
        self.assertEqual(scan.steps,101)
        self.assertEqual(len(scan.samples),3)
        self.assertEqual([frame.timestamp for _,frame in scan.samples],[1.,2.,3.])
        self.assertEqual(len(scan.goals),6)

    def test_one_supported_rim_keeps_the_original_observation_plan(self):
        scan,motion,packet,targets=self.fixture()
        third=self.prepare_third_capture(scan);original_goals=scan.goals
        def observed(frame,*args,**kwargs):
            if frame.index==0 and abs(kwargs['focal_scale']-.975)<1e-8:return targets[0],0.
            raise ValueError('not a complete rim at this scale')
        with patch('a3_continuability.handeye_calibration.observe_rim',side_effect=observed):
            scan.act(packet,third)
        self.assertFalse(scan.retry_used)
        self.assertEqual(scan.index,4)
        self.assertIs(scan.goals,original_goals)
        self.assertEqual(scan.max_steps,400)
