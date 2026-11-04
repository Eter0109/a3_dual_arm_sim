import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from a3_continuability.batch_vision import LocalizationError
from a3_continuability.held_batch_vision import placement_goal
from a3_continuability.placement_planning import (
    PlacementCandidate,placement_candidates,placement_bounds,sample_segment,check_placement_chain,choose_placement)
from a3_continuability.native_sensor_policy import NativeSensorPolicy
from a3_continuability.sensor_batch_loop import SensorBatchLoop


class Kinematics:
    def __init__(self,*,reject_default=False,hole=False):
        self.last=np.r_[np.zeros(7),.4]
        self.reject_default=reject_default;self.hole=hole
    def pose(self,state):return np.array([0.,0.,.2]),np.eye(3)
    def placement_ik(self,p,r,seed):
        p=np.asarray(p).copy()
        if (self.reject_default and p[2]<.04 and r[0,0]>.5) or (self.hole and .035<p[0]<.065):
            p[0]+=.004
        return np.asarray(seed).copy(),p,np.asarray(r).copy()


class MarginalKinematics(Kinematics):
    """Default descent is feasible but near the planner's residual boundary."""
    def __init__(self,*,position_error=.0013,rotation_error=0.):
        super().__init__();self.position_error=position_error;self.rotation_error=rotation_error

    def placement_ik(self,p,r,seed):
        p=np.asarray(p).copy();r=np.asarray(r).copy()
        # bounds_fixture's held offset puts the tool endpoint at z=0.082 m;
        # only its last descent samples, not the high transport, are marginal.
        if p[2]<.10 and np.allclose(r,np.eye(3),atol=1e-10):
            p[0]+=self.position_error
            r=r@Rotation.from_euler('z',self.rotation_error).as_matrix()
        return np.asarray(seed).copy(),p,r


class PlacementPlanningTests(unittest.TestCase):
    def estimates(self):
        return [SimpleNamespace(position=np.array([.01,.004+i*.007,.18]),rotation=np.eye(3)) for i in range(5)]

    def test_default_candidate_is_exactly_the_original_held_offset_compensation(self):
        p=np.array([.03,.02,.2]);r=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
        box=Rotation.from_euler('z',.2).as_matrix();center=np.array([.2,.3,.02])
        candidates=placement_candidates(self.estimates(),p,r,box,center)
        expected=placement_goal(self.estimates(),p,r,box,center)
        np.testing.assert_array_equal(candidates[0].position,expected[0])
        np.testing.assert_array_equal(candidates[0].rotation,expected[1])

    def test_half_turn_keeps_cookie_normal_and_measured_center_at_the_target(self):
        p=np.array([0.,0.,.2]);r=Rotation.from_euler('z',.3).as_matrix()
        box=Rotation.from_euler('z',-.2).as_matrix();center=np.array([.1,.1,.02])
        offsets=(np.mean([e.position for e in self.estimates()],axis=0)-p)@r
        for candidate in placement_candidates(self.estimates(),p,r,box,center):
            np.testing.assert_allclose(candidate.position+candidate.rotation@offsets,center,atol=1e-12)
            np.testing.assert_allclose(candidate.rotation@r.T@np.array([0.,0.,1.]),box[:,2],atol=1e-12)

    def bounds_fixture(self):
        estimates=[SimpleNamespace(position=np.array([0.,i*.008,.02]),rotation=np.eye(3)) for i in range(5)]
        box=SimpleNamespace(position=np.zeros(3),rotation=np.eye(3),inner_half_size=np.array([.10,.10]))
        args=(estimates,np.array([0.,0.,.08]),np.eye(3),np.eye(3),np.array([0.,0.,.022]))
        return args,box

    def test_small_tilt_is_bounded_by_full_observed_slab_volume(self):
        args,box=self.bounds_fixture();candidates=placement_candidates(*args,include_tilt=True)
        candidate=next(pose for pose in candidates if pose.tilt_radians==-.06)
        check=placement_bounds(candidate,candidates[0],args[0],args[1],args[2],box,[.02,.003,.02],0.)
        self.assertTrue(check['safe']);self.assertGreater(check['min_floor_clearance_m'],0.)
        self.assertLess(check['packed_row_expansion_m'],1e-12)

    def test_floor_penetration_is_rejected_even_when_robot_ik_would_allow_it(self):
        args,box=self.bounds_fixture();original=placement_candidates(*args)[0]
        candidate=PlacementCandidate(original.position-[0.,0.,.004],original.rotation,0.)
        check=placement_bounds(candidate,original,args[0],args[1],args[2],box,[.02,.003,.02],0.)
        self.assertFalse(check['safe']);self.assertLess(check['min_floor_clearance_m'],0.)

    def test_bin_wall_and_packed_row_intrusions_reject_candidates(self):
        args,box=self.bounds_fixture();original=placement_candidates(*args)[0]
        for displacement in [[.10,0.,0.],[0.,.001,0.]]:
            candidate=PlacementCandidate(original.position+displacement,original.rotation,0.)
            check=placement_bounds(candidate,original,args[0],args[1],args[2],box,[.02,.003,.02],0.)
            self.assertFalse(check['safe'])

    def test_full_path_checks_intermediate_positions_not_only_endpoints(self):
        candidate=PlacementCandidate(np.array([.1,0.,.02]),np.eye(3),0.)
        result=check_placement_chain(Kinematics(hole=True),np.zeros(8),candidate)
        self.assertFalse(result['feasible']);self.assertEqual(result['failed_phase'],'TRANSPORT')

    def test_alternate_upright_pose_selected_when_default_descent_is_unreachable(self):
        motion=Kinematics(reject_default=True);before=motion.last.copy();events=[]
        estimates=[SimpleNamespace(position=np.array([0.,0.,.2]),rotation=np.eye(3))]*5
        candidate,checks=choose_placement(motion,np.zeros(8),estimates,np.array([0.,0.,.2]),np.eye(3),
            np.eye(3),np.array([0.,0.,.02]),emit=lambda **row:events.append(row))
        self.assertEqual(candidate.yaw_radians,np.pi)
        self.assertEqual(checks[0]['failed_phase'],'INSERT')
        self.assertTrue(checks[1]['feasible'])
        np.testing.assert_array_equal(motion.last,before)

    def test_feasible_original_pose_does_not_need_a_symmetry_change(self):
        estimates=[SimpleNamespace(position=np.array([0.,0.,.2]),rotation=np.eye(3))]*5
        candidate,checks=choose_placement(Kinematics(),np.zeros(8),estimates,np.array([0.,0.,.2]),np.eye(3),
            np.eye(3),np.array([0.,0.,.02]))
        self.assertEqual(candidate.yaw_radians,0.);self.assertEqual(len(checks),1)

    def test_marginal_default_compares_bounded_tilts_and_selects_more_accurate_path(self):
        args,box=self.bounds_fixture();events=[]
        candidate,checks=choose_placement(MarginalKinematics(),np.zeros(8),*args,
            target_box=box,half_size=[.02,.003,.02],floor_offset=0.,
            emit=lambda **row:events.append(row))
        self.assertNotEqual(candidate.tilt_radians,0.)
        self.assertEqual(len(checks),6)
        self.assertTrue(checks[0]['feasible'])
        self.assertAlmostEqual(checks[0]['max_position_error_m'],.0013)
        self.assertLess(checks[-1]['max_position_error_m'],.0003)
        self.assertEqual(events[-1]['event'],'placement_plan_selected')
        self.assertTrue(all(row['geometry']['safe'] for row in checks))

    def test_rotation_margin_alone_also_requires_comparing_alternatives(self):
        args,box=self.bounds_fixture()
        candidate,checks=choose_placement(MarginalKinematics(position_error=0.,rotation_error=.024),
            np.zeros(8),*args,target_box=box,half_size=[.02,.003,.02],floor_offset=0.)
        self.assertGreater(len(checks),1)
        self.assertNotEqual(candidate.tilt_radians,0.)
        self.assertLess(checks[-1]['max_rotation_error_rad'],.005)

    def test_precise_default_is_preserved_even_if_an_alternative_has_zero_residual(self):
        args,box=self.bounds_fixture()
        candidate,checks=choose_placement(MarginalKinematics(position_error=.0002,rotation_error=.003),
            np.zeros(8),*args,target_box=box,half_size=[.02,.003,.02],floor_offset=0.)
        original=placement_candidates(*args)[0]
        self.assertEqual(len(checks),1)
        np.testing.assert_array_equal(candidate.position,original.position)
        np.testing.assert_array_equal(candidate.rotation,original.rotation)

    def test_lower_residual_cannot_override_a_hard_geometry_rejection(self):
        args,box=self.bounds_fixture()
        # The original fits; every tilt crosses the unchanged floor guard.
        # The half-turn still fits and remains a potential safe alternative.
        args=(*args[:-1],np.array([0.,0.,.0198]))
        candidate,checks=choose_placement(MarginalKinematics(),np.zeros(8),*args,
            target_box=box,half_size=[.02,.003,.02],floor_offset=0.)
        self.assertEqual(candidate.tilt_radians,0.)
        tilted=[row for row in checks if row['tilt_radians']!=0.]
        self.assertEqual(len(tilted),4)
        self.assertTrue(all(not row['geometry']['safe'] and row['failed_phase']=='GEOMETRY'
            for row in tilted))

    def test_unknown_or_unreachable_paths_do_not_produce_a_placement(self):
        estimates=[SimpleNamespace(position=np.array([0.,0.,.2]),rotation=np.eye(3))]*5
        with self.assertRaisesRegex(LocalizationError,'No upright placement'):
            choose_placement(Kinematics(hole=True),np.zeros(8),estimates,np.array([0.,0.,.2]),np.eye(3),
                np.eye(3),np.array([.1,0.,.02]))

    def test_rotation_and_translation_interpolation_are_bounded_and_end_exactly(self):
        final=Rotation.from_euler('z',np.pi).as_matrix()
        samples=sample_segment(np.zeros(3),np.eye(3),np.array([.12,0.,0.]),final)
        previous_p=np.zeros(3);previous_r=np.eye(3)
        for p,r in samples:
            self.assertLessEqual(np.linalg.norm(p-previous_p),.012+1e-12)
            self.assertLessEqual(Rotation.from_matrix(previous_r.T@r).magnitude(),.06+1e-12)
            previous_p,previous_r=p,r
        np.testing.assert_allclose(samples[-1][0],[.12,0.,0.])
        np.testing.assert_allclose(samples[-1][1],final,atol=1e-12)

    def test_native_planning_flag_is_opt_in(self):
        for enabled in [False,True]:
            policy=NativeSensorPolicy();policy.sensor=SimpleNamespace(protocol={})
            with patch.dict(os.environ,dict(A3_SENSOR_PLACEMENT_PLAN=str(int(enabled)),A3_SENSOR_ACTIVE_RIGHT='0',
                A3_SENSOR_RIGHT_EACH_BATCH='0',A3_SENSOR_SCENE_MEMORY='0',A3_SENSOR_VISUAL_MEMORY='0')):
                policy.reset()
            self.assertEqual(policy.placement_planning,enabled)
            self.assertEqual(policy.sensor.protocol['placement_path_planning'],enabled)

    def stalled_loop(self):
        loop=SensorBatchLoop.__new__(SensorBatchLoop)
        loop.placement_planning=True;loop.refine=True;loop.age=80;loop.phase='PREINSERT'
        loop.encoder_settling=SimpleNamespace(count=12)
        loop.placement_reference_by_phase={'PREINSERT':np.ones(7)*.2}
        loop.placement_guided_phase=None;loop.placement_guidance_activations=0
        loop.move_goal=(np.zeros(3),np.eye(3));loop.emit=lambda *args,**kwargs:None
        loop.motion=SimpleNamespace(last=np.zeros(8),reference_q=None,errors=lambda *args:(.004,0.))
        return loop

    def test_stalled_guidance_requires_fresh_settled_tracking_and_activates_once(self):
        loop=self.stalled_loop()
        loop._maybe_guide_placement(np.zeros(8),measured_new=False)
        self.assertIsNone(loop.motion.reference_q)
        loop.encoder_settling.count=2;loop._maybe_guide_placement(np.zeros(8),measured_new=True)
        self.assertIsNone(loop.motion.reference_q)
        loop.encoder_settling.count=12;loop._maybe_guide_placement(np.zeros(8),measured_new=True)
        np.testing.assert_array_equal(loop.motion.reference_q,np.ones(7)*.2)
        loop._maybe_guide_placement(np.zeros(8),measured_new=True)
        self.assertEqual(loop.placement_guidance_activations,1)

    def test_tracking_resistance_does_not_activate_a_model_only_correction(self):
        loop=self.stalled_loop();loop.motion.last[:7]=.02
        loop._maybe_guide_placement(np.zeros(8),measured_new=True)
        self.assertIsNone(loop.motion.reference_q)

    def test_in_tolerance_pose_does_not_activate_guidance(self):
        loop=self.stalled_loop();loop.motion.errors=lambda *args:(.001,.01)
        loop._maybe_guide_placement(np.zeros(8),measured_new=True)
        self.assertIsNone(loop.motion.reference_q)


if __name__=='__main__':unittest.main()
