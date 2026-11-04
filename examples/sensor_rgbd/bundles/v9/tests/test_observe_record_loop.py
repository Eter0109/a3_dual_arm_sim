"""Causal checks for explicit observe-record preparation and separate retries."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from a3_continuability.sensor_batch_loop import SensorBatchLoop
from a3_batch_skills.rgbd_localization import LocalizationFailure


class Motion:
    def __init__(self):
        self.last=np.r_[np.zeros(7),1.]
        self.pad_offset=np.zeros(3)

    def hold(self,opening=None):
        if opening is not None:self.last[7]=opening
        return self.last.copy()

    def pose(self,state):
        return np.array([0.,0.,1.]),np.eye(3)


class StateMemory:
    is_scene_state_memory=True

    def __init__(self):
        self.batch=None;self.records=set();self.observations=[];self.clears=[]
        self.make_goal=Mock()

    def begin_batch(self,batch):
        if self.batch!=batch:self.batch=batch;self.records=set()

    def clear(self,reason):
        self.clears.append(reason);self.records=set()

    def observe(self,packet,scene,*,batch,phase,stationary):
        self.begin_batch(batch)
        self.observations.append((packet,scene,phase,stationary))
        self.records.update(('source','target'))

    def has_record(self,kind,batch):
        return self.batch==batch and kind in self.records

    def snapshot_summary(self):
        return dict(source_records=int('source' in self.records),target_records=int('target' in self.records))


class ObserveRecordLoopTests(unittest.TestCase):
    def spec(self):
        return dict(schema='a3_rgbd_geometry_v2',required_cookies=10,
            cookie_half_size=[.025,.003166666666667,.0125],wall_thickness_m=.006,
            source_floor_top_offset_m=.756,floor_offset_m=.006,
            target_slots_local=[[x,y] for x in [-.0252,.0252] for y in np.arange(5)*.0088-.0176])

    def loop(self,memory=None):
        memory=StateMemory() if memory is None else memory
        return SensorBatchLoop(Motion(),self.spec(),visual_memory=memory)

    def packet(self,time):
        return SimpleNamespace(timestamp=float(time),jaw_gap_m=.04,rgbd=object(),
            pad_midpoint_world=np.array([0.,0.,1.]),
            observation={'observation.state':np.zeros(16)},
            calibrations={'front':(np.array([.5,-.18,1.25]),np.eye(3),48.)})

    def scene(self):
        source=SimpleNamespace(position=np.zeros(3),rotation=np.eye(3),
            inner_half_size=np.array([.1,.1]),rim_z=.786)
        target=SimpleNamespace(position=np.array([.2,.0,.75]),rotation=np.eye(3),
            inner_half_size=np.array([.0532,.03]),rim_z=.784)
        cookies=[SimpleNamespace(position=np.array([-.075,-.02+.0088*i,.7685]),column=1) for i in range(5)]
        return SimpleNamespace(source_box=source,target_box=target,source_cookies=cookies)

    def request(self,kind,index=0):
        exc=LocalizationFailure('observed floor insufficient')
        exc.missing_kind=kind;exc.region_index=index
        return exc

    def test_new_mode_requires_literal_flag_and_automatically_enables_hold(self):
        loop=self.loop()
        self.assertTrue(loop.scene_state_memory)
        self.assertTrue(loop.settle_observe)
        baseline=self.loop(Mock())
        self.assertFalse(baseline.scene_state_memory)
        self.assertFalse(baseline.settle_observe)

    def test_initial_locate_cannot_skip_real_stationary_capture(self):
        loop=self.loop();scene=self.scene()
        loop.localizer=SimpleNamespace(observe=Mock(return_value=scene))
        loop.choose_pick=Mock(side_effect=AssertionError('must hold before planning'))
        loop.act(self.packet(0),fresh=True)
        self.assertEqual(loop.phase,'OBSERVE_HOLD')
        self.assertEqual(loop.steps,1)
        self.assertEqual(loop.scene_batch_hold_count,0)
        self.assertEqual(loop.visual_memory.observations,[])
        for i in range(1,4):loop.act(self.packet(i),fresh=True)
        self.assertEqual(loop.visual_memory.observations,[])
        final=self.packet(4);loop.act(final,fresh=True)
        self.assertEqual(loop.phase,'LOCATE')
        self.assertEqual(loop.scene_batch_hold_count,1)
        self.assertIs(loop.visual_memory.observations[0][0],final)
        self.assertEqual(loop.visual_memory.observations[0][2:4],('OBSERVE_HOLD',True))

    def test_source_and_target_retry_candidates_start_at_zero_independently(self):
        loop=self.loop();loop._ensure_scene_observation_batch()
        actual=[]
        loop.start_observation=lambda packet:actual.append((loop.view_focus,loop.view_attempt))
        for kind in ('source_entry','target_empty','source_entry','target_empty'):
            loop._request_scene_observation(self.packet(1),self.request(kind,1))
        self.assertEqual(actual,[('source_entry',0),('target_empty',0),('source_entry',1),('target_empty',1)])
        self.assertEqual(loop.scene_observations_by_focus,{'source_entry':2,'target_empty':2})

    def test_each_target_region_has_its_own_first_candidate(self):
        loop=self.loop();loop._ensure_scene_observation_batch();actual=[]
        loop.start_observation=lambda packet:actual.append((loop.blocked_region,loop.view_attempt))
        for index in (0,1,0):loop._request_scene_observation(self.packet(1),self.request('target_empty',index))
        self.assertEqual(actual,[(0,0),(1,0),(0,1)])

    def test_observation_limits_fail_without_blind_goal_or_threshold_relaxation(self):
        loop=self.loop();loop._ensure_scene_observation_batch()
        loop.start_observation=Mock()
        for _ in range(3):loop._request_scene_observation(self.packet(1),self.request('source_entry'))
        loop._request_scene_observation(self.packet(1),self.request('source_entry'))
        self.assertEqual(loop.start_observation.call_count,3)
        self.assertEqual(loop.phase,'FAILED')
        self.assertIn('exhausted',loop.failure_reason)
        self.assertIsNone(loop.goal)

    def test_preparation_budget_counts_actual_control_steps(self):
        loop=self.loop();loop._ensure_scene_observation_batch();loop.steps=600
        loop.act(self.packet(1),fresh=True)
        self.assertEqual(loop.steps,601)
        self.assertEqual(loop.phase,'FAILED')
        self.assertIn('preparation budget',loop.failure_reason)

    def test_new_batch_clears_previous_evidence_and_focus_attempts(self):
        loop=self.loop();loop._ensure_scene_observation_batch()
        loop.visual_memory.records={'source','target'}
        loop.scene_view_attempts[('source_entry',None)]=2;loop.scene_batch_hold_count=3
        loop.completed=[SimpleNamespace(index=0)];loop.steps=100
        loop._ensure_scene_observation_batch()
        self.assertEqual(loop.scene_view_attempts,{})
        self.assertEqual(loop.scene_batch_hold_count,0)
        self.assertEqual(loop.scene_preparation_start,100)
        self.assertFalse(loop.visual_memory.has_record('target',1))

    def test_record_statistics_do_not_replace_current_batch_record_existence(self):
        loop=self.loop();loop._ensure_scene_observation_batch();loop.scene_batch_hold_count=1
        loop.visual_memory.snapshot_summary=lambda:dict(source_records=99,target_records=99)
        with self.assertRaises(LocalizationFailure) as caught:
            loop._choose_scene_state_pick(self.packet(1),self.scene())
        self.assertEqual(caught.exception.missing_kind,'source_entry')
        loop.visual_memory.records={'source'}
        with self.assertRaises(LocalizationFailure) as caught:
            loop._choose_scene_state_pick(self.packet(1),self.scene())
        self.assertEqual(caught.exception.missing_kind,'target_empty')

    def test_new_mode_does_not_bypass_independent_region_goal_check(self):
        loop=self.loop();packet=self.packet(1);scene=self.scene();goal=object()
        loop.visual_memory.make_goal.return_value=goal
        with patch('a3_continuability.sensor_batch_loop.make_visual_goal') as legacy:
            self.assertIs(loop._make_goal(packet,scene,1,1),goal)
        legacy.assert_not_called()
        loop.visual_memory.make_goal.assert_called_once_with(packet,scene,loop.spec,1,column=1,batch=0)

    def test_scene_memory_never_uses_passive_moving_capture(self):
        loop=self.loop();loop.memory_settling.count=8
        loop._remember_view(self.packet(1),self.scene())
        self.assertEqual(loop.visual_memory.observations,[])

    def test_source_roi_matches_whole_gate_corridor_and_actual_floor(self):
        loop=self.loop();scene=self.scene();five=scene.source_cookies
        with patch('a3_continuability.sensor_batch_loop.select_visible_five',return_value=five):
            points=loop._scene_observation_roi(scene,focus='source_entry')
        self.assertEqual(points.shape,(121,3))
        np.testing.assert_allclose(points[:,2],.756)
        self.assertAlmostEqual(points[:,0].min(),-.075-.018)
        self.assertAlmostEqual(points[:,0].max(),-.075+.018)
        self.assertAlmostEqual(points[:,1].min(),-.099)
        self.assertGreater(np.ptp(points[:,1]),.07)

    def test_target_roi_uses_target_geometry_not_source_height(self):
        loop=self.loop();scene=self.scene()
        scene.target_box.rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        points=loop._scene_observation_roi(scene,focus='target_empty',index=0)
        self.assertEqual(points.shape,(121,3))
        local=(points-scene.target_box.position)@scene.target_box.rotation
        np.testing.assert_allclose(local[:,2],loop.spec['floor_offset_m'])
        self.assertGreater(np.ptp(local[:,0]),.047)
        self.assertGreater(np.ptp(local[:,1]),.038)

    def test_manipulation_start_invalidates_recorded_empty_space(self):
        loop=self.loop();loop._ensure_scene_observation_batch()
        loop.visual_memory.records={'source','target'}
        loop.transition('APPROACH')
        self.assertEqual(loop.visual_memory.records,set())
        self.assertEqual(loop.visual_memory.clears[-1],'manipulation_started')


if __name__=='__main__':unittest.main()
