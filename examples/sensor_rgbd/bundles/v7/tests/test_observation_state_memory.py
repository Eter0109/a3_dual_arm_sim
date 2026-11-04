import json
import unittest
from types import SimpleNamespace
import numpy as np
from a3_batch_skills.rgbd import RGBDFrame, RGBDViews
from a3_batch_skills.rgbd_localization import LocalizationFailure
from a3_continuability.observation_state_memory import SceneStateMemory, ObservationRequirement


class ObservationStateMemoryTests(unittest.TestCase):
    def box(self,x,*,sides=4,yaw=0.):
        c,s=np.cos(yaw),np.sin(yaw)
        return SimpleNamespace(position=np.array([x,0.,0.]),
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]]),
            inner_half_size=np.array([.10,.10]),supported_sides=sides,extent_error_m=0.)

    def scene(self,time,*,source_sides=4,target_sides=4,target_shift=0.,target_yaw=0.,five=True,source_y=0.):
        source=self.box(-.15,sides=source_sides)
        target=self.box(.15+target_shift,sides=target_sides,yaw=target_yaw)
        cookies=tuple(SimpleNamespace(position=np.array([-.15,-.02+i*.007+source_y,.02]),
            rotation=np.eye(3),column=1) for i in range(5)) if five else ()
        return SimpleNamespace(timestamp=float(time),source_box=source,target_box=target,
            source_cookies=cookies,target_cookies=())

    def packet(self,time,*,source=1.,target=1.,n=64,target_obstacle=False):
        pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[:3,3]=[0,0,1.]
        depth=np.empty((n,n));depth[:,:n//2]=source;depth[:,n//2:]=target
        if target_obstacle:
            # One sampled interior target ray sees a 30 mm obstacle. All other
            # target rays see real floor, so live coverage remains above .90.
            depth[int(np.rint(n/2-.5)),int(np.rint(n/2-.5+.15*n))]=.97
        frame=RGBDFrame(np.zeros((n,n,3),np.uint8),depth,
            np.array([[n,0.,n/2],[0.,n,n/2],[0.,0.,1.]]),pose,float(time))
        return SimpleNamespace(timestamp=float(time),rgbd=RGBDViews((frame,frame,frame)))

    def spec(self):
        return dict(cookie_half_size=[.02,.003,.02],floor_offset_m=0.,
            source_floor_top_offset_m=0.,target_slots_local=[[0.,-.014+i*.007] for i in range(5)])

    def observe(self,memory,time,**kwargs):
        scene=self.scene(time,**kwargs.pop('scene_kwargs',{}))
        packet=self.packet(time,**kwargs)
        memory.observe(packet,scene,batch=0,phase='VIEW',stationary=True)
        return packet,scene

    def goal(self,memory,packet,scene,batch=0):
        return memory.make_goal(packet,scene,self.spec(),0,column=1,batch=batch)

    def test_same_pose_different_times_preserve_complementary_floor_rays(self):
        memory=SceneStateMemory()
        self.observe(memory,1.,source=1.,target=.8)
        self.observe(memory,2.,source=.8,target=1.)
        self.assertEqual(len(memory.views),6)
        goal=self.goal(memory,self.packet(3.,source=1.,target=.8),self.scene(3.))
        self.assertEqual(goal.source_clear_fraction,1.)
        self.assertEqual(goal.target_empty_fraction,1.)
        self.assertGreater(memory.history_uses,0)
        self.assertEqual({view.frame.timestamp for view in memory.views},{1.,2.})
        self.assertEqual(goal.timestamp,3.)

    def test_source_floor_history_and_current_target_can_also_complete_goal(self):
        memory=SceneStateMemory()
        self.observe(memory,1.,source=1.,target=.8)
        goal=self.goal(memory,self.packet(2.,source=.8,target=1.),self.scene(2.))
        self.assertEqual(goal.source_clear_fraction,1.)
        self.assertEqual(memory.last_gate_evidence['source_evidence']['current_fraction'],0.)
        self.assertEqual(memory.goal_accepts,1)

    def test_history_supported_goals_count_only_accepted_history_dependent_actions(self):
        memory=SceneStateMemory()
        self.observe(memory,1.,source=.8,target=1.)
        with self.assertRaises(ObservationRequirement):
            self.goal(memory,self.packet(2.,source=.8,target=.8),self.scene(2.))
        self.assertGreater(memory.history_uses,0)
        self.assertEqual(memory.history_supported_goals,0)
        self.assertEqual(memory.goal_accepts,0)
        self.goal(memory,self.packet(3.,source=1.,target=.8),self.scene(3.))
        self.assertEqual(memory.history_supported_goals,1)
        self.assertEqual(memory.goal_accepts,1)
        self.goal(memory,self.packet(4.),self.scene(4.))
        summary=memory.snapshot_summary()
        self.assertEqual(summary['history_supported_goals'],1)
        self.assertEqual(summary['goal_accepts'],2)

    def test_observation_records_one_supported_side_without_passing_either_gate(self):
        events=[]
        # Match NativeSensorPolicy's event wrapper to catch duplicate phase keys.
        def emit(**row):events.append(dict(step=0,phase='SCENE_MEMORY',**row))
        memory=SceneStateMemory(emit=emit)
        self.observe(memory,1.,source=.8,target=.8,scene_kwargs=dict(target_sides=2))
        summary=memory.snapshot_summary()
        self.assertEqual(summary['scene_records'],1)
        self.assertEqual(summary['source_records'],1)
        self.assertEqual(summary['target_records'],0)
        self.assertTrue(memory.has_record('source',0))
        self.assertFalse(memory.has_record('target',0))
        self.assertFalse(memory.has_record('source',1))
        self.assertEqual(summary['goal_queries'],0)
        self.assertEqual(summary['goal_accepts'],0)
        self.assertTrue(events)
        json.dumps(summary)

    def test_no_current_five_never_executes_cached_cookie_positions(self):
        memory=SceneStateMemory()
        self.observe(memory,1.)
        with self.assertRaises(ObservationRequirement) as raised:
            self.goal(memory,self.packet(2.),self.scene(2.,five=False))
        self.assertEqual(raised.exception.missing_kind,'source_entry')
        self.assertFalse(raised.exception.has_current_five)
        self.assertEqual(raised.exception.target_fraction,1.)
        self.assertEqual(memory.goal_accepts,0)

    def test_grasp_positions_always_use_current_source_measurements(self):
        memory=SceneStateMemory()
        self.observe(memory,1.)
        current=self.scene(2.,source_y=.004)
        goal=self.goal(memory,self.packet(2.,target=.8),current)
        np.testing.assert_array_equal(goal.source_positions,
            np.asarray([cookie.position for cookie in current.source_cookies]))
        self.assertNotEqual(goal.source_positions[0,1],memory.scene_observations[0].source_cookies[0].position[1])

    def test_new_obstacle_veto_rejects_even_high_live_free_fraction_and_invalidates_history(self):
        memory=SceneStateMemory()
        self.observe(memory,1.)
        with self.assertRaises(ObservationRequirement) as raised:
            self.goal(memory,self.packet(2.,n=256,target_obstacle=True),self.scene(2.))
        self.assertEqual(raised.exception.missing_kind,'target_empty')
        evidence=raised.exception.evidence['target_evidence']
        self.assertGreater(evidence['current_fraction'],.90)
        self.assertTrue(evidence['current_obstacle_veto'])
        self.assertFalse(evidence['history_used'])
        self.assertEqual(memory.snapshot_summary()['target_records'],0)
        with self.assertRaises(ObservationRequirement):
            self.goal(memory,self.packet(3.,target=.8),self.scene(3.))

    def test_expired_observations_do_not_supply_floor(self):
        memory=SceneStateMemory(max_age_s=.5)
        self.observe(memory,1.)
        with self.assertRaises(ObservationRequirement) as raised:
            self.goal(memory,self.packet(2.,source=.8,target=.8),self.scene(2.))
        self.assertEqual(raised.exception.source_fraction,0.)
        self.assertEqual(raised.exception.target_fraction,0.)
        summary=memory.snapshot_summary()
        self.assertEqual(summary['source_records'],0)
        self.assertEqual(summary['target_records'],0)

    def test_moved_or_rotated_box_rejects_registered_floor_but_not_other_side(self):
        for kwargs in (dict(target_shift=.004),dict(target_yaw=np.deg2rad(2.))):
            memory=SceneStateMemory();self.observe(memory,1.)
            with self.assertRaises(ObservationRequirement) as raised:
                self.goal(memory,self.packet(2.,source=.8,target=.8),self.scene(2.,**kwargs))
            self.assertEqual(raised.exception.missing_kind,'target_empty')
            self.assertEqual(raised.exception.source_fraction,1.)
            self.assertEqual(raised.exception.target_fraction,0.)
            self.assertFalse(memory.has_record('target',0))
            self.assertTrue(memory.has_record('source',0))

    def test_batch_change_and_manipulation_clear_records_and_floor(self):
        memory=SceneStateMemory();self.observe(memory,1.)
        memory.begin_batch(1)
        self.assertEqual(memory.scene_observations,[])
        self.assertEqual(memory.views,[])
        self.observe(memory,2.)
        memory.clear('manipulation_started')
        self.assertEqual(memory.scene_observations,[])
        self.assertEqual(memory.views,[])
        self.assertEqual(memory.scene_records,2)

    def test_contact_or_unsettled_samples_cannot_refresh_scene_records(self):
        memory=SceneStateMemory();packet=self.packet(1.);scene=self.scene(1.)
        for phase in ('PINCH','DESCEND','INSERT','OPEN','VERIFY','TRANSPORT'):
            memory.observe(packet,scene,batch=0,phase=phase,stationary=True)
        memory.observe(packet,scene,batch=0,phase='VIEW',stationary=False)
        self.assertEqual(memory.scene_records,0)
        self.assertEqual(memory.captures,0)
        self.assertEqual(memory.views,[])

    def test_both_gates_are_evaluated_before_selecting_missing_side(self):
        memory=SceneStateMemory()
        with self.assertRaises(ObservationRequirement) as raised:
            self.goal(memory,self.packet(1.,source=.8,target=.8),self.scene(1.))
        error=raised.exception
        self.assertIsInstance(error,LocalizationFailure)
        self.assertTrue(error.has_current_five)
        self.assertEqual(error.missing_kind,'source_entry')
        self.assertEqual(error.region_index,0)
        self.assertEqual(set(error.coverage_by_kind),{'source_entry','target_empty'})
        self.assertEqual(memory.queries,2)

    def test_when_both_gates_fail_unrecorded_side_is_requested_first(self):
        memory=SceneStateMemory()
        self.observe(memory,1.,source=.8,target=.8,scene_kwargs=dict(target_sides=2))
        with self.assertRaises(ObservationRequirement) as raised:
            self.goal(memory,self.packet(2.,source=.8,target=.8),self.scene(2.))
        self.assertEqual(raised.exception.missing_kind,'target_empty')

    def test_scene_and_camera_records_are_independent_copies_with_original_time(self):
        memory=SceneStateMemory();packet,scene=self.observe(memory,1.)
        record=memory.scene_observations[0]
        original=record.boxes['source'].position.copy()
        scene.source_box.position[:]+=1.
        scene.source_cookies[0].position[:]+=1.
        np.testing.assert_array_equal(record.boxes['source'].position,original)
        self.assertLess(record.source_cookies[0].position[0],0.)
        self.assertEqual(record.timestamp,1.)
        self.assertEqual(record.camera_records[0]['timestamp'],1.)
        self.assertFalse(record.camera_records[0]['world_from_optical'].flags.writeable)
        np.testing.assert_array_equal(record.camera_records[0]['world_from_optical'],
            packet.rgbd.frames[0].world_from_optical)

    def test_storage_is_bounded_and_same_camera_timestamp_not_duplicated(self):
        memory=SceneStateMemory(max_views=5)
        packet,scene=self.observe(memory,1.)
        memory.observe(packet,scene,batch=0,phase='VIEW',stationary=True)
        self.assertEqual(len(memory.views),3)
        self.assertEqual(memory.captures,1)
        self.assertEqual(memory.scene_records,1)
        for i in range(2,12):self.observe(memory,float(i))
        self.assertLessEqual(len(memory.views),5)
        self.assertLessEqual(len(memory.scene_observations),5)


if __name__=='__main__':unittest.main()
