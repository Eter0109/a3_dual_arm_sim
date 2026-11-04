import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from a3_batch_skills.rgbd import RGBDFrame,RGBDViews
from a3_continuability.visual_memory import BatchVisualMemory


class VisualMemoryTests(unittest.TestCase):
    def box(self,shift=0.,yaw=0.,sides=4):
        c,s=np.cos(yaw),np.sin(yaw)
        return SimpleNamespace(position=np.array([shift,0.,0.]),
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]]),supported_sides=sides,
            inner_half_size=np.array([.15,.15]),extent_error_m=0.)

    def packet(self,time,depth=1.,x=0.):
        pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[:3,3]=[x,0,1.]
        f=RGBDFrame(np.zeros((64,64,3),np.uint8),np.full((64,64),depth,dtype=float),
            np.array([[64.,0,32.],[0,64.,32.],[0,0,1.]]),pose,time)
        return SimpleNamespace(timestamp=float(time),rgbd=RGBDViews((f,f,f)))

    def scene(self,time,box=None):
        box=box or self.box()
        return SimpleNamespace(timestamp=float(time),source_box=box,target_box=box)

    def xy(self):
        x,y=np.meshgrid(np.linspace(-.04,.04,11),np.linspace(-.04,.04,11))
        return np.c_[x.ravel(),y.ravel()]

    def recorded(self,**kwargs):
        memory=BatchVisualMemory(**kwargs)
        packet=self.packet(1.)
        memory.observe(packet,self.scene(1.),batch=0,phase='RIGHT_SCAN',stationary=True)
        return memory

    def query(self,memory,packet,box=None,batch=0):
        return memory.fraction(packet,box or self.box(),self.xy(),0.,kind='target',batch=batch)

    def test_recent_floor_complements_unknown_high_foreground_without_retiming(self):
        memory=self.recorded();current=self.packet(2.,depth=.8)
        result=self.query(memory,current)
        self.assertEqual(result['current_fraction'],0.)
        self.assertEqual(result['combined_fraction'],1.)
        self.assertTrue(result['history_used'])
        self.assertEqual(memory.views[0].frame.timestamp,1.)
        self.assertEqual(current.timestamp,2.)
        with self.assertRaises(ValueError):RGBDViews((memory.views[0].frame,current.rgbd.frames[0]))

    def test_new_near_floor_obstacle_invalidates_past_empty_region(self):
        memory=self.recorded()
        result=self.query(memory,self.packet(2.,depth=.97))
        self.assertTrue(result['current_obstacle_veto'])
        self.assertFalse(result['history_used'])
        self.assertEqual(result['combined_fraction'],0.)
        self.assertFalse(self.query(memory,self.packet(3.,depth=.8))['history_used'])

    def test_expired_future_and_unobserved_floor_are_never_relabelled_free(self):
        memory=self.recorded(max_age_s=3.)
        for time in (0.,5.):
            result=self.query(memory,self.packet(time,depth=np.nan))
            self.assertEqual(result['combined_fraction'],0.)
            self.assertFalse(result['history_used'])
        empty=BatchVisualMemory()
        self.assertEqual(self.query(empty,self.packet(2.,depth=np.nan))['combined_fraction'],0.)

    def test_moved_or_rotated_rim_rejects_old_registered_view(self):
        for box in (self.box(shift=.004),self.box(yaw=np.deg2rad(2.))):
            memory=self.recorded()
            result=self.query(memory,self.packet(2.,depth=np.nan),box)
            self.assertFalse(result['history_used'])
            self.assertEqual(result['combined_fraction'],0.)

    def test_current_rim_support_is_required_for_history(self):
        result=self.query(self.recorded(),self.packet(2.,depth=np.nan),self.box(sides=2))
        self.assertFalse(result['history_used'])

    def test_batch_change_and_manipulation_invalidation_remove_memory(self):
        memory=self.recorded()
        self.assertFalse(self.query(memory,self.packet(2.,depth=np.nan),batch=1)['history_used'])
        self.assertEqual(memory.views,[])
        memory=self.recorded();memory.clear('manipulation_started')
        self.assertFalse(self.query(memory,self.packet(2.,depth=np.nan))['history_used'])

    def test_capture_transform_is_not_replaced_by_current_camera_pose(self):
        memory=self.recorded()
        result=self.query(memory,self.packet(2.,depth=np.nan,x=.30))
        self.assertEqual(result['combined_fraction'],1.)
        self.assertEqual(memory.views[0].frame.world_from_optical[0,3],0.)

    def test_stationary_capture_replaces_by_identity_and_storage_is_bounded(self):
        memory=self.recorded(max_views=5)
        memory.observe(self.packet(2.),self.scene(2.),batch=0,phase='VIEW',stationary=True)
        self.assertEqual(len(memory.views),3)
        for i in range(10):
            p=self.packet(3.+i,x=i*.02)
            memory.observe(p,self.scene(3.+i),batch=0,phase='VIEW',stationary=True)
            self.assertLessEqual(len(memory.views),5)

    def test_moving_or_contact_stage_observations_cannot_refresh_history(self):
        memory=BatchVisualMemory()
        for phase in ('PINCH','INSERT','OPEN','VERIFY'):
            memory.observe(self.packet(1.),self.scene(1.),batch=0,phase=phase,stationary=True)
        memory.observe(self.packet(1.),self.scene(1.),batch=0,phase='VIEW',stationary=False)
        self.assertEqual(memory.views,[])

    def test_goal_uses_current_five_positions_and_never_cached_cookie_pose(self):
        memory=self.recorded()
        packet=self.packet(2.,depth=np.nan)
        box=self.box();scene=self.scene(2.,box)
        current_positions=np.array([[.03,-.02+i*.007,.02] for i in range(5)])
        scene.source_cookies=[SimpleNamespace(position=p,column=1) for p in current_positions]
        spec=dict(cookie_half_size=[.02,.003,.02],target_slots_local=[],floor_offset_m=0.,
            source_floor_top_offset_m=0.)
        target=np.array([[0,-.02+i*.007,.02] for i in range(5)])
        with patch('a3_continuability.visual_memory.target_regions',return_value=[target]), \
             patch('a3_continuability.visual_memory.select_visible_five',return_value=scene.source_cookies):
            goal=memory.make_goal(packet,scene,spec,0,column=1,batch=0)
        np.testing.assert_array_equal(goal.source_positions,current_positions)
        self.assertEqual(goal.timestamp,2.)
        self.assertGreater(memory.history_uses,0)


if __name__=='__main__':unittest.main()
