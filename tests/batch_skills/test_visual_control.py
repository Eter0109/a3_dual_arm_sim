import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from a3_batch_skills.rgbd import RGBDFrame,RGBDViews
from a3_batch_skills.rgbd_localization import BoxEstimate,CookieEstimate,VisualScene,LocalizationFailure
from a3_batch_skills.visual_goals import (VisualBatchGoal,VisualGoalPolicy,floor_visibility,
    visual_region_complete)
from a3_batch_skills.visual_controller import VisualBatchController


class FakePolicy:
    def __init__(self):self.starts=[];self.pauses=0;self.actions=0
    def start(self,goal,context):self.starts.append(goal.index)
    def pause(self,context):self.pauses+=1
    def act(self,obs,calib):self.actions+=1;return np.zeros(8)


class VisualControlTests(unittest.TestCase):
    def frame(self,depth=1.):
        transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=1.
        return RGBDFrame(np.zeros((40,40,3),np.uint8),np.full((40,40),float(depth)),
            np.array([[100.,0,20],[0,100.,20],[0,0,1]]),transform,0.)

    def test_occluded_and_missing_depth_never_count_as_empty_floor(self):
        box=BoxEstimate(np.zeros(3),np.eye(3),np.ones(2),0.,4,0.)
        xy=np.array([[0.,0.],[.01,.01]])
        self.assertEqual(floor_visibility(self.frame(),box,xy,0.),1.)
        self.assertEqual(floor_visibility(self.frame(.9),box,xy,0.),0.)
        self.assertEqual(floor_visibility(self.frame(np.nan),box,xy,0.),0.)
        self.assertEqual(floor_visibility(RGBDViews((self.frame(.9),self.frame())),box,xy,0.),1.)

    def test_multiview_requires_synchronous_physics(self):
        with self.assertRaises(ValueError):RGBDViews((self.frame(),replace(self.frame(),timestamp=.1)))

    def goal(self,index=0):
        positions=np.c_[np.full(5,index*.056),np.arange(5)*.006333,np.full(5,.0185)]
        return VisualBatchGoal(index,np.zeros((5,3)),np.eye(3),positions,np.zeros(3),np.eye(3),
            np.array([.025,.0031665,.0125]),0.,1,1.,1.)

    def scene(self,positions,timestamp):
        box=BoxEstimate(np.zeros(3),np.eye(3),np.array([.1,.1]),.026,4,0.)
        cookies=tuple(CookieEstimate(p,np.eye(3),None,1.,0.,"rgbd_rectangle_xyz") for p in positions)
        return VisualScene(box,box,(),cookies,timestamp,0.)

    def test_shifted_or_stacked_objects_do_not_pass_visual_gate(self):
        g=self.goal();p=g.target_positions_box.copy()
        self.assertTrue(visual_region_complete(self.scene(p,0),g))
        p[:,0]+=.008
        self.assertFalse(visual_region_complete(self.scene(p,0),g))
        p=g.target_positions_box.copy();p[:,2]+=.025
        self.assertFalse(visual_region_complete(self.scene(p,0),g))

    def test_old_or_untrained_checkpoint_cannot_enable_visual_mode(self):
        with self.assertRaises(ValueError):VisualGoalPolicy(None,{"goal_conditioning":"projected_batch_targets_v1"})

    def test_release_pauses_once_and_frozen_image_cannot_finish_batch(self):
        policy=FakePolicy();goal0=self.goal();goal1=self.goal(1)
        state=[self.scene([],0.)]
        class Localizer:
            def observe(self,frame):return state[0]
        controller=VisualBatchController(policy,dict(required_cookies=10,cookie_half_size=[.025,.0031665,.0125]),
            localizer=Localizer(),stable_observations=3)
        def run():return controller.act({}, {},frame=object(),pad_midpoint_world=[0,0,.2],jaw_gap_m=.05)
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=lambda f,s,p,i,**k:self.goal(i)):
            run();self.assertEqual(policy.starts,[0])
            state[0]=self.scene(goal0.target_positions_box,.1)
            run();run();run()
            self.assertEqual(controller.stable,1)  # repeated capture is not time passing
            self.assertEqual(controller.phase,"VERIFY_RELEASE")
            self.assertEqual(policy.pauses,1);self.assertEqual(policy.actions,1)
            controller.act({},{});self.assertEqual(policy.actions,1)  # no fresh frame: keep holding
            state[0]=replace(state[0],timestamp=.2);run()
            state[0]=replace(state[0],timestamp=.3);run()
            self.assertEqual(controller.phase,"LOCATE");self.assertEqual(policy.pauses,1)
            state[0]=replace(state[0],timestamp=.4);run()
            self.assertEqual(policy.starts,[0,1])
            for time in (.5,.6,.7):
                state[0]=self.scene(np.r_[goal0.target_positions_box,goal1.target_positions_box],time);run()
            self.assertEqual(controller.phase,"DONE");self.assertEqual(policy.pauses,2)

    def controller(self,required=10,**kwargs):
        state=[self.scene([],0.)];policy=FakePolicy()
        class Localizer:
            def observe(self,frame):return state[0]
        ctrl=VisualBatchController(policy,dict(required_cookies=required,cookie_half_size=[.025,.0031665,.0125]),
            localizer=Localizer(),**kwargs)
        return ctrl,policy,state

    def run_controller(self,ctrl):
        return ctrl.act({}, {},frame=object(),pad_midpoint_world=[0,0,.2],jaw_gap_m=.05)

    def test_misplaced_batch_pauses_before_another_pick_and_reports_offset(self):
        ctrl,policy,state=self.controller()
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=lambda f,s,p,i,**k:self.goal(i)):
            self.run_controller(ctrl)
            shifted=self.goal().target_positions_box+np.array([.020,.009,0.])
            for time in (.1,.2,.3):
                state[0]=self.scene(shifted,time);self.assertIsNone(self.run_controller(ctrl))
            self.assertEqual(ctrl.phase,"RECOVERY_REQUIRED")
            self.assertEqual(policy.starts,[0]);self.assertEqual(policy.actions,1);self.assertEqual(policy.pauses,1)
            np.testing.assert_allclose(ctrl.diagnostics['current_region']['mean_offset_box_m'],[.020,.009,0.])
            self.assertEqual(ctrl.completed,[])

    def test_previously_placed_progress_loss_requires_recovery(self):
        ctrl,policy,state=self.controller();g0=self.goal();g1=self.goal(1)
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=lambda f,s,p,i,**k:self.goal(i)):
            self.run_controller(ctrl)
            for time in (.1,.2,.3):
                state[0]=self.scene(g0.target_positions_box,time);self.run_controller(ctrl)
            state[0]=replace(state[0],timestamp=.4);self.run_controller(ctrl)
            for time in (.5,.6,.7):
                state[0]=self.scene(g1.target_positions_box,time);self.run_controller(ctrl)
            self.assertEqual(ctrl.phase,"RECOVERY_REQUIRED");self.assertEqual(len(ctrl.completed),1)
            self.assertIn("Previously verified",ctrl.failure_reason);self.assertEqual(policy.pauses,2)
            state[0]=self.scene(np.r_[g0.target_positions_box,g1.target_positions_box],.8)
            self.assertIsNone(self.run_controller(ctrl));self.assertEqual(ctrl.phase,"RECOVERY_REQUIRED")

    def test_partial_detection_cannot_end_batch_and_verification_loss_keeps_holding(self):
        ctrl,policy,state=self.controller(verification_steps=3)
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=lambda f,s,p,i,**k:self.goal(i)):
            self.run_controller(ctrl)
            state[0]=self.scene(self.goal().target_positions_box[:4],.1);self.run_controller(ctrl)
            self.assertEqual(ctrl.phase,"VERIFY_RELEASE");self.assertEqual(policy.pauses,1)
            self.assertEqual(ctrl.completed,[])  # partial detection is not completion
            state[0]=self.scene(self.goal().target_positions_box,.2);self.run_controller(ctrl)
            with patch.object(ctrl.localizer,'observe',side_effect=LocalizationFailure("Occluded")):
                for _ in range(4):self.assertIsNone(self.run_controller(ctrl))
            self.assertEqual(ctrl.phase,"RECOVERY_REQUIRED");self.assertEqual(policy.pauses,1)

    def test_stable_partial_release_does_not_continue_grasping(self):
        ctrl,policy,state=self.controller()
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=lambda f,s,p,i,**k:self.goal(i)):
            self.run_controller(ctrl)
            for time in (.1,.2,.3):
                state[0]=self.scene(self.goal().target_positions_box[:4],time);self.run_controller(ctrl)
            self.assertEqual(ctrl.phase,"RECOVERY_REQUIRED");self.assertEqual(ctrl.completed,[])
            self.assertEqual(policy.actions,1);self.assertEqual(policy.pauses,1)

    def test_occupied_next_region_never_starts_another_batch(self):
        ctrl,policy,state=self.controller()
        def goal(frame,scene,spec,index,**kwargs):
            if index:raise LocalizationFailure("Target region is occupied or not visibly empty")
            return self.goal()
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=goal):
            self.run_controller(ctrl)
            for time in (.1,.2,.3):
                state[0]=self.scene(self.goal().target_positions_box,time);self.run_controller(ctrl)
            self.assertEqual(ctrl.phase,"RECOVERY_REQUIRED");self.assertEqual(policy.starts,[0])
            self.assertIn("Next batch",ctrl.failure_reason)


if __name__=="__main__":unittest.main()
