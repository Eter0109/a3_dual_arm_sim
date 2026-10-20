import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from a3_batch_skills.rgbd import RGBDFrame,MujocoRGBDViews
from a3_batch_skills.visual_goals import VisualBatchGoal
from a3_batch_skills.placement_observation import PlacementProbeRecorder
from a3_batch_skills.placement_audit import PlacementTruthAudit
from a3_batch_skills.rgbd_held import HeldBatchLocalizer
from a3_batch_skills.rgbd_held import HeldBatchPose
from a3_batch_skills.rgbd_localization import rotation_z,LocalizationFailure
from a3_batch_skills.rgbd_placement import (MeasuredPlacementServo,RGBDPlacementPolicy,
    RobotPlacementState,relative_goal_features)


class PlacementProbeTests(unittest.TestCase):
    def test_privileged_audit_keeps_identity_separate_from_later_noisy_measurements(self):
        local=np.c_[np.zeros(5),np.arange(-2,3)*.006333,np.zeros(5)]
        first=local+[0,0,.07];other=local+[.15,0,.07]
        positions=np.r_[first,other,np.zeros((1,3))]
        env=SimpleNamespace(_cookie_bodies=np.arange(10),_target_bin_body=10,
            data=SimpleNamespace(xpos=positions,xmat=np.tile(np.eye(3).ravel(),(11,1))))
        goal=SimpleNamespace(index=0,target_positions_box=local+[0,0,.0185])
        pose=HeldBatchPose(first.mean(0),np.eye(3),first,0.,0.,5,0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.jsonl';audit=PlacementTruthAudit(path)
            self.assertIsNone(audit.write_after_decision(env,goal,pose,0.))
            positions[:5,0]+=.004
            noisy=replace(pose,positions=other,center=other.mean(0),timestamp=.05)
            audit.write_after_decision(env,goal,noisy,.05);audit.close()
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(rows[0]['audit_cookie_ids'],rows[1]['audit_cookie_ids'])
            self.assertAlmostEqual(rows[1]['true_xy_error_m'],.004)
            self.assertGreater(rows[1]['visual_center_error_m'],.1)
            self.assertTrue(all(not r['used_for_control'] for r in rows))

    def test_additional_sensor_views_are_reported_as_five(self):
        with patch('a3_batch_skills.rgbd.MujocoRGBDSensor',return_value=SimpleNamespace(protocol={})):
            sensor=MujocoRGBDViews(None,views=((-65,180),(-90,90),(-65,90),(-25,0),(-25,180)))
        self.assertEqual(sensor.protocol['view_count'],5)
        self.assertEqual(sensor.protocol['sensor'],'fixed_rgbd_views')

    def test_bounded_synced_records_preserve_proprioception(self):
        frame=RGBDFrame(np.zeros((4,4,3),np.uint8),np.ones((4,4)),np.eye(3),np.eye(4),1.)
        goal=VisualBatchGoal(0,np.zeros((5,3)),np.eye(3),np.zeros((5,3)),np.zeros(3),np.eye(3),
            np.array([.025,.0031665,.0125]),0.,1,1.,1.)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'probes';rec=PlacementProbeRecorder(root,{},limit=1)
            args=dict(tool_position=np.zeros(3),tool_rotation=np.eye(3),pad_midpoint=np.zeros(3),jaw_gap=.03,opening=.2)
            rec.add(101,frame,{'observation.state':np.arange(16)},goal,**args)
            rec.add(121,frame,{'observation.state':np.arange(8)},goal,**args)
            self.assertEqual(rec.count,1)
            np.testing.assert_array_equal(np.load(root/'step_0101/policy_observation.npz')['observation.state'],np.arange(8))
            self.assertFalse(json.loads((root/'protocol.json').read_text())['object_truth'])


class HeldGeometryTests(unittest.TestCase):
    def cloud(self,count=5):
        x,y=np.meshgrid(np.linspace(-.025,.025,85),np.linspace(-.002,.002,9))
        patches=[np.c_[x.ravel(),y.ravel()+i*.006333,np.full(x.size,.0125)] for i in np.arange(count)-(count-1)/2]
        rotation=rotation_z(.12)
        center=np.array([.1,.2,.9])
        return np.concatenate(patches)@rotation.T+center,center,rotation

    def localizer(self):
        return HeldBatchLocalizer(dict(schema='a3_rgbd_geometry_v2',cookie_half_size=[.025,.0031665,.0125],
                                       wall_thickness_m=.006))

    def test_five_raised_cookies_have_measured_pose_not_floor_height(self):
        cloud,center,rotation=self.cloud()
        pose=self.localizer().fit(cloud,center+[0,0,.010],np.eye(3),2.)
        np.testing.assert_allclose(pose.center,center,atol=.0008)
        np.testing.assert_allclose(pose.rotation,rotation,atol=.03)
        self.assertEqual(pose.visible_count,5)

    def test_partial_group_is_unknown_not_filled_from_packing(self):
        cloud,center,_=self.cloud(3)
        with self.assertRaises(LocalizationFailure):self.localizer().fit(cloud,center+[0,0,.01],np.eye(3),1.)

    def test_both_visible_ends_support_pose_despite_middle_occlusion(self):
        cloud,center,rotation=self.cloud()
        local=(cloud-center)@rotation
        cloud=cloud[abs(local[:,0])>.010]
        pose=self.localizer().fit(cloud,center+[0,0,.010],np.eye(3),1.)
        np.testing.assert_allclose(pose.center,center,atol=.001)

    def test_one_visible_end_does_not_prove_full_longitudinal_pose(self):
        cloud,center,rotation=self.cloud();local=(cloud-center)@rotation
        cloud=cloud[local[:,0]>-.005]
        with self.assertRaises(LocalizationFailure):self.localizer().fit(cloud,center+[0,0,.010],np.eye(3),1.)


class FakeServo:
    def __init__(self):self.position=self.rotation=None;self.openings=[]
    def hold(self,opening):self.openings.append(opening);return np.r_[np.zeros(7),opening]
    def move(self,position,rotation,opening,speed):
        self.position,self.rotation=np.array(position),np.array(rotation)
        self.openings.append(opening);return np.r_[np.zeros(7),opening],True
    def opening_for_width(self,width,margin):return .45


class PlacementControlTests(unittest.TestCase):
    def pose(self,center=(.02,.01,.08),rotation=None,time=0.):
        rotation=np.eye(3) if rotation is None else rotation;center=np.asarray(center,dtype=float)
        positions=np.c_[np.zeros(5),np.arange(-2,3)*.006333,np.zeros(5)]@rotation.T+center
        return HeldBatchPose(center,rotation,positions,time,0.,5,0.)

    def robot(self,pose):
        return RobotPlacementState(pose.center+[.016,0,.010],np.eye(3),pose.center+[.016,0,.010],.031666,.25,pose.timestamp)

    def test_relative_context_is_invariant_to_world_frame_and_keeps_18_features(self):
        pose=self.pose();robot=self.robot(pose);target=np.array([0.,0.,.0185]);rotation=rotation_z(.1)
        expected=relative_goal_features(target,rotation,pose,robot)
        q=rotation_z(.3);shift=np.array([.4,.5,.1])
        p=replace(pose,center=q@pose.center+shift,rotation=q@pose.rotation)
        r=replace(robot,tool_position=q@robot.tool_position+shift,tool_rotation=q@robot.tool_rotation)
        np.testing.assert_allclose(relative_goal_features(q@target+shift,q@rotation,p,r),expected,atol=1e-12)
        self.assertEqual(expected.shape,(18,))

    def test_missing_or_repeated_geometry_never_opens_the_gripper(self):
        fake=FakeServo();servo=MeasuredPlacementServo(fake);pose=self.pose();robot=self.robot(pose)
        servo.start(pose,robot,np.array([0.,0.,.0185]),np.eye(3))
        for i in range(30):servo.act(None,replace(robot,timestamp=i*.05),np.array([0.,0.,.0185]),np.eye(3))
        self.assertIsNotNone(servo.failure);self.assertTrue(all(x==.25 for x in fake.openings))
        servo.start(pose,robot,np.zeros(3),np.eye(3))
        servo.act(pose,robot,np.zeros(3),np.eye(3));servo.act(pose,robot,np.zeros(3),np.eye(3))
        self.assertEqual(servo.missing,1)

    def test_measured_group_offset_is_corrected_before_release(self):
        fake=FakeServo();servo=MeasuredPlacementServo(fake)
        pose=self.pose(rotation=rotation_z(.1));robot=self.robot(pose)
        offset=robot.tool_rotation.T@(pose.center-robot.tool_position)
        relative=robot.tool_rotation.T@pose.rotation
        target=np.array([0.,0.,.0185]);servo.start(pose,robot,target,np.eye(3))
        for i in range(250):
            servo.act(pose,robot,target,np.eye(3))
            if servo.stage=='OPEN':
                self.assertLess(np.linalg.norm(pose.center-target),.0012);break
            if fake.position is not None:
                pos=fake.position+fake.rotation@offset;rot=fake.rotation@relative
                pose=self.pose(pos,rot,(i+1)*.05)
                robot=replace(robot,tool_position=fake.position,tool_rotation=fake.rotation,
                              pad_midpoint=pos+[.016,0,.010],timestamp=pose.timestamp)
        self.assertEqual(servo.stage,'OPEN');self.assertIsNone(servo.failure)

    def test_handoff_with_unobservable_objects_holds_and_reports_failure(self):
        class Policy:
            def start(self,*args):pass
            def pause(self,*args):pass
            def act(self,*args):raise AssertionError('VLA may not continue into blind placement')
        fake=FakeServo();wrapped=RGBDPlacementPolicy(Policy(),fake,
            dict(schema='a3_rgbd_geometry_v2',cookie_half_size=[.025,.0031665,.0125],wall_thickness_m=.006))
        goal=VisualBatchGoal(0,np.zeros((5,3)),np.eye(3),np.c_[np.zeros(5),np.arange(-2,3)*.006333,np.full(5,.0185)],
                            np.zeros(3),np.eye(3),np.array([.025,.0031665,.0125]),0.,1,1.,1.)
        wrapped.start(goal)
        pose=self.pose((0,0,.12));robot=self.robot(pose)
        wrapped.scene_provider=lambda:SimpleNamespace(timestamp=wrapped.robot.timestamp,
            target_box=SimpleNamespace(position=np.zeros(3),rotation=np.eye(3)))
        with patch.object(wrapped.localizer,'observe',side_effect=LocalizationFailure('occluded')):
            for i in range(40):
                wrapped.update(object(),replace(robot,timestamp=i*.05));wrapped.act({}, {})
        self.assertIsNotNone(wrapped.failure);self.assertEqual(wrapped.counts['vla'],0)
        self.assertTrue(all(x==.25 for x in fake.openings))

    def test_single_frame_drift_holds_without_resetting_descent(self):
        fake=FakeServo();servo=MeasuredPlacementServo(fake)
        target=np.array([0.,0.,.0185]);pose=self.pose((0,0,.07));robot=self.robot(pose)
        servo.start(pose,robot,target,np.eye(3));servo.transition('DESCEND')
        spike=self.pose((0,0,.07),rotation_z(np.deg2rad(3.6)),.05)
        servo.act(spike,replace(robot,timestamp=.05),target,np.eye(3))
        self.assertEqual(servo.stage,'DESCEND');self.assertEqual(servo.realignments,0)
        self.assertEqual(servo.diagnostics['decision'],'hold_recheck_drift');self.assertIsNone(fake.position)
        good=replace(pose,timestamp=.10)
        servo.act(good,replace(robot,timestamp=.10),target,np.eye(3))
        self.assertEqual(servo.stage,'DESCEND');self.assertEqual(servo.drift_observations,0)
        self.assertIsNotNone(fake.position);self.assertTrue(all(x==.25 for x in fake.openings))

    def test_persistent_drift_needs_three_fresh_observations_before_lifting(self):
        fake=FakeServo();servo=MeasuredPlacementServo(fake);target=np.array([0.,0.,.0185])
        pose=self.pose((.008,0,.07));robot=self.robot(pose)
        servo.start(pose,robot,target,np.eye(3));servo.transition('DESCEND')
        for i in range(1,4):
            p=replace(pose,timestamp=i*.05);servo.act(p,replace(robot,timestamp=p.timestamp),target,np.eye(3))
            if i<3:self.assertEqual(servo.stage,'DESCEND')
        self.assertEqual(servo.stage,'LIFT');self.assertEqual(servo.realignments,1)
        self.assertTrue(all(x==.25 for x in fake.openings))


if __name__=='__main__':unittest.main()
