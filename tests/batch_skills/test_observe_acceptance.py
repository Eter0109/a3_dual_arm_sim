import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from a3_batch_skills.rgbd_localization import BoxEstimate,CookieEstimate,VisualScene,LocalizationFailure
from a3_batch_skills.visual_acceptance import contained_report
from a3_batch_skills.visual_slabs import face_candidates
from a3_batch_skills.visual_evidence import PlanningEvidence
from a3_batch_skills.visual_controller import VisualBatchController
from a3_batch_skills.visual_goals import VisualBatchGoal,visual_region_complete


HALF=np.array([.025,.0031665,.0125])
SPEC=dict(cookie_half_size=HALF.tolist(),floor_offset_m=.006,required_cookies=10)


def scene(cookies=(),time=0.):
    box=BoxEstimate(np.zeros(3),np.eye(3),np.array([.057,.045]),.04,4,0.)
    return VisualScene(box,box,(),tuple(cookies),time,0.)


def cookie(position,rotation=None):
    return CookieEstimate(np.array(position),np.eye(3) if rotation is None else rotation,None,1.,0.)


def goal():
    p=np.c_[np.full(5,-.027),np.arange(-2,3)*2*HALF[1],np.full(5,.0185)]
    return VisualBatchGoal(0,p,np.eye(3),p,np.zeros(3),np.eye(3),HALF,0.,1,1.,1.)


class ObserveAcceptanceTests(unittest.TestCase):
    def test_observation_after_release_uses_measured_gap_and_finger_clearance(self):
        from a3_batch_skills.active_observation import observation_entry_ready
        self.assertTrue(observation_entry_ready(.92,.84,.04,.035,False))
        self.assertFalse(observation_entry_ready(.89,.84,.04,.035,False))
        self.assertFalse(observation_entry_ready(.92,.84,.03,.035,False))
        self.assertFalse(observation_entry_ready(.92,.84,.04,.035,True))
    def test_shifted_five_pass_containment_but_not_strict_slots(self):
        g=goal();s=scene([cookie(p+[.0057,0,0]) for p in g.target_positions_box])
        self.assertTrue(contained_report(s,SPEC,5)['complete'])
        self.assertFalse(visual_region_complete(s,g))

    def test_lying_and_tilted_can_pass_but_stack_and_outside_cannot(self):
        lying=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
        flat=cookie([0,0,.006+HALF[1]],lying)
        self.assertTrue(contained_report(scene([flat]),SPEC,1)['complete'])
        self.assertFalse(contained_report(scene([replace(flat,position=flat.position+[0,0,2*HALF[1]])]),SPEC,1)['complete'])
        self.assertFalse(contained_report(scene([replace(flat,position=flat.position+[.05,0,0])]),SPEC,1)['complete'])
        a=.22;r=np.array([[1.,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        tilted=cookie([0,0,.006+(abs(r)@HALF)[2]],r)
        self.assertTrue(contained_report(scene([tilted]),SPEC,1)['complete'])

    def test_unknown_or_missing_previous_instances_never_pass(self):
        s=scene([cookie(p) for p in goal().target_positions_box])
        self.assertFalse(contained_report(s,SPEC,5,unknown=True)['complete'])
        self.assertFalse(contained_report(s,SPEC,10)['complete'])

    def test_measured_flat_face_recovers_lying_cookie(self):
        x,y=np.meshgrid(np.linspace(-HALF[0],HALF[0],65),np.linspace(-HALF[2],HALF[2],35))
        q=np.c_[x.ravel(),y.ravel(),np.full(x.size,.006+2*HALF[1])]
        found=face_candidates(q,np.array([0,0,1.]),HALF)
        self.assertTrue(found)
        self.assertTrue(any(contained_report(scene([c]),SPEC,1)['complete'] for c in found))
        np.testing.assert_allclose(found[0].position,[0,0,.006+HALF[1]],atol=.001)

    def test_partial_face_does_not_invent_full_cookie(self):
        x,y=np.meshgrid(np.linspace(-.010,.010,25),np.linspace(-.008,.008,25))
        q=np.c_[x.ravel(),y.ravel(),np.full(x.size,.012333)]
        self.assertEqual(face_candidates(q,np.array([0,0,1.]),HALF),[])

    def test_independent_evidence_keeps_original_timestamps_then_expires(self):
        from types import SimpleNamespace
        e=PlanningEvidence();s=scene();g=goal()
        source=(g.source_positions,g.source_rotation,1,1.);target=(g.target_positions_box,1.)
        with patch('a3_batch_skills.visual_goals.measured_goal_parts',side_effect=[(source,None),(None,target)]):
            e.update(SimpleNamespace(timestamp=1.),s,SPEC,0,'auto',safe_static=True)
            e.update(SimpleNamespace(timestamp=2.),replace(s,timestamp=2.),SPEC,0,'auto',safe_static=True)
        self.assertEqual(e.provenance(),{'source':1.,'target':2.})
        measured=e.goal(s,SPEC,0,3.)
        self.assertEqual(measured.index,0)
        self.assertEqual((measured.timestamp,measured.source_timestamp,measured.target_timestamp),(1.,1.,2.))
        with self.assertRaises(LocalizationFailure):e.goal(s,SPEC,0,12.)
        e.invalidate('contact','source')
        with self.assertRaises(LocalizationFailure):e.goal(s,SPEC,0,3.)

    def test_box_motion_invalidates_old_other_region_evidence(self):
        from types import SimpleNamespace
        e=PlanningEvidence();s=scene();g=goal()
        with patch('a3_batch_skills.visual_goals.measured_goal_parts',side_effect=[
            ((g.source_positions,g.source_rotation,1,1.),None),
            (None,(g.target_positions_box,1.))]):
            e.update(SimpleNamespace(timestamp=1.),s,SPEC,0,'auto',safe_static=True)
            moved=replace(s,target_box=replace(s.target_box,position=np.array([.01,0,0.])),timestamp=2.)
            e.update(SimpleNamespace(timestamp=2.),moved,SPEC,0,'auto',safe_static=True)
        self.assertIsNone(e.source)

    def test_loose_boundary_does_not_require_next_goal_and_uses_time_span(self):
        class Policy:
            boundary_ready=True
            def pause(self,*args):pass
        ctrl=VisualBatchController(Policy(),SPEC,placement_mode='contained_unstacked',localizer=object())
        ctrl.goal=goal();ctrl.phase='EXECUTE';ctrl._paused=False
        cookies=[cookie(p+[.0057,0,0]) for p in goal().target_positions_box]
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=LocalizationFailure('Target hidden')):
            for t in (.10,.15,.20):ctrl._verify(scene(cookies,t),None,True)
            self.assertEqual(len(ctrl.completed),0)
            ctrl._verify(scene(cookies,.30),None,True)
        self.assertEqual(len(ctrl.completed),1);self.assertEqual(ctrl.phase,'LOCATE')

    def test_repeated_frame_does_not_verify_or_request_unbounded_observations(self):
        class Policy:
            def pause(self,*args):pass
        ctrl=VisualBatchController(Policy(),SPEC,observation_mode='active-left-wrist',localizer=object())
        self.assertTrue(ctrl.request_observation('Source entry missing'))
        ctrl.observation_finished(None,100)
        ctrl.request_observation('Target empty missing');ctrl.observation_finished(None,100)
        ctrl.request_observation('Source entry missing')
        self.assertTrue(ctrl.finished);self.assertEqual(ctrl.total_observation_steps,200)

    def test_verified_progress_can_bridge_only_clear_static_occlusion(self):
        ctrl=VisualBatchController(object(),SPEC,placement_mode='contained_unstacked',localizer=object())
        cookies=[cookie(p) for p in goal().target_positions_box]
        ctrl._completion_scene=scene(cookies,1.)
        hidden=scene([],2.)
        self.assertTrue(ctrl.retained_evidence_valid(hidden,[0,0,.2],.05))
        self.assertEqual(len(hidden.target_cookies),0)  # no fake current detections
        self.assertFalse(ctrl.retained_evidence_valid(replace(hidden,timestamp=12.),[0,0,.2],.05))
        self.assertFalse(ctrl.retained_evidence_valid(hidden,[0,0,.08],.05))
        moved=scene([replace(cookies[0],position=cookies[0].position+[.006,0,0])],2.)
        self.assertFalse(ctrl.retained_evidence_valid(moved,[0,0,.2],.05))

    def test_observation_view_verification_waits_for_return_and_keeps_live_counts(self):
        class Policy:
            def pause(self,*args):pass
        ctrl=VisualBatchController(Policy(),SPEC,placement_mode='contained_unstacked',localizer=object())
        ctrl.goal=goal();ctrl.phase='OBSERVE';ctrl._after_observation='VERIFY_RELEASE'
        cookies=[cookie(p) for p in goal().target_positions_box]
        for t in (1.,1.05,1.10,1.15,1.20):ctrl.observe_placement_at_view(scene(cookies,t))
        self.assertEqual(len(ctrl.completed),0);self.assertEqual(ctrl.phase,'OBSERVE')
        ctrl.observation_finished(None,40)
        live=scene(cookies[:4],1.7)
        with patch('a3_batch_skills.visual_controller.make_visual_goal',side_effect=LocalizationFailure('Source occluded')):
            ctrl._verify(live,None,True,pad=[0,0,.2],gap=.05)
        self.assertEqual(len(ctrl.completed),1)
        self.assertEqual(ctrl.diagnostics['target_instances'],4)
        self.assertEqual(ctrl.diagnostics['acceptance_evidence_instances'],5)
        self.assertEqual(ctrl.diagnostics['acceptance_evidence_timestamp'],1.20)


if __name__=='__main__':unittest.main()
