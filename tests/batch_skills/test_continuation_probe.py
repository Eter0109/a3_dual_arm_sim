from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from a3_batch_skills.execution_settings import set_action_steps,target_planning_settings
from a3_batch_skills.visual_goals import measured_goal_parts
from a3_batch_skills.visual_evidence import PlanningEvidence
from a3_batch_skills.rgbd_localization import BoxEstimate,CookieEstimate,VisualScene


HALF=np.array([.025,.0031666667,.0125])
SLOTS=np.array([[x,y] for x in (-.028,.028) for y in np.arange(-2,3)*2*HALF[1]])
SPEC=dict(cookie_half_size=HALF.tolist(),target_slots_local=SLOTS.tolist(),
    floor_offset_m=.006,source_floor_top_offset_m=.006,required_cookies=10)


def example():
    box=BoxEstimate(np.zeros(3),np.eye(3),np.array([.057,.029]),.04,4,0.)
    cookies=tuple(CookieEstimate(np.array([-.012,y,.0185]),np.eye(3),None,1.,0.)
                  for y in np.arange(-2,3)*2*HALF[1])
    source=tuple(CookieEstimate(np.array([0.,y,.0185]),np.eye(3),1,1.,0.)
                 for y in np.arange(-2,3)*2*HALF[1])
    return VisualScene(box,box,source,cookies,1.,0.)


class ContinuationProbeTests(unittest.TestCase):
    def test_lower_floor_threshold_alone_cannot_override_measured_overlap(self):
        spec=dict(SPEC,target_floor_fraction=.80)
        with patch('a3_batch_skills.visual_goals.floor_visibility',return_value=.84):
            _,target=measured_goal_parts(SimpleNamespace(timestamp=1.),example(),spec,1)
        self.assertIsNone(target)

    def test_explicit_contact_probe_allows_bounded_overlap_and_reports_real_floor(self):
        spec=dict(SPEC,target_floor_fraction=.80,target_overlap_tolerance_m=.012)
        # Exclude the region already assigned to the first batch.
        first=np.c_[SLOTS[:5],np.full(5,.0185)]
        completed=(SimpleNamespace(target_positions_box=first),)
        with patch('a3_batch_skills.visual_goals.floor_visibility',return_value=.84):
            _,target=measured_goal_parts(SimpleNamespace(timestamp=1.),example(),spec,1,completed=completed)
        self.assertIsNotNone(target)
        np.testing.assert_allclose(target[0][:,0],.028)
        self.assertEqual(target[1],.84)

    def test_contact_allowance_never_fills_unobserved_floor(self):
        spec=dict(SPEC,target_floor_fraction=.80,target_overlap_tolerance_m=.012)
        with patch('a3_batch_skills.visual_goals.floor_visibility',return_value=0.):
            _,target=measured_goal_parts(SimpleNamespace(timestamp=1.),example(),spec,1)
        self.assertIsNone(target)

    def test_contact_settings_use_same_rule_when_revalidating_cached_evidence(self):
        scene=example();spec=dict(SPEC,target_floor_fraction=.80,target_overlap_tolerance_m=.012)
        target=np.c_[SLOTS[5:],np.full(5,.0185)]
        evidence=PlanningEvidence()
        with patch('a3_batch_skills.visual_goals.floor_visibility',return_value=.84):
            for t in (1.,1.1):
                evidence.update(SimpleNamespace(timestamp=t),replace(scene,timestamp=t),spec,1,'auto',safe_static=True,
                    completed=(SimpleNamespace(target_positions_box=np.c_[SLOTS[:5],np.full(5,.0185)]),))
        self.assertIsNotNone(evidence.target)
        self.assertNotIn('target_occupancy_changed',evidence.invalidations)

    def test_shared_model_horizon_override_crosses_process_interface(self):
        from a3_batch_skills.visual_suite import RemotePolicy
        class Pipe:
            def __init__(self):self.request=None
            def send(self,request):self.request=request
            def recv(self):
                return dict(ok=True,value=dict(n_action_steps=16,chunk_size=50)
                    if self.request['op']=='new' else self.request['n_action_steps'])
        pipe=Pipe();policy=RemotePolicy(pipe,123)
        set_action_steps(policy,4)
        self.assertEqual(pipe.request,dict(op='configure_execution',n_action_steps=4))
        self.assertEqual(policy.config.n_action_steps,4)
        with self.assertRaises(ValueError):set_action_steps(policy,51)

    def test_session_horizon_change_preserves_other_sessions_and_shared_weights(self):
        from a3_batch_skills.visual_suite import clone_session
        class Policy:
            def __init__(self):self.model=object();self.config=SimpleNamespace(chunk_size=50,n_action_steps=16)
            def reset(self):self.queue=[]
        policy=Policy();base=SimpleNamespace(policy=policy,config=policy.config,pre={},post={})
        a=clone_session(base);b=clone_session(base)
        set_action_steps(a,4)
        self.assertEqual((a.policy.config.n_action_steps,b.policy.config.n_action_steps,base.config.n_action_steps),(4,16,16))
        self.assertIs(a.policy.model,b.policy.model)

    def test_batch_handoff_cannot_start_next_model_before_it_finishes(self):
        from a3_batch_skills.visual_controller import VisualBatchController
        class Policy:
            def act(self,*args):raise AssertionError('Model called before handoff')
        ctrl=VisualBatchController(Policy(),SPEC,placement_mode='contained_unstacked',
            batch_handoff='initial-open',localizer=object())
        ctrl.goal=SimpleNamespace(index=0);ctrl._accept_batch(example())
        self.assertEqual(ctrl.phase,'HANDOFF')
        self.assertIsNone(ctrl.act({},{}))
        ctrl.handoff_finished(None,45)
        self.assertEqual(ctrl.phase,'LOCATE');self.assertEqual(ctrl.total_observation_steps,45)
        ctrl.phase='HANDOFF';ctrl.handoff_finished('return failed',20)
        self.assertTrue(ctrl.finished)

    def test_observer_handoff_keeps_gripper_open_and_requires_measured_settling(self):
        from a3_batch_skills.active_observation import LeftWristObservation
        observer=object.__new__(LeftWristObservation)
        observer._phase='HANDOFF';observer.target=np.zeros(7);observer._stable=0
        observer.used_steps=0;observer.limit=100;observer.active=True;observer.done=False;observer.error=None
        observer.events=[];observer._safe=lambda q:True
        state=np.r_[np.full(7,.1),.4,np.zeros(8)]
        action=observer.act(state,0.)
        self.assertAlmostEqual(action[7],.425)
        self.assertFalse(observer.done)
        for i in range(2):observer.act(np.r_[np.zeros(7),1.,np.zeros(8)],1.+i*.05)
        self.assertFalse(observer.done)
        observer.act(np.r_[np.zeros(7),1.,np.zeros(8)],1.1)
        self.assertTrue(observer.done)
        self.assertEqual(observer.events[-1]['event'],'boundary_handoff_returned')

    def test_tolerance_limits_and_legacy_defaults(self):
        self.assertEqual(target_planning_settings({}),(.90,.001))
        for spec in [dict(target_floor_fraction=.79),dict(target_overlap_tolerance_m=.013)]:
            with self.assertRaises(ValueError):target_planning_settings(spec)


if __name__=='__main__':unittest.main()
