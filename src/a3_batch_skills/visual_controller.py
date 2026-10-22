"""RGB-D batch supervisor. No object truth, native score, or expert takeover."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .rgbd_localization import RGBDLocalizer,LocalizationFailure
from .visual_goals import make_visual_goal,visual_region_complete,visual_region_diagnostics

SUPERVISOR_VERSION="rgbd_release_pause_verify_v3_retreat_complete"


class VisualBatchController:
    def __init__(self,policy,spec,*,column="auto",batch_steps=750,stable_observations=3,
                 verification_steps=60,context=None,localizer=None,placement_mode='strict',
                 observation_mode='passive',modern=False):
        self.policy,self.spec,self.column=policy,spec,column
        self.batch_steps,self.stable_observations=int(batch_steps),int(stable_observations)
        self.verification_steps=int(verification_steps)
        if min(self.batch_steps,self.stable_observations,self.verification_steps)<1:
            raise ValueError("Invalid visual control budget")
        self.localizer=localizer or RGBDLocalizer(spec)
        self.context=context
        self.goal=None
        self.completed=[]
        self.step=self.batch_started=self.locate_attempts=self.stable=0
        self.phase="LOCATE"
        self.failure_reason=None
        self.last_scene=None
        self.events=[]
        self.diagnostics={}
        self._last_verified_timestamp=None
        self._stable_positions=None
        self._paused=True
        self._verification_started=None
        if placement_mode not in ('strict','contained_unstacked') or observation_mode not in ('passive','active-left-wrist'):
            raise ValueError('Invalid observation/placement mode')
        self.placement_mode=placement_mode;self.observation_mode=observation_mode
        self.modern=bool(modern or placement_mode!='strict' or observation_mode!='passive')
        from .visual_evidence import PlanningEvidence
        self.evidence=PlanningEvidence();self.observation_attempts=0;self.observation_steps=0
        self.total_observation_steps=0;self.observation_reason=None;self._after_observation=None
        self._stable_since=None;self._stable_corners=None;self._last_locate_timestamp=None
        self._completion_scene=None
        self._scan_placement_scene=None;self._scan_anchor=None;self._scan_since=None;self._scan_last=None;self._scan_count=0

    def retained_evidence_valid(self,scene,pad,gap,proof=None):
        """Bridge an occluded view only during a bounded non-contact interval.

        This proof is planning memory with its original timestamp. It is never
        inserted into current images, current instance counts or model state.
        """
        old=proof if proof is not None else self._completion_scene
        if old is None or not 0<=scene.timestamp-old.timestamp<=10:return False
        if pad is None or gap is None or gap<=10*self.spec['cookie_half_size'][1]+.003:return False
        if np.asarray(pad)[2]<=max(scene.source_box.rim_z,scene.target_box.rim_z)+.065:return False
        for name in ('source_box','target_box'):
            a,b=getattr(old,name),getattr(scene,name)
            if np.linalg.norm(a.position-b.position)>.003 or np.trace(a.rotation.T@b.rotation)<1+2*np.cos(np.deg2rad(2)):return False
        if len(scene.target_cookies)>len(old.target_cookies):return False
        if scene.target_cookies:
            from .visual_acceptance import contained_report
            if not contained_report(scene,self.spec,len(scene.target_cookies))['complete']:return False
            q=np.asarray([c.position for c in scene.target_cookies]);p=np.asarray([c.position for c in old.target_cookies])
            distance=np.linalg.norm(q[:,None]-p[None],axis=2);a,b=linear_sum_assignment(distance)
            if np.max(distance[a,b])>.002:return False
            from .visual_acceptance import stable_surface_points
            corners=stable_surface_points(scene,self.spec['cookie_half_size'])[a]
            prior=stable_surface_points(old,self.spec['cookie_half_size'])[b]
            pair=np.linalg.norm(corners[:,:,None,:]-prior[:,None,:,:],axis=-1)
            if np.max(np.min(pair,axis=-1))>.002:return False
        return True

    def observe_placement_at_view(self,scene):
        """Certify a stable placement at a settled view, before returning.

        No task transition is allowed here: the observer must return first.
        """
        from .visual_acceptance import contained_report,stable_surface_points
        report=contained_report(scene,self.spec,5*(len(self.completed)+1),
            unknown=getattr(self.localizer,'target_unknown',False))
        if not report['complete']:
            self._scan_anchor=None;self._scan_count=0;self._scan_since=None;return
        if self._scan_last is not None and scene.timestamp<=self._scan_last+1e-6:return
        self._scan_last=scene.timestamp;corners=stable_surface_points(scene,self.spec['cookie_half_size'])
        stable=False
        if self._scan_anchor is not None and corners.shape==self._scan_anchor.shape:
            distance=np.linalg.norm(corners.mean(1)[:,None]-self._scan_anchor.mean(1)[None],axis=2)
            a,b=linear_sum_assignment(distance)
            pair=np.linalg.norm(corners[a,:,None]-self._scan_anchor[b,None,:],axis=-1)
            stable=bool(np.max(np.min(pair,axis=-1))<.0015)
        if not stable:self._scan_anchor=corners.copy();self._scan_since=scene.timestamp;self._scan_count=0
        self._scan_count+=1
        if self._scan_count>=3 and scene.timestamp-self._scan_since>=.2-1e-6:
            self._scan_placement_scene=scene
            self.events.append(dict(step=self.step,event='placement_verified_at_observation_view',
                evidence_timestamp=scene.timestamp,observed_total=len(scene.target_cookies)))

    def _accept_batch(self,scene,source='current_view'):
        self.completed.append(self.goal)
        if self.placement_mode=='contained_unstacked':self._completion_scene=scene
        self.events.append(dict(step=self.step,event='visual_batch_verified',index=self.goal.index,
            placement_mode=self.placement_mode,observed_total=len(scene.target_cookies),
            acceptance_source=source,acceptance_evidence_timestamp=scene.timestamp))
        self.goal=None;self.stable=0;self._stable_positions=None
        self._last_verified_timestamp=None;self._verification_started=None;self.batch_started=self.step
        self._stable_corners=None;self._stable_since=None;self.observation_attempts=0;self.observation_steps=0
        self._scan_placement_scene=None;self._scan_anchor=None;self._scan_since=None;self._scan_last=None;self._scan_count=0
        self.evidence.invalidate('release_and_batch_boundary')
        self.phase='DONE' if 5*len(self.completed)==self.spec['required_cookies'] else 'LOCATE'

    def request_observation(self,reason):
        if 'Conflicting' in str(reason):self.evidence.invalidate('calibration_conflict')
        if self.observation_mode!='active-left-wrist':return False
        if self.observation_attempts>=2 or self.observation_steps>=200:
            self._fail('OBSERVATION_EXHAUSTED: '+str(reason),recovery=self.phase=='VERIFY_RELEASE');return True
        self._after_observation=self.phase;self.observation_reason=str(reason)
        self.observation_attempts+=1;self._pause();self.phase='OBSERVE'
        self.events.append(dict(step=self.step,event='observation_requested',reason=str(reason),
            attempt=self.observation_attempts,evidence=self.evidence.provenance()))
        return True

    def observation_finished(self,error,steps):
        self.observation_steps+=steps;self.total_observation_steps+=steps
        self.events.append(dict(step=self.step,event='observation_finished',error=error,steps=steps))
        if error and any(x in error.lower() for x in ('return','clearance','clear, open','path')):
            self._fail('OBSERVATION_UNSAFE_OR_RETURN_FAILED: '+error,recovery=True);return
        self.phase=self._after_observation;self._last_locate_timestamp=None
        if self.phase=='VERIFY_RELEASE':self._verification_started=self.step

    @property
    def finished(self):return self.phase in ("DONE","FAILED","RECOVERY_REQUIRED")

    def require_recovery(self,reason):self._fail(reason,recovery=True)

    def _pause(self):
        if not self._paused:self.policy.pause(self.context)
        self._paused=True

    def _fail(self,reason,*,recovery=False):
        self.phase="RECOVERY_REQUIRED" if recovery else "FAILED"
        self.failure_reason=str(reason)
        self.events.append(dict(step=self.step,event=self.phase.lower(),reason=self.failure_reason,
                                diagnostics=dict(self.diagnostics)))
        self._pause()

    def _verify(self,scene,frame,clear,pad=None,gap=None):
        current=visual_region_diagnostics(scene,self.goal)
        previous=[visual_region_diagnostics(scene,g) for g in self.completed]
        loose=self.placement_mode=='contained_unstacked'
        if loose:
            from .visual_acceptance import contained_report
            current=contained_report(scene,self.spec,5*(len(self.completed)+1),
                unknown=getattr(self.localizer,'target_unknown',False));previous=[]
        next_available=True;next_error=None
        if 5*(len(self.completed)+1)<self.spec["required_cookies"]:
            try:make_visual_goal(frame,scene,self.spec,len(self.completed)+1,column=self.column)
            except LocalizationFailure as exc:next_available=False;next_error=str(exc)
        self.diagnostics=dict(frame_timestamp=float(scene.timestamp),target_instances=len(scene.target_cookies),
            hand_clear_and_open=bool(clear),current_region=current,previous_regions=previous,
            source_visible_by_column={str(col):sum(c.column==col for c in scene.source_cookies)
                                      for col in sorted({c.column for c in scene.source_cookies})},
            placement_boundary_ready=bool(getattr(self.policy,'boundary_ready',True)),
            next_goal_available=next_available,next_goal_error=next_error)
        if (loose and clear and self.phase=='VERIFY_RELEASE' and self._scan_placement_scene is not None
                and self.retained_evidence_valid(scene,pad,gap,proof=self._scan_placement_scene)):
            self.diagnostics.update(acceptance_source='settled_observation_view',
                acceptance_evidence_timestamp=self._scan_placement_scene.timestamp,
                acceptance_evidence_instances=len(self._scan_placement_scene.target_cookies))
            self._accept_batch(self._scan_placement_scene,'settled_observation_view');return
        # Observe release even when placement is wrong: slot agreement must not
        # be required to stop an expired goal from driving the next grasp.
        retained=0
        if loose:retained=5*len(self.completed)
        elif self.completed and scene.target_cookies:
            actual=(np.asarray([c.position for c in scene.target_cookies])-scene.target_box.position)@scene.target_box.rotation
            known=np.concatenate([g.target_positions_box for g in self.completed])
            distance=np.linalg.norm(actual[:,None,:2]-known[None,:,:2],axis=2)
            a,b=linear_sum_assignment(distance)
            retained=int(np.sum((distance[a,b]<.004)&(abs(actual[a,2]-known[b,2])<.0025)))
        new_visible=len(scene.target_cookies)-retained
        self.diagnostics["new_or_displaced_target_instances"]=new_visible
        # A partially dropped batch also needs inspection, not another grasp.
        # A complete previous batch alone must not trigger a new release.
        release=clear and new_visible>0 and getattr(self.policy,'boundary_ready',True)
        if self.phase=="EXECUTE" and release:
            self._pause();self.phase="VERIFY_RELEASE";self._verification_started=self.step
            self.events.append(dict(step=self.step,event="release_observed_queue_paused",index=self.goal.index))
        if self.phase!="VERIFY_RELEASE":return
        fresh=self._last_verified_timestamp is None or scene.timestamp>self._last_verified_timestamp+1e-6
        if not fresh:return
        self._last_verified_timestamp=scene.timestamp
        positions=np.asarray([c.position for c in scene.target_cookies]).reshape(-1,3)
        if not clear or not len(positions):
            self.stable=0;self._stable_positions=None;self._stable_since=None;return
        if self._stable_positions is not None and len(positions)==len(self._stable_positions):
            distance=np.linalg.norm(positions[:,None]-self._stable_positions[None],axis=2)
            a,b=linear_sum_assignment(distance)
            stable_motion=bool(np.max(distance[a,b])<.0015)
        else:stable_motion=self._stable_positions is None
        if loose:
            from .visual_acceptance import stable_surface_points
            corners=stable_surface_points(scene,self.spec['cookie_half_size'])
            if self._stable_corners is not None and self._stable_corners.shape==corners.shape:
                distances=np.linalg.norm(corners.mean(1)[:,None]-self._stable_corners.mean(1)[None],axis=2)
                a,b=linear_sum_assignment(distances)
                pair=np.linalg.norm(corners[a,:,None,:]-self._stable_corners[b,None,:,:],axis=-1)
                stable_motion=bool(np.max(np.min(pair,axis=-1))<.0015)
            else:stable_motion=self._stable_corners is None
            if not stable_motion or self._stable_corners is None:self._stable_corners=corners.copy()
        self.stable=self.stable+1 if stable_motion else 1
        if self._stable_since is None or not stable_motion:self._stable_since=scene.timestamp
        self._stable_positions=positions
        if self.stable<self.stable_observations:return
        if loose and scene.timestamp-self._stable_since<.2-1e-6:return
        if loose and current['unknown']:
            if self.request_observation('Target instances unresolved or occluded after release'):return
            return  # passive mode waits for its bounded verification timeout
        reasons=[]
        if not current["complete"]:reasons.append(('Released batch failed contained_unstacked: '+','.join(current['reasons']))
            if loose else 'Released batch is outside its placement tolerance')
        if not all(r["complete"] for r in previous):reasons.append("Previously verified progress is no longer visible and valid")
        if not next_available and not self.modern:reasons.append("Next batch has no visibly safe goal: "+str(next_error))
        if reasons:
            self._fail("; ".join(reasons),recovery=True);return
        self._accept_batch(scene)

    def act(self,observation,calibrations,*,frame=None,pad_midpoint_world=None,jaw_gap_m=None,observation_settled=False):
        if self.finished:return None
        self.step+=1
        if self.phase=="VERIFY_RELEASE" and self.step-self._verification_started>self.verification_steps:
            self._fail("Release could not be visually verified within its time budget",recovery=True);return None
        if self.step-self.batch_started>self.batch_steps:
            self._fail("Single-batch time budget exceeded");return None
        scene=None
        if frame is not None:
            try:scene=self.localizer.observe(frame)
            except LocalizationFailure as exc:
                self.stable=0;self._stable_positions=None
                self.diagnostics=dict(localization_error=str(exc))
                if self.modern and 'Conflicting' in str(exc):
                    self.evidence.invalidate('calibration_conflict');self._completion_scene=None
                if self.modern and self.phase=='LOCATE':
                    fresh=self._last_locate_timestamp is None or frame.timestamp>self._last_locate_timestamp+1e-6
                    if fresh:self.locate_attempts+=1;self._last_locate_timestamp=frame.timestamp
                    if (self.last_scene is not None and 0<=frame.timestamp-self.last_scene.timestamp<=10
                            and self.request_observation('LOCALIZATION_REOBSERVE: '+str(exc))):return None
                    if self.locate_attempts>=30:self._fail('LOCALIZATION_UNAVAILABLE: '+str(exc))
            if scene is not None:self.last_scene=scene
        if self.modern and scene is not None and self.phase in ('LOCATE','OBSERVE'):
            pad=np.asarray(pad_midpoint_world) if pad_midpoint_world is not None else np.zeros(3)
            safe=bool(pad[2]>max(scene.source_box.rim_z,scene.target_box.rim_z)+.065 and
                jaw_gap_m is not None and jaw_gap_m>10*self.spec['cookie_half_size'][1]+.003)
            retained=self.retained_evidence_valid(scene,pad,jaw_gap_m) if self.placement_mode=='contained_unstacked' else False
            if self._completion_scene is not None and not retained:self._completion_scene=None
            occupancy=self._completion_scene.target_cookies if retained else ()
            try:self.evidence.update(frame,scene,self.spec,len(self.completed),self.column,
                safe_static=safe,completed=self.completed,known_occupancy=occupancy)
            except LocalizationFailure as exc:self.diagnostics['evidence_error']=str(exc)
        if self.phase=='OBSERVE':
            if (scene is not None and observation_settled and self._after_observation=='VERIFY_RELEASE'
                    and self.placement_mode=='contained_unstacked'):self.observe_placement_at_view(scene)
            return None
        if self.phase=="LOCATE":
            if scene is None:return None
            if self.completed:
                if self.placement_mode=='contained_unstacked':
                    from .visual_acceptance import contained_report
                    current_valid=contained_report(scene,self.spec,5*len(self.completed),
                        unknown=getattr(self.localizer,'target_unknown',False))['complete']
                    retained=self.retained_evidence_valid(scene,pad_midpoint_world,jaw_gap_m)
                    if not current_valid and not retained:
                        if self.request_observation('Target previous progress is unresolved'):return None
                        return None
                elif not all(visual_region_complete(scene,g) for g in self.completed):return None
            try:
                goal=(self.evidence.goal(scene,self.spec,len(self.completed),frame.timestamp) if self.modern else
                    make_visual_goal(frame,scene,self.spec,len(self.completed),column=self.column))
            except LocalizationFailure as exc:
                fresh=not self.modern or self._last_locate_timestamp is None or frame.timestamp>self._last_locate_timestamp+1e-6
                if fresh:self.locate_attempts+=1;self._last_locate_timestamp=frame.timestamp
                if self.modern and self.request_observation(str(exc)):return None
                if self.locate_attempts>=30:self._fail(str(exc))
                return None
            self.goal=goal;self.policy.start(goal,self.context);self._paused=False
            self.phase="EXECUTE"
            if not self.modern:self.batch_started=self.step
            self.locate_attempts=0;self.evidence.invalidate('model_operation_started')
            retained_time=self._completion_scene.timestamp if self._completion_scene is not None else None
            self._completion_scene=None
            self.events.append(dict(step=self.step,event="visual_batch_started",index=goal.index,
                source_column=goal.source_column,source_clear_fraction=goal.source_clear_fraction,
                target_empty_fraction=goal.target_empty_fraction,
                source_evidence_timestamp=goal.source_timestamp,target_evidence_timestamp=goal.target_timestamp,
                retained_progress_evidence_timestamp=retained_time))
        elif scene is not None:
            pad=np.asarray(pad_midpoint_world) if pad_midpoint_world is not None else None
            open_width=10*self.spec["cookie_half_size"][1]+.003
            clear=(pad is not None and pad.shape==(3,) and np.isfinite(pad).all()
                   and pad[2]>scene.target_box.rim_z+.120
                   and jaw_gap_m is not None and jaw_gap_m>open_width)
            self._verify(scene,frame,clear,pad=pad,gap=jaw_gap_m)
        # Pausing clears the policy's goal and remaining actions. Only a new
        # measured, verified batch can start it again; verification always holds.
        if self.phase!="EXECUTE":return None
        return self.policy.act(observation,calibrations)
