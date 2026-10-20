"""RGB-D batch supervisor. No object truth, native score, or expert takeover."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .rgbd_localization import RGBDLocalizer,LocalizationFailure
from .visual_goals import make_visual_goal,visual_region_complete,visual_region_diagnostics

SUPERVISOR_VERSION="rgbd_release_pause_verify_v2"


class VisualBatchController:
    def __init__(self,policy,spec,*,column="auto",batch_steps=750,stable_observations=3,
                 verification_steps=60,context=None,localizer=None):
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

    def _verify(self,scene,frame,clear):
        current=visual_region_diagnostics(scene,self.goal)
        previous=[visual_region_diagnostics(scene,g) for g in self.completed]
        next_available=True;next_error=None
        if 5*(len(self.completed)+1)<self.spec["required_cookies"]:
            try:make_visual_goal(frame,scene,self.spec,len(self.completed)+1,column=self.column)
            except LocalizationFailure as exc:next_available=False;next_error=str(exc)
        self.diagnostics=dict(frame_timestamp=float(scene.timestamp),target_instances=len(scene.target_cookies),
            hand_clear_and_open=bool(clear),current_region=current,previous_regions=previous,
            next_goal_available=next_available,next_goal_error=next_error)
        # Observe release even when placement is wrong: slot agreement must not
        # be required to stop an expired goal from driving the next grasp.
        retained=0
        if self.completed and scene.target_cookies:
            actual=(np.asarray([c.position for c in scene.target_cookies])-scene.target_box.position)@scene.target_box.rotation
            known=np.concatenate([g.target_positions_box for g in self.completed])
            distance=np.linalg.norm(actual[:,None,:2]-known[None,:,:2],axis=2)
            a,b=linear_sum_assignment(distance)
            retained=int(np.sum((distance[a,b]<.004)&(abs(actual[a,2]-known[b,2])<.0025)))
        new_visible=len(scene.target_cookies)-retained
        self.diagnostics["new_or_displaced_target_instances"]=new_visible
        # A partially dropped batch also needs inspection, not another grasp.
        # A complete previous batch alone must not trigger a new release.
        release=clear and new_visible>0
        if self.phase=="EXECUTE" and release:
            self._pause();self.phase="VERIFY_RELEASE";self._verification_started=self.step
            self.events.append(dict(step=self.step,event="release_observed_queue_paused",index=self.goal.index))
        if self.phase!="VERIFY_RELEASE":return
        fresh=self._last_verified_timestamp is None or scene.timestamp>self._last_verified_timestamp+1e-6
        if not fresh:return
        self._last_verified_timestamp=scene.timestamp
        positions=np.asarray([c.position for c in scene.target_cookies]).reshape(-1,3)
        if not clear or not len(positions):
            self.stable=0;self._stable_positions=None;return
        if self._stable_positions is not None and len(positions)==len(self._stable_positions):
            distance=np.linalg.norm(positions[:,None]-self._stable_positions[None],axis=2)
            a,b=linear_sum_assignment(distance)
            stable_motion=bool(np.max(distance[a,b])<.0015)
        else:stable_motion=self._stable_positions is None
        self.stable=self.stable+1 if stable_motion else 1
        self._stable_positions=positions
        if self.stable<self.stable_observations:return
        reasons=[]
        if not current["complete"]:reasons.append("Released batch is outside its placement tolerance")
        if not all(r["complete"] for r in previous):reasons.append("Previously verified progress is no longer visible and valid")
        if not next_available:reasons.append("Next batch has no visibly safe goal: "+str(next_error))
        if reasons:
            self._fail("; ".join(reasons),recovery=True);return
        self.completed.append(self.goal)
        self.events.append(dict(step=self.step,event="visual_batch_verified",index=self.goal.index))
        self.goal=None;self.stable=0;self._stable_positions=None
        self._last_verified_timestamp=None;self._verification_started=None;self.batch_started=self.step
        self.phase="DONE" if 5*len(self.completed)==self.spec["required_cookies"] else "LOCATE"

    def act(self,observation,calibrations,*,frame=None,pad_midpoint_world=None,jaw_gap_m=None):
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
            if scene is not None:self.last_scene=scene
        if self.phase=="LOCATE":
            if scene is None:return None
            if self.completed and not all(visual_region_complete(scene,g) for g in self.completed):return None
            try:goal=make_visual_goal(frame,scene,self.spec,len(self.completed),column=self.column)
            except LocalizationFailure as exc:
                self.locate_attempts+=1
                if self.locate_attempts>=30:self._fail(str(exc))
                return None
            self.goal=goal;self.policy.start(goal,self.context);self._paused=False
            self.phase="EXECUTE";self.batch_started=self.step;self.locate_attempts=0
            self.events.append(dict(step=self.step,event="visual_batch_started",index=goal.index,
                source_column=goal.source_column,source_clear_fraction=goal.source_clear_fraction,
                target_empty_fraction=goal.target_empty_fraction))
        elif scene is not None:
            pad=np.asarray(pad_midpoint_world) if pad_midpoint_world is not None else None
            open_width=10*self.spec["cookie_half_size"][1]+.003
            clear=(pad is not None and pad.shape==(3,) and np.isfinite(pad).all()
                   and pad[2]>scene.target_box.rim_z+.120
                   and jaw_gap_m is not None and jaw_gap_m>open_width)
            self._verify(scene,frame,clear)
        # Pausing clears the policy's goal and remaining actions. Only a new
        # measured, verified batch can start it again; verification always holds.
        if self.phase!="EXECUTE":return None
        return self.policy.act(observation,calibrations)
