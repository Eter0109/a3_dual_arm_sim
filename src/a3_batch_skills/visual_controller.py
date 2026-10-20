"""One neural policy per short batch, with an RGB-D-only boundary supervisor.

Inputs: calibrated image/depth packets, robot proprioception/FK and dimensions.
No simulator objects, contact graph, native score, or teacher can enter decisions.
"""
import numpy as np
from .rgbd_localization import RGBDLocalizer,LocalizationFailure
from .visual_goals import make_visual_goal,visual_region_complete


class VisualBatchController:
    def __init__(self,policy,spec,*,column="auto",batch_steps=750,stable_observations=3,
                 context=None,localizer=None):
        self.policy,self.spec,self.column=policy,spec,column
        self.batch_steps,self.stable_observations=int(batch_steps),int(stable_observations)
        if min(self.batch_steps,self.stable_observations)<1:raise ValueError("Invalid visual control budget")
        self.localizer=localizer or RGBDLocalizer(spec)
        self.context=context
        self.goal=None
        self.completed=[]
        self.step=0
        self.batch_started=0
        self.locate_attempts=0
        self.stable=0
        self.phase="LOCATE"
        self.failure_reason=None
        self.last_scene=None
        self.events=[]
        self._last_verified_timestamp=None
        self._stable_positions=None

    @property
    def finished(self):return self.phase in ("DONE","FAILED")

    def _fail(self,reason):
        self.phase="FAILED";self.failure_reason=str(reason)
        self.events.append(dict(step=self.step,event="failed",reason=self.failure_reason))
        self.policy.pause(self.context)

    def act(self,observation,calibrations,*,frame=None,pad_midpoint_world=None,jaw_gap_m=None):
        if self.finished:return None
        self.step+=1
        if self.step-self.batch_started>self.batch_steps:
            self._fail("Single-batch time budget exceeded");return None
        scene=None
        if frame is not None:
            try:scene=self.localizer.observe(frame)
            except LocalizationFailure:
                self.stable=0
                self._stable_positions=None
            if scene is not None:self.last_scene=scene
        if self.phase=="LOCATE":
            if scene is None:return None
            if self.completed and not all(visual_region_complete(scene,g) for g in self.completed):return None
            try:goal=make_visual_goal(frame,scene,self.spec,len(self.completed),column=self.column)
            except LocalizationFailure as exc:
                self.locate_attempts+=1
                if self.locate_attempts>=30:self._fail(str(exc))
                return None
            self.goal=goal
            self.policy.start(goal,self.context)
            self.phase="EXECUTE"
            self.batch_started=self.step
            self.locate_attempts=0
            self.events.append(dict(step=self.step,event="visual_batch_started",index=goal.index,
                source_column=goal.source_column,source_clear_fraction=goal.source_clear_fraction,
                target_empty_fraction=goal.target_empty_fraction))
        elif scene is not None:
            # Kinematics/gripper opening are measurable robot signals. Stable
            # image detections alone cannot establish that a held batch released.
            pad=np.asarray(pad_midpoint_world) if pad_midpoint_world is not None else None
            open_width=10*self.spec["cookie_half_size"][1]+.003
            clear=(pad is not None and pad.shape==(3,) and np.isfinite(pad).all()
                   and pad[2]>scene.target_box.rim_z+.120
                   and jaw_gap_m is not None and jaw_gap_m>open_width)
            all_goals=[*self.completed,self.goal]
            verified=clear and all(visual_region_complete(scene,g) for g in all_goals)
            if verified and 5*len(all_goals)<self.spec["required_cookies"]:
                try:make_visual_goal(frame,scene,self.spec,len(all_goals),column=self.column)
                except LocalizationFailure:verified=False
            positions=np.asarray([c.position for c in scene.target_cookies])
            # Different-time confirmations and geometric stability are required;
            # reading the same frozen image three times cannot finish a batch.
            fresh=self._last_verified_timestamp is None or scene.timestamp>self._last_verified_timestamp+1e-6
            if verified and fresh:
                if self._stable_positions is not None and len(positions)==len(self._stable_positions):
                    from scipy.optimize import linear_sum_assignment
                    distance=np.linalg.norm(positions[:,None]-self._stable_positions[None],axis=2)
                    a,b=linear_sum_assignment(distance)
                    stable_motion=bool(np.max(distance[a,b])<.0015)
                else:stable_motion=True
                self.stable=self.stable+1 if stable_motion else 1
                self._stable_positions=positions
                self._last_verified_timestamp=scene.timestamp
            elif not verified:
                self.stable=0;self._stable_positions=None
            if self.stable>=self.stable_observations:
                self.completed.append(self.goal)
                self.events.append(dict(step=self.step,event="visual_batch_verified",index=self.goal.index))
                self.policy.pause(self.context)
                self.goal=None;self.stable=0;self._stable_positions=None
                self._last_verified_timestamp=None;self.batch_started=self.step
                self.phase="DONE" if 5*len(self.completed)==self.spec["required_cookies"] else "LOCATE"
                return None
        return self.policy.act(observation,calibrations)
