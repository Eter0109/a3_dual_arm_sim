"""Short-lived planning evidence, never a forged current sensor packet."""
import numpy as np
from dataclasses import replace
from .rgbd_localization import LocalizationFailure
from .visual_goals import make_visual_goal


class PlanningEvidence:
    def __init__(self,ttl=10.):
        self.ttl=ttl;self.source=None;self.target=None;self.invalidations=[];self.target_reason='TARGET_EMPTY_UNOBSERVED'

    def invalidate(self,reason,region=None):
        if region in (None,'source'):self.source=None
        if region in (None,'target'):self.target=None
        self.invalidations.append(str(reason))

    def update(self,frame,scene,spec,index,column,*,safe_static,completed=(),known_occupancy=()):
        if not safe_static:
            self.invalidate('robot_not_clear_of_objects');return
        # make_visual_goal exposes its independently measured partial goals.
        from .visual_goals import measured_goal_parts
        source,target=measured_goal_parts(frame,scene,spec,index,column=column,completed=completed,known_occupancy=known_occupancy)
        from .visual_goals import measured_target_capacity
        if 'target_slots_local' in spec:
            self.target_reason='TARGET_EMPTY_UNOBSERVED' if measured_target_capacity(scene,spec,completed,known_occupancy) else 'NO_FREE_TARGET_REGION'
        if self.target is not None:
            from .planning import rectangle_overlap
            half=np.asarray(spec['cookie_half_size']);region=self.target['value'][0]
            for c in scene.target_cookies:
                p=(c.position-scene.target_box.position)@scene.target_box.rotation
                r=scene.target_box.rotation.T@c.rotation
                if any(rectangle_overlap(p,r,half,t,np.eye(3),half,margin=-.001) for t in region):
                    self.invalidate('target_occupancy_changed','target');break
        for name,value in (('source',source),('target',target)):
            old=getattr(self,name)
            if old:
                for a,b in ((old['scene'].source_box,scene.source_box),(old['scene'].target_box,scene.target_box)):
                    angle=np.arccos(np.clip((np.trace(a.rotation.T@b.rotation)-1)/2,-1,1))
                    if np.linalg.norm(a.position-b.position)>.003 or angle>np.deg2rad(2):
                        self.invalidate('box_pose_changed_or_calibration_conflict');break
                old=getattr(self,name)
                if old and name=='source' and source:
                    p=old['value'][0];q=source[0]
                    if p.shape!=q.shape or np.max(np.linalg.norm(p-q,axis=1))>.002:
                        self.invalidate('source_instances_moved','source')
            if value is not None:setattr(self,name,dict(value=value,scene=scene,timestamp=float(frame.timestamp)))

    def goal(self,scene,spec,index,now):
        from .visual_goals import VisualBatchGoal
        for name in ('source','target'):
            item=getattr(self,name)
            if item is None or not 0<=now-item['timestamp']<=self.ttl:
                raise LocalizationFailure(('SOURCE_ENTRY_UNOBSERVED: Source entry' if name=='source' else self.target_reason+': Target empty region')+' evidence unavailable or expired')
        positions,rotation,column,clear=self.source['value'];target,empty=self.target['value']
        source_time=self.source['timestamp'];target_time=self.target['timestamp']
        return VisualBatchGoal(index,positions,rotation,target,scene.target_box.position,scene.target_box.rotation,
            np.asarray(spec['cookie_half_size']),min(source_time,target_time),column,clear,empty,
            source_timestamp=source_time,target_timestamp=target_time)

    def provenance(self):
        return {name:(None if getattr(self,name) is None else getattr(self,name)['timestamp']) for name in ('source','target')}
