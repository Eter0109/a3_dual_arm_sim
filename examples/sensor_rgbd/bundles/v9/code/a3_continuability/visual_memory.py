"""Bounded, batch-local depth evidence; historical frames stay asynchronous.

Only static free-floor observations can complement a currently occluded region.
Current cookie poses, current rim geometry and release verification remain live.
No old frame is retimestamped or inserted into a synchronized RGBDViews packet.
"""
from dataclasses import dataclass
import copy
import numpy as np
from a3_batch_skills.visual_goals import floor_visibility,rectangle_samples,VisualBatchGoal
from a3_batch_skills.planning import target_regions
from a3_batch_skills.rgbd_localization import LocalizationFailure,select_visible_five


def live_floor_evidence(frame,box,xy,floor_offset,tolerance=.002):
    """Separate positive floor rays, low obstacles and higher occlusion.

    A high foreground arm is unknown floor, not proof that the floor is occupied.
    A low depth return inside the actual queried footprint vetoes old free space.
    Neither unknown rays nor a packing prior generate positive free evidence.
    """
    xy=np.asarray(xy,dtype=float)
    free=floor_visibility(frame,box,xy,floor_offset,tolerance,return_mask=True)
    world=np.c_[xy,np.full(len(xy),floor_offset)]@box.rotation.T+box.position
    transform=frame.world_from_optical
    optical=(world-transform[:3,3])@transform[:3,:3]
    projected=optical@frame.intrinsic.T
    pixel=np.rint(projected[:,:2]/np.maximum(projected[:,2:3],1e-8)-.5).astype(int)
    h,w=frame.depth_m.shape
    valid=(optical[:,2]>.01)&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    uv=pixel[valid];depth=frame.depth_m[uv[:,1],uv[:,0]]
    finite=np.isfinite(depth)
    uv=uv[finite];depth=depth[finite]
    hit=np.c_[uv+.5,np.ones(len(uv))]@np.linalg.inv(frame.intrinsic).T
    hit=hit*depth[:,None]
    local=(hit@transform[:3,:3].T+transform[:3,3]-box.position)@box.rotation
    low,high=xy.min(0),xy.max(0)
    near=(local[:,2]>floor_offset+.003)&(local[:,2]<floor_offset+.070)
    inside=np.all((local[:,:2]>=low-.0005)&(local[:,:2]<=high+.0005),axis=1)
    # Quantized ray samples can be duplicated. One distinct near-floor hit is
    # enough to withhold history; this intentionally favors reobservation.
    obstacle=bool(np.any(near&inside))
    return free,obstacle


@dataclass(eq=False)
class MemoryView:
    camera: str
    frame: object
    boxes: dict
    valid_kinds: set


class BatchVisualMemory:
    def __init__(self,*,max_age_s=15.,max_views=12,pose_tolerance_m=.002,
                 angle_tolerance_rad=np.deg2rad(1.),emit=None):
        if max_age_s<=0 or max_views<1:raise ValueError('Invalid memory bounds')
        self.max_age_s=float(max_age_s);self.max_views=int(max_views)
        self.pose_tolerance_m=pose_tolerance_m;self.angle_tolerance_rad=angle_tolerance_rad
        self.emit=emit or (lambda **row:None)
        self.batch=None;self.views=[];self.queries=0;self.history_uses=0
        self.captures=0;self.views_recorded=0

    def clear(self,reason):
        if self.views:self.emit(event='visual_memory_cleared',reason=reason,batch=self.batch,views=len(self.views))
        self.views=[]

    def begin_batch(self,batch):
        if self.batch!=batch:
            self.clear('batch_changed');self.batch=batch

    def invalidate(self,kind,reason):
        changed=False
        for view in self.views:
            if kind in view.valid_kinds:view.valid_kinds.remove(kind);changed=True
        if changed:self.emit(event='visual_memory_invalidated',kind=kind,reason=reason,batch=self.batch)

    def observe(self,packet,scene,*,batch,phase,stationary):
        if not stationary or phase not in ('RIGHT_SCAN','LOCATE','VIEW','BATCH_READY','OBSERVE_HOLD'):return
        self.begin_batch(batch)
        if abs(scene.timestamp-packet.timestamp)>1e-6:raise ValueError('Memory needs current observed rims')
        boxes={kind:copy.deepcopy(getattr(scene,kind+'_box')) for kind in ('source','target')}
        added=[]
        for name,frame in zip(('front','left_wrist','right_wrist'),packet.rgbd.frames):
            if np.isfinite(frame.depth_m).mean()<.10:continue
            valid={kind for kind,box in boxes.items() if box.supported_sides>=3}
            if not valid:continue
            previous=next((v for v in reversed(self.views) if v.camera==name),None)
            if previous is not None:
                a,b=previous.frame.world_from_optical,frame.world_from_optical
                angle=np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1))
                if np.linalg.norm(a[:3,3]-b[:3,3])<.008 and angle<.06:
                    self.views.remove(previous)
            # RGBDFrame owns immutable copied arrays. Keep the original capture
            # transform and time, never the camera's later encoder transform.
            self.views.append(MemoryView(name,frame,boxes,valid))
            self.views=self.views[-self.max_views:]
            added.append(name)
        if added:
            self.captures+=1;self.views_recorded+=len(added)
        self.emit(event='visual_memory_capture',batch=batch,capture_phase=phase,
            cameras=added,stored_views=len(self.views),timestamp=packet.timestamp)

    def fraction(self,packet,box,xy,floor_offset,*,kind,batch):
        if kind not in ('source','target'):raise ValueError(kind)
        self.begin_batch(batch);self.queries+=1
        live=np.zeros(len(xy),dtype=bool);obstacle=False
        for frame in packet.rgbd.frames:
            mask,blocked=live_floor_evidence(frame,box,xy,floor_offset)
            live|=mask;obstacle|=blocked
        original=float(live.mean());union=live.copy();ages=[];cameras=[]
        reason='current_view_sufficient' if original>=.90 else 'no_usable_history'
        rejected={};historical_floor=[]
        def reject(reason):rejected[reason]=rejected.get(reason,0)+1
        if obstacle:
            self.invalidate(kind,'current_near_floor_obstacle')
            reason='current_obstacle_veto'
        elif original<.90 and box.supported_sides>=3:
            for view in self.views:
                if kind not in view.valid_kinds:reject('invalid_kind');continue
                age=packet.timestamp-view.frame.timestamp
                if age<=1e-6:reject('same_time_or_future');continue
                if age>self.max_age_s:reject('expired');continue
                old=view.boxes[kind]
                angle=np.arccos(np.clip((np.trace(old.rotation.T@box.rotation)-1)/2,-1,1))
                if (np.linalg.norm(old.position-box.position)>self.pose_tolerance_m
                        or angle>self.angle_tolerance_rad):
                    view.valid_kinds.discard(kind);reject('changed_rim');continue
                mask=floor_visibility(view.frame,old,xy,floor_offset,return_mask=True)
                historical_floor.append(dict(camera=view.camera,age_s=float(age),fraction=float(mask.mean())))
                if not np.any(mask&~union):reject('no_additional_floor')
                if np.any(mask&~union):
                    union|=mask;ages.append(float(age));cameras.append(view.camera)
            if ages:reason='historical_observed_floor_complements_current_unknown'
        if ages:self.history_uses+=1
        result=dict(kind=kind,batch=batch,current_fraction=original,combined_fraction=float(union.mean()),
            history_used=bool(ages),historical_ages_s=ages,historical_cameras=cameras,
            current_obstacle_veto=obstacle,reason=reason,timestamp=packet.timestamp,
            stored_history_views=len(self.views),history_rejections=rejected,historical_floor=historical_floor)
        self.emit(event='visual_memory_query',**result)
        return result

    def make_goal(self,packet,scene,spec,index,*,column='auto',batch):
        """The baseline goal rules with an explicit asynchronous floor gate."""
        half=np.asarray(spec['cookie_half_size'])
        regions=target_regions(spec['target_slots_local'],spec['floor_offset_m']+half[2])
        if not 0<=index<len(regions):raise LocalizationFailure('Target batch index out of range')
        target=regions[index]
        xy=rectangle_samples(target[:,:2].min(0)-half[:2]+.001,
                             target[:,:2].max(0)+half[:2]-.001)
        evidence=self.fraction(packet,scene.target_box,xy,spec['floor_offset_m'],kind='target',batch=batch)
        empty=evidence['combined_fraction']
        if empty<.90:raise LocalizationFailure('Target region is occupied or not visibly empty')
        for col in sorted({c.column for c in scene.source_cookies}):
            if column!='auto' and col!=column:continue
            try:five=select_visible_five(scene,column=col)
            except LocalizationFailure:continue
            positions=np.asarray([c.position for c in five])
            local=(positions-scene.source_box.position)@scene.source_box.rotation
            low=np.array([local[:,0].mean()-.018,-scene.source_box.inner_half_size[1]+.001])
            high=np.array([local[:,0].mean()+.018,local[:,1].min()-half[1]-.0007])
            if high[1]<=low[1]:continue
            evidence=self.fraction(packet,scene.source_box,rectangle_samples(low,high),
                spec['source_floor_top_offset_m'],kind='source',batch=batch)
            clear=evidence['combined_fraction']
            if clear<.90:continue
            return VisualBatchGoal(index,positions,scene.source_box.rotation,target,scene.target_box.position,
                scene.target_box.rotation,half,packet.timestamp,col,clear,empty)
        raise LocalizationFailure('No five-cookie end has a visibly clear approach; acquire another view')
