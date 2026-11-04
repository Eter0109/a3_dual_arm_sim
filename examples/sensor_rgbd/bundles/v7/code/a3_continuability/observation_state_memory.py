"""Observed scene records and asynchronous floor evidence for preparation only.

Objects are measured by the current RGB-D localizer. These records never read
simulator objects/contacts, invent depth, or replace current grasp positions.
"""
from dataclasses import dataclass
import copy
import numpy as np
from a3_batch_skills.rgbd_localization import LocalizationFailure, select_visible_five
from a3_batch_skills.visual_goals import rectangle_samples, VisualBatchGoal
from a3_batch_skills.planning import target_regions
from .visual_memory import BatchVisualMemory, MemoryView


class ObservationRequirement(LocalizationFailure):
    """Actionable missing measurement, independent of exception text."""
    def __init__(self, missing_kind, region_index, *, source_fraction=0.,
                 target_fraction=0., has_current_five=False, evidence=None):
        if missing_kind not in ('source_entry', 'target_empty'):
            raise ValueError('Unknown observation requirement')
        self.missing_kind = missing_kind
        self.region_index = int(region_index)
        self.source_fraction = float(source_fraction)
        self.target_fraction = float(target_fraction)
        self.has_current_five = bool(has_current_five)
        self.coverage_by_kind = dict(source_entry=self.source_fraction,
                                     target_empty=self.target_fraction)
        self.evidence = copy.deepcopy(evidence or {})
        super().__init__('Need observed '+missing_kind+' for target region '+str(region_index)
            +' (source floor %.3f; target floor %.3f; current five %s)'
            % (self.source_fraction,self.target_fraction,self.has_current_five))


@dataclass
class SceneObservationRecord:
    batch: int
    timestamp: float
    phase: str
    boxes: dict
    source_cookies: tuple
    target_cookies: tuple
    camera_records: tuple
    valid_kinds: set


def _immutable_array(value):
    result=np.asarray(value).copy()
    result.setflags(write=False)
    return result


class SceneStateMemory(BatchVisualMemory):
    """Independent preparation observations; same guarded floor-ray storage."""
    is_scene_state_memory = True
    CAPTURE_PHASES = ('RIGHT_SCAN','LOCATE','VIEW','BATCH_READY','OBSERVE_HOLD')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene_observations=[]
        self.scene_records=0
        self.goal_queries=0
        self.goal_accepts=0
        self.history_supported_goals=0
        self.last_gate_evidence=None
        self.last_requirement=None

    def clear(self, reason):
        super().clear(reason)
        self.scene_observations=[]
        self.last_gate_evidence=None
        self.last_requirement=None

    def invalidate(self, kind, reason):
        super().invalidate(kind,reason)
        for record in self.scene_observations:
            record.valid_kinds.discard(kind)

    def observe(self, packet, scene, *, batch, phase, stationary):
        if not stationary or phase not in self.CAPTURE_PHASES:
            return
        # No source/target gate is evaluated before capture. A useful observation
        # of one side remains recordable even if the other side is occluded.
        self.begin_batch(batch)
        if abs(scene.timestamp-packet.timestamp)>1e-6:
            raise ValueError('Scene records require current observed geometry')
        supported={kind for kind in ('source','target')
            if getattr(scene,kind+'_box').supported_sides>=3}
        cameras=[]
        for name,frame in zip(('front','left_wrist','right_wrist'),packet.rgbd.frames):
            if np.isfinite(frame.depth_m).mean()<.10:
                continue
            cameras.append(dict(camera=name,timestamp=float(frame.timestamp),
                world_from_optical=_immutable_array(frame.world_from_optical),
                intrinsic=_immutable_array(frame.intrinsic)))
        duplicate=any(record.timestamp==float(packet.timestamp)
            for record in self.scene_observations)
        if supported and cameras and not duplicate:
            record=SceneObservationRecord(batch=int(batch),timestamp=float(packet.timestamp),
                phase=phase,boxes={kind:copy.deepcopy(getattr(scene,kind+'_box'))
                    for kind in ('source','target')},
                source_cookies=copy.deepcopy(tuple(getattr(scene,'source_cookies',()))),
                target_cookies=copy.deepcopy(tuple(getattr(scene,'target_cookies',()))),
                camera_records=tuple(cameras),valid_kinds=set(supported))
            self.scene_observations.append(record)
            self.scene_observations=self.scene_observations[-self.max_views:]
            self.scene_records+=1
            self.emit(event='scene_state_record',batch=batch,capture_phase=phase,
                timestamp=float(packet.timestamp),valid_kinds=sorted(supported),
                source_cookie_observations=len(record.source_cookies),
                target_cookie_observations=len(record.target_cookies),
                cameras=[c['camera'] for c in cameras],
                stored_scene_records=len(self.scene_observations))
        added=[]
        boxes={kind:copy.deepcopy(getattr(scene,kind+'_box')) for kind in ('source','target')}
        for name,frame in zip(('front','left_wrist','right_wrist'),packet.rgbd.frames):
            if not supported or np.isfinite(frame.depth_m).mean()<.10:
                continue
            if any(view.camera==name and view.frame.timestamp==frame.timestamp for view in self.views):
                continue
            # Keep distinct times even at the same camera pose: an arm moving
            # in front of a fixed camera must not overwrite earlier real floor
            # rays. Bounded storage and per-query change/age guards still apply.
            self.views.append(MemoryView(name,frame,boxes,set(supported)))
            self.views=self.views[-self.max_views:]
            added.append(name)
        if added:
            self.captures+=1;self.views_recorded+=len(added)
        self.emit(event='visual_memory_capture',batch=batch,capture_phase=phase,
            cameras=added,stored_views=len(self.views),timestamp=float(packet.timestamp))

    def _validate_records(self, timestamp, box, kind):
        for record in self.scene_observations:
            if kind not in record.valid_kinds:
                continue
            age=float(timestamp)-record.timestamp
            old=record.boxes[kind]
            angle=np.arccos(np.clip((np.trace(old.rotation.T@box.rotation)-1)/2,-1,1))
            if (age<0 or age>self.max_age_s
                    or np.linalg.norm(old.position-box.position)>self.pose_tolerance_m
                    or angle>self.angle_tolerance_rad):
                record.valid_kinds.discard(kind)

    def fraction(self, packet, box, xy, floor_offset, *, kind, batch):
        self.begin_batch(batch)
        self._validate_records(packet.timestamp,box,kind)
        # Parent keeps original times/transforms, demands live rim support,
        # vetoes new near-floor obstacles, and never calls unknown rays free.
        return super().fraction(packet,box,xy,floor_offset,kind=kind,batch=batch)

    def _has_record(self, kind):
        return any(kind in record.valid_kinds for record in self.scene_observations)

    def has_record(self, kind, batch):
        if kind not in ('source','target'):
            raise ValueError('Unknown scene observation kind')
        return self.batch==batch and self._has_record(kind)

    def _missing_kind(self, *, source_ok, target_ok, has_five, source, target):
        if not has_five:
            return 'source_entry'
        if source_ok:
            return 'target_empty'
        if target_ok:
            return 'source_entry'
        source_seen=self._has_record('source');target_seen=self._has_record('target')
        if source_seen!=target_seen:
            return 'target_empty' if source_seen else 'source_entry'
        return 'source_entry' if source<=target else 'target_empty'

    def make_goal(self, packet, scene, spec, index, *, column='auto', batch):
        self.begin_batch(batch)
        self.goal_queries+=1
        if abs(scene.timestamp-packet.timestamp)>1e-6:
            raise ValueError('Grasp positions require the current RGB-D scene')
        half=np.asarray(spec['cookie_half_size'])
        regions=target_regions(spec['target_slots_local'],spec['floor_offset_m']+half[2])
        if not 0<=index<len(regions):
            raise LocalizationFailure('Target batch index out of range')
        target=regions[index]
        target_xy=rectangle_samples(target[:,:2].min(0)-half[:2]+.001,
            target[:,:2].max(0)+half[:2]-.001)
        target_evidence=self.fraction(packet,scene.target_box,target_xy,spec['floor_offset_m'],
            kind='target',batch=batch)
        target_fraction=target_evidence['combined_fraction']
        target_ok=(target_fraction>=.90 and not target_evidence['current_obstacle_veto']
            and scene.target_box.supported_sides>=3)

        # Always inspect current source instances/corridor even if target failed.
        # Cached cookies are for audit only: none can become a grasp command.
        candidates=[];has_five=False
        for col in sorted({c.column for c in scene.source_cookies}):
            if column!='auto' and col!=column:
                continue
            try:five=select_visible_five(scene,column=col)
            except LocalizationFailure:continue
            has_five=True
            positions=np.asarray([cookie.position for cookie in five])
            local=(positions-scene.source_box.position)@scene.source_box.rotation
            low=np.array([local[:,0].mean()-.018,-scene.source_box.inner_half_size[1]+.001])
            high=np.array([local[:,0].mean()+.018,local[:,1].min()-half[1]-.0007])
            if high[1]<=low[1]:
                continue
            evidence=self.fraction(packet,scene.source_box,rectangle_samples(low,high),
                spec['source_floor_top_offset_m'],kind='source',batch=batch)
            fraction=evidence['combined_fraction']
            valid=(fraction>=.90 and not evidence['current_obstacle_veto']
                and scene.source_box.supported_sides>=3)
            candidates.append(dict(column=col,positions=positions,evidence=evidence,
                fraction=float(fraction),valid=bool(valid)))
        valid=[candidate for candidate in candidates if candidate['valid']]
        best=max(valid or candidates,key=lambda candidate:candidate['fraction']) if candidates else None
        source_fraction=best['fraction'] if best else 0.
        source_ok=bool(best and best['valid'])
        self.last_gate_evidence=dict(region_index=int(index),batch=int(batch),
            timestamp=float(packet.timestamp),source_fraction=float(source_fraction),
            target_fraction=float(target_fraction),has_current_five=has_five,
            source_ok=source_ok,target_ok=bool(target_ok),
            source_evidence=copy.deepcopy(best['evidence']) if best else None,
            target_evidence=copy.deepcopy(target_evidence))
        self.emit(event='scene_state_joint_gate',**self.last_gate_evidence)
        if source_ok and target_ok:
            self.goal_accepts+=1;self.last_requirement=None
            history_supported=any(evidence['history_used']
                and evidence['current_fraction']<.90<=evidence['combined_fraction']
                for evidence in (best['evidence'],target_evidence))
            if history_supported:self.history_supported_goals+=1
            self.emit(event='scene_state_goal_accepted',region_index=int(index),batch=int(batch),
                timestamp=float(packet.timestamp),history_supported=bool(history_supported),
                history_supported_goals=self.history_supported_goals,
                goal_accepts=self.goal_accepts)
            return VisualBatchGoal(index,best['positions'],scene.source_box.rotation,target,
                scene.target_box.position,scene.target_box.rotation,half,packet.timestamp,
                best['column'],source_fraction,target_fraction)
        missing=self._missing_kind(source_ok=source_ok,target_ok=target_ok,
            has_five=has_five,source=source_fraction,target=target_fraction)
        self.last_requirement=dict(missing_kind=missing,region_index=int(index),
            source_fraction=float(source_fraction),target_fraction=float(target_fraction),
            has_current_five=has_five)
        raise ObservationRequirement(missing,index,source_fraction=source_fraction,
            target_fraction=target_fraction,has_current_five=has_five,
            evidence=self.last_gate_evidence)

    def snapshot_summary(self):
        latest={kind:max((record.timestamp for record in self.scene_observations
            if kind in record.valid_kinds),default=None) for kind in ('source','target')}
        return dict(enabled=True,is_scene_state_memory=True,batch=self.batch,
            captures=self.captures,history_uses=self.history_uses,queries=self.queries,
            views_recorded=self.views_recorded,stored_views=len(self.views),
            scene_records=self.scene_records,stored_scene_records=len(self.scene_observations),
            source_records=sum('source' in record.valid_kinds for record in self.scene_observations),
            target_records=sum('target' in record.valid_kinds for record in self.scene_observations),
            latest_observation_times=latest,goal_queries=self.goal_queries,goal_accepts=self.goal_accepts,
            history_supported_goals=self.history_supported_goals,
            last_requirement=copy.deepcopy(self.last_requirement),
            last_gate_evidence=copy.deepcopy(self.last_gate_evidence))
