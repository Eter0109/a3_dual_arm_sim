"""Full before-action sensor clips; failed/uncertified clips never train actions."""
import json
from pathlib import Path
import numpy as np
from .geometry_context import GEOMETRY_VERSION,GEOMETRY_DIM
from .markers import CAMERAS,MARKED_INSTRUCTION
from .recording import LosslessRGBVideo
from .rgbd_localization import LOCALIZER_VERSION
from .visual_goals import VISUAL_MARKER_VERSION


class SensorPolicyClipRecorder:
    def __init__(self,root,fps,*,provenance,video_factory=LosslessRGBVideo):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.fps,self.provenance,self.video_factory=fps,provenance,video_factory
        self.active=self.pending=None;self.clips=[]

    @property
    def goal(self):return self.active['goal'] if self.active else None

    def prepare(self,observation,geometry,goal,global_step,action_source):
        if self.pending is not None:raise RuntimeError('Previous frame has no applied action')
        if self.active is None:
            path=self.root/f'batch_{goal.index:02d}';path.mkdir(exist_ok=False)
            self.active=dict(goal=goal,path=path,rows=[],videos={})
        if self.active['goal'].index!=goal.index:raise RuntimeError('Unfinished training batch')
        geometry=np.asarray(geometry,dtype=np.float32).copy()
        state=np.asarray(observation['observation.state'][:8],dtype=np.float32).copy()
        if geometry.shape!=(GEOMETRY_DIM,) or state.shape!=(8,) or not np.isfinite(np.r_[state,geometry]).all():
            raise ValueError('Invalid pre-action geometry/state')
        self.pending=dict(state=state,geometry=geometry,global_step=global_step,action_source=action_source,
            images={key:np.asarray(observation['observation.images.'+key]).copy() for key in CAMERAS})

    def commit(self,action,*,visual_verified=False,native_count=0,failed=False,reason=None):
        row=self.pending
        if row is None:raise RuntimeError('Missing before-action frame')
        row['action']=np.asarray(action,dtype=np.float32).copy()
        if row['action'].shape!=(8,) or not np.isfinite(row['action']).all():raise ValueError('Invalid applied left8')
        for name,image in row.pop('images').items():
            if name not in self.active['videos']:
                self.active['videos'][name]=self.video_factory(self.active['path']/(name+'.mkv'),self.fps,image.shape)
            self.active['videos'][name].add(image)
        self.active['rows'].append(row);self.pending=None
        if visual_verified or failed:
            accepted=visual_verified and native_count==5*(self.goal.index+1)
            self.finish(accepted,native_count=native_count,reason=reason or (
                None if accepted else 'Visual/native acceptance failed'))

    def finish(self,success,*,native_count=0,reason=None):
        if self.active is None:return
        active=self.active;rows=active['rows'];goal=active['goal']
        for video in active['videos'].values():video.close()
        if rows:
            keys=('state','geometry','action','global_step','action_source')
            np.savez_compressed(active['path']/'trajectory.npz',**{key:np.asarray([r[key] for r in rows]) for key in keys})
            sources=[r['action_source'] for r in rows]
            metadata=dict(schema='a3_single_batch_v1',clip_success=bool(success),frames=len(rows),fps=self.fps,
                batch_index=goal.index,entry_kind='initial' if goal.index==0 else 'after_previous_release',
                source_column=goal.source_column,
                state_dim=8,action_dim=8,geometry_conditioning=GEOMETRY_VERSION,geometry_dim=GEOMETRY_DIM,
                per_frame_geometry=True,geometry_source='calibrated_RGBD_and_robot_FK_only',
                goal_conditioning=VISUAL_MARKER_VERSION,localization=LOCALIZER_VERSION,task=MARKED_INSTRUCTION,
                clip_role='hybrid_correction' if 'visual_servo' in sources else 'learned_rollout',
                visual_servo_frames=sources.count('visual_servo'),failure_reason=reason,
                native_count_at_accept=int(native_count),visual_boundary_verified=bool(success),
                action_source_counts={key:sources.count(key) for key in sorted(set(sources))},
                first_global_step=int(rows[0]['global_step']),last_global_step=int(rows[-1]['global_step']),
                timing='observation_before_step -> applied_joint_target_at_that_step',
                physics_resets_between_batches=0,provenance=self.provenance)
            (active['path']/'metadata.json').write_text(json.dumps(metadata,indent=2))
            self.clips.append(metadata)
        self.active=None

    def close(self):
        self.pending=None;self.finish(False,reason='Run ended before visual/native acceptance')
        (self.root/'manifest.json').write_text(json.dumps(dict(clips=self.clips),indent=2))
