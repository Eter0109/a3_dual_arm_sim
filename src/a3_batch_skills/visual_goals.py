"""RGB-D goal creation and the same visual conditioning for training/inference.

This module receives camera packets and static dimensions, never an environment.
Source cues stay fixed in world coordinates for one batch. They are not secretly
tracked using simulator object poses while the learned policy grasps/transfers.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import linear_sum_assignment
from .markers import CAMERAS,MARKED_INSTRUCTION,draw_box,project_points
from .planning import target_regions
from .rgbd_localization import LOCALIZER_VERSION,LocalizationFailure,select_visible_five

VISUAL_MARKER_VERSION="rgbd_latched_batch_targets_v1"


def floor_visibility(frame, box, xy, floor_offset, tolerance=.002,*,return_mask=False):
    """Depth-ray test: unseen/occluded pixels never count as empty floor."""
    from .rgbd import RGBDViews
    if isinstance(frame,RGBDViews):
        mask=np.any([floor_visibility(f,box,xy,floor_offset,tolerance,return_mask=True)
                     for f in frame.frames],axis=0)
        return mask if return_mask else float(mask.mean())
    world=np.c_[np.asarray(xy),np.full(len(xy),floor_offset)]@box.rotation.T+box.position
    t=frame.world_from_optical
    optical=(world-t[:3,3])@t[:3,:3]
    uv=optical@frame.intrinsic.T
    positive=optical[:,2]>.01
    uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-8)-.5
    pixel=np.rint(uv).astype(int)
    h,w=frame.depth_m.shape
    valid=positive&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    observed=np.full(len(xy),np.nan)
    observed[valid]=frame.depth_m[pixel[valid,1],pixel[valid,0]]
    empty=valid&np.isfinite(observed)&(abs(observed-optical[:,2])<tolerance)
    return empty if return_mask else float(empty.mean())


def rectangle_samples(lower,upper,n=11):
    xx,yy=np.meshgrid(np.linspace(lower[0],upper[0],n),np.linspace(lower[1],upper[1],n))
    return np.c_[xx.ravel(),yy.ravel()]


@dataclass(frozen=True)
class VisualBatchGoal:
    index: int
    source_positions: np.ndarray
    source_rotation: np.ndarray
    target_positions_box: np.ndarray
    target_position: np.ndarray
    target_rotation: np.ndarray
    half_size: np.ndarray
    timestamp: float
    source_column: int
    source_clear_fraction: float
    target_empty_fraction: float

    def __post_init__(self):
        shapes=dict(source_positions=(5,3),source_rotation=(3,3),target_positions_box=(5,3),
                    target_position=(3,),target_rotation=(3,3),half_size=(3,))
        for name,shape in shapes.items():
            value=np.asarray(getattr(self,name),dtype=float).copy()
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError("Invalid visual goal "+name)
            value.setflags(write=False);object.__setattr__(self,name,value)

    @property
    def world_corners(self):
        corners=np.asarray(list(product((-1,1),repeat=3)))*self.half_size
        source=np.concatenate([p+corners@self.source_rotation.T for p in self.source_positions])
        target=np.concatenate([self.target_position+(p+corners)@self.target_rotation.T
                               for p in self.target_positions_box])
        return source,target


def make_visual_goal(frame,scene,spec,index,*,column="auto"):
    half=np.asarray(spec["cookie_half_size"])
    regions=target_regions(spec["target_slots_local"],spec["floor_offset_m"]+half[2])
    if not 0<=index<len(regions):raise LocalizationFailure("Target batch index out of range")
    target=regions[index]
    lower=target[:,:2].min(0)-half[:2]+.001
    upper=target[:,:2].max(0)+half[:2]-.001
    empty=floor_visibility(frame,scene.target_box,rectangle_samples(lower,upper),spec["floor_offset_m"])
    if empty<.90:raise LocalizationFailure("Target region is occupied or not visibly empty")
    columns=sorted({c.column for c in scene.source_cookies})
    for col in columns:
        if column!="auto" and col!=column:continue
        try:five=select_visible_five(scene,column=col)
        except LocalizationFailure:continue
        positions=np.asarray([c.position for c in five])
        local=(positions-scene.source_box.position)@scene.source_box.rotation
        # Certify that a preceding hidden/remaining cookie is not being skipped.
        # An apparently missing row caused by arm occlusion is not free space.
        low=np.array([local[:,0].mean()-.018,-scene.source_box.inner_half_size[1]+.001])
        high=np.array([local[:,0].mean()+.018,local[:,1].min()-half[1]-.0007])
        if high[1]<=low[1]:continue
        clear=floor_visibility(frame,scene.source_box,rectangle_samples(low,high),
                               spec["source_floor_top_offset_m"])
        if clear<.90:continue
        return VisualBatchGoal(index,positions,scene.source_box.rotation,target,scene.target_box.position,
            scene.target_box.rotation,half,frame.timestamp,col,clear,empty)
    raise LocalizationFailure("No five-cookie end has a visibly clear approach; acquire another view")


def camera_calibrations_from_fk(env):
    """Simulation adapter for camera calibration/FK only; no object queries."""
    result={}
    for name in CAMERAS:
        cid=env.model.camera(name).id
        result[name]=(env.data.cam_xpos[cid].copy(),env.data.cam_xmat[cid].reshape(3,3).copy(),
                      float(env.model.cam_fovy[cid]))
    return result


def annotate_visual_goal(observation,calibrations,goal):
    result=dict(observation)
    source,target=goal.world_corners
    for name in CAMERAS:
        key="observation.images."+name
        rgb=np.asarray(observation[key])
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError("Expected RGB policy views")
        args=(*calibrations[name],*rgb.shape[:2])
        uv,visible=project_points(source,*args)
        marked=draw_box(rgb,uv,visible,(0,220,255))
        uv,visible=project_points(target,*args)
        result[key]=draw_box(marked,uv,visible,(80,255,80))
    return result


def visual_region_diagnostics(scene,goal,*,xy_tolerance=.004,z_tolerance=.0025):
    """Measured goal agreement, separately from instance detection/native score."""
    result=dict(complete=False,matched=0,xy_tolerance_m=xy_tolerance,z_tolerance_m=z_tolerance)
    if not scene.target_cookies:return result
    actual=(np.asarray([c.position for c in scene.target_cookies])-scene.target_box.position)@scene.target_box.rotation
    desired=goal.target_positions_box
    distance=np.linalg.norm(actual[:,None,:2]-desired[None,:,:2],axis=2)
    rows,cols=linear_sum_assignment(distance)
    xy=distance[rows,cols];z=abs(actual[rows,2]-desired[cols,2])
    result.update(matched=len(cols),max_xy_error_m=float(xy.max()),max_z_error_m=float(z.max()),
        mean_offset_box_m=(actual[rows]-desired[cols]).mean(0).tolist(),
        complete=bool(len(cols)==5 and np.all(xy<xy_tolerance) and np.all(z<z_tolerance)))
    return result


def visual_region_complete(scene,goal,*,xy_tolerance=.004,z_tolerance=.0025):
    """Conservative visible upright-footprint gate, not native benchmark success."""
    return visual_region_diagnostics(scene,goal,xy_tolerance=xy_tolerance,
                                     z_tolerance=z_tolerance)["complete"]


class VisualGoalPolicy:
    """One adapted policy executes the complete batch; no geometric takeover."""
    def __init__(self,policy,metadata):
        if (metadata.get("goal_conditioning")!=VISUAL_MARKER_VERSION
                or metadata.get("localization")!=LOCALIZER_VERSION
                or metadata.get("training_completed") is not True):
            raise ValueError("RGB-D goal mode requires weights actually trained with these measured, latched cues")
        self.policy=policy
        self.goal=None

    def start(self,goal,context=None):
        self.goal=goal
        self._reset_queue(context)

    def _reset_queue(self,context):
        # The legacy benchmark adapter reseeds torch inside reset(). A batch
        # boundary should clear actions, not restart the same noise sequence.
        import torch
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):self.policy.reset(context)

    def act(self,observation,calibrations):
        if self.goal is None:raise RuntimeError("No RGB-D batch goal")
        return self.policy.act(annotate_visual_goal(observation,calibrations,self.goal),MARKED_INSTRUCTION)

    def pause(self,context=None):
        self._reset_queue(context)
        self.goal=None
