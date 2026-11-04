"""Measured held-batch geometry and robot-only placement reachability.

No simulator object state is used. A half-turn around the desired cookie normal
preserves the upright rectangular slab geometry while changing the wrist pose.
This is a kinematic feasibility check, not a collision-free contact guarantee.
"""
from dataclasses import dataclass
import numpy as np
from .batch_vision import LocalizationError
from .held_batch_vision import placement_goal


@dataclass(frozen=True)
class PlacementCandidate:
    position: np.ndarray
    rotation: np.ndarray
    yaw_radians: float
    tilt_radians: float = 0.


def placement_candidates(estimates, tool_position, tool_rotation, box_rotation, center_world,*,include_tilt=False):
    """Maintain the measured cookie/tool transform for both upright symmetries."""
    result=[]
    for yaw, symmetry in [(0.,np.eye(3)),(np.pi,np.diag([-1.,-1.,1.]))]:
        position,rotation=placement_goal(estimates,tool_position,tool_rotation,
            np.asarray(box_rotation)@symmetry,center_world)
        result.append(PlacementCandidate(position,rotation,yaw))
    if include_tilt:
        for tilt in [-.03,.03,-.06,.06]:
            c,s=np.cos(tilt),np.sin(tilt)
            local=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
            position,rotation=placement_goal(estimates,tool_position,tool_rotation,
                np.asarray(box_rotation)@local,center_world)
            result.append(PlacementCandidate(position,rotation,0.,tilt))
    return result


def placement_bounds(candidate,original,estimates,tool_position,tool_rotation,
                     target_box,half_size,floor_offset):
    """Guard the measured slabs' full volume, retaining the packed-row extent."""
    from itertools import product
    corners=np.asarray(list(product((-1.,1.),repeat=3)))*np.asarray(half_size)
    measured=np.concatenate([np.asarray(estimate.position)+corners@np.asarray(estimate.rotation).T
        for estimate in estimates])
    relative=(measured-np.asarray(tool_position))@np.asarray(tool_rotation)
    def project(pose):
        world=relative@pose.rotation.T+pose.position
        return (world-target_box.position)@target_box.rotation
    local=project(candidate);baseline=project(original)
    wall_margin=float(np.min(np.asarray(target_box.inner_half_size)-np.max(abs(local[:,:2]),axis=0)))
    floor_clearance=float(np.min(local[:,2])-floor_offset)
    # Tilting about the thin-slab normal changes X/Z but should not invade a
    # neighbouring packed row in Y. All checks use actual measured held offsets.
    row_expansion=float(max(baseline[:,1].min()-local[:,1].min(),
        local[:,1].max()-baseline[:,1].max(),0.))
    safe=bool(wall_margin>=.0005 and floor_clearance>=-.00025 and row_expansion<=.0001)
    return dict(safe=safe,wall_margin_m=wall_margin,min_floor_clearance_m=floor_clearance,
        packed_row_expansion_m=row_expansion)


def sample_segment(start_position,start_rotation,end_position,end_rotation,*,
                   max_translation=.012,max_rotation=.06):
    from scipy.spatial.transform import Rotation,Slerp
    start_position=np.asarray(start_position);end_position=np.asarray(end_position)
    angle=Rotation.from_matrix(np.asarray(start_rotation).T@end_rotation).magnitude()
    count=max(1,int(np.ceil(np.linalg.norm(end_position-start_position)/max_translation)),
        int(np.ceil(angle/max_rotation)))
    if count>200:raise LocalizationError('Placement path exceeds bounded interpolation length')
    fractions=np.linspace(0.,1.,count+1)[1:]
    rotations=Slerp([0.,1.],Rotation.from_matrix(np.stack([start_rotation,end_rotation])))(fractions).as_matrix()
    return [(start_position+(end_position-start_position)*fraction,rotation)
        for fraction,rotation in zip(fractions,rotations)]


def check_placement_chain(motion,state,candidate,*,position_tolerance=.0015,
                          rotation_tolerance=.025):
    """Continue bounded IK along the complete Cartesian path, including descent."""
    state=np.asarray(state,dtype=float)
    start_position,start_rotation=motion.pose(state)
    seed=state[:7].copy();path=[];maximum_position=0.;maximum_rotation=0.;endpoint_joints={}
    endpoints=[('TRANSPORT',candidate.position+[0.,0.,.10]),
               ('PREINSERT',candidate.position+[0.,0.,.035]),
               ('INSERT',candidate.position)]
    for phase,position in endpoints:
        for point,rotation in sample_segment(start_position,start_rotation,position,candidate.rotation):
            q,actual_position,actual_rotation=motion.placement_ik(point,rotation,seed)
            pe=float(np.linalg.norm(actual_position-point))
            re=float(np.arccos(np.clip((np.trace(actual_rotation.T@rotation)-1)/2,-1.,1.)))
            maximum_position=max(maximum_position,pe);maximum_rotation=max(maximum_rotation,re)
            if (not np.isfinite(q).all() or pe>position_tolerance or re>rotation_tolerance):
                return dict(feasible=False,failed_phase=phase,position_error_m=pe,
                    rotation_error_rad=re,max_position_error_m=maximum_position,
                    max_rotation_error_rad=maximum_rotation,checked_samples=len(path)+1)
            path.append(np.asarray(q).copy());seed=np.asarray(q).copy()
        endpoint_joints[phase]=seed.tolist()
        start_position,start_rotation=position,candidate.rotation
    return dict(feasible=True,failed_phase=None,max_position_error_m=maximum_position,
        max_rotation_error_rad=maximum_rotation,checked_samples=len(path),endpoint_joints=endpoint_joints)


def choose_placement(motion,state,estimates,tool_position,tool_rotation,box_rotation,
                     center_world,*,emit=None,target_box=None,half_size=None,floor_offset=None):
    """Keep an accurate default; compare alternatives for marginal IK paths.

    Passing the hard path limits only certifies a candidate, not a useful servo
    margin. In particular, default paths near joint limits can barely pass the
    1.5 mm / 0.025 rad planner gate and still stall during bounded execution.
    Preserve the exact default fast path only when its residual is comfortably
    below those limits. Other feasible defaults participate in the same cost
    comparison as the existing bounded alternatives; no geometry gate changes.
    """
    emit=emit or (lambda **row:None)
    diagnostics=[];feasible=[]
    bounded=target_box is not None and half_size is not None and floor_offset is not None
    candidates=placement_candidates(estimates,tool_position,tool_rotation,box_rotation,center_world,
        include_tilt=bounded)
    try:
        for candidate in candidates:
            geometry=(placement_bounds(candidate,candidates[0],estimates,tool_position,tool_rotation,
                target_box,half_size,floor_offset) if bounded else dict(safe=True))
            if not geometry['safe']:
                row=dict(yaw_radians=candidate.yaw_radians,tilt_radians=candidate.tilt_radians,
                    feasible=False,failed_phase='GEOMETRY',geometry=geometry)
                diagnostics.append(row);emit(event='placement_candidate_checked',**row);continue
            check=check_placement_chain(motion,state,candidate)
            row=dict(yaw_radians=candidate.yaw_radians,tilt_radians=candidate.tilt_radians,geometry=geometry,**check)
            diagnostics.append(row);emit(event='placement_candidate_checked',**row)
            if check['feasible']:
                cost=(check['max_position_error_m']+.01*check['max_rotation_error_rad']
                    +.005*abs(candidate.tilt_radians)+.0001*abs(candidate.yaw_radians))
                feasible.append((cost,candidate,row))
                high_margin=(check['max_position_error_m']<=.0003 and
                             check['max_rotation_error_rad']<=.005)
                if candidate.yaw_radians==0. and candidate.tilt_radians==0. and high_margin:break
        if feasible:
            _,selected,check=min(feasible,key=lambda item:item[0])
            # The last diagnostic is the selected solution for the loop's
            # validated endpoint references, independent of evaluation order.
            ordered=[row for row in diagnostics if row is not check]+[check]
            emit(event='placement_plan_selected',position=selected.position.tolist(),
                rotation=selected.rotation.tolist(),kinematic_check_only=True,object_truth_used=False,**check)
            return selected,ordered
    finally:
        # Diagnostic IK may use a separate robot work buffer; the real encoder
        # pose remains the only seed for subsequent physical control.
        motion.pose(state)
    raise LocalizationError('No upright placement candidate has a bounded reachable full path')
