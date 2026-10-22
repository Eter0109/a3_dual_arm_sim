"""Sensor-only, orientation-aware containment. Native scores never enter."""
from itertools import product
import numpy as np
from .planning import rectangle_overlap


def cookie_corners(cookie,half):
    return np.asarray(list(product((-1,1),repeat=3)))*half@cookie.rotation.T+cookie.position


def contained_report(scene,spec,expected,*,unknown=False):
    half=np.asarray(spec['cookie_half_size']);box=scene.target_box
    floor=float(spec['floor_offset_m']);accepted=[];reasons=[];footprints=[]
    for cookie in scene.target_cookies:
        points=(cookie_corners(cookie,half)-box.position)@box.rotation
        bottom=float(points[:,2].min());local=(cookie.position-box.position)@box.rotation
        rotation=box.rotation.T@cookie.rotation
        inside=bool(np.all(abs(points[:,:2])<=box.inner_half_size+.001))
        supported=bool(floor-.001<=bottom<=floor+.002)
        valid=inside and supported and cookie.coverage>=.75 and cookie.top_residual_m<=.0015
        accepted.append(valid);footprints.append((local,rotation,bottom))
        if not inside:reasons.append('outside_box')
        if not supported:reasons.append('not_supported_by_box_floor_or_stacked')
        if not valid and inside and supported:reasons.append('uncertain_instance_fit')
    # If vertically separated objects overlap in XY, treat that as stacking.
    stacked=False
    for i,(a,ra,za) in enumerate(footprints):
        for b,rb,zb in footprints[i+1:]:
            if abs(za-zb)>.003 and rectangle_overlap(a,ra,half,b,rb,half,margin=-.001):stacked=True
    if stacked:reasons.append('stacked_overlap')
    if len(accepted)!=expected:reasons.append('unexpected_or_missing_instance_count')
    if unknown:reasons.append('unresolved_or_occluded_instances')
    return dict(complete=bool(len(accepted)==expected and all(accepted) and not stacked and not unknown),
        expected=expected,detected=len(accepted),accepted=sum(accepted),unknown=bool(unknown),
        reasons=sorted(set(reasons)),containment_tolerance_m=.001,floor_clearance_m=.002,
        orientation_required=False)


def stable_surface_points(scene,half):
    """Corners, rather than centres alone, reveal a cookie still tipping over."""
    return np.asarray([cookie_corners(c,np.asarray(half)) for c in scene.target_cookies])
