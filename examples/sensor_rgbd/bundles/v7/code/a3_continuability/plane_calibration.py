"""RGB-D calibration against a declared, measured horizontal tabletop.

This estimates plane-normal and height errors. A plane alone cannot determine
camera XY translation or yaw; these remain at the nominal calibration.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlaneCorrection:
    rotation: np.ndarray
    translation: np.ndarray
    residual_m: float
    inlier_fraction: float
    tilt_degrees: float
    height_shift_m: float


def fit_table_correction(points,camera_position,*,table_height_m=.75,seed=0):
    xyz=np.asarray(points,dtype=float).reshape(-1,3)
    valid=np.isfinite(xyz).all(1)
    valid&=(xyz[:,0]>-.12)&(xyz[:,0]<.55)&(xyz[:,1]>-.20)&(xyz[:,1]<.60)
    valid&=(abs(xyz[:,2]-table_height_m)<.055)
    # Declared calibration region outside the entire bin/search volume. Never
    # fit a bin floor or the tops of packed cookies as the tabletop reference.
    valid&=(xyz[:,0]<.015)|(xyz[:,0]>.33)|(xyz[:,1]<-.04)|(xyz[:,1]>.49)
    xyz=xyz[valid]
    if len(xyz)<200:raise ValueError('Insufficient observed tabletop support')
    rng=np.random.default_rng(seed)
    if len(xyz)>4000:xyz=xyz[rng.choice(len(xyz),4000,replace=False)]
    best=None
    for _ in range(60):
        a,b,c=xyz[rng.choice(len(xyz),3,replace=False)]
        n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<1e-8:continue
        n/=length
        if n[2]<0:n=-n
        if n[2]<np.cos(np.deg2rad(6)):continue
        d=-n@a;mask=abs(xyz@n+d)<.001
        count=int(mask.sum())
        if best is None or count>best[0]:best=(count,mask)
    if best is None or best[0]<max(200,len(xyz)*.25):raise ValueError('No supported tabletop plane')
    patch=xyz[best[1]]
    if np.prod(np.ptp(patch[:,:2],axis=0))<.025:raise ValueError('Table support has insufficient spatial extent')
    center=patch.mean(0);_,_,vh=np.linalg.svd(patch-center,full_matrices=False)
    n=vh[-1]
    if n[2]<0:n=-n
    d=-n@center;residual=float(np.sqrt(np.mean((patch@n+d)**2)))
    tilt=float(np.rad2deg(np.arccos(np.clip(n[2],-1,1))))
    if residual>.0007 or tilt>6:raise ValueError('Uncertain tabletop plane')
    axis=np.cross(n,[0.,0.,1.]);s=np.linalg.norm(axis);c=float(n[2])
    skew=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rot=np.eye(3)+skew+skew@skew*((1-c)/max(s*s,1e-12))
    cp=np.asarray(camera_position,dtype=float)
    level_height=cp[2]-(n@cp+d)
    shift=float(table_height_m-level_height)
    if abs(shift)>.035:raise ValueError('Table height correction exceeds declared calibration bounds')
    translation=cp-rot@cp+np.array([0.,0.,shift])
    return PlaneCorrection(rot,translation,residual,best[0]/len(xyz),tilt,shift)


def rectify_frame(frame,*,table_height_m=.75,focal_scale=1.):
    from a3_batch_skills.rgbd import RGBDFrame,unproject
    intrinsic=frame.intrinsic.copy();intrinsic[:2,:2]*=focal_scale
    candidate=RGBDFrame(frame.rgb,frame.depth_m,intrinsic,frame.world_from_optical,
        frame.timestamp,'nominal_plus_measured_table_plane')
    correction=fit_table_correction(unproject(candidate)[::5,::5],frame.world_from_optical[:3,3],table_height_m=table_height_m)
    transform=frame.world_from_optical.copy()
    transform[:3,:3]=correction.rotation@transform[:3,:3]
    transform[:3,3]=correction.rotation@transform[:3,3]+correction.translation
    return RGBDFrame(frame.rgb,frame.depth_m,intrinsic,transform,frame.timestamp,
        'measured_table_plane_normal_and_height_only'),correction
