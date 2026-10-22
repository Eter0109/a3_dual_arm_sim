"""Visible rectangular faces propose measured oriented thin objects.

Merged, incomplete or inconsistent faces stay unknown. No expected-count prior
is used to invent hidden instances. Includes the existing upright ridge fits.
"""
from itertools import product
import numpy as np
from scipy import ndimage
from .rgbd_localization import CookieEstimate,minimum_rectangle


def inside_cookie(points,cookie,half,margin=.0015):
    return (abs((points-cookie.position)@cookie.rotation)<=half+margin).all(1)


def face_candidates(points,camera,half):
    if len(points)<60:return []
    # Deterministic voxel/sample budget; fitting does not use policy RNG.
    _,indices=np.unique(np.floor(points/.001).astype(int),axis=0,return_index=True)
    remaining=points[np.sort(indices)];remaining=remaining[::max(1,len(remaining)//3500)]
    rng=np.random.default_rng(0);result=[]
    for _ in range(8):
        if len(remaining)<60:break
        best=None
        for trial in range(48):
            a,b,c=remaining[rng.choice(len(remaining),3,replace=False)]
            normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
            if length<1e-7:continue
            normal/=length;mask=abs((remaining-a)@normal)<.00065
            if best is None or mask.sum()>best.sum():best=mask
        if best is None or best.sum()<60:break
        patch=remaining[best];remaining=remaining[~best]
        center=patch.mean(0);_,_,vt=np.linalg.svd(patch-center,full_matrices=False)
        normal=vt[-1]
        if normal@(camera-center)<0:normal=-normal
        u=vt[0];v=np.cross(normal,u);plane=(patch-center)@np.column_stack((u,v))
        lo=plane.min(0)-.002;grid_index=np.floor((plane-lo)/.001).astype(int)
        shape=grid_index.max(0)+2
        if np.prod(shape)>500000:continue
        grid=np.zeros(tuple(shape[::-1]),bool);grid[grid_index[:,1],grid_index[:,0]]=True
        grid=ndimage.binary_closing(grid,structure=np.ones((2,2)))
        labels,_=ndimage.label(grid)
        for region in ndimage.find_objects(labels):
            y,x=region;sub=patch[(grid_index[:,0]>=x.start)&(grid_index[:,0]<x.stop)&
                               (grid_index[:,1]>=y.start)&(grid_index[:,1]<y.stop)]
            if len(sub)<55:continue
            planar=(sub-center)@np.column_stack((u,v))
            try:r,lower,upper=minimum_rectangle(planar)
            except Exception:continue
            extent=upper-lower;mid=(lower+upper)/2
            basis=np.column_stack((u,v))@r
            for axes in ((0,1,2),(1,0,2),(0,2,1),(2,0,1),(2,1,0),(1,2,0)):
                expected=2*half[list(axes[:2])]
                if np.any(abs(extent-expected)>np.maximum(.0015,expected*.08)):continue
                relative=(sub-center)@basis
                bins=np.floor((relative-lower)/np.maximum(extent/np.array([12,6]),1e-8)).astype(int)
                bins=np.clip(bins,[0,0],[11,5]);coverage=len(np.unique(bins,axis=0))/72
                if coverage<.75:continue
                rotation=np.zeros((3,3));rotation[:,axes[0]]=basis[:,0];rotation[:,axes[1]]=basis[:,1]
                rotation[:,axes[2]]=normal
                if np.linalg.det(rotation)<0:rotation[:,axes[1]]*=-1
                position=center+basis@mid-normal*half[axes[2]]
                residual=float(np.std((sub-position)@normal))
                result.append(CookieEstimate(position,rotation,None,coverage,residual,'measured_oriented_slab_v1'))
                break
    return result


def oriented_targets(frames,clouds,box,spec,upright):
    half=np.asarray(spec['cookie_half_size']);result=list(upright);all_points=[]
    for frame,cloud in zip(frames,clouds):
        r,g,b=frame.rgb.astype(float).transpose(2,0,1);local=(cloud-box.position)@box.rotation
        mask=(r>70)&(r>g*1.13)&(g>b*1.25)&np.isfinite(cloud).all(2)
        mask&=(abs(local[:,:,:2])<box.inner_half_size+.035).all(2)
        mask&=(local[:,:,2]>spec['floor_offset_m']+.001)&(local[:,:,2]<spec['floor_offset_m']+.12)
        q=cloud[mask];all_points.append(q)
        for c in result:q=q[~inside_cookie(q,c,half)]
        for candidate in face_candidates(q,frame.world_from_optical[:3,3],half):
            if not any(np.linalg.norm(candidate.position-c.position)<.004 for c in result):result.append(candidate)
    unknown=False
    for q in all_points:
        if not len(q):continue
        unexplained=np.ones(len(q),bool)
        for c in result:unexplained&=~inside_cookie(q,c,half,margin=.002)
        # The crop includes a narrow outer margin so partially out-of-box
        # objects cannot vanish. Source objects outside the target ROI are not
        # counted. A substantial unresolved surface cannot certify success.
        local=(q-box.position)@box.rotation
        relevant=(abs(local[:,:2])<box.inner_half_size+.003).all(1)
        if np.sum(unexplained&relevant)>max(70,.08*np.sum(relevant)):unknown=True
    return tuple(result),unknown
