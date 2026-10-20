"""Visible, approximately upright five-cookie poses near the robot gripper.

Only calibrated RGB-D, robot FK and public dimensions enter this module. A
rigid formation is an explicit acceptance test, not a fill-in for unseen objects.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .rgbd import RGBDViews,unproject
from .rgbd_localization import RGBDLocalizer,BoxEstimate,LocalizationFailure,rotation_z
from .motion import average_rotation

HELD_POSE_VERSION="rgbd_visible_held_five_v1"


@dataclass(frozen=True)
class HeldBatchPose:
    center: np.ndarray
    rotation: np.ndarray
    positions: np.ndarray
    timestamp: float
    formation_error_m: float
    visible_count: int
    plane_residual_m: float


class HeldBatchLocalizer:
    def __init__(self,spec):
        self.spec=spec;self.half=np.asarray(spec["cookie_half_size"])
        self.fitter=RGBDLocalizer(spec,min_coverage=.45)

    def observe(self,packet,pad_midpoint,reference_rotation):
        frames=packet.frames if isinstance(packet,RGBDViews) else (packet,)
        pad=np.asarray(pad_midpoint,dtype=float);reference=np.asarray(reference_rotation,dtype=float)
        clouds=[]
        for frame in frames:
            points=unproject(frame)
            local=(points-pad)@reference
            r,g,b=frame.rgb.astype(float).transpose(2,0,1)
            orange=(r>70)&(r>g*1.13)&(g>b*1.25)&np.isfinite(points).all(2)
            roi=(abs(local[:,:,0])<.060)&(abs(local[:,:,1])<.043)&(local[:,:,2]>-.022)&(local[:,:,2]<.030)
            clouds.append(points[orange&roi])
        return self.fit(np.concatenate(clouds),pad,reference,packet.timestamp)

    def fit(self,points,pad,reference,timestamp):
        """Point-cloud entry for sensor-independent geometry regression tests."""
        points=np.asarray(points,dtype=float);pad=np.asarray(pad);reference=np.asarray(reference)
        points=points[np.isfinite(points).all(1)]
        if len(points)<150:raise LocalizationFailure("Insufficient visible orange surface near gripper")
        rng=np.random.default_rng(0)  # independent of policy sampling/randomization
        sample=points[rng.choice(len(points),min(2500,len(points)),replace=False)]
        best=None
        for _ in range(192):
            a,b,c=sample[rng.choice(len(sample),3,replace=False)]
            normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
            if length<1e-9:continue
            normal/=length
            if normal@reference[:,2]<0:normal=-normal
            if normal@reference[:,2]<np.cos(np.deg2rad(15)):continue
            distance=abs((sample-a)@normal)
            support=int(np.sum(distance<.00065))
            if best is None or support>best[0]:best=(support,a,normal)
        if best is None or best[0]<100:raise LocalizationFailure("No supported upright top plane near gripper")
        _,origin,normal=best
        selected=points[abs((points-origin)@normal)<.00065]
        center=selected.mean(0)
        _,_,vt=np.linalg.svd(selected-center,full_matrices=False)
        normal=vt[-1]
        if normal@reference[:,2]<0:normal=-normal
        if normal@reference[:,2]<np.cos(np.deg2rad(15)):raise LocalizationFailure("Held surface tilt exceeds supported range")
        axis=reference[:,0]-normal*(reference[:,0]@normal);axis/=np.linalg.norm(axis)
        base_rotation=np.column_stack((axis,np.cross(normal,axis),normal))
        # The intact five-cookie group is longer than it is wide. Resolve a
        # modest measured yaw before projecting rows, including split ends.
        planar=(selected-center)@base_rotation[:,:2]
        values,vectors=np.linalg.eigh(np.cov(planar.T));direction=vectors[:,-1]
        if direction[0]<0:direction=-direction
        angles=list(np.deg2rad(np.arange(-15,16,3)))
        if direction[0]>np.cos(np.deg2rad(25)) and values[-1]>1.15*max(values[0],1e-12):
            angles.append(float(np.arctan2(direction[1],direction[0])))
        top=float(np.median((selected-pad)@normal))
        options=[];best_count=0
        # Partial visibility biases global PCA. Compare measured ridge support
        # over modest yaw hypotheses instead of treating that PCA as the pose.
        for angle in angles:
            rotation=base_rotation@rotation_z(angle)
            local=(points-pad)@rotation;q=local[abs(local[:,2]-top)<.001]
            box=BoxEstimate(pad,rotation,np.array([.061,.044]),0.,0,0.)
            cookies=self.fitter._free_targets(q,box,allow_internal_occlusion=True)
            best_count=max(best_count,len(cookies))
            if not 5<=len(cookies)<=10:continue
            for choice in combinations(cookies,5):
                pos=np.asarray([c.position for c in choice]);rot=average_rotation([c.rotation for c in choice])
                relative=(pos-pos.mean(0))@rot;relative=relative[np.argsort(relative[:,1])]
                desired=np.c_[np.zeros(5),np.arange(-2,3)*2*self.half[1],np.zeros(5)]
                error=float(np.max(np.linalg.norm(relative-desired,axis=1)))
                near=np.linalg.norm((pos.mean(0)-pad)@reference[:,:2])
                if error<.003 and near<.030:options.append((error+near*.15,pos,rot,error))
        if not options:raise LocalizationFailure(f"No supported intact held group; best visible count {best_count}")
        _,pos,rot,error=min(options,key=lambda row:row[0])
        residual=float(np.std((selected-center)@normal))
        return HeldBatchPose(pos.mean(0),rot,pos,float(timestamp),error,5,residual)
