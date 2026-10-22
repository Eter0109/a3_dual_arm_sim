"""Visible upright slab instances: metric shape plus broad chromaticity.

Adapted from the pinned a3_batch_skills RGB-D detector. The orange-only gate is
replaced by chromaticity so the benchmark's cocoa/matcha/berry palettes work.
Packing cells propose search strips; returned objects still require visible
surfaces, width, height and coverage support.
"""
import numpy as np
from scipy import ndimage


def slab_instances(frame,points,box,spec,*,source,min_coverage=.9,source_columns=4):
    from a3_batch_skills.rgbd_localization import CookieEstimate
    half=np.asarray(spec['cookie_half_size']);rgb=frame.rgb.astype(float)
    local=(points-box.position)@box.rotation
    chromatic=(np.ptp(rgb,axis=2)>40)&(np.max(rgb,axis=2)>45)
    inside=(abs(local[:,:,:2])<box.inner_half_size).all(2)
    floor=spec['source_floor_top_offset_m'] if source else spec['floor_offset_m']
    mask=chromatic&inside&np.isfinite(points).all(2)
    mask&=abs(local[:,:,2]-(floor+2*half[2]))<(.004 if source else .001)
    q=local[mask]
    if not len(q):return ()
    xs=((np.arange(source_columns)-(source_columns-1)/2)*2*half[0]
        if source else np.unique(np.asarray(spec['target_slots_local'])[:,0]))
    estimates=[]
    for column,x in enumerate(xs):
        strip=q[abs(q[:,0]-x)<half[0]+(-.001 if source else .003)]
        if len(strip)<30:continue
        step=.0004;edges=np.arange(-box.inner_half_size[1],box.inner_half_size[1]+step,step)
        counts,_=np.histogram(strip[:,1],edges)
        smooth=ndimage.gaussian_filter1d(counts.astype(float),1.3)
        occupied=ndimage.binary_closing(smooth>max(3,float(smooth.max())*.15),structure=np.ones(2))
        labels,_=ndimage.label(occupied)
        for region in ndimage.find_objects(labels):
            segment=region[0];width=(segment.stop-segment.start)*step
            if not (.0035 if source else .0008)<width<.0085:continue
            lower,upper=edges[segment.start]-.0002,edges[segment.stop]+.0002
            patch=strip[(strip[:,1]>lower)&(strip[:,1]<upper)]
            if len(patch)<25:continue
            bins=np.linspace(x-half[0]+.001,x+half[0]-.001,25)
            coverage=float(np.mean(np.histogram(patch[:,0],bins)[0]>=2))
            if coverage<min_coverage:continue
            yl,yh=np.quantile(patch[:,1],[.02,.98]);measured_x=x
            if not source:
                xl,xh=np.quantile(patch[:,0],[.01,.99])
                if not 1.80*half[0]<xh-xl<2.15*half[0]:continue
                measured_x=float((xl+xh)/2)
            z=float(np.quantile(patch[:,2],.9)-half[2])
            top=patch[patch[:,2]>np.quantile(patch[:,2],.65)]
            residual=float(np.std(top[:,2])) if len(top) else 0.
            if residual>.0015:continue
            estimates.append(CookieEstimate(box.position+box.rotation@np.array([measured_x,(yl+yh)/2,z]),
                box.rotation.copy(),column+1 if source else None,coverage,residual,
                'packing_x_rgbd_yz' if source else 'rgbd_rectangle_xyz'))
    return tuple(sorted(estimates,key=lambda c:(c.column or 0,float((c.position-box.position)@box.rotation[:,1]))))
