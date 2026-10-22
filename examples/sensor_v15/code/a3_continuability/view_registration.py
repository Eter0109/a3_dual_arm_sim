"""Align cameras using independently observed known-size bin rims.

The common frame is anchored to a measured full-bin image, normally the front
camera. Absolute XY/yaw error of that reference remains unobservable here.
"""
import numpy as np


def box_alignment(reference,observed):
    if min(reference.supported_sides,observed.supported_sides)<3:
        raise ValueError('Three supported rim sides are needed for registration')
    yaw=np.arctan2(reference.rotation[1,0],reference.rotation[0,0])-np.arctan2(observed.rotation[1,0],observed.rotation[0,0])
    yaw=(yaw+np.pi)%(2*np.pi)-np.pi
    c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shift=reference.position-rot@observed.position;shift[2]=0.
    displacement=np.linalg.norm((rot@observed.position+shift)-observed.position)
    if abs(yaw)>np.deg2rad(3) or displacement>.015:
        raise ValueError('Cross-view correction exceeds nominal calibration bounds')
    return rot,shift


def register_views(views,spec,*,table_z=.75):
    from a3_batch_skills.rgbd import RGBDViews,RGBDFrame,unproject
    from a3_batch_skills.rgbd_localization import RGBDLocalizer,LocalizationFailure
    anchors=[]
    for i,frame in enumerate(views.frames):
        try:anchors.append((i,RGBDLocalizer(spec).observe(frame)))
        except LocalizationFailure:pass
    if not anchors:return views,{'reference':None,'reason':'No full-bin image anchors relative calibration'}
    # Prefer the stable external camera; do not switch reference because a wrist
    # happens to have a slightly smaller rim fitting error at this instant.
    index,base=min(anchors,key=lambda pair:(pair[0]!=0,pair[0]))
    result=[];details={'reference':index,'cameras':[]}
    for i,frame in enumerate(views.frames):
        if i==index:result.append(frame);details['cameras'].append({'reference':True});continue
        points=unproject(frame);matches=[]
        for kind in ('source','target'):
            clone=dict(spec)
            if kind=='source':
                clone['target_inner_half_size']=spec['source_inner_half_size']
                clone['wall_height_m']=spec['source_wall_top_offset_m']
            else:
                clone['source_inner_half_size']=spec['target_inner_half_size']
                clone['source_wall_top_offset_m']=spec['wall_height_m']
            try:
                a,b=RGBDLocalizer(clone)._boxes(frame,points,table_z)
                observed=a if kind=='source' else b
                reference=getattr(base,kind+'_box')
                rot,shift=box_alignment(reference,observed)
                matches.append((observed.extent_error_m,rot,shift,kind))
            except (LocalizationFailure,ValueError):pass
        if not matches:
            result.append(frame);details['cameras'].append({'registered':False,'reason':'Insufficient independent rim support'});continue
        matches.sort(key=lambda v:v[0]);_,rot,shift,kind=matches[0]
        if any(np.linalg.norm((r@frame.world_from_optical[:3,3]+s)-(rot@frame.world_from_optical[:3,3]+shift))>.003 for _,r,s,_ in matches[1:]):
            result.append(frame);details['cameras'].append({'registered':False,'reason':'Source/target registrations disagree'});continue
        transform=frame.world_from_optical.copy()
        transform[:3,:3]=rot@transform[:3,:3];transform[:3,3]=rot@transform[:3,3]+shift
        result.append(RGBDFrame(frame.rgb,frame.depth_m,frame.intrinsic,transform,frame.timestamp,
            'measured_table_plane_and_cross_view_bin_registration'))
        details['cameras'].append(dict(registered=True,anchor_bin=kind,
            camera_shift_m=float(np.linalg.norm(transform[:3,3]-frame.world_from_optical[:3,3]))))
    return RGBDViews(tuple(result)),details
