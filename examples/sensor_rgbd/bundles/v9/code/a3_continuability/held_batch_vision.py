"""RGB-D geometry of an airborne batch. No simulation object identities/poses."""
import numpy as np
from .batch_vision import CookieEstimate, LocalizationError, colored_cookie_points, _components

CANONICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def fit_airborne_cookie(points, half_size, top_bevel=.0025, bottom_bevel=.001):
    """Fit in an approximately cookie-aligned frame, without a floor prior."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    p = np.asarray(points); h = np.asarray(half_size)
    if len(p) < 25 or np.ptp(p[:, 0]) < h[0] * 1.2:
        raise LocalizationError('Airborne slab has insufficient visible width')
    p = p[np.linspace(0, len(p)-1, min(800, len(p))).astype(int)]
    # Existing calibrated views show the near side and upper edge. Bounds are
    # initialization only; all three translations and rotations are optimized.
    initial = np.array([(np.quantile(p[:, 0], .01)+np.quantile(p[:, 0], .99))/2,
                        np.quantile(p[:, 1], .01)+h[1], np.quantile(p[:, 2], .99)-h[2]])
    def residual(v):
        local = (p-v[:3]) @ Rotation.from_rotvec(v[3:]).as_matrix()
        x,y,z = local.T
        return np.stack([np.abs(x)-h[0], np.abs(y)-h[1], np.abs(z)-h[2],
            (np.abs(y)+z-h[1]-h[2]+top_bevel)/np.sqrt(2),
            (np.abs(y)-z-h[1]-h[2]+bottom_bevel)/np.sqrt(2)]).max(axis=0)
    v0 = np.r_[initial, np.zeros(3)]
    bounds = np.array([.004, .009, .009, .35, .35, .20])
    result = least_squares(residual, v0, bounds=(v0-bounds, v0+bounds),
                           loss='soft_l1', f_scale=.0004, max_nfev=160)
    error = float(np.quantile(np.abs(residual(result.x)), .8))
    if not result.success or error > .0015:
        raise LocalizationError(f'Airborne slab fit rejected: {error*1000:.2f} mm')
    return CookieEstimate(result.x[:3], Rotation.from_rotvec(result.x[3:]).as_matrix(), error, len(p))


def localize_held_batch(frames, tool_position, tool_rotation, pad_offset, half_size,
                       *, source_floor_z, top_bevel=.0025, bottom_bevel=.001):
    try:
        return _localize_held_by_hue(frames,tool_position,tool_rotation,pad_offset,half_size,
            source_floor_z=source_floor_z,top_bevel=top_bevel,bottom_bevel=bottom_bevel)
    except LocalizationError:
        return _localize_held_by_ridges(frames,tool_position,tool_rotation,pad_offset,half_size,
            source_floor_z=source_floor_z,top_bevel=top_bevel,bottom_bevel=bottom_bevel)


def _localize_held_by_hue(frames, tool_position, tool_rotation, pad_offset, half_size,
                       *, source_floor_z, top_bevel=.0025, bottom_bevel=.001):
    """Count and fit visible airborne slabs near the physical gripper."""
    origin = np.asarray(tool_position) + np.asarray(tool_rotation) @ pad_offset
    basis = np.asarray(tool_rotation) @ CANONICAL.T
    points, hsv = colored_cookie_points(frames,
        lower=np.maximum(origin-[.075,.075,.065], [-np.inf,-np.inf,source_floor_z+.035]),
        upper=origin+[.075,.075,.065])
    if len(points) < 150:
        raise LocalizationError('Held batch is not sufficiently visible above the tray')
    local = (points-origin) @ basis
    keep = (np.abs(local[:,0]) < .036) & (np.abs(local[:,1]) < .055) & (np.abs(local[:,2]) < .045)
    local,hsv = local[keep],hsv[keep]
    estimates=[]
    for bright in (False,True):
        p=local[(hsv[:,0]>=19)==bright]
        if not len(p): continue
        bins=np.floor(p[:,1:]/.0005).astype(int)
        _,inverse=np.unique(bins,axis=0,return_inverse=True)
        yz=np.zeros((inverse.max()+1,2)); counts=np.bincount(inverse)
        np.add.at(yz,inverse,p[:,1:]); yz/=counts[:,None]
        for group in _components(yz,.0018):
            cloud=p[np.isin(inverse,group)]
            if len(cloud)<25 or np.ptp(cloud[:,0])<np.asarray(half_size)[0]*1.2:continue
            estimate=fit_airborne_cookie(cloud,half_size,top_bevel,bottom_bevel)
            estimates.append(CookieEstimate(origin+basis@estimate.position,
                basis@estimate.rotation,estimate.fit_error_m,estimate.points))
    estimates.sort(key=lambda e:float((e.position-origin)@basis[:,1]))
    if len(estimates)!=5:
        raise LocalizationError(f'Expected exactly five airborne slabs; observed {len(estimates)}')
    centers=np.stack([e.position for e in estimates]); gaps=np.diff(centers@basis[:,1])
    if np.any(gaps<.003) or np.any(gaps>.012):
        raise LocalizationError('Airborne slabs do not form one contiguous five-cookie batch')
    return estimates


def ridge_centers(local):
    """Five individually observed bevel ridges, independent of cookie hue."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    p=np.asarray(local);step=.0003
    p=p[np.abs(p[:,0])<.018]
    if len(p)<150:raise LocalizationError('Too little visible airborne top surface')
    low,high=np.quantile(p[:,1],[.005,.995])
    edges=np.arange(low-.001,high+.001+step,step)
    profile=np.full(len(edges)-1,np.nan)
    for i in range(len(profile)):
        patch=p[(p[:,1]>=edges[i])&(p[:,1]<edges[i+1]),2]
        if len(patch)>=3:profile[i]=np.quantile(patch,.85)
    supported=np.isfinite(profile)
    if supported.sum()<25:raise LocalizationError('Airborne ridges not sufficiently observed')
    profile=np.interp(np.arange(len(profile)),np.flatnonzero(supported),profile[supported])
    smooth=gaussian_filter1d(profile,.7)
    peaks,_=find_peaks(smooth,distance=int(.0045/step),prominence=.00065)
    peaks=peaks[supported[peaks]]
    centers=edges[peaks]+step/2
    if len(centers)!=5 or np.any(np.diff(centers)>.012):
        raise LocalizationError(f'Expected five measured bevel ridges; observed {len(centers)}')
    return centers


def _localize_held_by_ridges(frames,tool_position,tool_rotation,pad_offset,half_size,
                          *,source_floor_z,top_bevel,bottom_bevel):
    origin=np.asarray(tool_position)+np.asarray(tool_rotation)@pad_offset
    basis=np.asarray(tool_rotation)@CANONICAL.T
    points,_=colored_cookie_points(frames,lower=np.maximum(origin-[.075,.075,.065],
        [-np.inf,-np.inf,source_floor_z+.035]),upper=origin+[.075,.075,.065])
    local=(points-origin)@basis
    local=local[(np.abs(local[:,0])<.036)&(np.abs(local[:,1])<.055)&(np.abs(local[:,2])<.045)]
    centers=ridge_centers(local)
    boundaries=np.r_[-np.inf,(centers[:-1]+centers[1:])/2,np.inf]
    result=[]
    for i in range(5):
        cloud=local[(local[:,1]>=boundaries[i])&(local[:,1]<boundaries[i+1])]
        estimate=fit_airborne_cookie(cloud,half_size,top_bevel,bottom_bevel)
        result.append(CookieEstimate(origin+basis@estimate.position,basis@estimate.rotation,
            estimate.fit_error_m,estimate.points))
    if np.any(np.abs(np.diff(np.array([e.position for e in result])@basis[:,1])-np.diff(centers))>.002):
        raise LocalizationError('Fitted slabs disagree with measured ridges')
    return result


def placement_goal(estimates, tool_position, tool_rotation, box_rotation, center_world):
    """Preserve observed cookie-to-tool offset/rotation while making batch upright."""
    if len(estimates)!=5:raise LocalizationError('Placement requires exactly five measured cookies')
    center=np.mean([e.position for e in estimates],axis=0)
    u,_,vh=np.linalg.svd(np.mean([e.rotation for e in estimates],axis=0))
    cookie_rotation=u@np.diag([1,1,np.linalg.det(u@vh)])@vh
    local_rotation=np.asarray(tool_rotation).T@cookie_rotation
    local_offset=np.asarray(tool_rotation).T@(center-tool_position)
    rotation=np.asarray(box_rotation)@local_rotation.T
    return np.asarray(center_world)-rotation@local_offset,rotation
