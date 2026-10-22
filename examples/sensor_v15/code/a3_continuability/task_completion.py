"""Measured settling and image coverage, independent of simulator state."""
import numpy as np


class EncoderSettling:
    def __init__(self):
        self.previous=None;self.timestamp=None;self.count=0

    def update(self,state,timestamp):
        state=np.asarray(state,dtype=float)[:7]
        if state.shape!=(7,) or not np.isfinite(state).all() or not np.isfinite(timestamp):
            raise ValueError('Settling requires finite measured encoders and time')
        if self.timestamp is not None and timestamp<=self.timestamp:return False
        stable=self.previous is not None and np.max(abs(state-self.previous))<.0008
        self.count=self.count+1 if stable else 0
        self.previous=state.copy();self.timestamp=float(timestamp)
        return True


def observed_roi(frame,points,*,padding=12):
    """All requested points project inside a current valid-depth image.

    This only certifies a view, never free floor or object completion. Those are
    checked by localization/occupancy and release verification afterwards.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():return False
    t=frame.world_from_optical;optical=(points-t[:3,3])@t[:3,:3]
    if np.any(optical[:,2]<=.05):return False
    uv=optical@frame.intrinsic.T;pixels=np.rint(uv[:,:2]/uv[:,2:3]-.5).astype(int)
    h,w=frame.depth_m.shape
    if np.any(pixels[:,0]<padding) or np.any(pixels[:,0]>=w-padding) or np.any(pixels[:,1]<padding) or np.any(pixels[:,1]>=h-padding):return False
    depth=frame.depth_m[pixels[:,1],pixels[:,0]]
    return bool(np.isfinite(depth).all() and np.all(depth>.05))


def view_completion_ready(*,age,stationary_frames,pad_z,rim_z,jaw_gap,roi_visible):
    return bool(age>=30 and stationary_frames>=6 and np.isfinite([pad_z,rim_z]).all()
        and pad_z>=rim_z+.065 and jaw_gap is not None and jaw_gap>.0347 and roi_visible)


def insertion_release_ready(*,age,stationary_frames,visual_frames,position_error,
                            rotation_error,current_complete,previous_complete):
    return bool(age>=40 and stationary_frames>=6 and visual_frames>=3
        and np.isfinite([position_error,rotation_error]).all()
        and position_error<=.006 and rotation_error<=.06
        and current_complete and previous_complete)
