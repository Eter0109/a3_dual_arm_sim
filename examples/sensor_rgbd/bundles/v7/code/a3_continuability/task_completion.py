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


class FinalVisualSettling:
    """A continuous measured settling window after the last verified batch.

    The same live visual completion gates remain authoritative. This class adds
    time and independent acquisition requirements, never native success/score.
    The robot stays at its withdrawn observation command with the jaw open.
    Motion is bounded relative to both the previous sample and the beginning
    of the window, so slow accumulated drift cannot pass as a settled scene.
    """
    def __init__(self,*,min_duration_s=.5,min_frames=6,max_gap_s=.2):
        if (not np.isfinite([min_duration_s,max_gap_s]).all()
                or min_duration_s<=0 or max_gap_s<=0
                or isinstance(min_frames,bool) or not isinstance(min_frames,int)
                or min_frames<2):
            raise ValueError('Final settling requires positive time and independent frames')
        self.min_duration_s=float(min_duration_s);self.min_frames=min_frames
        self.max_gap_s=float(max_gap_s)
        self.timestamp=None;self.previous_state=None;self.previous_positions=None
        self.start_time=None;self.anchor_state=None;self.anchor_positions=None
        self.frames=0;self.elapsed_s=0.;self.reset_reason='new_final_settle'

    def _reset_window(self,reason):
        self.start_time=None;self.anchor_state=None;self.anchor_positions=None
        self.frames=0;self.elapsed_s=0.;self.reset_reason=reason

    @staticmethod
    def _same_positions(previous,current):
        if previous is None or previous.shape!=current.shape:return False
        from scipy.optimize import linear_sum_assignment
        distance=np.linalg.norm(previous[:,None]-current[None,:],axis=2)
        rows,cols=linear_sum_assignment(distance)
        return bool(len(rows)==len(current) and np.max(distance[rows,cols])<.0015)

    def update(self,state,timestamp,positions,*,complete,jaw_gap):
        """Return True only on a new, fully qualified RGB-D/encoder sample."""
        if not np.isfinite(timestamp):
            self._reset_window('invalid_timestamp');return False
        timestamp=float(timestamp)
        if self.timestamp is not None and timestamp<=self.timestamp:return False
        prior_time=self.timestamp;prior_state=self.previous_state
        prior_positions=self.previous_positions;self.timestamp=timestamp
        state=np.asarray(state,dtype=float)[:7]
        valid_state=state.shape==(7,) and np.isfinite(state).all()
        self.previous_state=state.copy() if valid_state else None
        positions=None if positions is None else np.asarray(positions,dtype=float)
        valid_positions=(positions is not None and positions.ndim==2
            and positions.shape[1]==3 and len(positions)>0 and np.isfinite(positions).all())
        self.previous_positions=positions.copy() if valid_positions else None
        open_jaw=jaw_gap is not None and np.isfinite(jaw_gap) and jaw_gap>.0347
        if not (complete and open_jaw and valid_state and valid_positions):
            self._reset_window('current_observation_not_complete_open_and_valid');return False
        if prior_time is None or timestamp-prior_time>self.max_gap_s:
            self._reset_window('independent_observation_gap');return False
        encoder_stable=prior_state is not None and np.max(abs(state-prior_state))<.0008
        if not encoder_stable or not self._same_positions(prior_positions,positions):
            self._reset_window('measured_motion');return False
        if self.start_time is not None:
            if (np.max(abs(state-self.anchor_state))>=.0008
                    or not self._same_positions(self.anchor_positions,positions)):
                self._reset_window('accumulated_measured_motion');return False
        else:
            self.start_time=timestamp;self.anchor_state=state.copy()
            self.anchor_positions=positions.copy()
        self.frames+=1;self.elapsed_s=timestamp-self.start_time
        return bool(self.frames>=self.min_frames
            and self.elapsed_s+1e-9>=self.min_duration_s)


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
