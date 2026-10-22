"""Minimal vertical clearance from measured robot FK and visible bin rims."""
import numpy as np


def observation_lift(position, pad_midpoint, rim_z, *, clearance_m=.065):
    position=np.asarray(position,dtype=float)
    pad_midpoint=np.asarray(pad_midpoint,dtype=float)
    if (position.shape!=(3,) or pad_midpoint.shape!=(3,)
            or not np.isfinite(np.r_[position,pad_midpoint,rim_z,clearance_m]).all()
            or clearance_m<.04):
        raise ValueError('Observation clearance needs measured finite robot geometry')
    # The long open fingers remain below their midpoint. The 65 mm allowance
    # includes that geometry; merely clearing the EEF is insufficient.
    delta=max(0.,float(rim_z)+clearance_m-float(pad_midpoint[2]))
    target=position.copy();target[2]+=delta
    return target,delta
