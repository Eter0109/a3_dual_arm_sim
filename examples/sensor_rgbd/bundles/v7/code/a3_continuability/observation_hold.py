"""Bounded measured-pose dwell; no sensor replay, view motion, or scene inputs."""
import numpy as np
from .task_completion import EncoderSettling


class ObservationHold:
    def __init__(self, *, max_steps=40, required_stable=3):
        if min(max_steps, required_stable) < 1:
            raise ValueError('Positive observation dwell bounds required')
        self.max_steps = int(max_steps)
        self.required_stable = int(required_stable)
        self.steps = 0
        self.monitor = EncoderSettling()
        self.target = None
        self.previous = None
        self.last_delta = None
        self.max_delta = 0.
        self.tracking_error = None
        self.max_tracking_error = 0.
        self.fresh_samples = 0

    def update(self, state, timestamp, jaw_gap, *, fresh, opening):
        self.steps += 1
        latched = False
        independent = False
        # Never latch from a repeated/old packet, nor repeatedly chase measured
        # encoders. The first current packet fixes all seven joint targets once.
        if fresh:
            measured = np.asarray(state, dtype=float)[:7]
            independent = self.monitor.update(measured, timestamp)
            if independent:
                self.fresh_samples += 1
                if self.target is None:
                    if not np.isfinite(opening) or not 0 <= opening <= 1:
                        raise ValueError('Finite normalized gripper command required')
                    self.target = np.r_[measured.copy(), float(opening)]
                    latched = True
                self.last_delta = None if self.previous is None else float(np.max(abs(measured-self.previous)))
                if self.last_delta is not None:
                    self.max_delta = max(self.max_delta, self.last_delta)
                self.previous = measured.copy()
                self.tracking_error = float(np.max(abs(measured-self.target[:7])))
                self.max_tracking_error = max(self.max_tracking_error, self.tracking_error)
        open_jaw = jaw_gap is not None and np.isfinite(jaw_gap) and jaw_gap > .0347
        # Readiness needs a new, independent image/encoder sample. A duplicate
        # timestamp with fresh=True cannot progress or complete this dwell.
        ready = bool(independent and open_jaw and self.monitor.count >= self.required_stable)
        if not independent:
            reason = 'awaiting_independent_fresh_packet'
        elif not open_jaw:
            reason = 'measured_open_jaw_required'
        elif not ready:
            reason = 'measured_joints_not_yet_stationary'
        else:
            reason = 'ready'
        return dict(ready=ready, timed_out=not ready and self.steps >= self.max_steps,
            latched=latched, independent=independent, reason=reason,
            hold_steps=self.steps, fresh_samples=self.fresh_samples,
            stationary_frames=self.monitor.count, joint_delta_rad=self.last_delta,
            max_joint_delta_rad=self.max_delta, tracking_error_rad=self.tracking_error,
            max_tracking_error_rad=self.max_tracking_error,
            timestamp=float(timestamp))
