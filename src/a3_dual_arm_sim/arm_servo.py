"""Rate-limited Cartesian setpoints, planned on the pose an arm was *told* to hold.

Both batch experts drive an arm with the same primitive: an absolute Cartesian
goal, approached at a bounded rate, converted to joint commands by IK against
the previous command.  What makes a fast rate safe lives here rather than in
either expert, so the two cannot drift apart:

* **The plan is built from the command, not the measurement.**  A plan built from
  the measured pose has to be clamped, or a blocked move winds it up into a
  virtual spring; and that clamp also caps the rate, because the lag it is
  willing to tolerate is what has to pay for the speed.  Building from the
  command needs no clamp: the plan holds its rate and the arm trails it.
* **The rate ramps down inside the last ``DECEL_DISTANCE_M``.**  Without it the
  only thing keeping an arrival gentle is the rate itself, which is the thing
  being raised.  The ramp is what lets the approach rate be fast and still end
  without carrying that rate into whatever the tool meets.

The plan is a position only: orientation is passed straight through to the IK, so
an arm that cannot reach the goal orientation does not silently creep towards it.
"""

from __future__ import annotations

import mujoco
import numpy as np

#: Within this distance of its target a move ramps its rate down.
DECEL_DISTANCE_M = 0.020
#: A move already slower than this is gentle enough not to be worth ramping.
RAMP_ABOVE_M_PER_STEP = 0.0020
#: Floor of the ramp, so a ramped arrival still terminates in bounded steps.
RAMP_FLOOR_M_PER_STEP = 0.0012


def ramp(rate: float, distance: float) -> float:
    """``rate``, ramped down inside the last ``DECEL_DISTANCE_M``."""
    if 0.0 < distance < DECEL_DISTANCE_M and rate > RAMP_ABOVE_M_PER_STEP:
        return max(RAMP_FLOOR_M_PER_STEP, rate * distance / DECEL_DISTANCE_M)
    return rate


class CommandedPose:
    """Forward kinematics of one arm at the joints it was last commanded to hold.

    ``env.last_applied_action`` is the rate-limited command the actuators were
    given, not the state they produced, which is what makes it the right thing to
    plan from: it is where the arm *should* be, and every command the expert
    issues is relative to it.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_id: int,
        qpos_ids: np.ndarray,
        action_slice: slice,
    ) -> None:
        self.model = model
        self.data = data
        self.site = site_id
        self.qpos_ids = np.asarray(qpos_ids, dtype=np.int32)
        self.action_slice = action_slice
        # One scratch MjData, reused: the servo runs on every step of every
        # phase, and a fresh MjData per call is 0.8 ms of pure allocation.
        self._work = mujoco.MjData(model)

    def position(self, commanded: np.ndarray) -> np.ndarray:
        """World position of this arm's tool at ``commanded``'s joint values."""
        work = self._work
        work.qpos[:] = self.data.qpos
        work.qpos[self.qpos_ids] = commanded[self.action_slice]
        mujoco.mj_kinematics(self.model, work)
        return work.site_xpos[self.site].copy()

    def setpoint(self, commanded: np.ndarray, position, rate: float) -> np.ndarray:
        """The pose to aim at this step: at most ``rate`` from the commanded one."""
        current = self.position(commanded)
        error = np.asarray(position, dtype=np.float64) - current
        distance = float(np.linalg.norm(error))
        if distance <= 1e-9:
            return current
        step = min(distance, ramp(rate, distance))
        return current + error * (step / distance)
