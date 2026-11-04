"""Optional batch-skill scoring; the pinned benchmark and its score are untouched.

``contained_unstacked`` means contained, released, settled, and with its bottom
within 4 mm of the target floor. This threshold is below a full cookie's
smallest thickness, rejecting a complete second layer. It is not a general
detector of every partially overlapping leaned pile. Direct floor contact is
not required: adjacent upright cookies may support one another by friction.
Simulator geometry is evaluation evidence, not a visual observation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlacementRules:
    mode: str = "strict"
    floor_clearance_m: float = 0.004

    def __post_init__(self):
        if self.mode not in ("strict", "contained_unstacked"):
            raise ValueError("Unknown placement mode")
        if not np.isfinite(self.floor_clearance_m) or self.floor_clearance_m < 0:
            raise ValueError("Invalid floor clearance")


def cookie_is_accepted(*, position, rotation, half_size, inner_half_size,
                       floor_top, wall_height, floor_clearance, held,
                       linear_speed, angular_speed, max_linear_speed,
                       max_angular_speed, containment_tolerance):
    """All poses are expressed in the target-box frame; no upright constraint."""
    position = np.asarray(position, dtype=float)
    extent = np.abs(np.asarray(rotation, dtype=float)) @ np.asarray(half_size)
    if not np.all(np.isfinite(np.r_[position, extent, linear_speed, angular_speed])):
        return False
    bottom = position[2] - extent[2]
    return bool(
        np.all(np.abs(position[:2]) + extent[:2]
               <= np.asarray(inner_half_size) + containment_tolerance)
        and floor_top - 0.003 <= bottom < wall_height
        and bottom <= floor_top + floor_clearance and not held
        and linear_speed <= max_linear_speed
        and angular_speed <= max_angular_speed
    )


class BatchPlacementScorer:
    """Independent scorer; success must persist for the native hold duration."""
    version = "contained-unstacked-floor-clearance-v2"

    def __init__(self, env, rules=None):
        self.env = env
        self.rules = rules or PlacementRules()
        self.hold_count = 0
        if self.rules.floor_clearance_m >= 2 * min(env.COOKIE_HALF_SIZE):
            raise ValueError("Floor clearance must be smaller than one cookie's thickness")

    def reset(self):
        """Reset this counter only at episode start, never reset the simulator."""
        self.hold_count = 0

    def accepted_ids(self):
        env = self.env
        if self.rules.mode == "strict":
            return frozenset(i for i in range(env.task_config.cookie_count)
                             if env.privileged_cookie_in_target(i))
        center = env.data.xpos[env._target_bin_body]
        box_rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
        scene = env.config.cookie_transfer
        accepted = set()
        for i, body in enumerate(env._cookie_bodies):
            linear, angular = env._cookie_speed(i, relative_to_target=True)
            if cookie_is_accepted(
                position=box_rotation.T @ (env.data.xpos[body] - center),
                rotation=box_rotation.T @ env.data.xmat[body].reshape(3, 3),
                half_size=env.COOKIE_HALF_SIZE, inner_half_size=env.TARGET_INNER_HALF_SIZE,
                floor_top=scene.target_floor_z_m + scene.bin_wall_thickness_m / 2,
                wall_height=env.TARGET_WALL_HEIGHT, floor_clearance=self.rules.floor_clearance_m,
                held=any(env.privileged_left_finger_contacts(i)),
                linear_speed=linear, angular_speed=angular,
                max_linear_speed=max(env.task_config.max_linear_speed_m_s, 0.09),
                max_angular_speed=env.task_config.max_angular_speed_rad_s,
                containment_tolerance=env.CONTACT_CONTAINMENT_TOLERANCE_M,
            ):
                accepted.add(i)
        return frozenset(accepted)

    def update(self, native_info):
        from .outcomes import native_success
        accepted = self.accepted_ids()
        safe = not native_info.get("safety_reason") and not self.env._safety_stop
        filled = len(accepted) == self.env.task_config.required_cookies and safe
        self.hold_count = self.hold_count + 1 if filled else 0
        original_success = native_success(self.env, native_info)
        selected_success = original_success if self.rules.mode == "strict" else (
            self.hold_count >= self.env.task_config.success_hold_steps)
        return {
            "placement_mode": self.rules.mode,
            "placement_version": "native" if self.rules.mode == "strict" else self.version,
            "accepted_ids": sorted(accepted), "accepted_count": len(accepted),
            "success": bool(selected_success), "native_success": original_success,
            "native_count": int(native_info.get("cookies_in_target", 0)),
            "hold_count": self.hold_count, "safety_stop": not bool(safe),
        }


class _ControllerView:
    """Only the privileged expert's release check uses the selected criterion.

    The real env.step still computes the untouched original benchmark score.
    This view never writes physics, observations, or native scoring methods.
    """
    def __init__(self, env, scorer):
        self._env, self._scorer = env, scorer

    def __getattr__(self, name):
        return getattr(self._env, name)

    def privileged_cookie_in_target(self, index):
        return index in self._scorer.accepted_ids()


def make_batch_expert(env, scorer, *, seed=0):
    """Reuse the existing 5+5 / 5+5+5+5 contact-driven expert as an oracle baseline.

    Its native RETRACT -> VERIFY_RELEASE -> SELECT_COOKIE loop already clears
    the local servo target and remeasures the next batch; do not call reset()
    at each handoff, which would erase completed-cookie records.
    """
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
    from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
    from a3_dual_arm_sim.contracts import EpisodeContext
    relaxed = scorer.rules.mode == "contained_unstacked"
    view = _ControllerView(env, scorer) if relaxed else env

    class Relaxed2x10Expert(A3Cookie2x10Expert):
        def _verify_complete_fill(self):
            accepted = scorer.accepted_ids()
            if len(accepted) != self.env.task_config.required_cookies:
                self._fail(f"floor-supported target count={len(accepted)}/20")

    if env.task_config.required_cookies == 20:
        cls = Relaxed2x10Expert if relaxed else A3Cookie2x10Expert
        expert = cls(view, first_grasp=5, sampling_seed=seed,
                     requested_source_column_index=0)
    elif env.task_config.required_cookies == 10:
        expert = A3SameColumnBatchExpert(view)
    else:
        raise ValueError("This pilot supports only 2x5 and 2x10")
    expert.reset(EpisodeContext(seed=seed, action_mode="cartesian_delta",
                               task="Repeat five-cookie transfers until the box is filled."))
    return expert
